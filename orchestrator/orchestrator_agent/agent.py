import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, AsyncIterable, List

import httpx
import nest_asyncio
from a2a.client import A2ACardResolver
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
)
from dotenv import load_dotenv
from google.adk import Agent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from models import model # Assuming 'models' is a custom module with a 'model' class


model = model()

# NOTE: Domain-specific tools should be passed in during initialization
# instead of being hardcoded, for a truly general orchestrator.
# from .pickleball_tools import (
#     book_pickleball_court,
#     list_court_availabilities,
# )
from .remote_agent_connection import RemoteAgentConnections

load_dotenv()
nest_asyncio.apply()


class OrchestratorAgent:
    """The Orchestrator agent coordinates tasks among a network of other agents."""

    def __init__(self, tools: List[Any] = None):
        """
        Initializes the OrchestratorAgent.

        Args:
            tools: A list of tool functions for the agent to use, in addition
                   to its built-in `send_message` tool.
        """
        self.remote_agent_connections: dict[str, RemoteAgentConnections] = {}
        self.cards: dict[str, AgentCard] = {}
        self.agents: str = ""
        self._agent = self.create_agent(additional_tools=tools or [])
        self._user_id = "orchestrator_agent"
        self._runner = Runner(
            app_name=self._agent.name,
            agent=self._agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def _async_init_components(self, remote_agent_addresses: List[str]):
        """
        Asynchronously discovers and establishes connections with remote agents.
        """
        async with httpx.AsyncClient(timeout=30) as client:
            for address in remote_agent_addresses:
                card_resolver = A2ACardResolver(client, address)
                try:
                    card = await card_resolver.get_agent_card()
                    remote_connection = RemoteAgentConnections(
                        agent_card=card, agent_url=address
                    )
                    self.remote_agent_connections[card.name] = remote_connection
                    self.cards[card.name] = card
                except httpx.ConnectError as e:
                    print(f"ERROR: Failed to get agent card from {address}: {e}")
                except Exception as e:
                    print(f"ERROR: Failed to initialize connection for {address}: {e}")

        agent_info = [
            json.dumps({"name": card.name, "description": card.description})
            for card in self.cards.values()
        ]
        print("agent_info:", agent_info)
        self.agents = "\n".join(agent_info) if agent_info else "No connected agents found."

    @classmethod
    async def create(
        cls, remote_agent_addresses: List[str], tools: List[Any] = None
    ):
        """
        Factory method to create and asynchronously initialize an OrchestratorAgent instance.
        """
        instance = cls(tools=tools)
        await instance._async_init_components(remote_agent_addresses)
        return instance

    def create_agent(self, additional_tools: List[Any]) -> Agent:
        """
        Creates the underlying ADK Agent object.
        """
        base_tools = [self.send_message]
        all_tools = base_tools + additional_tools

        return Agent(
            model=model,
            name="Orchestrator_Agent",
            instruction=self.root_instruction,
            description="This agent orchestrates tasks by delegating to a network of other specialized agents.",
            tools=all_tools,
        )

    def root_instruction(self, context: ReadonlyContext) -> str:
        """
        Provides the root instruction prompt for the orchestrator model.
        """
        return f"""
        **Role:** You are an Orchestrator Agent, an expert in coordinating tasks among a group of specialized agents. Your primary function is to understand user requests, delegate sub-tasks to the appropriate agent, synthesize the results, and provide a final answer.

        **Core Directives:**

        * **Analyze the Request:** Break down the user's request into smaller, manageable sub-tasks.
        * **Task Delegation:** Identify the best agent for each sub-task based on their description. Use the `send_message` tool to delegate.
            * Frame your request clearly and provide all necessary context.
            * You must pass the official name of the agent to the `agent_name` parameter.
        * **Synthesize Results:** Once you receive responses from the agents, combine and analyze the information to formulate a comprehensive answer.
        * **Manage Tools:** If the request requires a capability you possess (i.e., a tool other than `send_message`), use that tool directly.
        * **Transparent Communication:** Keep the user informed of your progress. Relay final answers in a clear and easy-to-read format (e.g., using bullet points).
        * **Tool Reliance:** Strictly rely on the available tools to address user requests. Do not generate responses based on assumptions.
        * **Agent Awareness:** The agents listed below are the only ones available to you for delegation.

        **Today's Date (YYYY-MM-DD):** {datetime.now().strftime("%Y-%m-%d")}

        <Available Agents>
        {self.agents}
        </Available Agents>
        """

    async def stream(
        self, query: str, session_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """
        Streams the agent's response to a given query for a specific session.
        """
        session = await self._runner.session_service.get_session(
            app_name=self._agent.name,
            user_id=self._user_id,
            session_id=session_id,
        )
        content = types.Content(role="user", parts=[types.Part.from_text(text=query)])
        if session is None:
            session = await self._runner.session_service.create_session(
                app_name=self._agent.name,
                user_id=self._user_id,
                state={},
                session_id=session_id,
            )
        async for event in self._runner.run_async(
            user_id=self._user_id, session_id=session.id, new_message=content
        ):
            if event.is_final_response():
                response = ""
                if (
                    event.content
                    and event.content.parts
                    and event.content.parts[0].text
                ):
                    response = "\n".join(
                        [p.text for p in event.content.parts if p.text]
                    )
                yield {
                    "is_task_complete": True,
                    "content": response,
                }
            else:
                yield {
                    "is_task_complete": False,
                    "updates": "The orchestrator agent is thinking...",
                }

    async def send_message(self, agent_name: str, task: str, tool_context: ToolContext):
        """
        Sends a task to a connected remote agent and returns its response.

        Args:
            agent_name: The name of the target agent (e.g., 'Weather_Agent').
            task: The specific task or question to send to the agent.
            tool_context: The context provided by the ADK tool-calling framework.
        """
        if agent_name not in self.remote_agent_connections:
            raise ValueError(f"Agent '{agent_name}' not found. Available agents: {list(self.remote_agent_connections.keys())}")
        
        client = self.remote_agent_connections[agent_name]

        if not client:
            raise ValueError(f"Client connection not available for {agent_name}")

        state = tool_context.state
        task_id = state.get("task_id", str(uuid.uuid4()))
        context_id = state.get("context_id", str(uuid.uuid4()))
        message_id = str(uuid.uuid4())

        payload = {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": task}],
                "messageId": message_id,
                "taskId": task_id,
                "contextId": context_id,
            },
        }

        message_request = SendMessageRequest(
            id=message_id, params=MessageSendParams.model_validate(payload)
        )
        send_response: SendMessageResponse = await client.send_message(message_request)
        print("send_response", send_response)

        if not isinstance(
            send_response.root, SendMessageSuccessResponse
        ) or not isinstance(send_response.root.result, Task):
            print("Received a non-success or non-task response. Cannot proceed.")
            return "Error: Failed to get a valid response from the agent."

        response_content = send_response.root.model_dump_json(exclude_none=True)
        json_content = json.loads(response_content)

        resp = []
        if json_content.get("result", {}).get("artifacts"):
            for artifact in json_content["result"]["artifacts"]:
                if artifact.get("parts"):
                    resp.extend(artifact["parts"])
        return resp


def _get_initialized_orchestrator_agent_sync():
    """
    Synchronously creates and initializes the OrchestratorAgent.
    This is a helper function for environments where top-level await is not available.
    """

    async def _async_main():
        # --- CONFIGURATION ---
        # List the network addresses for all agents the orchestrator should connect to.
        remote_agent_urls = [
            "http://localhost:10002",  # Example: Weather_Agent
            "http://localhost:10003",  # Example: Calendar_Agent
            "http://localhost:10004",  # Example: Database_Agent
        ]
        
        # List any additional tools the orchestrator itself should have.
        # For example, a tool to perform a web search.
        # from .custom_tools import web_search_tool
        # orchestrator_tools = [web_search_tool]
        orchestrator_tools = []


        print("Initializing Orchestrator Agent...")
        orchestrator_instance = await OrchestratorAgent.create(
            remote_agent_addresses=remote_agent_urls,
            tools=orchestrator_tools
        )
        print("Orchestrator Agent initialized.")
        return orchestrator_instance.create_agent(additional_tools=orchestrator_tools)

    try:
        # Attempt to run the async main function
        return asyncio.run(_async_main())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print(
                f"Warning: Could not initialize OrchestratorAgent with asyncio.run(): {e}. "
                "This can happen if an event loop is already running (e.g., in Jupyter). "
                "Consider initializing the agent within an async function in your application."
            )
        else:
            raise


# This call creates the agent instance when the module is loaded.
# It can be modified to be instantiated on demand.
root_agent = _get_initialized_orchestrator_agent_sync()