import datetime
import toml
import os
import openai # Import the openai library
from zoneinfo import ZoneInfo

# We'll no longer use the Google ADK's Agent class directly for the LLM part
# from google.adk.agents import Agent

def get_config():
    """Reads configuration from config.toml file."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.toml")
        
        with open(config_path, "r") as f:
            config = toml.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"config.toml not found at '{config_path}'. Please create one.")

def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        return {
            "status": "success",
            "report": (
                "The weather in New York is sunny with a temperature of 25 degrees"
                " Celsius (77 degrees Fahrenheit)."
            ),
        }
    else:
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available.",
        }

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """
    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }
    try:
        tz = ZoneInfo(tz_identifier)
        now = datetime.datetime.now(tz)
        report = (
            f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
        )
        return {"status": "success", "report": report}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}

# --- Refactored Part for OpenRouter Integration ---

# Load configuration from the TOML file
config = get_config()
llm_config = config.get("llm", {})

# Initialize the OpenAI client with OpenRouter's details
client = openai.OpenAI(
    base_url=llm_config.get("base_url"),
    api_key=llm_config.get("api_key"),
)

def ask_agent(prompt: str) -> str:
    """
    Simulates the agent's behavior by calling the LLM with the provided tools.
    
    Args:
        prompt (str): The user's query.
        
    Returns:
        str: The LLM's response.
    """
    
    # Define the available tools for the LLM to use
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Retrieves the current weather report for a specified city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city for which to retrieve the weather report.",
                        },
                    },
                    "required": ["city"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Returns the current time in a specified city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city for which to retrieve the current time.",
                        },
                    },
                    "required": ["city"],
                },
            },
        },
    ]

    try:
        # Call the LLM with the user's prompt and tool definitions
        response = client.chat.completions.create(
            model=llm_config.get("model"),
            messages=[{"role": "user", "content": prompt}],
            tools=tools,
            tool_choice="auto", # Let the model decide whether to call a tool
            temperature=llm_config.get("temperature"),
            max_tokens=llm_config.get("max_tokens"),
        )
        
        # Check if the LLM decided to call a tool
        tool_calls = response.choices[0].message.tool_calls
        
        if tool_calls:
            # If a tool call is detected, execute the function
            function_name = tool_calls[0].function.name
            function_args = tool_calls[0].function.arguments
            
            print(f"**LLM selected tool: {function_name} with args: {function_args}**")
            
            # Execute the function based on the name
            if function_name == "get_weather":
                # Note: eval() can be a security risk with untrusted input, but here
                # we're using it to parse the LLM's structured JSON output.
                args_dict = eval(function_args)
                result = get_weather(city=args_dict.get("city"))
            elif function_name == "get_current_time":
                args_dict = eval(function_args)
                result = get_current_time(city=args_dict.get("city"))
            else:
                return f"Error: Unknown tool {function_name}"
            
            # Return the result from the tool execution
            return result.get("report", result.get("error_message"))
            
        else:
            # If no tool is called, return the LLM's direct response
            return response.choices[0].message.content

    except Exception as e:
        return f"An error occurred: {e}"

# --- Interactive Loop for the new agent ---

if __name__ == "__main__":
    print("Running custom OpenRouter agent, type exit to exit.")
    while True:
        user_input = input("[user]: ")
        if user_input.lower() == "exit":
            break
        
        response = ask_agent(user_input)
        print(f"[agent]: {response}")