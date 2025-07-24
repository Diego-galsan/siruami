import os
from google.adk.models.lite_llm import LiteLlm

def model():
    model = LiteLlm(
        model = "openrouter/google/gemini-2.5-flash",
        api_key = os.getenv("OPENROUTER_API_KEY")
    )
    return model
