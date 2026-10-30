
import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

AGENTS = ("planner", "analyst", "verifier", "reporter", "followup")


def get_llm_for_agent(agent_name: str) -> BaseChatModel:
    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent: {agent_name}")

    provider = os.getenv(
        f"{agent_name.upper()}_LLM_PROVIDER", "gemini"
    ).lower()

    model = os.getenv(
        f"{agent_name.upper()}_LLM_MODEL", "gemini-3.5-flash-lite"
    )

    
    if provider == "gemini":
        api_key = os.getenv(f"{agent_name.upper()}_GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
