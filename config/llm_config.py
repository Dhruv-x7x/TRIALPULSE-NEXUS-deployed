
from pydantic import BaseModel
from typing import Optional

class ProviderConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "llama-3-70b"
    host: Optional[str] = None

class LLMConfig(BaseModel):
    groq: ProviderConfig = ProviderConfig(api_key="mock", model="llama-3.3-70b-versatile")
    ollama: ProviderConfig = ProviderConfig(model="trialpulse-nexus", host="http://localhost:11434")

def get_config():
    return LLMConfig()
