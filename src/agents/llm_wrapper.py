import os
import time
import logging
from typing import Dict, Any, Optional, List
import requests
from groq import Groq
from config.llm_config import get_config

logger = logging.getLogger(__name__)

class LLMResponse:
    def __init__(self, content: str, model: str, total_tokens: int = 0, latency_ms: float = 0):
        self.content = content
        self.model = model
        self.total_tokens = total_tokens
        self.latency_ms = latency_ms

class LLMWrapper:
    """
    Standard interface for Groq/Ollama inference.
    """
    def __init__(self):
        self.config = get_config()
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            # Prefer environment variable if set, otherwise use config
            api_key = os.getenv("GROQ_API_KEY") or self.config.groq.api_key
            if api_key and api_key != "mock":
                self.client = Groq(api_key=api_key)
                logger.info("Groq client initialized")
            else:
                logger.warning("GROQ_API_KEY not found or mock. Checking for Ollama...")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")

    def generate(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.", model: Optional[str] = None) -> LLMResponse:
        start_time = time.time()
        
        # Determine model
        groq_model = model or self.config.groq.model or "llama-3.3-70b-versatile"
        
        # Try Groq first as primary
        if self.client and os.getenv("GROQ_API_KEY"):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    model=groq_model,
                    temperature=0.7,
                    max_tokens=2048,
                )
                
                latency = (time.time() - start_time) * 1000
                content = chat_completion.choices[0].message.content or ""
                tokens = chat_completion.usage.total_tokens if chat_completion.usage else 0
                
                return LLMResponse(
                    content=content,
                    model=groq_model,
                    total_tokens=tokens,
                    latency_ms=latency
                )
            except Exception as e:
                logger.error(f"Groq primary generation failed, falling back to Ollama: {e}")

        # Fallback to Ollama if Groq failed or is not configured
        try:
            ollama_host = self.config.ollama.host or "http://localhost:11434"
            ollama_model = model if model and "/" not in model else self.config.ollama.model or "trialpulse-nexus"
            
            # Use /api/chat if possible for better system prompt handling
            response = requests.post(
                f"{ollama_host}/api/chat",
                json={
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 2048
                    }
                },
                timeout=60 # Increased timeout for local models
            )
            
            if response.status_code == 200:
                data = response.json()
                latency = (time.time() - start_time) * 1000
                message = data.get("message", {})
                content = message.get("content", "")
                
                return LLMResponse(
                    content=content,
                    model=f"ollama/{ollama_model}",
                    total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                    latency_ms=latency
                )
            else:
                # Fallback to old /api/generate if /api/chat fails
                response = requests.post(
                    f"{ollama_host}/api/generate",
                    json={
                        "model": ollama_model,
                        "prompt": f"{system_prompt}\n\nUser: {prompt}\nAssistant:",
                        "stream": False
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    data = response.json()
                    latency = (time.time() - start_time) * 1000
                    return LLMResponse(
                        content=data.get("response", ""),
                        model=f"ollama/{ollama_model}",
                        total_tokens=data.get("eval_count", 0),
                        latency_ms=latency
                    )
        except Exception as e:
            logger.error(f"Ollama fallback generation failed: {e}")

        return LLMResponse(
            content="[LLM Unavailable] All providers failed. Check your configuration and connection.",
            model="none"
        )

    def health_check(self) -> Dict[str, Any]:
        groq_avail = False
        ollama_avail = False
        
        if self.client:
            groq_avail = True
            
        try:
            res = requests.get(f"{self.config.ollama.host}/api/tags", timeout=2)
            if res.status_code == 200:
                ollama_avail = True
        except:
            pass
        
        return {
            "groq": {"available": groq_avail},
            "ollama": {"available": ollama_avail}
        }

def get_llm():
    return LLMWrapper()
