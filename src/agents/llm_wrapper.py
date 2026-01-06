"""
LLM Wrapper for TRIALPULSE NEXUS 10X
Phase 5.1: LLM Setup

Provides unified interface for:
- Ollama (local, primary)
- Groq (cloud, backup/fast)

Features:
- Automatic fallback
- Retry logic
- Response caching
- Token tracking
- Error handling
"""

import os
import sys
import json
import time
import hashlib
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from functools import lru_cache

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.llm_config import (
    LLMConfig, LLMProvider, get_config, 
    SYSTEM_PROMPTS, TASK_MODEL_MAPPING
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Structured LLM response"""
    content: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None and len(self.content) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
            "timestamp": self.timestamp,
            "error": self.error,
            "success": self.success
        }


class LLMWrapper:
    """
    Unified LLM wrapper with fallback and caching
    
    Usage:
        llm = LLMWrapper()
        response = llm.generate("What is SDV?")
        print(response.content)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize LLM wrapper"""
        self.config = config or get_config()
        self._cache: Dict[str, LLMResponse] = {}
        self._ollama_client = None
        self._groq_client = None
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "ollama_calls": 0,
            "groq_calls": 0,
            "errors": 0,
            "total_tokens": 0
        }
        
        # Initialize clients
        self._init_clients()
    
    def _init_clients(self):
        """Initialize LLM clients"""
        # Initialize Groq first (more reliable for cloud deployment)
        if self.config.groq.api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.config.groq.api_key)
                logger.info("✅ Groq API configured")
            except Exception as e:
                logger.warning(f"⚠️ Groq not available: {e}")
                self._groq_client = None
        else:
            logger.info("ℹ️ Groq API key not configured")
            self._groq_client = None
        
        # Initialize Ollama (may not be available on cloud)
        try:
            import ollama
            self._ollama_client = ollama.Client(host=self.config.ollama.host)
            # Test connection with timeout
            self._ollama_client.list()
            logger.info(f"✅ Ollama connected at {self.config.ollama.host}")
        except Exception as e:
            logger.info(f"ℹ️ Ollama not available: {e}")
            self._ollama_client = None
        
        # Validate at least one provider is available
        if self._ollama_client is None and self._groq_client is None:
            logger.warning("⚠️ No LLM providers available! AI features will be limited.")
    
    def _get_cache_key(self, prompt: str, system_prompt: str, model: str) -> str:
        """Generate cache key for request"""
        content = f"{prompt}|{system_prompt}|{model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_cache(self, cache_key: str) -> Optional[LLMResponse]:
        """Check if response is cached"""
        if not self.config.enable_cache:
            return None
        
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Check TTL
            cached_time = datetime.fromisoformat(cached.timestamp)
            age = (datetime.now() - cached_time).total_seconds()
            if age < self.config.cache_ttl:
                self._stats["cache_hits"] += 1
                cached.cached = True
                return cached
            else:
                # Expired
                del self._cache[cache_key]
        return None
    
    def _call_ollama(
        self,
        prompt: str,
        system_prompt: str,
        model: Optional[str] = None
    ) -> LLMResponse:
        """Call Ollama API"""
        if not self._ollama_client:
            return LLMResponse(
                content="",
                provider="ollama",
                model=model or self.config.ollama.model,
                error="Ollama not available"
            )
        
        model = model or self.config.ollama.model
        start_time = time.time()
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self._ollama_client.chat(
                model=model,
                messages=messages,
                options=self.config.ollama.options
            )
            
            latency_ms = (time.time() - start_time) * 1000
            self._stats["ollama_calls"] += 1
            
            # Extract token counts if available
            prompt_tokens = response.get("prompt_eval_count", 0)
            completion_tokens = response.get("eval_count", 0)
            
            return LLMResponse(
                content=response["message"]["content"],
                provider="ollama",
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            self._stats["errors"] += 1
            return LLMResponse(
                content="",
                provider="ollama",
                model=model,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    def _call_groq(
        self,
        prompt: str,
        system_prompt: str,
        model: Optional[str] = None
    ) -> LLMResponse:
        """Call Groq API"""
        if not self._groq_client:
            return LLMResponse(
                content="",
                provider="groq",
                model=model or self.config.groq.model,
                error="Groq not available"
            )
        
        model = model or self.config.groq.model
        start_time = time.time()
        
        try:
            response = self._groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.groq.max_tokens,
                temperature=self.config.groq.temperature
            )
            
            latency_ms = (time.time() - start_time) * 1000
            self._stats["groq_calls"] += 1
            
            usage = response.usage
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider="groq",
                model=model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                total_tokens=usage.total_tokens if usage else 0,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            self._stats["errors"] += 1
            return LLMResponse(
                content="",
                provider="groq",
                model=model,
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
        use_cache: bool = True,
        provider: Optional[LLMProvider] = None
    ) -> LLMResponse:
        """
        Generate LLM response
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (uses default if None)
            model: Specific model to use
            task_type: Task type for model selection
            use_cache: Whether to use caching
            provider: Force specific provider
            
        Returns:
            LLMResponse with content and metadata
        """
        self._stats["total_requests"] += 1
        
        # Get system prompt
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPTS.get("default", "")
        elif system_prompt in SYSTEM_PROMPTS:
            system_prompt = SYSTEM_PROMPTS[system_prompt]
        
        # Get model based on task type
        if model is None and task_type:
            model = TASK_MODEL_MAPPING.get(task_type)
        
        # Check cache
        if use_cache and self.config.enable_cache:
            cache_key = self._get_cache_key(prompt, system_prompt, model or "default")
            cached = self._check_cache(cache_key)
            if cached:
                logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                return cached
        
        # Determine provider order
        providers = []
        if provider:
            providers = [provider]
        else:
            providers = [self.config.primary_provider]
            if self.config.fallback_provider:
                providers.append(self.config.fallback_provider)
        
        # Try each provider
        response = None
        for prov in providers:
            if prov == LLMProvider.OLLAMA:
                response = self._call_ollama(prompt, system_prompt, model)
            elif prov == LLMProvider.GROQ:
                response = self._call_groq(prompt, system_prompt, model)
            
            if response and response.success:
                break
            else:
                logger.warning(f"Provider {prov.value} failed: {response.error if response else 'Unknown'}")
        
        # Cache successful response
        if response and response.success and use_cache and self.config.enable_cache:
            cache_key = self._get_cache_key(prompt, system_prompt, model or "default")
            self._cache[cache_key] = response
        
        # Update stats
        if response:
            self._stats["total_tokens"] += response.total_tokens
        
        # Log if enabled
        if self.config.log_requests:
            logger.info(f"LLM Request: {response.provider}/{response.model} "
                       f"({response.latency_ms:.0f}ms, {response.total_tokens} tokens)")
        
        return response
    
    def generate_with_context(
        self,
        prompt: str,
        context: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response with RAG context
        
        Args:
            prompt: User prompt
            context: List of context documents
            system_prompt: System prompt
            **kwargs: Additional generate() arguments
        """
        # Build context-enhanced prompt
        context_text = "\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(context)])
        enhanced_prompt = f"""Based on the following context, answer the question.

{context_text}

Question: {prompt}

Answer based on the context provided. If the context doesn't contain relevant information, say so."""
        
        return self.generate(enhanced_prompt, system_prompt=system_prompt, **kwargs)
    
    def generate_json(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON response
        
        Args:
            prompt: User prompt
            schema: Expected JSON schema (for validation)
            **kwargs: Additional generate() arguments
            
        Returns:
            Parsed JSON dict
        """
        json_prompt = f"""{prompt}

Respond ONLY with valid JSON. No markdown, no explanation, just JSON."""
        
        response = self.generate(json_prompt, **kwargs)
        
        if not response.success:
            return {"error": response.error, "raw": response.content}
        
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            
            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "raw": response.content}
    
    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts
        
        Args:
            prompts: List of prompts
            system_prompt: Shared system prompt
            **kwargs: Additional generate() arguments
            
        Returns:
            List of LLMResponse objects
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, system_prompt=system_prompt, **kwargs)
            responses.append(response)
        return responses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        stats = self._stats.copy()
        stats["cache_size"] = len(self._cache)
        stats["cache_hit_rate"] = (
            self._stats["cache_hits"] / self._stats["total_requests"] 
            if self._stats["total_requests"] > 0 else 0
        )
        return stats
    
    def clear_cache(self):
        """Clear response cache"""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def health_check(self) -> Dict[str, Any]:
        """Check health of LLM providers"""
        status = {
            "ollama": {"available": False, "models": []},
            "groq": {"available": False}
        }
        
        # Check Ollama
        if self._ollama_client:
            try:
                models = self._ollama_client.list()
                status["ollama"]["available"] = True
                status["ollama"]["models"] = [m["name"] for m in models.get("models", [])]
            except Exception as e:
                status["ollama"]["error"] = str(e)
        
        # Check Groq
        if self._groq_client:
            try:
                # Simple test call
                test_response = self._call_groq("Hello", "Reply with 'OK'", None)
                status["groq"]["available"] = test_response.success
                if not test_response.success:
                    status["groq"]["error"] = test_response.error
            except Exception as e:
                status["groq"]["error"] = str(e)
        
        return status
    
    def __repr__(self):
        return (f"LLMWrapper(primary={self.config.primary_provider.value}, "
                f"ollama={'✓' if self._ollama_client else '✗'}, "
                f"groq={'✓' if self._groq_client else '✗'})")


# Convenience function
def get_llm(config: Optional[LLMConfig] = None) -> LLMWrapper:
    """Get configured LLM wrapper instance"""
    return LLMWrapper(config)