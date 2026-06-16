"""
OpenAI-compatible API wrapper for the COT project.

Supports any OpenAI-compatible endpoint, including:
- DeepSeek Official API (https://api.deepseek.com/v1)
- Official OpenAI API
- Other third-party compatible services
"""
import os
from typing import List, Dict, Any, Optional

from openai import OpenAI

from .base import BaseModel


class OpenAIModel(BaseModel):
    """OpenAI-compatible API model wrapper."""

    def __init__(
        self,
        model_name: str = "deepseek-v4-flash",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model_name, **kwargs)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key is required. Set OPENAI_API_KEY env var or pass --api_key."
            )
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        n: int = 1,
        **kwargs
    ) -> List[str]:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            n=n,
            **kwargs
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop: Optional[List[str]] = None,
        n: int = 1,
        **kwargs
    ) -> List[str]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
            n=n,
            **kwargs
        )
        return [choice.message.content for choice in response.choices]

    def get_model_info(self) -> Dict[str, Any]:
        info = super().get_model_info()
        info["provider"] = "deepseek_openai_compatible"
        return info
