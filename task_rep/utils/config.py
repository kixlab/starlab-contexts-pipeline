from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

@dataclass
class ModelConfig:
    """VLM/Embedding provider + model config."""
    category: Literal["vlm", "embedding"]
    provider: Literal["openai", "anthropic"]
    model: Optional[str] = None

    def __post_init__(self):
        """Initialize default model based on provider (only if not explicitly set)."""
        if self.category not in ["vlm", "embedding"]:
            raise ValueError(f"Invalid category: {self.category}")
        if self.model is not None:
            return
        if self.provider == "openai":
            if self.category == "vlm":
                self.model = "gpt-4o-mini"
            elif self.category == "embedding":
                self.model = "text-embedding-3-small"
        elif self.provider == "anthropic":
            if self.category == "vlm":
                self.model = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
            elif self.category == "embedding":
                raise ValueError(f"Embedding models are not supported for Anthropic")
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

    def cache_suffix(self) -> str:
        model_name = self.model.split("/")[-1].replace("-", "").replace("_", "")
        return f"{model_name}"

    def to_dict(self) -> Dict[str, Any]:
        return {"category": self.category, "provider": self.provider, "model": self.model}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelConfig":
        return ModelConfig(
            category=data["category"],
            provider=data["provider"],
            model=data.get("model"),
        )