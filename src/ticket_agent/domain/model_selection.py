from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSelection:
    capability: str
    primary_model: str
    fallback_models: tuple[str, ...]
