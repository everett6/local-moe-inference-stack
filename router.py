"""Prompt-bucket routing -- kept in spirit with your original LM Studio
script, minus anything that talked to LM Studio's REST API."""
import json
import os
from typing import Any, Dict

from config import Paths


def prompt_bucket(prompt: str) -> str:
    text = (prompt or "").strip().lower()
    if not text:
        return "quick"
    word_count = len(text.split())
    code_terms = ["debug", "code", "fix", "bug", "error", "trace", "optimi", "benchmark", "profile", "cli", "api", "python", "c++", "rust", "cuda", "gpu"]
    long_terms = ["plan", "design", "architect", "compare", "analyze", "write", "summarize", "explain", "draft", "proposal"]
    if word_count <= 12 or any(t in text for t in ["what", "why", "who", "when", "where", "yes", "no", "fix"]):
        return "quick"
    if any(t in text for t in code_terms):
        return "code"
    if any(t in text for t in long_terms):
        return "analysis"
    if word_count >= 80:
        return "long"
    return "analysis"


def load_router_state(paths: Paths) -> Dict[str, Any]:
    if not os.path.exists(paths.router_state_path):
        return {"buckets": {}}
    try:
        with open(paths.router_state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"buckets": {}}


def save_router_state(paths: Paths, state: Dict[str, Any]) -> None:
    try:
        with open(paths.router_state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass
