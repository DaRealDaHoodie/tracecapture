"""Canonical training example schema for Qwen chat SFT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SYSTEM_DEFAULT = (
    "You are a careful coding agent. Think step by step inside <think>...</think>, "
    "then either call a tool or give a final answer. Prefer small, verifiable steps. "
    "Use tools when you need to inspect or change the codebase. Do not invent tool "
    "results — wait for real observations."
)


@dataclass
class Example:
    """One supervised chat example ready for Qwen SFT."""

    messages: list[dict[str, str]]
    teacher: str
    source_dataset: str
    source_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def wrap_think(cot: str | None, action: str) -> str:
    """Combine reasoning + action into Qwen-style assistant content."""
    cot = (cot or "").strip()
    action = (action or "").strip()
    if cot:
        return f"<think>\n{cot}\n</think>\n{action}".strip()
    # Still emit think tags so the student learns the habit
    return f"<think>\n\n</think>\n{action}".strip()


def make_example(
    *,
    user: str,
    assistant: str,
    teacher: str,
    source_dataset: str,
    source_id: str = "",
    system: str | None = None,
    history: list[dict[str, str]] | None = None,
    meta: dict[str, Any] | None = None,
) -> Example | None:
    user = (user or "").strip()
    assistant = (assistant or "").strip()
    if not user or not assistant:
        return None
    # Keep a low bar so short prompts still train; junk is filtered elsewhere.
    if len(assistant) < 4:
        return None

    messages: list[dict[str, str]] = [
        {"role": "system", "content": (system or SYSTEM_DEFAULT).strip()},
    ]
    if history:
        for m in history:
            role = m.get("role")
            content = (m.get("content") or "").strip()
            if role in {"user", "assistant", "tool"} and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant})
    return Example(
        messages=messages,
        teacher=teacher,
        source_dataset=source_dataset,
        source_id=str(source_id),
        meta=meta or {},
    )


def looks_like_junk(text: str) -> bool:
    t = text.lower().strip()
    if not t:
        return True
    junk_markers = (
        "queue-operation",
        "as an ai language model",
        "***",
        "decoy",
        "placeholder",
    )
    if any(m in t for m in junk_markers) and len(t) < 80:
        return True
    return False
