"""Convert heterogeneous HF agent/SFT rows into canonical Qwen examples."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Iterator

from src.schema import looks_like_junk, make_example, wrap_think

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (dict, list)):
        return json.dumps(x, ensure_ascii=False)
    return str(x)


def _messages_from_sharegpt_or_chat(row: dict[str, Any]) -> list[dict[str, str]] | None:
    """Try common chat layouts: messages, conversations, traj, etc."""
    for key in ("messages", "conversation", "conversations", "traj", "trajectory"):
        if key not in row or row[key] is None:
            continue
        raw = row[key]
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if not isinstance(raw, list) or not raw:
            continue
        out: list[dict[str, str]] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or m.get("from") or m.get("speaker")
            content = m.get("content") or m.get("value") or m.get("text") or ""
            if role in ("human", "user", "Human"):
                role = "user"
            elif role in ("gpt", "assistant", "bot", "Assistant", "model"):
                role = "assistant"
            elif role in ("function", "tool", "observation", "environment"):
                role = "tool"
            elif role == "system":
                role = "system"
            else:
                continue
            content = _as_str(content).strip()
            if content:
                out.append({"role": role, "content": content})
        if out:
            return out
    return None


def _split_last_assistant(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]], str, str] | None:
    """Return (history_without_last_asst, user_or_context, last_assistant)."""
    if not messages:
        return None
    # Find last assistant
    last_i = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_i = i
            break
    if last_i is None:
        return None
    asst = messages[last_i]["content"]
    prior = messages[:last_i]
    # Build a single user string from prior non-system content for simple SFT,
    # or keep multi-turn history if we have clear user/assistant alternation.
    system = [m for m in prior if m["role"] == "system"]
    rest = [m for m in prior if m["role"] != "system"]
    if not rest:
        return None
    # Prefer multi-turn: last user before assistant is the prompt; rest is history
    user_text = ""
    history: list[dict[str, str]] = []
    # Drop leading system from history; system injected later
    for m in rest:
        if m["role"] == "user":
            user_text = m["content"]
            # previous turns already in history
        elif m["role"] in ("assistant", "tool"):
            if user_text and history:
                pass
            history.append(m)
        # rebuild properly:
    # Simpler robust approach: pack everything before last assistant as one user context
    # except keep structured multi-turn when possible.
    if len(rest) >= 1 and rest[-1]["role"] == "user":
        user = rest[-1]["content"]
        hist = rest[:-1]
        return hist, user, asst
    # Fallback: dump transcript as user context
    packed = []
    for m in rest:
        packed.append(f"[{m['role'].upper()}]\n{m['content']}")
    return [], "\n\n".join(packed), asst


def _extract_cot_and_action(assistant: str) -> tuple[str, str]:
    """Pull think/cot blocks if present; else empty cot + full action."""
    text = assistant.strip()
    # <think>...</think>
    m = re.search(r"<think>(.*?)</think>\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # <thinking>...</thinking>
    m = re.search(r"<thinking>(.*?)</thinking>\s*(.*)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # reasoning_content style already separated upstream
    return "", text


def _tool_action_from_glint_output(output: Any, output_type: str) -> str:
    if output_type == "tool_use" or (isinstance(output, dict) and ("name" in output or "tool" in output)):
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except json.JSONDecodeError:
                return output
        name = ""
        args: Any = {}
        if isinstance(output, dict):
            name = output.get("name") or output.get("tool") or output.get("tool_name") or "tool"
            args = (
                output.get("arguments")
                or output.get("args")
                or output.get("input")
                or output.get("parameters")
                or {k: v for k, v in output.items() if k not in {"name", "tool", "tool_name", "type"}}
            )
        return (
            f"<tool_call>\n"
            f'{{"name": {json.dumps(name)}, "arguments": {json.dumps(args, ensure_ascii=False)}}}\n'
            f"</tool_call>"
        )
    return _as_str(output)


# ---------------------------------------------------------------------------
# per-dataset converters
# ---------------------------------------------------------------------------


def convert_fable_glint(row: dict[str, Any], teacher: str = "fable") -> Iterator:
    # Flat merged schema
    if "context" in row and ("cot" in row or "output" in row or "completion" in row):
        context = _as_str(row.get("context"))
        cot = _as_str(row.get("cot"))
        output = row.get("output")
        output_type = _as_str(row.get("output_type") or "")
        if output is None and row.get("completion"):
            completion = _as_str(row.get("completion"))
            # try pull think from completion
            c_cot, c_act = _extract_cot_and_action(completion)
            cot = cot or c_cot
            action = c_act
        else:
            action = _tool_action_from_glint_output(output, output_type)
        if looks_like_junk(context) or looks_like_junk(action):
            return
        asst = wrap_think(cot, action)
        ex = make_example(
            user=context,
            assistant=asst,
            teacher=teacher,
            source_dataset="Glint-Research/Fable-5-traces",
            source_id=_as_str(row.get("uid") or row.get("session") or ""),
            meta={"output_type": output_type},
        )
        if ex:
            yield ex
        return

    # Pi / agent messages
    msgs = _messages_from_sharegpt_or_chat(row)
    if msgs:
        yield from _examples_from_messages(msgs, teacher, "Glint-Research/Fable-5-traces", row)


def convert_crownelius(row: dict[str, Any], teacher: str = "fable") -> Iterator:
    raw = row.get("row_json")
    if isinstance(raw, str):
        try:
            inner = json.loads(raw)
        except json.JSONDecodeError:
            return
    elif isinstance(raw, dict):
        inner = raw
    else:
        return
    # Reuse glint/chat converters on inner
    yielded = False
    for ex in convert_fable_glint(inner, teacher=teacher):
        ex.source_dataset = "Crownelius/Complete-FABLE.5-traces-2M"
        ex.meta["first_source"] = _as_str(row.get("first_source_dataset"))
        yield ex
        yielded = True
    if not yielded:
        msgs = _messages_from_sharegpt_or_chat(inner)
        if msgs:
            yield from _examples_from_messages(
                msgs, teacher, "Crownelius/Complete-FABLE.5-traces-2M", row
            )


def convert_generic_chat(row: dict[str, Any], teacher: str, source: str) -> Iterator:
    # reasoning_content + content (Nemotron-style)
    if "messages" in row or "conversations" in row or "conversation" in row:
        msgs = _messages_from_sharegpt_or_chat(row)
        if msgs:
            # Inject reasoning into last assistant if separate field
            rc = row.get("reasoning_content") or row.get("reasoning") or row.get("cot")
            if rc and msgs and msgs[-1]["role"] == "assistant":
                cot, action = _extract_cot_and_action(msgs[-1]["content"])
                if not cot:
                    msgs[-1]["content"] = wrap_think(_as_str(rc), action or msgs[-1]["content"])
            yield from _examples_from_messages(msgs, teacher, source, row)
            return

    # prompt / response
    prompt = row.get("prompt") or row.get("instruction") or row.get("input") or row.get("question")
    response = row.get("response") or row.get("output") or row.get("answer") or row.get("completion")
    cot = row.get("reasoning_content") or row.get("reasoning") or row.get("cot") or row.get("think")
    if prompt and response:
        asst = wrap_think(_as_str(cot), _as_str(response)) if cot else wrap_think(*_extract_cot_and_action(_as_str(response)))
        if looks_like_junk(_as_str(prompt)) or looks_like_junk(asst):
            return
        ex = make_example(
            user=_as_str(prompt),
            assistant=asst,
            teacher=teacher,
            source_dataset=source,
            source_id=_as_str(row.get("id") or row.get("uuid") or ""),
        )
        if ex:
            yield ex
        return

    # agent trajectory fields
    for key in ("trajectory", "traj", "steps", "trace"):
        if key in row and row[key]:
            msgs = _messages_from_sharegpt_or_chat({key: row[key]})
            if msgs:
                yield from _examples_from_messages(msgs, teacher, source, row)
                return


def convert_coderforge(row: dict[str, Any], teacher: str = "open_swe") -> Iterator:
    # Prefer success when present
    success = row.get("success")
    if success is None:
        success = row.get("passed")
    if success is None:
        success = row.get("resolved")
    if success is False or success == 0 or str(success).lower() in {"false", "fail", "failed"}:
        # keep ~15% fails later via sampling; skip hard fails here by default
        if hash(_as_str(row.get("id") or row.get("instance_id") or row)) % 100 >= 15:
            return
    yield from convert_generic_chat(row, teacher, "togethercomputer/CoderForge-Preview")


def convert_open_swe(row: dict[str, Any], teacher: str = "open_swe") -> Iterator:
    yield from convert_generic_chat(row, teacher, "nvidia/Open-SWE-Traces")


def convert_nebius(row: dict[str, Any], teacher: str = "open_swe") -> Iterator:
    yield from convert_generic_chat(row, teacher, "nebius/SWE-rebench-openhands-trajectories")


def convert_nemotron(row: dict[str, Any], teacher: str, source: str) -> Iterator:
    yield from convert_generic_chat(row, teacher, source)


def convert_gpt55(row: dict[str, Any], teacher: str = "gpt55") -> Iterator:
    # Teich / agent-traces style JSONL events sometimes
    if "message" in row and isinstance(row["message"], dict):
        # session event stream — handle in batch elsewhere; single event weak
        role = row["message"].get("role")
        content = _as_str(row["message"].get("content"))
        if role == "assistant" and content:
            # need prior context; skip orphan events
            return
    yield from convert_generic_chat(row, teacher, "AletheiaResearch/GPT-5.5-Codex")


def convert_claude_merged(row: dict[str, Any], teacher: str = "fable") -> Iterator:
    yield from convert_generic_chat(row, teacher, "nlile/misc-merged-claude-code-traces-v1")


def convert_local_jsonl(row: dict[str, Any], teacher: str, source: str) -> Iterator:
    yield from convert_generic_chat(row, teacher, source)


def _examples_from_messages(
    messages: list[dict[str, str]],
    teacher: str,
    source: str,
    row: dict[str, Any],
) -> Iterator:
    split = _split_last_assistant(messages)
    if not split:
        return
    history, user, asst_raw = split
    cot, action = _extract_cot_and_action(asst_raw)
    asst = wrap_think(cot, action) if (cot or action) else asst_raw
    if looks_like_junk(user) or looks_like_junk(asst):
        return
    # history should not include system
    hist = [m for m in history if m["role"] != "system"]
    ex = make_example(
        user=user,
        assistant=asst,
        teacher=teacher,
        source_dataset=source,
        source_id=_as_str(row.get("id") or row.get("session_id") or row.get("uid") or ""),
        history=hist if hist else None,
        meta={"n_history": len(hist)},
    )
    if ex:
        yield ex


CONVERTERS = {
    "coderforge": convert_coderforge,
    "open_swe": convert_open_swe,
    "nebius_openhands": convert_nebius,
    "nemotron_swe": lambda r: convert_nemotron(r, "open_swe", "nvidia/Nemotron-SFT-SWE-v2"),
    "fable_glint": convert_fable_glint,
    "fable_crownelius": convert_crownelius,
    "claude_code_merged": convert_claude_merged,
    "gpt55_codex": convert_gpt55,
    "nemotron_agentic": lambda r: convert_nemotron(r, "tools", "nvidia/Nemotron-SFT-Agentic-v2"),
    "nemotron_opencode": lambda r: convert_nemotron(r, "tools", "nvidia/Nemotron-SFT-OpenCode-v1"),
    "grok_local": lambda r: convert_local_jsonl(r, "grok", "local/grok"),
}


def convert_rows(teacher_id: str, rows: Iterable[dict[str, Any]]) -> Iterator:
    fn = CONVERTERS.get(teacher_id)
    if fn is None:
        for r in rows:
            yield from convert_generic_chat(r, teacher_id, teacher_id)
        return
    for r in rows:
        try:
            yield from fn(r)
        except Exception:
            continue
