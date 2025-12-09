#!/usr/bin/env python3
"""Convert one simulated conversation into an SFT training example for the judge.

The produced record mirrors the exact evaluation setting in
evaluate_simulated_conversations.py: the model receives the same
system message and user prompt (guidelines + conversation transcript)
and must reply with the strict JSON listing all violations.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_NAME = "providers.azure_gpt_call"
if MODULE_NAME not in sys.modules:
    stub = types.ModuleType(MODULE_NAME)

    def _unused_call_chat_completion(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[name-defined]
        raise RuntimeError("call_chat_completion is not available in training-data prep.")

    stub.call_chat_completion = _unused_call_chat_completion  # type: ignore[attr-defined]
    sys.modules[MODULE_NAME] = stub
    providers_pkg = importlib.import_module("providers")
    setattr(providers_pkg, "azure_gpt_call", stub)

from evaluation.shared import build_user_prompt, format_conversation, format_guidelines, infer_category_titles, normalize_category


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_violation(item: Dict[str, Any]) -> Dict[str, Any]:
    """Map raw mistake entries to the fields expected during evaluation."""

    def _first(*keys: str) -> str:
        for key in keys:
            if key in item and item[key] not in (None, ""):
                return str(item[key])
        return ""

    turn_index = int(item.get("turn_index", -1))
    category = normalize_category(
        _first(
            "guidance category",
            "guideline_category",
            "guideline type",
            "guideline_type",
        )
    )
    key = _first("guidance key", "guideline_key", "key", "guideline_type")
    phase_raw = _first("guideline_phase", "phase")
    try:
        phase = int(phase_raw)
    except Exception:
        phase = -1
    evidence = _first("evidence", "quote", "text")
    return {
        "turn_index": turn_index,
        "guidance_category": category,
        "guidance_key": key,
        "guideline_phase": phase,
        "evidence": evidence,
    }


def build_training_example(
    convo_path: Path,
    guidelines_path: Path,
) -> Dict[str, Any]:
    convo = _load_json(convo_path)
    oracle = _load_json(guidelines_path)

    message_list = convo.get("message_list") or convo.get("conversation") or []
    truth_list = convo.get("mistakes", [])
    conv_id = convo_path.stem

    cat_titles = infer_category_titles(oracle)
    guidelines_text = format_guidelines(oracle, cat_titles)
    conversation_text = format_conversation(message_list)
    user_prompt = build_user_prompt(guidelines_text, conversation_text, conv_id)

    normalized_violations = [
        _normalize_violation(it) for it in sorted(truth_list, key=lambda x: x.get("turn_index", -1))
    ]

    target_payload = {
        "conversation_id": conv_id,
        "violations": normalized_violations,
    }

    return {
        "conversation_id": conv_id,
        "input_messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "target_response": json.dumps(target_payload, ensure_ascii=False, indent=2),
        "source_conversation_file": str(convo_path),
        "guidelines_file": str(guidelines_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare one judge training example.")
    parser.add_argument("--input", required=True, help="Path to a simulated conversation JSON file.")
    parser.add_argument(
        "--guidelines",
        default="guidelines/SCAN/oracle.json",
        help="Guidelines JSON (same as evaluation).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the processed record (JSON). Defaults to stdout if omitted.",
    )
    args = parser.parse_args()

    convo_path = Path(args.input)
    guidelines_path = Path(args.guidelines)
    if not convo_path.exists():
        raise FileNotFoundError(f"Conversation file not found: {convo_path}")
    if not guidelines_path.exists():
        raise FileNotFoundError(f"Guidelines file not found: {guidelines_path}")

    record = build_training_example(convo_path, guidelines_path)
    payload = json.dumps(record, ensure_ascii=False, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
