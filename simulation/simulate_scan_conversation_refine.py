#!/usr/bin/env python3
"""Refined SCAN callback simulator built on the shared refine pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .refine import (
        StyleConfig,
        ScenarioHooks,
        build_standard_agent_prompt,
        load_config,
        normalize_to_titles,
        run_refine_pipeline,
    )
else:  # pragma: no cover - direct script execution (`python simulation/...py`)
    import sys

    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from simulation.refine import (  # type: ignore[no-redef]
        StyleConfig,
        ScenarioHooks,
        build_standard_agent_prompt,
        load_config,
        normalize_to_titles,
        run_refine_pipeline,
    )

CONFIG_FILE = Path(__file__).with_name("config").joinpath("scan_conversation_config.yaml")

CATEGORY_1 = "Category 1: Universal Compliance"
CATEGORY_2 = "Category 2: Intent Triggered Guidelines"
CATEGORY_3 = "Category 3: Condition Triggered Guidelines"
SCAN_TITLES = {"cat1": CATEGORY_1, "cat2": CATEGORY_2, "cat3": CATEGORY_3}


def build_agent_system_prompt(guidelines: dict[str, object]) -> str:
    return build_standard_agent_prompt(
        "You are the SCAN Health virtual assistant focused on callback scheduling and voicemail routing.",
        guidelines,
        SCAN_TITLES,
    )


def persona_to_user_system_prompt(persona: dict[str, object]) -> str:
    caller = persona.get("caller", {}) or {}
    member = persona.get("member", {}) or {}
    availability = persona.get("availability", {}) or {}
    reason = persona.get("call_reason", {}) or {}
    confirmation = persona.get("confirmation", {}) or {}
    voicemail = persona.get("voicemail", {}) or {}

    def fmt_window(win: dict[str, object]) -> str:
        label = win.get("label", "")
        start = win.get("start_local", "")
        end = win.get("end_local", "")
        note = win.get("note", "")
        pref = "preferred" if win.get("is_preferred") else "backup"
        return f"{label} ({pref}): {start}-{end} local. Note: {note}".strip()

    windows = [
        fmt_window(win)
        for win in availability.get("windows", []) or []
        if isinstance(win, dict)
    ]

    lines: list[str] = [
        "You are a tester role-playing a caller interacting with the SCAN Health virtual assistant.",
        "Speak naturally in first person, never reveal you're testing.",
        f"Intent: {persona.get('intent','unknown')}. Tone: {persona.get('tone','neutral')}. English proficiency: {persona.get('language_proficiency','native')}.",
        f"Preference for human agent: {persona.get('prefers_human_agent','medium')}.",
    ]

    caller_name = f"{caller.get('first_name','')} {caller.get('last_name','')}".strip()
    if caller_name:
        lines.append(f"Caller name: {caller_name} (pronouns: {caller.get('pronouns','')}).")
    lines.append(
        f"Primary callback number: {caller.get('phone_country_code','')} {caller.get('phone_number_only','')} "
        f"(last four {caller.get('phone_last_four','')})."
    )
    if caller.get("alternate_number"):
        lines.append(f"Alternate number available: {caller['alternate_number']}.")
    if caller.get("email_address"):
        lines.append(f"Email: {caller['email_address']}.")

    if windows:
        lines.append("Callback windows: " + " | ".join(windows))

    raw_reason = reason.get("raw_statement") or reason.get("concise_summary")
    if raw_reason:
        lines.append(f"Reason for callback (keep under 50 words): {raw_reason}")

    if confirmation.get("requires_last_four_only"):
        lines.append("Only confirm phone numbers using the last four digits.")

    if voicemail.get("wants_voicemail"):
        lines.append("Caller ultimately wants to leave a voicemail instead of choosing a slot.")

    for note in persona.get("notes", []) or []:
        lines.append(f"Note: {note}")

    lines.append("Stay in character, reply with one utterance per user turn.")
    return "\n".join(lines)


SCAN_STYLE = StyleConfig(
    discriminator_intro="You are a realism discriminator for SCAN callback callers.",
    human_cues="Human cues: mild fillers, soft hedges, brief pauses, natural rhythm.",
    synthetic_cues="Synthetic cues: overly formal, rigid, repetitive, robotic phrasing.",
    rewrite_instructions=(
        "Rewrite the caller line to sound more organic but keep the intent and facts identical.\n"
        "Favor mild fillers, informal tone, brief pauses, and gentle self-corrections.\n"
        'Return strict JSON only: {"rewrite": "<string>" }.'
    ),
)


ROOT = Path(__file__).resolve().parent.parent

SCAN_SCENARIO = ScenarioHooks(
    name="SCAN",
    category_titles=SCAN_TITLES,
    build_agent_prompt=build_agent_system_prompt,
    build_user_prompt=persona_to_user_system_prompt,
    normalize_category=lambda cat: normalize_to_titles(cat, SCAN_TITLES),
    style=SCAN_STYLE,
    oracle_path=ROOT / "guidelines" / "SCAN" / "oracle.json",
    modified_path=ROOT / "guidelines" / "SCAN" / "modified.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine SCAN callback simulations.")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="Path to YAML config.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    run_refine_pipeline(cfg, SCAN_SCENARIO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
