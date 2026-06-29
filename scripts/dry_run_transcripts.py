#!/usr/bin/env python3
"""Dry-run transcript routing — no mic, no tool execution, no browser."""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.layer_pipeline import LayerPipeline
from src.core.logging_config import setup_logging
from src.core.ollama_client import OllamaClient
from src.core.settings_loader import load_settings
from src.core.transcript_resolver import resolve_transcript

# Phrases from real session logs + common mishearings
DEFAULT_PHRASES = [
    "En nuit.",
    "L.E.N.U.I.T.",
    "They call this love.",
    "I think they call this love, play it.",
    "Buy VideoClub.",
    "Play En nuit by VideoClub.",
    "Play playdate by malenie martizen",
    "Yes.",
    "Open Task Manager.",
    "Ask Manager.",
    "Play Samjha One by Arigit Singh.",
    "Play Dedicate by Taylor Swift",
    "Pahla Pyaar by Kabir Singh",
    "A Sunday I'll Find My Way Home by Carol and Tuesday",
    "pause",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run Jarvis transcript routing")
    parser.add_argument("phrases", nargs="*", help="Transcripts to test")
    parser.add_argument("--with-llm", action="store_true", help="Enable layer pipeline (Ollama + DDG)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    settings = load_settings()
    settings["browse_enabled"] = False
    settings["layer0_search_engines"] = []

    layer_pipeline = None
    if args.with_llm:
        ollama = OllamaClient(settings)
        if not ollama.check_model():
            print("WARNING: Ollama model not available — layer pipeline disabled")
        else:
            layer_pipeline = LayerPipeline(settings, ollama, None, None, {})
            settings["layer0_search_enabled"] = True

    phrases = args.phrases or DEFAULT_PHRASES
    print(f"\n{'='*72}")
    print(f"Dry-run: {len(phrases)} phrases (with_llm={args.with_llm})")
    print(f"{'='*72}\n")

    ok = 0
    for phrase in phrases:
        r = resolve_transcript(phrase, settings, layer_pipeline)
        tool = r.route.get("tool_name") if r.route else "-"
        args_str = r.route.get("arguments") if r.route else {}
        status = "OK" if r.route and not r.ignored else ("SKIP" if r.ignored else "MISS")
        if r.route and not r.ignored:
            ok += 1
        print(f"[{status:4}] {phrase!r}")
        if r.normalized != phrase.strip():
            print(f"       norm: {r.normalized!r}")
        print(f"       path={r.path} tool={tool} args={args_str}")
        if r.plan_verb:
            print(f"       layer1: verb={r.plan_verb} phrase={r.plan_phrase!r} conf={r.plan_confidence}")
        print()

    print(f"Resolved: {ok}/{len(phrases)}")
    return 0 if ok >= len(phrases) * 0.6 else 1


if __name__ == "__main__":
    raise SystemExit(main())
