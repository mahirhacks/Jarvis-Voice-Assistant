#!/usr/bin/env python3
"""Batch stress-test transcript routing — no tool execution."""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.layer_pipeline import LayerPipeline
from src.core.logging_config import setup_logging
from src.core.ollama_client import OllamaClient
from src.core.settings_loader import load_settings
from src.core.transcript_resolver import resolve_transcript

PHRASES_FILE = Path(__file__).parent / "stress_phrases.json"
REPORT_FILE = Path(__file__).parent / "stress_report.json"


def main() -> int:
    setup_logging(logging.WARNING)
    settings = load_settings()
    settings["browse_enabled"] = False
    settings["layer0_search_enabled"] = False

    layer_pipeline = None
    if "--with-llm" in sys.argv:
        ollama = OllamaClient(settings)
        if ollama.check_model():
            from src.music.vocabulary_matcher import load_vocabulary

            vocab = load_vocabulary(settings.get("music_vocabulary_path", "config/music_vocabulary.json"))
            layer_pipeline = LayerPipeline(settings, ollama, None, None, vocab)
            settings["layer0_search_enabled"] = True

    phrases = json.loads(PHRASES_FILE.read_text(encoding="utf-8"))
    print(f"Stress test: {len(phrases)} phrases\n")

    results = []
    path_counts: Counter = Counter()
    misses = []

    for phrase in phrases:
        r = resolve_transcript(phrase, settings, layer_pipeline)
        path_counts[r.path] += 1
        entry = {
            "phrase": phrase,
            "normalized": r.normalized,
            "path": r.path,
            "tool": r.route.get("tool_name") if r.route else None,
            "args": r.route.get("arguments") if r.route else None,
            "ignored": r.ignored,
        }
        results.append(entry)
        if r.ignored:
            status = "SKIP"
        elif r.route:
            status = "OK"
        else:
            status = "MISS"
            misses.append(entry)
        print(f"[{status:4}] {phrase[:55]:55} | {r.path:18} | {entry['tool'] or '-'}")

    resolved = sum(1 for r in results if r["tool"] and not r["ignored"])
    skipped = sum(1 for r in results if r["ignored"])
    report = {
        "total": len(phrases),
        "resolved": resolved,
        "skipped": skipped,
        "missed": len(misses),
        "path_counts": dict(path_counts),
        "misses": misses,
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'='*72}")
    print(f"Resolved: {resolved}/{len(phrases)} ({100*resolved/len(phrases):.1f}%)")
    print(f"Skipped:  {skipped}")
    print(f"Missed:   {len(misses)}")
    print(f"Paths:    {dict(path_counts)}")
    print(f"Report:   {REPORT_FILE}")
    return 0 if len(misses) <= len(phrases) * 0.15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
