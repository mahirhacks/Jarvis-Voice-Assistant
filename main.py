"""Jarvis 3.0 — Hands-free voice assistant entry point."""

import asyncio
import logging
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("CT2_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.core.agent import JarvisAgent
from src.core.layer_pipeline import LayerPipeline
from src.core.logging_config import setup_logging
from src.core.ollama_client import OllamaClient
from src.core.session import Session
from src.core.settings_loader import load_settings
from src.core.tool_registry import ToolRegistry
from src.core.version_check import check_python_version
from src.music.enrich_pipeline import EnrichPipeline
from src.music.entity_lookup import EntityLookup
from src.music.search_cache import SearchCache
from src.music.vocabulary_matcher import load_vocabulary
from src.music.youtube_search import YouTubeSearch
from src.tools.music import MusicTools
from src.tools.web_browse import get_browser
from src.voice import tts
from src.voice.audio_devices import validate_microphone
from src.voice.stt import STTEngine
from src.voice.wake_word import WakeWordListener

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging(logging.INFO)

    version = (sys.version_info.major, sys.version_info.minor)
    settings = load_settings()
    required = (
        settings.get("python_required_major", 3),
        settings.get("python_required_minor", 11),
    )
    if not check_python_version(version, required):
        msg = f"Jarvis 3.0 requires Python {required[0]}.{required[1]} or newer (you have {version[0]}.{version[1]})."
        print(msg)
        sys.exit(1)

    logger.info("[Startup] Python %d.%d — loading modules.", *version)
    tts.configure(settings)

    logger.info("[Startup] Probing microphone...")
    try:
        mic = validate_microphone(settings)
    except RuntimeError as e:
        logger.error("%s", e)
        tts.speak(str(e), blocking=True)
        sys.exit(1)

    wake = WakeWordListener(settings, mic)
    try:
        stt = STTEngine(settings, mic)
    except RuntimeError as e:
        tts.speak(str(e), blocking=True)
        sys.exit(1)

    ollama = OllamaClient(settings)
    if not ollama.check_model():
        msg = f"Model not installed. Run: ollama pull {settings.get('llm_model', 'qwen3.5:4b')}"
        logger.error(msg)
        tts.speak(msg, blocking=True)
    else:
        if not ollama.warm_up():
            logger.warning("[Startup] LLM warm-up failed — will retry on first request.")

    session = Session()
    cache = SearchCache(settings)
    youtube = YouTubeSearch(settings)
    vocabulary = load_vocabulary(settings.get("music_vocabulary_path", "config/music_vocabulary.json"))
    entity_lookup = EntityLookup(settings, cache)
    enrich = EnrichPipeline(settings, cache, entity_lookup, youtube, vocabulary)

    registry = ToolRegistry(settings)
    music_tools = MusicTools(settings, session, cache, youtube, enrich, stt)
    registry.set_music_tools(music_tools)

    browser = get_browser(settings) if settings.get("browse_enabled", True) else None
    layer_pipeline = (
        LayerPipeline(settings, ollama, browser, entity_lookup, vocabulary)
        if settings.get("layer_pipeline_enabled", True)
        else None
    )
    logger.info("[Startup] Layer pipeline enabled (qwen3.5:4b).")

    agent = JarvisAgent(
        settings, wake, stt, ollama, registry, session,
        layer_pipeline=layer_pipeline,
    )

    tts.speak("Jarvis online.", blocking=True)
    print('[Jarvis] Listening for "Hey Jarvis"... (Ctrl+C to quit)')
    logger.info("[Startup] Ready — say \"Hey Jarvis\" to give a command.")
    asyncio.run(agent.run())


if __name__ == "__main__":
    main()
