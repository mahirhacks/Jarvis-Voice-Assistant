"""Text-to-speech with Kokoro primary and pyttsx3 fallback."""

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_tts_lock = threading.Lock()
_kokoro_pipeline = None
_kokoro_available = None
_settings: dict = {}
_temp_dir = Path("data/temp")


def configure(settings: dict) -> None:
    global _settings
    _settings = settings
    _temp_dir.mkdir(parents=True, exist_ok=True)


def _speak_kokoro(text: str) -> bool:
    global _kokoro_pipeline, _kokoro_available
    if _kokoro_available is False:
        return False
    try:
        if _kokoro_pipeline is None:
            from kokoro import KPipeline
            _kokoro_pipeline = KPipeline(lang_code="a")
            _kokoro_available = True
        import numpy as np
        import sounddevice as sd
        import soundfile as sf

        voice = _settings.get("tts_voice", "af_heart")
        speed = _settings.get("tts_speed", 1.0)
        sample_rate = _settings.get("tts_sample_rate", 24000)
        chunks = []
        for _, _, audio in _kokoro_pipeline(text, voice=voice, speed=speed):
            if audio is not None:
                chunks.append(audio)
        if not chunks:
            return False
        full_audio = np.concatenate(chunks)
        sd.play(full_audio, samplerate=sample_rate)
        sd.wait()
        return True
    except Exception as e:
        logger.warning("Kokoro TTS failed: %s", e)
        _kokoro_available = False
        return False


def _speak_pyttsx3(text: str) -> bool:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception as e:
        logger.warning("pyttsx3 failed: %s", e)
        return False


def _speak_engine(text: str) -> None:
    with _tts_lock:
        if not _settings.get("tts_enabled", True):
            return
        engine = _settings.get("tts_engine", "pyttsx3")
        fallback = _settings.get("tts_fallback_engine", "pyttsx3")
        if engine == "kokoro" and _speak_kokoro(text):
            return
        if fallback == "pyttsx3" or engine == "pyttsx3":
            if _speak_pyttsx3(text):
                return
        logger.warning("All TTS engines failed")


def speak(text: str, blocking: bool | None = None) -> None:
    logger.info("[TTS] %s", text)
    if not _settings.get("tts_enabled", True):
        return
    if blocking is None:
        blocking = not _settings.get("tts_non_blocking", True)
    if blocking:
        _speak_engine(text)
    else:
        threading.Thread(target=_speak_engine, args=(text,), daemon=True).start()
