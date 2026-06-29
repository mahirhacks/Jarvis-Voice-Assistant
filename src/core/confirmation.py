"""Voice-only confirmation."""

import logging
import re
from typing import Optional, Protocol

from src.core.utils import match_confirmation_phrase
from src.voice import tts

logger = logging.getLogger(__name__)


class STTProtocol(Protocol):
    def record_and_transcribe(self) -> Optional[str]: ...


def ask_confirmation(stt: STTProtocol, message: str, max_attempts: int = 2) -> bool:
    tts.speak(message, blocking=True)

    for attempt in range(max_attempts):
        response = stt.record_and_transcribe()
        if not response:
            if attempt < max_attempts - 1:
                tts.speak("I didn't hear anything. Yes or no?", blocking=True)
            continue

        logger.info("[Confirm] '%s'", response)
        classification = match_confirmation_phrase(response)
        if classification == "affirmative":
            return True
        if classification == "negative":
            return False
        if attempt < max_attempts - 1:
            tts.speak("Sorry, I didn't understand. Yes or no?", blocking=True)

    return False


def normalize_confirmation(text: str) -> str:
    return re.sub(r"[.,!?;:'\"]+", "", text.strip().lower())
