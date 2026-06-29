"""OpenWakeWord-based wake word listener."""

import logging

import numpy as np
import sounddevice as sd

from src.voice.audio_devices import MicConfig, to_mono_int16

logger = logging.getLogger(__name__)

_MODEL_DOWNLOAD_NAMES = {
    "hey_jarvis": "hey_jarvis_v0.1",
    "alexa": "alexa_v0.1",
    "hey_mycroft": "hey_mycroft_v0.1",
    "hey_rhasspy": "hey_rhasspy_v0.1",
}


def _ensure_wake_models(model_key: str) -> None:
    from openwakeword.utils import download_models

    download_name = _MODEL_DOWNLOAD_NAMES.get(model_key, model_key)
    logger.info("[WakeWord] Ensuring models are present (%s)...", download_name)
    download_models(model_names=[download_name])


class WakeWordListener:
    def __init__(self, settings: dict, mic: MicConfig) -> None:
        from openwakeword.model import Model as OWWModel

        self._model_name: str = settings.get("wake_word_model", "hey_jarvis")
        self._mic = mic
        self._threshold: float = settings.get("wake_word_threshold", 0.2)

        _ensure_wake_models(self._model_name)

        logger.info(
            "[WakeWord] Loading '%s' (device=%d, threshold=%.2f)",
            self._model_name, self._mic.device_id, self._threshold,
        )
        self._model = OWWModel(
            wakeword_models=[self._model_name],
            inference_framework="onnx",
        )

    def wait_for_wake_word(self) -> None:
        chunk_size = 1280
        with sd.InputStream(
            samplerate=self._mic.samplerate,
            channels=self._mic.channels,
            dtype="int16",
            blocksize=chunk_size,
            device=self._mic.device_id,
        ) as stream:
            while True:
                audio, _ = stream.read(chunk_size)
                audio_np = to_mono_int16(np.asarray(audio))
                try:
                    self._model.predict(audio_np)
                except MemoryError:
                    logger.warning("[WakeWord] MemoryError — resetting model.")
                    from openwakeword.model import Model as OWWModel
                    _ensure_wake_models(self._model_name)
                    self._model = OWWModel(
                        wakeword_models=[self._model_name],
                        inference_framework="onnx",
                    )
                    continue

                for name, score in self._model.prediction_buffer.items():
                    if score and score[-1] >= self._threshold:
                        logger.info("[WakeWord] Detected '%s' (score=%.3f)", name, score[-1])
                        self._model.reset()
                        return
