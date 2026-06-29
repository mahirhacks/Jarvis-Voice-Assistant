"""faster-whisper STT with VAD recording and fallback chain."""

import logging
import time
from typing import Optional

import numpy as np
import sounddevice as sd

from src.voice.audio_devices import MicConfig, to_mono_int16

logger = logging.getLogger(__name__)

_STT_FALLBACK_CHAIN = [
    ("large-v3", "cuda", "float16"),
    ("large-v3-turbo", "cuda", "float16"),
    ("large-v3-turbo", "cuda", "int8_float16"),
    ("medium", "cuda", "int8_float16"),
    ("small", "cpu", "int8"),
]


def _build_stt_chain(settings: dict) -> list[tuple[str, str, str]]:
    primary = settings.get("stt_model", "small")
    device = settings.get("stt_device_preference", "cuda")
    gpu_types = settings.get("stt_gpu_compute_types", ["float16", "int8_float16"])
    fallbacks = settings.get("stt_fallback_models", ["medium", "small", "base"])
    cpu_type = settings.get("stt_cpu_compute_type", "int8")

    chain: list[tuple[str, str, str]] = []
    if device == "cuda":
        for ct in gpu_types:
            chain.append((primary, "cuda", ct))
        for fb in fallbacks:
            if fb != primary:
                chain.append((fb, "cuda", "int8_float16"))
    for model in [primary] + fallbacks:
        entry = (model, "cpu", cpu_type)
        if entry not in chain:
            chain.append(entry)
    return chain if chain else _STT_FALLBACK_CHAIN


class STTEngine:
    def __init__(self, settings: dict, mic: MicConfig) -> None:
        self._settings = settings
        self._mic = mic
        self._beam_size: int = settings.get("stt_beam_size", 1)
        self._best_of: int = settings.get("stt_best_of", 1)
        self._condition_on_prev: bool = settings.get("stt_condition_on_previous_text", False)
        self._language: Optional[str] = settings.get("stt_language")
        self._initial_prompt: Optional[str] = settings.get("stt_initial_prompt")
        self._hotwords: Optional[str] = self._resolve_hotwords(settings)
        self._no_speech_threshold: float = settings.get("stt_no_speech_threshold", 0.6)
        self._log_prob_threshold: float = settings.get("stt_log_prob_threshold", -1.0)
        self._compression_ratio_threshold: float = settings.get(
            "stt_compression_ratio_threshold", 2.4
        )
        self._patience: float = settings.get("stt_patience", 2.0)
        self._repetition_penalty: float = settings.get("stt_repetition_penalty", 1.1)
        self._no_repeat_ngram_size: int = settings.get("stt_no_repeat_ngram_size", 3)
        self._vad_filter: bool = settings.get("stt_vad_filter", True)
        self._temperature = self._resolve_temperature(settings)
        self._pre_roll: float = settings.get("vad_pre_roll_seconds", 0.6)
        self._min_record: float = settings.get("vad_min_record_seconds", 0.8)
        self._max_record: float = settings.get("vad_max_record_seconds", 12)
        self._silence_dur: float = settings.get("vad_silence_duration_seconds", 0.85)
        self._start_thresh: int = settings.get("vad_start_threshold", 450)
        self._stop_thresh: int = settings.get("vad_stop_threshold", 250)
        self._min_volume: int = settings.get("minimum_audio_volume", 350)
        self._speech_start_timeout: float = settings.get("stt_speech_start_timeout_seconds", 8.0)
        self._cpu_threads: int = settings.get("stt_cpu_threads", 1)
        self._num_workers: int = settings.get("stt_num_workers", 1)
        self._model = self._load_model_with_fallback(settings)

    @staticmethod
    def _resolve_hotwords(settings: dict) -> Optional[str]:
        explicit = settings.get("stt_hotwords")
        if explicit:
            return explicit
        if not settings.get("stt_hotwords_from_vocabulary", True):
            return None
        from src.music.vocabulary_matcher import load_vocabulary

        vocab_path = settings.get("music_vocabulary_path", "config/music_vocabulary.json")
        vocabulary = load_vocabulary(vocab_path)
        terms: list[str] = []
        for entry in vocabulary.values():
            if isinstance(entry, dict) and entry.get("canonical_form"):
                terms.append(str(entry["canonical_form"]))
        return ", ".join(terms) if terms else None

    @staticmethod
    def _resolve_temperature(settings: dict):
        """Beam search uses 0; a fallback tuple retries harder cases."""
        raw = settings.get("stt_temperature_fallback")
        if raw is not None:
            if isinstance(raw, (list, tuple)):
                return tuple(raw)
            return raw
        return settings.get("stt_temperature", 0)

    def _load_model_with_fallback(self, settings: dict):
        from faster_whisper import WhisperModel

        chain = _build_stt_chain(settings)
        last_error = None
        for model_name, device, compute_type in chain:
            try:
                logger.info("[STT] Trying %s on %s (%s)...", model_name, device, compute_type)
                model = WhisperModel(
                    model_name, device=device, compute_type=compute_type,
                    cpu_threads=self._cpu_threads, num_workers=self._num_workers,
                )
                logger.info("[STT] Loaded %s on %s (%s)", model_name, device, compute_type)
                return model
            except Exception as e:
                last_error = e
                logger.warning("[STT] Failed %s on %s: %s", model_name, device, e)
        raise RuntimeError(
            "STT model loading failed. Try run_jarvis.ps1 or set stt_model to 'small'."
        ) from last_error

    @staticmethod
    def _resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == 16000:
            return audio
        target_len = int(len(audio) * 16000 / orig_sr)
        if target_len < 1:
            return audio
        x_old = np.linspace(0.0, 1.0, len(audio))
        x_new = np.linspace(0.0, 1.0, target_len)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def record_and_transcribe(self) -> Optional[str]:
        # Whisper expects 16 kHz mono — record at 16 kHz when the mic supports it.
        sample_rate = 16000
        if self._mic.samplerate != sample_rate:
            logger.info(
                "[STT] Resampling command audio %d Hz -> %d Hz",
                self._mic.samplerate, sample_rate,
            )
        mic_rate = self._mic.samplerate
        chunk_duration = 0.05
        chunk_size = max(1, int(mic_rate * chunk_duration))
        pre_roll_chunks = int(self._pre_roll / chunk_duration)

        pre_roll_buffer: list = []
        recording: list = []
        is_recording = False
        silence_start = None
        total_duration = 0.0
        listen_started = time.monotonic()
        last_wait_log = listen_started

        try:
            with sd.InputStream(
                samplerate=mic_rate,
                channels=self._mic.channels,
                dtype="int16",
                blocksize=chunk_size,
                device=self._mic.device_id,
            ) as stream:
                while True:
                    if not is_recording:
                        elapsed = time.monotonic() - listen_started
                        if elapsed >= self._speech_start_timeout:
                            logger.info(
                                "[STT] No speech within %.1fs — giving up",
                                self._speech_start_timeout,
                            )
                            return None
                        if time.monotonic() - last_wait_log >= 3.0:
                            logger.info("[STT] Waiting for speech (%.0fs)...", elapsed)
                            last_wait_log = time.monotonic()

                    audio, _ = stream.read(chunk_size)
                    audio_np = to_mono_int16(np.asarray(audio))
                    volume = int(np.abs(audio_np).mean())

                    if not is_recording:
                        pre_roll_buffer.append(audio_np.copy())
                        if len(pre_roll_buffer) > pre_roll_chunks:
                            pre_roll_buffer.pop(0)
                        if volume >= self._start_thresh:
                            is_recording = True
                            logger.info("[STT] Speech detected (volume=%d)", volume)
                            recording.extend(pre_roll_buffer)
                            recording.append(audio_np.copy())
                            total_duration = len(recording) * chunk_duration
                            silence_start = None
                    else:
                        recording.append(audio_np.copy())
                        total_duration += chunk_duration
                        if volume < self._stop_thresh:
                            if silence_start is None:
                                silence_start = total_duration
                            elif total_duration - silence_start >= self._silence_dur:
                                break
                        else:
                            silence_start = None
                        if total_duration >= self._max_record:
                            break
        except Exception:
            logger.exception("[STT] Recording error")
            return None

        if not recording:
            return None

        if total_duration < self._min_record:
            all_audio = np.concatenate(recording)
            if int(np.abs(all_audio).mean()) < self._min_volume:
                return None

        all_audio = np.concatenate(recording).astype(np.float32) / 32768.0
        all_audio = self._resample_to_16k(all_audio, mic_rate)

        logger.info("[STT] Transcribing %.1fs of audio...", total_duration)
        try:
            segments, info = self._model.transcribe(
                all_audio,
                beam_size=self._beam_size,
                best_of=self._best_of,
                patience=self._patience,
                repetition_penalty=self._repetition_penalty,
                no_repeat_ngram_size=self._no_repeat_ngram_size,
                temperature=self._temperature,
                condition_on_previous_text=self._condition_on_prev,
                language=self._language,
                initial_prompt=self._initial_prompt,
                hotwords=self._hotwords,
                no_speech_threshold=self._no_speech_threshold,
                log_prob_threshold=self._log_prob_threshold,
                compression_ratio_threshold=self._compression_ratio_threshold,
                vad_filter=self._vad_filter,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            logger.info(
                "[STT] '%s' (lang=%s, prob=%.2f)",
                text, info.language, getattr(info, "language_probability", 0.0),
            )
            return text if text else None
        except Exception:
            logger.exception("[STT] Transcription failed")
            return None
