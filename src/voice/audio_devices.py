"""Microphone device probing and safe InputStream configuration."""

import logging
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


def list_input_devices() -> list[dict]:
    """Return input-capable devices with index, name, channels, sample rate."""
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            devices.append({
                "index": idx,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": int(dev["default_samplerate"]),
            })
    return devices


def print_input_devices() -> None:
    print("\n=== Available microphones ===")
    for d in list_input_devices():
        print(
            f"  [{d['index']}] {d['name']} "
            f"(channels={d['channels']}, {d['sample_rate']} Hz)"
        )
    default_in = sd.default.device[0]
    if default_in is not None:
        print(f"\n  Windows default input device index: {default_in}")
    print("  Set wake_microphone_device_id and command_microphone_device_id in config/settings.json\n")


def _probe_stream(device_id: int, channels: int, samplerate: int) -> bool:
    try:
        with sd.InputStream(
            device=device_id,
            channels=channels,
            samplerate=samplerate,
            dtype="int16",
            blocksize=256,
        ):
            pass
        return True
    except Exception:
        return False


def probe_input_device(device_id: int, preferred_rate: int = 16000) -> tuple[int, int]:
    """Find (channels, samplerate) that open successfully on this device."""
    info = sd.query_devices(device_id)
    if info["max_input_channels"] < 1:
        raise ValueError(f"Device {device_id} ({info['name']}) has no microphone input.")

    max_ch = int(info["max_input_channels"])
    default_sr = int(info["default_samplerate"])
    channel_candidates = []
    for ch in (1, 2, max_ch):
        if ch not in channel_candidates and ch <= max_ch:
            channel_candidates.append(ch)

    rate_candidates = []
    for sr in (preferred_rate, default_sr, 44100, 48000):
        if sr not in rate_candidates:
            rate_candidates.append(sr)

    for ch in channel_candidates:
        for sr in rate_candidates:
            if _probe_stream(device_id, ch, sr):
                return ch, sr

    raise ValueError(
        f"Could not open microphone on device {device_id} ({info['name']}). "
        "Try a different device index in config/settings.json."
    )


def resolve_input_device(settings: dict) -> int:
    """Pick a working input device: configured -> default -> first input."""
    configured = settings.get("command_microphone_device_id")
    if configured is None:
        configured = settings.get("wake_microphone_device_id")

    candidates: list[int] = []
    if configured is not None:
        candidates.append(int(configured))
    default_in = sd.default.device[0]
    if default_in is not None and int(default_in) not in candidates:
        candidates.append(int(default_in))
    for d in list_input_devices():
        if d["index"] not in candidates:
            candidates.append(d["index"])

    last_error: Optional[Exception] = None
    for device_id in candidates:
        try:
            probe_input_device(device_id)
            info = sd.query_devices(device_id)
            logger.info(
                "[Audio] Using device %d: %s",
                device_id, info["name"],
            )
            return device_id
        except Exception as exc:
            last_error = exc
            logger.warning("[Audio] Device %d failed probe: %s", device_id, exc)

    print_input_devices()
    raise RuntimeError(
        "No working microphone found. Update wake_microphone_device_id and "
        "command_microphone_device_id in config/settings.json."
    ) from last_error


class MicConfig:
    """Resolved microphone stream parameters."""

    def __init__(self, device_id: int, channels: int, samplerate: int):
        self.device_id = device_id
        self.channels = channels
        self.samplerate = samplerate

    @classmethod
    def from_settings(cls, settings: dict) -> "MicConfig":
        device_id = resolve_input_device(settings)
        channels, samplerate = probe_input_device(device_id)
        return cls(device_id, channels, samplerate)


def to_mono_int16(audio: np.ndarray) -> np.ndarray:
    """Convert int16 audio to mono 1-D array."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[1] == 1:
            return audio[:, 0]
        return audio.mean(axis=1).astype(np.int16)
    return np.squeeze(audio)


def validate_microphone(settings: dict) -> MicConfig:
    """Probe mic at startup; log config or raise with device list."""
    mic = MicConfig.from_settings(settings)
    info = sd.query_devices(mic.device_id)
    logger.info(
        "[Audio] Microphone OK — device=%d (%s), channels=%d, rate=%d Hz",
        mic.device_id, info["name"], mic.channels, mic.samplerate,
    )
    return mic
