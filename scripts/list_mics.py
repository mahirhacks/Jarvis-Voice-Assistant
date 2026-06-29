"""List microphones — run from project root: python scripts/list_mics.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.voice.audio_devices import print_input_devices

if __name__ == "__main__":
    print_input_devices()
