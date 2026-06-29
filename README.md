# Jarvis 3.0

Hands-free, voice-only local assistant for Windows. Wake word → speech-to-text → layered intent routing → tool execution → TTS. Uses [Ollama](https://ollama.com) (`qwen3.5:4b`) for ambiguous commands; most everyday phrases route instantly without the LLM.

## Features

- **Wake word** — "Hey Jarvis" via openWakeWord
- **STT** — faster-whisper (GPU or CPU)
- **Music** — YouTube search, vocabulary corrections, MusicBrainz/Wikidata enrichment
- **Media & browser** — play/pause, volume, tabs, search, navigation
- **System** — open apps, workflows, shutdown/restart/lock (with voice confirmation)
- **Layered routing** — fast path → vocabulary → heuristics → web search → LLM (only when needed)

## Requirements

- **Windows 10/11**
- **Python 3.11+**
- **Ollama** running locally
- Microphone (configure device index after install)
- Optional: NVIDIA GPU for faster-whisper (`stt_device_preference: cuda`)

## Quick start

```powershell
git clone https://github.com/YOUR_USERNAME/jarvis-3.0.git
cd jarvis-3.0

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install firefox
ollama pull qwen3.5:4b
```

### Microphone

List input devices and note your mic index:

```powershell
.\venv\Scripts\python.exe scripts\list_mics.py
```

Copy the local settings template and set your device IDs:

```powershell
copy config\settings.local.json.example config\settings.local.json
# Edit config\settings.local.json — set wake_microphone_device_id and command_microphone_device_id
```

`config/settings.local.json` is gitignored and overrides `config/settings.json`.

### Allowed apps

Edit `config/allowed_apps.json` to match apps on your machine. Paths can use Windows env vars (e.g. `%LOCALAPPDATA%`).

### Run

```powershell
.\run_jarvis.ps1
```

Jarvis speaks **"Jarvis online"**, then listens for **"Hey Jarvis"**.

On first run, openWakeWord downloads small model files from GitHub (internet required once).

## Architecture

```
Transcript → normalize → Tier 0 fast router
         → music vocabulary → phrase heuristics (bare song, questions, URLs)
         → Layer 0 search (vocab, DDG, Google/Bing/Yahoo)
         → Layer 1 intent (qwen3.5:4b JSON loops)
         → Layer 2 execution (same model, narrowed schema when ambiguous)
```

| Tier | Role |
|------|------|
| **Tier 0** | Exact commands: pause, volume, play/open prefixes, workflows |
| **Vocab** | STT corrections from `config/music_vocabulary.json` |
| **Heuristics** | Bare `title artist`, questions, `search …`, URLs |
| **Layer 0** | Web/music entity context (skipped for obvious commands) |
| **Layer 1–2** | LLM intent + tool call when confidence is low |

## Tools

**Music:** `search_youtube`, `play_video`, `next_candidate`, `previous_candidate`, `replay_last`

**Media:** play/pause, volume, next/previous track, mute, fullscreen

**Browser:** tabs, scroll, zoom, search, navigate

**System:** shutdown, restart, lock (voice confirmation)

**Apps:** `open_app` — see `config/allowed_apps.json`

**Workflows:** `study_mode`, `assignment_mode`, `bug_bounty_mode`, `play_music` — see `config/workflows/`

## Tests

```powershell
pytest tests/ -q
```

Routing stress test (233 phrases, no tool execution, no LLM by default):

```powershell
python scripts/stress_test_phrases.py
python scripts/stress_test_phrases.py --with-llm   # includes Ollama layers
```

Dry-run single transcripts:

```powershell
python scripts/dry_run_transcripts.py
```

## Project layout

```
jarvis-3.0/
├── config/           # settings, apps, vocabulary, workflows
├── data/             # runtime logs/temp (gitignored contents)
├── scripts/          # mic list, stress tests, dry-run
├── src/
│   ├── core/         # agent, routing, layers, Ollama
│   ├── music/        # YouTube, enrichment, vocabulary
│   ├── tools/        # browser, media, system, apps
│   └── voice/        # wake word, STT, TTS
├── tests/
├── main.py
└── run_jarvis.ps1
```

## Configuration

| File | Purpose |
|------|---------|
| `config/settings.json` | Committed defaults |
| `config/settings.local.json` | Your machine overrides (gitignored) |
| `config/allowed_apps.json` | Apps Jarvis may launch |
| `config/music_vocabulary.json` | Song/artist STT corrections |
| `config/workflows/*.json` | Multi-step voice workflows |

Runtime caches (`music_entity_cache.json`, `music_corrections_cache.json`) are created automatically and are gitignored.

## License

MIT — see [LICENSE](LICENSE).
