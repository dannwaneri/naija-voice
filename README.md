# Nigerian-accented voice cloning pipeline

Two-stage TTS pipeline that generates speech in a target voice while preserving Nigerian accent:

```
text
  → Afro-TTS (accent + phonetics)   [XTTS-v2 fine-tuned on African voices]
  → RVC       (target voice timbre) [Retrieval-based Voice Conversion]
  → WAV
```

Built for content creation where existing TTS options either wash out the Nigerian accent (XTTS, F5-TTS) or produce unreliable audio quality (YarnGPT).

## Status

Working. See [PLAYBOOK.md](PLAYBOOK.md) for realistic quality expectations, tuned parameters, and input formatting rules.

Known ceiling: ~95% intelligibility on simple content, ~85% on slang-heavy content, voice drift on multi-minute takes. Not a code problem — it's the training-data-per-billion-params floor of 2024-era open-source TTS. See "Upgrade paths" below.

## Layout

- `speak_afro.py` — main pipeline (Afro-TTS + RVC). **Use this.**
- `speak.py` — legacy pipeline (YarnGPT + RVC). Kept for reference.
- `convert.py` — standalone RVC inference on any input WAV.
- `prep_training_audio.py` — combines dataset clips into a single training WAV for RVC.
- `rvc_convert.py` — early prototype, superseded by `convert.py`.
- `PLAYBOOK.md` — production usage and lessons learned.
- `models/` (gitignored) — RVC model checkpoints.
- `outputs/` (gitignored) — generated audio.
- `Applio/` (gitignored) — [Applio](https://github.com/IAHispano/Applio) RVC fork, cloned locally.

## External dependencies (not in this repo)

The pipeline shells out to two sibling projects:

- **YarnGPT** at `C:\Users\DELL\yarngpt\` (only used by legacy `speak.py`) — [saheedniyi/YarnGPT](https://github.com/saheedniyi02/yarngpt)
- **Afro-TTS** at `C:\Users\DELL\afro-tts\` — [intronhealth/afro-tts](https://huggingface.co/intronhealth/afro-tts)

Each has its own venv because their dep trees conflict.

## Setup (short version)

1. Clone this repo
2. Install [Applio](https://github.com/IAHispano/Applio) into `./Applio/`
3. Create `.venv/` with Python 3.11, install PyTorch CPU + Applio requirements
4. Train an RVC model on ~20+ min of your target voice (see Applio docs), drop artifacts into `models/<speaker>/`
5. Install Afro-TTS at `../afro-tts/` (uses [Coqui XTTS](https://github.com/coqui-ai/TTS) with the Intron Health checkpoint)
6. Record a 60-90 sec single-take reference clip, save to `speakers/<speaker>/reference.wav`
7. Update paths in `speak_afro.py` if your layout differs

## Usage

```powershell
.\.venv\Scripts\activate
python speak_afro.py --text "Your script here"
python speak_afro.py --text-file script.txt --out outputs/episode1.wav
```

See `python speak_afro.py --help` for all flags.

## Upgrade paths (2026)

The 2024-era models this pipeline uses have real ceilings. Modern alternatives worth swapping in when you have time:

- **[VoxCPM2](https://voxcpm.net/)** — 2B params, SOTA voice similarity, cross-lingual with accent preservation
- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)** — Alibaba, 5M hrs training, best zero-shot cloning
- **[Fish Audio S2](https://fish.audio/)** — designed for long-form (audiobook) consistency, 15-sec reference
- **[dots.tts](https://github.com/HiLab-git/dots.tts)** — streaming, accent-preserving

Any of these could replace the Afro-TTS stage with tighter accent capture and less voice drift on longer content.

## License

MIT.

The trained voice model (`models/`) is intentionally not published — a voice fingerprint is biometric data and should not be committed to public repositories. If you fork this, generate your own from your own voice.
