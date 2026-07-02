# Nigerian-accented voice cloning pipeline

Single-stage TTS pipeline that generates speech in a target voice while preserving Nigerian accent:

```
text → Qwen3-TTS (voice + accent + intelligibility) → WAV
```

Built for content creation where existing options either wash out the Nigerian accent (XTTS, F5-TTS), produce unreliable audio (YarnGPT), or drift mid-audio on long content (Afro-TTS).

Historical two-stage pipelines using RVC on top of Afro-TTS or YarnGPT are kept as legacy fallbacks — see PLAYBOOK.md.

## Status

Working. See [PLAYBOOK.md](PLAYBOOK.md) for realistic quality expectations, tuned parameters, and input formatting rules.

Known ceiling: ~95% intelligibility on simple content, ~85% on slang-heavy content, voice drift on multi-minute takes. Not a code problem — it's the training-data-per-billion-params floor of 2024-era open-source TTS. See "Upgrade paths" below.

## Layout

- `speak_qwen.py` — **current pipeline (Qwen3-TTS single-stage). Use this.**
- `speak_afro.py` — legacy pipeline (Afro-TTS + RVC). Kept as fallback.
- `speak.py` — legacy pipeline (YarnGPT + RVC). Historical reference.
- `convert.py` — standalone RVC inference on any input WAV.
- `prep_training_audio.py` — combines dataset clips into a single training WAV for RVC.
- `rvc_convert.py` — early prototype, superseded by `convert.py`.
- `PLAYBOOK.md` — production usage and lessons learned.
- `models/` (gitignored) — RVC model checkpoints.
- `outputs/` (gitignored) — generated audio.
- `Applio/` (gitignored) — [Applio](https://github.com/IAHispano/Applio) RVC fork, cloned locally.

## External dependencies (not in this repo)

The pipeline shells out to sibling projects:

- **Qwen3-TTS** at `C:\Users\DELL\qwen-tts\` (current default) — [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- **Afro-TTS** at `C:\Users\DELL\afro-tts\` (legacy) — [intronhealth/afro-tts](https://huggingface.co/intronhealth/afro-tts)
- **YarnGPT** at `C:\Users\DELL\yarngpt\` (legacy) — [saheedniyi/YarnGPT](https://github.com/saheedniyi02/yarngpt)

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
python speak_qwen.py --text "Your script here"
python speak_qwen.py --text-file script.txt --out outputs/episode1.wav
```

See `python speak_qwen.py --help` for all flags.

## Upgrade paths (future)

Current pipeline uses [Qwen3-TTS 1.7B](https://github.com/QwenLM/Qwen3-TTS), which as of Q2 2026 handles voice + accent + long-form intelligibility from a single reference clip. If a better model appears, the pipeline is architecturally trivial to swap (one subprocess call).

Watch:
- **[VoxCPM2](https://voxcpm.net/)** — 2B params, SOTA voice similarity, but CUDA-first (Windows CPU currently blocked)
- **[Fish Audio S2](https://fish.audio/)** — designed for long-form audiobook consistency
- **[dots.tts](https://github.com/HiLab-git/dots.tts)** — 54ms streaming for real-time use cases

## License

MIT.

The trained voice model (`models/`) is intentionally not published — a voice fingerprint is biometric data and should not be committed to public repositories. If you fork this, generate your own from your own voice.
