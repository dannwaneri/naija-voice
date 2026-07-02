# Daniel Voice Pipeline — Production Playbook

## The shipped pipeline (current — Qwen3-TTS single-stage)

```
text → Qwen3-TTS (voice + accent + intelligibility, all in one)
     → final WAV
```

No RVC needed. Qwen3-TTS's 5M-hour training captures voice + accent tightly enough from a single reference clip.

### Runtime: use Colab GPU, not local CPU

- **Colab T4 GPU (recommended):** open `qwen_colab.ipynb` in Colab, set T4 GPU runtime. Generation ~5-15 sec per line.
- **Local CPU:** works but prohibitively slow when `min_new_tokens` is forced high enough to prevent truncation (10+ min per line). Only use for very short lines.

### Critical settings (baked into `qwen_colab.ipynb`)

- **Model:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- **Patched:** `min_new_tokens: 2` (hardcoded in stock package) → `kwargs.pop("min_new_tokens", 2)`. Without this patch, model truncates mid-sentence on ~half of inputs.
- **Seed:** 3 for Daniel's voice (found via seed sweep)
- **Temperature:** 0.7 (lower = more consistent voice match)
- **Min tokens per char:** 15 (forces model to speak full sentence)
- **Reference:** `my_voice_full.wav` (59.2s single-take)
- **Reference transcript:** `reference_transcript.txt` (Whisper-generated)

Tested 2026-07-02 on Colab GPU: voice matches, accent preserved, full sentence delivered, generation fast.

## Legacy pipelines (kept for reference)

- `speak_afro.py` — Afro-TTS + RVC. Works, but voice drifts on multi-sentence content. Superseded by `speak_qwen.py`.
- `speak.py` — YarnGPT + RVC. Superseded twice. Kept as historical reference only.

## Quick start

```powershell
C:\Users\DELL\rvc-pipeline\.venv\Scripts\activate
python C:\Users\DELL\rvc-pipeline\speak_qwen.py --text "Your script here"
```

Output lands in `C:\Users\DELL\rvc-pipeline\outputs\`.

Useful flags:
- `--text-file script.txt` — read from file instead of CLI string
- `--out path.wav` — explicit output path
- `--reference other.wav --reference-text other.txt` — clone a different voice
- `--split-threshold 300` — long input auto-splits into sentences (default 300 chars)

## Realistic expectations

- **Simple content (intros, narration, explainers): ~95% intelligible.** Ship it.
- **Slang-heavy, abbreviation-heavy text: ~85% intelligible.** Preprocess first or expect to cherry-pick.
- **Voice match: solid** once the RVC params are at the tuned defaults (pitch -2, index 1.0, protect 0.0).
- **Nigerian accent: preserved** (Afro-TTS's training data, not the reference clip alone).

## Input formatting rules (the actual leverage)

What hurts quality:
- ALL CAPS for emphasis → write `i cannot` not `I CANNOT`
- Abbreviations → expand them: `DM` → `direct message`, `2am` → `two in the morning`
- Raw digits → spell them out: `1500 naira` → `one thousand five hundred naira`
- Sentence-end slang → bury it mid-sentence or pad with a normal word: instead of `Not one drop.` try `Not even one drop here.`

What helps:
- Lowercase with normal punctuation
- Common, well-known proper nouns (or phonetic respelling: `Tinubu` → `Ti-noo-boo` if it gets garbled)
- Periods and commas where you actually want pauses
- Avoiding sentence-final words that need to land cleanly

## When something sounds wrong

1. Re-run — Afro-TTS is mostly deterministic but RVC has some variance
2. Adjust the input text using the rules above
3. If a specific word fails, phonetic respell only that word
4. Generate with `--keep-intermediate` to check whether the Afro-TTS stage or the RVC stage degraded it

## Files

- `speak_afro.py` — production pipeline (Afro-TTS + RVC). **Use this.**
- `speak.py` — legacy YarnGPT pipeline. Kept for reference; quality ceiling lower.
- `convert.py` — standalone RVC inference. Run on any pre-existing WAV.
- `outputs/daniel_baseline_good.wav` — the original "this sounds like me" sample. Reference floor.
- `outputs/afro_002_voiced.wav` — best known intro take. Reference ceiling.

## Backup of what works (current)

- Pipeline: `speak_qwen.py` (single-stage Qwen3-TTS, no RVC)
- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base` (~5 GB, cached in `~/.cache/huggingface/`)
- Reference clip: `C:\Users\DELL\xtts-clone\speakers\daniel\my_voice_full.wav` (59.2 sec, single continuous take)
- Reference transcript: `C:\Users\DELL\qwen-tts\reference_transcript.txt` (Whisper-generated)
- Auto-split threshold: 300 chars

## Backup for legacy fallback (if Qwen3-TTS ever breaks)

- Afro-TTS + RVC pipeline: `speak_afro.py`
- Tuned RVC params: `--pitch -4 --index-rate 1.0 --protect 0.0`
- Tuned Afro-TTS: `--gpt-cond-len 15`
- RVC model: 50 epochs, at `C:\Users\DELL\rvc-pipeline\models\daniel\daniel_50e_2950s.pth`

## Hard-learned lessons (do not relearn)

1. **Reference clip must be ONE continuous take from one recording session.** Stitching multiple clips together produces robotic output — XTTS's speaker encoder reads acoustic discontinuities (different mic distance, room tone, levels) as inconsistency and fails. Tested 2026-06-23 with a 4-clip montage: result was unusable.

2. **For better voice cloning, RE-RECORD a fresh single-take clip.** Don't try to assemble one from the existing 140 training clips.

3. **Sentence-final words get compressed.** If a critical word lands at the very end of a sentence, the model may swallow it. Bury important words mid-sentence, or pad the end with a throwaway word.

4. **Pipeline evolution:**
   - YarnGPT+RVC: ~30-40% intelligibility on hard content (unreliable pronunciation)
   - Afro-TTS+RVC: ~85% intelligibility, but voice drifts on multi-sentence content
   - Qwen3-TTS alone: voice + accent + intelligibility from single reference clip, no drift observed. **Current default.**

6. **RVC ceases being useful when upstream TTS is strong enough.** Qwen3-TTS's 5M-hour training makes RVC (trained on 20 min) net-negative — it adds artifacts rather than fixing voice. Tested 2026-07-02: `qwen_intro_qwen.wav` (no RVC) preferred over `qwen_intro.wav` (with RVC).

5. **Last word "distance" got eaten** on the connections script. Fix: end with a buffer word like "Yeah." or "basically." Then trim the buffer in audio editor if needed.

## Future upgrade vector (when you have time)

Record a fresh 60-90 second reference clip in **one continuous take**:

- Quiet room, no AC/fan/traffic
- Mic 6-8 inches from mouth, consistent
- Phone voice recorder is fine; save as WAV if possible
- Script that hits varied intonations (statements + a question + some enthusiasm + reflection)
- Save to `C:\Users\DELL\xtts-clone\speakers\daniel\my_voice_v2.wav` and update `speak_afro.py` `--reference` default if it sounds better in A/B
