"""
Single-stage pipeline: text -> Qwen3-TTS (Nigerian accent, clean phonetics,
long-form stable, voice cloned from reference clip) -> WAV

Qwen3-TTS's 5M-hour training data captures voice + accent tightly enough from
a single reference clip that no downstream RVC stage is needed. Simpler pipeline,
better quality than YarnGPT/Afro-TTS + RVC combined.

Usage:
    python speak_qwen.py --text "Hello, this is Daniel."
    python speak_qwen.py --text-file script.txt --out outputs/episode1.wav
    python speak_qwen.py --text "..." --reference other_voice.wav --reference-text other_voice.txt
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def trim_tail_guard(wav_path: Path, silence_thresh: float = 0.015, min_silence_ms: int = 80):
    """After appending a tail-guard word, trim it off. Strategy:
      1. Strip trailing silence.
      2. From there, walk backwards through the guard word (non-silence).
      3. Find the silence gap that precedes it (between real content and guard).
      4. Cut at the START of that gap so the real sentence keeps its natural trailing silence.
    Falls back to trimming a fixed 700ms of audio if no such structure is found."""
    import soundfile as sf
    import numpy as np
    audio, sr = sf.read(str(wav_path))
    mono = audio.mean(axis=-1) if audio.ndim > 1 else audio
    win = int(0.02 * sr)
    energy = np.array([np.abs(mono[i:i+win]).mean() for i in range(0, len(mono), win)])
    is_silent = energy < silence_thresh
    min_silence_wins = max(1, min_silence_ms // 20)

    # 1. Walk back past trailing silence
    idx = len(is_silent) - 1
    while idx >= 0 and is_silent[idx]:
        idx -= 1
    # 2. Walk back past the guard word (non-silence)
    while idx >= 0 and not is_silent[idx]:
        idx -= 1
    # 3. Walk back through the silence gap before the guard, finding its start
    silence_end = idx + 1  # first silent frame going backwards
    while idx >= 0 and is_silent[idx]:
        idx -= 1
    silence_start = idx + 1

    if silence_end - silence_start >= min_silence_wins:
        # Cut at start of the pre-guard silence + a small tail (50ms) to preserve natural ending
        cut = min(len(mono), (silence_start + 3) * win)
    else:
        # Structure not found → fixed 700ms trim (guard word + short silence)
        cut = max(0, len(mono) - int(0.7 * sr))
    sf.write(str(wav_path), audio[:cut] if audio.ndim == 1 else audio[:cut, :], sr)

BASE = Path(__file__).parent
QWEN_BASE = Path(r"C:\Users\DELL\qwen-tts")
QWEN_PY = QWEN_BASE / ".venv" / "Scripts" / "python.exe"
QWEN_SCRIPT = QWEN_BASE / "generate_qwen.py"
DEFAULT_REF_AUDIO = Path(r"C:\Users\DELL\xtts-clone\speakers\daniel\my_voice_full.wav")
DEFAULT_REF_TEXT = QWEN_BASE / "reference_transcript.txt"
OUTPUTS = BASE / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="Text to speak")
    ap.add_argument("--text-file", help="Path to a text file to speak")
    ap.add_argument("--out", help="Output WAV path")
    ap.add_argument("--reference", default=str(DEFAULT_REF_AUDIO),
                    help="Speaker reference WAV")
    ap.add_argument("--reference-text", default=str(DEFAULT_REF_TEXT),
                    help="Transcript file for the reference clip")
    ap.add_argument("--split-threshold", type=int, default=300,
                    help="Auto-split into sentences when input exceeds this char count")
    ap.add_argument("--gap-ms", type=int, default=300)
    ap.add_argument("--tail-guard", default="Okay.",
                    help="Word appended to every sentence to protect the real final word from compression. Trimmed off before saving. Empty string disables.")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed for reproducible takes. If not set, each run is random.")
    ap.add_argument("--takes", type=int, default=1,
                    help="Generate N takes (saved as <out>_take1.wav, _take2.wav, ...). Voice varies take-to-take; pick the best.")
    args = ap.parse_args()

    if not args.text and not args.text_file:
        sys.exit("Provide --text or --text-file")
    text = args.text or Path(args.text_file).read_text(encoding="utf-8").strip()

    OUTPUTS.mkdir(exist_ok=True)
    if args.out:
        out_path = Path(args.out).resolve()
    else:
        n = len(list(OUTPUTS.glob("daniel_qwen_*.wav"))) + 1
        out_path = OUTPUTS / f"daniel_qwen_{n:03d}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if len(text) > args.split_threshold:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    else:
        sentences = [text]

    # Append tail-guard to each sentence so its real final word is never last.
    guard = args.tail_guard.strip()
    if guard:
        sentences = [s + " " + guard if s and s[-1] in ".!?" else s + ". " + guard for s in sentences]

    print(f"\nQwen3-TTS -> {out_path.name}  [{len(sentences)} sentence(s)]")
    if len(sentences) == 1:
        cmd = [
            str(QWEN_PY), str(QWEN_SCRIPT),
            "--text", sentences[0],
            "--ref-audio", args.reference,
            "--ref-text-file", args.reference_text,
            "--out", str(out_path),
        ]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        subprocess.run(cmd, check=True)
        if guard:
            trim_tail_guard(out_path)
    else:
        import soundfile as sf
        import numpy as np
        tmpdir = Path(tempfile.mkdtemp(prefix="qwen_"))
        parts, sr = [], None
        for i, s in enumerate(sentences, 1):
            preview = s[:60] + ("…" if len(s) > 60 else "")
            print(f"   [{i}/{len(sentences)}] {preview}")
            p = tmpdir / f"s{i:03d}.wav"
            cmd = [
                str(QWEN_PY), str(QWEN_SCRIPT),
                "--text", s,
                "--ref-audio", args.reference,
                "--ref-text-file", args.reference_text,
                "--out", str(p),
            ]
            if args.seed is not None:
                # Offset seed per sentence so they don't all sound identical
                cmd += ["--seed", str(args.seed + i)]
            subprocess.run(cmd, check=True)
            if guard:
                trim_tail_guard(p)
            audio, sr = sf.read(str(p))
            parts.append(audio)
        gap = np.zeros(int(args.gap_ms / 1000 * sr), dtype=parts[0].dtype)
        combined = []
        for i, p in enumerate(parts):
            combined.append(p)
            if i < len(parts) - 1:
                combined.append(gap)
        sf.write(str(out_path), np.concatenate(combined), sr)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\nFinal: {out_path}")


if __name__ == "__main__":
    main()
