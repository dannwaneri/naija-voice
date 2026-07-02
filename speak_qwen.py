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

    print(f"\nQwen3-TTS -> {out_path.name}  [{len(sentences)} sentence(s)]")
    if len(sentences) == 1:
        subprocess.run([
            str(QWEN_PY), str(QWEN_SCRIPT),
            "--text", sentences[0],
            "--ref-audio", args.reference,
            "--ref-text-file", args.reference_text,
            "--out", str(out_path),
        ], check=True)
    else:
        import soundfile as sf
        import numpy as np
        tmpdir = Path(tempfile.mkdtemp(prefix="qwen_"))
        parts, sr = [], None
        for i, s in enumerate(sentences, 1):
            preview = s[:60] + ("…" if len(s) > 60 else "")
            print(f"   [{i}/{len(sentences)}] {preview}")
            p = tmpdir / f"s{i:03d}.wav"
            subprocess.run([
                str(QWEN_PY), str(QWEN_SCRIPT),
                "--text", s,
                "--ref-audio", args.reference,
                "--ref-text-file", args.reference_text,
                "--out", str(p),
            ], check=True)
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
