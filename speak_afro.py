"""
End-to-end pipeline: text -> Afro-TTS (Nigerian accent, clean phonetics) -> RVC (Daniel's voice) -> WAV

Replaces YarnGPT with Afro-TTS for cleaner intelligibility while still preserving
Nigerian accent. RVC stage still locks in Daniel's voice timbre.

Usage:
    python speak_afro.py --text "Hello, this is Daniel."
    python speak_afro.py --text-file script.txt --out outputs/episode1.wav
    python speak_afro.py --text "..." --keep-intermediate
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

RVC_BASE = Path(__file__).parent
AFRO_BASE = Path(r"C:\Users\DELL\afro-tts")
XTTS_VENV_PY = Path(r"C:\Users\DELL\xtts-clone\venv\Scripts\python.exe")
AFRO_SCRIPT = AFRO_BASE / "speak.py"
OUTPUTS = RVC_BASE / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="Text to speak")
    ap.add_argument("--text-file", help="Path to a text file to speak")
    ap.add_argument("--out", help="Final output WAV path")
    ap.add_argument("--reference", default=r"C:\Users\DELL\xtts-clone\speakers\daniel\my_voice_full.wav",
                    help="Speaker reference WAV for Afro-TTS")
    ap.add_argument("--gpt-cond-len", type=int, default=15,
                    help="Seconds of reference used for Afro-TTS voice conditioning")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="Keep the Afro-TTS intermediate audio")
    # RVC params
    ap.add_argument("--split-threshold", type=int, default=200,
                    help="Auto-split into sentences when input exceeds this char count")
    ap.add_argument("--gap-ms", type=int, default=300)
    ap.add_argument("--pitch", type=int, default=-4)
    ap.add_argument("--index-rate", type=float, default=1.0)
    ap.add_argument("--protect", type=float, default=0.0)
    args = ap.parse_args()

    if not args.text and not args.text_file:
        sys.exit("Provide --text or --text-file")
    text = args.text or Path(args.text_file).read_text(encoding="utf-8").strip()

    OUTPUTS.mkdir(exist_ok=True)
    if args.out:
        final_out = Path(args.out).resolve()
    else:
        n = len(list(OUTPUTS.glob("daniel_*.wav"))) + 1
        final_out = OUTPUTS / f"daniel_{n:03d}.wav"
    final_out.parent.mkdir(parents=True, exist_ok=True)

    if args.keep_intermediate:
        afro_wav = final_out.with_name(final_out.stem + "_afro.wav")
    else:
        afro_wav = Path(tempfile.gettempdir()) / f"afro_{final_out.stem}.wav"

    # Stage 1: Afro-TTS (split long input on sentences to avoid XTTS rushing)
    if len(text) > args.split_threshold:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    else:
        sentences = [text]

    print(f"\n[1/2] Afro-TTS -> {afro_wav.name}  [{len(sentences)} sentence(s)]")
    if len(sentences) == 1:
        subprocess.run([
            str(XTTS_VENV_PY), str(AFRO_SCRIPT),
            "--text", sentences[0],
            "--reference", args.reference,
            "--out", str(afro_wav),
            "--gpt-cond-len", str(args.gpt_cond_len),
        ], check=True)
    else:
        import soundfile as sf
        import numpy as np
        tmpdir = Path(tempfile.mkdtemp(prefix="afro_"))
        parts, sr = [], None
        for i, s in enumerate(sentences, 1):
            preview = s[:60] + ("…" if len(s) > 60 else "")
            print(f"   [{i}/{len(sentences)}] {preview}")
            p = tmpdir / f"s{i:03d}.wav"
            subprocess.run([
                str(XTTS_VENV_PY), str(AFRO_SCRIPT),
                "--text", s,
                "--reference", args.reference,
                "--out", str(p),
                "--gpt-cond-len", str(args.gpt_cond_len),
            ], check=True)
            audio, sr = sf.read(str(p))
            parts.append(audio)
        gap = np.zeros(int(args.gap_ms / 1000 * sr), dtype=parts[0].dtype)
        combined = []
        for i, p in enumerate(parts):
            combined.append(p)
            if i < len(parts) - 1:
                combined.append(gap)
        sf.write(str(afro_wav), np.concatenate(combined), sr)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Stage 2: RVC (Daniel's voice timbre)
    print(f"\n[2/2] RVC (Daniel) -> {final_out.name}")
    subprocess.run([
        sys.executable, str(RVC_BASE / "convert.py"),
        "--input", str(afro_wav),
        "--output", str(final_out),
        "--pitch", str(args.pitch),
        "--index-rate", str(args.index_rate),
        "--protect", str(args.protect),
    ], check=True)

    if not args.keep_intermediate and afro_wav.exists():
        afro_wav.unlink()
    print(f"\nFinal: {final_out}")


if __name__ == "__main__":
    main()
