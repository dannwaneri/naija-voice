"""
End-to-end pipeline: text -> YarnGPT (Nigerian accent) -> RVC (Daniel's voice) -> WAV

Long inputs are auto-split on sentence boundaries (. ! ?). Each sentence is
generated fresh by YarnGPT to avoid the "rushing" that happens on long single
generations, then all pieces are concatenated with a small silence gap before
RVC runs on the combined audio.

Usage:
    python speak.py --text "Hello, this is Daniel."
    python speak.py --text-file script.txt --out outputs/episode1.wav
    python speak.py --text "..." --speaker tayo --keep-intermediate
    python speak.py --text "single sentence input" --no-split
"""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

RVC_BASE = Path(__file__).parent
YARN_BASE = Path(r"C:\Users\DELL\yarngpt")
YARN_PY = YARN_BASE / "venv" / "Scripts" / "python.exe"
YARN_SCRIPT = YARN_BASE / "generate_yarn.py"
OUTPUTS = RVC_BASE / "outputs"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def run_yarn(sentence: str, out_path: Path, speaker: str, temperature: float):
    subprocess.run([
        str(YARN_PY), str(YARN_SCRIPT),
        "--text", sentence,
        "--out", str(out_path),
        "--speaker", speaker,
        "--temperature", str(temperature),
    ], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", help="Text to speak")
    ap.add_argument("--text-file", help="Path to a text file to speak")
    ap.add_argument("--out", help="Final output WAV path")
    ap.add_argument("--speaker", default="jude",
                    help="YarnGPT source voice. Male: jude, tayo, umar, osagie, onye, emma. Female: idera, zainab, regina, chinenye, joke, remi.")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--keep-intermediate", action="store_true")
    ap.add_argument("--no-split", action="store_true",
                    help="Generate whole input in one pass (faster, but may rush long input)")
    ap.add_argument("--split-threshold", type=int, default=120,
                    help="Min char length before sentence-splitting kicks in (short input stays one pass for quality)")
    ap.add_argument("--gap-ms", type=int, default=250,
                    help="Silence between sentences (ms)")
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--index-rate", type=float, default=0.75)
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument("--takes", type=int, default=1,
                    help="Generate N takes of the same text (saved as *_take1.wav, *_take2.wav, ...)")
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
        yarn_wav = final_out.with_name(final_out.stem + "_yarn.wav")
    else:
        yarn_wav = Path(tempfile.gettempdir()) / f"yarn_{final_out.stem}.wav"

    if args.no_split or len(text) < args.split_threshold:
        sentences = [text]
    else:
        sentences = split_sentences(text)

    print(f"\n[1/2] YarnGPT ({args.speaker}) -> {yarn_wav.name}  [{len(sentences)} sentence(s)]")
    if len(sentences) == 1:
        run_yarn(sentences[0], yarn_wav, args.speaker, args.temperature)
    else:
        import soundfile as sf
        import numpy as np
        tmpdir = Path(tempfile.mkdtemp(prefix="yarn_"))
        parts, sr = [], None
        for i, s in enumerate(sentences, 1):
            preview = s[:60] + ("…" if len(s) > 60 else "")
            print(f"   [{i}/{len(sentences)}] {preview}")
            p = tmpdir / f"s{i:03d}.wav"
            run_yarn(s, p, args.speaker, args.temperature)
            audio, sr = sf.read(str(p))
            parts.append(audio)
        gap = np.zeros(int(args.gap_ms / 1000 * sr), dtype=parts[0].dtype)
        combined = []
        for i, p in enumerate(parts):
            combined.append(p)
            if i < len(parts) - 1:
                combined.append(gap)
        sf.write(str(yarn_wav), np.concatenate(combined), sr)
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n[2/2] RVC (Daniel) -> {final_out.name}")
    subprocess.run([
        sys.executable, str(RVC_BASE / "convert.py"),
        "--input", str(yarn_wav),
        "--output", str(final_out),
        "--pitch", str(args.pitch),
        "--index-rate", str(args.index_rate),
        "--protect", str(args.protect),
    ], check=True)

    if not args.keep_intermediate and yarn_wav.exists():
        yarn_wav.unlink()
    print(f"\nFinal: {final_out}")


if __name__ == "__main__":
    main()
