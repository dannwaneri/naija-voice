"""
Convert any input WAV to Daniel's voice using the trained RVC model.

Runs Applio's inference CLI locally (CPU). Expects the venv at
.venv/ and the Applio repo cloned at Applio/.

Usage:
    python convert.py --input some_audio.wav
    python convert.py --input some_audio.wav --output daniel_version.wav
    python convert.py --input some_audio.wav --pitch 2 --index-rate 0.5
"""
import argparse
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).parent
VENV_PY = BASE / ".venv" / "Scripts" / "python.exe"
APPLIO = BASE / "Applio"
MODEL = BASE / "models" / "daniel" / "daniel_50e_2950s.pth"
INDEX = BASE / "models" / "daniel" / "daniel.index"
OUTPUTS = BASE / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input WAV to convert")
    ap.add_argument("--output", help="Output WAV path (default: outputs/<input>_daniel.wav)")
    ap.add_argument("--f0-method", default="rmvpe",
                    choices=["rmvpe", "crepe", "crepe-tiny", "fcpe"])
    ap.add_argument("--pitch", type=int, default=0,
                    help="Pitch shift in semitones (0 = no change)")
    ap.add_argument("--index-rate", type=float, default=0.75,
                    help="How much to use the speaker index (0=ignore, 1=full)")
    ap.add_argument("--protect", type=float, default=0.33,
                    help="Protect voiceless consonants (0=off, 0.5=max)")
    ap.add_argument("--volume-envelope", type=float, default=1.0)
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        sys.exit(f"Input not found: {in_path}")

    OUTPUTS.mkdir(exist_ok=True)
    out_path = Path(args.output).resolve() if args.output else OUTPUTS / f"{in_path.stem}_daniel.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(VENV_PY), "core.py", "infer",
        "--input_path", str(in_path),
        "--output_path", str(out_path),
        "--pth_path", str(MODEL),
        "--index_path", str(INDEX),
        "--f0_method", args.f0_method,
        "--pitch", str(args.pitch),
        "--index_rate", str(args.index_rate),
        "--volume_envelope", str(args.volume_envelope),
        "--protect", str(args.protect),
        "--embedder_model", "contentvec",
    ]
    print(f"Converting {in_path.name} -> {out_path.name}...")
    subprocess.run(cmd, cwd=str(APPLIO), check=True)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
