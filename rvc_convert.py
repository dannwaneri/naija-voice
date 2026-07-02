"""
Convert any input WAV to Daniel's voice using the trained RVC model.

Usage:
    python rvc_convert.py --input some_audio.wav --output daniel_version.wav
    python rvc_convert.py --input some_audio.wav   # auto-names output
"""
import argparse
from pathlib import Path

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "daniel" / "daniel_50e_2950s.pth"
INDEX_PATH = BASE / "models" / "daniel" / "daniel.index"
OUTPUTS_DIR = BASE / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input WAV to convert")
    ap.add_argument("--output", help="Output WAV path (default: outputs/<input>_daniel.wav)")
    ap.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "crepe", "harvest", "pm"])
    ap.add_argument("--f0-up-key", type=int, default=0, help="Pitch shift in semitones (0 = no change)")
    ap.add_argument("--index-rate", type=float, default=0.75,
                    help="How much to use the index (0=ignore, 1=full). 0.75 = balance")
    ap.add_argument("--protect", type=float, default=0.33,
                    help="Protect voiceless consonants (0=off, 0.5=max). Default 0.33 keeps clarity")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    if args.output:
        out_path = Path(args.output).resolve()
    else:
        out_path = OUTPUTS_DIR / f"{in_path.stem}_daniel.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading RVC model from {MODEL_PATH.name}...")
    from rvc_python.infer import RVCInference
    rvc = RVCInference(device=args.device)
    rvc.load_model(str(MODEL_PATH), index_path=str(INDEX_PATH))

    # Set conversion params
    rvc.set_params(
        f0_method=args.f0_method,
        f0_up_key=args.f0_up_key,
        index_rate=args.index_rate,
        protect=args.protect,
    )

    print(f"Converting {in_path.name} → {out_path.name}...")
    rvc.infer_file(str(in_path), str(out_path))
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    main()
