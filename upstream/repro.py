"""
Repro for: qwen-tts 0.1.1 hardcoded min_new_tokens=2 overrides user kwarg,
causing premature EOS and mid-sentence truncation on ~half of seeds.

Usage:
    python repro.py --ref-audio path/to/ref.wav --ref-text-file path/to/transcript.txt

Expected:
    Some seeds produce short audio (~3 sec) that cuts off mid-sentence,
    despite min_new_tokens=500 being passed. Full sentence is ~4.5-5.5s.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-audio", required=True)
    ap.add_argument("--ref-text-file", required=True)
    ap.add_argument("--text", default="Hello, my name is Daniel. Let us get into it.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--min-new-tokens", type=int, default=500)
    args = ap.parse_args()

    ref_text = Path(args.ref_text_file).read_text(encoding="utf-8").strip()

    print(f"Loading Qwen/Qwen3-TTS-12Hz-1.7B-Base...")
    model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

    print(f"\nText: {args.text}")
    print(f"Passing min_new_tokens={args.min_new_tokens} to generate_voice_clone(...)\n")

    for seed in args.seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        wavs, sr = model.generate_voice_clone(
            text=args.text,
            language="English",
            ref_audio=args.ref_audio,
            ref_text=ref_text,
            temperature=0.7,
            min_new_tokens=args.min_new_tokens,
        )
        wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        duration = len(wav) / sr
        flag = " <-- likely truncated" if duration < 4.0 else ""
        print(f"seed={seed}: {duration:.2f}s{flag}")


if __name__ == "__main__":
    main()
