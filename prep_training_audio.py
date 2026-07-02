"""
Concatenate all Daniel voice clips (m4a + wav) into one long WAV for RVC training.

RVC trains best on continuous speech. We:
  1. Decode every m4a/wav via PyAV (handles fp32 amplitude correctly)
  2. Convert to 40kHz mono (RVC's standard training rate)
  3. Concatenate with 200ms silence between clips
  4. Save as daniel_rvc_training.wav

Output:
  C:\\Users\\DELL\\rvc-pipeline\\daniel_rvc_training.wav
"""
import os
from pathlib import Path
import av
import numpy as np
import torch
import torchaudio

DATASET = Path(r"C:\Users\DELL\yarngpt\finetune\dataset")
OUT = Path(r"C:\Users\DELL\rvc-pipeline\daniel_rvc_training.wav")
TARGET_SR = 40000  # RVC standard
GAP_SEC = 0.2

def decode_audio(path: Path) -> tuple[np.ndarray, int]:
    container = av.open(str(path))
    stream = container.streams.audio[0]
    sr = stream.sample_rate
    frames = [f.to_ndarray() for f in container.decode(audio=0)]
    container.close()
    audio = np.concatenate(frames, axis=-1)  # (channels, samples) float32 or int
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    return audio, sr

def main():
    # Collect all source files, sorted numerically
    files = []
    for ext in ("*.wav", "*.m4a"):
        for f in DATASET.glob(ext):
            try:
                num = int(f.stem)
                files.append((num, f))
            except ValueError:
                continue
    files.sort()
    print(f"Found {len(files)} clips")

    gap = torch.zeros(1, int(GAP_SEC * TARGET_SR))
    chunks = []
    total_sec = 0.0
    for num, f in files:
        try:
            audio, sr = decode_audio(f)
            t = torch.from_numpy(audio)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            if t.shape[0] > 1:
                t = t.mean(dim=0, keepdim=True)  # mono
            if sr != TARGET_SR:
                t = torchaudio.functional.resample(t, sr, TARGET_SR)
            peak = t.abs().max().item()
            dur = t.shape[-1] / TARGET_SR
            total_sec += dur + GAP_SEC
            if peak < 0.01:
                print(f"  [skip] {f.name}: silent (peak={peak:.4f})")
                continue
            # normalize per-clip to ~0.7 peak so loud and soft clips match
            t = t / peak * 0.7
            chunks.append(t)
            chunks.append(gap)
            print(f"  {f.name}: {dur:.1f}s  peak={peak:.3f}")
        except Exception as e:
            print(f"  [error] {f.name}: {e}")

    if not chunks:
        raise SystemExit("No usable clips.")

    full = torch.cat(chunks, dim=-1)
    OUT.parent.mkdir(exist_ok=True)
    torchaudio.save(str(OUT), full, TARGET_SR)
    final_dur = full.shape[-1] / TARGET_SR
    print(f"\n✅ Saved {OUT}")
    print(f"   Duration: {final_dur/60:.1f} minutes ({final_dur:.1f}s)")
    print(f"   Sample rate: {TARGET_SR} Hz mono")

if __name__ == "__main__":
    main()
