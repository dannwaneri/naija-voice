# Hardcoded `min_new_tokens=2` overrides user-supplied value, causing premature EOS on ~half of seeds

## Environment

- `qwen-tts==0.1.1` (pip install qwen-tts)
- Python 3.11 / CPU inference (also reproduces on GPU per community reports)
- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`

## Summary

The [README explicitly documents](https://github.com/QwenLM/Qwen3-TTS#python-package-usage) that users can pass Hugging Face Transformers `generate()` kwargs to `generate_voice_clone(...)`:

> "besides the parameters shown and explicitly documented, you can also pass generation kwargs supported by Hugging Face Transformers `model.generate`, e.g., `max_new_tokens`, `top_p`, etc."

`min_new_tokens` is one of those kwargs. It's accepted at the top of the call chain (via `**kwargs`) and forwarded through `_merge_generate_kwargs`. However, the low-level `SpeechTalkerModel.generate()` at `qwen_tts/core/models/modeling_qwen3_tts.py:2046` **hardcodes** `"min_new_tokens": 2` when it builds `talker_kwargs`, silently discarding whatever the user passed.

As a result, the model can emit its EOS code within the first few tokens, producing truncated audio on many otherwise valid seeds. Behaviorally: `generate_voice_clone` returns cleanly (no error), the audio file is written, but only ~30-40% of the requested sentence is spoken — the remainder is cut off mid-phrase.

## Reproducer

`repro.py`:

```python
import torch, random, numpy as np
from qwen_tts import Qwen3TTSModel

REF_AUDIO = "your_ref.wav"   # any 30–60s clean single-speaker clip
REF_TEXT  = "transcript of the reference clip"
TEXT      = "Hello, my name is Daniel. Let us get into it."

model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

for seed in range(1, 6):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    wavs, sr = model.generate_voice_clone(
        text=TEXT,
        language="English",
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        temperature=0.7,
        min_new_tokens=500,   # explicitly requesting >= 500 tokens
    )
    wav = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
    if hasattr(wav, "detach"):
        wav = wav.detach().cpu().numpy()
    print(f"seed={seed}: {len(wav)/sr:.2f}s of audio ({len(wav)} samples @ {sr}Hz)")
```

Observed durations on my hardware (same 45-char input, 5 seeds, `min_new_tokens=500`):

```
seed=1: 4.1s
seed=2: 3.8s
seed=3: 3.0s   <- truncated: only "Hello, my name is Daniel." (missing "Let us get into it.")
seed=4: 3.7s
seed=5: 4.4s
```

The 45-char sentence should be ~4.5–5.5 seconds. Seeds 2, 3, 4 are all short of that. Seed 3 cuts off mid-sentence audibly.

With `min_new_tokens=500` respected, the model would be forced to keep generating past its early EOS candidate. Instead, the hardcoded `min_new_tokens=2` at line 2046 wins, and the model can EOS at will.

## Root cause

`qwen_tts/core/models/modeling_qwen3_tts.py`, around line 2044 (in the `SpeechTalkerModel.generate()` method):

```python
talker_kwargs = {
    "max_new_tokens": max_new_tokens,
    "min_new_tokens": 2,                      # <-- hardcoded, overrides user kwarg
    "do_sample": do_sample,
    ...
```

The surrounding signature accepts `**kwargs`, so the user's `min_new_tokens` reaches this function but is discarded when the dict is built.

## Proposed fix

One line — read from kwargs, fall back to the current default:

```python
talker_kwargs = {
    "max_new_tokens": max_new_tokens,
    "min_new_tokens": kwargs.pop("min_new_tokens", 2),
    "do_sample": do_sample,
    ...
```

This preserves current behavior for callers that don't pass `min_new_tokens`, and honors it for callers that do. PR incoming (or happy to open one if the maintainers prefer).

## Why it matters

Voice cloning workloads that need reliable full-sentence completion on short-to-medium inputs are impacted on every seed. Downstream users currently work around this by seed-sweeping until they find one that happens to not EOS early — expensive on GPU and effectively unreliable.

I hit this while building an open-source Nigerian-accented voice cloning pipeline ([dannwaneri/naija-voice](https://github.com/dannwaneri/naija-voice)). Happy to provide the audio artifacts if useful.
