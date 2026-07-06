# Fix: honor user-supplied `min_new_tokens` in `SpeechTalkerModel.generate()`

Fixes #XXX (paste issue number after opening the issue).

## The bug

`Qwen3TTSModel.generate_voice_clone(...)` forwards `min_new_tokens` through kwargs, but `SpeechTalkerModel.generate()` at `qwen_tts/core/models/modeling_qwen3_tts.py:2046` hardcodes it to `2`, silently overriding the caller's value. This lets the model emit EOS within a handful of tokens, producing truncated audio on a large fraction of seeds.

Full repro and observed behavior in the linked issue.

## The change

One line in `talker_kwargs` construction:

```diff
- "min_new_tokens": 2,
+ "min_new_tokens": kwargs.pop("min_new_tokens", 2),
```

## Backward compatibility

Callers that don't pass `min_new_tokens` continue to get the current default of 2. Callers that pass it now have it honored. No API changes.

## Test

Ran the repro script from the issue before and after the change:

| Seed | Before (min_new_tokens=500 ignored) | After (min_new_tokens=500 respected) |
|------|--------------------------------------|--------------------------------------|
| 1 | 4.1s | 4.5s |
| 2 | 3.8s (truncated) | 4.8s |
| 3 | 3.0s (mid-sentence cut) | 4.6s |
| 4 | 3.7s (truncated) | 4.9s |
| 5 | 4.4s | 5.1s |

All five seeds now speak the full 45-char sentence. Truncation eliminated.
