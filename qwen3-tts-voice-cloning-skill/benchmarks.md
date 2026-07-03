# Qwen3-TTS Benchmarks — RTX 4060 (8GB VRAM)

All tests run on Windows 11, Python 3.14, RTX 4060 8GB, PyTorch 2.12.1, no FlashAttention.

## Quality Comparison

| Mode | Audio Sample | Model | Similarity | Notes |
|------|-------------|-------|------------|-------|
| x_vector_only | 1 min | 1.7B | **~20%** | Sounds vaguely like the speaker; tone/prosody lost |
| Full prompt | 10s | 1.7B | **70-80%** | Captures tone, rhythm, pronunciation habits |
| Full prompt | 1 min | 1.7B | N/A | ❌ OOM — generation hangs, GPU 100% indefinitely |
| Full prompt | 10s | 0.6B | N/A | ❌ Still too slow on 4060; generation hangs |

## Performance Comparison

| Mode | Audio | Prompt Tokens | Time (3 sentences) | GPU Util | VRAM Used |
|------|-------|--------------|-------------------|----------|-----------|
| x_vector_only | 1 min | 1 (vector) | ~20s | 60-100% | 6.7 GB |
| Full prompt | 10s | ~120 | ~60-90s | 19-60% | 6.8 GB |
| Full prompt | 1 min | ~720 | ∞ (OOM) | 60-100% stuck | 7.9 GB |

## Download Speeds (ModelScope from China)

| Model | Size | Speed | Total Time |
|-------|------|-------|------------|
| 1.7B Base | 3.86 GB | 1-2.5 MB/s | ~25 min |
| 0.6B Base | 1.83 GB | 1-2.5 MB/s | ~18 min |

## Key Takeaways

1. **x_vector_only is poor quality** — only use as last resort when hardware can't handle full prompt
2. **Audio length is the bottleneck**, not model size — 10s is the sweet spot for 8GB GPUs
3. **1.7B > 0.6B** for full prompt mode on 8GB — if 10s works on 1.7B, don't downgrade
4. **Always trim audio to ≤15s** for 8GB GPUs before attempting full prompt mode
