---
name: qwen3-tts-voice-cloning
description: Use when the user wants to clone their voice with Qwen3-TTS, generate speech from text with a cloned voice, or set up TTS voice cloning on a local GPU. Covers installation, hardware assessment, audio sample length recommendations, model selection (1.7B vs 0.6B), and mode selection (full prompt vs x_vector_only).
---

# Qwen3-TTS Voice Cloning

## Overview

Set up Qwen3-TTS locally to clone a user's voice from a short audio sample and generate new speech. Covers the full pipeline: environment setup, model download (via ModelScope for China users), voice cloning with quality/performance tradeoffs, and batch generation.

**Core principle:** Full prompt mode (`x_vector_only_mode=False`) gives much better quality than x_vector_only mode, at the cost of more GPU memory and time. Always aim for full prompt mode first.

## When to Use

```
User wants voice cloning?
├─ Has GPU? ─── No ──> Suggest cloud API (DashScope) or CPU-only (0.6B, very slow)
├─ Yes
├─ How much VRAM?
│   ├─ 4-6 GB ──> 0.6B model + x_vector_only mode only
│   ├─ 8 GB ────> 1.7B model + full prompt with ≤15s audio, or x_vector_only
│   ├─ 12+ GB ──> 1.7B model + full prompt with ≤30s audio
│   └─ 16+ GB ──> 1.7B model + full prompt with ≤60s audio
└─ Audio length > recommended? ──> Trim to best segment first
```

## Installation

### Step 1: Create venv

```bash
python -m venv qwen3-tts
qwen3-tts\Scripts\activate   # Windows
source qwen3-tts/bin/activate # Linux/Mac
```

### Step 2: Install package

```bash
pip install -U qwen-tts
```

Skip FlashAttention on Windows (it won't compile; PyTorch fallback works fine).

### Step 3: Download model

**Do NOT use HuggingFace directly from China** — it will timeout. Use ModelScope instead:

```python
from modelscope import snapshot_download

# For 1.7B model (recommended if GPU ≥ 8GB):
snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_dir="./models/Qwen3-TTS-12Hz-1.7B-Base")

# For 0.6B model (lighter, for 4-6GB GPUs):
snapshot_download("Qwen/Qwen3-TTS-12Hz-0.6B-Base", local_dir="./models/Qwen3-TTS-12Hz-0.6B-Base")
```

Expected download: 1.7B ≈ 3.9GB, 0.6B ≈ 1.8GB. Speed ~1-2.5 MB/s on ModelScope from China.

### Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| HF download timeout | `Read timed out` | Use ModelScope instead |
| HF mirror blocked | `hf-mirror.com` unreachable | Use ModelScope |
| Corrupted HF cache | `pytorch_model.bin not found` / TensorFlow weights error | `rm -rf ~/.cache/huggingface/hub/models--Qwen--*` |
| SoX warning on import | `SoX could not be found!` | Safe to ignore — 12Hz models don't use SoX |
| FlashAttn won't install | Compile hangs on Windows | Skip it; manual PyTorch attention works |

## Hardware Assessment

**MUST ask the user their GPU model/VRAM before making recommendations.** Use `nvidia-smi` if available.

### Recommendation Matrix

| GPU VRAM | Max Audio (full prompt) | Recommended Model | Mode |
|----------|------------------------|-------------------|------|
| 4-6 GB | 5s | 0.6B | Full prompt |
| 8 GB (e.g. RTX 4060) | **10-15s** | 1.7B | Full prompt |
| 12 GB (e.g. RTX 4070 Ti) | 30s | 1.7B | Full prompt |
| 16+ GB (e.g. RTX 4080/4090) | 60s | 1.7B | Full prompt |

**Critical:** Audio length is the bottleneck, not the output text length. 1 minute of audio ≈ 720 prompt tokens — the attention matrix alone consumes ~1.5 GB VRAM. 10 seconds ≈ 120 tokens, which is ~14x cheaper.

## Audio Sample Recommendations

1. **Trim to the cleanest segment** — 10-15 seconds of clear, well-pronounced speech with no background noise
2. **Provide an exact transcript** of the trimmed segment — accuracy directly impacts clone quality
3. **Use WAV format** if possible (MP3 adds compression artifacts)
4. **Trim with Python if ffmpeg unavailable:**
```python
import soundfile as sf
audio, sr = sf.read("recording.mp3")
start_s = int(start_sec * sr)
end_s = int(end_sec * sr)
sf.write("clip.wav", audio[start_s:end_s], sr)
```

## Quality Modes

### Full Prompt Mode (`x_vector_only_mode=False`) — **PREFERRED**

The model encodes the entire reference audio as tokens and conditions generation on them. Captures tone, rhythm, and pronunciation habits.

- Quality: ★★★★☆ (70-80% similarity on 10s sample)
- Speed: 1-2 min per sentence on RTX 4060
- Requires: transcript of reference audio

### x_vector_only Mode (`x_vector_only_mode=True`) — **FALLBACK ONLY**

Only extracts a speaker embedding vector. Much faster but loses prosody and pronunciation character.

- Quality: ★★☆☆☆ (~20% similarity — sounds vaguely like you)
- Speed: ~20s per sentence on RTX 4060
- Use only when: GPU can't handle full prompt mode for the given audio length

**⚠️ Never recommend x_vector_only first.** Always try full prompt mode with appropriately trimmed audio. Only fall back to x_vector_only if the user refuses to trim their audio and their GPU can't handle it.

## Usage Template

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "path/to/model",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

prompt = model.create_voice_clone_prompt(
    ref_audio="clip.wav",
    ref_text="exact transcript of the clip",
    x_vector_only_mode=False,  # Full mode for best quality
)

wavs, sr = model.generate_voice_clone(
    text=["Sentence one.", "Sentence two."],
    language=["English", "English"],
    voice_clone_prompt=prompt,
)

for i, w in enumerate(wavs):
    sf.write(f"output_{i+1}.wav", w, sr)
```

For batch generation with intentionally misspelled words (e.g., English pronunciation practice), simply provide misspelled text — the TTS will read it phonetically, creating a "mispronounced" effect.

## Benchmark Reference

See [benchmarks.md](benchmarks.md) for measured results on RTX 4060 (8GB).

## Common Mistakes

| Mistake | Why wrong | Fix |
|---------|-----------|-----|
| Using 1-minute audio on 8GB GPU | 720 prompt tokens → OOM | Trim to 10-15s |
| Using x_vector_only as default | Quality is poor (~20%) | Always try full prompt first |
| Not providing transcript | Full prompt mode needs it | Transcribe the clip exactly |
| Using HuggingFace from China | Timeout | Use ModelScope |
| Installing FlashAttn on Windows | Won't compile | Skip it |
| Using 0.6B when 1.7B would fit | Lower quality for no reason | Check VRAM, prefer 1.7B |
