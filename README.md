# Qwen3-TTS Voice Cloning — 本地声音克隆 + 英语口语练习

基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 的本地声音克隆方案。用一段 10-15 秒的录音克隆你的声音，然后让 AI 用你的声音朗读任意文本

## 效果对比（RTX 4060 8GB）

| 模式 | 音频 | 相似度 | 耗时 |
|------|------|--------|------|
| 声纹模式 | 任意 | ~20% | 20s |
| **完整模式** | **10s** | **70-80%** | 1 min |
| 完整模式 | 1 min | ❌ 显存溢出 | — |

## 安装

```bash
python -m venv qwen3-tts
qwen3-tts\Scripts\activate
pip install -U qwen-tts modelscope
```

## 下载模型

```python
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", local_dir="./models/Qwen3-TTS-12Hz-1.7B-Base")
```

> 国内用户用 ModelScope，HuggingFace 直连会超时。

## 快速开始

```python
import torch, soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "./models/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

prompt = model.create_voice_clone_prompt(
    ref_audio="my_voice_10s.wav",
    ref_text="录音逐字稿...",
    x_vector_only_mode=False,
)

wavs, sr = model.generate_voice_clone(
    text=["Hello, this is my voice!"],
    language=["English"],
    voice_clone_prompt=prompt,
)

sf.write("output.wav", wavs[0], sr)
```

## 坑与解法

| 坑 | 解法 |
|----|------|
| HF 下载超时 | 用 ModelScope |
| 1 分钟音频跑不动 | 截到 10-15 秒 |
| FlashAttn 编译失败 | 跳过，PyTorch 原生也能跑 |
| HF 缓存损坏 | `rm -rf ~/.cache/huggingface` 重来 |
| SoX 警告 | 无视，12Hz 模型不依赖 |

## 硬件建议

| 显存 | 推荐音频 | 推荐模型 |
|------|---------|---------|
| 4-6 GB | 5s | 0.6B |
| **8 GB** | **10-15s** | **1.7B** |
| 12 GB | 30s | 1.7B |
| 16+ GB | 60s | 1.7B |



## 项目结构

```
├── batch_clone.py              # 批量生成脚本
├── models/
│   └── Qwen3-TTS-12Hz-1.7B-Base/  # 模型权重（需下载）
└── skills/qwen3-tts-voice-cloning-skill/  # AI Skill
    ├── SKILL.md                # 安装 + 评估 + 推荐
    └── benchmarks.md           # 实测数据
```

## License

MIT
