# ruri_sentencepiece_lite

[English](README_en.md) | [日本語](README.md)

An ultra-lightweight, zero-torch Japanese embedding inference environment powered strictly by **`sentencepiece_lite` + `onnxruntime` + `numpy`**.

- **Target Model**: `cl-nagoya/ruri-v3` series (ModernBERT-based)
- **Hugging Face Model Hub**:
  - **30m (256-dim)**: [`Chottokun/ruri-v3-30m-lite`](https://huggingface.co/Chottokun/ruri-v3-30m-lite)
  - **70m (384-dim)**: [`Chottokun/ruri-v3-70m-lite`](https://huggingface.co/Chottokun/ruri-v3-70m-lite)
  - **130m (512-dim)**: [`Chottokun/ruri-v3-130m-lite`](https://huggingface.co/Chottokun/ruri-v3-130m-lite)
  - **310m (768-dim)**: [`Chottokun/ruri-v3-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-310m-lite)

---

## 💡 Key Features & Solved Challenges

| Traditional PyTorch / Transformers Pipeline | Our Solution (Zero-Torch & SBP) |
|---|---|
| **Bloated package size** (Several gigabytes for PyTorch alone) | **Ultra-compact footprint**: Runtime only requires `sentencepiece_lite` + `onnxruntime` + `numpy`. Significantly reduces disk usage and container cold-start times. |
| **SentencePiece compilation hurdles** (Requires CMake, gcc, Python headers) | **Zero-compilation setup**: Pre-compiled Python wheels (`.whl`) provided for one-line installation. |
| **CPU preprocessing bottleneck** (Slow tokenization leaves GPU idle) | **Lightning-fast multi-threaded tokenization**: Uses Google's SentencePiece Lite with **Safe Boundary Pre-tokenization (SBP)**. Over **13.4x faster** than Hugging Face Fast Tokenizer. |
| **FP16 slowdown on older GPUs** (e.g. GTX 1080 / Pascal architecture) | **Hardware-adaptive auto-switching**: Dynamically queries GPU Compute Capability via `ctypes`. Automatically selects FP16 for Turing/Ampere+ and FP32 for Pascal/CPU. |
| **Divergent outputs from custom inference** | **Mathematical equivalence**: Verified cosine similarity of **1.0000** against official PyTorch outputs (Mean Pooling + L2 normalization). |

---

## 📁 Repository Structure

```text
ruri_sentencepiece_lite/
├── AGENTS.md                  # Strict design constraints & security hygiene (uv, secret protection)
├── .gitignore                 # Exclusion rules for large binaries, weights, and secrets
├── .env.example               # Template for environment variables (HF_TOKEN, HF_USERNAME)
├── docs/
│   └── benchmark.md           # Comprehensive benchmark report (Tokenizer & Model inference)
├── dist_assets/               # Deployment assets (uploaded to Hugging Face Model Hub)
│   ├── README.md              # Model card documentation for Hugging Face
│   ├── ruri_v3_lite.py        # Self-contained, PyTorch-free inference wrapper module
│   ├── ruri_v3_30m.spm.fb     # FlatBuffers-format tokenizer model (4.57MB)
│   ├── tokenizer.model        # Original SentencePiece model
│   ├── model.onnx             # FP32 ONNX model (for CPU / Pascal GPUs)
│   ├── model.onnx.data        # External tensor data binary
│   ├── model_fp16.onnx        # FP16 ONNX model (for Turing/Ampere+ Tensor Core GPUs)
│   └── wheels/                # Pre-built wheels for sentencepiece_lite
│       └── sentencepiece_lite-0.1.0-cp311-cp311-linux_x86_64.whl
└── scripts/
    ├── verify_tokenizer.py    # Phase 0: Verify BOS/EOS/PAD tokens and official output alignment
    ├── build_and_export.py    # Phase 1: Build wheels, convert FlatBuffers, export ONNX models
    ├── deploy_to_hf.py        # Phase 3: Secure upload script to Hugging Face Hub
    ├── test_inference.py      # Phase 5: Verification of mathematical equivalence with PyTorch
    └── benchmark.py           # Throughput & latency benchmark script
```

---

## ⚡ Benchmark Highlights (SentencePiece Lite vs Hugging Face)

Detailed Report: [docs/benchmark.md](docs/benchmark.md)

| Metric | Hugging Face Fast Tokenizer | SentencePiece Lite (Ours) | Speedup / Difference |
|---|---|---|---|
| **10,000 Sentences Time** | 455.03 ms | **33.97 ms** | **~13.4x Faster** ⚡ |
| **Throughput** | 21,976 sent/s | **294,395 sent/s** | **~300k sentences/sec** |
| **Per-sentence Latency** | 45.5 µs | **3.40 µs** | **Near-zero overhead** |

---

## 🚀 Getting Started (End-User Environment)

End users **do not need PyTorch, Transformers, or C++ compilers**.

### 1. Installation

```bash
# Install pre-built wheel and runtime dependencies
pip install https://huggingface.co/Chottokun/ruri-v3-30m-lite/resolve/main/wheels/sentencepiece_lite-0.1.0-cp311-cp311-linux_x86_64.whl \
            onnxruntime numpy huggingface_hub

# For GPU acceleration (CUDA)
pip install onnxruntime-gpu
```

### 2. Quickstart Inference

```python
from ruri_v3_lite import RuriV3Lite
import numpy as np

# Automatically downloads model assets and chooses optimal precision (FP16/FP32)
model = RuriV3Lite(repo_id="Chottokun/ruri-v3-30m-lite")

# Queries & documents formatted with ruri-v3 prefixes
queries = ["検索クエリ: 日本の首都はどこですか？"]
documents = [
    "文章: 日本の首都は東京都です。",
    "文章: 大阪は関西地方の主要都市です。"
]

# Generate normalized embeddings (shape: (N, 256))
q_emb = model.encode(queries)
d_emb = model.encode(documents)

# Compute cosine similarity via dot product
similarities = np.dot(q_emb, d_emb.T)[0]

print(f"Query vs Tokyo: {similarities[0]:.4f}")
print(f"Query vs Osaka: {similarities[1]:.4f}")
```

---

## 🛠️ Development & Reproduction Workflow

All developer commands are executed with [`uv`](https://github.com/astral-sh/uv).

### 1. Set Up Environment
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### 2. Phase 0: Verify Tokenizer Specs
```bash
uv run python scripts/verify_tokenizer.py
```
> Confirms special token IDs `<s>` (1), `</s>` (2), `<pad>` (3) and 100% token ID alignment.

### 3. Phase 1: Build & Export Models
```bash
uv run python scripts/build_and_export.py
```
> Compiles SentencePiece Lite, converts dictionaries to FlatBuffers (`.spm.fb`), exports FP32 ONNX, and quantizes to FP16.

### 4. Phase 5: Verify Mathematical Equivalence
```bash
uv run python scripts/test_inference.py
```
> Validates that cosine similarity against PyTorch official model is $\ge 0.9999$ (measured: `1.00000000`).

### 5. Phase 3: Deploy to Hugging Face
Copy `.env.example` to create your local `.env`:
```bash
cp .env.example .env
# Edit .env and set your HF_TOKEN and optional HF_USERNAME
```

Run deployment:
```bash
# Deploy a specific model size
uv run python scripts/deploy_to_hf.py --model 70m

# Or provide HF_TOKEN via shell environment variable
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
uv run python scripts/deploy_to_hf.py --model 130m
```

---

## 🔒 Security & Secret Hygiene
- Hugging Face write tokens (`HF_TOKEN`) and credentials are never hardcoded and must be provided via environment variables or `.env`.
- Large model binaries (`*.onnx`, `*.onnx.data`, `*.spm.fb`, `*.whl`) and local caches are strictly ignored by `.gitignore`.
