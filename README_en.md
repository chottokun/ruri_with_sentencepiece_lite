# ruri_sentencepiece_lite

[English](README_en.md) | [日本語](README.md)

A lightweight Japanese text embedding and cross-encoder reranker inference environment powered strictly by **`sentencepiece_lite` + `onnxruntime` + `numpy`**—completely free of `torch` and `transformers` runtime dependencies.

- **Target Models**: `cl-nagoya/ruri-v3` series (ModernBERT-Ja architecture)
- **Hugging Face Model Hub**:
  - **30m Embedding (256-dim)**: [`Chottokun/ruri-v3-30m-lite`](https://huggingface.co/Chottokun/ruri-v3-30m-lite)
  - **70m Embedding (384-dim)**: [`Chottokun/ruri-v3-70m-lite`](https://huggingface.co/Chottokun/ruri-v3-70m-lite)
  - **130m Embedding (512-dim)**: [`Chottokun/ruri-v3-130m-lite`](https://huggingface.co/Chottokun/ruri-v3-130m-lite)
  - **310m Embedding (768-dim)**: [`Chottokun/ruri-v3-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-310m-lite)
  - **310m Reranker (Cross-Encoder)**: [`Chottokun/ruri-v3-reranker-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-reranker-310m-lite)

**Quick Links**: [Key Features](#key-features--technical-approach) | [Benchmark Highlights](#benchmark-highlights) | [Embedding Usage](#embedding-usage-encode) | [Reranker Usage](#reranker-usage-rerank) | [Development Workflow](#development-and-build-workflow) | [License & Citations](#license--redistribution-terms-apache-license-20)

---

## Key Features & Technical Approach

| Traditional PyTorch / Transformers Stack | Zero-Torch Architecture (This Repo) | Technical Rationale & Characteristics |
|---|---|---|
| **Large package footprint** (Several GBs for PyTorch alone) | **Minimal runtime footprint** (~70–100 MB installed) | Runtime dependencies restricted strictly to `sentencepiece_lite`, `onnxruntime`, `numpy`, and `huggingface_hub`. Lowers container sizes and cold-start latencies. |
| **SentencePiece build dependencies** (Requires CMake, gcc, Python dev headers) | **Pre-compiled Wheel distribution** | Pre-built Wheels (`.whl`) provided for one-line installation without requiring a C++ compiler toolchain. |
| **CPU tokenization bottlenecks** (GPU idle time during preprocessing) | **C++20 Zero-copy tokenization** | Leverages FlatBuffers (`mmap`) and Safe Boundary Pre-tokenization (SBP), completing tokenization in ~3.4 µs per sentence. |
| **FP16 performance degradation on legacy GPUs** (Pascal architecture) | **Hardware-adaptive automatic dispatch** | Dynamically checks GPU Compute Capability via `ctypes`. Automatically selects FP16 for CC $\ge 7.0$ (Turing/Ampere+) and FP32 for CC $< 7.0$ (Pascal) or CPU. |
| **Numerical divergence in custom wrappers** | **Mathematical equivalence verified** | Cosine similarity of 1.0000 against official PyTorch outputs (Mean Pooling + L2 normalization) validated on reference datasets. |

---

## Repository Structure

```text
ruri_sentencepiece_lite/
├── AGENTS.md                  # Strict engineering constraints & security policies (uv, secret protection)
├── .gitignore                 # Exclusion rules for large binaries, model weights, and cache dirs
├── .env.example               # Template for environment variables (HF_TOKEN, etc.)
├── docs/
│   ├── benchmark.md           # Detailed benchmark report for embedding models
│   └── reranker_benchmark.md  # Detailed benchmark & architectural trade-off report for reranker
├── dist_assets/               # Deployment assets for Hugging Face Model Hub
│   ├── README.md              # Model card for Hugging Face
│   ├── ruri_v3_lite.py        # Embedding wrapper module (~190 lines, zero torch)
│   ├── ruri_v3_reranker_lite.py # Reranker wrapper module (~310 lines, zero torch)
│   ├── ruri_v3_30m.spm.fb     # FlatBuffers-format tokenizer dictionary (4.57MB)
│   ├── tokenizer.model        # Original SentencePiece model
│   ├── model.onnx             # FP32 ONNX model
│   ├── model_fp16.onnx        # FP16 ONNX model
│   └── wheels/                # Pre-built wheels for sentencepiece_lite
└── scripts/
    ├── verify_tokenizer.py    # Verify special tokens and tokenizer compatibility
    ├── build_and_export.py    # Build wheels, convert FlatBuffers, and export ONNX models
    ├── deploy_to_hf.py        # Secure deployment script for embedding models
    ├── deploy_reranker_to_hf.py # Secure deployment script for reranker model
    ├── test_inference.py      # Verification of mathematical equivalence with PyTorch
    ├── test_reranker_robustness.py # Automated edge-case & robustness test suite for reranker
    └── benchmark.py           # Latency and throughput benchmark script
```

---

## Benchmark Highlights

For full experimental setups, profile breakdowns, and trade-off analyses, please see the individual benchmark reports:
- Embedding Models: [docs/benchmark.md](docs/benchmark.md)
- Cross-Encoder Reranker: [docs/reranker_benchmark.md](docs/reranker_benchmark.md)

### 1. Tokenizer Preprocessing Benchmark (10,000 Sentences, CPU)
| Tokenizer Implementation | 10,000 Sentences Time | Throughput | Per-sentence Latency | Speedup |
|---|---|---|---|---|
| **Hugging Face Fast Tokenizer** (Rust-based) | 455.03 ms | 21,976 sent/s | 45.50 µs | 1.00x (Baseline) |
| **SentencePiece Lite (Ours)** (C++20 Zero-copy) | **33.97 ms** | **294,395 sent/s** | **3.40 µs** | **~13.4x Faster** |

### 2. End-to-End Inference Latency across Batch Sizes (30m / CPU)
| Batch Size | PyTorch 2.14 CPU | RuriV3Lite (Ours) CPU | Latency Ratio | Assessment |
|---|---|---|---|---|
| **Batch = 1 (Single Query)** | 5.90 ms | **2.98 ms** | **1.98x Faster** | Ours advantageous (minimal Python overhead) |
| **Batch = 8** | 15.22 ms | **11.39 ms** | **1.34x Faster** | Ours advantageous |
| **Batch = 16** | 27.35 ms | **26.79 ms** | 1.02x (Tied) | Performance Breakeven point |
| **Batch = 64** | 93.97 ms | 124.08 ms | 0.76x (PyTorch faster) | Matrix multiplication (GEMM) dominant |

> **Architectural Selection Guidelines**:
> - **WebAPI, Search, & Serverless Services (Batch 1–8)**: Our implementation is recommended due to minimal Python dispatch overhead, ~90% smaller installation size, and faster container startup.
> - **Offline Bulk Batch Processing (Batch 32–64)**: PyTorch (LibTorch) with Intel MKL optimized matrix kernels exhibits ~1.25–1.3x higher CPU throughput.

### 3. Package Footprint & Memory Consumption
| Evaluation Metric | Standard PyTorch + Transformers | Zero-Torch Stack (Ours) | Reduction |
|---|---|---|---|
| **Python Runtime Footprint (CPU)** | ~1,240 MB (~1.2 GB) | **~100 MB** | **~92% Reduction** |
| **Peak Resident Memory (30m RAM RSS)** | 980.6 MB | **324.1 MB** | **~67% Reduction** |
| **Tokenizer Wheel Package** | ~10 MB (`sentencepiece`) | **1.7 MB** (`sentencepiece_lite`) | **~83% Reduction** |
| **Dictionary Loading Overhead** | In-memory parse & allocation | **4.57 MB (mmap zero-copy)** | **Zero allocation overhead** |

---

## Embedding Usage (Encode)

### 1. Installation

```bash
# Install pre-built wheel and runtime dependencies (CPU)
pip install https://huggingface.co/Chottokun/ruri-v3-30m-lite/resolve/main/wheels/sentencepiece_lite-0.1.0-cp311-cp311-linux_x86_64.whl \
            onnxruntime numpy huggingface_hub

# For GPU acceleration (CUDA)
pip install onnxruntime-gpu
```

### 2. Inference Code

```python
from ruri_v3_lite import RuriV3Lite
import numpy as np

# Automatically loads FP16 on modern GPUs (CC >= 7.0) and FP32 on CPU / Pascal GPUs
model = RuriV3Lite(repo_id="Chottokun/ruri-v3-30m-lite")

# Formatted with official ruri-v3 task prefixes
queries = ["検索クエリ: 日本の首都はどこですか？"]
documents = [
    "文章: 日本の首都は東京都です。",
    "文章: 大阪は関西地方の主要都市です。"
]

# Generate normalized embeddings (shape: (N, 256))
q_emb = model.encode(queries)
d_emb = model.encode(documents)

# Cosine similarity via dot product of L2-normalized vectors
similarities = np.dot(q_emb, d_emb.T)[0]

print(f"Query vs Tokyo: {similarities[0]:.4f}")
print(f"Query vs Osaka: {similarities[1]:.4f}")
```

---

## Reranker Usage (Rerank)

Cross-Encoder model [`Chottokun/ruri-v3-reranker-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-reranker-310m-lite) for high-precision second-stage ranking in RAG and semantic search systems.

### Precision & Model Tiers

| `precision` Parameter | Disk Size | Download (Gzip) | CPU Latency (Top-10) | Rank Preservation | Target Use Case |
|---|---|---|---|---|---|
| `"auto"` (Default) | - | - | - | - | Selects FP16 on modern GPUs, INT8-Full on CPU |
| `"int8_full"` | 301 MB | 202 MB | 2,116 ms | 100% Identical | **Recommended CPU Standard**: Minimal download, exact ranking |
| `"pruned_16l"` | 220 MB | 178 MB | 1,382 ms | Top-1 Preserved | **Ultra-Low-Latency**: 16-layer pruned model (1.87x faster) |
| `"int8"` | 526 MB | - | 2,072 ms | Minor diff | Dynamic quantization of linear layers only |
| `"fp16"` | 601 MB | - | (GPU only) | 100% Identical | Optimized for Tensor Core GPUs |
| `"fp32"` | 1,202 MB | - | 2,589 ms | Baseline | Full precision FP32 model |

### Quickstart

```python
from ruri_v3_reranker_lite import RuriV3RerankerLite

# Load model (transparently streams and decompresses Gzip if available)
reranker = RuriV3RerankerLite(
    repo_id="Chottokun/ruri-v3-reranker-310m-lite",
    precision="int8_full",  # Or "pruned_16l" for lowest latency
    device="cpu"
)

query = "日本の首都はどこですか？"
documents = [
    "日本の首都は東京都です。政治・経済の中枢が集約されています。",
    "東京は日本の政治と文化の中心都市であり、多くの観光客が訪れます。",
    "大阪は関西地方の主要都市で、独自の食文化やお笑いで知られています。",
    "明日の天気は全国的に晴れのち曇りとなる見込みです。",
    "ピタゴラスの定理は直角三角形の斜辺の長さを計算するための幾何学の基本法則です。"
]

# Get results sorted in descending score order
results = reranker.rerank(query, documents, top_k=3, normalize=True)

for rank, item in enumerate(results, start=1):
    print(f"Rank {rank}: Score={item['score']:.4f} (Index {item['index']}) -> {item['document']}")
```

> **Recommended Two-Stage Retrieval Pipeline**:
> Since Cross-Encoders exhibit $O(N)$ computational complexity, it is best practice to first retrieve **Top-20 to Top-30** candidate documents via `ruri-v3-30m-lite` (Bi-Encoder embeddings in $<10$ ms), and then rerank them using `ruri-v3-reranker-310m-lite` before passing them to downstream LLMs.

---

## Development and Build Workflow

All tasks within this repository are managed using `uv`.

### 1. Environment Setup
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### 2. Tokenizer Specification Verification
```bash
uv run python scripts/verify_tokenizer.py
```

### 3. Asset Build & Model Export
```bash
uv run python scripts/build_and_export.py
```

### 4. Mathematical Equivalence Verification
```bash
uv run python scripts/test_inference.py
```

### 5. Reranker Robustness & Edge-Case Testing
```bash
uv run python scripts/test_reranker_robustness.py
```

### 6. Deployment to Hugging Face
```bash
cp .env.example .env
# Edit .env with HF_TOKEN and HF_USERNAME

# Deploy embedding models
uv run python scripts/deploy_to_hf.py --model 30m

# Deploy reranker model
uv run python scripts/deploy_reranker_to_hf.py
```

---

## Security and Secret Hygiene
- All Hugging Face credentials (`HF_TOKEN`) are handled strictly via environment variables. No secrets are committed.
- Large binaries (`*.onnx`, `*.onnx.data`, `*.spm.fb`, `*.whl`) and local caches are ignored via `.gitignore`.

---

## License & Redistribution Terms (Apache License, Version 2.0)

This repository and its distributed assets are licensed under the **Apache License, Version 2.0**. Full terms can be found in [LICENSE](LICENSE).

### 1. Base Model
- **Model**: [`cl-nagoya/ruri-v3`](https://huggingface.co/collections/cl-nagoya/ruri-v3-67c006886e0621255e7fcb99) (`30m`, `70m`, `130m`, `310m`, `reranker-310m`)
- **Developers**: Nagoya University NLP Laboratory (cl-nagoya)
- **Authors**: Hayato Tsukagoshi, Ryohei Sasano
- **License**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Modifications**:
  - Exported weights to ONNX (FP32, FP16, INT8, Pruned) formats.
  - Converted SentencePiece vocabulary to FlatBuffers serialization (`.spm.fb`).
  - Implemented standalone zero-torch inference wrappers (`ruri_v3_lite.py`, `ruri_v3_reranker_lite.py`).

### 2. Tokenizer Core (SentencePiece Lite)
- **Library**: [Google SentencePiece Lite](https://google.github.io/sentencepiece/lite/)
- **Repository**: [github.com/google/sentencepiece](https://github.com/google/sentencepiece)
- **Copyright**: Copyright 2018 Google LLC
- **License**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

### 3. Citations

#### Ruri (Embedding & Reranker Models)
```bibtex
@misc{Ruri,
  title={{Ruri: Japanese General Text Embeddings}}, 
  author={Hayato Tsukagoshi and Ryohei Sasano},
  year={2024},
  eprint={2409.07737},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2409.07737}, 
}
```

#### SentencePiece (Tokenizer)
```bibtex
@inproceedings{kudo-richardson-2018-sentencepiece,
  title = "{S}entence{P}iece: A simple and language independent subword tokenizer and detokenizer for {N}eural {T}ext {P}rocessing",
  author = "Kudo, Taku and Richardson, John",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
  month = nov,
  year = "2018",
  address = "Brussels, Belgium",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/D18-2012",
  doi = "10.18653/v1/D18-2012",
  pages = "66--71",
}
```
