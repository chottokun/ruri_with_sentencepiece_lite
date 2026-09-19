# ruri_sentencepiece_lite

[English](README_en.md) | [日本語](README.md)

An ultra-lightweight, zero-torch Japanese embedding inference environment powered strictly by **`sentencepiece_lite` + `onnxruntime` + `numpy`**.

- **Target Models**: `cl-nagoya/ruri-v3` series (ModernBERT-based)
- **Hugging Face Model Hub**:
  - **30m Embedding (256-dim)**: [`Chottokun/ruri-v3-30m-lite`](https://huggingface.co/Chottokun/ruri-v3-30m-lite)
  - **70m Embedding (384-dim)**: [`Chottokun/ruri-v3-70m-lite`](https://huggingface.co/Chottokun/ruri-v3-70m-lite)
  - **130m Embedding (512-dim)**: [`Chottokun/ruri-v3-130m-lite`](https://huggingface.co/Chottokun/ruri-v3-130m-lite)
  - **310m Embedding (768-dim)**: [`Chottokun/ruri-v3-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-310m-lite)
  - **310m Reranker (Cross-Encoder)**: [`Chottokun/ruri-v3-reranker-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-reranker-310m-lite) ⚡ *New!*

**Quick Links**: [Key Features](#-key-features--solved-challenges) | [Benchmark](#-benchmark-highlights) | [Getting Started](#-getting-started-end-user-environment) | [Reranker Usage](#-using-the-reranker-cross-encoder) | [Development](#️-development--reproduction-workflow) | [License & Citations](#️-license--attribution)


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

## ⚡ Benchmark Highlights
- Embedding Models Detailed Report: [docs/benchmark.md](docs/benchmark.md)
- Cross-Encoder Reranker Detailed Report: [docs/reranker_benchmark.md](docs/reranker_benchmark.md) ⚡ *New!*

### 1. Tokenizer Preprocessing Benchmark (10,000 Sentences)
| Metric | Hugging Face Fast Tokenizer | SentencePiece Lite (Ours) | Speedup / Difference |
|---|---|---|---|
| **10,000 Sentences Time** | 455.03 ms | **33.97 ms** | **~13.4x Faster** ⚡ |
| **Throughput** | 21,976 sent/s | **294,395 sent/s** | **~300k sentences/sec** |
| **Per-sentence Latency** | 45.5 µs | **3.40 µs** | **Near-zero overhead** |

### 2. End-to-End Embedding Generation (`encode()` Full Pipeline)
| Metric (30m / CPU) | PyTorch 2.14 (Transformers) | RuriV3Lite (Ours) | Difference |
|---|---|---|---|
| **Single Query Latency** | 6.03 ms | **2.94 ms** | **~2.05x Faster** ⚡ |

### 3. Disk Footprint, Binary Sizes & Codebase Comparison
| Metric | Standard Stack (PyTorch + Transformers) | Our Stack (Zero-Torch & SBP) | Reduction |
|---|---|---|---|
| **Python Runtime Footprint (CPU)** | ~1,240 MB (~1.2 GB) | **~100 MB** | **🔥 92% Reduction** |
| **Python Runtime Footprint (GPU)** | ~3,600 MB (~3.6 GB) | **~380 MB** | **🔥 89% Reduction** |
| **Peak Runtime Memory (30m RAM RSS)** | 980.6 MB | **324.1 MB** | **🔥 67% Reduction** |
| **Tokenizer Wheel Package** | ~10 MB (`sentencepiece`) | **1.7 MB** (`sentencepiece_lite`) | **83% Reduction** |
| **Dictionary Model Loading** | In-memory parse & allocation | **4.57 MB (mmap zero-copy)** | **Zero allocation overhead** |
| **Inference Wrapper Code Length** | 100,000+ lines (heavy dependency tree) | **Only 194 lines** | **Auditable & minimal** |




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

## 🎯 Using the Reranker (Cross-Encoder)

Maximize retrieval precision in RAG systems with the **Ultra-Fast Cross-Encoder Reranker** [`Chottokun/ruri-v3-reranker-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-reranker-310m-lite).  
Detailed Benchmark Report: [docs/reranker_benchmark.md](docs/reranker_benchmark.md)

### Quickstart

```python
from ruri_v3_reranker_lite import RuriV3RerankerLite

# Load model (automatically cached from Hugging Face Hub)
# Precision options:
# - "int8_full": 301 MB ultra-compact (fastest on CPU, 1/4 size, 100% rank preservation) ⚡ [Recommended]
# - "int8": 526 MB linear-only quantized
# - "fp16": 601 MB GPU-optimized (Turing/Ampere+)
# - "fp32": 1,202 MB baseline full precision
reranker = RuriV3RerankerLite(
    repo_id="Chottokun/ruri-v3-reranker-310m-lite",
    precision="int8_full",  # 301 MB ultra-compact model
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

# Get ranked results sorted in descending score order (Top-3)
results = reranker.rerank(query, documents, top_k=3, normalize=True)

for rank, item in enumerate(results, start=1):
    print(f"Rank {rank}: Score={item['score']:.4f} (Index {item['index']}) -> {item['document']}")

# Output:
# Rank 1: Score=0.9990 (Index 0) -> 日本の首都は東京都です。政治・経済の中枢が集約されています。
# Rank 2: Score=0.5633 (Index 1) -> 東京は日本の政治と文化の中心都市であり、多くの観光客が訪れます。
# Rank 3: Score=0.0093 (Index 2) -> 大阪は関西地方の主要都市で、独自の食文化やお笑いで知られています。
```

> **💡 Best Practice: Two-Stage Hybrid Retrieval Pipeline**:
> Since Cross-Encoder inference runs in $O(N)$ computational complexity per query, first retrieve **Top-20~30 candidates** within $< 10$ ms using `ruri-v3-30m-lite` (Bi-Encoder), then rerank them with `ruri-v3-reranker-310m-lite` to balance extreme low latency and maximal ranking precision.

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

---

## ⚖️ License & Redistribution Terms (Apache License, Version 2.0)

This project, converted model binaries, and distribution assets are licensed under the **Apache License, Version 2.0** (the "License"). You may not use these files except in compliance with the License. You may obtain a copy of the License at:

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. A complete copy of the license is included in [LICENSE](LICENSE).


### 1. Base Model Credits
- **Model**: [`cl-nagoya/ruri-v3`](https://huggingface.co/collections/cl-nagoya/ruri-v3-67c006886e0621255e7fcb99) (`30m`, `70m`, `130m`, `310m`)
- **Developer**: Nagoya University Natural Language Processing Laboratory (cl-nagoya)
- **Original Authors**: Hayato Tsukagoshi, Ryohei Sasano
- **License**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Modifications**:
  - Converted and exported official Safetensors weights to ONNX (FP32) and ONNX (FP16) formats.
  - Converted official SentencePiece models into FlatBuffers binaries (`.spm.fb`) for zero-copy memory mapping.
  - Authored a self-contained, PyTorch-free inference wrapper (`ruri_v3_lite.py`).

### 2. Tokenizer Core (SentencePiece Lite)
- **Library**: [Google SentencePiece Lite](https://google.github.io/sentencepiece/lite/)
- **Repository**: [github.com/google/sentencepiece](https://github.com/google/sentencepiece)
- **Copyright**: Copyright 2018 Google LLC
- **License**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **Features**: Ultra-fast C++20 subword tokenizer with Safe Boundary Pre-tokenization (SBP) for multi-threaded parallel execution and FlatBuffers serialization.

### 3. Citations
When using or evaluating Ruri-v3 or these derivative models, please cite the corresponding original papers:

#### Ruri (Embedding Model)
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


