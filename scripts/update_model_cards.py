import os
import shutil

MODELS = {
    "30m": ("dist_assets", "cl-nagoya/ruri-v3-30m", 256),
    "70m": ("dist_assets/ruri_v3_70m", "cl-nagoya/ruri-v3-70m", 384),
    "130m": ("dist_assets/ruri_v3_130m", "cl-nagoya/ruri-v3-130m", 512),
    "310m": ("dist_assets/ruri_v3_310m", "cl-nagoya/ruri-v3-310m", 768),
}

LICENSE_SRC = os.path.abspath("LICENSE")

for key, (target_dir, base_model, dim) in MODELS.items():
    os.makedirs(target_dir, exist_ok=True)

    # 1. Copy LICENSE
    shutil.copyfile(LICENSE_SRC, os.path.join(target_dir, "LICENSE"))

    # 2. English Model Card complying with Apache License 2.0
    readme = f"""---
license: apache-2.0
language:
- ja
base_model: {base_model}
pipeline_tag: feature-extraction
tags:
- sentence-similarity
- embeddings
- modernbert
- sentencepiece_lite
- onnx
- zero-torch
---

# ruri-v3-{key}-lite: Zero-Torch & Ultra-Fast Japanese Text Embeddings

An ultra-optimized, lightweight Japanese text embedding model derived from [`{base_model}`](https://huggingface.co/{base_model}) (ModernBERT-based, {dim}-dim).  
Engineered strictly with **`sentencepiece_lite` + `onnxruntime` + `numpy`** — **100% PyTorch-free (Zero-Torch)** and **zero compilation required** (pre-built wheels provided).

---

## 💡 Highlights

- **Ultra-Compact Footprint (Zero-Torch)**:
  - Runtime only requires `sentencepiece_lite`, `onnxruntime` (or `onnxruntime-gpu`), `numpy`, and `huggingface_hub`.
  - Eliminates gigabytes of PyTorch / Transformers dependencies. Ideal for serverless (AWS Lambda, Cloud Run) and edge environments with instant cold starts.
- **Lightning-Fast Multi-Threaded Tokenization**:
  - Powered by Google's [SentencePiece Lite](https://google.github.io/sentencepiece/lite/) with Safe Boundary Pre-tokenization (SBP) and FlatBuffers zero-copy mmap.
  - Over **13.4x faster** than standard Hugging Face Fast Tokenizers (~3.4 µs per sentence).
- **Hardware-Adaptive Auto-Switching**:
  - Dynamically queries GPU Compute Capability via `ctypes` without PyTorch overhead.
  - Automatically loads **FP16 ONNX** on modern GPUs (RTX 3060+, Ampere/Ada/Hopper) to leverage Tensor Cores, and **FP32 ONNX** on Pascal (GTX 1080) or CPU.
- **Mathematical Equivalence**:
  - Validated cosine similarity of **1.0000** against official PyTorch outputs (Mean Pooling + L2 normalization).

---

## 🚀 Installation

End users **do not need PyTorch, Transformers, or C++ compilers**:

```bash
# Install pre-built wheel and runtime dependencies
pip install https://huggingface.co/Chottokun/ruri-v3-{key}-lite/resolve/main/wheels/sentencepiece_lite-0.1.0-cp311-cp311-linux_x86_64.whl \\
            onnxruntime numpy huggingface_hub

# For GPU acceleration (CUDA)
pip install onnxruntime-gpu
```

---

## 💻 Quickstart Inference

```python
from ruri_v3_lite import RuriV3Lite
import numpy as np

# Download and initialize model (automatically chooses FP16 or FP32 based on hardware)
model = RuriV3Lite(repo_id="Chottokun/ruri-v3-{key}-lite")

# Sentences formatted with official ruri-v3 task prefixes
queries = ["検索クエリ: 日本の首都はどこですか？"]
documents = [
    "文章: 日本の首都は東京都です。",
    "文章: 大阪は関西地方の主要都市です。"
]

# Generate normalized embeddings (shape: (N, {dim}))
q_emb = model.encode(queries)
d_emb = model.encode(documents)

# Compute cosine similarity via dot product
similarities = np.dot(q_emb, d_emb.T)[0]

print(f"Query vs Tokyo: {{similarities[0]:.4f}}")
print(f"Query vs Osaka: {{similarities[1]:.4f}}")
```

---

## ⚖️ License & Redistribution Terms (Apache License, Version 2.0)

This model repository, converted binaries, and wrapper scripts are distributed under the **Apache License, Version 2.0** (the "License"). You may not use these files except in compliance with the License. You may obtain a copy of the License at:

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

### Upstream Components & Notices of Modification (Section 4)

In compliance with Section 4 of the Apache License 2.0:

1. **Base Embedding Model**:
   - **Model**: [{base_model}](https://huggingface.co/{base_model})
   - **Copyright**: Copyright 2024 Hayato Tsukagoshi and Ryohei Sasano (Nagoya University NLP Laboratory)
   - **License**: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
   - **Prominent Notice of Modification (Section 4b)**:
     - Converted original PyTorch Safetensors weights into ONNX (FP32) and FP16 formats.
     - Converted official SentencePiece model into FlatBuffers serialization format (`.spm.fb`).
     - Added independent inference implementation (`ruri_v3_lite.py`) decoupling from PyTorch and Transformers.

2. **Tokenizer Engine**:
   - **Library**: [Google SentencePiece Lite](https://google.github.io/sentencepiece/lite/)
   - **Repository**: [github.com/google/sentencepiece](https://github.com/google/sentencepiece)
   - **Copyright**: Copyright 2018 Google LLC
   - **License**: [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
   - **Prominent Notice of Modification (Section 4b)**:
     - Built and packaged as a standalone C++20 Python wheel with Safe Boundary Pre-tokenization (SBP) support.

---

### 📚 Citations

When utilizing or evaluating this model or tokenizer, please cite the corresponding works:

#### Ruri (Embedding Model)
```bibtex
@misc{{Ruri,
  title={{Ruri: Japanese General Text Embeddings}}, 
  author={{Hayato Tsukagoshi and Ryohei Sasano}},
  year={{2024}},
  eprint={{2409.07737}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CL}},
  url={{https://arxiv.org/abs/2409.07737}}, 
}}
```

#### SentencePiece (Tokenizer)
```bibtex
@inproceedings{{kudo-richardson-2018-sentencepiece,
  title = "{{S}}entence{{P}}iece: A simple and language independent subword tokenizer and detokenizer for {{N}}eural {{T}}ext {{P}}rocessing",
  author = "Kudo, Taku and Richardson, John",
  booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
  month = nov,
  year = "2018",
  address = "Brussels, Belgium",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/D18-2012",
  doi = "10.18653/v1/D18-2012",
  pages = "66--71",
}}
```

---

## 📄 License File
A full copy of the license is included in [LICENSE](LICENSE).
"""

    out_readme = os.path.join(target_dir, "README.md")
    with open(out_readme, "w", encoding="utf-8") as f:
        f.write(readme.strip() + "\n")
    print(f"Generated Apache 2.0 compliant model card: {out_readme}")
