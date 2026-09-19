---
license: apache-2.0
language:
- ja
pipeline_tag: feature-extraction
tags:
- sentence-similarity
- embeddings
- sentencepiece_lite
- onnx
- zero-torch
---

# ruri-v3-30m-lite: Zero-Torch & Ultra-Fast Japanese Embeddings

[`cl-nagoya/ruri-v3-30m`](https://huggingface.co/cl-nagoya/ruri-v3-30m)（ModernBERTベース・256次元）を、**PyTorch / Transformers 一切不要（Zero-Torch）** かつ **C++ コンパイル不要（事前ビルドWheel配布）** で実行できるように最適化した軽量日本語埋め込み環境です。

## 特徴

1. **極小フットプリント (Zero-Torch)**:
   - 実行時依存関係は `sentencepiece_lite`, `onnxruntime` (または `onnxruntime-gpu`), `numpy`, `huggingface_hub` のみ。
   - `torch` や `transformers` によるギガバイト級のディスク消費を回避し、サーバーレスやエッジ環境でも即時起動可能。
2. **完全ノーコンパイル導入**:
   - Googleの SentencePiece Lite (C++20, SBP対応) を事前ビルド済み Wheel として提供。
3. **ハードウェア適応型推論**:
   - `ctypes` による CUDA Compute Capability の自動判定により、Turing/Ampere以降（RTX 3060等）では Tensor コアに最適化された **FP16 モデル**、CPUや旧世代GPU（GTX 1080等）では **FP32 モデル** を自動選択。
4. **数学的等価性**:
   - PyTorch 公式出力（Mean Pooling + L2正規化）とのコサイン類似度 **1.0000** を実証済み。

---

## インストール手順

### 1. 必要パッケージのインストール（PyTorch不要）

```bash
# Wheel の直接インストールと ONNX Runtime の導入
pip install https://huggingface.co/Chottokun/ruri-v3-30m-lite/resolve/main/wheels/sentencepiece_lite-0.1.0-cp311-cp311-linux_x86_64.whl onnxruntime numpy huggingface_hub

# GPU を使用する場合
pip install onnxruntime-gpu
```

---

## クイックスタート

`ruri_v3_lite.py` をダウンロードするか、同ファイル内の `RuriV3Lite` クラスをインポートして使用します。

```python
from ruri_v3_lite import RuriV3Lite
import numpy as np

# インスタンス化 (Hugging Face Hub からモデルとトークナイザーを自動キャッシュ)
model = RuriV3Lite(repo_id="Chottokun/ruri-v3-30m-lite")

# ruri-v3 推奨のプレフィックス付きテキスト
queries = ["検索クエリ: 日本の首都はどこですか？"]
documents = [
    "文章: 日本の首都は東京都です。",
    "文章: 大阪は関西地方の主要都市です。"
]

# 埋め込みベクトルの取得 (L2正規化済み, shape: (N, 256))
q_emb = model.encode(queries)
d_emb = model.encode(documents)

# コサイン類似度計算 (内積)
similarities = np.dot(q_emb, d_emb.T)[0]

print(f"クエリ vs 東京: {similarities[0]:.4f}")
print(f"クエリ vs 大阪: {similarities[1]:.4f}")
```

---

## リポジトリ構成

```text
chottokun/ruri-v3-30m-lite/
├── README.md
├── ruri_v3_lite.py               # 推論用ラッパーモジュール (Zero-Torch)
├── ruri_v3_30m.spm.fb            # FlatBuffers 形式トークナイザー (Zero-Copy mmap)
├── tokenizer.model                # 元の SentencePiece モデル (フォールバック用)
├── model.onnx                     # FP32 ONNX モデル (CPU / 旧世代GPU用)
├── model.onnx.data                # FP32 外部テンソルデータ
├── model_fp16.onnx                # FP16 ONNX モデル (モダンGPU用)
└── wheels/
    └── sentencepiece_lite-0.1.0-cp311-cp311-linux_x86_64.whl
```
