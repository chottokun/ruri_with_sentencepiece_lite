# ruri_sentencepiece_lite

[English](README_en.md) | [日本語](README.md)

PyTorch や Transformers に依存せず、**`sentencepiece_lite` + `onnxruntime` + `numpy`** の軽量な依存関係のみで動作する日本語テキスト埋め込み（Embedding）およびリランカー（Cross-Encoder）推論環境です。

- **ターゲットモデル**: `cl-nagoya/ruri-v3` シリーズ（ModernBERT-Ja アーキテクチャ）
- **Hugging Face モデルリポジトリ**:
  - **30m 埋め込み (256次元)**: [`Chottokun/ruri-v3-30m-lite`](https://huggingface.co/Chottokun/ruri-v3-30m-lite)
  - **70m 埋め込み (384次元)**: [`Chottokun/ruri-v3-70m-lite`](https://huggingface.co/Chottokun/ruri-v3-70m-lite)
  - **130m 埋め込み (512次元)**: [`Chottokun/ruri-v3-130m-lite`](https://huggingface.co/Chottokun/ruri-v3-130m-lite)
  - **310m 埋め込み (768次元)**: [`Chottokun/ruri-v3-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-310m-lite)
  - **310m リランカー (Cross-Encoder)**: [`Chottokun/ruri-v3-reranker-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-reranker-310m-lite)

**ナビゲーション**: [特徴と技術的アプローチ](#技術的アプローチと解決する課題) | [性能評価要約](#性能評価要約) | [埋め込みの利用方法](#埋め込みモデルの利用方法encode) | [リランカーの利用方法](#リランカーの利用方法rerank) | [開発・検証手順](#開発検証ビルド手順開発者向け) | [ライセンス・帰属表示](#ライセンス再配布条件-apache-license-20)

---

## 技術的アプローチと解決する課題

| 従来の PyTorch / Transformers 構成 | 本実装（Zero-Torch 構成） | 技術的要因と特性 |
|---|---|---|
| **パッケージ容量の肥大化**<br>（PyTorch 等で数GB消費） | **軽量フットプリント**<br>（実行環境 約70〜100MB） | 実行時依存を `sentencepiece_lite`, `onnxruntime`, `numpy`, `huggingface_hub` に限定。コンテナサイズ縮小とコールドスタートを短縮。 |
| **SentencePiece のビルド環境依存**<br>（CMake/gcc等のコンパイルが必要） | **事前ビルド済み Wheel の提供** | 事前ビルド済み Wheel（`.whl`）を提供し、C++ コンパイル不要で即時導入可能。 |
| **前処理（トークナイズ）の遅延**<br>（GPUが遊休する要因） | **C++20 によるゼロコピー前処理** | FlatBuffers（`mmap`）および Safe Boundary Pre-tokenization（SBP）により、1文あたり約 3.4 µs で処理。 |
| **旧世代 GPU での FP16 遅延**<br>（Pascal 世代等の制約） | **ハードウェア適応型自動切り替え** | `ctypes` で GPU Compute Capability を判定。7.0 以上は FP16、7.0 未満または CPU は FP32 を自動選択。 |
| **自作ラッパーによる数値乖離の懸念** | **数学的等価性の検証済み** | PyTorch 公式出力（Mean Pooling + L2正規化）とのコサイン類似度 1.0000 を実機検証。 |

---

## ディレクトリ構成

```text
ruri_sentencepiece_lite/
├── AGENTS.md                  # 厳格な開発制約・セキュリティ方針 (uv利用・秘密情報保護)
├── .gitignore                 # 巨大モデル重み・秘密情報・キャッシュの除外設定
├── .env.example               # 環境変数テンプレート (HF_TOKEN 等)
├── docs/
│   ├── benchmark.md           # 埋め込みモデルの性能評価レポート
│   └── reranker_benchmark.md  # リランカーモデルの性能評価・トレードオフ考察レポート
├── dist_assets/               # 配布用資産 (Hugging Face リポジトリ配信用)
│   ├── README.md              # Hugging Face モデルカード
│   ├── ruri_v3_lite.py        # 埋め込み推論ラッパー (PyTorch非依存, 約190行)
│   ├── ruri_v3_reranker_lite.py # リランカー推論ラッパー (PyTorch非依存, 約310行)
│   ├── ruri_v3_30m.spm.fb     # FlatBuffers 形式トークナイザー辞書 (4.57MB)
│   ├── tokenizer.model        # 元の SentencePiece 辞書
│   ├── model.onnx             # FP32 ONNX モデル
│   ├── model_fp16.onnx        # FP16 ONNX モデル
│   └── wheels/                # 事前ビルド済み sentencepiece_lite wheel
└── scripts/
    ├── verify_tokenizer.py    # トークナイザー仕様および特殊トークン ID 検証
    ├── build_and_export.py    # Wheel ビルド、FlatBuffers 変換、ONNX エクスポート
    ├── deploy_to_hf.py        # Hugging Face への埋め込みモデル自動デプロイ
    ├── deploy_reranker_to_hf.py # Hugging Face へのリランカー自動デプロイ
    ├── test_inference.py      # PyTorch 公式出力との数学的等価性検証
    ├── test_reranker_robustness.py # リランカーの境界値・頑健性自動テストスイート
    └── benchmark.py           # 推論レイテンシ・スループット測定スクリプト
```

---

## 性能評価要約

詳細な実測値・プロファイリング・分析については各レポートをご参照ください：
- 埋め込みモデル詳細: [docs/benchmark.md](docs/benchmark.md)
- リランカー詳細: [docs/reranker_benchmark.md](docs/reranker_benchmark.md)

### 1. 前処理トークナイザー単体性能 (10,000 件, CPU)
| トークナイザー実装 | 10,000件処理時間 | スループット | 1件あたり平均レイテンシ | 速度比 |
|---|---|---|---|---|
| **Hugging Face Fast Tokenizer** (Rustベース) | 455.03 ms | 21,976 sent/s | 45.50 µs | 1.00x (基準) |
| **SentencePiece Lite (本実装)** (C++20 Zero-copy) | **33.97 ms** | **294,395 sent/s** | **3.40 µs** | **約 13.4 倍 高速** |

### 2. End-to-End 推論性能とバッチサイズによる特性 (30m / CPU)
| バッチサイズ | PyTorch 2.14 CPU | RuriV3Lite (本実装) CPU | レイテンシ比 | 特性判定 |
|---|---|---|---|---|
| **Batch = 1 (単一クエリ)** | 5.90 ms | **2.98 ms** | **1.98x 高速** | 本実装優位（Pythonオーバーヘッド極小） |
| **Batch = 8** | 15.22 ms | **11.39 ms** | **1.34x 高速** | 本実装優位 |
| **Batch = 16** | 27.35 ms | **26.79 ms** | 1.02x (同等) | 性能交差点（Breakeven point） |
| **Batch = 64** | 93.97 ms | 124.08 ms | 0.76x (PyTorch優位) | 行列演算時間（GEMM）が支配的 |

> **アーキテクチャ上の選定指針**:
> - **WebAPI・検索・サーバーレス環境（Batch 1〜8）**: Python ディスパッチコストが小さく、インストール容量とコールドスタートを大幅に抑制できる本実装が適しています。
> - **オフライン一括バッチ処理（Batch 32〜64）**: Intel MKL 最適化カーネルを持つ PyTorch (LibTorch) がスループット面で約 1.25〜1.3 倍優位となります。

### 3. パッケージ容量・メモリ消費量（実測値）
| 評価項目 | 従来の標準構成 (PyTorch + Transformers) | 本実装 (Zero-Torch 構成) | 削減率 |
|---|---|---|---|
| **Python ランタイム総容量 (CPU)** | 約 1,240 MB (~1.2 GB) | **約 100 MB** | **約 92% 削減** |
| **実行時常駐メモリ (30m RAM RSS)** | 980.6 MB | **324.1 MB** | **約 67% 削減** |
| **トークナイザー Wheel** | 約 10 MB (`sentencepiece`) | **1.7 MB** (`sentencepiece_lite`) | **約 83% 削減** |
| **辞書バイナリ読み込み** | メモリパース展開あり | **4.57 MB (mmap ゼロコピー)** | **メモリ展開負荷ゼロ** |

---

## 埋め込みモデルの利用方法（Encode）

### 1. インストール

```bash
# 事前ビルド済み Wheel と推論ランタイムの導入 (CPU)
pip install https://huggingface.co/Chottokun/ruri-v3-30m-lite/resolve/main/wheels/sentencepiece_lite-0.1.0-cp311-cp311-linux_x86_64.whl \
            onnxruntime numpy huggingface_hub

# GPU 推論を行う場合 (CUDA)
pip install onnxruntime-gpu
```

### 2. 推論コード

```python
from ruri_v3_lite import RuriV3Lite
import numpy as np

# インスタンス化 (ハードウェア世代に応じて FP16 / FP32 を自動選択)
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

# コサイン類似度計算 (L2正規化済みの内積)
similarities = np.dot(q_emb, d_emb.T)[0]

print(f"クエリ vs 東京: {similarities[0]:.4f}")
print(f"クエリ vs 大阪: {similarities[1]:.4f}")
```

---

## リランカーの利用方法（Rerank）

検索システムや RAG の適合度を向上させる Cross-Encoder モデル [`Chottokun/ruri-v3-reranker-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-reranker-310m-lite) です。

### 多段階モデルの選択肢

| `precision` 引数 | 実体サイズ | 配信サイズ (Gzip) | CPU レイテンシ (Top-10) | 順位一致度 | 特徴と用途 |
|---|---|---|---|---|---|
| `"auto"` (デフォルト) | - | - | - | - | GPU で FP16、CPU で INT8-Full を自動選択 |
| `"int8_full"` | 301 MB | 202 MB | 2,116 ms | 100% 一致 | **CPU推奨本流**: ダウンロード最小化・順位完全一致 |
| `"pruned_16l"` | 220 MB | 178 MB | 1,382 ms | Top-1 一致 | **低遅延用途**: 16層間引きによる 1.87倍速（Top-1維持） |
| `"int8"` | 526 MB | - | 2,072 ms | 微小差 | 線形層のみ 8bit 動的量子化 |
| `"fp16"` | 601 MB | - | (GPU専用) | 100% 一致 | Tensor Core GPU 最適化 |
| `"fp32"` | 1,202 MB | - | 2,589 ms | 基準 | FP32 フル精度モデル |

### クイックスタート

```python
from ruri_v3_reranker_lite import RuriV3RerankerLite

# モデルのロード (Hugging Face Hub から自動ダウンロード・透過展開)
reranker = RuriV3RerankerLite(
    repo_id="Chottokun/ruri-v3-reranker-310m-lite",
    precision="int8_full",  # または "pruned_16l"
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

# 降順ソートされたリランキング結果を取得 (Top-3)
results = reranker.rerank(query, documents, top_k=3, normalize=True)

for rank, item in enumerate(results, start=1):
    print(f"Rank {rank}: Score={item['score']:.4f} (Index {item['index']}) -> {item['document']}")
```

> **実務における検索パイプライン設計（2段階検索）**:
> Cross-Encoder は計算量が $O(N)$ となるため、まず `ruri-v3-30m-lite`（Bi-Encoder 埋め込み）で大量の候補から **Top-20〜30 件** を数ミリ秒で絞り込み、その後に `ruri-v3-reranker-310m-lite` で精密リランキングを行うことで、レイテンシと精度のバランスが取れたシステムを構築できます。

---

## LLMライブラリ連携 (LangChain / LlamaIndex)

`RuriV3Lite` および `RuriV3RerankerLite` は LangChain や LlamaIndex などの標準インターフェースをダックタイピングでサポートしており、追加の標準アダプターなしでそのまま各種 VectorStore や Retriever に組み込むことができます。

### 1. LangChain での埋め込み (VectorStore / FAISS)

```python
from ruri_v3_lite import RuriV3Lite
from langchain_community.vectorstores import FAISS

# RuriV3Lite は LangChain Embeddings プロトコル (embed_documents, embed_query) を実装
# 自動的に "検索クエリ: " および "文章: " のプレフィックスを付与します
embeddings = RuriV3Lite(repo_id="Chottokun/ruri-v3-30m-lite")

texts = [
    "日本の首都は東京都です。",
    "大阪は関西地方の主要都市です。"
]

# ベクトルデータベースの作成
db = FAISS.from_texts(texts, embeddings)

# 類似検索
docs = db.similarity_search("日本の首都はどこですか？", k=1)
print(docs[0].page_content)
```

### 2. LlamaIndex での埋め込み (VectorStoreIndex)

```python
from ruri_v3_lite import RuriV3Lite
from llama_index.core import VectorStoreIndex, Document

# RuriV3Lite は LlamaIndex BaseEmbedding プロトコル (get_text_embedding, get_query_embedding) を実装
embed_model = RuriV3Lite(repo_id="Chottokun/ruri-v3-30m-lite")

documents = [
    Document(text="日本の首都は東京都です。"),
    Document(text="大阪は関西地方の主要都市です。")
]

index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = index.as_query_engine()
```

### 3. LangChain でのリランカー (ContextualCompressionRetriever)

```python
from ruri_v3_reranker_lite import RuriV3RerankerLite
from langchain.retrievers import ContextualCompressionRetriever

# RuriV3RerankerLite は LangChain BaseDocumentCompressor プロトコル (compress_documents) を実装
reranker = RuriV3RerankerLite(precision="int8_full")

# 既存のレトリバーと組み合わせた 2 段階検索 pipeline
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=db.as_retriever(search_kwargs={"k": 10})
)

# 精密リランキングされたドキュメントを取得
compressed_docs = compression_retriever.invoke("日本の首都はどこですか？")

---

## 開発・検証・ビルド手順（開発者向け）

本リポジトリでの環境構築・操作は `uv` を使用します。

### 1. 仮想環境の準備
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### 2. トークナイザー仕様の検証
```bash
uv run python scripts/verify_tokenizer.py
```
> `<s>` (1), `</s>` (2), `<pad>` (3) の特殊トークン仕様および公式出力との一致を確認します。

### 3. 資産のビルドとエクスポート
```bash
uv run python scripts/build_and_export.py
```
> SentencePiece Lite の Wheel コンパイル、FlatBuffers 形式への変換、FP32/FP16 ONNX へのエクスポートを一括実行します。

### 4. 推論精度の完全一致検証
```bash
uv run python scripts/test_inference.py
```
> PyTorch 公式モデル出力と本実装の出力を照合し、コサイン類似度 0.9999 以上（実測 1.0000）を検証します。

### 5. リランカーの頑健性テスト
```bash
uv run python scripts/test_reranker_robustness.py
```
> 空入力、超長文、特殊文字、バッチ境界値を含む全テスト項目を検証します。

### 6. Hugging Face へのデプロイ

設定用テンプレートから `.env` を作成します（Git 管理からは自動除外されます）：
```bash
cp .env.example .env
# .env を編集して HF_TOKEN="hf_xxx", 任意で HF_USERNAME="xxx" を設定
```

指定モデルをアップロードします：
```bash
# 埋め込みモデルのデプロイ
uv run python scripts/deploy_to_hf.py --model 30m

# リランカーモデルのデプロイ
uv run python scripts/deploy_reranker_to_hf.py
```

---

## セキュリティと機微情報保護方針
- Hugging Face の書き込みトークン（`HF_TOKEN`）等の秘密情報はコード内に記述せず、環境変数経由でのみ受け取ります。
- 巨大なモデルバイナリ（`*.onnx`, `*.onnx.data`, `*.spm.fb`, `*.whl`）やキャッシュディレクトリは `.gitignore` により Git 管理から除外されています。

---

## ライセンス・再配布条件 (Apache License 2.0)

本プロジェクトのソースコード、変換済みモデルバイナリ、および配布資材は、**Apache License, Version 2.0** に基づいて提供されます。ライセンスの写しは [LICENSE](LICENSE) ファイルに収録されています。

### 1. ベースモデル (Base Model)
- **モデル**: [`cl-nagoya/ruri-v3`](https://huggingface.co/collections/cl-nagoya/ruri-v3-67c006886e0621255e7fcb99) (`30m`, `70m`, `130m`, `310m`, `reranker-310m`)
- **開発元**: 名古屋大学 自然言語処理研究室 (Nagoya University, cl-nagoya)
- **原著作者**: 塚越 隼人 (Hayato Tsukagoshi), 笹野 遼平 (Ryohei Sasano)
- **ライセンス**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **改変内容**:
  - 公式 Safetensors 重みから ONNX (FP32, FP16, INT8, Pruned) 形式への変換・エクスポート
  - 公式 SentencePiece 辞書を SentencePiece Lite 用 FlatBuffers バイナリ (`.spm.fb`) に変換
  - PyTorch / Transformers 非依存の推論ラッパー (`ruri_v3_lite.py`, `ruri_v3_reranker_lite.py`) の新規実装

### 2. トークナイザーコア (SentencePiece Lite)
- **ライブラリ**: [Google SentencePiece Lite](https://google.github.io/sentencepiece/lite/)
- **リポジトリ**: [github.com/google/sentencepiece](https://github.com/google/sentencepiece)
- **権利表記**: Copyright 2018 Google LLC
- **ライセンス**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

### 3. 引用 (Citations)

#### Ruri (埋め込み・リランカーモデル)
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

#### SentencePiece (トークナイザー)
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
