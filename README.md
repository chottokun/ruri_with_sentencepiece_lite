# ruri_sentencepiece_lite

[English](README_en.md) | [日本語](README.md)

`torch` や `transformers` を一切使用せず、**`sentencepiece_lite` + `onnxruntime` + `numpy`** だけで動作する最速・極小フットプリントの日本語埋め込み（Embedding）環境です。

ターゲットモデル: `cl-nagoya/ruri-v3` シリーズ（ModernBERTベース）  
配布先モデルハブ（Hugging Face）:
- **30m (256次元)**: [`Chottokun/ruri-v3-30m-lite`](https://huggingface.co/Chottokun/ruri-v3-30m-lite)
- **70m (384次元)**: [`Chottokun/ruri-v3-70m-lite`](https://huggingface.co/Chottokun/ruri-v3-70m-lite)
- **130m (512次元)**: [`Chottokun/ruri-v3-130m-lite`](https://huggingface.co/Chottokun/ruri-v3-130m-lite)
- **310m (768次元)**: [`Chottokun/ruri-v3-310m-lite`](https://huggingface.co/Chottokun/ruri-v3-310m-lite)

**ナビゲーション**: [特徴](#-特徴と解決する課題) | [ベンチマーク](#-ベンチマーク要約-sentencepiece-lite--ort-vs-pytorch) | [利用方法](#-利用方法エンドユーザー環境) | [開発手順](#️-開発検証ビルド手順開発者向け) | [ライセンス・帰属表示・引用](#️-ライセンス帰属表示引用-license--attribution)


---

## 💡 特徴と解決する課題

| 課題（従来の PyTorch / Transformers 構成） | 本リポジトリの解決策（Zero-Torch & SBP） |
|---|---|
| **パッケージサイズが巨大**（PyTorch等で数GB消費） | **極小フットプリント**: 実行時は `sentencepiece_lite` + `onnxruntime` + `numpy` のみで動作。ディスク・起動オーバーヘッドを大幅削減。 |
| **SentencePiece のビルド失敗**（CMake/gcc等のコンパイルが必要） | **完全ノーコンパイル導入**: 事前ビルド済み Wheel（`.whl`）を提供。`pip` 1行で導入可能。 |
| **GPU が遊ぶ CPU ボトルネック**（前処理トークナイズが遅い） | **超高速並列トークナイズ**: Google の SentencePiece Lite と **Safe Boundary Pre-tokenization (SBP)** によるマルチスレッド処理。 |
| **Pascal 世代（GTX 1080等）での FP16 低速化** | **ハードウェア適応型自動切り替え**: `ctypes` で GPU 世代（Compute Capability）を判定し、新世代は FP16、旧世代/CPU は FP32 を自動選択。 |
| **自作実装による精度のズレ** | **数学的等価性**: PyTorch 公式出力（Mean Pooling + L2正規化）とのコサイン類似度 **1.0000** を実機実証済み。 |

---

## 📁 ディレクトリ構成

```text
ruri_sentencepiece_lite/
├── AGENTS.md                  # 厳格な制約事項・セキュリティ方針 (uv利用・秘密情報保護)
├── .gitignore                 # 巨大モデル重み・秘密情報・キャッシュの完全除外設定
├── docs/
│   └── benchmark.md           # 性能評価・トークナイザー＆推論ベンチマークレポート
├── dist_assets/               # 配布用資産 (Hugging Face にアップロードされる成果物)
│   ├── README.md              # Hugging Face モデルカード用ドキュメント
│   ├── ruri_v3_lite.py        # 超軽量推論ラッパーモジュール (PyTorch非依存)
│   ├── ruri_v3_30m.spm.fb     # FlatBuffers 形式トークナイザー辞書 (4.57MB)
│   ├── tokenizer.model        # 元の SentencePiece 辞書
│   ├── model.onnx             # FP32 ONNX モデル (CPU / GTX 1080等)
│   ├── model.onnx.data        # FP32 外部テンソルバイナリ (141MB)
│   ├── model_fp16.onnx        # FP16 ONNX モデル (RTX 3060 / A100等, 71MB)
│   └── wheels/                # 事前ビルド済み sentencepiece_lite wheel
│       └── sentencepiece_lite-0.1.0-cp311-cp311-linux_x86_64.whl
└── scripts/
    ├── verify_tokenizer.py    # Phase 0: 特殊トークン ID & プレフィックス検証
    ├── build_and_export.py    # Phase 1: Wheel ビルド・FlatBuffers 変換・ONNX エクスポート
    ├── deploy_to_hf.py        # Phase 3: Hugging Face への安全な自動デプロイ
    ├── test_inference.py      # Phase 5: PyTorch公式出力との数学的等価性検証
    └── benchmark.py           # 推論速度・スループット測定スクリプト
```

---

## ⚡ ベンチマーク要約 (SentencePiece Lite & ORT vs PyTorch)

詳細レポート: [docs/benchmark.md](docs/benchmark.md)

### 1. 前処理トークナイザー単体性能 (10,000件)
| 項目 | Hugging Face Fast Tokenizer | SentencePiece Lite (本実装) | 性能差 |
|---|---|---|---|
| **10,000件処理時間** | 455.03 ms | **33.97 ms** | **約 13.4 倍 高速** ⚡ |
| **スループット** | 21,976 sent/s | **294,395 sent/s** | **毎秒約30万文** |
| **1件あたり平均レイテンシ** | 45.5 µs | **3.40 µs** | **極小オーバーヘッド** |

### 2. End-to-End 埋め込み生成性能 (`encode()` 総合)
| 項目 (30m / CPU) | PyTorch 2.14 (Transformers) | RuriV3Lite (本実装) | 性能差 |
|---|---|---|---|
| **単一クエリ応答レイテンシ** | 6.03 ms | **2.94 ms** | **約 2.05 倍 高速** ⚡ |

### 3. パッケージ容量・バイナリサイズ・コード規模の実測比較
| 項目 | 従来の標準構成 (PyTorch + Transformers) | 本実装 (Zero-Torch 構成) | 削減率 |
|---|---|---|---|
| **Python ランタイム総容量 (CPU)** | 約 1,240 MB (~1.2 GB) | **約 100 MB** | **🔥 92% 削減** |
| **Python ランタイム総容量 (GPU)** | 約 3,600 MB (~3.6 GB) | **約 380 MB** | **🔥 89% 削減** |
| **実行時ピークメモリ (30m RAM RSS)** | 980.6 MB | **324.1 MB** | **🔥 67% 削減** |
| **トークナイザー Wheel** | 約 10 MB (`sentencepiece`) | **1.7 MB** (`sentencepiece_lite`) | **83% 削減** |
| **辞書バイナリ読み込み** | メモリパース展開あり | **4.57 MB (mmap ゼロコピー)** | **メモリ展開オーバーヘッド 0** |
| **推論ラッパーコード行数** | 数十万行（巨大な依存ツリー） | **わずか 194 行** | **可読性・監査容易** |




---

## 🚀 利用方法（エンドユーザー環境）

エンドユーザー環境では、**PyTorch や Transformers、C++ コンパイラをインストールする必要はありません。**

### 1. インストール

```bash
# 事前ビルド済み Wheel と推論ランタイムの導入
pip install https://huggingface.co/Chottokun/ruri-v3-30m-lite/resolve/main/wheels/sentencepiece_lite-0.1.0-cp311-cp311-linux_x86_64.whl \
            onnxruntime numpy huggingface_hub

# GPU 推論を行う場合 (CUDA)
pip install onnxruntime-gpu
```

### 2. 推論コード

```python
from ruri_v3_lite import RuriV3Lite
import numpy as np

# インスタンス化 (GPU の世代を自動判定し、最適な FP16 または FP32 モデルをロード)
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

# コサイン類似度計算 (内積で算出可能)
similarities = np.dot(q_emb, d_emb.T)[0]

print(f"クエリ vs 東京: {similarities[0]:.4f}")
print(f"クエリ vs 大阪: {similarities[1]:.4f}")
```

---

## 🛠️ 開発・検証・ビルド手順（開発者向け）

本リポジトリでの全操作は `uv` を使用します。

### 1. 仮想環境の準備
```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
```

### 2. トークナイザー仕様の検証 (Phase 0)
```bash
uv run python scripts/verify_tokenizer.py
```
> `<s>` (1), `</s>` (2), `<pad>` (3) の特殊トークン仕様および公式出力との 100% 一致を確認します。

### 3. 資産のビルドとエクスポート (Phase 1)
```bash
uv run python scripts/build_and_export.py
```
> SentencePiece Lite の Wheel コンパイル、FlatBuffers 形式への変換、FP32/FP16 ONNX へのエクスポートを一括実行します。

### 4. 推論精度の完全一致検証 (Phase 5)
```bash
uv run python scripts/test_inference.py
```
> PyTorch 公式モデル出力と本実装の出力を直接照合し、コサイン類似度 **0.9999以上（実測 1.00000000）** を検証します。

### 5. Hugging Face へのデプロイ (Phase 3)

設定用テンプレートから `.env` を作成します（`.gitignore` により Git 管理からは自動除外されます）：
```bash
cp .env.example .env
# .env を編集して HF_TOKEN="hf_xxx", 任意で HF_USERNAME="xxx" を設定
```

指定モデルまたは全モデルをアップロードします：
```bash
# 70m をデプロイする場合
uv run python scripts/deploy_to_hf.py --model 70m

# または環境変数を直接渡してデプロイ
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
uv run python scripts/deploy_to_hf.py --model 130m
```

---

## 🔒 セキュリティと機微情報保護方針
- Hugging Face の書き込みトークン（`HF_TOKEN`）を含む一切の秘密情報はコード内に記述せず、環境変数経由でのみ受け取ります。
- 巨大なモデルバイナリ（`*.onnx`, `*.onnx.data`, `*.spm.fb`, `*.whl`）やキャッシュディレクトリは `.gitignore` により Git 追跡から完全に除外されています。

---

## ⚖️ ライセンス・再配布条件 (Apache License 2.0)

本プロジェクトのソースコード、変換済みモデルバイナリ、および配布資材は、**Apache License, Version 2.0**（以下「本ライセンス」）に基づいて提供されます。本ライセンスに準拠しない限り、これらのファイルを使用することはできません。ライセンスの写しは以下から入手できます：

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

適用される法律で義務付けられている場合、または書面で合意されている場合を除き、本ライセンスに基づいて配布されるソフトウェアは、明示的または黙示的を問わず、いかなる種類の保証も条件もなしに「現状のまま（AS IS）」で配布されます。本ライセンスに基づく権限および制限を規定する特定の言語については、本ライセンスを参照してください（リポジトリ内の [LICENSE](LICENSE) ファイルに全文を収録）。


### 1. ベースモデル (Base Model)
- **モデル**: [`cl-nagoya/ruri-v3`](https://huggingface.co/collections/cl-nagoya/ruri-v3-67c006886e0621255e7fcb99) (`30m`, `70m`, `130m`, `310m`)
- **開発元**: 名古屋大学 自然言語処理研究室 (Nagoya University, cl-nagoya)
- **原著作者**: 塚越 隼人 (Hayato Tsukagoshi), 笹野 遼平 (Ryohei Sasano)
- **ライセンス**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **改変内容**:
  - 公式 Safetensors 重みから ONNX (FP32) および ONNX (FP16) 形式への変換・エクスポート
  - 公式 SentencePiece 辞書を SentencePiece Lite 用 FlatBuffers バイナリ (`.spm.fb`) に変換
  - PyTorch / Transformers を一切使用しない高速推論ラッパー (`ruri_v3_lite.py`) の新規実装

### 2. トークナイザーコア (SentencePiece Lite)
- **ライブラリ**: [Google SentencePiece Lite](https://google.github.io/sentencepiece/lite/)
- **リポジトリ**: [github.com/google/sentencepiece](https://github.com/google/sentencepiece)
- **権利表記**: Copyright 2018 Google LLC
- **ライセンス**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **特徴**: C++20 で実装された極小・超高速なサブワードトークナイザー。Safe Boundary Pre-tokenization (SBP) によるマルチスレッド並列処理と FlatBuffers によるゼロコピー mmap 辞書読み込みに対応。

### 3. 原著論文の引用 (Citations)
本リポジトリおよびモデルをご利用の際は、以下の原著論文の引用をお願いいたします：

#### Ruri (埋め込みモデル)
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


