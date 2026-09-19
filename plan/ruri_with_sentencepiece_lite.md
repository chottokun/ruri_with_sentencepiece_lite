# 軽量日本語埋め込み環境（ruri-v3-30m × sentencepiece_lite）完全実装計画書

---

### 1. 背景と目的

* **背景:**
  * 通常の `transformers` + `torch` 構成は依存パッケージだけで数GBに達し、コンテナ起動やサーバーレス環境のボトルネックになる。
  * `sentencepiece` は環境によって C++ コンパイル（CMake/gcc/MSVC）が走り、導入失敗の原因になりやすい。
  * GPU 推論環境であっても、前処理（テキストトークナイズ）は CPU が担当するため、トークナイザーが遅いと GPU が遊ぶ「CPU ボトルネック」が発生する。

* **目的:**
  * **完全ノーコンパイル導入:** SentencePiece Lite の事前ビルド済み Wheel を Hugging Face に集約し、`pip` 1 行で導入可能にする。
  * **極限のフットプリント削減:** PyTorch を完全排除し、実行時依存を `sentencepiece_lite` + `onnxruntime(-gpu)` + `numpy` のみに絞る。
  * **ハードウェア適応型アーキテクチャ:** 実行環境（CPU / GTX 1080 / RTX 3060 等）を自動検知し、最適な精度（FP32/FP16）で最大性能を引き出す。
  * **高速トークナイズ:** SentencePiece Lite の Safe Boundary Pre-tokenization (SBP) によるマルチスレッド並列トークナイズを活用する。

* **ターゲットモデル:** `cl-nagoya/ruri-v3-30m`（37M パラメータ、256次元埋め込み、ModernBERT ベース）
* **配布先リポジトリ:** `chottokun/ruri-v3-30m-lite`

---

### 2. システムアーキテクチャ設計

トークナイズ（CPU）とテンソル演算（GPU）の強みを分離したハイブリッド構成を採用します。

```text
[入力テキスト]
       │
       ▼
【CPU: sentencepiece_lite (SBP対応)】
  ・FlatBuffers (.spm.fb) による zero-copy mmap ロード
  ・文字列正規化 & BPE エンコード（SBP 並列化対応）
  ・特殊トークン付与 & パディング (pad_token_id=3)
       │
       ▼ (NumPy int64 配列)
【GPU / CPU: ONNX Runtime】
  ・Compute Capability 判定 (>= 7.0: FP16 / < 7.0: FP32)
  ・CUDAExecutionProvider による Transformer 行列演算
       │
       ▼ (テンソル出力)
【CPU: NumPy】
  ・Mean Pooling (attention_mask 考慮) ──> L2 正規化 ──> [最終埋め込みベクトル]

```

#### ハードウェア自動判別マトリクス

| 実行環境 | アーキテクチャ | Compute Capability | 選択モデル | 理由 |
| --- | --- | --- | --- | --- |
| **CPU のみ** | x86_64 | - | `model.onnx` (FP32) | CPU ネイティブ演算が最速 |
| **GTX 1080** | Pascal | 6.1 (< 7.0) | `model.onnx` (FP32) | Pascal 世代は FP16 演算器が弱く、FP32 の方が高速 |
| **RTX 3060** | Ampere | 8.6 (≥ 7.0) | `model_fp16.onnx` (FP16) | Tensor コアにより **速度約2倍・VRAM消費半減** |

#### トークン仕様（cl-nagoya/ruri-v3-30m）

| トークン | ID | 役割 |
| --- | --- | --- |
| `<unk>` | 0 | 未知語 |
| `<s>` (BOS) | 1 | 文頭 |
| `</s>` (EOS) | 2 | 文末 |
| `<pad>` | 3 | パディング |
| `<sep>` | 4 | セパレータ |
| `<mask>` | 5 | マスク |
| `<cls>` | 6 | 分類トークン |

> **注意:** `tokenizer_config.json` では `add_bos_token: true, add_eos_token: true` が設定されている。
> Phase 0 で `AutoTokenizer` の出力と照合し、実際に使用される特殊トークン ID を確定する。

---

### 3. Hugging Face リポジトリ構成（配置目標）

Hugging Face の単一リポジトリ（`chottokun/ruri-v3-30m-lite`）にすべての必要資産を集約します。

```text
chottokun/ruri-v3-30m-lite/
├── README.md
├── wheels/
│   ├── sentencepiece_lite-0.1.0-cp311-cp311-manylinux2014_x86_64.whl
│   └── sentencepiece_lite-0.1.0-cp311-cp311-win_amd64.whl
├── ruri_v3_30m.spm.fb             # FlatBuffers 形式トークナイザー (SBP 並列化対応)
├── tokenizer.model                 # 元の SentencePiece モデル (フォールバック用)
├── model.onnx                     # FP32 モデル (CPU / GTX 1080 等)
├── model_fp16.onnx                # FP16 モデル (RTX 3060 / A100 等)
└── ruri_v3_lite.py                # 配布用ラッパーモジュール

```

---

### 4. Phase 0: 特殊トークン検証スクリプト (`verify_tokenizer.py`)

Phase 1 に先立ち、`transformers.AutoTokenizer` の出力を検証し、SentencePiece Lite での手動トークナイズで再現すべき特殊トークンのパターンを確定します。

```python
"""特殊トークン検証スクリプト
transformers.AutoTokenizer の出力を検証し、SentencePiece Lite で再現すべき
特殊トークンのパターンを確定する。
"""
from transformers import AutoTokenizer

MODEL_ID = "cl-nagoya/ruri-v3-30m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# テストケース（ruri-v3 推奨プレフィックス付き）
texts = [
    "検索クエリ: 日本の首都はどこですか？",
    "文章: 日本の首都は東京都です。",
    "短いテスト",
]

print(f"=== 特殊トークン情報 ===")
print(f"  bos_token: {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
print(f"  eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
print(f"  cls_token: {tokenizer.cls_token!r} (id={tokenizer.cls_token_id})")
print(f"  sep_token: {tokenizer.sep_token!r} (id={tokenizer.sep_token_id})")
print(f"  pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
print()

for text in texts:
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"Text: {text}")
    print(f"  先頭トークン: {tokens[0]} (id={ids[0]})")
    print(f"  末尾トークン: {tokens[-1]} (id={ids[-1]})")
    print(f"  IDs (先頭5): {ids[:5]}")
    print(f"  Tokens (先頭5): {tokens[:5]}")
    print(f"  合計長: {len(ids)}")
    print()

```

```bash
uv run --with "torch" --with "transformers" --with "sentencepiece" python scripts/verify_tokenizer.py
```

---

### 5. Phase 1: 資産生成スクリプト (`build_and_export.py`)

初回にビルド環境（PyTorch / C++ コンパイラ搭載環境）で 1 度だけ実行し、必要なすべての成果物を生成します。

```python
"""資産生成スクリプト
SentencePiece Lite wheel のビルド、トークナイザー辞書の取得と FlatBuffers 変換、
FP32 ONNX エクスポート、FP16 変換を行う。
"""
import os
import subprocess
import shutil
import onnx
import torch
from onnxconverter_common import float16
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

MODEL_ID = "cl-nagoya/ruri-v3-30m"
OUTPUT_DIR = "./dist_assets"
os.makedirs(f"{OUTPUT_DIR}/wheels", exist_ok=True)

# 1. sentencepiece_lite の wheel ビルド
# google/sentencepiece リポジトリをクローンし、lite/ 以下の C++ ソースと
# pybind11 バインディングをビルドして wheel を生成する
print("[1/6] Building sentencepiece_lite wheel...")
subprocess.run(["python", "-m", "build", "--wheel", "-o", f"{OUTPUT_DIR}/wheels"], check=True)

# 2. Tokenizer 辞書モデルの取得
# 注意: ruri-v3-30m では "tokenizer.model" が正しいファイル名
#       ("spiece.model" は存在せず 404 になる)
print("[2/6] Fetching tokenizer model...")
sp_path = hf_hub_download(repo_id=MODEL_ID, filename="tokenizer.model")
shutil.copy(sp_path, f"{OUTPUT_DIR}/tokenizer.model")

# 3. FlatBuffers 形式への変換 (SBP 並列化に必要)
print("[3/6] Converting to FlatBuffers format...")
subprocess.run([
    "./spm_to_fb",
    f"--model={sp_path}",
    f"--output={OUTPUT_DIR}/ruri_v3_30m.spm.fb"
], check=True)

# 4. FP32 ONNX エクスポート
print("[4/6] Exporting FP32 ONNX model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()

dummy_inputs = tokenizer(["ダミーテキスト"], return_tensors="pt")
fp32_onnx_path = f"{OUTPUT_DIR}/model.onnx"

torch.onnx.export(
    model,
    (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
    fp32_onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=17,
    do_constant_folding=True
)

# 5. FP16 変換 (RTX 3060 等のモダン GPU 用)
print("[5/6] Converting to FP16...")
fp32_model = onnx.load(fp32_onnx_path)
fp16_model = float16.convert_float_to_float16(
    fp32_model,
    keep_io_types=True,  # input_ids / attention_mask の int64 を保持
    disable_shape_infer=False
)
fp16_onnx_path = f"{OUTPUT_DIR}/model_fp16.onnx"
onnx.save(fp16_model, fp16_onnx_path)

print(f"[6/6] All assets generated successfully in {OUTPUT_DIR}/")

```

```bash
uv run --with "torch" --with "transformers" --with "onnx" --with "onnxconverter-common" --with "build" --with "huggingface_hub" python scripts/build_and_export.py
```

---

### 6. Phase 2: 単体配布用ラッパーモジュール (`ruri_v3_lite.py`)

利用側で動作するクラスです。PyTorch に依存せず `ctypes` で GPU 世代を自動検出し、モデルの切り替えから **Mean Pooling**、L2 正規化までを単一クラスで完結させます。

```python
"""ruri-v3-30m 軽量推論ラッパー
PyTorch 完全不要。sentencepiece_lite + onnxruntime + numpy のみで動作。
"""
import ctypes
import os
import numpy as np
import onnxruntime as ort
import sentencepiece_lite as spl
from huggingface_hub import hf_hub_download

# ruri-v3-30m の特殊トークン ID (Phase 0 で確定した値を記入)
BOS_ID = 1   # <s>
EOS_ID = 2   # </s>
PAD_ID = 3   # <pad>

def get_cuda_compute_capability() -> float:
    """CUDA ドライバ経由で GPU の Compute Capability を取得 (PyTorch 完全不要)"""
    try:
        cuda = ctypes.CDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so")
        if cuda.cuInit(0) != 0:
            return 0.0
        device = ctypes.c_int()
        if cuda.cuDeviceGet(ctypes.byref(device), 0) != 0:
            return 0.0

        major = ctypes.c_int()
        minor = ctypes.c_int()
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
        cuda.cuDeviceGetAttribute(ctypes.byref(major), 75, device)
        cuda.cuDeviceGetAttribute(ctypes.byref(minor), 76, device)
        return major.value + (minor.value / 10.0)
    except Exception:
        return 0.0

class RuriV3Lite:
    def __init__(self, repo_id: str = "chottokun/ruri-v3-30m-lite", force_fp32: bool = False):
        available_providers = ort.get_available_providers()
        use_cuda = "CUDAExecutionProvider" in available_providers

        # 1. ハードウェアに応じたモデル自動判定
        if use_cuda and not force_fp32:
            capability = get_cuda_compute_capability()
            # Turing (7.5) / Ampere (8.6: RTX 3060) 以降は FP16
            # Pascal (6.1: GTX 1080) 等は 7.0 未満のため FP32
            if capability >= 7.0:
                model_filename = "model_fp16.onnx"
                print(f"[RuriV3Lite] GPU Capability {capability:.1f} >= 7.0 detected -> Loading FP16")
            else:
                model_filename = "model.onnx"
                print(f"[RuriV3Lite] GPU Capability {capability:.1f} < 7.0 detected (GTX 1080 etc.) -> Loading FP32")
        else:
            model_filename = "model.onnx"
            print("[RuriV3Lite] Running on CPU or force_fp32=True -> Loading FP32")

        # 2. モデル & トークナイザーの自動キャッシュ取得
        model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
        fb_path = hf_hub_download(repo_id=repo_id, filename="ruri_v3_30m.spm.fb")

        # 3. トークナイザーの初期化 (SentencePiece Lite, FlatBuffers, CPU固定)
        self.tokenizer = spl.FastSBPTokenizer(fb_path)

        # 4. ONNX Runtime 初期化
        if use_cuda:
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }),
                "CPUExecutionProvider"
            ]
        else:
            providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)

    def encode(self, texts: list[str], max_length: int = 512, batch_size: int = 64) -> np.ndarray:
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # CPU 上で sentencepiece_lite による高速トークナイズ
            batch_ids = []
            for text in batch_texts:
                tokens = self.tokenizer.encode(text)
                # BOS/EOS を手動で付与（Phase 0 で確定した ID を使用）
                ids = [BOS_ID] + tokens[:max_length - 2] + [EOS_ID]
                batch_ids.append(ids)

            seq_len = max(len(ids) for ids in batch_ids)
            # pad_token_id = 3 で初期化（ruri-v3-30m の config.json に準拠）
            input_ids = np.full((len(batch_texts), seq_len), fill_value=PAD_ID, dtype=np.int64)
            attention_mask = np.zeros((len(batch_texts), seq_len), dtype=np.int64)

            for b_idx, ids in enumerate(batch_ids):
                input_ids[b_idx, :len(ids)] = ids
                attention_mask[b_idx, :len(ids)] = 1

            # GPU / CPU 推論
            outputs = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

            # Mean Pooling (attention_mask を考慮した加重平均)
            # ruri-v3-30m の公式設定: pooling_mode_mean_tokens = true
            hidden_states = outputs[0].astype(np.float32)
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_embeddings = np.sum(hidden_states * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask

            # L2 正規化 (コサイン類似度直結)
            norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
            normalized = mean_pooled / np.clip(norms, a_min=1e-12, a_max=None)
            all_embeddings.append(normalized)

        return np.vstack(all_embeddings)

```

---

### 7. Phase 3: Hugging Face アップロードスクリプト (`deploy_to_hf.py`)

生成物とラッパーモジュールを Hugging Face リポジトリへ一括転送します。

```python
"""HuggingFace デプロイスクリプト
dist_assets/ 以下の成果物を chottokun/ruri-v3-30m-lite にアップロードする。
"""
import os
import shutil
from huggingface_hub import HfApi

# トークンは環境変数から取得（ハードコード禁止）
HF_TOKEN = os.environ["HF_TOKEN"]
REPO_ID = "chottokun/ruri-v3-30m-lite"

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

# ラッパースクリプト自体もリポジトリにコピー
shutil.copy("dist_assets/ruri_v3_lite.py", "./dist_assets/ruri_v3_lite.py")

print(f"Uploading assets to https://huggingface.co/{REPO_ID} ...")
api.upload_folder(
    folder_path="./dist_assets",
    repo_id=REPO_ID,
    repo_type="model"
)
print("Upload completed.")

```

```bash
export HF_TOKEN="hf_xxx"
uv run --with "huggingface_hub" python scripts/deploy_to_hf.py
```

---

### 8. Phase 4: 利用側でのワンライナー導入・実行手順

配布先の別マシン、コンテナ、または別ユーザーは、C++ コンパイラや PyTorch を一切インストールすることなく利用できます。

#### 1. インストール（コンパイル不要・1行）

```bash
# 環境に合わせて Linux 用または Windows 用の wheel を指定
pip install https://huggingface.co/chottokun/ruri-v3-30m-lite/resolve/main/wheels/sentencepiece_lite-0.1.0-cp311-cp311-manylinux2014_x86_64.whl onnxruntime-gpu numpy huggingface_hub

```

#### 2. 推論コード（自動最適化）

```python
from ruri_v3_lite import RuriV3Lite
import numpy as np

# インスタンス化 (GPU 種別を自動判定し、最適な重みをキャッシュ取得)
model = RuriV3Lite(repo_id="chottokun/ruri-v3-30m-lite")

# ruri-v3 推奨のプレフィックスを付与して埋め込み
queries = ["検索クエリ: 日本の首都はどこですか？"]
documents = [
    "文章: 日本の首都は東京都です。",
    "文章: 大阪は関西地方の主要都市です。"
]

q_vec = model.encode(queries)
d_vec = model.encode(documents)

# コサイン類似度計算 (正規化済みのため内積で算出可能)
similarities = np.dot(q_vec, d_vec.T)[0]

print(f"クエリとの類似度:")
print(f" - 東京の文: {similarities[0]:.4f}")
print(f" - 大阪の文: {similarities[1]:.4f}")

```

---

### 9. Phase 5: 推論精度検証スクリプト (`test_inference.py`)

`sentence-transformers` のリファレンス出力と ONNX + SentencePiece Lite の出力を比較し、コサイン類似度 0.99 以上で一致することを検証します。

```python
"""推論精度検証スクリプト
sentence-transformers のリファレンス出力と ONNX + SentencePiece Lite の出力を比較し、
コサイン類似度が 0.99 以上で一致することを検証する。
"""
import numpy as np

# 1. リファレンス出力 (sentence-transformers)
from sentence_transformers import SentenceTransformer
ref_model = SentenceTransformer("cl-nagoya/ruri-v3-30m")

test_texts = [
    "検索クエリ: 日本の首都はどこですか？",
    "文章: 日本の首都は東京都です。",
    "文章: 大阪は関西地方の主要都市です。",
]
ref_embeddings = ref_model.encode(test_texts)

# 2. ONNX + SentencePiece Lite 推論
from ruri_v3_lite import RuriV3Lite
lite_model = RuriV3Lite(repo_id="chottokun/ruri-v3-30m-lite")
lite_embeddings = lite_model.encode(test_texts)

# 3. コサイン類似度で検証 (閾値 0.99 以上)
print("=== 推論精度検証 ===")
all_passed = True
for i, text in enumerate(test_texts):
    cos_sim = np.dot(ref_embeddings[i], lite_embeddings[i])
    status = "PASS ✅" if cos_sim >= 0.99 else "FAIL ❌"
    if cos_sim < 0.99:
        all_passed = False
    print(f"  [{status}] Text {i}: cosine_sim = {cos_sim:.6f}")

print(f"\n{'全テスト合格 ✅' if all_passed else '一部テスト不合格 ❌'}")

```

```bash
# CPU テスト
uv run --with "onnxruntime" --with "numpy" --with "huggingface_hub" python scripts/test_inference.py

# GPU テスト
uv run --with "onnxruntime-gpu" --with "numpy" --with "huggingface_hub" python scripts/test_inference.py
```