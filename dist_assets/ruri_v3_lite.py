"""ruri-v3-30m 超軽量・高速推論ラッパーモジュール
PyTorch / Transformers は完全不要。
依存パッケージ: sentencepiece_lite, onnxruntime / onnxruntime-gpu, numpy, huggingface_hub
"""
import ctypes
import os
from typing import List, Union
import numpy as np
import onnxruntime as ort
import sentencepiece_lite as spl
from huggingface_hub import hf_hub_download

# ruri-v3-30m の確定済み特殊トークン ID
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
    """ruri-v3-30m 軽量推論クラス"""

    def __init__(
        self,
        repo_id: str = "Chottokun/ruri-v3-30m-lite",
        model_dir: str = None,
        force_fp32: bool = False,
        device: str = "auto",  # "auto", "cuda", "cpu"
        query_prefix: str = "検索クエリ: ",
        document_prefix: str = "文章: "
    ):
        """
        初期化:
        model_dir が指定されている場合はローカルパスから読み込み、
        未指定の場合は Hugging Face Hub (repo_id) からダウンロード・キャッシュします。

        Args:
            repo_id: Hugging Face リポジトリID
            model_dir: ローカルモデル保存ディレクトリ
            force_fp32: True の場合強制的に FP32
            device: "auto", "cuda", "cpu"
            query_prefix: LangChain / LlamaIndex の embed_query 等で自動付与する接頭辞
            document_prefix: LangChain / LlamaIndex の embed_documents 等で自動付与する接頭辞
        """
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        available_providers = ort.get_available_providers()
        want_cuda = (device == "cuda") or (device == "auto" and "CUDAExecutionProvider" in available_providers)

        # 1. ハードウェア世代に応じたモデル判定
        if want_cuda and not force_fp32:
            capability = get_cuda_compute_capability()
            # Turing (7.5) / Ampere (8.6: RTX 3060) 以降は FP16
            # Pascal (6.1: GTX 1080) 等は 7.0 未満のため FP32
            if capability >= 7.0:
                model_filename = "model_fp16.onnx"
                print(f"[RuriV3Lite] GPU Capability {capability:.1f} >= 7.0 検知 -> FP16 モデルをロード")
            else:
                model_filename = "model.onnx"
                print(f"[RuriV3Lite] GPU Capability {capability:.1f} < 7.0 検知 (GTX 1080等) -> FP32 モデルをロード")
        else:
            model_filename = "model.onnx"
            print("[RuriV3Lite] CPU 実行または force_fp32=True -> FP32 モデルをロード")

        # 2. モデル & トークナイザーの取得
        model_size = "30m"
        for s in ["310m", "130m", "70m", "30m"]:
            if s in repo_id.lower():
                model_size = s
                break

        fb_filename = f"ruri_v3_{model_size}.spm.fb"

        if model_dir and os.path.exists(model_dir):
            model_path = os.path.join(model_dir, model_filename)
            fb_path = os.path.join(model_dir, fb_filename)
            if not os.path.exists(fb_path):
                fbs = [f for f in os.listdir(model_dir) if f.endswith(".spm.fb")]
                if fbs:
                    fb_path = os.path.join(model_dir, fbs[0])
        else:
            model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
            try:
                fb_path = hf_hub_download(repo_id=repo_id, filename=fb_filename)
            except Exception:
                fb_path = hf_hub_download(repo_id=repo_id, filename="ruri_v3_30m.spm.fb")

        # 3. SentencePiece Lite トークナイザーの初期化 (FlatBuffers, SBP並列対応)
        self.tokenizer = spl.FastSBPTokenizer(fb_path)

        # 4. ONNX Runtime セッション初期化 (CUDA 失敗時は自動的に CPU へフォールバック)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if want_cuda:
            providers = [
                ("CUDAExecutionProvider", {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "DEFAULT",
                    "do_copy_in_default_stream": True,
                }),
                "CPUExecutionProvider"
            ]
            try:
                self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
                # 実際に CUDA が選ばれたか確認
                active_provider = self.session.get_providers()[0]
                print(f"[RuriV3Lite] InferenceSession 初期化完了 (プロバイダ: {active_provider})")
            except Exception as e:
                print(f"[RuriV3Lite] CUDA 初期化失敗 ({e}) -> CPUExecutionProvider にフォールバックします")
                self.session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        else:
            self.session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
            print("[RuriV3Lite] InferenceSession 初期化完了 (プロバイダ: CPUExecutionProvider)")

    def encode(
        self,
        texts: Union[str, List[str]],
        max_length: int = 512,
        batch_size: int = 64,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        テキストのリストを高次元埋め込みベクトルに変換します。

        Args:
            texts: 単一文字列または文字列のリスト
            max_length: 最大トークン長 (BOS/EOS含む)
            batch_size: バッチサイズ
            normalize_embeddings: True の場合 L2 正規化を実行 (コサイン類似度用)

        Returns:
            np.ndarray: shape (len(texts), 256) の埋め込みベクトル
        """
        if isinstance(texts, str):
            texts = [texts]

        if len(texts) == 0:
            return np.empty((0, 256), dtype=np.float32)

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # CPU 上で sentencepiece_lite による高速トークナイズ
            batch_ids = []
            for text in batch_texts:
                tokens = self.tokenizer.encode(text)
                # BOS (1) + tokens + EOS (2)
                # max_length - 2 で切り詰め
                clipped_tokens = tokens[:max(0, max_length - 2)]
                ids = [BOS_ID] + clipped_tokens + [EOS_ID]
                batch_ids.append(ids)

            seq_len = max(len(ids) for ids in batch_ids)
            # PAD_ID = 3 でパディング初期化
            input_ids = np.full((len(batch_texts), seq_len), fill_value=PAD_ID, dtype=np.int64)
            attention_mask = np.zeros((len(batch_texts), seq_len), dtype=np.int64)

            for b_idx, ids in enumerate(batch_ids):
                input_ids[b_idx, :len(ids)] = ids
                attention_mask[b_idx, :len(ids)] = 1

            # ONNX Runtime 推論
            outputs = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

            # Mean Pooling (attention_mask を考慮した加重平均)
            # ruri-v3-30m 公式仕様: pooling_mode_mean_tokens = true
            hidden_states = outputs[0].astype(np.float32)
            mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
            sum_embeddings = np.sum(hidden_states * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            mean_pooled = sum_embeddings / sum_mask

            # L2 正規化
            if normalize_embeddings:
                norms = np.linalg.norm(mean_pooled, axis=1, keepdims=True)
                mean_pooled = mean_pooled / np.clip(norms, a_min=1e-12, a_max=None)

            all_embeddings.append(mean_pooled)

        return np.vstack(all_embeddings)

    # =========================================================================
    # LangChain / LlamaIndex 互換インターフェース (Embeddings Protocol)
    # =========================================================================

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        LangChain Embeddings 互換メソッド: ドキュメントのリストを埋め込みベクトルに変換します。
        自動的に `document_prefix` (デフォルト: "文章: ") を付与します。
        """
        prefixed_texts = [
            f"{self.document_prefix}{text}" if self.document_prefix and not text.startswith(self.document_prefix) else text
            for text in texts
        ]
        embeddings = self.encode(prefixed_texts)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        LangChain Embeddings 互換メソッド: 単一の検索クエリを埋め込みベクトルに変換します。
        自動的に `query_prefix` (デフォルト: "検索クエリ: ") を付与します。
        """
        prefixed_text = f"{self.query_prefix}{text}" if self.query_prefix and not text.startswith(self.query_prefix) else text
        embedding = self.encode([prefixed_text])[0]
        return embedding.tolist()

    def get_text_embedding(self, text: str) -> List[float]:
        """LlamaIndex BaseEmbedding 互換メソッド (ドキュメント単体用)"""
        return self.embed_documents([text])[0]

    def get_query_embedding(self, query: str) -> List[float]:
        """LlamaIndex BaseEmbedding 互換メソッド (クエリ用)"""
        return self.embed_query(query)

    def get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """LlamaIndex BaseEmbedding 互換メソッド (ドキュメント複数用)"""
        return self.embed_documents(texts)
