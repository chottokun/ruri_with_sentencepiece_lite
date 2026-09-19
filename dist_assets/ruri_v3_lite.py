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
        force_fp32: bool = False
    ):
        """
        初期化:
        model_dir が指定されている場合はローカルパスから読み込み、
        未指定の場合は Hugging Face Hub (repo_id) からダウンロード・キャッシュします。
        """
        available_providers = ort.get_available_providers()
        use_cuda = "CUDAExecutionProvider" in available_providers

        # 1. ハードウェア世代に応じたモデル判定
        if use_cuda and not force_fp32:
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
        if model_dir and os.path.exists(model_dir):
            model_path = os.path.join(model_dir, model_filename)
            fb_path = os.path.join(model_dir, "ruri_v3_30m.spm.fb")
        else:
            model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
            fb_path = hf_hub_download(repo_id=repo_id, filename="ruri_v3_30m.spm.fb")

        # 3. SentencePiece Lite トークナイザーの初期化 (FlatBuffers, SBP並列対応)
        self.tokenizer = spl.FastSBPTokenizer(fb_path)

        # 4. ONNX Runtime セッション初期化
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
