"""ruri-v3-reranker-310m 超軽量・高速リランカー推論ラッパーモジュール
PyTorch / Transformers は完全不要。
依存パッケージ: sentencepiece_lite, onnxruntime / onnxruntime-gpu, numpy, huggingface_hub
"""
import ctypes
import os
import gzip
import shutil
from typing import List, Tuple, Dict, Any, Union, Optional
import numpy as np
import onnxruntime as ort
import sentencepiece_lite as spl
from huggingface_hub import hf_hub_download

# ruri-v3 (ModernBERT) の確定済み特殊トークン ID
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

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """オーバーフロー耐性を持つ Sigmoid 関数"""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    )

class RuriV3RerankerLite:
    """ruri-v3-reranker-310m 超軽量リランカー推論クラス (Zero-Torch)"""

    def __init__(
        self,
        repo_id: str = "Chottokun/ruri-v3-reranker-310m-lite",
        model_dir: Optional[str] = None,
        force_fp32: bool = False,
        precision: str = "auto",  # "auto", "fp32", "fp16", "int8", "int8_full"
        device: str = "auto"  # "auto", "cuda", "cpu"
    ):
        """
        初期化:
        model_dir が指定されている場合はローカルパスから読み込み、
        未指定の場合は Hugging Face Hub (repo_id) からダウンロード・キャッシュします。

        Args:
            repo_id: Hugging Face リポジトリID
            model_dir: ローカルモデル保存ディレクトリ
            force_fp32: True の場合強制的に FP32 (互換性用引数)
            precision: モデル精度・サイズ指定
                - "auto": GPU (CC >= 7.0) では FP16、CPU では INT8-Full (301MB / 202MB Gzip)
                - "fp32": FP32 元モデル (約 1,200 MB, 最高精度)
                - "fp16": FP16 半精度 (約 601 MB, Tensor Core GPU 最適化)
                - "int8": INT8 線形層量子化 (約 526 MB, CPU高速)
                - "int8_full": INT8 線形層+語彙テーブル完全量子化 (約 301 MB / Gzip配信 202MB, 極小・高精度)
                - "pruned_16l": 16層剪定 INT8 モデル (約 220 MB / Gzip配信 178MB, 最速・超軽量)
            device: "auto", "cuda", "cpu"
        """
        available_providers = ort.get_available_providers()
        want_cuda = (device == "cuda") or (device == "auto" and "CUDAExecutionProvider" in available_providers)

        # 1. 精度・モデルファイル名の決定
        if precision == "pruned_16l":
            model_filename = "model_pruned_16l_int8.onnx"
            print("[RuriV3RerankerLite] precision='pruned_16l' 指定 -> 220MB (Gzip 178MB) 16層超軽量モデルをロード")
        elif precision == "int8_full":
            model_filename = "model_int8_full.onnx"
            print("[RuriV3RerankerLite] precision='int8_full' 指定 -> 301MB (Gzip 202MB) 極小量子化モデルをロード")
        elif precision == "int8":
            model_filename = "model_int8.onnx"
            print("[RuriV3RerankerLite] precision='int8' 指定 -> 526MB INT8 モデルをロード")
        elif precision == "fp16":
            model_filename = "model_fp16.onnx"
            print("[RuriV3RerankerLite] precision='fp16' 指定 -> 601MB FP16 モデルをロード")
        elif precision == "fp32" or force_fp32:
            model_filename = "model.onnx"
            print("[RuriV3RerankerLite] precision='fp32' 指定 -> 1,202MB FP32 モデルをロード")
        else:  # auto
            if want_cuda:
                capability = get_cuda_compute_capability()
                if capability >= 7.0:
                    model_filename = "model_fp16.onnx"
                    print(f"[RuriV3RerankerLite] GPU Capability {capability:.1f} >= 7.0 検知 -> FP16 モデルをロード")
                else:
                    model_filename = "model.onnx"
                    print(f"[RuriV3RerankerLite] GPU Capability {capability:.1f} < 7.0 検知 -> FP32 モデルをロード")
            else:
                model_filename = "model_int8_full.onnx"
                print("[RuriV3RerankerLite] CPU 実行 (precision='auto') -> 最適な 301MB (Gzip 202MB) INT8-Full モデルをロード")

        # 2. FlatBuffers トークナイザー辞書ファイル名の決定
        fb_filename = "ruri_v3_reranker_310m.spm.fb"

        if model_dir and os.path.exists(model_dir):
            model_path = os.path.join(model_dir, model_filename)
            # ローカルに .onnx がなく .onnx.gz がある場合は自動展開
            if not os.path.exists(model_path) and os.path.exists(model_path + ".gz"):
                print(f"[RuriV3RerankerLite] ローカルの {model_filename}.gz を自動展開中...")
                with gzip.open(model_path + ".gz", "rb") as f_in, open(model_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

            fb_path = os.path.join(model_dir, fb_filename)
            if not os.path.exists(fb_path):
                # 代替候補探索
                fbs = [f for f in os.listdir(model_dir) if f.endswith(".spm.fb")]
                if fbs:
                    fb_path = os.path.join(model_dir, fbs[0])
                else:
                    raise FileNotFoundError(f"FlatBuffers 辞書 (*.spm.fb) が {model_dir} に見つかりません。")
        else:
            # Hugging Face Hub からのダウンロード
            # まず高速な Gzip 配信版 (*.onnx.gz) の存在を優先確認して透過解凍
            gz_filename = model_filename + ".gz"
            try:
                gz_download_path = hf_hub_download(repo_id=repo_id, filename=gz_filename)
                target_onnx_path = os.path.splitext(gz_download_path)[0]
                if not os.path.exists(target_onnx_path) or os.path.getsize(target_onnx_path) == 0:
                    print(f"[RuriV3RerankerLite] 透過的 Gzip 高速展開中 ({gz_filename} -> {model_filename})...")
                    with gzip.open(gz_download_path, "rb") as f_in, open(target_onnx_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                model_path = target_onnx_path
            except Exception:
                # Gzip 版が存在しない場合は直接 .onnx をダウンロード
                model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)

            try:
                fb_path = hf_hub_download(repo_id=repo_id, filename=fb_filename)
            except Exception:
                # 辞書は 310m と完全バイナリ一致のためフォールバック可能
                fb_path = hf_hub_download(repo_id=repo_id, filename="ruri_v3_310m.spm.fb")

        # 3. SentencePiece Lite トークナイザーの初期化 (FlatBuffers ゼロコピーロード)
        self.tokenizer = spl.FastSBPTokenizer(fb_path)

        # 4. ONNX Runtime セッション初期化
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
                active_provider = self.session.get_providers()[0]
                print(f"[RuriV3RerankerLite] InferenceSession 初期化完了 (プロバイダ: {active_provider})")
            except Exception as e:
                print(f"[RuriV3RerankerLite] CUDA 初期化失敗 ({e}) -> CPUExecutionProvider にフォールバックします")
                self.session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
        else:
            self.session = ort.InferenceSession(model_path, sess_options, providers=["CPUExecutionProvider"])
            print("[RuriV3RerankerLite] InferenceSession 初期化完了 (プロバイダ: CPUExecutionProvider)")

    def _tokenize_pair(self, query: str, document: str, max_length: int) -> List[int]:
        """
        クエリとドキュメントのペアを ModernBERT ペア形式にエンコードします。
        形式: [<s>, q_tokens..., </s>, <s>, doc_tokens..., </s>]
        
        トランケーション方針:
        質問の意味損失を防ぐため、クエリを優先的に保持し、超過分はドキュメントの後方を切り捨てます。
        """
        q_tokens = self.tokenizer.encode(query)
        d_tokens = self.tokenizer.encode(document)

        # 特殊トークン合計は 4 個: <s> q </s> <s> d </s>
        overhead = 4
        available_len = max(0, max_length - overhead)

        if len(q_tokens) + len(d_tokens) > available_len:
            # クエリが長すぎる場合はクエリも切り詰める (最大で available_len の半分まで)
            max_q_len = min(len(q_tokens), available_len // 2)
            if len(q_tokens) > max_q_len:
                q_tokens = q_tokens[:max_q_len]
            max_d_len = available_len - len(q_tokens)
            d_tokens = d_tokens[:max_d_len]

        return [BOS_ID] + q_tokens + [EOS_ID] + [BOS_ID] + d_tokens + [EOS_ID]

    def score(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
        normalize: bool = True,
        max_length: int = 512
    ) -> np.ndarray:
        """
        (query, document) ペアのリストに対してリランキングスコアを算出します。

        Args:
            pairs: (query, document) 文字列タプルのリスト
            batch_size: バッチサイズ
            normalize: True の場合、Sigmoid を適用して 0.0〜1.0 の確率スコアを返します。
                       False の場合、生の logits (未正規化スコア) を返します。
            max_length: 最大シーケンス長 (デフォルト: 512, 最大 8192 まで対応)

        Returns:
            np.ndarray: shape (len(pairs),) の 1次元浮動小数点配列
        """
        if not pairs:
            return np.empty((0,), dtype=np.float32)

        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]

            # 1. CPU 上で sentencepiece_lite による高速ペアトークナイズ
            batch_ids = [
                self._tokenize_pair(q, d, max_length=max_length)
                for q, d in batch_pairs
            ]

            # 2. バッチ内動的パディング (無駄なパディングを排除して計算量削減)
            seq_len = max(len(ids) for ids in batch_ids)
            input_ids = np.full((len(batch_pairs), seq_len), fill_value=PAD_ID, dtype=np.int64)
            attention_mask = np.zeros((len(batch_pairs), seq_len), dtype=np.int64)

            for b_idx, ids in enumerate(batch_ids):
                input_ids[b_idx, :len(ids)] = ids
                attention_mask[b_idx, :len(ids)] = 1

            # 3. ONNX Runtime 推論実行
            outputs = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

            logits = outputs[0].reshape(-1).astype(np.float32)
            if normalize:
                batch_scores = _sigmoid(logits)
            else:
                batch_scores = logits

            all_scores.append(batch_scores)

        return np.concatenate(all_scores, axis=0)

    def rerank(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 32,
        top_k: Optional[int] = None,
        return_documents: bool = True,
        normalize: bool = True,
        max_length: int = 512
    ) -> List[Dict[str, Any]]:
        """
        クエリに対してドキュメント群をスコアリングし、スコア降順でリランキングした結果を返します。

        Args:
            query: 検索クエリ文字列
            documents: 候補ドキュメント文字列のリスト
            batch_size: バッチサイズ
            top_k: 返却する上位件数 (None の場合は全件)
            return_documents: 結果辞書に元のドキュメント文字列を含めるかどうか
            normalize: True の場合 Sigmoid 確率値 (0.0〜1.0)、False の場合 logits
            max_length: 最大シーケンス長

        Returns:
            List[Dict[str, Any]]: 降順ソートされたリランキング結果リスト
            例:
            [
                {"index": 2, "score": 0.9852, "document": "..."},
                {"index": 0, "score": 0.4210, "document": "..."},
                ...
            ]
        """
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self.score(pairs, batch_size=batch_size, normalize=normalize, max_length=max_length)

        # スコア降順にインデックスをソート
        ranked_indices = np.argsort(scores)[::-1]

        if top_k is not None and top_k > 0:
            ranked_indices = ranked_indices[:top_k]

        results = []
        for idx in ranked_indices:
            item: Dict[str, Any] = {
                "index": int(idx),
                "score": float(scores[idx]),
            }
            if return_documents:
                item["document"] = documents[idx]
            results.append(item)

        return results

    # =========================================================================
    # LangChain / LlamaIndex 互換インターフェース (BaseDocumentCompressor / Postprocessor)
    # =========================================================================

    def compress_documents(
        self,
        documents: List[Any],
        query: str,
        callbacks: Any = None
    ) -> List[Any]:
        """
        LangChain BaseDocumentCompressor 互換メソッド:
        LangChain の Document オブジェクトのリストをクエリとの関連度スコアでリランキングし、
        `metadata["rerank_score"]` にスコアを設定してソートしたドキュメントリストを返します。

        Args:
            documents: LangChain Document オブジェクト (page_content 属性を持つオブジェクト) のリスト
            query: 検索クエリ文字列
            callbacks: LangChain コールバックハンドラ (互換性用引数)

        Returns:
            List[Any]: スコア降順に並び替えられ metadata に rerank_score が挿入された Document リスト
        """
        if not documents:
            return []

        # Document オブジェクトまたは文字列の抽出
        doc_texts = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in documents
        ]

        reranked = self.rerank(query, doc_texts, return_documents=False)

        compressed_docs = []
        for res in reranked:
            idx = res["index"]
            score = res["score"]
            doc = documents[idx]

            # Document オブジェクトの metadata を更新 (オブジェクトコピーを推奨)
            if hasattr(doc, "page_content"):
                # LangChain Document オブジェクトの場合
                if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                    doc.metadata["rerank_score"] = score
                compressed_docs.append(doc)
            else:
                compressed_docs.append(doc)

        return compressed_docs
