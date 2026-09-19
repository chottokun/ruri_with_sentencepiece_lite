"""ruri-v3-reranker-310m INT8 動的量子化 & 評価スクリプト
1. dist_assets/ruri_v3_reranker_310m/model.onnx の古い value_info をサニタイズ
2. onnxruntime.quantization.quantize_dynamic により INT8 量子化モデル model_int8.onnx を生成
3. ファイルサイズ、推論レイテンシ (CPU)、ランキング順位保持率を厳密測定
"""
import os
import sys
import time
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_DIR = os.path.abspath("./dist_assets/ruri_v3_reranker_310m")
FP32_PATH = os.path.join(MODEL_DIR, "model.onnx")
CLEANED_FP32_PATH = os.path.join(MODEL_DIR, "model_clean.onnx")
INT8_PATH = os.path.join(MODEL_DIR, "model_int8.onnx")

def main():
    print("=" * 70)
    print("  ruri-v3-reranker-310m INT8 動的量子化 (Dynamic Quantization)")
    print("=" * 70)

    if not os.path.exists(FP32_PATH):
        print(f"エラー: {FP32_PATH} が存在しません。")
        sys.exit(1)

    fp32_size_mb = os.path.getsize(FP32_PATH) / (1024 * 1024)
    print(f"元の FP32 モデルサイズ: {fp32_size_mb:.1f} MB ({os.path.getsize(FP32_PATH):,} bytes)")

    # 1. 形状推論競合のサニタイズ (古い value_info の除去)
    if not os.path.exists(INT8_PATH):
        print("\n[1/4] ONNX グラフの形状メタデータをサニタイズ中...")
        m = onnx.load(FP32_PATH)
        m.graph.ClearField("value_info")
        onnx.save(m, CLEANED_FP32_PATH)
        print("  サニタイズ完了")

        # 2. INT8 動的量子化の実行
        print("\n[2/4] INT8 動的量子化を実行中 (MatMul / Gemm 重みの INT8 化)...")
        t0 = time.perf_counter()
        quantize_dynamic(
            model_input=CLEANED_FP32_PATH,
            model_output=INT8_PATH,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"]
        )
        quant_time = time.perf_counter() - t0
        int8_size_mb = os.path.getsize(INT8_PATH) / (1024 * 1024)
        compression_ratio = fp32_size_mb / int8_size_mb

        print(f"  量子化完了 ({quant_time:.2f} 秒)")
        print(f"  INT8 モデルサイズ: {int8_size_mb:.1f} MB ({os.path.getsize(INT8_PATH):,} bytes)")
        print(f"  圧縮倍率: {compression_ratio:.2f}x (サイズ削減率: {(1 - int8_size_mb/fp32_size_mb)*100:.1f}%)")

        if os.path.exists(CLEANED_FP32_PATH):
            os.remove(CLEANED_FP32_PATH)
    else:
        int8_size_mb = os.path.getsize(INT8_PATH) / (1024 * 1024)
        print(f"\n[2/4] 既存の INT8 モデルを使用: {int8_size_mb:.1f} MB")

    # 3. CPU 推論速度比較 (Threads=4)
    print("\n[3/4] CPU 推論速度比較 (Threads=4)...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_fp32 = ort.InferenceSession(FP32_PATH, opts, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(INT8_PATH, opts, providers=["CPUExecutionProvider"])

    # 典型的なバッチ (Top-10, seq=128)
    dummy_ids = np.ones((10, 128), dtype=np.int64)
    dummy_mask = np.ones((10, 128), dtype=np.int64)

    # ウォームアップ
    _ = sess_fp32.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})
    _ = sess_int8.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})

    times_fp32 = []
    for _ in range(5):
        t = time.perf_counter()
        _ = sess_fp32.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})
        times_fp32.append(time.perf_counter() - t)

    times_int8 = []
    for _ in range(5):
        t = time.perf_counter()
        _ = sess_int8.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})
        times_int8.append(time.perf_counter() - t)

    med_fp32_ms = np.median(times_fp32) * 1000
    med_int8_ms = np.median(times_int8) * 1000
    speedup = med_fp32_ms / med_int8_ms

    print(f"  FP32 レイテンシ (Top-10, seq=128): {med_fp32_ms:6.1f} ms")
    print(f"  INT8 レイテンシ (Top-10, seq=128): {med_int8_ms:6.1f} ms")
    print(f"  推論速度比: {speedup:.2f}x ({'⚡ 高速化' if speedup >= 1.05 else '同等'})")

    # 4. 実データでの精度・順位維持率の検証
    print("\n[4/4] 実際の日本語テキストでのスコア差分・順位整合性検証...")
    sys.path.insert(0, os.path.abspath("./dist_assets"))
    import sentencepiece_lite as spl
    fb_path = os.path.join(MODEL_DIR, "ruri_v3_reranker_310m.spm.fb")
    tokenizer = spl.FastSBPTokenizer(fb_path)

    query = "日本の地方自治体におけるDX推進の現状と課題"
    docs = [
        "自治体DX推進計画では、情報システムの標準化・共通化や行政手続きのオンライン化が進められています。課題として専門人材の不足が挙げられます。",
        "地方都市における人口減少対策と産業振興の取り組みについて。地域資源を活用した観光振興が各自治体で実施されています。",
        "クラウドサービスの導入により業務効率化を図る民間企業の事例紹介。コスト削減とセキュリティ対策が重要です。",
        "明日の天気は全国的に晴れのち曇りとなる見込みです。",
        "ピタゴラスの定理は直角三角形の斜辺の長さを求める幾何学の定理です。"
    ]

    batch_ids = []
    for doc in docs:
        q_ids = tokenizer.encode(query)
        d_ids = tokenizer.encode(doc)
        ids = [1] + q_ids + [2] + [1] + d_ids + [2]
        batch_ids.append(ids)

    seq_len = max(len(ids) for ids in batch_ids)
    inp_ids = np.full((len(docs), seq_len), 3, dtype=np.int64)
    inp_mask = np.zeros((len(docs), seq_len), dtype=np.int64)
    for b_idx, ids in enumerate(batch_ids):
        inp_ids[b_idx, :len(ids)] = ids
        inp_mask[b_idx, :len(ids)] = 1

    out_fp32 = sess_fp32.run(None, {"input_ids": inp_ids, "attention_mask": inp_mask})[0].reshape(-1)
    out_int8 = sess_int8.run(None, {"input_ids": inp_ids, "attention_mask": inp_mask})[0].reshape(-1)

    prob_fp32 = 1.0 / (1.0 + np.exp(-out_fp32))
    prob_int8 = 1.0 / (1.0 + np.exp(-out_int8))

    rank_fp32 = list(np.argsort(prob_fp32)[::-1])
    rank_int8 = list(np.argsort(prob_int8)[::-1])

    max_diff = float(np.max(np.abs(out_fp32 - out_int8)))
    print(f"  Logits 最大絶対誤差: {max_diff:.4f}")
    print(f"  FP32 ランキング順位: {rank_fp32}")
    print(f"  INT8 ランキング順位: {rank_int8}")
    print(f"  ランキング順位一致: {'✅ 完全一致 (100%)' if rank_fp32 == rank_int8 else '❌ 順位変動あり'}")

    for r_idx in range(len(docs)):
        idx = rank_int8[r_idx]
        print(f"    Rank {r_idx+1}: doc[{idx}] FP32_prob={prob_fp32[idx]:.4f} -> INT8_prob={prob_int8[idx]:.4f} | {docs[idx][:35]}...")

    print("\n" + "=" * 70)
    print("✨ INT8 量子化モデル検証完了！")
    print("=" * 70)

if __name__ == "__main__":
    main()
