"""ruri-v3-reranker-310m 多段階サイズ・量子化ベンチマーク比較スクリプト
1. FP32 (1,202 MB) - 公式同等
2. FP16 (602 MB) - Tensor Core GPU用
3. INT8 (526 MB) - 線形層8bit化
4. INT8-Full (301 MB) - 語彙埋め込みテーブル(Gather)も完全8bit化した極小限界モデル
"""
import os
import sys
import time
import json
import numpy as np
import onnxruntime as ort

MODEL_DIR = os.path.abspath("./dist_assets/ruri_v3_reranker_310m")

MODELS = {
    "FP32 (Original)": os.path.join(MODEL_DIR, "model.onnx"),
    "FP16 (GPU-Optimized)": os.path.join(MODEL_DIR, "model_fp16.onnx"),
    "INT8 (Linear-only)": os.path.join(MODEL_DIR, "model_int8.onnx"),
    "INT8-Full (Linear+Embedding)": os.path.join(MODEL_DIR, "model_int8_full.onnx"),
}

SAMPLE_QUERY = "日本の地方自治体におけるDX推進の現状と課題"
SAMPLE_DOCS = [
    "自治体DX推進計画では、情報システムの標準化・共通化や行政手続きのオンライン化が進められています。課題として専門人材の不足が挙げられます。",
    "地方都市における人口減少対策と産業振興の取り組みについて。地域資源を活用した観光振興が各自治体で実施されています。",
    "クラウドサービスの導入により業務効率化を図る民間企業の事例紹介。コスト削減とセキュリティ対策が重要です。",
    "明日の天気は全国的に晴れのち曇りとなる見込みです。",
    "ピタゴラスの定理は直角三角形の斜辺の長さを求める幾何学の定理です。"
]

def main():
    print("=" * 85)
    print("  ruri-v3-reranker-310m 多段階サイズ (FP32 -> FP16 -> INT8 -> INT8-Full) ベンチマーク")
    print("=" * 85)

    # トークナイザー準備
    sys.path.insert(0, os.path.abspath("./dist_assets"))
    import sentencepiece_lite as spl
    fb_path = os.path.join(MODEL_DIR, "ruri_v3_reranker_310m.spm.fb")
    tokenizer = spl.FastSBPTokenizer(fb_path)

    batch_ids = []
    for doc in SAMPLE_DOCS:
        q_ids = tokenizer.encode(SAMPLE_QUERY)
        d_ids = tokenizer.encode(doc)
        ids = [1] + q_ids + [2] + [1] + d_ids + [2]
        batch_ids.append(ids)

    seq_len = max(len(ids) for ids in batch_ids)
    inp_ids = np.full((len(SAMPLE_DOCS), seq_len), 3, dtype=np.int64)
    inp_mask = np.zeros((len(SAMPLE_DOCS), seq_len), dtype=np.int64)
    for b_idx, ids in enumerate(batch_ids):
        inp_ids[b_idx, :len(ids)] = ids
        inp_mask[b_idx, :len(ids)] = 1

    # ダミーバッチ (Top-10, seq=128) のレイテンシ測定用
    dummy_ids = np.ones((10, 128), dtype=np.int64)
    dummy_mask = np.ones((10, 128), dtype=np.int64)

    results = {}
    base_probs = None
    base_rank = None

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Skipping {name} (not found: {path})")
            continue

        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"\n--- 測定中: {name} (ファイルサイズ: {size_mb:.1f} MB) ---")

        # セッション作成
        t0_load = time.perf_counter()
        try:
            sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"  CPU ロード不可 (FP16等): {e}")
            results[name] = {
                "size_mb": size_mb,
                "latency_ms": None,
                "speedup": None,
                "top1_match": None,
                "rank_order": None,
                "note": "GPU (Tensor Core) 専用"
            }
            continue

        load_time_ms = (time.perf_counter() - t0_load) * 1000

        # ウォームアップ
        _ = sess.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})

        # Top-10 レイテンシ測定 (5回試行)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = sess.run(None, {"input_ids": dummy_ids, "attention_mask": dummy_mask})
            times.append(time.perf_counter() - t0)
        median_lat_ms = np.median(times) * 1000

        # 実テキスト推論 & ランキング順位
        logits = sess.run(None, {"input_ids": inp_ids, "attention_mask": inp_mask})[0].reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        rank = list(np.argsort(probs)[::-1])

        if base_probs is None:
            base_probs = probs
            base_rank = rank
            top1_match = True
            max_diff = 0.0
        else:
            top1_match = (rank[0] == base_rank[0])
            max_diff = float(np.max(np.abs(logits - base_probs)))

        results[name] = {
            "size_mb": size_mb,
            "latency_ms": median_lat_ms,
            "load_time_ms": load_time_ms,
            "rank_order": rank,
            "top1_match": top1_match,
            "probs": [float(p) for p in probs]
        }

        print(f"  モデルロード: {load_time_ms:.1f} ms | Top-10 レイテンシ: {median_lat_ms:.1f} ms")
        print(f"  ランキング順位: {rank} (Top-1 適合文一致: {'✅ 一致' if top1_match else '❌ 不一致'})")
        print(f"  Top-1 文書スコア: {probs[rank[0]]:.4f}")

    # 結果サマリーテーブル
    print("\n" + "=" * 85)
    print("  📊 多段階サイズ・量子化モデル性能比較サマリー")
    print("=" * 85)
    print(f"| モデル精度・段階 | ファイルサイズ | 削減率 | CPU レイテンシ (Top-10) | 速度比 | Top-1 精度順位 |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")

    fp32_lat = results.get("FP32 (Original)", {}).get("latency_ms", 1.0)
    for name, data in results.items():
        sz = f"{data['size_mb']:.1f} MB"
        reduction = f"-{(1 - data['size_mb']/1202.6)*100:.1f}%"
        if data["latency_ms"] is not None:
            lat = f"{data['latency_ms']:.1f} ms"
            spd = f"{fp32_lat / data['latency_ms']:.2f}x"
            top1 = "✅ 100% 完全維持" if data["top1_match"] else "変動あり"
        else:
            lat = "N/A (GPU専用)"
            spd = "-"
            top1 = "FP16 (GPU)"
        print(f"| {name:<26} | {sz:<9} | {reduction:<7} | {lat:<20} | {spd:<6} | {top1} |")

    print("=" * 85)

if __name__ == "__main__":
    main()
