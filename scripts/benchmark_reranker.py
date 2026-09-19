"""ruri-v3-reranker-310m 包括的ベンチマーク測定スクリプト (最適化版)
1. RAG Top-N リランキング速度 (N=5, 10, 20, 50, 100)
2. テキスト長別レイテンシ (短文 60文字, 中文 150文字, 長文 750文字)
3. CPU スレッド数別スケーリング (intra_op threads = 1, 2, 4, 8)
4. プロセス分離による最大物理メモリ消費量 (Peak RSS) の厳密測定
5. PyTorch (Transformers) vs Zero-Torch (RuriV3RerankerLite) の対比
"""
import sys
import os
import time
import json
import subprocess
import resource
import numpy as np

SAMPLE_QUERY = "日本の地方自治体におけるDX推進の現状と課題"
SHORT_DOC = "自治体DXでは手続きオンライン化と業務効率化が重要課題です。" * 2 # 約60文字
MEDIUM_DOC = "自治体DX推進計画では、情報システムの標準化・共通化や行政手続きのオンライン化が進められています。現場の課題として専門人材の不足や既存レガシーシステムの移行コストが挙げられます。住民サービスの利便性向上に向けた実効的な取り組みが求められます。" # 約150文字
LONG_DOC = MEDIUM_DOC * 5 # 約750文字

# ワーカー実行モード用の処理
if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    mode = sys.argv[2]       # "full" or "thread_only"
    framework = sys.argv[3]  # "pytorch" or "lite"
    threads = int(sys.argv[4])
    
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)

    MODEL_ID = "cl-nagoya/ruri-v3-reranker-310m"
    MODEL_DIR = os.path.abspath("./dist_assets/ruri_v3_reranker_310m")

    t_start_load = time.perf_counter()
    if framework == "pytorch":
        import torch
        torch.set_num_threads(threads)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    else:
        import onnxruntime as ort
        sys.path.insert(0, os.path.abspath("./dist_assets"))
        from ruri_v3_reranker_lite import RuriV3RerankerLite
        lite = RuriV3RerankerLite(model_dir=MODEL_DIR, device="cpu")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        lite.session = ort.InferenceSession(
            os.path.join(MODEL_DIR, "model.onnx"),
            opts,
            providers=["CPUExecutionProvider"]
        )

    load_time_ms = (time.perf_counter() - t_start_load) * 1000

    # ウォームアップ
    if framework == "pytorch":
        w_inp = tokenizer([[SAMPLE_QUERY, "ウォームアップ文書"]], return_tensors="pt")
        with torch.no_grad():
            _ = model(**w_inp)
    else:
        _ = lite.rerank(SAMPLE_QUERY, ["ウォームアップ文書"])

    if mode == "thread_only":
        # スレッド数スケーリング測定時は N=20 のみ測定 (3試行)
        docs = [f"文書{i}: {MEDIUM_DOC}" for i in range(20)]
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            if framework == "pytorch":
                inputs = tokenizer([[SAMPLE_QUERY, d] for d in docs], padding=True, truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    out = model(**inputs)
                    scores = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy()
                    ranked = np.argsort(scores)[::-1]
            else:
                res = lite.rerank(SAMPLE_QUERY, docs, batch_size=32, normalize=True, max_length=512)
            times.append(time.perf_counter() - t0)
        output = {"e2e_ms": float(np.median(times) * 1000)}
        print("BENCHMARK_RESULT_JSON:" + json.dumps(output))
        sys.exit(0)

    # 1. RAG Top-N ベンチマーク (3試行中央値)
    top_n_results = {}
    for n in [5, 10, 20, 50, 100]:
        docs = [f"文書{i}: {MEDIUM_DOC}" for i in range(n)]
        times = []
        tok_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            if framework == "pytorch":
                t_tok0 = time.perf_counter()
                inputs = tokenizer([[SAMPLE_QUERY, d] for d in docs], padding=True, truncation=True, max_length=512, return_tensors="pt")
                tok_times.append(time.perf_counter() - t_tok0)
                with torch.no_grad():
                    out = model(**inputs)
                    scores = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy()
                    ranked = np.argsort(scores)[::-1]
            else:
                t_tok0 = time.perf_counter()
                _ = [lite._tokenize_pair(SAMPLE_QUERY, d, 512) for d in docs]
                tok_times.append(time.perf_counter() - t_tok0)
                res = lite.rerank(SAMPLE_QUERY, docs, batch_size=32, normalize=True, max_length=512)
            times.append(time.perf_counter() - t0)

        top_n_results[str(n)] = {
            "tok_ms": float(np.median(tok_times) * 1000),
            "e2e_ms": float(np.median(times) * 1000),
            "throughput": float(n / np.median(times))
        }

    # 2. テキスト長別ベンチマーク (N=10, 3試行)
    len_results = {}
    for label, doc_text in [("short (~60字)", SHORT_DOC), ("medium (~150字)", MEDIUM_DOC), ("long (~750字)", LONG_DOC)]:
        docs = [f"文書{i}: {doc_text}" for i in range(10)]
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            if framework == "pytorch":
                inputs = tokenizer([[SAMPLE_QUERY, d] for d in docs], padding=True, truncation=True, max_length=512, return_tensors="pt")
                with torch.no_grad():
                    out = model(**inputs)
                    scores = torch.sigmoid(out.logits.squeeze(-1)).cpu().numpy()
                    ranked = np.argsort(scores)[::-1]
            else:
                res = lite.rerank(SAMPLE_QUERY, docs, batch_size=32, normalize=True, max_length=512)
            times.append(time.perf_counter() - t0)
        len_results[label] = float(np.median(times) * 1000)

    # 最大メモリ (RSS)
    peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)

    output = {
        "load_time_ms": load_time_ms,
        "peak_rss_mb": peak_rss_mb,
        "top_n": top_n_results,
        "lengths": len_results
    }
    print("BENCHMARK_RESULT_JSON:" + json.dumps(output))
    sys.exit(0)

# 親プロセスマネージャー
def run_worker(mode: str, framework: str, threads: int = 4):
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        mode,
        framework,
        str(threads)
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in res.stdout.split("\n"):
        if line.startswith("BENCHMARK_RESULT_JSON:"):
            return json.loads(line.replace("BENCHMARK_RESULT_JSON:", "").strip())
    print("Worker stderr:", res.stderr)
    raise RuntimeError(f"Worker failed for {framework} threads={threads}")

def main():
    print("=" * 80)
    print("  ruri-v3-reranker-310m 多角的包括的ベンチマーク測定")
    print("=" * 80)

    # 1. 基準比較 (4 スレッド, Full)
    print("\n>>> 測定中: PyTorch (Transformers) [Threads=4]...")
    pt_data = run_worker("full", "pytorch", threads=4)
    print(f">>> 測定完了: PyTorch (Load={pt_data['load_time_ms']:.1f}ms, RSS={pt_data['peak_rss_mb']:.1f}MB)")

    print("\n>>> 測定中: Zero-Torch (RuriV3RerankerLite) [Threads=4]...")
    lite_data = run_worker("full", "lite", threads=4)
    print(f">>> 測定完了: Zero-Torch (Load={lite_data['load_time_ms']:.1f}ms, RSS={lite_data['peak_rss_mb']:.1f}MB)")

    # 2. スレッド数別スケーリング (N=20 ドキュメント, thread_only)
    print("\n>>> 測定中: CPU スレッド数別スケーリング測定 (N=20)...")
    thread_scaling = {"pytorch": {}, "lite": {}}
    for th in [1, 2, 4, 8]:
        p = run_worker("thread_only", "pytorch", threads=th)
        l = run_worker("thread_only", "lite", threads=th)
        thread_scaling["pytorch"][th] = p["e2e_ms"]
        thread_scaling["lite"][th] = l["e2e_ms"]
        print(f"  Threads={th:2d}: PyTorch={thread_scaling['pytorch'][th]:6.1f} ms vs Lite={thread_scaling['lite'][th]:6.1f} ms (高速化: {thread_scaling['pytorch'][th] / thread_scaling['lite'][th]:.2f}x)")

    # 結果サマリー出力
    print("\n" + "=" * 80)
    print("  📊 ベンチマーク 1: 基本性能・リソース消費 (Threads=4)")
    print("=" * 80)
    print(f"| 評価指標 | PyTorch (Transformers) | Zero-Torch (RuriV3RerankerLite) | 改善率 / 優位性 |")
    print(f"| :--- | :--- | :--- | :--- |")
    print(f"| モデルロード時間 | {pt_data['load_time_ms']:.1f} ms | {lite_data['load_time_ms']:.1f} ms | **{pt_data['load_time_ms'] / lite_data['load_time_ms']:.2f}x 高速** |")
    print(f"| ピークメモリ (RSS) | {pt_data['peak_rss_mb']:.1f} MB | {lite_data['peak_rss_mb']:.1f} MB | **{pt_data['peak_rss_mb'] - lite_data['peak_rss_mb']:.1f} MB 削減 ({lite_data['peak_rss_mb'] / pt_data['peak_rss_mb'] * 100:.1f}%)** |")

    print("\n" + "=" * 80)
    print("  📊 ベンチマーク 2: RAG Top-N リランキングレイテンシ & スループット (Threads=4)")
    print("=" * 80)
    print(f"| 件数 (N) | PyTorch E2E (ms) | Lite E2E (ms) | 高速化倍率 | Lite スループット (docs/s) | トークナイズ高速化 |")
    print(f"| :--- | :--- | :--- | :--- | :--- | :--- |")
    for n_str in ["5", "10", "20", "50", "100"]:
        pt_n = pt_data["top_n"][n_str]
        lt_n = lite_data["top_n"][n_str]
        e2e_speedup = pt_n["e2e_ms"] / lt_n["e2e_ms"]
        tok_speedup = pt_n["tok_ms"] / lt_n["tok_ms"]
        print(f"| Top-{int(n_str):<3d} | {pt_n['e2e_ms']:7.1f} ms | {lt_n['e2e_ms']:7.1f} ms | **{e2e_speedup:5.2f}x** | {lt_n['throughput']:6.1f} docs/s | **{tok_speedup:.2f}x** |")

    print("\n" + "=" * 80)
    print("  📊 ベンチマーク 3: テキスト長別レイテンシ (N=10, Threads=4)")
    print("=" * 80)
    print(f"| ドキュメント長 | PyTorch E2E (ms) | Lite E2E (ms) | 高速化倍率 |")
    print(f"| :--- | :--- | :--- | :--- |")
    for k in pt_data["lengths"]:
        pt_len = pt_data["lengths"][k]
        lt_len = lite_data["lengths"][k]
        print(f"| {k:<15} | {pt_len:7.1f} ms | {lt_len:7.1f} ms | **{pt_len / lt_len:.2f}x** |")

    print("\n" + "=" * 80)
    print("  📊 ベンチマーク 4: CPU スレッド数別スケーリング (N=20 ドキュメント)")
    print("=" * 80)
    print(f"| スレッド数 | PyTorch E2E (ms) | Lite E2E (ms) | 高速化倍率 |")
    print(f"| :--- | :--- | :--- | :--- |")
    for th in [1, 2, 4, 8]:
        pt_t = thread_scaling["pytorch"][th]
        lt_t = thread_scaling["lite"][th]
        print(f"| Threads={th:<2d} | {pt_t:7.1f} ms | {lt_t:7.1f} ms | **{pt_t / lt_t:.2f}x** |")
    print("=" * 80)

if __name__ == "__main__":
    main()
