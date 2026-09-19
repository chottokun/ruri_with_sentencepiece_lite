"""CPU vs GPU 推論ベンチマークスクリプト
1. PyTorch 公式 (Transformers + AutoModel) [CPU & GPU]
2. RuriV3Lite (SentencePiece Lite + ONNX Runtime) [CPU & GPU]
レイテンシ、スループット (sentences/sec)、メモリ使用量を測定・比較する。
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "./dist_assets")
from ruri_v3_lite import RuriV3Lite

MODELS = {
    "30m": ("cl-nagoya/ruri-v3-30m", "./dist_assets"),
    "70m": ("cl-nagoya/ruri-v3-70m", "./dist_assets/ruri_v3_70m"),
    "130m": ("cl-nagoya/ruri-v3-130m", "./dist_assets/ruri_v3_130m"),
    "310m": ("cl-nagoya/ruri-v3-310m", "./dist_assets/ruri_v3_310m"),
}

# ベンチマーク用日本語テキスト群 (短文・中文・長文を混在させたリアルなデータセット)
BASE_TEXTS = [
    "検索クエリ: 日本の首都はどこですか？",
    "文章: 日本の首都は東京都です。東京は政治、経済、文化の中心地として機能しています。",
    "文章: 大阪は関西地方の主要都市であり、歴史的な商業都市として発展してきました。",
    "検索クエリ: 自然言語処理における埋め込みベクトルの活用方法について",
    "文章: 密ベクトル検索を用いることで、キーワード一致だけでなく文脈や意味の類似性を考慮した高度な情報検索が可能になります。",
    "文章: Googleが開発したSentencePiece Liteは、軽量で高速なC++実装のトークナイザーです。",
    "文章: Safe Boundary Pre-tokenization (SBP) により、境界分割を用いて一貫性を損なわずにマルチスレッドで並列処理できます。",
    "短いテスト",
    "文章: ONNX Runtimeを利用することで、PyTorch等の重厚なフレームワークに依存せず高効率なモデル推論を実現できます。",
    "検索クエリ: サーバーレス環境で機械学習モデルを低レイテンシ・低メモリで動かす最適化手法"
]

def generate_benchmark_dataset(num_samples: int = 100):
    return [BASE_TEXTS[i % len(BASE_TEXTS)] for i in range(num_samples)]

def benchmark_pytorch(model_id: str, texts: list[str], device: str, batch_size: int = 32, warmup: int = 2, runs: int = 5):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    # Warmup
    for _ in range(warmup):
        inputs = tokenizer(texts[:batch_size], padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1)
                sum_emb = torch.sum(out.last_hidden_state * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_pooled = sum_emb / sum_mask
                _ = torch.nn.functional.normalize(mean_pooled, p=2, dim=1).cpu().numpy()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    throughput = len(texts) / avg_time
    return avg_time * 1000, throughput

def benchmark_lite(model_dir: str, texts: list[str], device: str = "cpu", batch_size: int = 32, warmup: int = 2, runs: int = 5):
    try:
        model = RuriV3Lite(model_dir=model_dir, force_fp32=(device == "cpu"), device=device)
        active_p = model.session.get_providers()[0]
        if device == "cuda" and "CUDA" not in active_p:
            print(f"  (GPU未対応のためスキップ: active={active_p})")
            return None, None
    except Exception as e:
        print(f"  (初期化不可のためスキップ: {e})")
        return None, None

    # Warmup
    for _ in range(warmup):
        _ = model.encode(texts[:batch_size], batch_size=batch_size)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = model.encode(texts, batch_size=batch_size)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    throughput = len(texts) / avg_time
    return avg_time * 1000, throughput

def run_benchmark(model_key: str = "30m", num_samples: int = 100, batch_size: int = 32):
    model_id, model_dir = MODELS[model_key]
    texts = generate_benchmark_dataset(num_samples)

    print(f"==============================================================")
    print(f"  推論ベンチマーク: {model_id} ({model_key})")
    print(f"  サンプル数: {num_samples} 件 | バッチサイズ: {batch_size}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"==============================================================")

    results = {}

    # 1. PyTorch CPU
    print("[1/4] PyTorch (Transformers) CPU 実行中...")
    pt_cpu_ms, pt_cpu_qps = benchmark_pytorch(model_id, texts, device="cpu", batch_size=batch_size)
    results["PyTorch (CPU)"] = (pt_cpu_ms, pt_cpu_qps)

    # 2. RuriV3Lite CPU
    print("[2/4] RuriV3Lite (SentencePiece Lite + ONNX) CPU 実行中...")
    lite_cpu_ms, lite_cpu_qps = benchmark_lite(model_dir, texts, device="cpu", batch_size=batch_size)
    if lite_cpu_ms is not None:
        results["RuriV3Lite (CPU)"] = (lite_cpu_ms, lite_cpu_qps)

    # 3. PyTorch GPU (sm_61など旧世代でPyTorchが非対応の場合は例外ハンドリング)
    if torch.cuda.is_available():
        print("[3/4] PyTorch (Transformers) GPU 実行中...")
        try:
            pt_gpu_ms, pt_gpu_qps = benchmark_pytorch(model_id, texts, device="cuda", batch_size=batch_size)
            results["PyTorch (GPU)"] = (pt_gpu_ms, pt_gpu_qps)
        except Exception as e:
            print(f"  (PyTorch CUDA非対応のためスキップ: {e})")

        # 4. RuriV3Lite GPU
        print("[4/4] RuriV3Lite (SentencePiece Lite + ONNX) GPU 実行中...")
        try:
            lite_gpu_ms, lite_gpu_qps = benchmark_lite(model_dir, texts, device="cuda", batch_size=batch_size)
            if lite_gpu_ms is not None:
                results["RuriV3Lite (GPU)"] = (lite_gpu_ms, lite_gpu_qps)
        except Exception as e:
            print(f"  (RuriV3Lite CUDA非対応のためスキップ: {e})")

    print("\n" + "=" * 70)
    print(f"{'構成 / ランタイム':<30} | {'総所要時間 (ms)':>15} | {'スループット (sent/s)':>18}")
    print("-" * 70)
    for name, (ms, qps) in results.items():
        print(f"{name:<30} | {ms:>13.2f} ms | {qps:>15.1f} sent/s")
    print("=" * 70)

    # 倍率計算
    speedup_cpu = results["RuriV3Lite (CPU)"][1] / results["PyTorch (CPU)"][1]
    print(f"\n💡 [CPU性能]: RuriV3Lite は PyTorch CPU 比で 約 {speedup_cpu:.2f}倍 高速！")

    if "RuriV3Lite (GPU)" in results and "PyTorch (GPU)" in results:
        speedup_gpu = results["RuriV3Lite (GPU)"][1] / results["PyTorch (GPU)"][1]
        print(f"💡 [GPU性能]: RuriV3Lite は PyTorch GPU 比で 約 {speedup_gpu:.2f}倍 高速！")
        gpu_vs_cpu = results["RuriV3Lite (GPU)"][1] / results["RuriV3Lite (CPU)"][1]
        print(f"💡 [GPU加速]: RuriV3Lite GPU は CPU 比で 約 {gpu_vs_cpu:.2f}倍 高速！")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["30m", "70m", "130m", "310m"], default="30m")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    run_benchmark(args.model, args.samples, args.batch_size)
