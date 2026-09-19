"""End-to-End Embedding 推論ベンチマークスクリプト
テキスト前処理（トークナイズ）からモデル推論、Pooling、L2正規化までの一連の
埋め込みベクトル生成処理全体 (encode) のパフォーマンスを比較測定する。
"""
import sys
import time
import os
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

SAMPLE_SENTENCES = [
    "検索クエリ: 日本の首都はどこですか？",
    "文章: 日本の首都は東京都です。政治、経済、文化の中心地として機能しています。",
    "文章: 大阪は関西地方の主要都市であり、歴史的な商業都市として知られています。",
    "検索クエリ: 自然言語処理における埋め込みベクトルの高速化手法",
    "文章: 密ベクトル検索によりキーワード一致だけでなく文脈の意味類似性を考慮した検索が可能です。",
    "文章: GoogleのSentencePiece LiteによりCPUトークナイズのオーバーヘッドを劇的に削減します。",
    "文章: ONNX Runtimeを用いることでPyTorchに依存せず軽量かつ高速な本番デプロイが実現します。",
    "短いテストテキストです。"
]

def benchmark_model(model_key: str, num_samples: int = 100, batch_size: int = 32):
    hf_id, local_dir = MODELS[model_key]
    texts = [SAMPLE_SENTENCES[i % len(SAMPLE_SENTENCES)] for i in range(num_samples)]

    print(f"\n==================================================================")
    print(f"  モデル: {hf_id} ({model_key}) | 件数: {num_samples}件 | バッチ: {batch_size}")
    print(f"==================================================================")

    # 1. PyTorch (Transformers) CPU
    tok = AutoTokenizer.from_pretrained(hf_id)
    pt_model = AutoModel.from_pretrained(hf_id)
    pt_model.eval()

    # Warmup
    inp = tok(texts[:batch_size], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        _ = pt_model(**inp)

    pt_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tok(batch, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                out = pt_model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1)
                sum_emb = torch.sum(out.last_hidden_state * mask, dim=1)
                sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                mean_p = sum_emb / sum_mask
                _ = torch.nn.functional.normalize(mean_p, p=2, dim=1).numpy()
        pt_times.append(time.perf_counter() - t0)

    pt_median = np.median(pt_times)
    pt_qps = num_samples / pt_median

    # 2. RuriV3Lite CPU
    lite_model = RuriV3Lite(model_dir=local_dir, force_fp32=True, device="cpu")
    # Warmup
    _ = lite_model.encode(texts[:batch_size], batch_size=batch_size)

    lite_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = lite_model.encode(texts, batch_size=batch_size)
        lite_times.append(time.perf_counter() - t0)

    lite_median = np.median(lite_times)
    lite_qps = num_samples / lite_median

    print(f"  PyTorch 2.14 CPU : {pt_median * 1000:7.2f} ms ({pt_qps:6.1f} sent/s, 1件: {pt_median*1000/num_samples:5.2f} ms)")
    print(f"  RuriV3Lite CPU   : {lite_median * 1000:7.2f} ms ({lite_qps:6.1f} sent/s, 1件: {lite_median*1000/num_samples:5.2f} ms)")

    return {
        "pt_ms": pt_median * 1000,
        "pt_qps": pt_qps,
        "lite_ms": lite_median * 1000,
        "lite_qps": lite_qps,
    }

def main():
    print("🚀 End-to-End Embedding ベンチマーク開始")
    all_results = {}
    for k in ["30m", "70m", "130m", "310m"]:
        all_results[k] = benchmark_model(k, num_samples=100, batch_size=32)

    print("\n\n" + "=" * 78)
    print(f"{'モデル':<8} | {'PyTorch CPU (ms)':>16} | {'RuriV3Lite CPU (ms)':>18} | {'スループット比':>14}")
    print("-" * 78)
    for k, res in all_results.items():
        ratio = res["lite_qps"] / res["pt_qps"]
        print(f"{k:<8} | {res['pt_ms']:>13.1f} ms | {res['lite_ms']:>15.1f} ms | {res['lite_qps']:>6.1f} sent/s ({ratio:.2f}x)")
    print("=" * 78)

if __name__ == "__main__":
    main()
