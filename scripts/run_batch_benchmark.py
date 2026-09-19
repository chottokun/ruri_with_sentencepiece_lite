"""丁寧な再テストスクリプト: バッチサイズ別 End-to-End 推論レイテンシ・スループット測定
十分なウォームアップ（各5回）を行い、各バッチサイズで10回の試行の中央値（Median）を測定する。
"""
import sys
import time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "./dist_assets")
from ruri_v3_lite import RuriV3Lite

def main():
    model_id = "cl-nagoya/ruri-v3-30m"
    tok = AutoTokenizer.from_pretrained(model_id)
    pt_model = AutoModel.from_pretrained(model_id).eval()
    lite = RuriV3Lite(model_dir="dist_assets", force_fp32=True, device="cpu")

    base_query = "検索クエリ: 日本の首都はどこですか？"

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = []

    print("\n=========================================================================================")
    print("  徹底検証: バッチサイズ別 End-to-End 性能 (30m / CPU / 5 Warmup + 10 Runs Median)")
    print("=========================================================================================")
    print(f"{'Batch Size':<10} | {'PyTorch 2.14':>14} | {'RuriV3Lite':>14} | {'レイテンシ比':>14} | {'RuriV3Lite スループット':>22}")
    print("-" * 89)

    for bs in batch_sizes:
        texts = [base_query] * bs

        # 1. 丁寧なウォームアップ (各5回)
        for _ in range(5):
            inp = tok(texts, return_tensors="pt", padding=True)
            with torch.no_grad():
                out = pt_model(**inp)
                m = inp["attention_mask"].unsqueeze(-1)
                _ = torch.sum(out.last_hidden_state * m, dim=1)
            _ = lite.encode(texts, batch_size=bs)

        # 2. PyTorch 本計測 (10回)
        pt_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            inp = tok(texts, return_tensors="pt", padding=True)
            with torch.no_grad():
                out = pt_model(**inp)
                m = inp["attention_mask"].unsqueeze(-1)
                emb = torch.sum(out.last_hidden_state * m, dim=1) / torch.clamp(m.sum(dim=1), min=1e-9)
                _ = torch.nn.functional.normalize(emb, p=2, dim=1).numpy()
            pt_times.append(time.perf_counter() - t0)

        # 3. RuriV3Lite 本計測 (10回)
        lite_times = []
        for _ in range(10):
            t0 = time.perf_counter()
            _ = lite.encode(texts, batch_size=bs)
            lite_times.append(time.perf_counter() - t0)

        pt_ms = np.median(pt_times) * 1000
        lite_ms = np.median(lite_times) * 1000
        ratio = pt_ms / lite_ms
        qps = bs / (np.median(lite_times))

        tag = "Lite優位" if ratio >= 1.05 else ("同等" if ratio >= 0.95 else "PT優位")
        print(f"{bs:<10} | {pt_ms:11.2f} ms | {lite_ms:11.2f} ms | {ratio:7.2f}x ({tag:<4}) | {qps:15.1f} sent/s")
        results.append((bs, pt_ms, lite_ms, ratio, tag, qps))

    print("=" * 89)

if __name__ == "__main__":
    main()
