"""Hugging Face Hub (Chottokun/ruri-v3-reranker-310m-lite) からの自動ダウンロード推論テスト
PyTorch / Transformers 一切不要の完全 Zero-Torch 動作を実証します。
"""
import sys
import os

sys.path.insert(0, os.path.abspath("./dist_assets"))
from ruri_v3_reranker_lite import RuriV3RerankerLite

def main():
    print("=" * 70)
    print("  Hugging Face Hub 自動ダウンロード推論テスト")
    print("=" * 70)

    # model_dir を指定せず repo_id から直接取得
    reranker = RuriV3RerankerLite(repo_id="Chottokun/ruri-v3-reranker-310m-lite", device="cpu")

    query = "日本の首都はどこですか？"
    documents = [
        "日本の首都は東京都です。",
        "大阪は関西地方の中心です。"
    ]

    results = reranker.rerank(query, documents, top_k=2)
    for r, item in enumerate(results, 1):
        print(f"Rank {r}: Score={item['score']:.4f} (Index {item['index']}) -> {item['document']}")

    assert results[0]["index"] == 0, "東京が1位になっていません"
    print("✅ Hugging Face Hub からのダウンロード・推論に完全成功しました！")

if __name__ == "__main__":
    main()
