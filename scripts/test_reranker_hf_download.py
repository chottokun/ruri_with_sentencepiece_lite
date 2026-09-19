"""Hugging Face Hub からの precision="int8_full" (301MB) 自動ダウンロード・推論テスト
"""
import sys
import os

sys.path.insert(0, os.path.abspath("./dist_assets"))
from ruri_v3_reranker_lite import RuriV3RerankerLite

def main():
    print("=" * 70)
    print("  Hugging Face Hub precision='int8_full' (301MB) 自動ダウンロード推論テスト")
    print("=" * 70)

    reranker = RuriV3RerankerLite(
        repo_id="Chottokun/ruri-v3-reranker-310m-lite",
        precision="int8_full",
        device="cpu"
    )

    query = "日本の首都はどこですか？"
    documents = [
        "日本の首都は東京都です。",
        "大阪は関西地方の中心です。"
    ]

    results = reranker.rerank(query, documents, top_k=2)
    for r, item in enumerate(results, 1):
        print(f"Rank {r}: Score={item['score']:.4f} (Index {item['index']}) -> {item['document']}")

    assert results[0]["index"] == 0, "東京が1位になっていません"
    print("✅ 301MB 極小モデルの Hub ダウンロード・推論に完全成功しました！")

if __name__ == "__main__":
    main()
