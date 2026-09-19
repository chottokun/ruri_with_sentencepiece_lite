"""ruri-v3-reranker-310m の数学的等価性およびランキング順序検証スクリプト
公式 PyTorch 実装 (AutoModelForSequenceClassification) と
Zero-Torch ラッパー (RuriV3RerankerLite) の出力 logits および ranking を厳密比較します。
"""
import sys
import os
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# dist_assets を sys.path に追加
sys.path.insert(0, os.path.abspath("./dist_assets"))
from ruri_v3_reranker_lite import RuriV3RerankerLite

MODEL_ID = "cl-nagoya/ruri-v3-reranker-310m"
MODEL_DIR = os.path.abspath("./dist_assets/ruri_v3_reranker_310m")

TEST_CASES = [
    {
        "query": "日本の首都はどこですか？",
        "docs": [
            "日本の首都は東京都です。政治・経済の中枢が集約されています。",
            "東京は日本の政治と文化の中心都市であり、多くの観光客が訪れます。",
            "大阪は関西地方の主要都市で、独自の食文化やお笑いで知られています。",
            "明日の天気は全国的に晴れのち曇りとなる見込みです。",
            "ピタゴラスの定理は直角三角形の斜辺の長さを計算するための幾何学の基本法則です。",
        ]
    },
    {
        "query": "機械学習における過学習を防ぐための効果的な手法",
        "docs": [
            "過学習（過適合）を防ぐためには、L2正則化（Weight Decay）、ドロップアウト、データ拡張、アーリーストッピングなどが広く用いられます。",
            "機械学習のモデル学習では、訓練損失の減少だけでなく検証セットでの汎化性能を評価することが極めて重要です。",
            "深層学習（ディープラーニング）はニューラルネットワークを多層に重ねた学習アルゴリズムです。",
            "おいしいスパイスカレーの作り方：玉ねぎを飴色になるまでじっくり炒めるのがコツです。",
            "サッカーワールドカップで日本代表は歴史的な勝利を収めました。",
        ]
    },
    {
        "query": "Pythonでリストを辞書に変換する",
        "docs": [
            "Pythonでは dict() 関数や辞書内包表記 {k: v for k, v in pairs} を使ってキーと値のペアから辞書を生成できます。",
            "zip関数とdictを組み合わせることで、2つのリストから簡単に1つの辞書を作ることができます。",
            "リストの要素数を取得するには len() 関数を使用します。",
            "JavaにおけるHashMapの初期化方法と基本的な操作についての解説記事です。",
            "今日は天気が良いので公園を散歩しました。",
        ]
    }
]

def main():
    print("=" * 70)
    print("  ruri-v3-reranker-310m 数学的等価性・ランキング順位 検証")
    print("=" * 70)

    # 1. 公式 PyTorch モデルのロード
    print("\n[1/3] 公式 PyTorch モデルのロード中...")
    pt_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    pt_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

    # 2. Zero-Torch RuriV3RerankerLite のロード
    print("\n[2/3] Zero-Torch RuriV3RerankerLite のロード中...")
    lite_model = RuriV3RerankerLite(model_dir=MODEL_DIR, device="cpu")

    # 3. テストケースごとの厳密比較
    print("\n[3/3] 各テストケースの推論比較実行...")
    all_passed = True
    max_logit_diff_overall = 0.0

    for c_idx, case in enumerate(TEST_CASES, start=1):
        query = case["query"]
        docs = case["docs"]
        print(f"\n--- [ケース {c_idx}] クエリ: '{query}' ({len(docs)} 件) ---")

        # PyTorch 推論
        pt_pairs = [[query, doc] for doc in docs]
        pt_inputs = pt_tokenizer(
            pt_pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        with torch.no_grad():
            pt_out = pt_model(**pt_inputs)
            pt_logits = pt_out.logits.squeeze(-1).cpu().numpy().astype(np.float32)
            pt_probs = torch.sigmoid(pt_out.logits.squeeze(-1)).cpu().numpy().astype(np.float32)

        # Lite 推論
        lite_pairs = [(query, doc) for doc in docs]
        lite_logits = lite_model.score(lite_pairs, normalize=False, max_length=512)
        lite_probs = lite_model.score(lite_pairs, normalize=True, max_length=512)
        lite_rerank_res = lite_model.rerank(query, docs, normalize=True, max_length=512)

        # 差分検証
        logit_diff = np.abs(pt_logits - lite_logits)
        prob_diff = np.abs(pt_probs - lite_probs)
        max_ld = float(np.max(logit_diff))
        max_pd = float(np.max(prob_diff))
        if max_ld > max_logit_diff_overall:
            max_logit_diff_overall = max_ld

        print(f"  Logits 最大絶対誤差: {max_ld:.6e}")
        print(f"  Probs  最大絶対誤差: {max_pd:.6e}")

        # ランキング順位比較
        pt_ranking = np.argsort(pt_probs)[::-1]
        lite_ranking = [item["index"] for item in lite_rerank_res]

        print(f"  PyTorch 順位: {list(pt_ranking)}")
        print(f"  Lite    順位: {lite_ranking}")

        rank_match = (list(pt_ranking) == lite_ranking)
        tolerance = 1e-4
        val_match = (max_ld < tolerance)

        if rank_match and val_match:
            print(f"  -> 結果: ✅ 合格 (順位一致: {rank_match}, Logits誤差 < {tolerance})")
        else:
            print(f"  -> 結果: ❌ 不合格 (順位一致: {rank_match}, Logits誤差: {max_ld})")
            all_passed = False

        # 詳細スコア表示
        for r_i, item in enumerate(lite_rerank_res):
            orig_idx = item["index"]
            print(f"    Rank {r_i+1}: doc[{orig_idx}] PT_prob={pt_probs[orig_idx]:.4f}, Lite_prob={item['score']:.4f} | {item['document'][:35]}...")

    print("\n" + "=" * 70)
    print(f"総合判定: 全テストケースの最大 Logits 差分 = {max_logit_diff_overall:.6e}")
    if all_passed:
        print("🎉 すべての数学的等価性およびランキング順序検証に【合格】しました！")
    else:
        print("⚠️ 一部の検証で許容値を超過またはランキングの不一致が発生しました。")
        sys.exit(1)
    print("=" * 70)

if __name__ == "__main__":
    main()
