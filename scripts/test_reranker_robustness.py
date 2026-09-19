"""ruri-v3-reranker-310m 頑健性・エッジケース自動テストスイート
空入力、超長文トランケーション、Unicode・絵文字、バッチ境界、APIオプションを網羅的に検証します。
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath("./dist_assets"))
from ruri_v3_reranker_lite import RuriV3RerankerLite

MODEL_DIR = os.path.abspath("./dist_assets/ruri_v3_reranker_310m")

def run_tests():
    print("=" * 70)
    print("  ruri-v3-reranker-310m 頑健性・エッジケース検証テスト")
    print("=" * 70)

    lite = RuriV3RerankerLite(model_dir=MODEL_DIR, device="cpu")
    passed = 0
    total = 0

    def assert_test(name: str, condition: bool, msg: str = ""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name}: {msg}")

    # Test 1: 空リストドキュメント
    print("\n--- Test 1: 空リスト / 空ドキュメント入力 ---")
    res_empty_docs = lite.rerank("テスト", [])
    assert_test("空ドキュメントリスト -> 空リスト返却", res_empty_docs == [])
    scores_empty = lite.score([])
    assert_test("空ペアリスト -> shape (0,) 返却", scores_empty.shape == (0,))

    # Test 2: 空文字列の処理
    res_empty_query = lite.rerank("", ["ドキュメント1", "ドキュメント2"])
    assert_test("空クエリでもクラッシュせず2件返却", len(res_empty_query) == 2)

    res_empty_pair = lite.score([("", ""), ("東京", "")])
    assert_test("空文字列ペアのスコアリング成功", len(res_empty_pair) == 2 and not np.isnan(res_empty_pair).any())

    # Test 3: 超長文トランケーション (クエリ2000文字、ドキュメント10000文字)
    print("\n--- Test 2: 超長文・トランケーション境界 ---")
    long_query = "日本の歴史と文化についての詳細な解説。" * 100 # 約2100文字
    long_doc = "縄文時代から現代に至るまでの日本の歴史的変遷を考察する。" * 300 # 約9000文字
    res_long = lite.rerank(long_query, [long_doc, "短い文書"], max_length=128)
    assert_test("max_length=128 での超長文トランケーション成功", len(res_long) == 2)
    assert_test("スコアが NaN や Inf でない", not np.isnan(res_long[0]["score"]) and not np.isinf(res_long[0]["score"]))

    # Test 4: 多様な max_length (16, 64, 512, 1024)
    for ml in [16, 64, 512, 1024]:
        res_ml = lite.rerank("クエリ", ["テスト文書1", "テスト文書2"], max_length=ml)
        assert_test(f"max_length={ml} 正常動作", len(res_ml) == 2)

    # Test 5: 特殊文字・Unicode・絵文字・制御コード
    print("\n--- Test 3: 特殊文字・絵文字・制御コード ---")
    special_query = "🍣 ラーメン & 餃子 @ 渋谷 🗼 <script>alert('xss');</script> \n\t\r"
    special_docs = [
        "美味しいお寿司とラーメンのお店です！ 🍣🍜 #東京グルメ",
        "セキュリティ対策とXSS脆弱性の回避手法について解説します。",
        "半角ｶﾀｶﾅと全角文字：ＡＢＣ１２３！？＆％＄"
    ]
    res_special = lite.rerank(special_query, special_docs)
    assert_test("絵文字・制御文字・HTMLタグ・半角カナの正常処理", len(res_special) == 3)
    assert_test("絵文字クエリで寿司・ラーメン文書が最高スコアを獲得", res_special[0]["index"] == 0)

    # Test 6: バッチサイズの境界値 (1, 2, 5, 7, 100)
    print("\n--- Test 4: バッチサイズの境界値 ---")
    docs_batch = [f"テストドキュメント {i}" for i in range(13)]
    for bs in [1, 2, 5, 7, 13, 32, 100]:
        res_bs = lite.rerank("テスト", docs_batch, batch_size=bs)
        assert_test(f"batch_size={bs:3d} (13件対象) 正常完了", len(res_bs) == 13)

    # Test 7: API オプション (top_k, return_documents, normalize)
    print("\n--- Test 5: API オプション検証 ---")
    # top_k
    assert_test("top_k=3 で上位3件のみ返却", len(lite.rerank("テスト", docs_batch, top_k=3)) == 3)
    assert_test("top_k=100 (実件数超) で全件返却", len(lite.rerank("テスト", docs_batch, top_k=100)) == 13)

    # return_documents
    res_with_doc = lite.rerank("テスト", ["あ", "い"], return_documents=True)
    assert_test("return_documents=True で 'document' キー存在", "document" in res_with_doc[0])
    res_no_doc = lite.rerank("テスト", ["あ", "い"], return_documents=False)
    assert_test("return_documents=False で 'document' キー除外", "document" not in res_no_doc[0])

    # normalize=True (Sigmoid: 0〜1) vs False (logits)
    norm_scores = lite.score([("クエリ", "ドキュメント")], normalize=True)
    raw_scores = lite.score([("クエリ", "ドキュメント")], normalize=False)
    assert_test("normalize=True で 0.0 <= score <= 1.0", 0.0 <= norm_scores[0] <= 1.0)
    # raw logits と sigmoid の整合性
    expected_prob = 1.0 / (1.0 + np.exp(-raw_scores[0]))
    assert_test("raw logits と sigmoid 確率の数学的一致", abs(norm_scores[0] - expected_prob) < 1e-6)

    # Test 8: score() と rerank() のスコア順序完全一致
    pairs = [("日本の首都", d) for d in ["東京", "大阪", "京都", "北海道"]]
    scores_direct = lite.score(pairs, normalize=True)
    rerank_items = lite.rerank("日本の首都", ["東京", "大阪", "京都", "北海道"], normalize=True)
    assert_test("rerank() のスコアが降順ソートされている", all(rerank_items[i]["score"] >= rerank_items[i+1]["score"] for i in range(len(rerank_items)-1)))
    assert_test("score() と rerank() の個別スコア一致", abs(rerank_items[0]["score"] - scores_direct[rerank_items[0]["index"]]) < 1e-6)

    print("\n" + "=" * 70)
    print(f"  結果: {passed} / {total} テスト合格 (合格率: {passed/total*100:.1f}%)")
    if passed == total:
        print("  🎉 すべての頑健性テストに【完全合格】しました！")
    else:
        print("  ❌ 一部のテストが失敗しました。")
        sys.exit(1)
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
