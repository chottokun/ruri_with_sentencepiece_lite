"""推論精度検証スクリプト (マルチモデル対応版)
元の PyTorch (sentence-transformers / transformers) の公式出力と、
本実装 (sentencepiece_lite + onnxruntime + numpy) の埋め込みベクトルを直接比較し、
コサイン類似度が 0.9999 以上 (数学的等価性) で一致することを検証する。

使用例:
  uv run python scripts/test_inference.py --model 70m
  uv run python scripts/test_inference.py --model 130m
  uv run python scripts/test_inference.py --model 310m
"""
import os
import sys
import argparse
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "./dist_assets")
from ruri_v3_lite import RuriV3Lite

SUPPORTED_MODELS = {
    "30m": "cl-nagoya/ruri-v3-30m",
    "70m": "cl-nagoya/ruri-v3-70m",
    "130m": "cl-nagoya/ruri-v3-130m",
    "310m": "cl-nagoya/ruri-v3-310m",
}

test_texts = [
    "検索クエリ: 日本の首都はどこですか？",
    "文章: 日本の首都は東京都です。",
    "文章: 大阪は関西地方の主要都市です。",
    "短いテスト",
    "SentencePiece Lite Safe Boundary Pre-tokenization による高速日本語埋め込み環境の実装テストです。"
]

def verify_model(model_key: str):
    model_id = SUPPORTED_MODELS[model_key]
    print(f"\n==================================================")
    print(f"  推論精度検証: {model_id} ({model_key})")
    print(f"==================================================")

    if model_key == "30m":
        model_dir = "./dist_assets"
    else:
        model_dir = f"./dist_assets/ruri_v3_{model_key}"

    print("=== [1/3] リファレンス埋め込みベクトルの生成 (PyTorch + Transformers) ===")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    with torch.no_grad():
        inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        # 公式 Mean Pooling + L2 正規化
        sum_embeddings = torch.sum(last_hidden_state * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        ref_embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1).cpu().numpy()

    print("リファレンス計算完了。shape:", ref_embeddings.shape)

    print("\n=== [2/3] 本実装推論 (sentencepiece_lite + onnxruntime + numpy) ===")
    lite_model = RuriV3Lite(model_dir=model_dir)
    lite_embeddings = lite_model.encode(test_texts)

    print("本実装推論完了。shape:", lite_embeddings.shape)

    print("\n=== [3/3] コサイン類似度による精度検証 ===")
    all_passed = True
    for i, text in enumerate(test_texts):
        ref_vec = ref_embeddings[i]
        lite_vec = lite_embeddings[i]
        cos_sim = float(np.dot(ref_vec, lite_vec))

        passed = cos_sim >= 0.9999
        status = "PASS ✅" if passed else "FAIL ❌"
        if not passed:
            all_passed = False
        print(f"[{status}] テキスト[{i}]: cos_sim = {cos_sim:.8f} | \"{text[:30]}...\"")

    print("\n" + ("=" * 50))
    if all_passed:
        print(f"【検証成功】{model_id} PyTorch公式出力と本実装出力の完全一致 (cos_sim >= 0.9999) を確認しました！ ✅")
        return True
    else:
        print(f"【検証失敗】{model_id} 一部のベクトルに乖離があります。")
        return False

def main():
    parser = argparse.ArgumentParser(description="推論精度検証スクリプト")
    parser.add_argument("--model", choices=["30m", "70m", "130m", "310m"], default="70m", help="検証対象モデル")
    args = parser.parse_args()

    success = verify_model(args.model)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
