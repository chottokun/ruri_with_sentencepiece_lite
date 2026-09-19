"""推論精度検証スクリプト
元の PyTorch (sentence-transformers / transformers) の公式出力と、
本実装 (sentencepiece_lite + onnxruntime + numpy) の埋め込みベクトルを直接比較し、
コサイン類似度が 0.9999 以上 (数学的等価性) で一致することを検証する。
"""
import sys
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, "./dist_assets")
from ruri_v3_lite import RuriV3Lite

MODEL_ID = "cl-nagoya/ruri-v3-30m"

test_texts = [
    "検索クエリ: 日本の首都はどこですか？",
    "文章: 日本の首都は東京都です。",
    "文章: 大阪は関西地方の主要都市です。",
    "短いテスト",
    "SentencePiece Lite Safe Boundary Pre-tokenization による高速日本語埋め込み環境の実装テストです。"
]

print("=== [1/3] リファレンス埋め込みベクトルの生成 (PyTorch + Transformers) ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
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
lite_model = RuriV3Lite(model_dir="./dist_assets")
lite_embeddings = lite_model.encode(test_texts)

print("本実装推論完了。shape:", lite_embeddings.shape)

print("\n=== [3/3] コサイン類似度による精度検証 ===")
all_passed = True
for i, text in enumerate(test_texts):
    ref_vec = ref_embeddings[i]
    lite_vec = lite_embeddings[i]
    cos_sim = float(np.dot(ref_vec, lite_vec))
    
    # 閾値: 0.9999 以上 (実質的な完全一致)
    passed = cos_sim >= 0.9999
    status = "PASS ✅" if passed else "FAIL ❌"
    if not passed:
        all_passed = False
    print(f"[{status}] テキスト[{i}]: cos_sim = {cos_sim:.8f} | \"{text[:30]}...\"")

print("\n" + ("=" * 50))
if all_passed:
    print("【検証成功】PyTorch公式出力と本実装出力の完全一致 (コサイン類似度 >= 0.9999) を確認しました！ ✅")
else:
    print("【検証失敗】一部のベクトルに乖離があります。")
    sys.exit(1)
