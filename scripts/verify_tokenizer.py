"""特殊トークン検証スクリプト
transformers.AutoTokenizer の出力を検証し、SentencePiece Lite で再現すべき
特殊トークンのパターンを確定する。
"""
from transformers import AutoTokenizer
import sentencepiece as sp_original

MODEL_ID = "cl-nagoya/ruri-v3-30m"

# === 1. transformers の AutoTokenizer で検証 ===
print("=" * 60)
print("1. AutoTokenizer の特殊トークン情報")
print("=" * 60)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"  bos_token: {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
print(f"  eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
print(f"  cls_token: {tokenizer.cls_token!r} (id={tokenizer.cls_token_id})")
print(f"  sep_token: {tokenizer.sep_token!r} (id={tokenizer.sep_token_id})")
print(f"  pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
print(f"  unk_token: {tokenizer.unk_token!r} (id={tokenizer.unk_token_id})")
print(f"  tokenizer_class: {type(tokenizer).__name__}")
print()

# テストケース（ruri-v3 推奨プレフィックス付き）
texts = [
    "検索クエリ: 日本の首都はどこですか？",
    "文章: 日本の首都は東京都です。",
    "短いテスト",
]

print("=" * 60)
print("2. AutoTokenizer の出力")
print("=" * 60)
for text in texts:
    encoded = tokenizer(text, return_tensors="pt")
    ids = encoded["input_ids"][0].tolist()
    mask = encoded["attention_mask"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(f"Text: {text}")
    print(f"  先頭トークン: {tokens[0]} (id={ids[0]})")
    print(f"  末尾トークン: {tokens[-1]} (id={ids[-1]})")
    print(f"  全IDs: {ids}")
    print(f"  全Tokens: {tokens}")
    print(f"  attention_mask: {mask}")
    print(f"  合計長: {len(ids)}")
    print()

# === 2. 生の sentencepiece で検証 ===
print("=" * 60)
print("3. 生の sentencepiece (add_bos/eos なし) の出力")
print("=" * 60)
from huggingface_hub import hf_hub_download
sp_path = hf_hub_download(repo_id=MODEL_ID, filename="tokenizer.model")

sp = sp_original.SentencePieceProcessor()
sp.Load(sp_path)

print(f"  sp.bos_id(): {sp.bos_id()}")
print(f"  sp.eos_id(): {sp.eos_id()}")
print(f"  sp.pad_id(): {sp.pad_id()}")
print(f"  sp.unk_id(): {sp.unk_id()}")
print(f"  vocab_size: {sp.GetPieceSize()}")
print()

for text in texts:
    raw_ids = sp.Encode(text)
    # BOS/EOS 手動付与パターン
    manual_ids = [sp.bos_id()] + raw_ids + [sp.eos_id()]
    print(f"Text: {text}")
    print(f"  raw encode (no special): {raw_ids[:10]}... (len={len(raw_ids)})")
    print(f"  with BOS/EOS: {manual_ids[:10]}... (len={len(manual_ids)})")

    # AutoTokenizer の出力と比較
    auto_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
    match = manual_ids == auto_ids
    print(f"  AutoTokenizer出力と一致: {'✅ YES' if match else '❌ NO'}")
    if not match:
        print(f"    AutoTokenizer: {auto_ids[:10]}... (len={len(auto_ids)})")
        print(f"    Manual:        {manual_ids[:10]}... (len={len(manual_ids)})")
        # 差分を特定
        for j in range(min(len(auto_ids), len(manual_ids))):
            if auto_ids[j] != manual_ids[j]:
                print(f"    最初の差分位置: index={j}, auto={auto_ids[j]}, manual={manual_ids[j]}")
                break
    print()
