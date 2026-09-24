"""Hugging Face 形式トークナイザー (tokenizer.json) エクスポートスクリプト

SentencePiece バイナリ (tokenizer.model) だけでなく、Transformers / tokenizers / Transformers.js 互換の
Fast Tokenizer 形式 (tokenizer.json, tokenizer_config.json, special_tokens_map.json)
を生成し、指定ディレクトリおよび dist_assets 配下に配備します。

【重要】
Chottokun/ruri-v3-*-lite には現状 tokenizer.model しかアップロードされていないため、
特殊トークンルール (BOS=<s>, EOS=</s> の自動付与 post-processor) を正しく含めるために、
デフォルトでは公式ベースモデル (cl-nagoya/ruri-v3-30m) からトークナイザー設定を読み込みます。

使用例:
  # ruri_tokenizer_out/ および dist_assets 配下に一括生成
  uv run --with "transformers" --with "sentencepiece" python scripts/export_tokenizer_json.py

  # 特定のディレクトリのみに出力する場合 (例: Jules / 拡張機能用)
  uv run --with "transformers" --with "sentencepiece" python scripts/export_tokenizer_json.py --output-dir ./ruri_tokenizer_out
"""

import os
import sys
import argparse
import shutil
from transformers import AutoTokenizer

DEFAULT_BASE_MODEL = "cl-nagoya/ruri-v3-30m"

DIST_DIRS = [
    "./dist_assets",
    "./dist_assets/ruri_v3_70m",
    "./dist_assets/ruri_v3_130m",
    "./dist_assets/ruri_v3_310m",
    "./dist_assets/ruri_v3_reranker_310m",
]

def load_tokenizer(model_id: str):
    """ベースモデルから Fast Tokenizer を読み込む"""
    print(f"[{model_id}] Tokenizer を読み込み中...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if not tokenizer.is_fast:
        print("警告: 読み込まれたトークナイザーは Fast Tokenizer ではありません。")
    else:
        print(f"Fast Tokenizer のロードに成功しました: {type(tokenizer).__name__}")
        print(f"  bos_token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
        print(f"  eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    return tokenizer

def save_and_verify(tokenizer, target_dir: str):
    """指定ディレクトリに tokenizer 資産を書き出し、整合性を検証する"""
    os.makedirs(target_dir, exist_ok=True)
    tokenizer.save_pretrained(target_dir)

    json_path = os.path.join(target_dir, "tokenizer.json")
    config_path = os.path.join(target_dir, "tokenizer_config.json")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"tokenizer.json が生成されませんでした: {json_path}")

    json_size = os.path.getsize(json_path)
    print(f"  -> 保存完了: {json_path} ({json_size:,} bytes)")

    # 生成された tokenizer.json を使って再ロード検証
    reloaded_tok = AutoTokenizer.from_pretrained(target_dir)
    test_text = "検索クエリ: 日本の首都はどこですか？"
    encoded = reloaded_tok(test_text)
    
    # 基本検証: input_ids が空でなく、BOS/EOS が付与されていること
    input_ids = encoded["input_ids"]
    assert len(input_ids) > 0, "トークナイズ結果が空です。"
    assert input_ids[0] == tokenizer.bos_token_id, f"先頭トークンが BOS ではありません (got {input_ids[0]}, expected {tokenizer.bos_token_id})"
    assert input_ids[-1] == tokenizer.eos_token_id, f"末尾トークンが EOS ではありません (got {input_ids[-1]}, expected {tokenizer.eos_token_id})"

    print(f"  -> 検証OK: BOS/EOS付与確認済み (tokens: {len(input_ids)})")
    return json_path

def main():
    parser = argparse.ArgumentParser(description="tokenizer.json エクスポートスクリプト")
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"ベースとなるモデルのリポジトリID (デフォルト: {DEFAULT_BASE_MODEL})"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./ruri_tokenizer_out",
        help="出力先ディレクトリ (デフォルト: ./ruri_tokenizer_out)"
    )
    parser.add_argument(
        "--no-dist-assets",
        action="store_true",
        help="dist_assets 配下の各モデルディレクトリへのコピーをスキップする"
    )
    args = parser.parse_args()

    print("==================================================")
    print("  tokenizer.json (Fast Tokenizer) エクスポート開始")
    print("==================================================")

    tokenizer = load_tokenizer(args.model_id)

    # 1. ユーザー指定の出力先へ保存
    print(f"\n[1] 指定出力先への保存: {args.output_dir}")
    save_and_verify(tokenizer, args.output_dir)

    # 2. dist_assets 配下への保存
    if not args.no_dist_assets:
        print("\n[2] dist_assets 配下への一括配備")
        for d in DIST_DIRS:
            target_path = os.path.abspath(d)
            if os.path.exists(target_path):
                print(f"配備中: {target_path}")
                save_and_verify(tokenizer, target_path)

    print("\n==================================================")
    print("✨ すべての tokenizer.json のエクスポートおよび検証が完了しました！")
    print("==================================================")

if __name__ == "__main__":
    main()
