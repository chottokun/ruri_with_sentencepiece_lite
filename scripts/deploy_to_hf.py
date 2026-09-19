"""Hugging Face デプロイスクリプト (マルチモデル対応版)
指定したモデル (30m, 70m, 130m, 310m) または全モデルをデプロイする。

使用例:
  uv run python scripts/deploy_to_hf.py --model 70m
  uv run python scripts/deploy_to_hf.py --model 130m
  uv run python scripts/deploy_to_hf.py --model 310m
"""
import os
import sys
import argparse
from huggingface_hub import HfApi, get_token

BASE_DIST_DIR = os.path.abspath("./dist_assets")

def deploy_model(model_key: str, api: HfApi, username: str):
    repo_id = f"{username}/ruri-v3-{model_key}-lite"

    if model_key == "30m":
        target_dir = BASE_DIST_DIR
    else:
        target_dir = os.path.join(BASE_DIST_DIR, f"ruri_v3_{model_key}")

    if not os.path.exists(target_dir):
        print(f"エラー: デプロイ対象ディレクトリが存在しません: {target_dir}")
        print(f"先に `uv run python scripts/build_and_export.py --model {model_key}` を実行してください。")
        return False

    print(f"\n==================================================")
    print(f"  デプロイ開始: {repo_id}")
    print(f"  送信元ディレクトリ: {target_dir}")
    print(f"==================================================")

    print(f"=== [1/3] リポジトリ存在確認・作成: {repo_id} ===")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"リポジトリ準備完了: https://huggingface.co/{repo_id}")

    print(f"\n=== [2/3] アップロード対象ファイルの確認 ===")
    files_to_upload = []
    for root, dirs, files in os.walk(target_dir):
        # サブディレクトリとしての他モデルディレクトリやキャッシュは除外
        rel_root = os.path.relpath(root, target_dir)
        if rel_root.startswith("ruri_v3_") or "__pycache__" in rel_root:
            continue
        for f in files:
            if f.endswith(".pyc") or f == "model_fixed.onnx":
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, target_dir)
            files_to_upload.append(rel_path)
            print(f"  - {rel_path} ({os.path.getsize(full_path):,} bytes)")

    print(f"\n=== [3/3] 成果物の一括アップロード開始 ===")
    api.upload_folder(
        folder_path=target_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"feat: upload sentencepiece_lite zero-torch ruri-v3-{model_key} assets",
        ignore_patterns=["ruri_v3_*/**", "__pycache__/**", "*.pyc", "model_fixed.onnx"]
    )
    print(f"\n【デプロイ完了】{repo_id} のアップロードが完了しました！ 🎉")
    print(f"URL: https://huggingface.co/{repo_id}")
    return True

def load_env_file():
    """ローカルの .env ファイルが存在する場合に環境変数を読み込む (サードパーティ非依存)"""
    env_path = os.path.abspath("./.env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("'\"")
                    if k and v and k not in os.environ:
                        os.environ[k] = v

def main():
    load_env_file()

    parser = argparse.ArgumentParser(description="Hugging Face デプロイスクリプト")
    parser.add_argument("--model", choices=["30m", "70m", "130m", "310m"], default="70m", help="デプロイ対象モデル")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN") or get_token()
    if not hf_token:
        print("エラー: 有効な Hugging Face トークンが見つかりません。")
        print("設定方法:")
        print("  1. .env ファイルに `HF_TOKEN=hf_xxx` を記述")
        print("  2. または環境変数 `export HF_TOKEN=hf_xxx` を設定")
        sys.exit(1)

    api = HfApi(token=hf_token)
    
    # ユーザー名: HF_USERNAME 環境変数があれば最優先、なければトークンから自動取得
    username = os.environ.get("HF_USERNAME")
    if not username:
        user_info = api.whoami()
        username = user_info["name"]

    deploy_model(args.model, api, username)

if __name__ == "__main__":
    main()
