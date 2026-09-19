"""Hugging Face デプロイスクリプト
dist_assets/ 以下の全成果物を Hugging Face Hub (chottokun/ruri-v3-30m-lite) にアップロードする。

※ セキュリティ規定:
  HF_TOKEN は環境変数または実行時に入力し、スクリプト内にハードコードしないこと。
"""
import os
import sys
from huggingface_hub import HfApi

REPO_ID = "chottokun/ruri-v3-30m-lite"
DIST_DIR = os.path.abspath("./dist_assets")

def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("エラー: 環境変数 'HF_TOKEN' が設定されていません。")
        print("実行例:")
        print("  export HF_TOKEN=\"hf_xxxxxxxxxxxxxxxxxxxx\"")
        print("  uv run python scripts/deploy_to_hf.py")
        sys.exit(1)

    api = HfApi(token=hf_token)

    print(f"=== [1/3] リポジトリ存在確認・作成: {REPO_ID} ===")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print(f"リポジトリ準備完了: https://huggingface.co/{REPO_ID}")

    print(f"\n=== [2/3] アップロード対象ファイルの確認 ===")
    files_to_upload = []
    for root, dirs, files in os.walk(DIST_DIR):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, DIST_DIR)
            files_to_upload.append(rel_path)
            print(f"  - {rel_path} ({os.path.getsize(full_path):,} bytes)")

    print(f"\n=== [3/3] 成果物の一括アップロード開始 ===")
    api.upload_folder(
        folder_path=DIST_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="feat: upload sentencepiece_lite zero-torch ruri-v3-30m assets"
    )
    print("\n【デプロイ完了】すべての成果物が正常にアップロードされました！ 🎉")
    print(f"URL: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
