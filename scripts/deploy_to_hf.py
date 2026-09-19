"""Hugging Face デプロイスクリプト
dist_assets/ 以下の全成果物を Hugging Face Hub (chottokun/ruri-v3-30m-lite) にアップロードする。

※ セキュリティ規定:
  HF_TOKEN は環境変数または実行時に入力し、スクリプト内にハードコードしないこと。
"""
import os
import sys
from huggingface_hub import HfApi, get_token

def main():
    hf_token = os.environ.get("HF_TOKEN") or get_token()
    if not hf_token:
        print("エラー: 有効な Hugging Face トークンが見つかりません。")
        print("実行例:")
        print("  export HF_TOKEN=\"hf_xxxxxxxxxxxxxxxxxxxx\"")
        print("  または huggingface-cli login")
        sys.exit(1)

    api = HfApi(token=hf_token)
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/ruri-v3-30m-lite"

    print(f"=== [1/3] リポジトリ存在確認・作成: {repo_id} ===")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"リポジトリ準備完了: https://huggingface.co/{repo_id}")

    DIST_DIR = os.path.abspath("./dist_assets")

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
        repo_id=repo_id,
        repo_type="model",
        commit_message="feat: upload sentencepiece_lite zero-torch ruri-v3-30m assets"
    )
    print("\n【デプロイ完了】すべての成果物が正常にアップロードされました！ 🎉")
    print(f"URL: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
