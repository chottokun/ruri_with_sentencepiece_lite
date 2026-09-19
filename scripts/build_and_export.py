"""資産生成スクリプト (マルチモデル対応版)
ruri-v3 シリーズ (30m, 70m, 130m, 310m) の資産を一括または指定して生成する。

使用例:
  uv run python scripts/build_and_export.py --model 70m
  uv run python scripts/build_and_export.py --model 130m
  uv run python scripts/build_and_export.py --model 310m
  uv run python scripts/build_and_export.py --all
"""
import os
import sys
import shutil
import argparse
import subprocess
import onnx
import torch
from onnxconverter_common import float16
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

SUPPORTED_MODELS = {
    "30m": "cl-nagoya/ruri-v3-30m",
    "70m": "cl-nagoya/ruri-v3-70m",
    "130m": "cl-nagoya/ruri-v3-130m",
    "310m": "cl-nagoya/ruri-v3-310m",
}

BASE_DIST_DIR = os.path.abspath("./dist_assets")
BUILD_DIR = os.path.abspath("./build_spm")
GLOBAL_WHEELS_DIR = os.path.join(BASE_DIST_DIR, "wheels")

os.makedirs(BASE_DIST_DIR, exist_ok=True)
os.makedirs(GLOBAL_WHEELS_DIR, exist_ok=True)
os.makedirs(BUILD_DIR, exist_ok=True)

def ensure_sentencepiece_tools_and_wheel():
    """sentencepiece_lite の C++ ツール (spm_to_fb) および Wheel をビルド (未作成時のみ)"""
    spm_repo_dir = os.path.join(BUILD_DIR, "sentencepiece")
    if not os.path.exists(spm_repo_dir):
        print("Cloning google/sentencepiece...")
        subprocess.run([
            "git", "clone", "--depth=1", "--recurse-submodules", "--shallow-submodules",
            "https://github.com/google/sentencepiece.git", spm_repo_dir
        ], check=True)

    spm_build_dir = os.path.join(spm_repo_dir, "build")
    spm_to_fb_bin = os.path.join(spm_build_dir, "lite", "spm_to_fb")
    if not os.path.exists(spm_to_fb_bin):
        print("Building sentencepiece and spm_to_fb with cmake...")
        subprocess.run([
            "cmake", "-B", spm_build_dir, "-S", spm_repo_dir,
            "-DSPM_ENABLE_SHARED=OFF", "-DSPM_BUILD_TEST=OFF", "-DCMAKE_BUILD_TYPE=Release"
        ], check=True)
        subprocess.run([
            "cmake", "--build", spm_build_dir, "--config", "Release", "-j", str(os.cpu_count() or 4)
        ], check=True)

    whl_files = [os.path.join(GLOBAL_WHEELS_DIR, f) for f in os.listdir(GLOBAL_WHEELS_DIR) if f.endswith(".whl")]
    if not whl_files:
        print("Building sentencepiece_lite Python wheel...")
        pkg_dir = os.path.join(BUILD_DIR, "sentencepiece_lite_pkg")
        subprocess.run([
            sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", GLOBAL_WHEELS_DIR, pkg_dir
        ], check=True)
        whl_files = [os.path.join(GLOBAL_WHEELS_DIR, f) for f in os.listdir(GLOBAL_WHEELS_DIR) if f.endswith(".whl")]

    print(f"sentencepiece_lite wheel ready: {whl_files[0]}")
    print(f"spm_to_fb binary ready: {spm_to_fb_bin}")
    return spm_to_fb_bin, whl_files[0]

def export_model(model_key: str, spm_to_fb_bin: str, wheel_path: str):
    repo_id = SUPPORTED_MODELS[model_key]
    print(f"\n==================================================")
    print(f"  処理開始: {repo_id} ({model_key})")
    print(f"==================================================")

    # 30m はルートの dist_assets 互換、70m以降は dist_assets/ruri-v3-<key> またはモデル別
    out_dir = os.path.join(BASE_DIST_DIR, f"ruri_v3_{model_key}")
    os.makedirs(out_dir, exist_ok=True)
    wheels_sub_dir = os.path.join(out_dir, "wheels")
    os.makedirs(wheels_sub_dir, exist_ok=True)
    shutil.copyfile(wheel_path, os.path.join(wheels_sub_dir, os.path.basename(wheel_path)))

    # 1. tokenizer.model & FlatBuffers (.spm.fb)
    print(f"[{model_key}] 1/4: Tokenizer 辞書の取得と FlatBuffers 変換")
    dst_sp = os.path.join(out_dir, "tokenizer.model")
    if not os.path.exists(dst_sp):
        sp_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.model")
        shutil.copyfile(sp_path, dst_sp)
        os.chmod(dst_sp, 0o644)

    fb_path = os.path.join(out_dir, f"ruri_v3_{model_key}.spm.fb")
    if not os.path.exists(fb_path):
        print(f"Converting {dst_sp} -> {fb_path}...")
        subprocess.run([
            spm_to_fb_bin,
            f"--model={dst_sp}",
            f"--output={fb_path}"
        ], check=True)
    print(f"FlatBuffers: {fb_path} ({os.path.getsize(fb_path):,} bytes)")

    # 2. FP32 ONNX エクスポート
    print(f"[{model_key}] 2/4: FP32 ONNX エクスポート")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModel.from_pretrained(repo_id)
    model.eval()

    dummy_inputs = tokenizer(["テストテキストです。"], return_tensors="pt")
    fp32_onnx_path = os.path.join(out_dir, "model.onnx")

    torch.onnx.export(
        model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        fp32_onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=17,
        do_constant_folding=True
    )

    # PyTorch 2.14 Split 属性サニタイズ
    fp32_model = onnx.load(fp32_onnx_path)
    for node in fp32_model.graph.node:
        if node.op_type == "Split":
            attrs_to_remove = [a for a in node.attribute if a.name == "num_outputs"]
            for a in attrs_to_remove:
                node.attribute.remove(a)
    onnx.save(fp32_model, fp32_onnx_path)
    print(f"FP32 ONNX ready: {fp32_onnx_path}")

    # 3. FP16 変換
    print(f"[{model_key}] 3/4: FP16 ONNX 変換")
    fp16_model = float16.convert_float_to_float16(
        fp32_model,
        keep_io_types=True,
        disable_shape_infer=False
    )
    fp16_onnx_path = os.path.join(out_dir, "model_fp16.onnx")
    onnx.save(fp16_model, fp16_onnx_path)
    print(f"FP16 ONNX ready: {fp16_onnx_path} ({os.path.getsize(fp16_onnx_path):,} bytes)")

    # 4. ラッパー & README 生成
    print(f"[{model_key}] 4/4: ラッパーおよび README の配置")
    shutil.copyfile("dist_assets/ruri_v3_lite.py", os.path.join(out_dir, "ruri_v3_lite.py"))

    readme_content = f"""---
license: apache-2.0
language:
- ja
pipeline_tag: feature-extraction
tags:
- sentence-similarity
- embeddings
- sentencepiece_lite
- onnx
- zero-torch
---

# ruri-v3-{model_key}-lite: Zero-Torch & Ultra-Fast Japanese Embeddings

[`{repo_id}`](https://huggingface.co/{repo_id}) を、**PyTorch / Transformers 一切不要（Zero-Torch）** かつ **C++ コンパイル不要（事前ビルドWheel配布）** で実行できるように最適化した軽量日本語埋め込み環境です。

## インストール手順

```bash
pip install https://huggingface.co/Chottokun/ruri-v3-{model_key}-lite/resolve/main/wheels/{os.path.basename(wheel_path)} onnxruntime numpy huggingface_hub
```

## クイックスタート

```python
from ruri_v3_lite import RuriV3Lite
import numpy as np

# インスタンス化
model = RuriV3Lite(repo_id="Chottokun/ruri-v3-{model_key}-lite")

queries = ["検索クエリ: 日本の首都はどこですか？"]
documents = [
    "文章: 日本の首都は東京都です。",
    "文章: 大阪は関西地方の主要都市です。"
]

q_emb = model.encode(queries)
d_emb = model.encode(documents)

similarities = np.dot(q_emb, d_emb.T)[0]
print(f"クエリ vs 東京: {{similarities[0]:.4f}}")
print(f"クエリ vs 大阪: {{similarities[1]:.4f}}")
```
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    print(f"✨ {repo_id} ({model_key}) の資産生成が完了しました: {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="ruri-v3 資産生成スクリプト")
    parser.add_argument("--model", choices=["30m", "70m", "130m", "310m"], default="70m", help="対象モデル")
    parser.add_argument("--all", action="store_true", help="全モデルを一括生成")
    args = parser.parse_args()

    spm_to_fb_bin, wheel_path = ensure_sentencepiece_tools_and_wheel()

    if args.all:
        for k in ["70m", "130m", "310m"]:
            export_model(k, spm_to_fb_bin, wheel_path)
    else:
        export_model(args.model, spm_to_fb_bin, wheel_path)

if __name__ == "__main__":
    main()
