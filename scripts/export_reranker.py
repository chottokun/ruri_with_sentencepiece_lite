"""ruri-v3-reranker-310m ONNX エクスポートスクリプト
1. Hugging Face より cl-nagoya/ruri-v3-reranker-310m をダウンロード
2. ModernBertForSequenceClassification を FP32 ONNX にエクスポート
3. Split ノードの無効属性サニタイズ
4. onnxconverter-common により FP16 ONNX へ変換
5. FlatBuffers 形式トークナイザー辞書およびラッパー、LICENSE、README を配備
"""
import os
import sys
import shutil
import torch
import onnx
from onnxconverter_common import float16
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "cl-nagoya/ruri-v3-reranker-310m"
OUT_DIR = os.path.abspath("./dist_assets/ruri_v3_reranker_310m")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"==================================================================")
    print(f"  ruri-v3-reranker-310m エクスポート開始")
    print(f"  出力先: {OUT_DIR}")
    print(f"==================================================================")

    # 1. モデルとトークナイザーのロード
    print("[1/5] モデルおよびトークナイザーのロード中...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

    # 2. FP32 ONNX エクスポート
    print("[2/5] FP32 ONNX エクスポート中...")
    fp32_onnx_path = os.path.join(OUT_DIR, "model.onnx")

    dummy_input_ids = torch.ones((2, 64), dtype=torch.long)
    dummy_attention_mask = torch.ones((2, 64), dtype=torch.long)

    # ModernBertForSequenceClassification のラッパー (logits のみを返す)
    class RerankerExportWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask)
            return out.logits

    wrapper = RerankerExportWrapper(model).eval()

    torch.onnx.export(
        wrapper,
        (dummy_input_ids, dummy_attention_mask),
        fp32_onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  FP32 ONNX 保存完了: {fp32_onnx_path}")

    # 3. ONNX グラフのサニタイズ (PyTorch 2.14 Split num_outputs 属性バグの除去)
    print("[3/5] ONNX グラフの属性サニタイズ中...")
    fp32_model = onnx.load(fp32_onnx_path)
    for node in fp32_model.graph.node:
        if node.op_type == "Split":
            attrs_to_remove = [a for a in node.attribute if a.name == "num_outputs"]
            for a in attrs_to_remove:
                node.attribute.remove(a)
    onnx.save(fp32_model, fp32_onnx_path)
    print("  ONNX グラフサニタイズ完了")

    # 4. FP16 変換
    print("[4/5] FP16 ONNX 変換中 (Tensor コア最適化)...")
    fp16_model = float16.convert_float_to_float16(
        fp32_model,
        keep_io_types=True,
        disable_shape_infer=False
    )
    fp16_onnx_path = os.path.join(OUT_DIR, "model_fp16.onnx")
    onnx.save(fp16_model, fp16_onnx_path)
    print(f"  FP16 ONNX 保存完了: {fp16_onnx_path} ({os.path.getsize(fp16_onnx_path):,} bytes)")

    # 5. 資材の配置 (FlatBuffers 辞書, wheels, LICENSE)
    print("[5/5] トークナイザー辞書・ホイール・ライセンスの配置中...")
    spm_fb_src = "./dist_assets/ruri_v3_310m/ruri_v3_310m.spm.fb"
    spm_fb_dst = os.path.join(OUT_DIR, "ruri_v3_reranker_310m.spm.fb")
    shutil.copyfile(spm_fb_src, spm_fb_dst)

    # tokenizer.model のフォールバックコピー
    spm_src = "./dist_assets/ruri_v3_310m/tokenizer.model"
    spm_dst = os.path.join(OUT_DIR, "tokenizer.model")
    if os.path.exists(spm_src):
        shutil.copyfile(spm_src, spm_dst)

    # LICENSE のコピー
    shutil.copyfile("LICENSE", os.path.join(OUT_DIR, "LICENSE"))

    # Wheels ディレクトリの配置
    os.makedirs(os.path.join(OUT_DIR, "wheels"), exist_ok=True)
    wheel_src = "./dist_assets/wheels/sentencepiece_lite-0.1.0-cp311-cp311-linux_x86_64.whl"
    shutil.copyfile(wheel_src, os.path.join(OUT_DIR, "wheels", os.path.basename(wheel_src)))

    print(f"\n✨ ruri-v3-reranker-310m の資材生成がすべて完了しました: {OUT_DIR}")

if __name__ == "__main__":
    main()
