import os
import gzip
import shutil
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_ID = "cl-nagoya/ruri-v3-reranker-310m"
OUTPUT_DIR = "dist_assets/ruri_v3_reranker_310m"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== 1. 16層プルーニング ModernBERT の構築 ===")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

keep_indices = np.linspace(0, 24, 16, dtype=int)
print(f"Selecting 16 layers from 25: {list(keep_indices)}")
model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in keep_indices])
model.config.num_hidden_layers = 16

print("=== 2. ONNX (FP32) へのエクスポート (opset_version=17) ===")
fp32_onnx_path = os.path.join(OUTPUT_DIR, "model_pruned_16l.onnx")
dummy_input_ids = torch.ones((2, 64), dtype=torch.long)
dummy_attention_mask = torch.ones((2, 64), dtype=torch.long)

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
print(f"FP32 ONNX raw exported: {os.path.getsize(fp32_onnx_path)/1024/1024:.1f} MB")

print("=== 3. ONNX グラフのサニタイズ (Split num_outputs & value_info 除去) ===")
fp32_model = onnx.load(fp32_onnx_path)
for node in fp32_model.graph.node:
    if node.op_type == "Split":
        attrs_to_remove = [a for a in node.attribute if a.name == "num_outputs"]
        for a in attrs_to_remove:
            node.attribute.remove(a)
# 形状推論競合を防ぐため value_info をクリア
fp32_model.graph.ClearField("value_info")
onnx.save(fp32_model, fp32_onnx_path)
print("ONNX グラフサニタイズ完了")

print("=== 4. INT8 Full 量子化 (Linear + Embedding) ===")
int8_onnx_path = os.path.join(OUTPUT_DIR, "model_pruned_16l_int8.onnx")
quantize_dynamic(
    model_input=fp32_onnx_path,
    model_output=int8_onnx_path,
    op_types_to_quantize=["MatMul", "Gemm", "Gather"],
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
)
int8_size = os.path.getsize(int8_onnx_path) / 1024 / 1024
print(f"INT8 Pruned ONNX generated: {int8_size:.1f} MB")

print("=== 5. Gzip 圧縮版の生成 ===")
# 1. model_int8_full.onnx.gz
int8_full_src = os.path.join(OUTPUT_DIR, "model_int8_full.onnx")
int8_full_gz = os.path.join(OUTPUT_DIR, "model_int8_full.onnx.gz")
with open(int8_full_src, "rb") as f_in, gzip.open(int8_full_gz, "wb", compresslevel=6) as f_out:
    shutil.copyfileobj(f_in, f_out)
print(f"INT8-Full Gzip: {os.path.getsize(int8_full_gz)/1024/1024:.1f} MB")

# 2. model_pruned_16l_int8.onnx.gz
pruned_gz = os.path.join(OUTPUT_DIR, "model_pruned_16l_int8.onnx.gz")
with open(int8_onnx_path, "rb") as f_in, gzip.open(pruned_gz, "wb", compresslevel=6) as f_out:
    shutil.copyfileobj(f_in, f_out)
print(f"Pruned-16L INT8 Gzip: {os.path.getsize(pruned_gz)/1024/1024:.1f} MB")

# 一時ファイル削除
for p in [fp32_onnx_path, fp32_onnx_path + ".data", os.path.join(OUTPUT_DIR, "model_pruned_16l-inferred.onnx")]:
    if os.path.exists(p):
        os.remove(p)
        print(f"Cleaned up {p}")

print("\n=== すべての処理が正常終了しました ===")
