"""資産生成スクリプト
1. sentencepiece_lite のソース取得、C++ツール (spm_to_fb) のビルド
2. pybind11 による Python wheel のビルド
3. ruri-v3-30m の tokenizer.model 取得と FlatBuffers (.spm.fb) 形式への変換
4. FP32 ONNX エクスポート (ModernBertModel, input_ids + attention_mask)
5. FP16 ONNX への変換 (onnxconverter-common)
"""
import os
import sys
import shutil
import subprocess
import urllib.request
import onnx
import torch
from onnxconverter_common import float16
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

MODEL_ID = "cl-nagoya/ruri-v3-30m"
DIST_DIR = os.path.abspath("./dist_assets")
BUILD_DIR = os.path.abspath("./build_spm")
WHEELS_DIR = os.path.join(DIST_DIR, "wheels")

os.makedirs(DIST_DIR, exist_ok=True)
os.makedirs(WHEELS_DIR, exist_ok=True)
os.makedirs(BUILD_DIR, exist_ok=True)

print("=== [1/6] SentencePiece リポジトリの準備 & spm_to_fb ビルド ===")
spm_repo_dir = os.path.join(BUILD_DIR, "sentencepiece")
if not os.path.exists(spm_repo_dir):
    print("Cloning google/sentencepiece...")
    subprocess.run([
        "git", "clone", "--depth=1", "--recurse-submodules", "--shallow-submodules",
        "https://github.com/google/sentencepiece.git", spm_repo_dir
    ], check=True)

spm_build_dir = os.path.join(spm_repo_dir, "build")
os.makedirs(spm_build_dir, exist_ok=True)
spm_to_fb_bin = os.path.join(spm_build_dir, "src", "spm_to_fb")
if not os.path.exists(spm_to_fb_bin):
    spm_to_fb_alt = os.path.join(spm_build_dir, "lite", "spm_to_fb")
    if os.path.exists(spm_to_fb_alt):
        spm_to_fb_bin = spm_to_fb_alt

if not os.path.exists(spm_to_fb_bin):
    print("Building sentencepiece and spm_to_fb with cmake...")
    subprocess.run([
        "cmake", "-B", spm_build_dir, "-S", spm_repo_dir,
        "-DSPM_ENABLE_SHARED=OFF", "-DSPM_BUILD_TEST=OFF", "-DCMAKE_BUILD_TYPE=Release"
    ], check=True)
    subprocess.run([
        "cmake", "--build", spm_build_dir, "--config", "Release", "-j", str(os.cpu_count() or 4)
    ], check=True)

    # 探索
    for root, dirs, files in os.walk(spm_build_dir):
        if "spm_to_fb" in files:
            spm_to_fb_bin = os.path.join(root, "spm_to_fb")
            break

print(f"spm_to_fb binary found at: {spm_to_fb_bin}")

print("\n=== [2/6] sentencepiece_lite Python Wheel のビルド ===")
# pybind11 を使った sentencepiece_lite パッケージのセットアップ
pkg_dir = os.path.join(BUILD_DIR, "sentencepiece_lite_pkg")
os.makedirs(pkg_dir, exist_ok=True)

sbp_module_cc = """#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>
#include "sentencepiece_lite.h"

namespace py = pybind11;

class FastSBPTokenizer {
public:
  FastSBPTokenizer(const std::string& model_path) {
    fd_ = open(model_path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("Could not open model file: " + model_path);
    struct stat sb;
    if (fstat(fd_, &sb) != 0) {
      close(fd_);
      throw std::runtime_error("fstat failed");
    }
    size_ = sb.st_size;
    addr_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (addr_ == MAP_FAILED) {
      close(fd_);
      throw std::runtime_error("mmap failed");
    }
    std::string_view view(static_cast<const char*>(addr_), size_);
    processor_ = std::make_unique<sentencepiece::lite::SentencePieceLiteProcessor>(view);
    if (processor_->status() != sentencepiece::lite::StatusCode::kOk) {
      munmap(addr_, size_);
      close(fd_);
      throw std::runtime_error("SentencePieceLiteProcessor failed to load model");
    }
  }

  ~FastSBPTokenizer() {
    if (addr_ && addr_ != MAP_FAILED) munmap(addr_, size_);
    if (fd_ >= 0) close(fd_);
  }

  std::vector<int> encode(const std::string& text, size_t chunk_size = 16384, int num_threads = 0) {
    if (num_threads > 0) omp_set_num_threads(num_threads);

    std::string_view sv(text);
    std::vector<std::string_view> chunks;
    size_t start = 0;
    while (start < sv.size()) {
      if (start + chunk_size >= sv.size()) {
        chunks.push_back(sv.substr(start));
        break;
      }
      size_t split_pos = sv.find('\\n', start + chunk_size);
      if (split_pos == std::string_view::npos) split_pos = sv.find(' ', start + chunk_size);
      if (split_pos != std::string_view::npos && split_pos < sv.size()) {
        chunks.push_back(sv.substr(start, (split_pos + 1) - start));
        start = split_pos + 1;
      } else {
        chunks.push_back(sv.substr(start));
        break;
      }
    }

    std::vector<std::vector<int>> per_chunk(chunks.size());
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < chunks.size(); ++i) {
      processor_->Encode(chunks[i], &per_chunk[i]);
    }

    size_t total = 0;
    for (const auto& ids : per_chunk) total += ids.size();
    std::vector<int> res;
    res.reserve(total);
    for (auto& ids : per_chunk) {
      res.insert(res.end(), ids.begin(), ids.end());
    }
    return res;
  }

private:
  int fd_ = -1;
  size_t size_ = 0;
  void* addr_ = nullptr;
  std::unique_ptr<sentencepiece::lite::SentencePieceLiteProcessor> processor_;
};

PYBIND11_MODULE(sentencepiece_lite, m) {
  m.doc() = "SentencePiece Lite Python Wrapper with Fast SBP Tokenization";
  py::class_<FastSBPTokenizer>(m, "FastSBPTokenizer")
      .def(py::init<const std::string&>(), py::arg("model_path"))
      .def("encode", &FastSBPTokenizer::encode,
           py::arg("text"), py::arg("chunk_size") = 16384, py::arg("num_threads") = 0);
}
"""

with open(os.path.join(pkg_dir, "sbp_module.cc"), "w", encoding="utf-8") as f:
    f.write(sbp_module_cc)

# sentencepiece_lite.h / .cc をコピー
spm_lite_dir = os.path.join(spm_repo_dir, "src", "lite")
if not os.path.exists(spm_lite_dir):
    spm_lite_dir = os.path.join(spm_repo_dir, "lite")

shutil.copy(os.path.join(spm_lite_dir, "sentencepiece_lite.h"), pkg_dir)
shutil.copy(os.path.join(spm_lite_dir, "sentencepiece_lite.cc"), pkg_dir)

# FlatBuffers インクルードディレクトリの特定
flatbuffers_include = os.path.join(spm_build_dir, "third_party", "flatbuffers-src", "include")
if not os.path.exists(flatbuffers_include):
    flatbuffers_include = os.path.join(spm_repo_dir, "third_party", "flatbuffers", "include")

# 自動生成ヘッダーの検索 (sentencepiece_lite_generated.h)
generated_header_dir = os.path.join(spm_build_dir, "lite")
for root, dirs, files in os.walk(spm_build_dir):
    if "sentencepiece_lite_generated.h" in files:
        generated_header_dir = root
        break

absl_include_dir = os.path.join(spm_build_dir, "third_party", "abseil-cpp-src")
if not os.path.exists(absl_include_dir):
    absl_include_dir = os.path.join(spm_repo_dir, "third_party", "abseil-cpp")

setup_py_content = f"""from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'sentencepiece_lite',
        ['sbp_module.cc', 'sentencepiece_lite.cc'],
        include_dirs=[
            pybind11.get_include(),
            '.',
            r'{flatbuffers_include}',
            r'{generated_header_dir}',
            r'{absl_include_dir}',
        ],
        extra_compile_args=['-O3', '-std=c++20', '-fPIC', '-fopenmp'],
        extra_link_args=['-fopenmp'],
        language='c++'
    ),
]

setup(
    name='sentencepiece_lite',
    version='0.1.0',
    description='SentencePiece Lite with Safe Boundary Pre-tokenization',
    ext_modules=ext_modules,
)
"""

pyproject_toml_content = """[build-system]
requires = ["setuptools>=40.8.0", "wheel", "pybind11"]
build-backend = "setuptools.build_meta"
"""
with open(os.path.join(pkg_dir, "pyproject.toml"), "w", encoding="utf-8") as f:
    f.write(pyproject_toml_content)

with open(os.path.join(pkg_dir, "setup.py"), "w", encoding="utf-8") as f:
    f.write(setup_py_content)

print("Building wheel with python -m build --no-isolation...")
subprocess.run([
    sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", WHEELS_DIR, pkg_dir
], check=True)

# 動作検証用にカレント環境にもインストール
whl_files = [os.path.join(WHEELS_DIR, f) for f in os.listdir(WHEELS_DIR) if f.endswith(".whl")]
print(f"Generated wheels: {whl_files}")

print("\n=== [3/6] tokenizer.model の取得 & FlatBuffers (.spm.fb) への変換 ===")
dst_sp = os.path.join(DIST_DIR, "tokenizer.model")
if not os.path.exists(dst_sp):
    sp_path = hf_hub_download(repo_id=MODEL_ID, filename="tokenizer.model")
    if os.path.exists(dst_sp):
        os.remove(dst_sp)
    shutil.copyfile(sp_path, dst_sp)
    os.chmod(dst_sp, 0o644)

fb_path = os.path.join(DIST_DIR, "ruri_v3_30m.spm.fb")
if not os.path.exists(fb_path):
    print(f"Converting {dst_sp} to {fb_path} using {spm_to_fb_bin}...")
    subprocess.run([
        spm_to_fb_bin,
        f"--model={dst_sp}",
        f"--output={fb_path}"
    ], check=True)
print(f"FlatBuffers model ready: {fb_path} (size: {os.path.getsize(fb_path)} bytes)")

print("\n=== [4/6] FP32 ONNX エクスポート ===")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()

dummy_inputs = tokenizer(["テストテキストです。"], return_tensors="pt")
fp32_onnx_path = os.path.join(DIST_DIR, "model.onnx")

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
print(f"FP32 ONNX exported: {fp32_onnx_path} (size: {os.path.getsize(fp32_onnx_path)} bytes)")

print("\n=== [5/6] FP16 ONNX への変換 ===")
fp32_model = onnx.load(fp32_onnx_path)
fp16_model = float16.convert_float_to_float16(
    fp32_model,
    keep_io_types=True,
    disable_shape_infer=False
)
fp16_onnx_path = os.path.join(DIST_DIR, "model_fp16.onnx")
onnx.save(fp16_model, fp16_onnx_path)
print(f"FP16 ONNX saved: {fp16_onnx_path} (size: {os.path.getsize(fp16_onnx_path)} bytes)")

print(f"\n=== [6/6] 全資産の生成が正常に完了しました: {DIST_DIR} ===")
for item in sorted(os.listdir(DIST_DIR)):
    item_path = os.path.join(DIST_DIR, item)
    if os.path.isfile(item_path):
        print(f"  - {item} ({os.path.getsize(item_path):,} bytes)")
    else:
        print(f"  - {item}/")
