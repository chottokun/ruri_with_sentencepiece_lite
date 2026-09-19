# AGENTS.md

## Strict Constraints
- **Package Manager**: All executions and installations must use `uv`. Direct usage of `pip` or `conda` is forbidden.
- **Runtime Isolation**:
  - `dist_assets/ruri_v3_lite.py` must NEVER import `torch` or `transformers`.
  - Allowed runtime dependencies: `sentencepiece_lite`, `onnxruntime` / `onnxruntime-gpu`, `numpy`, `huggingface_hub`.
- **Inference Separation**:
  - CPU: Tokenization via `sentencepiece_lite` (BOS/EOS injection, padding, truncation).
  - GPU/CPU: Matrix multiplication via `onnxruntime`.
- **Model Switching**:
  - Check CUDA Compute Capability via `ctypes` (`nvcuda.dll` / `libcuda.so`). Do NOT import PyTorch for this check.
  - Capability < 7.0 (e.g. GTX 1080) or CPU: Load `model.onnx` (FP32).
  - Capability >= 7.0 (e.g. RTX 3060): Load `model_fp16.onnx` (FP16).
- **Security & Secret Hygiene (GitHub Public Readiness)**:
  - **No Hardcoded Tokens/Secrets**: NEVER commit API tokens, Hugging Face write tokens (`HF_TOKEN`), credentials, or private file paths into any file.
  - Token input must exclusively come from environment variables (e.g. `os.environ.get("HF_TOKEN")`) or user prompt.
  - Large binaries (`*.onnx`, `*.spm.fb`, `*.safetensors`, `.whl`) and cache directories (`.venv/`, `.cache/`, `hf_cache/`) must NOT be committed to git (ensure `.gitignore` exclusion).
  - Clean local logs and scratchpads before committing. Verify `git status` / `git diff` before every commit.

---

## File Structure & Roles
- `scripts/verify_tokenizer.py`: Phase 0 verification to confirm BOS/EOS token IDs, special tokens, and pooling behavior.
- `scripts/build_and_export.py`: Build `.whl` from source, export FP32 ONNX, and convert to FP16. (PyTorch/Transformers allowed here).
- `scripts/deploy_to_hf.py`: Upload `dist_assets/` to target Hugging Face model repo.
- `scripts/test_inference.py`: Verification script executing `dist_assets/ruri_v3_lite.py` under various text inputs.
- `dist_assets/ruri_v3_lite.py`: Self-contained wrapper class for end-users.
- `dist_assets/wheels/`: Built `sentencepiece_lite` wheel files.

---

## Command Reference (uv)

### 1. Environment Setup
```bash
uv init --app
uv venv --python 3.11
```

### 2. Phase 0 Verification
```bash
uv run --with "torch" --with "transformers" --with "sentencepiece" --with "huggingface_hub" python scripts/verify_tokenizer.py
```

### 3. Asset Build & Export
```bash
uv run --with "torch" --with "transformers" --with "onnx" --with "onnxconverter-common" --with "build" --with "huggingface_hub" python scripts/build_and_export.py
```

### 4. Deploy to Hugging Face
```bash
export HF_TOKEN="hf_xxx" # Pass via environment variable, never write in script
uv run --with "huggingface_hub" python scripts/deploy_to_hf.py
```

### 5. Lightweight Verification (No PyTorch)
```bash
# CPU test
uv run --with "onnxruntime" --with "numpy" --with "huggingface_hub" python scripts/test_inference.py

# GPU test
uv run --with "onnxruntime-gpu" --with "numpy" --with "huggingface_hub" python scripts/test_inference.py
```