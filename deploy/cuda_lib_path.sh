#!/usr/bin/env bash
# Prints a LD_LIBRARY_PATH value pointing at the pip-installed NVIDIA CUDA/cuDNN
# libs inside .venv (onnxruntime-gpu can't find them on its own). Empty if none.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shopt -s nullglob
dirs=("$PROJECT_DIR"/.venv/lib/python3*/site-packages/nvidia/*/lib)
IFS=:
echo "${dirs[*]}"
