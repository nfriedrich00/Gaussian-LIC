#!/usr/bin/env bash
set -e

ENV_NAME=spnet

echo ">>> Activating conda env: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo ">>> Export ONNX (512x640)"
python export_onnx_512_640.py

echo ">>> Export ONNX (480x640)"
python export_onnx_480_640.py

echo ">>> Export ONNX (720x1280)"
python export_onnx_720_1280.py

echo ">>> Export ONNX (1080x1920)"
python export_onnx_1080_1920.py

echo ">>> All ONNX exports finished successfully."