#!/usr/bin/env bash
set -e

TENSORRT_ROOT=~/Software/TensorRT-8.6.1.6
TRT_BIN=$TENSORRT_ROOT/bin/trtexec
TRT_LIB=$TENSORRT_ROOT/targets/x86_64-linux-gnu/lib

echo ">>> Deactivating conda env (if any)"
conda deactivate || true

echo ">>> Setting TensorRT LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TRT_LIB

echo ">>> Building TensorRT engine: 512x640"
$TRT_BIN \
  --onnx=spnet_512_640.onnx \
  --saveEngine=spnet_512_640.engine \
  --fp16 \
  --optShapes=rgb:1x3x512x640,depth:1x1x512x640,mask:1x1x512x640

echo ">>> Building TensorRT engine: 480x640"
$TRT_BIN \
  --onnx=spnet_480_640.onnx \
  --saveEngine=spnet_480_640.engine \
  --fp16 \
  --optShapes=rgb:1x3x480x640,depth:1x1x480x640,mask:1x1x480x640

echo ">>> Building TensorRT engine: 720x1280"
$TRT_BIN \
  --onnx=spnet_720_1280.onnx \
  --saveEngine=spnet_720_1280.engine \
  --fp16 \
  --optShapes=rgb:1x3x720x1280,depth:1x1x720x1280,mask:1x1x720x1280

echo ">>> Building TensorRT engine: 1080x1920"
$TRT_BIN \
  --onnx=spnet_1080_1920.onnx \
  --saveEngine=spnet_1080_1920.engine \
  --fp16 \
  --optShapes=rgb:1x3x1080x1920,depth:1x1x1080x1920,mask:1x1x1080x1920

echo ">>> TensorRT engine build finished."