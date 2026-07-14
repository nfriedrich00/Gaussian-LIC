# SPNet TensorRT Runtime Status

Status date: 2026-07-14

## Local Result

The SPNet depth-completion runtime artifact has been generated locally for the
RTX 5070 Ti Laptop GPU (`sm_120`) path:

```text
TensorRT SDK: /home/frank/Software/TensorRT-10.9.0.34-cuda12.8
Checkpoint:   external/Gaussian-LIC/ckpt/Large_300.pth
512 ONNX:     /home/frank/Software/TensorRT-engines/spnet_512_640.onnx
512 engine:   /home/frank/Software/TensorRT-engines/spnet_512_640_fp16.engine
480 ONNX:     /home/frank/Software/TensorRT-engines/spnet_480_640.onnx
480 engine:   /home/frank/Software/TensorRT-engines/spnet_480_640_fp16.engine
```

Artifact hashes:

```text
Large_300.pth:
  e4a3dbaae1aff425c26db6083cb6ea8e1d6f13d719cc567054b2522b3bf84559
spnet_512_640.onnx:
  d977ba4ddf843d224778170d43453d4be4c532a72945a4949b97d1874654c24d
spnet_512_640_fp16.engine:
  514b5e4bde4e39f3290af0f68ed3f4ed72bb1553e50f3a46819e3a0f9ec2804a
spnet_480_640.onnx:
  0d1ed2c97d7049af8d2712ab058dd0c8696c59c84bcb1362c512f899ee05c3ad
spnet_480_640_fp16.engine:
  d8ff46cf67912cf18913e07993bc65ff499a71508d3c9e7a993d376a210b3f4b
```

The engines are intentionally not checked into git. They are approximately
479 MB hardware/runtime-specific TensorRT plans. The previous 512 plan is
preserved as
`spnet_512_640_fp16.engine.67b0f9f8360dd771.bak`; the active plans were rebuilt
on this GPU and no longer emit TensorRT's cross-device-model warning.

## Benchmark

Command:

```bash
./scripts/install_local_tensorrt_10_9.sh
SPNET_PYTHON=/home/frank/miniconda3/envs/active-gs-original/bin/python \
TENSORRT_ROOT=/home/frank/Software/TensorRT-10.9.0.34-cuda12.8 \
  scripts/build_spnet_engine.sh \
  --download \
  --output-dir /home/frank/Software/TensorRT-engines

# Build the 480x640 profile used by m2dgr/mcd.
SPNET_PYTHON=/home/frank/miniconda3/envs/active-gs-original/bin/python \
TENSORRT_ROOT=/home/frank/Software/TensorRT-10.9.0.34-cuda12.8 \
  scripts/build_spnet_engine.sh \
  --height 480 \
  --output-dir /home/frank/Software/TensorRT-engines
```

Observed `trtexec` summary:

```text
TensorRT version: 10.9.0
Compute Capability: 12.0
512x640 engine size: 478,634,188 bytes
512x640 throughput: 36.8397 qps
512x640 GPU compute mean/median/p95: 26.927/24.8081/37.1979 ms

480x640 engine size: 480,817,404 bytes
480x640 throughput: 38.4239 qps
480x640 GPU compute mean: 25.8273 ms
```

Both profiles pass the native `depth_completer_probe` at their declared shape.
Their mean GPU compute time satisfies the depth-completion runtime target of
<= 30 ms/frame on the tested machine.

## Compatibility Note

TensorRT 8.6.1.6 can parse the exported ONNX, but engine creation fails on this
GPU with:

```text
Error Code 2: Internal Error (Assertion major >= 0 && major < 10 failed.)
```

That failure is a TensorRT 8.x `sm_120` support gap, not an SPNet model export
failure. The build helper now prefers the local TensorRT 10.9 CUDA 12.8 SDK and
emits an explicit error if a caller tries TensorRT 8.x on the local `sm_120`
GPU.
