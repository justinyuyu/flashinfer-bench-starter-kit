#!/bin/bash
# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Use paths relative to the uv environment
VENV_PATH="$PROJECT_ROOT/.venv"
NVCC=nvcc
TVM_FFI_PATH=$VENV_PATH/lib/python3.12/site-packages/tvm_ffi
MOE_MLP_DIR="$PROJECT_ROOT/solution/cuda/moe_expert_mlp"

if [ -z "${CUTLASS_DIR:-}" ]; then
    if [ -f "/workspace/cutlass/include/cutlass/cutlass.h" ]; then
        CUTLASS_DIR="/workspace/cutlass"
    elif [ -f "$PROJECT_ROOT/cutlass/include/cutlass/cutlass.h" ]; then
        CUTLASS_DIR="$PROJECT_ROOT/cutlass"
    else
        CUTLASS_DIR="$HOME/.local/src/cutlass"
    fi
fi

ENABLE_CUTLASS=0

# Check if TVM_FFI_PATH exists, if not, try to find it via python
if [ ! -d "$TVM_FFI_PATH" ]; then
    TVM_FFI_PATH=$(python3 -c "import tvm_ffi; import os; print(os.path.dirname(tvm_ffi.__file__))" 2>/dev/null)
fi

EXTRA_NVCC_FLAGS=()
if [ -f "$CUTLASS_DIR/include/cutlass/cutlass.h" ]; then
    ENABLE_CUTLASS=1
    EXTRA_NVCC_FLAGS+=("-I$CUTLASS_DIR/include" "-DK4_ENABLE_CUTLASS=1" "--expt-relaxed-constexpr")
    echo "CUTLASS detected at $CUTLASS_DIR, enabling fast grouped GEMM backends."
else
    echo "CUTLASS not found, building fallback/tiled backends only."
fi

echo "Building MoE FFI library..."
$NVCC -shared -Xcompiler -fPIC \
    -arch=sm_90 \
    -I$TVM_FFI_PATH/include \
    -L$TVM_FFI_PATH/lib \
    -ltvm_ffi \
    -I$MOE_MLP_DIR \
    "${EXTRA_NVCC_FLAGS[@]}" \
    -o $SCRIPT_DIR/librouter_ffi.so \
    $PROJECT_ROOT/solution/cuda/moe_ffi.cu

echo "Building MoE Expert MLP tests..."
if [ "$ENABLE_CUTLASS" -eq 1 ]; then
    make -C $MOE_MLP_DIR CUDA_ARCH=sm_90 CUTLASS_DIR="$CUTLASS_DIR" build-cutlass
else
    make -C $MOE_MLP_DIR CUDA_ARCH=sm_90 build-fallback
fi
