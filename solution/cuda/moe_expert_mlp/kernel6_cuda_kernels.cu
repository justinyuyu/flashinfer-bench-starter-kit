#include "kernel6_internal.cuh"

namespace kernel6_internal {

__global__ void fp8_gemm2_project_and_combine_kernel(
    const __nv_bfloat16* __restrict__ inter,
    const int*           __restrict__ local_expert_ids,
    const int*           __restrict__ token_indices,
    const float*         __restrict__ routing_w,
    const fp8_e4m3*      __restrict__ W2,
    const float*         __restrict__ W2_scale,
    float*               __restrict__ output_accum,
    int                  total_tokens)
{
    int tok = blockIdx.x;
    int out_col = threadIdx.x + blockIdx.y * blockDim.x;
    if (tok >= total_tokens || out_col >= HIDDEN_SIZE) return;

    int expert_id = load_cached(local_expert_ids + tok);

    const __nv_bfloat16* inter_tok = inter + (size_t)tok * INTERMEDIATE_SIZE;
    const fp8_e4m3* W_e = W2 + (size_t)expert_id * HIDDEN_SIZE * INTERMEDIATE_SIZE;
    const float* Ws_e = W2_scale + (size_t)expert_id * NUM_HIDDEN_BLOCKS * NUM_INTER_BLOCKS;

    int hb = out_col / BLOCK_SIZE;
    float acc = 0.f;
    for (int ib = 0; ib < NUM_INTER_BLOCKS; ++ib) {
        float tile_scale = load_cached(Ws_e + hb * NUM_INTER_BLOCKS + ib);
        int k_base = ib * BLOCK_SIZE;
        float dot = 0.f;
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a = __bfloat162float(inter_tok[k_base + k]);
            float w = fp8_to_float(load_cached(W_e + out_col * INTERMEDIATE_SIZE + k_base + k));
            dot += a * w;
        }
        acc += dot * tile_scale;
    }


    int orig_tok = load_cached(token_indices + tok);
    float rw = load_cached(routing_w + tok);
    atomicAdd(output_accum + (size_t)orig_tok * HIDDEN_SIZE + out_col, acc * rw);
}

__global__ void combine_projected_kernel(
    const float*   __restrict__ projected,
    const int*     __restrict__ token_indices,
    const float*   __restrict__ routing_w,
    float          routed_scaling_factor,
    float*         __restrict__ output_accum,
    int            total_tokens,
    int            seq_len)
{
    int tok = blockIdx.x;
    int out_col = threadIdx.x + blockIdx.y * blockDim.x;
    if (tok >= total_tokens || out_col >= HIDDEN_SIZE) return;

    int orig_tok = load_cached(token_indices + tok);
    if (orig_tok < 0 || orig_tok >= seq_len) return;

    float rw = load_cached(routing_w + tok);
    float value = projected[(size_t)tok * HIDDEN_SIZE + out_col];
    atomicAdd(output_accum + (size_t)orig_tok * HIDDEN_SIZE + out_col, value * rw);
}

__global__ void f32_to_bf16_kernel(
    const float* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

__global__ void bf16_rows_to_f32_kernel(
    const __nv_bfloat16* __restrict__ input,
    int                  token_offset,
    int                  token_count,
    int                  row_stride,
    float*               __restrict__ output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = token_count * row_stride;
    if (idx >= total) return;

    int local_tok = idx / row_stride;
    int col = idx % row_stride;
    output[idx] = __bfloat162float(input[(size_t)(token_offset + local_tok) * row_stride + col]);
}

#if defined(K4_ENABLE_CUTLASS)
__global__ void dequant_gemm2_weight_kernel(
    const fp8_e4m3* __restrict__ weights,
    const float*    __restrict__ scales,
    float*          __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
    if (idx >= total) return;

    int row = idx / INTERMEDIATE_SIZE;
    int k = idx % INTERMEDIATE_SIZE;
    int hb = row / BLOCK_SIZE;
    int ib = k / BLOCK_SIZE;
    float scale = scales[hb * NUM_INTER_BLOCKS + ib];
    out[idx] = fp8_to_float(weights[(size_t)row * INTERMEDIATE_SIZE + k]) * scale;
}
#endif

}  // namespace kernel6_internal
