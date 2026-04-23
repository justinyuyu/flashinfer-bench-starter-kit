#include "kernel6_internal.cuh"

namespace kernel6_internal {

cudaError_t launch_fallback_gemm2_combine(const Gemm2Problem& p,
                                          const Gemm2Workspace& workspace,
                                          int total_tokens) {
    K6_CUDA_CHECK(cudaMemsetAsync(
        workspace.output_accum,
        0,
        output_accum_bytes(p.seq_len),
        p.stream));

    dim3 block(256);
    dim3 gemm2_grid(total_tokens, (HIDDEN_SIZE + block.x - 1) / block.x);
    fp8_gemm2_project_and_combine_kernel<<<gemm2_grid, block, 0, p.stream>>>(
        p.hidden_states,
        p.local_expert_ids,
        p.token_indices,
        p.token_expert_weights,
        p.gemm2_weights,
        p.gemm2_weights_scale,
        workspace.output_accum,
        total_tokens
    );
    K6_CUDA_CHECK(cudaGetLastError());

    int total_output_elems = p.seq_len * HIDDEN_SIZE;
    dim3 pack_grid((total_output_elems + block.x - 1) / block.x);
    f32_to_bf16_kernel<<<pack_grid, block, 0, p.stream>>>(
        workspace.output_accum,
        p.output,
        total_output_elems
    );
    K6_CUDA_CHECK(cudaGetLastError());
    return cudaSuccess;
}

}  // namespace kernel6_internal
