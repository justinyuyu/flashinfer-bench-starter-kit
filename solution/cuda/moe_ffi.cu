#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/container/tensor.h>

// Include the existing CUDA kernel implementations
#include "moe_routing.cu"
#include "moe_scan.cu"
#include "moe_dispatch.cu"
#include "moe_reindex.cu"
#include "moe_expert_mlp/kernel4_cuda_kernels.cu"
#include "moe_expert_mlp/kernel4_backends.cu"
#include "moe_expert_mlp/kernel4_cutlass.cu"
#include "moe_expert_mlp/kernel4.cu"
#include "moe_expert_mlp/kernel6_cuda_kernels.cu"
#include "moe_expert_mlp/kernel6_backends.cu"
#include "moe_expert_mlp/kernel6_cutlass.cu"
#include "moe_expert_mlp/kernel6.cu"

#include <algorithm>
#include <cstdio>


namespace {

struct WorkspaceCache {
    void* ptr = nullptr;
    size_t bytes = 0;

    ~WorkspaceCache() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    void* reserve(size_t requested_bytes) {
        if (requested_bytes == 0) {
            return nullptr;
        }
        if (bytes >= requested_bytes && ptr) {
            return ptr;
        }
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
            bytes = 0;
        }
        if (cudaMalloc(&ptr, requested_bytes) != cudaSuccess) {
            ptr = nullptr;
            bytes = 0;
            return nullptr;
        }
        bytes = requested_bytes;
        return ptr;
    }
};

thread_local WorkspaceCache k4_workspace_cache;
thread_local WorkspaceCache k6_workspace_cache;

int choose_total_tokens(tvm::ffi::Tensor expert_token_offsets) {
    int total_tok = 0;
    cudaMemcpy(&total_tok,
               static_cast<int*>(expert_token_offsets.data_ptr()) + moe_spec::NUM_LOCAL_EXPERTS,
               sizeof(int),
               cudaMemcpyDeviceToHost);
    return total_tok;
}

Kernel4Backend choose_kernel4_backend_policy(int seq_len,
                                             int total_tok,
                                             bool has_local_expert_ids) {
    // Small: minimize setup overhead.
    if (total_tok <= 256 || seq_len <= 32) {
        return has_local_expert_ids ? Kernel4Backend::Fallback : Kernel4Backend::Tiled;
    }

    // Promote to CUTLASS earlier for medium/large workloads once the
    // grouped GEMM setup cost is amortized.
    if (k4_cutlass_available() && (total_tok >= 768 || seq_len >= 1024)) {
        return Kernel4Backend::Cutlass;
    }

    // Medium: hand-written tiled path usually wins over reference/fallback.
    if (total_tok <= 4096 || seq_len <= 512) {
        return Kernel4Backend::Tiled;
    }

    // Large: prefer grouped GEMM when available; otherwise keep the tiled path.
    if (k4_cutlass_available()) {
        return Kernel4Backend::Cutlass;
    }
    return Kernel4Backend::Tiled;
}

Kernel6Backend choose_kernel6_backend_policy(int seq_len, int total_tok) {
    // Small/medium: fused fallback avoids CUTLASS setup overhead.
    if (total_tok <= 256 || seq_len <= 64) {
        return Kernel6Backend::Fallback;
    }

    // Promote to CUTLASS earlier than kernel4 because fallback kernel6 still
    // carries high memory traffic even at moderate token counts.
    if (k6_cutlass_available() && (total_tok >= 768 || seq_len >= 1024)) {
        return Kernel6Backend::Cutlass;
    }

    // Large: CUTLASS wins when available.
    if (k6_cutlass_available()) {
        return Kernel6Backend::Cutlass;
    }
    return Kernel6Backend::Fallback;
}

}  // namespace

namespace ffi = tvm::ffi;

// ─── Router FFI ───────────────────────────────────────────────────────────────

void router_ffi_wrapper(ffi::Tensor routing_logits,         // [T, 256]
                        ffi::Tensor routing_bias,           // [256]
                        ffi::Tensor expert_token_counts,    // [256]
                        ffi::Tensor token_expert_indices,   // [T, 8]
                        ffi::Tensor token_expert_weights,   // [T, 8]
                        ffi::Tensor token_expert_slots,     // [T, 8]
                        int T, int local_expert_offset, float routed_scaling_factor) {
    
    const int E_GLOBAL = 256;
    const int E_LOCAL = 32;
    const int TOP_K = 8;
    
    dim3 threads(256);
    dim3 blocks(T);
    
    router<E_GLOBAL, E_LOCAL, TOP_K><<<blocks, threads>>>(
        static_cast<const float*>(routing_logits.data_ptr()),
        static_cast<const __nv_bfloat16*>(routing_bias.data_ptr()),
        static_cast<int*>(expert_token_counts.data_ptr()),
        static_cast<int*>(token_expert_indices.data_ptr()),
        static_cast<float*>(token_expert_weights.data_ptr()),
        static_cast<int*>(token_expert_slots.data_ptr()),
        T, local_expert_offset, routed_scaling_factor
    );
}

static auto _router = ffi::reflection::GlobalDef().def("router_ffi", router_ffi_wrapper);

// ─── Scan (Prefix Sum) FFI ───────────────────────────────────────────────────

void scan_ffi_wrapper(ffi::Tensor counts, ffi::Tensor offsets, int num_items) {
    // num_items is E_LOCAL + 1 (e.g., 33)
    // counts has E_LOCAL items (e.g., 32)
    int E = num_items - 1;
    exclusive_scan_cub(static_cast<int*>(counts.data_ptr()), 
                       static_cast<int*>(offsets.data_ptr()), E);
    
    // Compute the last element: offsets[E] = offsets[E-1] + counts[E-1]
    int last_c, last_o;
    cudaMemcpy(&last_c, static_cast<int*>(counts.data_ptr()) + E - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_o, static_cast<int*>(offsets.data_ptr()) + E - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int total = last_c + last_o;
    cudaMemcpy(static_cast<int*>(offsets.data_ptr()) + E, &total, sizeof(int), cudaMemcpyHostToDevice);
}

static auto _scan = ffi::reflection::GlobalDef().def("scan_ffi", scan_ffi_wrapper);

// ─── Dispatch (Permutation) FFI ──────────────────────────────────────────────

void dispatch_ffi_wrapper(ffi::Tensor hidden_states_fp8,     // [T, H]  FP8 E4M3
                          ffi::Tensor hidden_states_scale,   // [H/128, T]  float32
                          ffi::Tensor token_expert_indices,  // [T, TOP_K]
                          ffi::Tensor token_expert_slots,    // [T, TOP_K]
                          ffi::Tensor expert_offsets,        // [E_GLOBAL + 1]
                          ffi::Tensor permuted_tokens,       // [total, H]
                          int T, int TOP_K, int H) {
    launch_token_dispatch(
        static_cast<const __nv_fp8_storage_t*>(hidden_states_fp8.data_ptr()),
        static_cast<const float*>(hidden_states_scale.data_ptr()),
        static_cast<const int*>(token_expert_indices.data_ptr()),
        static_cast<const int*>(token_expert_slots.data_ptr()),
        static_cast<const int*>(expert_offsets.data_ptr()),
        static_cast<float*>(permuted_tokens.data_ptr()),
        T, TOP_K, H
    );
}

static auto _dispatch = ffi::reflection::GlobalDef().def("dispatch_ffi", dispatch_ffi_wrapper);

// ─── Reindex / Merge Weights FFI ────────────────────────────────────────────

void reindex_ffi_wrapper(ffi::Tensor token_expert_indices,   // [T, TOP_K]
                         ffi::Tensor token_expert_weights,   // [T, TOP_K]
                         ffi::Tensor token_expert_slots,     // [T, TOP_K]
                         ffi::Tensor expert_token_offsets,   // [E_LOCAL + 1]
                         ffi::Tensor token_indices,          // [T * TOP_K] capacity
                         ffi::Tensor local_expert_ids,       // [T * TOP_K] capacity
                         ffi::Tensor merged_token_weights,   // [T * TOP_K] capacity
                         int seq_len,
                         int local_expert_offset) {
    launch_moe_reindex(
        static_cast<const int*>(token_expert_indices.data_ptr()),
        static_cast<const float*>(token_expert_weights.data_ptr()),
        static_cast<const int*>(token_expert_slots.data_ptr()),
        static_cast<const int*>(expert_token_offsets.data_ptr()),
        static_cast<int*>(token_indices.data_ptr()),
        static_cast<int*>(local_expert_ids.data_ptr()),
        static_cast<float*>(merged_token_weights.data_ptr()),
        seq_len,
        local_expert_offset
    );
}

static auto _reindex = ffi::reflection::GlobalDef().def("reindex_ffi", reindex_ffi_wrapper);

// ─── Kernel4 FFI ─────────────────────────────────────────────────────────────

void kernel4_ffi_wrapper(ffi::Tensor hidden_states,
                         ffi::Tensor hidden_states_scale,
                         ffi::Tensor gemm1_weights,
                         ffi::Tensor gemm1_weights_scale,
                         ffi::Tensor gemm2_weights,
                         ffi::Tensor gemm2_weights_scale,
                         ffi::Tensor expert_token_offsets,
                         ffi::Tensor token_indices,
                         ffi::Tensor token_expert_weights,
                         ffi::Tensor output,
                         int seq_len, int local_expert_offset, float routed_scaling_factor) {
    int total_tok = choose_total_tokens(expert_token_offsets);

    size_t workspace_bytes = k4_query_workspace(seq_len, total_tok, 0);
    void* d_workspace = k4_workspace_cache.reserve(workspace_bytes);
    if (workspace_bytes > 0 && !d_workspace) {
        fprintf(stderr, "Failed to reserve kernel4 workspace (%zu bytes)\n", workspace_bytes);
        return;
    }

    Kernel4Workspace workspace = k4_bind_workspace(d_workspace, workspace_bytes, seq_len, total_tok, 0);

    Kernel4Problem problem{};
    problem.routing_logits = nullptr;
    problem.routing_bias = nullptr;
    problem.seq_len = seq_len;
    problem.hidden_states = static_cast<const fp8_e4m3*>(hidden_states.data_ptr());
    problem.hidden_states_scale = static_cast<const float*>(hidden_states_scale.data_ptr());
    problem.gemm1_weights = static_cast<const fp8_e4m3*>(gemm1_weights.data_ptr());
    problem.gemm1_weights_scale = static_cast<const float*>(gemm1_weights_scale.data_ptr());
    problem.gemm2_weights = static_cast<const fp8_e4m3*>(gemm2_weights.data_ptr());
    problem.gemm2_weights_scale = static_cast<const float*>(gemm2_weights_scale.data_ptr());
    problem.local_expert_offset = local_expert_offset;
    problem.routed_scaling_factor = routed_scaling_factor;
    problem.expert_token_offsets = static_cast<const int*>(expert_token_offsets.data_ptr());
    problem.token_indices = static_cast<const int*>(token_indices.data_ptr());
    problem.local_expert_ids = nullptr;
    problem.token_expert_weights = static_cast<const float*>(token_expert_weights.data_ptr());
    problem.output = static_cast<__nv_bfloat16*>(output.data_ptr());
    problem.backend = choose_kernel4_backend_policy(
        seq_len,
        total_tok,
        /*has_local_expert_ids=*/false
    );
    problem.stream = nullptr;

    k4_launch(problem, workspace);
}

static auto _kernel4 = ffi::reflection::GlobalDef().def("kernel4_ffi", kernel4_ffi_wrapper);

// ─── GEMM1+SwiGLU FFI ───────────────────────────────────────────────────────

void gemm1_swiglu_ffi_wrapper(ffi::Tensor hidden_states,
                              ffi::Tensor hidden_states_scale,
                              ffi::Tensor gemm1_weights,
                              ffi::Tensor gemm1_weights_scale,
                              ffi::Tensor expert_token_offsets,
                              ffi::Tensor token_indices,
                              ffi::Tensor local_expert_ids,
                              ffi::Tensor output,
                              int seq_len, int local_expert_offset) {
    int total_tok = choose_total_tokens(expert_token_offsets);

    size_t workspace_bytes = k4_query_workspace(seq_len, total_tok, 0);
    void* d_workspace = k4_workspace_cache.reserve(workspace_bytes);
    if (workspace_bytes > 0 && !d_workspace) {
        fprintf(stderr, "Failed to reserve kernel4 workspace (%zu bytes)\n", workspace_bytes);
        return;
    }

    Kernel4Workspace workspace = k4_bind_workspace(d_workspace, workspace_bytes, seq_len, total_tok, 0);

    Kernel4Problem problem{};
    problem.seq_len = seq_len;
    problem.hidden_states = static_cast<const fp8_e4m3*>(hidden_states.data_ptr());
    problem.hidden_states_scale = static_cast<const float*>(hidden_states_scale.data_ptr());
    problem.gemm1_weights = static_cast<const fp8_e4m3*>(gemm1_weights.data_ptr());
    problem.gemm1_weights_scale = static_cast<const float*>(gemm1_weights_scale.data_ptr());
    problem.local_expert_offset = local_expert_offset;
    problem.expert_token_offsets = static_cast<const int*>(expert_token_offsets.data_ptr());
    problem.token_indices = static_cast<const int*>(token_indices.data_ptr());
    problem.local_expert_ids = static_cast<const int*>(local_expert_ids.data_ptr());
    problem.output = nullptr; // Not used for gemm1 call
    problem.backend = choose_kernel4_backend_policy(
        seq_len,
        total_tok,
        /*has_local_expert_ids=*/true
    );
    problem.stream = nullptr;

    k4_launch_gemm1(problem, workspace);

    // Copy intermediate output from workspace to the provided output tensor
    cudaMemcpy(output.data_ptr(), workspace.gemm1_output, 
               (size_t)total_tok * moe_spec::INTERMEDIATE_SIZE * sizeof(__nv_bfloat16), 
               cudaMemcpyDeviceToDevice);
}

static auto _gemm1_swiglu = ffi::reflection::GlobalDef().def("gemm1_swiglu_ffi", gemm1_swiglu_ffi_wrapper);

// ─── Kernel6 FFI ─────────────────────────────────────────────────────────────

void kernel6_ffi_wrapper(ffi::Tensor hidden_states,
                         ffi::Tensor gemm2_weights,
                         ffi::Tensor gemm2_weights_scale,
                         ffi::Tensor expert_token_offsets,
                         ffi::Tensor token_indices,
                         ffi::Tensor local_expert_ids,
                         ffi::Tensor token_expert_weights,
                         ffi::Tensor output,
                         int seq_len, int local_expert_offset, float routed_scaling_factor) {
    int total_tok = choose_total_tokens(expert_token_offsets);

    size_t workspace_bytes = k6_query_workspace(seq_len, total_tok, 0);
    void* d_workspace = k6_workspace_cache.reserve(workspace_bytes);
    if (workspace_bytes > 0 && !d_workspace) {
        fprintf(stderr, "Failed to reserve kernel6 workspace (%zu bytes)\n", workspace_bytes);
        return;
    }

    Kernel6Workspace workspace = k6_bind_workspace(d_workspace, workspace_bytes, seq_len, total_tok, 0);

    Kernel6Problem problem{};
    problem.hidden_states = static_cast<const __nv_bfloat16*>(hidden_states.data_ptr());
    problem.seq_len = seq_len;
    problem.gemm2_weights = static_cast<const fp8_e4m3*>(gemm2_weights.data_ptr());
    problem.gemm2_weights_scale = static_cast<const float*>(gemm2_weights_scale.data_ptr());
    problem.local_expert_offset = local_expert_offset;
    problem.routed_scaling_factor = routed_scaling_factor;
    problem.expert_token_offsets = static_cast<const int*>(expert_token_offsets.data_ptr());
    problem.token_indices = static_cast<const int*>(token_indices.data_ptr());
    problem.local_expert_ids = static_cast<const int*>(local_expert_ids.data_ptr());
    problem.token_expert_weights = static_cast<const float*>(token_expert_weights.data_ptr());
    problem.output = static_cast<__nv_bfloat16*>(output.data_ptr());
    problem.backend = choose_kernel6_backend_policy(seq_len, total_tok);
    problem.stream = nullptr;

    k6_launch(problem, workspace);
}

static auto _kernel6 = ffi::reflection::GlobalDef().def("kernel6_ffi", kernel6_ffi_wrapper);

// ─── MoE Integration Function ──────────────────────────────────────────────────

// Expose a combined wrapper integrating the kernels
void moe_forward_ffi_wrapper(ffi::Tensor routing_logits,
                             ffi::Tensor routing_bias,
                             ffi::Tensor hidden_states,
                             ffi::Tensor hidden_states_scale,
                             ffi::Tensor gemm1_weights,
                             ffi::Tensor gemm1_weights_scale,
                             ffi::Tensor gemm2_weights,
                             ffi::Tensor gemm2_weights_scale,
                             ffi::Tensor expert_token_counts,
                             ffi::Tensor token_expert_indices,
                             ffi::Tensor token_expert_weights,
                             ffi::Tensor token_expert_slots,
                             ffi::Tensor expert_token_offsets,
                             ffi::Tensor token_indices,
                             ffi::Tensor merged_token_weights,
                             ffi::Tensor output,
                             int seq_len, int local_expert_offset, float routed_scaling_factor) {
    // 1. Route
    router_ffi_wrapper(routing_logits, routing_bias, expert_token_counts, 
                       token_expert_indices, token_expert_weights, token_expert_slots, 
                       seq_len, local_expert_offset, routed_scaling_factor);
    cudaDeviceSynchronize();
                       
    // 2. Scan (only for local experts)
    int* local_counts_ptr = static_cast<int*>(expert_token_counts.data_ptr()) + local_expert_offset;
    exclusive_scan_cub(local_counts_ptr, static_cast<int*>(expert_token_offsets.data_ptr()), moe_spec::NUM_LOCAL_EXPERTS);
    cudaDeviceSynchronize();

    // Compute total_tok for the 33rd offset
    int last_c, last_o;
    cudaMemcpy(&last_c, local_counts_ptr + moe_spec::NUM_LOCAL_EXPERTS - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_o, static_cast<int*>(expert_token_offsets.data_ptr()) + moe_spec::NUM_LOCAL_EXPERTS - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int total_tok = last_c + last_o;
    cudaMemcpy(static_cast<int*>(expert_token_offsets.data_ptr()) + moe_spec::NUM_LOCAL_EXPERTS, &total_tok, sizeof(int), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    size_t k4_ws_bytes = k4_query_workspace(seq_len, total_tok, 0);

    void* d_k4_ws = nullptr;
    if (k4_ws_bytes > 0) {
        if (cudaMalloc(&d_k4_ws, k4_ws_bytes) != cudaSuccess) {
            printf("CUDA malloc failed for k4_ws\n");
            return;
        }
    }
    
    Kernel4Workspace k4_ws = k4_bind_workspace(d_k4_ws, k4_ws_bytes, seq_len, total_tok, 0);


    
    Kernel4Problem k4_p{};
    k4_p.seq_len = seq_len;
    k4_p.hidden_states = static_cast<const fp8_e4m3*>(hidden_states.data_ptr());
    k4_p.hidden_states_scale = static_cast<const float*>(hidden_states_scale.data_ptr());
    k4_p.gemm1_weights = static_cast<const fp8_e4m3*>(gemm1_weights.data_ptr());
    k4_p.gemm1_weights_scale = static_cast<const float*>(gemm1_weights_scale.data_ptr());
    k4_p.local_expert_offset = local_expert_offset;
    k4_p.expert_token_offsets = static_cast<const int*>(expert_token_offsets.data_ptr());
    k4_p.token_indices = static_cast<const int*>(token_indices.data_ptr());
    k4_p.backend = Kernel4Backend::Auto;
    
    k4_launch_gemm1(k4_p, k4_ws);
    cudaDeviceSynchronize();
    
    // 4. Compute Expert MLP Part 2: GEMM2 + Accumulation (Kernel 6)

    size_t k6_ws_bytes = k6_query_workspace(seq_len, total_tok, 0);
    void* d_k6_ws = nullptr;
    if (k6_ws_bytes > 0) {
        if (cudaMalloc(&d_k6_ws, k6_ws_bytes) != cudaSuccess) {
            printf("CUDA malloc failed for k6_ws\n");
            if (d_k4_ws) cudaFree(d_k4_ws);
            return;
        }
    }
    
    Kernel6Workspace k6_ws = k6_bind_workspace(d_k6_ws, k6_ws_bytes, seq_len, total_tok, 0);

    
    Kernel6Problem k6_p{};
    k6_p.hidden_states = k4_ws.gemm1_output;
    k6_p.seq_len = seq_len;
    k6_p.gemm2_weights = static_cast<const fp8_e4m3*>(gemm2_weights.data_ptr());
    k6_p.gemm2_weights_scale = static_cast<const float*>(gemm2_weights_scale.data_ptr());
    k6_p.local_expert_offset = local_expert_offset;
    k6_p.routed_scaling_factor = routed_scaling_factor;
    k6_p.expert_token_offsets = static_cast<const int*>(expert_token_offsets.data_ptr());
    k6_p.token_indices = static_cast<const int*>(token_indices.data_ptr());
    k6_p.token_expert_weights = static_cast<const float*>(merged_token_weights.data_ptr());
    k6_p.output = static_cast<__nv_bfloat16*>(output.data_ptr());
    k6_p.backend = Kernel6Backend::Auto;
    
    k6_launch(k6_p, k6_ws);
    cudaDeviceSynchronize();
    
    if (cudaGetLastError() != cudaSuccess) {
        printf("Kernel 6 failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    }


}

static auto _moe_forward = ffi::reflection::GlobalDef().def("moe_forward_ffi", moe_forward_ffi_wrapper);
