#include "kernel6_internal.cuh"

#if defined(K4_ENABLE_CUTLASS)

#include <algorithm>
#include <vector>

#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

namespace kernel6_internal {

// Bump from 2 → NUM_LOCAL_EXPERTS so a typical workload (≤32 active experts)
// fits in a SINGLE grouped GEMM call.
constexpr int kCutlassGroupedBatchExperts = NUM_LOCAL_EXPERTS;

// BF16 Tensor-Op grouped GEMM (see kernel4_cutlass.cu for the rationale).
// hidden_states arrive in __nv_bfloat16 already, so we feed them straight
// into the GEMM as A without any FP32 conversion - eliminating both the
// inter_f32 scratch buffer and the bf16_rows_to_f32_kernel pass that the
// previous SIMT FP32 path required.
using CutlassElementA = cutlass::bfloat16_t;
using CutlassElementB = cutlass::bfloat16_t;
using CutlassElementC = float;
using CutlassElementAccumulator = float;

constexpr int kCutlassAlignA = 8;
constexpr int kCutlassAlignB = 8;

using CutlassGroupedConfig = cutlass::gemm::device::DefaultGemmConfiguration<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    CutlassElementA,
    CutlassElementB,
    CutlassElementC,
    CutlassElementAccumulator>;

using CutlassGroupedKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    CutlassElementA,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    kCutlassAlignA,
    CutlassElementB,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    kCutlassAlignB,
    CutlassElementC,
    cutlass::layout::RowMajor,
    CutlassElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    typename CutlassGroupedConfig::ThreadblockShape,
    typename CutlassGroupedConfig::WarpShape,
    typename CutlassGroupedConfig::InstructionShape,
    typename CutlassGroupedConfig::EpilogueOutputOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    CutlassGroupedConfig::kStages,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

using CutlassGroupedGemm = cutlass::gemm::device::GemmGrouped<CutlassGroupedKernel>;
using CutlassGroupedStrideA = typename CutlassGroupedGemm::LayoutA::Stride::LongIndex;
using CutlassGroupedStrideB = typename CutlassGroupedGemm::LayoutB::Stride::LongIndex;
using CutlassGroupedStrideC = typename CutlassGroupedGemm::LayoutC::Stride::LongIndex;

struct CutlassScratchView {
    // No more inter_f32 - hidden_states are already BF16.
    CutlassElementB* w2_dequant;       // BF16 dequantized weights
    cutlass::gemm::GemmCoord* problem_sizes;
    CutlassElementA** ptr_A;
    CutlassElementB** ptr_B;
    CutlassElementC** ptr_C;
    CutlassElementC** ptr_D;
    CutlassGroupedStrideA* lda;
    CutlassGroupedStrideB* ldb;
    CutlassGroupedStrideC* ldc;
    CutlassGroupedStrideC* ldd;
    void* gemm_workspace;
};

size_t cutlass_aux_bytes(int /*total_dispatched_tokens*/) {
    size_t total = 0;
    // BF16 weight scratch (was FP32 before; halved).
    total += align_up((size_t)kCutlassGroupedBatchExperts * HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(CutlassElementB));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(cutlass::gemm::GemmCoord));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassElementA*));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassElementB*));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassElementC*));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassElementC*));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideA));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideB));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideC));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideC));
    return total;
}

static CutlassScratchView bind_cutlass_scratch(void* storage,
                                               size_t storage_bytes,
                                               int /*total_dispatched_tokens*/) {
    CutlassScratchView view{};
    if (!storage) {
        return view;
    }

    uintptr_t base = reinterpret_cast<uintptr_t>(storage);
    uintptr_t cursor = align_up(base);

    view.w2_dequant = reinterpret_cast<CutlassElementB*>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * HIDDEN_SIZE * INTERMEDIATE_SIZE * sizeof(CutlassElementB));

    view.problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(cutlass::gemm::GemmCoord));

    view.ptr_A = reinterpret_cast<CutlassElementA**>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassElementA*));

    view.ptr_B = reinterpret_cast<CutlassElementB**>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassElementB*));

    view.ptr_C = reinterpret_cast<CutlassElementC**>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassElementC*));

    view.ptr_D = reinterpret_cast<CutlassElementC**>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassElementC*));

    view.lda = reinterpret_cast<CutlassGroupedStrideA*>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideA));

    view.ldb = reinterpret_cast<CutlassGroupedStrideB*>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideB));

    view.ldc = reinterpret_cast<CutlassGroupedStrideC*>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideC));

    view.ldd = reinterpret_cast<CutlassGroupedStrideC*>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideC));

    view.gemm_workspace = reinterpret_cast<void*>(cursor);
    if (cursor - base > storage_bytes) {
        return CutlassScratchView{};
    }
    return view;
}

static cudaError_t cutlass_status_to_cuda_error(cutlass::Status status) {
    switch (status) {
        case cutlass::Status::kSuccess:
            return cudaSuccess;
        case cutlass::Status::kErrorNotSupported:
        case cutlass::Status::kErrorArchMismatch:
            return cudaErrorNotSupported;
        case cutlass::Status::kErrorMemoryAllocation:
            return cudaErrorMemoryAllocation;
        case cutlass::Status::kErrorWorkspaceNull:
        case cutlass::Status::kErrorInvalidProblem:
        case cutlass::Status::kErrorMisalignedOperand:
        case cutlass::Status::kErrorInvalidLayout:
        case cutlass::Status::kErrorInvalidDataType:
            return cudaErrorInvalidValue;
        case cutlass::Status::kErrorInsufficientDriver:
            return cudaErrorInsufficientDriver;
        case cutlass::Status::kErrorInternal:
        case cutlass::Status::kInvalid:
            return cudaErrorUnknown;
    }
    return cudaErrorUnknown;
}

static cudaError_t run_cutlass_grouped_bf16_gemm(
    CutlassScratchView const& scratch,
    std::vector<cutlass::gemm::GemmCoord> const& host_problem_sizes,
    std::vector<CutlassElementA*> const& host_ptr_A,
    std::vector<CutlassElementB*> const& host_ptr_B,
    std::vector<CutlassElementC*> const& host_ptr_D,
    std::vector<CutlassGroupedStrideA> const& host_lda,
    std::vector<CutlassGroupedStrideB> const& host_ldb,
    std::vector<CutlassGroupedStrideC> const& host_ldd,
    cudaStream_t stream) {
    int problem_count = static_cast<int>(host_problem_sizes.size());
    if (problem_count == 0) {
        return cudaSuccess;
    }

    thread_local std::vector<CutlassElementC*> host_ptr_C;
    thread_local std::vector<CutlassGroupedStrideC> host_ldc;
    host_ptr_C.assign(host_ptr_D.begin(), host_ptr_D.end());
    host_ldc.assign(host_ldd.begin(), host_ldd.end());

    K6_CUDA_CHECK(cudaMemcpyAsync(
        scratch.problem_sizes,
        host_problem_sizes.data(),
        problem_count * sizeof(cutlass::gemm::GemmCoord),
        cudaMemcpyHostToDevice,
        stream));
    K6_CUDA_CHECK(cudaMemcpyAsync(
        scratch.ptr_A,
        host_ptr_A.data(),
        problem_count * sizeof(CutlassElementA*),
        cudaMemcpyHostToDevice,
        stream));
    K6_CUDA_CHECK(cudaMemcpyAsync(
        scratch.ptr_B,
        host_ptr_B.data(),
        problem_count * sizeof(CutlassElementB*),
        cudaMemcpyHostToDevice,
        stream));
    K6_CUDA_CHECK(cudaMemcpyAsync(
        scratch.ptr_C,
        host_ptr_C.data(),
        problem_count * sizeof(CutlassElementC*),
        cudaMemcpyHostToDevice,
        stream));
    K6_CUDA_CHECK(cudaMemcpyAsync(
        scratch.ptr_D,
        host_ptr_D.data(),
        problem_count * sizeof(CutlassElementC*),
        cudaMemcpyHostToDevice,
        stream));
    K6_CUDA_CHECK(cudaMemcpyAsync(
        scratch.lda,
        host_lda.data(),
        problem_count * sizeof(CutlassGroupedStrideA),
        cudaMemcpyHostToDevice,
        stream));
    K6_CUDA_CHECK(cudaMemcpyAsync(
        scratch.ldb,
        host_ldb.data(),
        problem_count * sizeof(CutlassGroupedStrideB),
        cudaMemcpyHostToDevice,
        stream));
    K6_CUDA_CHECK(cudaMemcpyAsync(
        scratch.ldc,
        host_ldc.data(),
        problem_count * sizeof(CutlassGroupedStrideC),
        cudaMemcpyHostToDevice,
        stream));
    K6_CUDA_CHECK(cudaMemcpyAsync(
        scratch.ldd,
        host_ldd.data(),
        problem_count * sizeof(CutlassGroupedStrideC),
        cudaMemcpyHostToDevice,
        stream));

    CutlassGroupedGemm gemm_op;
    int threadblock_count = CutlassGroupedGemm::sufficient(
        host_problem_sizes.data(), problem_count);
    if (!threadblock_count) {
        return cudaErrorNotSupported;
    }

    typename CutlassGroupedGemm::EpilogueOutputOp::Params epilogue_op(1.0f, 0.0f);
    typename CutlassGroupedGemm::Arguments args(
        scratch.problem_sizes,
        problem_count,
        threadblock_count,
        epilogue_op,
        scratch.ptr_A,
        scratch.ptr_B,
        scratch.ptr_C,
        scratch.ptr_D,
        scratch.lda,
        scratch.ldb,
        scratch.ldc,
        scratch.ldd,
        const_cast<cutlass::gemm::GemmCoord*>(host_problem_sizes.data()));

    cutlass::Status status = gemm_op(args, scratch.gemm_workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS grouped GEMM failed: %s\n", cutlass::cutlassGetStatusString(status));
    }
    return cutlass_status_to_cuda_error(status);
}

bool current_device_is_sm86_or_better() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return false;
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return false;
    }
    return prop.major > 8 || (prop.major == 8 && prop.minor >= 6);
}

cudaError_t launch_cutlass_gemm2_combine(const Gemm2Problem& p,
                                         const Gemm2Workspace& workspace,
                                         int total_tokens) {
    if (!workspace.cutlass_workspace) {
        return cudaErrorInvalidValue;
    }

    // Use the caller-provided host mirror when available so we skip the
    // synchronous cudaMemcpy that would otherwise stall the launch pipeline.
    int local_host_offsets[NUM_LOCAL_EXPERTS + 1];
    const int* host_offsets = p.host_expert_token_offsets;
    if (!host_offsets) {
        K6_CUDA_CHECK(cudaMemcpy(local_host_offsets,
                                 p.expert_token_offsets,
                                 sizeof(local_host_offsets),
                                 cudaMemcpyDeviceToHost));
        host_offsets = local_host_offsets;
    }

    CutlassScratchView scratch = bind_cutlass_scratch(
        workspace.cutlass_workspace,
        workspace.cutlass_workspace_bytes,
        total_tokens);
    if (!scratch.w2_dequant ||
        !scratch.problem_sizes || !scratch.ptr_A || !scratch.ptr_B ||
        !scratch.ptr_C || !scratch.ptr_D || !scratch.lda || !scratch.ldb ||
        !scratch.ldc || !scratch.ldd) {
        return cudaErrorInvalidValue;
    }

    K6_CUDA_CHECK(cudaMemsetAsync(
        workspace.output_accum,
        0,
        output_accum_bytes(p.seq_len),
        p.stream));

    constexpr int threads = 256;

    // hidden_states is already __nv_bfloat16 in expert-grouped order, so we
    // can point ptr_A directly at it. No bf16->fp32 conversion pass needed.
    CutlassElementA* a_base = reinterpret_cast<CutlassElementA*>(
        const_cast<__nv_bfloat16*>(p.hidden_states));

    struct ActiveExpert {
        int expert;
        int begin;
        int token_count;
    };
    // thread_local: keep the heap allocation across calls. clear() preserves
    // capacity, so after warm-up these never re-allocate.
    thread_local std::vector<ActiveExpert> active_experts;
    active_experts.clear();
    active_experts.reserve(NUM_LOCAL_EXPERTS);
    for (int expert = 0; expert < NUM_LOCAL_EXPERTS; ++expert) {
        int begin = host_offsets[expert];
        int end = host_offsets[expert + 1];
        if (end > begin) {
            active_experts.push_back({expert, begin, end - begin});
        }
    }

    thread_local std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    thread_local std::vector<CutlassElementA*> ptr_A;
    thread_local std::vector<CutlassElementB*> ptr_B;
    thread_local std::vector<CutlassElementC*> ptr_D;
    thread_local std::vector<CutlassGroupedStrideA> lda;
    thread_local std::vector<CutlassGroupedStrideB> ldb;
    thread_local std::vector<CutlassGroupedStrideC> ldd;
    problem_sizes.reserve(kCutlassGroupedBatchExperts);
    ptr_A.reserve(kCutlassGroupedBatchExperts);
    ptr_B.reserve(kCutlassGroupedBatchExperts);
    ptr_D.reserve(kCutlassGroupedBatchExperts);
    lda.reserve(kCutlassGroupedBatchExperts);
    ldb.reserve(kCutlassGroupedBatchExperts);
    ldd.reserve(kCutlassGroupedBatchExperts);

    int gemm2_weight_total = HIDDEN_SIZE * INTERMEDIATE_SIZE;
    for (size_t batch_start = 0; batch_start < active_experts.size(); batch_start += kCutlassGroupedBatchExperts) {
        size_t batch_count = std::min<size_t>(kCutlassGroupedBatchExperts, active_experts.size() - batch_start);
        problem_sizes.clear();
        ptr_A.clear();
        ptr_B.clear();
        ptr_D.clear();
        lda.clear();
        ldb.clear();
        ldd.clear();

        for (size_t i = 0; i < batch_count; ++i) {
            const ActiveExpert& info = active_experts[batch_start + i];
            const fp8_e4m3* w2_e = p.gemm2_weights +
                (size_t)info.expert * HIDDEN_SIZE * INTERMEDIATE_SIZE;
            const float* w2s_e = p.gemm2_weights_scale +
                (size_t)info.expert * NUM_HIDDEN_BLOCKS * NUM_INTER_BLOCKS;

            CutlassElementB* w2_slot = scratch.w2_dequant + i * (size_t)HIDDEN_SIZE * INTERMEDIATE_SIZE;
            dequant_gemm2_weight_bf16_kernel<<<(gemm2_weight_total + threads - 1) / threads, threads, 0, p.stream>>>(
                w2_e,
                w2s_e,
                reinterpret_cast<__nv_bfloat16*>(w2_slot)
            );
            K6_CUDA_CHECK(cudaGetLastError());

            problem_sizes.push_back({info.token_count, HIDDEN_SIZE, INTERMEDIATE_SIZE});
            ptr_A.push_back(a_base + (size_t)info.begin * INTERMEDIATE_SIZE);
            ptr_B.push_back(w2_slot);
            ptr_D.push_back(workspace.gemm2_output + (size_t)info.begin * HIDDEN_SIZE);
            lda.push_back(INTERMEDIATE_SIZE);
            ldb.push_back(INTERMEDIATE_SIZE);
            ldd.push_back(HIDDEN_SIZE);
        }

        K6_CUDA_CHECK(run_cutlass_grouped_bf16_gemm(
            scratch, problem_sizes, ptr_A, ptr_B, ptr_D, lda, ldb, ldd, p.stream));
    }

    dim3 block(256);
    dim3 combine_grid(total_tokens, (HIDDEN_SIZE + block.x - 1) / block.x);
    combine_projected_kernel<<<combine_grid, block, 0, p.stream>>>(
        workspace.gemm2_output,
        p.token_indices,
        p.token_expert_weights,
        p.routed_scaling_factor,
        workspace.output_accum,
        total_tokens,
        p.seq_len
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

#endif
