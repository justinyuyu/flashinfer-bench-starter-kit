#include "kernel4_internal.cuh"

#if defined(K4_ENABLE_CUTLASS)

#include <algorithm>
#include <vector>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/default_gemm_configuration.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

namespace kernel4_internal {

constexpr int kCutlassGroupedBatchExperts = 2;

using CutlassGroupedConfig = cutlass::gemm::device::DefaultGemmConfiguration<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    float,
    float,
    float,
    float>;

using CutlassGroupedKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float,
    cutlass::layout::RowMajor,
    cutlass::ComplexTransform::kNone,
    1,
    float,
    cutlass::layout::ColumnMajor,
    cutlass::ComplexTransform::kNone,
    1,
    float,
    cutlass::layout::RowMajor,
    float,
    cutlass::arch::OpClassSimt,
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
    float* a1_dequant;
    float* gemm1_up;
    float* gemm1_gate;
    float* w1_up;
    float* w1_gate;
    cutlass::gemm::GemmCoord* problem_sizes;
    float** ptr_A;
    float** ptr_B;
    float** ptr_C;
    float** ptr_D;
    CutlassGroupedStrideA* lda;
    CutlassGroupedStrideB* ldb;
    CutlassGroupedStrideC* ldc;
    CutlassGroupedStrideC* ldd;
    void* gemm_workspace;
};

static size_t gemm1_cutlass_aux_bytes(int total_dispatched_tokens) {
    size_t total = 0;
    total += align_up((size_t)total_dispatched_tokens * HIDDEN_SIZE * sizeof(float));
    total += align_up((size_t)total_dispatched_tokens * INTERMEDIATE_SIZE * sizeof(float));
    total += align_up((size_t)total_dispatched_tokens * INTERMEDIATE_SIZE * sizeof(float));
    total += align_up((size_t)kCutlassGroupedBatchExperts * INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(float));
    total += align_up((size_t)kCutlassGroupedBatchExperts * INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(float));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(cutlass::gemm::GemmCoord));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(float*));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(float*));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(float*));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(float*));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideA));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideB));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideC));
    total += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(CutlassGroupedStrideC));
    return total;
}

size_t cutlass_aux_bytes(int total_dispatched_tokens) {
    return gemm1_cutlass_aux_bytes(total_dispatched_tokens) +
        kernel6_internal::cutlass_aux_bytes(total_dispatched_tokens);
}

static CutlassScratchView bind_cutlass_scratch(void* storage,
                                               size_t storage_bytes,
                                               int total_dispatched_tokens) {
    CutlassScratchView view{};
    if (!storage) {
        return view;
    }

    uintptr_t base = reinterpret_cast<uintptr_t>(storage);
    uintptr_t cursor = align_up(base);

    view.a1_dequant = reinterpret_cast<float*>(cursor);
    cursor += align_up((size_t)total_dispatched_tokens * HIDDEN_SIZE * sizeof(float));

    view.gemm1_up = reinterpret_cast<float*>(cursor);
    cursor += align_up((size_t)total_dispatched_tokens * INTERMEDIATE_SIZE * sizeof(float));

    view.gemm1_gate = reinterpret_cast<float*>(cursor);
    cursor += align_up((size_t)total_dispatched_tokens * INTERMEDIATE_SIZE * sizeof(float));

    view.w1_up = reinterpret_cast<float*>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(float));

    view.w1_gate = reinterpret_cast<float*>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * INTERMEDIATE_SIZE * HIDDEN_SIZE * sizeof(float));

    view.problem_sizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(cutlass::gemm::GemmCoord));

    view.ptr_A = reinterpret_cast<float**>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(float*));

    view.ptr_B = reinterpret_cast<float**>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(float*));

    view.ptr_C = reinterpret_cast<float**>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(float*));

    view.ptr_D = reinterpret_cast<float**>(cursor);
    cursor += align_up((size_t)kCutlassGroupedBatchExperts * sizeof(float*));

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

static cudaError_t run_cutlass_grouped_float_gemm(
    CutlassScratchView const& scratch,
    std::vector<cutlass::gemm::GemmCoord> const& host_problem_sizes,
    std::vector<float*> const& host_ptr_A,
    std::vector<float*> const& host_ptr_B,
    std::vector<float*> const& host_ptr_D,
    std::vector<CutlassGroupedStrideA> const& host_lda,
    std::vector<CutlassGroupedStrideB> const& host_ldb,
    std::vector<CutlassGroupedStrideC> const& host_ldd,
    cudaStream_t stream) {
    int problem_count = static_cast<int>(host_problem_sizes.size());
    if (problem_count == 0) {
        return cudaSuccess;
    }

    std::vector<float*> host_ptr_C = host_ptr_D;
    std::vector<CutlassGroupedStrideC> host_ldc = host_ldd;

    CUDA_CHECK(cudaMemcpyAsync(
        scratch.problem_sizes,
        host_problem_sizes.data(),
        problem_count * sizeof(cutlass::gemm::GemmCoord),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        scratch.ptr_A,
        host_ptr_A.data(),
        problem_count * sizeof(float*),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        scratch.ptr_B,
        host_ptr_B.data(),
        problem_count * sizeof(float*),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        scratch.ptr_C,
        host_ptr_C.data(),
        problem_count * sizeof(float*),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        scratch.ptr_D,
        host_ptr_D.data(),
        problem_count * sizeof(float*),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        scratch.lda,
        host_lda.data(),
        problem_count * sizeof(CutlassGroupedStrideA),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        scratch.ldb,
        host_ldb.data(),
        problem_count * sizeof(CutlassGroupedStrideB),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
        scratch.ldc,
        host_ldc.data(),
        problem_count * sizeof(CutlassGroupedStrideC),
        cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(
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

cudaError_t launch_cutlass_backend(const Kernel4Problem& p,
                                   const Kernel4Workspace& workspace,
                                   int total_tokens,
                                   bool gemm1_only) {
    if (!workspace.cutlass_workspace) {
        return cudaErrorInvalidValue;
    }

    std::vector<int> host_offsets(NUM_LOCAL_EXPERTS + 1, 0);
    CUDA_CHECK(cudaMemcpy(host_offsets.data(),
                          p.expert_token_offsets,
                          host_offsets.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));

    CutlassScratchView scratch = bind_cutlass_scratch(
        workspace.cutlass_workspace,
        gemm1_cutlass_aux_bytes(total_tokens),
        total_tokens);
    if (!scratch.a1_dequant || !scratch.gemm1_up || !scratch.gemm1_gate ||
        !scratch.w1_up || !scratch.w1_gate ||
        !scratch.problem_sizes || !scratch.ptr_A || !scratch.ptr_B ||
        !scratch.ptr_C || !scratch.ptr_D || !scratch.lda || !scratch.ldb ||
        !scratch.ldc || !scratch.ldd) {
        return cudaErrorInvalidValue;
    }

    constexpr int threads = 256;
    int act_total_all = total_tokens * HIDDEN_SIZE;
    dequant_activations_kernel<<<(act_total_all + threads - 1) / threads, threads, 0, p.stream>>>(
        p.hidden_states,
        p.hidden_states_scale,
        p.token_indices,
        0,
        total_tokens,
        p.seq_len,
        scratch.a1_dequant
    );
    CUDA_CHECK(cudaGetLastError());

    struct ActiveExpert {
        int expert;
        int begin;
        int token_count;
    };
    std::vector<ActiveExpert> active_experts;
    for (int expert = 0; expert < NUM_LOCAL_EXPERTS; ++expert) {
        int begin = host_offsets[expert];
        int end = host_offsets[expert + 1];
        if (end > begin) {
            active_experts.push_back({expert, begin, end - begin});
        }
    }

    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<float*> ptr_A;
    std::vector<float*> ptr_B;
    std::vector<float*> ptr_D;
    std::vector<CutlassGroupedStrideA> lda;
    std::vector<CutlassGroupedStrideB> ldb;
    std::vector<CutlassGroupedStrideC> ldd;
    problem_sizes.reserve(kCutlassGroupedBatchExperts);
    ptr_A.reserve(kCutlassGroupedBatchExperts);
    ptr_B.reserve(kCutlassGroupedBatchExperts);
    ptr_D.reserve(kCutlassGroupedBatchExperts);
    lda.reserve(kCutlassGroupedBatchExperts);
    ldb.reserve(kCutlassGroupedBatchExperts);
    ldd.reserve(kCutlassGroupedBatchExperts);

    for (size_t batch_start = 0; batch_start < active_experts.size(); batch_start += kCutlassGroupedBatchExperts) {
        size_t batch_count = std::min<size_t>(kCutlassGroupedBatchExperts, active_experts.size() - batch_start);
        problem_sizes.clear();
        ptr_A.clear();
        ptr_B.clear();
        ptr_D.clear();
        lda.clear();
        ldb.clear();
        ldd.clear();

        int gemm1_weight_total = INTERMEDIATE_SIZE * HIDDEN_SIZE;
        for (size_t i = 0; i < batch_count; ++i) {
            const ActiveExpert& info = active_experts[batch_start + i];
            const fp8_e4m3* w1_e = p.gemm1_weights +
                (size_t)info.expert * GEMM1_OUT_SIZE * HIDDEN_SIZE;
            const float* w1s_e = p.gemm1_weights_scale +
                (size_t)info.expert * NUM_GEMM1_OUT_BLOCKS * NUM_HIDDEN_BLOCKS;

            float* w1_up_slot = scratch.w1_up + i * (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE;
            float* w1_gate_slot = scratch.w1_gate + i * (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE;

            dequant_gemm1_weight_half_kernel<<<(gemm1_weight_total + threads - 1) / threads, threads, 0, p.stream>>>(
                w1_e,
                w1s_e,
                0,
                w1_up_slot
            );
            CUDA_CHECK(cudaGetLastError());

            dequant_gemm1_weight_half_kernel<<<(gemm1_weight_total + threads - 1) / threads, threads, 0, p.stream>>>(
                w1_e,
                w1s_e,
                INTERMEDIATE_SIZE,
                w1_gate_slot
            );
            CUDA_CHECK(cudaGetLastError());

            problem_sizes.push_back({info.token_count, INTERMEDIATE_SIZE, HIDDEN_SIZE});
            ptr_A.push_back(scratch.a1_dequant + (size_t)info.begin * HIDDEN_SIZE);
            ptr_B.push_back(w1_up_slot);
            ptr_D.push_back(scratch.gemm1_up + (size_t)info.begin * INTERMEDIATE_SIZE);
            lda.push_back(HIDDEN_SIZE);
            ldb.push_back(HIDDEN_SIZE);
            ldd.push_back(INTERMEDIATE_SIZE);
        }

        CUDA_CHECK(run_cutlass_grouped_float_gemm(
            scratch, problem_sizes, ptr_A, ptr_B, ptr_D, lda, ldb, ldd, p.stream));

        for (size_t i = 0; i < batch_count; ++i) {
            ptr_B[i] = scratch.w1_gate + i * (size_t)INTERMEDIATE_SIZE * HIDDEN_SIZE;
            ptr_D[i] = scratch.gemm1_gate + (size_t)active_experts[batch_start + i].begin * INTERMEDIATE_SIZE;
        }

        CUDA_CHECK(run_cutlass_grouped_float_gemm(
            scratch, problem_sizes, ptr_A, ptr_B, ptr_D, lda, ldb, ldd, p.stream));

        for (size_t i = 0; i < batch_count; ++i) {
            const ActiveExpert& info = active_experts[batch_start + i];
            int inter_total = info.token_count * INTERMEDIATE_SIZE;
            swiglu_pack_kernel<<<(inter_total + threads - 1) / threads, threads, 0, p.stream>>>(
                scratch.gemm1_up + (size_t)info.begin * INTERMEDIATE_SIZE,
                scratch.gemm1_gate + (size_t)info.begin * INTERMEDIATE_SIZE,
                info.begin,
                info.token_count,
                workspace.gemm1_output
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }

    if (gemm1_only) {
        return cudaSuccess;
    }

    kernel6_internal::Gemm2Problem shared_problem{};
    shared_problem.hidden_states = workspace.gemm1_output;
    shared_problem.gemm2_weights = p.gemm2_weights;
    shared_problem.gemm2_weights_scale = p.gemm2_weights_scale;
    shared_problem.expert_token_offsets = p.expert_token_offsets;
    shared_problem.token_indices = p.token_indices;
    shared_problem.local_expert_ids = p.local_expert_ids;
    shared_problem.token_expert_weights = p.token_expert_weights;
    shared_problem.routed_scaling_factor = p.routed_scaling_factor;
    shared_problem.seq_len = p.seq_len;
    shared_problem.stream = p.stream;
    shared_problem.output = p.output;

    kernel6_internal::Gemm2Workspace shared_workspace{};
    shared_workspace.gemm2_output = workspace.gemm2_output;
    shared_workspace.output_accum = workspace.output_accum;
    shared_workspace.cutlass_workspace =
        reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(workspace.cutlass_workspace) +
                                gemm1_cutlass_aux_bytes(total_tokens));
    shared_workspace.cutlass_workspace_bytes =
        workspace.cutlass_workspace_bytes > gemm1_cutlass_aux_bytes(total_tokens)
            ? workspace.cutlass_workspace_bytes - gemm1_cutlass_aux_bytes(total_tokens)
            : 0;

    return kernel6_internal::launch_cutlass_gemm2_combine(
        shared_problem, shared_workspace, total_tokens);
}

}  // namespace kernel4_internal

#endif
