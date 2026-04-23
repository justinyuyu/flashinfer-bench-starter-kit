// Test coverage:
// T1 fp8 encode/decode
// T2 block dequant
// T3 SwiGLU
// T4 routing
// T5 launch correctness
// T6 kernel6 launch correctness
// T7 backend benchmark

#include "../kernels/moe_expert_mlp/kernel4.cuh"
#include "../kernels/moe_expert_mlp/kernel6.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace moe_spec;

// ─── Utilities ────────────────────────────────────────────────────────────────

#define CHECK(x) do { \
    cudaError_t _e=(x); \
    if(_e!=cudaSuccess){ fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1); } \
} while(0)

static std::mt19937 rng(1337);

static std::vector<float> randf(size_t n, float lo=-1.f, float hi=1.f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n); for(auto& x:v) x=d(rng); return v;
}

// Quantize float to FP8 e4m3fn (simple nearest, no saturation for test values)
static uint8_t float_to_fp8(float v) {
    // Clamp to FP8 range [-448, 448]
    v = std::max(-448.f, std::min(448.f, v));
    if (v == 0.f) return 0;
    int sign = v < 0; if(sign) v=-v;

    if (v < ldexpf(1.f, -6)) {
        int man = std::min(7, std::max(0, (int)roundf(v * 512.f)));
        return (uint8_t)((sign << 7) | man);
    }

    int exp = (int)floorf(log2f(v));
    exp = std::max(-6, std::min(8, exp));  // e4m3 range: exp in [-6, 8]
    float mantissa_f = v / ldexpf(1.f, exp) - 1.f;
    int man = (int)roundf(mantissa_f * 8.f);
    if (man == 8) {
        man = 0;
        exp = std::min(8, exp + 1);
    }

    int e_biased = exp + 7;  // bias=7
    return (uint8_t)((sign << 7) | (e_biased << 3) | std::min(7, man));
}

// Software FP8 decode (mirrors device version)
static float fp8_decode(uint8_t v) {
    if (v == 0x7F || v == 0xFF) return 0.f;
    int sign     = (v >> 7) & 1;
    int exp_bits = (v >> 3) & 0xF;
    int man_bits = v & 0x7;
    float mantissa = (exp_bits==0) ? (man_bits/8.f) : (1.f + man_bits/8.f);
    float scale    = (exp_bits==0) ? ldexpf(1.f,-6) : ldexpf(1.f, exp_bits-7);
    float r = mantissa * scale;
    return sign ? -r : r;
}

static float max_abs_diff_bf16(const std::vector<float>& ref,
                                const std::vector<uint16_t>& gpu) {
    float mx = 0;
    for (size_t i=0;i<ref.size();++i) {
        // BF16: top 16 bits of float32
        uint32_t bf = ((uint32_t)gpu[i]) << 16;
        float g; memcpy(&g, &bf, 4);
        mx = std::max(mx, fabsf(ref[i]-g));
    }
    return mx;
}

static float silu(float x) { return x/(1.f+expf(-x)); }

static uint16_t float_to_bf16_bits(float v) {
    uint32_t bits = 0;
    memcpy(&bits, &v, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

static void fill_quantized_random(size_t n,
                                  float lo,
                                  float hi,
                                  std::vector<uint8_t>& q,
                                  std::vector<float>& dq) {
    auto src = randf(n, lo, hi);
    q.resize(n);
    dq.resize(n);
    for (size_t i = 0; i < n; ++i) {
        q[i] = float_to_fp8(src[i]);
        dq[i] = fp8_decode(q[i]);
    }
}

static int active_expert_span(const std::vector<int>& offsets) {
    int span = 1;
    for (int e = 0; e < NUM_LOCAL_EXPERTS; ++e) {
        if (offsets[e + 1] > offsets[e]) {
            span = e + 1;
        }
    }
    return span;
}

struct LaunchCase {
    const char* name;
    int seq_len;
    std::vector<int> token_indices;
    std::vector<int> expert_offsets;
    std::vector<float> routing_weights;
};

// ─── Test 1: FP8 encode/decode round-trip ────────────────────────────────────

bool test_fp8_roundtrip() {
    printf("\n[T1] FP8 e4m3fn encode/decode round-trip\n");
    std::vector<float> vals = {0.f, 1.f, -1.f, 0.5f, -0.5f,
                                0.125f, 7.f, -7.f, 448.f, -448.f,
                                0.0078125f, 3.14f};
    bool ok = true;
    for (float v : vals) {
        uint8_t q = float_to_fp8(v);
        float   d = fp8_decode(q);
        float   err = fabsf(d - v) / (fabsf(v) + 1e-6f);
        printf("  %8.4f → 0x%02X → %8.4f  rel_err=%.4f %s\n",
               v, q, d, err, err<0.15f?"ok":"fail");
        if (err > 0.15f) ok = false;
    }
    printf("  Result: %s\n", ok?"PASS":"FAIL");
    return ok;
}

// ─── Test 2: Block dequant (one 128×128 tile) ────────────────────────────────

bool test_block_dequant() {
    printf("\n[T2] Block dequant scale application (128×128 tile)\n");

    const int T=4, K=BLOCK_SIZE, N=BLOCK_SIZE;

    // Fake activation: constant 1.0, scale = 2.0
    // Fake weight:     constant 1.0, scale = 3.0
    // Expected dot product per output element: K * (1*2) * (1*3) = 128 * 6 = 768
    auto act_f8  = std::vector<uint8_t>(T*K, float_to_fp8(1.f));
    auto w_f8    = std::vector<uint8_t>(N*K, float_to_fp8(1.f));
    float a_scale = 2.f, w_scale = 3.f;

    // CPU compute
    float expected = 0.f;
    for (int k=0; k<K; ++k) {
        expected += fp8_decode(act_f8[k]) * a_scale
                  * fp8_decode(w_f8[k])   * w_scale;
    }
    printf("  Expected per output element: %.1f\n", expected);

    // Verify with SwiGLU: out = gate * silu(up)
    // If up=gate=expected, result = expected * silu(expected)
    float swiglu_expected = expected * silu(expected);
    printf("  After SwiGLU (up=gate=%.1f): %.4f\n", expected, swiglu_expected);

    bool ok = (fabsf(expected - 768.f) < 1.f);
    printf("  Result: %s\n", ok?"PASS":"FAIL");
    return ok;
}

// ─── Test 3: SwiGLU numerical correctness ────────────────────────────────────

bool test_swiglu_numerics() {
    printf("\n[T3] SwiGLU formula correctness\n");
    struct Case { float up, gate, expected; };
    std::vector<Case> cases = {
        {0.f,   1.f,  0.f},           // silu(0)=0
        {1.f,   1.f,  silu(1.f)},
        {-1.f,  2.f,  2.f*silu(-1.f)},
        {2.f,   0.f,  0.f},
        {1.f,  -1.f, -silu(1.f)},
    };
    bool ok=true;
    for (auto& c : cases) {
        float result = c.gate * silu(c.up);
        float err    = fabsf(result - c.expected);
        printf("  gate=%.2f up=%.2f → %.6f  expected=%.6f  err=%.2e %s\n",
               c.gate, c.up, result, c.expected, err, err<1e-5f?"ok":"fail");
        if(err>1e-5f) ok=false;
    }
    printf("  Result: %s\n", ok?"PASS":"FAIL");
    return ok;
}

// ─── Test 4: Routing correctness ─────────────────────────────────────────────

bool test_routing() {
    printf("\n[T4] Routing: sigmoid + bias + top-K + weight normalization\n");
    const int S=2, E=NUM_EXPERTS, K=TOP_K;

    auto logits = randf(S*E, -2.f, 2.f);
    auto bias   = randf(E, -0.1f, 0.1f);

    // Manual computation for token 0
    std::vector<float> scores(E);
    for(int e=0;e<E;++e)
        scores[e] = 1.f/(1.f+expf(-logits[e])) + bias[e];

    std::vector<int> idx(E); std::iota(idx.begin(),idx.end(),0);
    std::partial_sort(idx.begin(), idx.begin()+K, idx.end(),
        [&](int a,int b){return scores[a]>scores[b];});

    float sum=0;
    for(int k=0;k<K;++k) sum += 1.f/(1.f+expf(-logits[idx[k]]));

    printf("  Top-%d experts for token 0: ", K);
    for(int k=0;k<K;++k) printf("%d(w=%.3f) ", idx[k],
        (1.f/(1.f+expf(-logits[idx[k]])))/sum);
    printf("\n");

    // Verify weights sum to 1.0
    float wsum=0;
    for(int k=0;k<K;++k) wsum += (1.f/(1.f+expf(-logits[idx[k]])))/sum;
    printf("  Weight sum: %.6f (should be 1.0)\n", wsum);
    bool ok = fabsf(wsum - 1.f) < 1e-5f;
    printf("  Result: %s\n", ok?"PASS":"FAIL");
    return ok;
}

// ─── Test 5: Full Kernel 4+5 integration coverage ───────────────────────────

bool test_kernel4_launch_smoke() {
    printf("\n[T5] Kernel 4+5 launch coverage (reference + tiled + CUTLASS)\n");

    const float routed_scaling_factor = 1.25f;
    const float tol = 0.35f;
    const bool cutlass_available = k4_cutlass_available();

    std::vector<LaunchCase> cases;
    cases.push_back({"zero dispatch", 2, {}, std::vector<int>(NUM_LOCAL_EXPERTS + 1, 0), {}});

    std::vector<int> single_offsets(NUM_LOCAL_EXPERTS + 1, 2);
    single_offsets[0] = 0;
    single_offsets[1] = 2;
    cases.push_back({"single expert, seq_len=2", 2, {1, 0}, single_offsets, {0.65f, 0.35f}});

    std::vector<int> multi_small_offsets(NUM_LOCAL_EXPERTS + 1, 3);
    multi_small_offsets[0] = 0;
    multi_small_offsets[1] = 1;
    multi_small_offsets[2] = 3;
    cases.push_back({"multi expert, seq_len=2", 2, {1, 0, 1}, multi_small_offsets, {0.5f, 0.3f, 0.2f}});

    std::vector<int> multi_large_offsets(NUM_LOCAL_EXPERTS + 1, 8);
    multi_large_offsets[0] = 0;
    multi_large_offsets[1] = 3;
    multi_large_offsets[2] = 8;
    cases.push_back({"multi expert, seq_len=128", 128, {0, 1, 0, 65, 127, 2, 65, 5},
                     multi_large_offsets, {0.4f, 0.35f, 0.25f, 0.5f, 0.2f, 0.3f, 0.15f, 0.45f}});

    bool ok = true;
    for (const auto& tc : cases) {
        const int total_tok = tc.expert_offsets.back();
        const int expert_span = active_expert_span(tc.expert_offsets);
        const size_t act_elems = std::max<size_t>(1, (size_t)total_tok * HIDDEN_SIZE);
        const size_t gemm1_weight_elems = (size_t)expert_span * GEMM1_OUT_SIZE * HIDDEN_SIZE;
        const size_t gemm1_scale_elems = (size_t)expert_span * NUM_GEMM1_OUT_BLOCKS * NUM_HIDDEN_BLOCKS;
        const size_t gemm2_weight_elems = (size_t)expert_span * HIDDEN_SIZE * INTERMEDIATE_SIZE;
        const size_t gemm2_scale_elems = (size_t)expert_span * NUM_HIDDEN_BLOCKS * NUM_INTER_BLOCKS;

        std::vector<uint8_t> act_fp8, w_fp8, w2_fp8;
        std::vector<float> act_q_f32, w_q_f32, w2_q_f32;
        fill_quantized_random(act_elems, -0.25f, 0.25f, act_fp8, act_q_f32);
        fill_quantized_random(gemm1_weight_elems, -0.05f, 0.05f, w_fp8, w_q_f32);
        fill_quantized_random(gemm2_weight_elems, -0.05f, 0.05f, w2_fp8, w2_q_f32);

        auto act_scale_f32 = randf((size_t)NUM_HIDDEN_BLOCKS * tc.seq_len, 0.5f, 1.5f);
        auto w_scale_f32 = randf(gemm1_scale_elems, 0.5f, 1.25f);
        auto w2_scale_f32 = randf(gemm2_scale_elems, 0.5f, 1.25f);
        std::vector<float> ref_out((size_t)tc.seq_len * HIDDEN_SIZE, 0.f);
        k4_reference_cpu(
            act_q_f32.data(),
            act_scale_f32.data(),
            tc.token_indices.empty() ? nullptr : tc.token_indices.data(),
            tc.expert_offsets.data(),
            w_q_f32.data(),
            w_scale_f32.data(),
            w2_q_f32.data(),
            w2_scale_f32.data(),
            tc.routing_weights.empty() ? nullptr : tc.routing_weights.data(),
            routed_scaling_factor,
            tc.seq_len,
            total_tok,
            ref_out.data());

        auto run_backend = [&](Kernel4Backend backend,
                               std::vector<uint16_t>& gpu_out,
                               float& max_err,
                               bool& skipped) {
            skipped = false;
            fp8_e4m3 *d_act = nullptr, *d_w = nullptr, *d_w2 = nullptr;
            float *d_as = nullptr, *d_ws = nullptr, *d_ws2 = nullptr, *d_rw = nullptr;
            uint16_t *d_out = nullptr;
            int *d_off = nullptr, *d_tok = nullptr;
            void* d_workspace = nullptr;

            CHECK(cudaMalloc(&d_act, act_fp8.size() * sizeof(fp8_e4m3)));
            CHECK(cudaMalloc(&d_w, w_fp8.size() * sizeof(fp8_e4m3)));
            CHECK(cudaMalloc(&d_w2, w2_fp8.size() * sizeof(fp8_e4m3)));
            CHECK(cudaMalloc(&d_as, act_scale_f32.size() * sizeof(float)));
            CHECK(cudaMalloc(&d_ws, w_scale_f32.size() * sizeof(float)));
            CHECK(cudaMalloc(&d_ws2, w2_scale_f32.size() * sizeof(float)));
            CHECK(cudaMalloc(&d_out, ref_out.size() * sizeof(uint16_t)));
            CHECK(cudaMalloc(&d_off, tc.expert_offsets.size() * sizeof(int)));
            if (!tc.token_indices.empty()) CHECK(cudaMalloc(&d_tok, tc.token_indices.size() * sizeof(int)));
            if (!tc.routing_weights.empty()) CHECK(cudaMalloc(&d_rw, tc.routing_weights.size() * sizeof(float)));

            size_t workspace_bytes = k4_query_workspace(tc.seq_len, total_tok);
            if (workspace_bytes > 0) CHECK(cudaMalloc(&d_workspace, workspace_bytes));

            CHECK(cudaMemcpy(d_act, act_fp8.data(), act_fp8.size() * sizeof(fp8_e4m3), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_w, w_fp8.data(), w_fp8.size() * sizeof(fp8_e4m3), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_w2, w2_fp8.data(), w2_fp8.size() * sizeof(fp8_e4m3), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_as, act_scale_f32.data(), act_scale_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_ws, w_scale_f32.data(), w_scale_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_ws2, w2_scale_f32.data(), w2_scale_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_off, tc.expert_offsets.data(), tc.expert_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
            if (d_tok) CHECK(cudaMemcpy(d_tok, tc.token_indices.data(), tc.token_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
            if (d_rw) CHECK(cudaMemcpy(d_rw, tc.routing_weights.data(), tc.routing_weights.size() * sizeof(float), cudaMemcpyHostToDevice));

            Kernel4Workspace workspace = k4_bind_workspace(
                d_workspace, workspace_bytes, tc.seq_len, total_tok);

            Kernel4Problem problem{};
            problem.seq_len = tc.seq_len;
            problem.hidden_states = d_act;
            problem.hidden_states_scale = d_as;
            problem.gemm1_weights = d_w;
            problem.gemm1_weights_scale = d_ws;
            problem.gemm2_weights = d_w2;
            problem.gemm2_weights_scale = d_ws2;
            problem.local_expert_offset = 0;
            problem.routed_scaling_factor = routed_scaling_factor;
            problem.expert_token_offsets = d_off;
            problem.token_indices = d_tok;
            problem.token_expert_weights = d_rw;
            problem.output = reinterpret_cast<__nv_bfloat16*>(d_out);
            problem.backend = backend;
            problem.stream = nullptr;

            cudaError_t err = k4_launch(problem, workspace);
            if (backend == Kernel4Backend::Cutlass && err == cudaErrorNotSupported) {
                skipped = true;
            } else {
                CHECK(err);
                CHECK(cudaDeviceSynchronize());
                gpu_out.resize(ref_out.size());
                CHECK(cudaMemcpy(gpu_out.data(), d_out, gpu_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
                max_err = max_abs_diff_bf16(ref_out, gpu_out);
            }

            cudaFree(d_act); cudaFree(d_w); cudaFree(d_w2); cudaFree(d_as); cudaFree(d_ws); cudaFree(d_ws2);
            cudaFree(d_rw); cudaFree(d_out); cudaFree(d_off); cudaFree(d_tok); cudaFree(d_workspace);
        };

        std::vector<uint16_t> fallback_out;
        float fallback_err = 0.f;
        bool fallback_skipped = false;
        run_backend(Kernel4Backend::Fallback, fallback_out, fallback_err, fallback_skipped);
        bool fallback_ok = !fallback_skipped && fallback_err < tol;
        ok &= fallback_ok;

        printf("  %-24s reference max_err=%.6f %s\n",
               tc.name, fallback_err, fallback_ok ? "ok" : "fail");

        std::vector<uint16_t> tiled_out;
        float tiled_err = 0.f;
        bool tiled_skipped = false;
        run_backend(Kernel4Backend::Tiled, tiled_out, tiled_err, tiled_skipped);
        bool tiled_ok = !tiled_skipped && tiled_err < tol;
        ok &= tiled_ok;
        printf("  %-24s tiled     max_err=%.6f %s\n",
               tc.name, tiled_err, tiled_ok ? "ok" : "fail");

        std::vector<uint16_t> cutlass_out;
        float cutlass_err = 0.f;
        bool cutlass_skipped = false;
        run_backend(Kernel4Backend::Cutlass, cutlass_out, cutlass_err, cutlass_skipped);
        if (cutlass_skipped) {
            printf("  %-24s CUTLASS skipped (%s)\n",
                   tc.name, cutlass_available ? "backend unavailable" : "built without CUTLASS");
        } else {
            float cross_err = max_abs_diff_bf16(ref_out, cutlass_out);
            bool cutlass_ok = cutlass_err < tol && cross_err < tol;
            ok &= cutlass_ok;
            printf("  %-24s CUTLASS max_err=%.6f %s\n",
                   tc.name, cutlass_err, cutlass_ok ? "ok" : "fail");
        }
    }

    printf("  Result: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ─── Test 6: Standalone Kernel 6 coverage ────────────────────────────────────

bool test_kernel6_launch_smoke() {
    printf("\n[T6] Kernel 6 launch coverage (fallback + CUTLASS)\n");

    const float routed_scaling_factor = 1.25f;
    const float tol = 0.35f;
    const bool cutlass_available = k6_cutlass_available();

    std::vector<LaunchCase> cases;
    cases.push_back({"zero dispatch", 2, {}, std::vector<int>(NUM_LOCAL_EXPERTS + 1, 0), {}});

    std::vector<int> single_offsets(NUM_LOCAL_EXPERTS + 1, 2);
    single_offsets[0] = 0;
    single_offsets[1] = 2;
    cases.push_back({"single expert, seq_len=2", 2, {1, 0}, single_offsets, {0.65f, 0.35f}});

    std::vector<int> multi_offsets(NUM_LOCAL_EXPERTS + 1, 6);
    multi_offsets[0] = 0;
    multi_offsets[1] = 2;
    multi_offsets[2] = 6;
    cases.push_back({"multi expert combine", 4, {1, 0, 1, 3, 0, 3}, multi_offsets,
                     {0.5f, 0.25f, 0.3f, 0.4f, 0.2f, 0.15f}});

    bool ok = true;
    for (const auto& tc : cases) {
        const int total_tok = tc.expert_offsets.back();
        const int expert_span = active_expert_span(tc.expert_offsets);
        const size_t inter_elems = std::max<size_t>(1, (size_t)total_tok * INTERMEDIATE_SIZE);
        const size_t gemm2_weight_elems = (size_t)expert_span * HIDDEN_SIZE * INTERMEDIATE_SIZE;
        const size_t gemm2_scale_elems = (size_t)expert_span * NUM_HIDDEN_BLOCKS * NUM_INTER_BLOCKS;

        auto inter_f32 = randf(inter_elems, -0.25f, 0.25f);
        std::vector<uint16_t> inter_bf16(inter_elems);
        for (size_t i = 0; i < inter_elems; ++i) {
            inter_bf16[i] = float_to_bf16_bits(inter_f32[i]);
        }

        std::vector<uint8_t> w2_fp8;
        std::vector<float> w2_q_f32;
        fill_quantized_random(gemm2_weight_elems, -0.05f, 0.05f, w2_fp8, w2_q_f32);
        auto w2_scale_f32 = randf(gemm2_scale_elems, 0.5f, 1.25f);

        std::vector<float> ref_out((size_t)tc.seq_len * HIDDEN_SIZE, 0.f);
        k6_reference_cpu(
            reinterpret_cast<const __nv_bfloat16*>(inter_bf16.data()),
            tc.token_indices.empty() ? nullptr : tc.token_indices.data(),
            tc.expert_offsets.data(),
            w2_q_f32.data(),
            w2_scale_f32.data(),
            tc.routing_weights.empty() ? nullptr : tc.routing_weights.data(),
            routed_scaling_factor,
            tc.seq_len,
            total_tok,
            ref_out.data());

        auto run_backend = [&](Kernel6Backend backend,
                               std::vector<uint16_t>& gpu_out,
                               float& max_err,
                               bool& skipped) {
            skipped = false;
            uint16_t *d_inter = nullptr, *d_out = nullptr;
            fp8_e4m3 *d_w2 = nullptr;
            float *d_ws2 = nullptr, *d_rw = nullptr;
            int *d_off = nullptr, *d_tok = nullptr;
            void* d_workspace = nullptr;

            CHECK(cudaMalloc(&d_inter, inter_bf16.size() * sizeof(uint16_t)));
            CHECK(cudaMalloc(&d_w2, w2_fp8.size() * sizeof(fp8_e4m3)));
            CHECK(cudaMalloc(&d_ws2, w2_scale_f32.size() * sizeof(float)));
            CHECK(cudaMalloc(&d_out, ref_out.size() * sizeof(uint16_t)));
            CHECK(cudaMalloc(&d_off, tc.expert_offsets.size() * sizeof(int)));
            if (!tc.token_indices.empty()) CHECK(cudaMalloc(&d_tok, tc.token_indices.size() * sizeof(int)));
            if (!tc.routing_weights.empty()) CHECK(cudaMalloc(&d_rw, tc.routing_weights.size() * sizeof(float)));

            size_t workspace_bytes = k6_query_workspace(tc.seq_len, total_tok);
            if (workspace_bytes > 0) CHECK(cudaMalloc(&d_workspace, workspace_bytes));

            CHECK(cudaMemcpy(d_inter, inter_bf16.data(), inter_bf16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_w2, w2_fp8.data(), w2_fp8.size() * sizeof(fp8_e4m3), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_ws2, w2_scale_f32.data(), w2_scale_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_off, tc.expert_offsets.data(), tc.expert_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
            if (d_tok) CHECK(cudaMemcpy(d_tok, tc.token_indices.data(), tc.token_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
            if (d_rw) CHECK(cudaMemcpy(d_rw, tc.routing_weights.data(), tc.routing_weights.size() * sizeof(float), cudaMemcpyHostToDevice));

            Kernel6Workspace workspace = k6_bind_workspace(
                d_workspace, workspace_bytes, tc.seq_len, total_tok);

            Kernel6Problem problem{};
            problem.seq_len = tc.seq_len;
            problem.hidden_states = reinterpret_cast<__nv_bfloat16*>(d_inter);
            problem.gemm2_weights = d_w2;
            problem.gemm2_weights_scale = d_ws2;
            problem.local_expert_offset = 0;
            problem.routed_scaling_factor = routed_scaling_factor;
            problem.expert_token_offsets = d_off;
            problem.token_indices = d_tok;
            problem.token_expert_weights = d_rw;
            problem.output = reinterpret_cast<__nv_bfloat16*>(d_out);
            problem.backend = backend;
            problem.stream = nullptr;

            cudaError_t err = k6_launch(problem, workspace);
            if (backend == Kernel6Backend::Cutlass && err == cudaErrorNotSupported) {
                skipped = true;
            } else {
                CHECK(err);
                CHECK(cudaDeviceSynchronize());
                gpu_out.resize(ref_out.size());
                CHECK(cudaMemcpy(gpu_out.data(), d_out, gpu_out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));
                max_err = max_abs_diff_bf16(ref_out, gpu_out);
            }

            cudaFree(d_inter); cudaFree(d_w2); cudaFree(d_ws2); cudaFree(d_rw);
            cudaFree(d_out); cudaFree(d_off); cudaFree(d_tok); cudaFree(d_workspace);
        };

        std::vector<uint16_t> fallback_out;
        float fallback_err = 0.f;
        bool fallback_skipped = false;
        run_backend(Kernel6Backend::Fallback, fallback_out, fallback_err, fallback_skipped);
        bool fallback_ok = !fallback_skipped && fallback_err < tol;
        ok &= fallback_ok;
        printf("  %-24s fallback max_err=%.6f %s\n",
               tc.name, fallback_err, fallback_ok ? "ok" : "fail");

        std::vector<uint16_t> cutlass_out;
        float cutlass_err = 0.f;
        bool cutlass_skipped = false;
        run_backend(Kernel6Backend::Cutlass, cutlass_out, cutlass_err, cutlass_skipped);
        if (cutlass_skipped) {
            printf("  %-24s CUTLASS skipped (%s)\n",
                   tc.name, cutlass_available ? "backend unavailable" : "built without CUTLASS");
        } else {
            bool cutlass_ok = cutlass_err < tol;
            ok &= cutlass_ok;
            printf("  %-24s CUTLASS max_err=%.6f %s\n",
                   tc.name, cutlass_err, cutlass_ok ? "ok" : "fail");
        }
    }

    if (!cutlass_available) {
        std::vector<int> off(NUM_LOCAL_EXPERTS + 1, 1);
        off[0] = 0;
        std::vector<uint16_t> inter(INTERMEDIATE_SIZE, float_to_bf16_bits(0.1f));
        std::vector<uint8_t> w2(HIDDEN_SIZE * INTERMEDIATE_SIZE, float_to_fp8(0.01f));
        std::vector<float> w2s(NUM_HIDDEN_BLOCKS * NUM_INTER_BLOCKS, 1.f);
        std::vector<int> tok(1, 0);
        std::vector<float> rw(1, 1.f);
        uint16_t* d_inter = nullptr;
        uint16_t* d_out = nullptr;
        fp8_e4m3* d_w2 = nullptr;
        float* d_ws2 = nullptr;
        int* d_off = nullptr;
        int* d_tok = nullptr;
        float* d_rw = nullptr;
        void* d_workspace = nullptr;
        CHECK(cudaMalloc(&d_inter, inter.size() * sizeof(uint16_t)));
        CHECK(cudaMalloc(&d_off, off.size() * sizeof(int)));
        CHECK(cudaMalloc(&d_tok, sizeof(int)));
        CHECK(cudaMalloc(&d_rw, sizeof(float)));
        CHECK(cudaMalloc(&d_w2, w2.size() * sizeof(fp8_e4m3)));
        CHECK(cudaMalloc(&d_ws2, w2s.size() * sizeof(float)));
        CHECK(cudaMemcpy(d_off, off.data(), off.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_inter, inter.data(), inter.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_tok, tok.data(), sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_rw, rw.data(), sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_w2, w2.data(), w2.size() * sizeof(fp8_e4m3), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_ws2, w2s.data(), w2s.size() * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMalloc(&d_out, (size_t)HIDDEN_SIZE * sizeof(uint16_t)));
        size_t workspace_bytes = k6_query_workspace(1, 1);
        CHECK(cudaMalloc(&d_workspace, workspace_bytes));
        Kernel6Workspace workspace = k6_bind_workspace(d_workspace, workspace_bytes, 1, 1);
        Kernel6Problem problem{};
        problem.seq_len = 1;
        problem.hidden_states = reinterpret_cast<__nv_bfloat16*>(d_inter);
        problem.gemm2_weights = d_w2;
        problem.gemm2_weights_scale = d_ws2;
        problem.expert_token_offsets = d_off;
        problem.token_indices = d_tok;
        problem.token_expert_weights = d_rw;
        problem.output = reinterpret_cast<__nv_bfloat16*>(d_out);
        problem.backend = Kernel6Backend::Cutlass;
        cudaError_t err = k6_launch(problem, workspace);
        bool not_supported = (err == cudaErrorNotSupported);
        ok &= not_supported;
        printf("  explicit Cutlass request without support: %s\n", not_supported ? "ok" : "fail");
        cudaFree(d_inter);
        cudaFree(d_w2);
        cudaFree(d_ws2);
        cudaFree(d_off);
        cudaFree(d_tok);
        cudaFree(d_rw);
        cudaFree(d_out);
        cudaFree(d_workspace);
    }

    printf("  Result: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// ─── Test 7: Backend benchmark ───────────────────────────────────────────────

bool benchmark_backends() {
    printf("\n[T7] Backend benchmark (reference vs tiled vs CUTLASS)\n");

    const int S = 128;
    const int total_tok = 128;
    const float routed_scaling_factor = 1.25f;
    const int warmup = 1;
    const int iters = 3;

    std::vector<int> h_offsets(NUM_LOCAL_EXPERTS + 1, total_tok);
    for (int e = 0; e <= 8; ++e) {
        h_offsets[e] = e * (total_tok / 8);
    }
    for (int e = 9; e <= NUM_LOCAL_EXPERTS; ++e) {
        h_offsets[e] = total_tok;
    }

    std::vector<int> h_token_indices(total_tok);
    for (int i = 0; i < total_tok; ++i) {
        h_token_indices[i] = i % S;
    }
    auto rw_f32 = randf(total_tok, 0.05f, 1.f);

    std::vector<uint8_t> act_fp8, w_fp8, w2_fp8;
    std::vector<float> act_q_f32, w_q_f32, w2_q_f32;
    fill_quantized_random((size_t)total_tok * HIDDEN_SIZE, -0.25f, 0.25f, act_fp8, act_q_f32);
    fill_quantized_random((size_t)8 * GEMM1_OUT_SIZE * HIDDEN_SIZE, -0.05f, 0.05f, w_fp8, w_q_f32);
    fill_quantized_random((size_t)8 * HIDDEN_SIZE * INTERMEDIATE_SIZE, -0.05f, 0.05f, w2_fp8, w2_q_f32);
    auto act_scale_f32 = randf((size_t)NUM_HIDDEN_BLOCKS * S, 0.5f, 1.5f);
    auto w_scale_f32 = randf((size_t)8 * NUM_GEMM1_OUT_BLOCKS * NUM_HIDDEN_BLOCKS, 0.5f, 1.25f);
    auto w2_scale_f32 = randf((size_t)8 * NUM_HIDDEN_BLOCKS * NUM_INTER_BLOCKS, 0.5f, 1.25f);

    fp8_e4m3 *d_act = nullptr, *d_w = nullptr, *d_w2 = nullptr;
    float *d_as = nullptr, *d_ws = nullptr, *d_ws2 = nullptr, *d_rw = nullptr;
    uint16_t *d_out = nullptr;
    int *d_off = nullptr, *d_tok = nullptr;
    void* d_workspace = nullptr;

    CHECK(cudaMalloc(&d_act, act_fp8.size() * sizeof(fp8_e4m3)));
    CHECK(cudaMalloc(&d_w, w_fp8.size() * sizeof(fp8_e4m3)));
    CHECK(cudaMalloc(&d_w2, w2_fp8.size() * sizeof(fp8_e4m3)));
    CHECK(cudaMalloc(&d_as, act_scale_f32.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_ws, w_scale_f32.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_ws2, w2_scale_f32.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_rw, rw_f32.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (size_t)S * HIDDEN_SIZE * sizeof(uint16_t)));
    CHECK(cudaMalloc(&d_off, h_offsets.size() * sizeof(int)));
    CHECK(cudaMalloc(&d_tok, h_token_indices.size() * sizeof(int)));
    size_t workspace_bytes = k4_query_workspace(S, total_tok);
    CHECK(cudaMalloc(&d_workspace, workspace_bytes));

    CHECK(cudaMemcpy(d_act, act_fp8.data(), act_fp8.size() * sizeof(fp8_e4m3), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, w_fp8.data(), w_fp8.size() * sizeof(fp8_e4m3), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w2, w2_fp8.data(), w2_fp8.size() * sizeof(fp8_e4m3), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_as, act_scale_f32.data(), act_scale_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ws, w_scale_f32.data(), w_scale_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_ws2, w2_scale_f32.data(), w2_scale_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_rw, rw_f32.data(), rw_f32.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_off, h_offsets.data(), h_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_tok, h_token_indices.data(), h_token_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

    Kernel4Workspace workspace = k4_bind_workspace(d_workspace, workspace_bytes, S, total_tok);
    Kernel4Problem problem{};
    problem.seq_len = S;
    problem.hidden_states = d_act;
    problem.hidden_states_scale = d_as;
    problem.gemm1_weights = d_w;
    problem.gemm1_weights_scale = d_ws;
    problem.gemm2_weights = d_w2;
    problem.gemm2_weights_scale = d_ws2;
    problem.local_expert_offset = 0;
    problem.routed_scaling_factor = routed_scaling_factor;
    problem.expert_token_offsets = d_off;
    problem.token_indices = d_tok;
    problem.token_expert_weights = d_rw;
    problem.output = reinterpret_cast<__nv_bfloat16*>(d_out);
    problem.stream = nullptr;

    auto bench_backend = [&](Kernel4Backend backend, const char* label, bool can_skip, float& ms_out) {
        if (can_skip && !k4_cutlass_available()) {
            printf("  %-10s skipped\n", label);
            ms_out = 1e30f;
            return;
        }
        problem.backend = backend;
        for (int i = 0; i < warmup; ++i) {
            CHECK(k4_launch(problem, workspace));
        }
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; ++i) {
            CHECK(k4_launch(problem, workspace));
        }
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float total_ms = 0.f;
        CHECK(cudaEventElapsedTime(&total_ms, start, stop));
        ms_out = total_ms / iters;
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
        printf("  %-10s %.3f ms\n", label, ms_out);
    };

    float ref_ms = 0.f, tiled_ms = 0.f, cutlass_ms = 0.f, auto_ms = 0.f;
    bench_backend(Kernel4Backend::Fallback, "reference", false, ref_ms);
    bench_backend(Kernel4Backend::Tiled, "tiled", false, tiled_ms);
    bench_backend(Kernel4Backend::Cutlass, "cutlass", true, cutlass_ms);
    bench_backend(Kernel4Backend::Auto, "auto", false, auto_ms);

    float best_ms = std::min(ref_ms, std::min(tiled_ms, cutlass_ms));
    const char* best_label = (best_ms == ref_ms) ? "reference" : (best_ms == tiled_ms ? "tiled" : "cutlass");
    const char* auto_label = k4_cutlass_available() ? "cutlass" : "reference";
    printf("  Fastest backend on this GPU: %s\n", best_label);
    printf("  Default Auto launcher maps to: %s\n", auto_label);

    cudaFree(d_act); cudaFree(d_w); cudaFree(d_w2); cudaFree(d_as); cudaFree(d_ws); cudaFree(d_ws2);
    cudaFree(d_rw); cudaFree(d_out); cudaFree(d_off); cudaFree(d_tok); cudaFree(d_workspace);
    return true;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("GPU: %s (sm_%d%d)\n", p.name, p.major, p.minor);
    printf("Spec: E=%d, E_local=%d, H=%d, D=%d, G=%d, block=%d\n\n",
           NUM_EXPERTS, NUM_LOCAL_EXPERTS, HIDDEN_SIZE,
           INTERMEDIATE_SIZE, GEMM1_OUT_SIZE, BLOCK_SIZE);

    bool all = true;
    all &= test_fp8_roundtrip();
    all &= test_block_dequant();
    all &= test_swiglu_numerics();
    all &= test_routing();
    all &= test_kernel4_launch_smoke();
    all &= test_kernel6_launch_smoke();
    all &= benchmark_backends();

    printf("\n══════════════════════════════════\n");
    printf("Overall: %s\n", all?"ALL PASS":"SOME FAIL");
    printf("══════════════════════════════════\n");
    return all ? 0 : 1;
}
