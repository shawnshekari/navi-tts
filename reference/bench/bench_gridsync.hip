// Build/run inside the navi31-llama toolbox (see docs/code_predictor_plan.md, Phase 0):
//   /opt/rocm/llvm/bin/clang++ -O2 --offload-arch=gfx1100 \
//     -resource-dir=$HOME/.local/share/rocm-clangrt-overlay --rtlib=compiler-rt -unwindlib=libgcc \
//     -x hip scripts/bench_gridsync.hip -o /tmp/bench_gridsync && /tmp/bench_gridsync
//
// Microbench for Phase 2's key assumption: cost of a cooperative-groups
// grid.sync() on gfx1100, vs the cost of a separate kernel launch.
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <chrono>
namespace cg = cooperative_groups;

__global__ void k_sync(float * buf, int n_sync) {
    cg::grid_group g = cg::this_grid();
    float acc = buf[blockIdx.x * blockDim.x + threadIdx.x];
    for (int i = 0; i < n_sync; i++) {
        acc = acc * 1.0001f + 1.0f;       // token work between syncs
        g.sync();
    }
    buf[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

__global__ void k_nosync(float * buf, int n_iter) {
    float acc = buf[blockIdx.x * blockDim.x + threadIdx.x];
    for (int i = 0; i < n_iter; i++) acc = acc * 1.0001f + 1.0f;
    buf[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}

__global__ void k_empty(float * buf) {
    if (threadIdx.x == 0 && blockIdx.x == 0) buf[0] += 1.0f;
}

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e)); return 1; } } while (0)

int main() {
    hipDeviceProp_t prop; CK(hipGetDeviceProperties(&prop, 0));
    printf("%s  CUs=%d  cooperativeLaunch=%d\n", prop.gcnArchName, prop.multiProcessorCount, prop.cooperativeLaunch);

    const int threads = 256;
    float * buf; CK(hipMalloc(&buf, 4096 * threads * sizeof(float)));
    CK(hipMemset(buf, 0, 4096 * threads * sizeof(float)));
    hipStream_t st; CK(hipStreamCreate(&st));

    // --- baseline: back-to-back empty launches on a stream ---
    for (int reps : {1000}) {
        k_empty<<<1, 64, 0, st>>>(buf); CK(hipStreamSynchronize(st));
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < reps; i++) k_empty<<<96, threads, 0, st>>>(buf);
        CK(hipStreamSynchronize(st));
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        printf("empty kernel launch (96 blocks), stream, %d back-to-back: %.2f us/launch\n", reps, us / reps);
    }

    // --- grid.sync cost at a few grid sizes ---
    for (int blocks : {48, 96, 192, 384}) {
        int max_blocks_per_cu = 0;
        CK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_cu, k_sync, threads, 0));
        if (blocks > max_blocks_per_cu * prop.multiProcessorCount) {
            printf("blocks=%d exceeds co-resident limit (%d), skipping\n", blocks, max_blocks_per_cu * prop.multiProcessorCount);
            continue;
        }
        for (int n_sync : {0, 1000}) {
            void * args[] = { &buf, &n_sync };
            // warmup
            CK(hipLaunchCooperativeKernel((const void *) k_sync, dim3(blocks), dim3(threads), args, 0, st));
            CK(hipStreamSynchronize(st));
            const int reps = 20;
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; r++)
                CK(hipLaunchCooperativeKernel((const void *) k_sync, dim3(blocks), dim3(threads), args, 0, st));
            CK(hipStreamSynchronize(st));
            double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            static double base = 0;
            if (n_sync == 0) { base = us; printf("blocks=%3d n_sync=%4d: %8.1f us/launch (cooperative launch overhead)\n", blocks, n_sync, us); }
            else             printf("blocks=%3d n_sync=%4d: %8.1f us/launch -> %.2f us per grid.sync\n", blocks, n_sync, us, (us - base) / n_sync);
        }
        // same arithmetic without syncs, to confirm the work itself is negligible
        int n = 1000; void * args[] = { &buf, &n };
        k_nosync<<<blocks, threads, 0, st>>>(buf, n); CK(hipStreamSynchronize(st));
        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < 20; r++) k_nosync<<<blocks, threads, 0, st>>>(buf, n);
        CK(hipStreamSynchronize(st));
        printf("blocks=%3d no-sync loop x1000:  %8.1f us/launch\n", blocks, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 20);
    }
    return 0;
}
