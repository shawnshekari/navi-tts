// Achieved VRAM bandwidth of the frame's grid matvec pattern at the real grid
// (1 block per WGP), a grid barrier after every matvec, for the layer shapes;
// variants of rows per wave / loads in flight / block size. Streams a 500 MB
// rotation so the 96 MB Infinity Cache does not flatter it. Build:
//   clang++ -x hip --offload-arch=gfx1100 --rocm-path=$ROCM -O3 matvec.hip -o matvec -L$ROCM/lib -lamdhip64
// Every barrier has a spin cap (CLAUDE.md); run it briefly, the card is shared.
// 2026-09-17, XTX: R4 U2 519-615 GB/s at the layer shapes, R2 U4 703-753 -> MV_ROWS 2 / unroll 4.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <vector>
#include <cstdlib>

__device__ __forceinline__ float wave_sum(float v) { for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, warpSize); return v; }
__device__ __forceinline__ float dot8(const uint4 & u, const float * xs) {
    const __half2 * h2 = reinterpret_cast<const __half2 *>(&u);
    const float2 f0 = __half22float2(h2[0]), f1 = __half22float2(h2[1]), f2 = __half22float2(h2[2]), f3 = __half22float2(h2[3]);
    return f0.x * xs[0] + f0.y * xs[1] + f1.x * xs[2] + f1.y * xs[3] + f2.x * xs[4] + f2.y * xs[5] + f3.x * xs[6] + f3.y * xs[7];
}
template <int R, int U>
__device__ __forceinline__ void wave_dot_rows(const uint16_t * W, int row0, int n_rows, int n_cols, const float * xs, float * acc) {
    const int lane = threadIdx.x % warpSize;
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll U
    for (int c = lane * 8; c < n_cols; c += warpSize * 8) {
        uint4 u[R];
#pragma unroll
        for (int r = 0; r < R; ++r) u[r] = (row0 + r < n_rows) ? *reinterpret_cast<const uint4 *>(W + static_cast<size_t>(row0 + r) * n_cols + c) : make_uint4(0, 0, 0, 0);
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] += dot8(u[r], xs + c);
    }
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = wave_sum(acc[r]);
}
template <int R, int U>
__device__ void grid_matvec(const uint16_t * W, int n_rows, int n_cols, const float * xs, float * y) {
    const int lane = threadIdx.x % warpSize, wpb = blockDim.x / warpSize;
    const int gwave = blockIdx.x * wpb + threadIdx.x / warpSize, nwave = gridDim.x * wpb;
    for (int row0 = gwave * R; row0 < n_rows; row0 += nwave * R) {
        float acc[R];
        wave_dot_rows<R, U>(W, row0, n_rows, n_cols, xs, acc);
        if (lane < R && row0 + lane < n_rows) { float d = acc[0];
#pragma unroll
            for (int r = 1; r < R; ++r) if (lane == r) d = acc[r];
            y[row0 + lane] = d; }
    }
}
__device__ void grid_barrier(volatile unsigned * bar, unsigned nblocks) {
    __shared__ unsigned lsense;
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned s = bar[1];
        __threadfence();
        const unsigned prev = atomicAdd((unsigned *) &bar[0], 1u);
        if (prev + 1u == nblocks) { bar[0] = 0u; __threadfence(); bar[1] = s ^ 1u; lsense = s ^ 1u; }
        else { unsigned long long spins = 0; while (bar[1] == s && !bar[2]) { if (++spins >= 20000000ull) { bar[2] = 1u; break; } } lsense = bar[1]; }
    }
    __syncthreads();
    (void) lsense;
}
constexpr int COPIES = 40;   // 40 x (up to 12.6 MB) = 500 MB rotation: past the 96 MB Infinity Cache
template <int R, int U>
__global__ void k_bench(const uint16_t * W, int n_rows, int n_cols, const float * x, float * y, volatile unsigned * bar, int iters) {
    __shared__ float xs[4096];
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    for (int it = 0; it < iters; ++it) {
        grid_matvec<R, U>(W + static_cast<size_t>(it % COPIES) * n_rows * n_cols, n_rows, n_cols, xs, y);
        grid_barrier(bar, gridDim.x);
        if (bar[2]) return;
    }
}
__global__ void k_barrier(volatile unsigned * bar, int iters) { for (int it = 0; it < iters; ++it) { grid_barrier(bar, gridDim.x); if (bar[2]) return; } }

template <int R, int U>
void run(const char * name, int block, int grid, const uint16_t * W, int rows, int cols, const float * x, float * y, unsigned * bar) {
    const int iters = 400;
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    void * args[] = {(void *) &W, (void *) &rows, (void *) &cols, (void *) &x, (void *) &y, (void *) &bar, (void *) &iters};
    hipMemset(bar, 0, 16);
    if (hipLaunchCooperativeKernel((const void *) k_bench<R, U>, dim3(grid), dim3(block), args, 0, 0) != hipSuccess) { printf("  %-22s block %3d: launch refused (%s)\n", name, block, hipGetErrorString(hipGetLastError())); return; }
    hipDeviceSynchronize();
    unsigned hb[4]; hipMemcpy(hb, bar, 16, hipMemcpyDeviceToHost); if (hb[2]) { printf("  %-22s block %3d: barrier timed out\n", name, block); return; }
    hipEventRecord(a);
    hipLaunchCooperativeKernel((const void *) k_bench<R, U>, dim3(grid), dim3(block), args, 0, 0);
    hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    const double us = ms * 1e3 / iters, bytes = 2.0 * rows * cols;
    printf("  %-22s block %3d  %5d x %4d: %6.2f us/matvec  %5.0f GB/s\n", name, block, rows, cols, us, bytes / us / 1e3);
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    hipDeviceProp_t prop; hipGetDeviceProperties(&prop, 0);
    const int grid = prop.multiProcessorCount;
    const int rows_max = 3072, cols_max = 3072;
    const size_t wbytes = static_cast<size_t>(COPIES) * 6144 * 1024 * 2;
    uint16_t * W; if (hipMalloc(&W, wbytes) != hipSuccess) { printf("no VRAM for %zu MB\n", wbytes >> 20); return 1; } hipMemset(W, 0x3c, wbytes);
    float * x; hipMalloc(&x, cols_max * 4); hipMemset(x, 0, cols_max * 4);
    float * y; hipMalloc(&y, rows_max * 4);
    unsigned * bar; hipMalloc(&bar, 16); hipMemset(bar, 0, 16);
    // barrier alone
    { const int iters = 4000; hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); void * args[] = {(void *) &bar, (void *) &iters};
      if (hipLaunchCooperativeKernel((const void *) k_barrier, dim3(grid), dim3(256), args, 0, 0) != hipSuccess) { printf("barrier launch refused: %s\n", hipGetErrorString(hipGetLastError())); return 1; }
      hipDeviceSynchronize(); unsigned hb[4]; hipMemcpy(hb, bar, 16, hipMemcpyDeviceToHost); if (hb[2]) { printf("barrier timed out\n"); return 1; }
      hipEventRecord(a); hipLaunchCooperativeKernel((const void *) k_barrier, dim3(grid), dim3(256), args, 0, 0); hipEventRecord(b); hipEventSynchronize(b);
      float ms; hipEventElapsedTime(&ms, a, b); printf("grid barrier alone: %.2f us (%d blocks)\n", ms * 1e3 / iters, grid); }
    const int shapes[][2] = {{2048, 1024}, {3072, 1024}, {1024, 3072}};   // 1024x1024 x 40 copies still half-fits the cache
    for (auto & s : shapes) {
        printf("shape %d x %d (%.1f MB)\n", s[0], s[1], 2.0 * s[0] * s[1] / 1e6);
        run<4, 2>("R4 U2 (current)", 256, grid, W, s[0], s[1], x, y, bar);
        run<4, 4>("R4 U4", 256, grid, W, s[0], s[1], x, y, bar);
        run<8, 2>("R8 U2", 256, grid, W, s[0], s[1], x, y, bar);
        run<2, 4>("R2 U4", 256, grid, W, s[0], s[1], x, y, bar);
        run<2, 8>("R2 U8", 256, grid, W, s[0], s[1], x, y, bar);
        run<1, 8>("R1 U8", 256, grid, W, s[0], s[1], x, y, bar);
        run<2, 4>("R2 U4", 512, grid, W, s[0], s[1], x, y, bar);
    }
    return 0;
}
