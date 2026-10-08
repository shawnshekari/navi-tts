// Achieved VRAM bandwidth of the frame's grid matvec pattern at the real grid
// (1 block per WGP), a grid barrier after every matvec, for the layer shapes;
// variants of rows per wave / loads in flight / block size. Streams a 500 MB
// rotation so the 96 MB Infinity Cache does not flatter it. Build:
//   clang++ -x hip --offload-arch=gfx1100 --rocm-path=$ROCM -O3 matvec.hip -o matvec -L$ROCM/lib -lamdhip64
// Every barrier has a spin cap; run it briefly, the card is shared.
// 2026-09-17, XTX: R4 U2 519-615 GB/s at the layer shapes, R2 U4 703-753 -> MV_ROWS 2 / unroll 4.
// 2026-09-18, int8 weights (M4's question), block-32 f16 scales, u8 + folded
// offset, timing only: at 256-thread blocks the convert per byte makes it
// VALU-bound (415-590 GB/s of int8, i.e. slower per weight than f16 on the
// wide shapes); at 512-thread blocks with a tree dot and U2, 590-870 GB/s:
// 2.6 / 5.3 / 5.6 us vs f16's 5.9 / 8.4 / 8.6 at the three shapes, a 0.55x
// per-layer matvec time. Against the frame-8 profile (matvec 4.7 of 6.0 ms:
// cp 3.5, talker 1.2) that is frame ~4.5 ms / RTF ~0.066 with the code
// predictor at int8, ~3.9 / ~0.059 with the talker too, ~3.7 / ~0.056 if
// int8 streamed at f16's rate (W8A8 with v_dot4 would be the way there).
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

// ---- int8 weights, the same pattern (M4's question: what does halving the
// bytes buy at this grid?). Weights stored as u8 = int8 + 128; the -128 is
// folded into a per-segment sum of x computed once. Timing only.
//   per-row scale : W [rows][cols] u8, scale [rows] f32          (1.00 B/weight)
//   block-32 scale: W as above,     scale [rows][cols/32] f16     (1.06 B/weight)
__device__ __forceinline__ float dot16_u8(const uint4 & u, const float * xs) {
    // four independent 4-term chains, then a tree: a quarter of the chain latency
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
    float d[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        d[k] = __uint2float_rn(w[k] & 0xffu) * xs[k * 4 + 0];
        d[k] += __uint2float_rn((w[k] >> 8) & 0xffu) * xs[k * 4 + 1];
        d[k] += __uint2float_rn((w[k] >> 16) & 0xffu) * xs[k * 4 + 2];
        d[k] += __uint2float_rn(w[k] >> 24) * xs[k * 4 + 3];
    }
    return (d[0] + d[1]) + (d[2] + d[3]);
}
template <int R, int U, bool BLOCK>
__device__ void grid_matvec_u8(const uint8_t * W, const float * rowscale, const uint16_t * blockscale, int n_rows, int n_cols,
                               const float * xs, const float * xsum16, float * y) {
    const int lane = threadIdx.x % warpSize, wpb = blockDim.x / warpSize;
    const int gwave = blockIdx.x * wpb + threadIdx.x / warpSize, nwave = gridDim.x * wpb;
    for (int row0 = gwave * R; row0 < n_rows; row0 += nwave * R) {
        float acc[R];
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll U
        for (int c = lane * 16; c < n_cols; c += warpSize * 16) {
            uint4 u[R];
#pragma unroll
            for (int r = 0; r < R; ++r) u[r] = (row0 + r < n_rows) ? *reinterpret_cast<const uint4 *>(W + static_cast<size_t>(row0 + r) * n_cols + c) : make_uint4(0, 0, 0, 0);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const float d = dot16_u8(u[r], xs + c) - 128.f * xsum16[c / 16];
                if (BLOCK) {
                    const uint16_t sb = blockscale[(static_cast<size_t>(row0 + r) * n_cols + c) / 32];
                    _Float16 h; __builtin_memcpy(&h, &sb, 2);
                    acc[r] += d * static_cast<float>(h);
                } else {
                    acc[r] += d;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = wave_sum(acc[r]);
        if (lane < R && row0 + lane < n_rows) { float d = acc[0];
#pragma unroll
            for (int r = 1; r < R; ++r) if (lane == r) d = acc[r];
            y[row0 + lane] = BLOCK ? d : d * rowscale[row0 + lane]; }
    }
}
template <int R, int U, bool BLOCK>
__global__ void k_bench_u8(const uint8_t * W, const float * rowscale, const uint16_t * blockscale, int n_rows, int n_cols, const float * x, float * y, volatile unsigned * bar, int iters) {
    __shared__ float xs[4096];
    __shared__ float xsum16[256];
    for (int i = threadIdx.x; i < n_cols; i += blockDim.x) xs[i] = x[i];
    __syncthreads();
    for (int i = threadIdx.x; i < n_cols / 16; i += blockDim.x) { float s = 0.f; for (int k = 0; k < 16; ++k) s += xs[i * 16 + k]; xsum16[i] = s; }
    __syncthreads();
    for (int it = 0; it < iters; ++it) {
        const size_t off = static_cast<size_t>(it % COPIES) * n_rows * n_cols;
        grid_matvec_u8<R, U, BLOCK>(W + off, rowscale, blockscale + off / 32, n_rows, n_cols, xs, xsum16, y);
        grid_barrier(bar, gridDim.x);
        if (bar[2]) return;
    }
}
template <int R, int U, bool BLOCK>
void run_u8(const char * name, int block, int grid, const uint8_t * W, const float * rowscale, const uint16_t * blockscale, int rows, int cols, const float * x, float * y, unsigned * bar) {
    const int iters = 400;
    hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
    void * args[] = {(void *) &W, (void *) &rowscale, (void *) &blockscale, (void *) &rows, (void *) &cols, (void *) &x, (void *) &y, (void *) &bar, (void *) &iters};
    hipMemset(bar, 0, 16);
    if (hipLaunchCooperativeKernel((const void *) k_bench_u8<R, U, BLOCK>, dim3(grid), dim3(block), args, 0, 0) != hipSuccess) { printf("  %-22s block %3d: launch refused (%s)\n", name, block, hipGetErrorString(hipGetLastError())); return; }
    hipDeviceSynchronize();
    unsigned hb[4]; hipMemcpy(hb, bar, 16, hipMemcpyDeviceToHost); if (hb[2]) { printf("  %-22s block %3d: barrier timed out\n", name, block); return; }
    hipEventRecord(a);
    hipLaunchCooperativeKernel((const void *) k_bench_u8<R, U, BLOCK>, dim3(grid), dim3(block), args, 0, 0);
    hipEventRecord(b); hipEventSynchronize(b);
    float ms; hipEventElapsedTime(&ms, a, b);
    const double us = ms * 1e3 / iters, bytes = (BLOCK ? 1.0625 : 1.0) * rows * cols;
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
    // int8 copies: the same rotation at 1 byte per weight, block scales alongside
    uint8_t * W8; hipMalloc(&W8, wbytes / 2); hipMemset(W8, 0x80, wbytes / 2);
    float * rs; hipMalloc(&rs, rows_max * 4); hipMemset(rs, 0, rows_max * 4);
    uint16_t * bs; hipMalloc(&bs, wbytes / 2 / 32 * 2); hipMemset(bs, 0x3c, wbytes / 2 / 32 * 2);
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
        run_u8<2, 4, false>("u8 row-scale R2 U4", 256, grid, W8, rs, bs, s[0], s[1], x, y, bar);
        run_u8<2, 4, true>("u8 block32 R2 U4", 256, grid, W8, rs, bs, s[0], s[1], x, y, bar);
        run_u8<4, 4, true>("u8 block32 R4 U4", 256, grid, W8, rs, bs, s[0], s[1], x, y, bar);
        run_u8<2, 4, true>("u8 block32 R2 U4", 512, grid, W8, rs, bs, s[0], s[1], x, y, bar);
        run_u8<4, 2, true>("u8 block32 R4 U2", 512, grid, W8, rs, bs, s[0], s[1], x, y, bar);
        run_u8<2, 2, true>("u8 block32 R2 U2", 512, grid, W8, rs, bs, s[0], s[1], x, y, bar);
    }
    return 0;
}
