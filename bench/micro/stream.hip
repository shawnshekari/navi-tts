// Plain weight streams at the prefill/frame shapes: what the card moves for a
// 2-6 MB matrix under different grid shapes and lane mappings, no arithmetic.
// The ceiling every GEMM/matvec kernel is measured against. Build:
//   clang++ -x hip --offload-arch=gfx1100 --rocm-path=$ROCM -O3 stream.hip -o stream -L$ROCM/lib -lamdhip64
// (gfx1151 on the mini PC; docs/reference/toolchain.md for its extra flags.)
// 2026-09-18:
//   XTX     coalesced (blocks/8rows, persistent)  500-730 GB/s at 2-6 MB
//           per-lane rows 150-330, quads of lanes per row 440-530
//   Strix   coalesced 140-190 GB/s (LPDDR5x, ~256 theoretical), from GTT;
//           a 34 MB rotation runs at 470-510 - the 32 MB MALL; per-lane rows 65-160
// main() runs the coalesced set; main_lanes() the lane-mapping set (swap the names).
#include <hip/hip_runtime.h>
#include <cstdio>
// a: one block per ROWS rows, threads stream the rows' bytes coalesced, D loads in flight, no LDS
template <int D>
__global__ void k_blocks(const uint4 * W, int rows, int cin16, unsigned * sink) {
    const int r0 = blockIdx.x * 8;   // 8 rows per block, 128 threads
    const uint4 * base = W + (std::size_t) r0 * cin16;
    const int n = 8 * cin16;
    unsigned acc = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x * D) {
        uint4 v[D];
#pragma unroll
        for (int d = 0; d < D; ++d) v[d] = (i + d * blockDim.x < n) ? base[i + d * blockDim.x] : make_uint4(0,0,0,0);
#pragma unroll
        for (int d = 0; d < D; ++d) acc ^= v[d].x ^ v[d].w;
    }
    if (acc == 0x12345678u) sink[0] = acc;
}
// b: persistent grid, each wave takes rows in turn, lanes contiguous, D in flight
template <int D>
__global__ void k_persist(const uint4 * W, int rows, int cin16, unsigned * sink) {
    const int lane = threadIdx.x % 32, wpb = blockDim.x / 32, gw = blockIdx.x * wpb + threadIdx.x / 32, nw = gridDim.x * wpb;
    unsigned acc = 0;
    for (int r = gw; r < rows; r += nw) {
        const uint4 * row = W + (std::size_t) r * cin16;
        for (int c = lane; c < cin16; c += 32 * D) {
            uint4 v[D];
#pragma unroll
            for (int d = 0; d < D; ++d) v[d] = (c + d * 32 < cin16) ? row[c + d * 32] : make_uint4(0,0,0,0);
#pragma unroll
            for (int d = 0; d < D; ++d) acc ^= v[d].x ^ v[d].w;
        }
    }
    if (acc == 0x12345678u) sink[0] = acc;
}
// c: persistent, each wave takes R rows at once (more independent streams per wave)
template <int D, int R>
__global__ void k_persist_r(const uint4 * W, int rows, int cin16, unsigned * sink) {
    const int lane = threadIdx.x % 32, wpb = blockDim.x / 32, gw = blockIdx.x * wpb + threadIdx.x / 32, nw = gridDim.x * wpb;
    unsigned acc = 0;
    for (int r0 = gw * R; r0 < rows; r0 += nw * R) {
        for (int c = lane; c < cin16; c += 32 * D) {
            uint4 v[R][D];
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int d = 0; d < D; ++d) v[r][d] = (r0 + r < rows && c + d * 32 < cin16) ? W[(std::size_t)(r0 + r) * cin16 + c + d * 32] : make_uint4(0,0,0,0);
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int d = 0; d < D; ++d) acc ^= v[r][d].x ^ v[r][d].w;
        }
    }
    if (acc == 0x12345678u) sink[0] = acc;
}
int main() {
    const int shapes[][2] = {{2048, 1024}, {1024, 1024}, {3072, 1024}, {1024, 3072}};
    const std::size_t maxW = 3072 * 3072, rot = (512u << 20) / 2 / maxW + 1;
    uint4 * W; unsigned * sink; hipMalloc(&W, maxW * rot * 2); hipMalloc(&sink, 4); hipMemset(W, 0x3c, maxW * rot * 2);
    auto time = [&](const char * name, int rows, int cin, auto launch) {
        const int cin16 = cin / 8; const std::size_t stride = (std::size_t) rows * cin / 8;
        for (int i = 0; i < 4; ++i) launch(W + (i % rot) * stride, rows, cin16);
        hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
        const int iters = 40; hipEventRecord(a);
        for (int i = 0; i < iters; ++i) launch(W + (i % rot) * stride, rows, cin16);
        hipEventRecord(b); hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms, a, b);
        const double us = ms * 1e3 / iters; printf("  %-28s %6.1f us  %5.0f GB/s\n", name, us, 2.0 * rows * cin / us / 1e3);
    };
    for (auto & s : shapes) {
        const int rows = s[0], cin = s[1];
        printf("%d rows x %d (%.1f MB)\n", rows, cin, 2.0 * rows * cin / 1e6);
        time("blocks/8rows t128 D4", rows, cin, [&](const uint4 * w, int r, int c) { k_blocks<4><<<r / 8, 128>>>(w, r, c, sink); });
        time("blocks/8rows t128 D8", rows, cin, [&](const uint4 * w, int r, int c) { k_blocks<8><<<r / 8, 128>>>(w, r, c, sink); });
        time("persist 96x256 D4", rows, cin, [&](const uint4 * w, int r, int c) { k_persist<4><<<96, 256>>>(w, r, c, sink); });
        time("persist 96x256 D8", rows, cin, [&](const uint4 * w, int r, int c) { k_persist<8><<<96, 256>>>(w, r, c, sink); });
        time("persist 192x256 D8", rows, cin, [&](const uint4 * w, int r, int c) { k_persist<8><<<192, 256>>>(w, r, c, sink); });
        time("persist 48x256 R2 D4", rows, cin, [&](const uint4 * w, int r, int c) { k_persist_r<4, 2><<<48, 256>>>(w, r, c, sink); });
        time("persist 96x256 R2 D4", rows, cin, [&](const uint4 * w, int r, int c) { k_persist_r<4, 2><<<96, 256>>>(w, r, c, sink); });
        time("persist 96x256 R4 D4", rows, cin, [&](const uint4 * w, int r, int c) { k_persist_r<4, 4><<<96, 256>>>(w, r, c, sink); });
        time("persist 192x256 R2 D4", rows, cin, [&](const uint4 * w, int r, int c) { k_persist_r<4, 2><<<192, 256>>>(w, r, c, sink); });
    }
    return 0;
}
// d: lane = row, each lane streams its own row, D consecutive uint4 in flight (128 B = one line at D=8)
template <int D, int RPW>
__global__ void k_rowlane(const uint4 * W, int rows, int cin16, unsigned * sink) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, wpb = blockDim.x / 32;
    const int r = (blockIdx.x * wpb + wave) * RPW + lane % RPW;
    const uint4 * row = W + (std::size_t) (r < rows ? r : 0) * cin16;
    unsigned acc = 0;
    for (int c = 0; c < cin16; c += D) {
        uint4 v[D];
#pragma unroll
        for (int d = 0; d < D; ++d) v[d] = row[c + d];
#pragma unroll
        for (int d = 0; d < D; ++d) acc ^= v[d].x ^ v[d].w;
    }
    if (acc == 0x12345678u) sink[0] = acc;
}
// e: quad of lanes per row: lane%4 loads piece lane%4 of a 64-byte segment; 8 rows per wave
template <int D>
__global__ void k_quad(const uint4 * W, int rows, int cin16, unsigned * sink) {
    const int lane = threadIdx.x % 32, wave = threadIdx.x / 32, wpb = blockDim.x / 32;
    const int r = (blockIdx.x * wpb + wave) * 8 + lane / 4, q = lane % 4;
    const uint4 * row = W + (std::size_t) (r < rows ? r : 0) * cin16;
    unsigned acc = 0;
    for (int c = 0; c < cin16; c += 4 * D) {
        uint4 v[D];
#pragma unroll
        for (int d = 0; d < D; ++d) v[d] = row[c + d * 4 + q];
#pragma unroll
        for (int d = 0; d < D; ++d) acc ^= v[d].x ^ v[d].w;
    }
    if (acc == 0x12345678u) sink[0] = acc;
}
int main() {
    const int shapes[][2] = {{2048, 1024}, {1024, 1024}, {3072, 1024}, {1024, 3072}};
    const std::size_t maxW = 3072 * 3072, rot = (512u << 20) / 2 / maxW + 1;
    uint4 * W; unsigned * sink; hipMalloc(&W, maxW * rot * 2); hipMalloc(&sink, 4); hipMemset(W, 0x3c, maxW * rot * 2);
    auto time = [&](const char * name, int rows, int cin, auto launch) {
        const int cin16 = cin / 8; const std::size_t stride = (std::size_t) rows * cin / 8;
        for (int i = 0; i < 4; ++i) launch(W + (i % rot) * stride, rows, cin16);
        hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b);
        const int iters = 40; hipEventRecord(a);
        for (int i = 0; i < iters; ++i) launch(W + (i % rot) * stride, rows, cin16);
        hipEventRecord(b); hipEventSynchronize(b); float ms; hipEventElapsedTime(&ms, a, b);
        const double us = ms * 1e3 / iters; printf("  %-28s %6.1f us  %5.0f GB/s\n", name, us, 2.0 * rows * cin / us / 1e3);
    };
    for (auto & s : shapes) {
        const int rows = s[0], cin = s[1];
        printf("%d rows x %d (%.1f MB)\n", rows, cin, 2.0 * rows * cin / 1e6);
        time("rowlane 32 rows/wave D8", rows, cin, [&](const uint4 * w, int r, int c) { k_rowlane<8, 32><<<r / 32 / 4, 128>>>(w, r, c, sink); });
        time("rowlane 8 rows/wave x4dup D8", rows, cin, [&](const uint4 * w, int r, int c) { k_rowlane<8, 8><<<r / 8 / 4, 128>>>(w, r, c, sink); });
        time("rowlane 8 rows/wave x4dup D16", rows, cin, [&](const uint4 * w, int r, int c) { k_rowlane<16, 8><<<r / 8 / 4, 128>>>(w, r, c, sink); });
        time("quad 8 rows/wave D4", rows, cin, [&](const uint4 * w, int r, int c) { k_quad<4><<<r / 8 / 4, 128>>>(w, r, c, sink); });
        time("quad 8 rows/wave D8", rows, cin, [&](const uint4 * w, int r, int c) { k_quad<8><<<r / 8 / 4, 128>>>(w, r, c, sink); });
    }
    return 0;
}
