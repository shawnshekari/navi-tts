// Microbench of the frame kernel's per-event sampling pieces, one 256-thread
// block: the old bitonic sort + serial draw against the radix select and the
// compacted draw that replaced them (2026-09-17: 46+102 us -> 4.6+8.9 at
// V=3072, 21+68 -> 3.6+6.9 at V=2048; draws bit-identical). Build as matvec.hip.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>
constexpr int BLOCK = 256;

__device__ void block_sort_desc(float * xs, int N) {
    for (int k = 2; k <= N; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            for (int i = threadIdx.x; i < (N >> 1); i += blockDim.x) {
                const int l = ((i & ~(j - 1)) << 1) | (i & (j - 1)), m = l + j;
                const bool desc = (l & k) == 0;
                const float x = xs[l], y = xs[m];
                if (desc ? (x < y) : (x > y)) { xs[l] = y; xs[m] = x; }
            }
        }
    __syncthreads();
}
__device__ int draw_serial(const float * p, int V, float inv_sum, float u) {
    float cum = 0.f; int chosen = V - 1;
    for (int i = 0; i < V; ++i) { cum += p[i] * inv_sum; if (cum >= u) { chosen = i; break; } }
    return chosen;
}
// monotone float->uint key (descending select = largest keys)
__device__ __forceinline__ unsigned fkey(float f) { unsigned u = __float_as_uint(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
// k-th largest value of xs[0..V) via 4-pass radix select on 8-bit digits. hist: 256 uints, red: a few ints.
__device__ float block_kth_largest(const float * xs, int V, int k, unsigned * hist, int * red) {
    unsigned prefix = 0, mask = 0;   // digits already fixed
    int need = k;                    // rank within the candidates (1-based)
    for (int pass = 3; pass >= 0; --pass) {
        const int shift = pass * 8;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            const unsigned key = fkey(xs[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {   // walk digits from the top; 256 steps
            int acc = 0, d = 255;
            for (; d >= 0; --d) { const int c = static_cast<int>(hist[d]); if (acc + c >= need) break; acc += c; }
            red[0] = d; red[1] = need - acc;
        }
        __syncthreads();
        const unsigned digit = static_cast<unsigned>(red[0]);
        need = red[1];
        prefix |= digit << shift;
        mask |= 0xFFu << shift;
        __syncthreads();
    }
    unsigned u = prefix;   // the key of the k-th largest
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}

// Variant: 256-bin histogram walked in parallel - one thread per bin, the count
// of keys above each bin by a block scan (descending), the bin holding `need` found by ballot.
__device__ float block_kth_largest2(const float * xs, int V, int k, unsigned * hist, int * red) {
    unsigned prefix = 0, mask = 0;
    int need = k;
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
    for (int pass = 3; pass >= 0; --pass) {
        const int shift = pass * 8;
        hist[threadIdx.x] = 0;          // BLOCK == 256 bins
        __syncthreads();
        for (int i = threadIdx.x; i < V; i += blockDim.x) {
            const unsigned key = fkey(xs[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 0xFF], 1u);
        }
        __syncthreads();
        // above[d] = count of keys with digit > d  ->  exclusive scan from the top. thread t owns bin 255 - t.
        const int d = 255 - static_cast<int>(threadIdx.x);
        const int c = static_cast<int>(hist[d]);
        int incl = c;                    // inclusive scan within the wave (top digits first)
        for (int o = 1; o < warpSize; o <<= 1) { const int v = __shfl_up(incl, o, warpSize); if (lane >= o) incl += v; }
        __syncthreads();
        if (lane == warpSize - 1) red[wid] = incl;   // wave totals
        __syncthreads();
        int base = 0;
        for (int w = 0; w < wid; ++w) base += red[w];
        const int above = base + incl - c;           // keys strictly above this bin
        const bool hit = above < need && above + c >= need;
        const unsigned long long m = __ballot(hit);
        __syncthreads();
        if (hit) { red[8] = d; red[9] = need - above; }
        __syncthreads();
        need = red[9];
        prefix |= static_cast<unsigned>(red[8]) << shift;
        mask |= 0xFFu << shift;
        (void) m;
        __syncthreads();
    }
    unsigned u = prefix;
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}
// Draw over the nonzero entries only, in index order: bit-identical to the serial walk
// (adding an exact 0 does not change the sum), but thread 0 visits <= k entries.
__device__ int draw_compact(const float * p, int V, float inv_sum, float u, int * idx, int * red) {
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = blockDim.x / warpSize;
    if (threadIdx.x == 0) red[0] = 0;
    __syncthreads();
    for (int base = 0; base < V; base += blockDim.x) {
        const int i = base + threadIdx.x;
        const bool nz = i < V && p[i] != 0.f;
        const unsigned long long m = __ballot(nz);
        const int in_wave = __popcll(m & ((1ull << lane) - 1ull));
        const int wave_n = __popcll(m);
        if (lane == 0) red[1 + wid] = wave_n;
        __syncthreads();
        int before = red[0];
        for (int w = 0; w < wid; ++w) before += red[1 + w];
        if (nz) idx[before + in_wave] = i;
        __syncthreads();
        if (threadIdx.x == 0) { int t = 0; for (int w = 0; w < nw; ++w) t += red[1 + w]; red[0] += t; }
        __syncthreads();
    }
    int chosen = V - 1;
    if (threadIdx.x == 0) {
        const int n = red[0];
        float cum = 0.f;
        for (int j = 0; j < n; ++j) { const int i = idx[j]; cum += p[i] * inv_sum; if (cum >= u) { chosen = i; break; } }
    }
    return chosen;   // meaningful on thread 0
}
__global__ void k_select2(float * out, int V, int k, int iters) {
    __shared__ float xs[4096]; __shared__ unsigned hist[256]; __shared__ int red[16];
    float t = 0;
    for (int it = 0; it < iters; ++it) {
        for (int i = threadIdx.x; i < V; i += blockDim.x) xs[i] = __sinf(i * 0.37f + it);
        __syncthreads();
        t += block_kth_largest2(xs, V, k, hist, red);
    }
    if (threadIdx.x == 0) out[0] = t;
}
__global__ void k_draw2(float * out, int V, int iters) {   // 50 nonzero entries spread over V
    __shared__ float xs[4096]; __shared__ int idx[4096]; __shared__ int red[16];
    for (int i = threadIdx.x; i < V; i += blockDim.x) xs[i] = (i % (V / 50) == 0) ? 1.0f : 0.f;
    __syncthreads();
    int c = 0;
    for (int it = 0; it < iters; ++it) { c += draw_compact(xs, V, 1.0f / 50, 0.999f, idx, red); __syncthreads(); }
    if (threadIdx.x == 0) out[0] = c;
}
__global__ void k_check2(float * out, int V, int k) {
    __shared__ float xs[4096]; __shared__ float ys[4096]; __shared__ unsigned hist[256]; __shared__ int red[16]; __shared__ int idx[4096];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) { const float v = i < V ? __sinf(i * 0.37f) * 7.0f - 3.0f : -INFINITY; xs[i] = v; ys[i] = v; }
    __syncthreads();
    const float sel = block_kth_largest2(xs, V, k, hist, red);
    block_sort_desc(ys, 4096);
    // draw equivalence: probabilities = top-k of xs
    for (int i = threadIdx.x; i < V; i += blockDim.x) ys[i] = (xs[i] >= sel) ? __expf(xs[i]) : 0.f;
    __syncthreads();
    float s = 0; for (int i = 0; i < V; ++i) s += ys[i];
    int a = 0, b = 0;
    for (int q = 0; q < 200; ++q) {
        const float u = (q + 0.5f) / 200.f;
        const int c1 = draw_compact(ys, V, 1.f / s, u, idx, red);
        int c0 = 0; if (threadIdx.x == 0) c0 = draw_serial(ys, V, 1.f / s, u);
        if (threadIdx.x == 0) { a += (c0 == c1); b++; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { out[0] = sel; out[1] = a == b ? 1.f : 0.f; }
}

__global__ void k_sort(float * out, int N, int iters) {
    __shared__ float xs[4096];
    for (int it = 0; it < iters; ++it) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) xs[i] = __sinf(i * 0.37f + it);
        block_sort_desc(xs, N);
    }
    if (threadIdx.x == 0) out[0] = xs[49];
}
__global__ void k_draw(float * out, int V, int iters) {
    __shared__ float xs[4096];
    for (int i = threadIdx.x; i < V; i += blockDim.x) xs[i] = 1.0f;
    __syncthreads();
    int c = 0;
    for (int it = 0; it < iters; ++it) { if (threadIdx.x == 0) c += draw_serial(xs, V, 1.0f / V, 0.999f); __syncthreads(); }
    if (threadIdx.x == 0) out[0] = c;
}
__global__ void k_select(float * out, int V, int k, int iters) {
    __shared__ float xs[4096]; __shared__ unsigned hist[256]; __shared__ int red[4];
    float t = 0;
    for (int it = 0; it < iters; ++it) {
        for (int i = threadIdx.x; i < V; i += blockDim.x) xs[i] = __sinf(i * 0.37f + it);
        __syncthreads();
        t += block_kth_largest(xs, V, k, hist, red);
    }
    if (threadIdx.x == 0) out[0] = t;
}
__global__ void k_check(float * out, int V, int k) {   // select vs sort agreement
    __shared__ float xs[4096]; __shared__ float ys[4096]; __shared__ unsigned hist[256]; __shared__ int red[4];
    for (int i = threadIdx.x; i < 4096; i += blockDim.x) { const float v = i < V ? __sinf(i * 0.37f) * 7.0f - 3.0f : -INFINITY; xs[i] = v; ys[i] = v; }
    __syncthreads();
    const float sel = block_kth_largest(xs, V, k, hist, red);
    block_sort_desc(ys, 4096);
    if (threadIdx.x == 0) { out[0] = sel; out[1] = ys[k - 1]; }
}

int main() {
    float * d; hipMalloc(&d, 64); hipEvent_t a, b; hipEventCreate(&a); hipEventCreate(&b); float ms; float h[2];
    const int iters = 2000;
    auto time = [&](const char * name, auto launch) { launch(); hipDeviceSynchronize(); hipEventRecord(a); launch(); hipEventRecord(b); hipEventSynchronize(b); hipEventElapsedTime(&ms, a, b); printf("%-28s %7.1f us/call\n", name, ms * 1e3 / iters); };
    time("sort 4096 (talker)", [&] { k_sort<<<1, BLOCK>>>(d, 4096, iters); });
    time("sort 2048 (cp)", [&] { k_sort<<<1, BLOCK>>>(d, 2048, iters); });
    time("draw serial 3072", [&] { k_draw<<<1, BLOCK>>>(d, 3072, iters); });
    time("draw serial 2048", [&] { k_draw<<<1, BLOCK>>>(d, 2048, iters); });
    time("radix select 3072 k=50", [&] { k_select<<<1, BLOCK>>>(d, 3072, 50, iters); });
    time("radix select 2048 k=50", [&] { k_select<<<1, BLOCK>>>(d, 2048, 50, iters); });
    time("select2 3072 k=50", [&] { k_select2<<<1, BLOCK>>>(d, 3072, 50, iters); });
    time("select2 2048 k=50", [&] { k_select2<<<1, BLOCK>>>(d, 2048, 50, iters); });
    time("draw compact 3072", [&] { k_draw2<<<1, BLOCK>>>(d, 3072, iters); });
    time("draw compact 2048", [&] { k_draw2<<<1, BLOCK>>>(d, 2048, iters); });
    k_check2<<<1, BLOCK>>>(d, 3072, 50); hipMemcpy(h, d, 8, hipMemcpyDeviceToHost); printf("select2 %g; draws identical over 200 u: %s\n", h[0], h[1] == 1.f ? "yes" : "NO");
    k_check2<<<1, BLOCK>>>(d, 2048, 50); hipMemcpy(h, d, 8, hipMemcpyDeviceToHost); printf("select2 %g; draws identical over 200 u: %s\n", h[0], h[1] == 1.f ? "yes" : "NO");
    k_check<<<1, BLOCK>>>(d, 3072, 50); hipMemcpy(h, d, 8, hipMemcpyDeviceToHost); printf("select %g vs sort %g: %s\n", h[0], h[1], h[0] == h[1] ? "equal" : "DIFFER");
    k_check<<<1, BLOCK>>>(d, 2048, 50); hipMemcpy(h, d, 8, hipMemcpyDeviceToHost); printf("select %g vs sort %g: %s\n", h[0], h[1], h[0] == h[1] ? "equal" : "DIFFER");
    return 0;
}
