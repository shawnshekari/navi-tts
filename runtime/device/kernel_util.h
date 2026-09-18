#pragma once

// Device-side helpers shared by the model kernels. wave32 on RDNA3.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

namespace navi::dev {

__device__ __forceinline__ float wave_sum(float v) {
    for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor(v, o, warpSize);
    return v;
}

__device__ __forceinline__ float wave_max(float v) {
    for (int o = warpSize / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, warpSize));
    return v;
}

// Block-wide sum, every thread gets the result. `red` is LDS scratch of >= 32 floats.
__device__ __forceinline__ float block_sum(float v, float * red) {
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = blockDim.x / warpSize;
    v = wave_sum(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    v = (threadIdx.x < static_cast<unsigned>(nw)) ? red[threadIdx.x] : 0.f;
    if (wid == 0) v = wave_sum(v);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    const float r = red[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float block_max(float v, float * red) {
    const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize, nw = blockDim.x / warpSize;
    v = wave_max(v);
    __syncthreads();
    if (lane == 0) red[wid] = v;
    __syncthreads();
    v = (threadIdx.x < static_cast<unsigned>(nw)) ? red[threadIdx.x] : -INFINITY;
    if (wid == 0) v = wave_max(v);
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    const float r = red[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float half_bits_to_float(unsigned short h) {
    _Float16 f;
    __builtin_memcpy(&f, &h, sizeof f);
    return static_cast<float>(f);
}

} // namespace navi::dev
