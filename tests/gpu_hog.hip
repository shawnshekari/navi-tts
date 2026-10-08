// The contention hog: fills every CU for N seconds so the
// cooperative frame kernel has to fight for residency. Two modes:
//   spin    resident blocks spinning on the wall clock (occupancy pressure)
//   stream  the same grid streaming through a 1 GiB buffer (bandwidth
//           pressure, what llama-server looks like)
// Relaunched until the deadline so it survives scheduler gaps. Used by
// tests/contention.sh, briefly.
//   gpu_hog SECONDS [BLOCKS_PER_CU=4] [spin|stream=stream]

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// wall_clock64() is the constant-rate counter (hipDeviceAttributeWallClockRate);
// clock64() does not count shader cycles on RDNA3 (a spin on it exits at once).
__global__ void spin(unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {}
}

// Every thread strides through the buffer `passes` times; the sum is written
// so the loads cannot be elided.
__global__ void stream(const float4 * __restrict__ buf, std::size_t n4, int passes, float * out) {
    float acc = 0.f;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (int p = 0; p < passes; ++p) {
        for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n4; i += stride) {
            const float4 v = buf[i];
            acc += v.x + v.y + v.z + v.w;
        }
    }
    if (acc == 12345.678f) out[0] = acc;
}

int main(int argc, char ** argv) {
    const int seconds = argc > 1 ? std::atoi(argv[1]) : 10;
    const int per_cu = argc > 2 ? std::atoi(argv[2]) : 4;
    const bool do_stream = argc > 3 ? std::strcmp(argv[3], "spin") != 0 : true;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) { std::fprintf(stderr, "gpu_hog: no device\n"); return 1; }
    const int grid = prop.multiProcessorCount * per_cu;
    int khz = 0;
    if (hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0) != hipSuccess || khz <= 0) { std::fprintf(stderr, "gpu_hog: no wall clock rate\n"); return 1; }
    const unsigned long long ticks = static_cast<unsigned long long>(khz) * 200ull;   // 0.2 s per spin launch
    const std::size_t bytes = std::size_t(1) << 30;
    float4 * buf = nullptr;
    float * out = nullptr;
    if (do_stream) {
        if (hipMalloc(&buf, bytes) != hipSuccess || hipMalloc(&out, 4) != hipSuccess) { std::fprintf(stderr, "gpu_hog: cannot allocate 1 GiB\n"); return 1; }
        hipMemset(buf, 0, bytes);
    }
    std::fprintf(stderr, "gpu_hog: %s, %d blocks x 256 threads on %s for %d s\n", do_stream ? "stream 1 GiB" : "spin", grid, prop.name, seconds);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    const auto t0 = std::chrono::steady_clock::now();
    std::size_t launches = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        if (do_stream) stream<<<grid, 256>>>(buf, bytes / sizeof(float4), 4, out);
        else spin<<<grid, 256>>>(ticks);
        if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) { std::fprintf(stderr, "gpu_hog: launch failed\n"); return 1; }
        ++launches;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (do_stream) std::fprintf(stderr, "gpu_hog: done, %zu launches, %.0f GB/s sustained\n", launches, launches * 4.0 * bytes / 1e9 / s);
    else std::fprintf(stderr, "gpu_hog: done, %zu launches (%.0f ms each)\n", launches, s * 1e3 / static_cast<double>(launches));
    return 0;
}
