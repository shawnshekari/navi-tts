#include "gpu_saturate.h"

#include <hip/hip_runtime.h>

#include <unistd.h>

#include <chrono>

namespace qwen3_tts {

// Occupies the CU for ~spin_cycles clock ticks. Pure compute spin: the
// point is block residency, not work.
__global__ void gpu_saturate_kernel(unsigned long long spin_cycles) {
    const unsigned long long start = clock64();
    while (clock64() - start < spin_cycles) {
    }
}

bool gpu_saturate_run(int seconds, int ready_fd) {
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return false;
    const int grid = prop.multiProcessorCount * 4;
    // ~0.5 s per launch at ~2.5 GHz; relaunch until the deadline so the
    // hog survives across scheduler gaps.
    const unsigned long long spin_cycles = 1250000000ull;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    bool first = true;
    while (std::chrono::steady_clock::now() < deadline) {
        gpu_saturate_kernel<<<grid, 256>>>(spin_cycles);
        if (hipGetLastError() != hipSuccess) return false;
        if (first) {
            first = false;
            const char r = 'R';
            if (write(ready_fd, &r, 1) != 1) return false;
        }
        if (hipDeviceSynchronize() != hipSuccess) return false;
    }
    return true;
}

} // namespace qwen3_tts
