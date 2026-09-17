#pragma once

// HIP device selection. One supported GPU or refuse to start (docs/DESIGN.md 1).

#include <hip/hip_runtime.h>

#include <cstddef>
#include <string>
#include <vector>

namespace navi {

struct DeviceInfo {
    int         index = -1;
    std::string name;            // marketing name, e.g. "AMD Radeon RX 7900 XTX"
    std::string arch;            // "gfx1100" - gcnArchName with the feature suffix stripped
    // hipDeviceProp_t::multiProcessorCount. On RDNA3 in the default WGP mode
    // this counts work-group processors, not CUs: the XTX (96 CU) reports 48.
    // The cooperative grid is sized from it at runtime, never from a constant.
    int         multiprocessors = 0;
    int         warp_size = 0;
    bool        cooperative_launch = false;
    int         max_threads_per_block = 0;
    std::size_t lds_per_block = 0;   // sharedMemPerBlock
    int         regs_per_block = 0;
    int         clock_khz = 0;
    int         mem_clock_khz = 0;
    int         mem_bus_width = 0;
    std::size_t vram_total = 0;
    std::size_t vram_free = 0;       // at open() time; the card is shared
    std::size_t l2_bytes = 0;
    int         hip_runtime_version = 0;
    int         hip_driver_version = 0;
};

// Architectures this binary can run on: exactly what its device code was
// compiled for (NAVI_GPU_ARCHS). Anything else is refused.
std::vector<std::string> supported_archs();

// Enumerate every visible GPU (for `navi-tts info`); does not select one.
std::vector<DeviceInfo> enumerate_devices();

class Device {
public:
    // Picks the first visible GPU whose arch is supported, sets it current,
    // spin-wait scheduling (host latency hygiene, docs/DESIGN.md 8b), and
    // creates the engine stream. Throws navi::Error if none qualifies.
    static Device open();

    ~Device();
    Device(Device &&) noexcept;
    Device & operator=(Device &&) noexcept;
    Device(const Device &) = delete;
    Device & operator=(const Device &) = delete;

    const DeviceInfo & info() const { return info_; }
    hipStream_t stream() const { return stream_; }

    // Refresh vram_free.
    void refresh_memory();

private:
    Device() = default;
    DeviceInfo  info_;
    hipStream_t stream_ = nullptr;
};

} // namespace navi
