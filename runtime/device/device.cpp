#include "runtime/device/device.h"

#include "navi/build_info.h"
#include "runtime/common/hip_check.h"

#include <hip/hip_version.h>

#include <sstream>
#include <utility>

namespace navi {

std::vector<std::string> supported_archs() {
    std::vector<std::string> out;
    std::stringstream ss(NAVI_GPU_ARCHS);
    for (std::string a; std::getline(ss, a, ',');) {
        if (!a.empty()) out.push_back(a);
    }
    return out;
}

static std::string strip_arch(const char * gcn_arch_name) {
    // "gfx1100:sramecc+:xnack-" -> "gfx1100"
    std::string s(gcn_arch_name);
    const auto colon = s.find(':');
    return colon == std::string::npos ? s : s.substr(0, colon);
}

static DeviceInfo query(int index) {
    hipDeviceProp_t p{};
    NAVI_HIP(hipGetDeviceProperties(&p, index));
    DeviceInfo d;
    d.index                 = index;
    d.name                  = p.name;
    d.arch                  = strip_arch(p.gcnArchName);
    d.multiprocessors       = p.multiProcessorCount;
    d.warp_size             = p.warpSize;
    d.cooperative_launch    = p.cooperativeLaunch != 0;
    d.max_threads_per_block = p.maxThreadsPerBlock;
    d.lds_per_block         = p.sharedMemPerBlock;
    d.regs_per_block        = p.regsPerBlock;
    d.clock_khz             = p.clockRate;
    d.mem_clock_khz         = p.memoryClockRate;
    d.mem_bus_width         = p.memoryBusWidth;
    d.vram_total            = p.totalGlobalMem;
    d.l2_bytes              = p.l2CacheSize;
    NAVI_HIP(hipRuntimeGetVersion(&d.hip_runtime_version));
    NAVI_HIP(hipDriverGetVersion(&d.hip_driver_version));
    return d;
}

std::vector<DeviceInfo> enumerate_devices() {
    int n = 0;
    const hipError_t e = hipGetDeviceCount(&n);
    if (e == hipErrorNoDevice) return {};
    hip_check(e, "hipGetDeviceCount", __FILE__, __LINE__);
    std::vector<DeviceInfo> out;
    for (int i = 0; i < n; ++i) out.push_back(query(i));
    return out;
}

// The HIP runtime that is loaded must be at least the one the binary was
// compiled against (hip_version.h): newer is fine, older is refused.
// hipRuntimeGetVersion encodes major * 10^7 + minor * 10^5 + patch.
static void check_hip_runtime() {
    int rt = 0;
    NAVI_HIP(hipRuntimeGetVersion(&rt));
    const int rt_major = rt / 10000000, rt_minor = (rt / 100000) % 100;
    if (rt_major < HIP_VERSION_MAJOR || (rt_major == HIP_VERSION_MAJOR && rt_minor < HIP_VERSION_MINOR)) {
        fail("HIP runtime " + std::to_string(rt_major) + "." + std::to_string(rt_minor) + " is older than the " +
             std::to_string(HIP_VERSION_MAJOR) + "." + std::to_string(HIP_VERSION_MINOR) +
             " this binary was built against (ROCm " NAVI_ROCM_VERSION "); ROCm " NAVI_ROCM_MIN_VERSION
             " or newer is required - check LD_LIBRARY_PATH");
    }
}

Device Device::open() {
    check_hip_runtime();
    const auto archs = supported_archs();
    const auto devs  = enumerate_devices();
    if (devs.empty()) fail("no HIP device visible (is the therock ROCm on LD_LIBRARY_PATH?)");

    for (const auto & d : devs) {
        bool ok = false;
        for (const auto & a : archs) ok = ok || (a == d.arch);
        if (!ok) continue;
        if (!d.cooperative_launch) {
            fail("device " + std::to_string(d.index) + " (" + d.name + ", " + d.arch +
                 ") does not support cooperative launch; the frame kernel needs it");
        }
        Device dev;
        dev.info_ = d;
        NAVI_HIP(hipSetDevice(d.index));
        // Spin on completion: one launch, one wait per frame; a yield costs
        // more than the barrier we are waiting for.
        NAVI_HIP(hipSetDeviceFlags(hipDeviceScheduleSpin));
        NAVI_HIP(hipStreamCreateWithFlags(&dev.stream_, hipStreamNonBlocking));
        dev.refresh_memory();
        return dev;
    }

    std::string seen;
    for (const auto & d : devs) seen += (seen.empty() ? "" : ", ") + d.arch + " (" + d.name + ")";
    std::string want;
    for (const auto & a : archs) want += (want.empty() ? "" : ", ") + a;
    fail("no supported GPU: this build runs on [" + want + "] only; visible: " + seen +
         ". navi-tts targets the RX 7900 XTX and Strix Halo by design (docs/DESIGN.md 1).");
}

void Device::refresh_memory() {
    std::size_t free_b = 0, total_b = 0;
    NAVI_HIP(hipMemGetInfo(&free_b, &total_b));
    info_.vram_free  = free_b;
    info_.vram_total = total_b;
}

Device::~Device() {
    if (stream_) (void) hipStreamDestroy(stream_);
}

Device::Device(Device && o) noexcept : info_(std::move(o.info_)), stream_(o.stream_) { o.stream_ = nullptr; }

Device & Device::operator=(Device && o) noexcept {
    if (this != &o) {
        if (stream_) (void) hipStreamDestroy(stream_);
        info_   = std::move(o.info_);
        stream_ = o.stream_;
        o.stream_ = nullptr;
    }
    return *this;
}

} // namespace navi
