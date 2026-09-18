#include "runtime/weights/device_weights.h"

#include "runtime/common/hip_check.h"

#include <chrono>
#include <utility>
#include <vector>

namespace navi {

namespace {
// 16-byte loads in the matvec inner loops; 256 keeps every tensor on its own
// cache-line group as well.
constexpr std::size_t ALIGN = 256;
constexpr std::size_t align_up(std::size_t n) { return (n + ALIGN - 1) / ALIGN * ALIGN; }
} // namespace

DeviceWeights DeviceWeights::upload(const NaviFile & file, const Select & select, const Groups & groups) {
    std::map<std::string, const TensorInfo *> by_name;
    std::map<std::string, std::size_t> group_of;   // member -> group index
    for (const TensorInfo & t : file.tensors()) by_name.emplace(t.name, &t);
    for (std::size_t g = 0; g < groups.size(); ++g)
        for (const std::string & n : groups[g]) group_of.emplace(n, g);

    std::vector<const TensorInfo *> chosen;
    std::vector<bool> group_done(groups.size(), false);
    std::size_t arena = 0, payload = 0;
    auto take = [&](const TensorInfo & t) {
        if (!(select ? select(t) : (t.dtype != DType::U8))) return;
        chosen.push_back(&t);
        arena += align_up(t.nbytes);
        payload += t.nbytes;
    };
    for (const TensorInfo & t : file.tensors()) {
        const auto g = group_of.find(t.name);
        if (g == group_of.end()) { take(t); continue; }
        if (group_done[g->second]) continue;
        group_done[g->second] = true;
        for (const std::string & n : groups[g->second]) {
            const auto m = by_name.find(n);
            if (m != by_name.end()) take(*m->second);
        }
    }

    DeviceWeights w;
    w.stats_.n_tensors = chosen.size();
    w.stats_.bytes = payload;
    w.stats_.arena_bytes = arena;
    if (arena == 0) return w;

    const auto t0 = std::chrono::steady_clock::now();
    const hipError_t e = hipMalloc(&w.arena_, arena);
    if (e != hipSuccess) {
        std::size_t free_b = 0, total_b = 0;
        (void) hipMemGetInfo(&free_b, &total_b);
        fail("cannot allocate " + std::to_string(arena >> 20) + " MiB of device memory for weights (" +
             hipGetErrorString(e) + "); " + std::to_string(free_b >> 20) + " MiB free of " +
             std::to_string(total_b >> 20));
    }
    std::size_t off = 0;
    for (const TensorInfo * t : chosen) {
        DeviceTensor d;
        d.info = *t;
        d.ptr = static_cast<std::uint8_t *>(w.arena_) + off;
        NAVI_HIP(hipMemcpy(d.ptr, file.data(*t), t->nbytes, hipMemcpyHostToDevice));
        w.tensors_.emplace(t->name, std::move(d));
        off += align_up(t->nbytes);
    }
    NAVI_HIP(hipDeviceSynchronize());
    w.stats_.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return w;
}

DeviceWeights::~DeviceWeights() {
    if (arena_) (void) hipFree(arena_);
}

DeviceWeights::DeviceWeights(DeviceWeights && o) noexcept
    : arena_(o.arena_), tensors_(std::move(o.tensors_)), stats_(o.stats_) {
    o.arena_ = nullptr;
}

DeviceWeights & DeviceWeights::operator=(DeviceWeights && o) noexcept {
    if (this != &o) {
        if (arena_) (void) hipFree(arena_);
        arena_ = o.arena_; tensors_ = std::move(o.tensors_); stats_ = o.stats_;
        o.arena_ = nullptr;
    }
    return *this;
}

const DeviceTensor * DeviceWeights::find(std::string_view name) const {
    const auto it = tensors_.find(std::string(name));
    return it == tensors_.end() ? nullptr : &it->second;
}

const DeviceTensor & DeviceWeights::get(std::string_view name) const {
    const DeviceTensor * t = find(name);
    if (!t) fail("weights: missing tensor '" + std::string(name) + "' on device");
    return *t;
}

} // namespace navi
