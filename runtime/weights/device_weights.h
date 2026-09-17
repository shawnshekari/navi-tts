#pragma once

// Weights resident in device memory for the process lifetime (DESIGN 5.1):
// one arena, one upload per tensor straight from the .navi mapping.

#include "runtime/weights/navi_file.h"

#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <string_view>

namespace navi {

struct DeviceTensor {
    TensorInfo info;      // copy: the NaviFile is gone after upload
    void *     ptr = nullptr;
    template <class T> const T * as() const { return static_cast<const T *>(ptr); }
};

struct UploadStats {
    std::size_t n_tensors = 0;
    std::size_t bytes = 0;         // payload
    std::size_t arena_bytes = 0;   // with per-tensor alignment
    double      seconds = 0.0;
};

class DeviceWeights {
public:
    using Select = std::function<bool(const TensorInfo &)>;

    // Uploads every tensor `select` accepts (default: all but U8 blobs).
    // Synchronous; the caller closes the NaviFile afterwards.
    static DeviceWeights upload(const NaviFile & file, const Select & select = nullptr);

    ~DeviceWeights();
    DeviceWeights(DeviceWeights &&) noexcept;
    DeviceWeights & operator=(DeviceWeights &&) noexcept;
    DeviceWeights(const DeviceWeights &) = delete;
    DeviceWeights & operator=(const DeviceWeights &) = delete;

    const DeviceTensor * find(std::string_view name) const;
    const DeviceTensor & get(std::string_view name) const;   // throws if missing
    const std::map<std::string, DeviceTensor> & tensors() const { return tensors_; }
    const UploadStats & stats() const { return stats_; }

private:
    DeviceWeights() = default;
    void * arena_ = nullptr;
    std::map<std::string, DeviceTensor> tensors_;
    UploadStats stats_;
};

} // namespace navi
