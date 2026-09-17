#pragma once

// .navi reader: mmap the file, parse the KV and tensor tables (docs/navi-format.md).
// Lives only as long as the upload; the mapping is dropped afterwards.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace navi {

enum class DType : std::uint8_t { F32 = 0, F16 = 1, BF16 = 2, I32 = 3, U8 = 4 };

const char * dtype_name(DType d);
std::size_t  dtype_size(DType d);

struct TensorInfo {
    std::string               name;
    DType                     dtype = DType::F32;
    std::vector<std::uint64_t> dims;   // row-major, outermost first
    std::uint64_t             offset = 0;   // from data_offset
    std::uint64_t             nbytes = 0;

    std::uint64_t numel() const;
    std::string   shape_str() const;
};

struct KV {
    enum class Type : std::uint8_t { I64 = 0, F64 = 1, STR = 2, I64_ARRAY = 3 };
    Type                      type = Type::I64;
    std::int64_t              i64 = 0;
    double                    f64 = 0.0;
    std::string               str;
    std::vector<std::int64_t> arr;
};

class NaviFile {
public:
    static NaviFile open(const std::string & path);
    ~NaviFile();
    NaviFile(NaviFile &&) noexcept;
    NaviFile & operator=(NaviFile &&) noexcept;
    NaviFile(const NaviFile &) = delete;
    NaviFile & operator=(const NaviFile &) = delete;

    const std::string & path() const { return path_; }
    std::uint64_t file_size() const { return size_; }
    std::uint64_t data_offset() const { return data_offset_; }

    const std::map<std::string, KV> & kv() const { return kv_; }
    const std::vector<TensorInfo> & tensors() const { return tensors_; }

    // Typed KV access; missing key or wrong type throws navi::Error.
    std::int64_t  kv_i64(std::string_view key) const;
    double        kv_f64(std::string_view key) const;   // accepts I64 too
    std::string   kv_str(std::string_view key) const;
    std::vector<std::int64_t> kv_i64_array(std::string_view key) const;
    bool          has_kv(std::string_view key) const;
    std::int64_t  kv_i64_or(std::string_view key, std::int64_t dflt) const;

    const TensorInfo * find(std::string_view name) const;
    const TensorInfo & get(std::string_view name) const;   // throws if missing

    // Pointer into the mapping for a tensor's bytes.
    const void * data(const TensorInfo & t) const;

    // Unmap now (the destructor does it too).
    void close();

private:
    NaviFile() = default;
    std::string   path_;
    int           fd_ = -1;
    const std::uint8_t * map_ = nullptr;
    std::uint64_t size_ = 0;
    std::uint64_t data_offset_ = 0;
    std::map<std::string, KV> kv_;
    std::vector<TensorInfo>   tensors_;
    std::map<std::string, std::size_t> index_;
};

} // namespace navi
