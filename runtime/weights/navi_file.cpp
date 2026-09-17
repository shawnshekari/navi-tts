#include "runtime/weights/navi_file.h"

#include "runtime/common/error.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <utility>

namespace navi {

const char * dtype_name(DType d) {
    switch (d) {
        case DType::F32:  return "f32";
        case DType::F16:  return "f16";
        case DType::BF16: return "bf16";
        case DType::I32:  return "i32";
        case DType::U8:   return "u8";
    }
    return "?";
}

std::size_t dtype_size(DType d) {
    switch (d) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::I32:  return 4;
        case DType::U8:   return 1;
    }
    return 0;
}

std::uint64_t TensorInfo::numel() const {
    std::uint64_t n = 1;
    for (auto d : dims) n *= d;
    return n;
}

std::string TensorInfo::shape_str() const {
    std::string s = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) s += (i ? ", " : "") + std::to_string(dims[i]);
    return s + "]";
}

namespace {

struct Cursor {
    const std::uint8_t * p;
    const std::uint8_t * end;
    const std::string &  path;

    void need(std::size_t n) const {
        if (static_cast<std::size_t>(end - p) < n) fail(path + ": truncated header");
    }
    template <class T> T take() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const auto n = take<std::uint32_t>();
        need(n);
        std::string s(reinterpret_cast<const char *>(p), n);
        p += n;
        return s;
    }
};

constexpr std::uint64_t PAGE = 4096;

} // namespace

NaviFile NaviFile::open(const std::string & path) {
    NaviFile f;
    f.path_ = path;
    f.fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (f.fd_ < 0) fail(path + ": " + std::strerror(errno));
    struct stat st{};
    if (fstat(f.fd_, &st) != 0) fail(path + ": fstat: " + std::strerror(errno));
    f.size_ = static_cast<std::uint64_t>(st.st_size);
    if (f.size_ < 32) fail(path + ": not a .navi file (too small)");
    void * m = mmap(nullptr, f.size_, PROT_READ, MAP_PRIVATE, f.fd_, 0);
    if (m == MAP_FAILED) fail(path + ": mmap: " + std::strerror(errno));
    f.map_ = static_cast<const std::uint8_t *>(m);
    // Weights are read once, front to back, during upload.
    (void) madvise(m, f.size_, MADV_SEQUENTIAL);

    Cursor c{f.map_, f.map_ + f.size_, path};
    if (std::memcmp(c.p, "NAVI", 4) != 0) fail(path + ": bad magic; not a .navi file");
    c.p += 4;
    const auto version = c.take<std::uint32_t>();
    if (version != 1) fail(path + ": .navi format version " + std::to_string(version) + " (this build reads 1)");
    f.data_offset_ = c.take<std::uint64_t>();
    const auto n_kv = c.take<std::uint32_t>();
    const auto n_t  = c.take<std::uint32_t>();
    const auto file_size = c.take<std::uint64_t>();
    if (file_size != f.size_) {
        fail(path + ": header says " + std::to_string(file_size) + " bytes, file is " + std::to_string(f.size_) +
             " (truncated or corrupt)");
    }
    if (f.data_offset_ % PAGE != 0 || f.data_offset_ > f.size_) fail(path + ": bad data_offset");
    c.end = f.map_ + f.data_offset_;

    for (std::uint32_t i = 0; i < n_kv; ++i) {
        std::string key = c.str();
        KV v;
        v.type = static_cast<KV::Type>(c.take<std::uint8_t>());
        switch (v.type) {
            case KV::Type::I64: v.i64 = c.take<std::int64_t>(); break;
            case KV::Type::F64: v.f64 = c.take<double>(); break;
            case KV::Type::STR: v.str = c.str(); break;
            case KV::Type::I64_ARRAY: {
                const auto n = c.take<std::uint32_t>();
                v.arr.reserve(n);
                for (std::uint32_t k = 0; k < n; ++k) v.arr.push_back(c.take<std::int64_t>());
                break;
            }
            default: fail(path + ": kv '" + key + "' has unknown type");
        }
        f.kv_.emplace(std::move(key), std::move(v));
    }
    f.tensors_.reserve(n_t);
    for (std::uint32_t i = 0; i < n_t; ++i) {
        TensorInfo t;
        t.name = c.str();
        t.dtype = static_cast<DType>(c.take<std::uint8_t>());
        if (dtype_size(t.dtype) == 0) fail(path + ": tensor '" + t.name + "' has unknown dtype");
        const auto ndim = c.take<std::uint8_t>();
        for (std::uint8_t k = 0; k < ndim; ++k) t.dims.push_back(c.take<std::uint64_t>());
        t.offset = c.take<std::uint64_t>();
        t.nbytes = c.take<std::uint64_t>();
        if (t.offset % PAGE != 0 || f.data_offset_ + t.offset + t.nbytes > f.size_) {
            fail(path + ": tensor '" + t.name + "' lies outside the file");
        }
        if (t.numel() * dtype_size(t.dtype) != t.nbytes) {
            fail(path + ": tensor '" + t.name + "' shape " + t.shape_str() + " does not match its byte length");
        }
        if (!f.index_.emplace(t.name, f.tensors_.size()).second) fail(path + ": duplicate tensor '" + t.name + "'");
        f.tensors_.push_back(std::move(t));
    }
    return f;
}

void NaviFile::close() {
    if (map_) { (void) munmap(const_cast<std::uint8_t *>(map_), size_); map_ = nullptr; }
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
}

NaviFile::~NaviFile() { close(); }

NaviFile::NaviFile(NaviFile && o) noexcept
    : path_(std::move(o.path_)), fd_(o.fd_), map_(o.map_), size_(o.size_), data_offset_(o.data_offset_),
      kv_(std::move(o.kv_)), tensors_(std::move(o.tensors_)), index_(std::move(o.index_)) {
    o.fd_ = -1;
    o.map_ = nullptr;
}

NaviFile & NaviFile::operator=(NaviFile && o) noexcept {
    if (this != &o) {
        close();
        path_ = std::move(o.path_); fd_ = o.fd_; map_ = o.map_; size_ = o.size_; data_offset_ = o.data_offset_;
        kv_ = std::move(o.kv_); tensors_ = std::move(o.tensors_); index_ = std::move(o.index_);
        o.fd_ = -1; o.map_ = nullptr;
    }
    return *this;
}

static const KV & kv_get(const std::map<std::string, KV> & m, std::string_view key, const std::string & path) {
    const auto it = m.find(std::string(key));
    if (it == m.end()) fail(path + ": missing key '" + std::string(key) + "'");
    return it->second;
}

bool NaviFile::has_kv(std::string_view key) const { return kv_.count(std::string(key)) != 0; }

std::int64_t NaviFile::kv_i64(std::string_view key) const {
    const KV & v = kv_get(kv_, key, path_);
    if (v.type != KV::Type::I64) fail(path_ + ": key '" + std::string(key) + "' is not an integer");
    return v.i64;
}

std::int64_t NaviFile::kv_i64_or(std::string_view key, std::int64_t dflt) const {
    return has_kv(key) ? kv_i64(key) : dflt;
}

double NaviFile::kv_f64(std::string_view key) const {
    const KV & v = kv_get(kv_, key, path_);
    if (v.type == KV::Type::F64) return v.f64;
    if (v.type == KV::Type::I64) return static_cast<double>(v.i64);
    fail(path_ + ": key '" + std::string(key) + "' is not a number");
}

std::string NaviFile::kv_str(std::string_view key) const {
    const KV & v = kv_get(kv_, key, path_);
    if (v.type != KV::Type::STR) fail(path_ + ": key '" + std::string(key) + "' is not a string");
    return v.str;
}

std::vector<std::int64_t> NaviFile::kv_i64_array(std::string_view key) const {
    const KV & v = kv_get(kv_, key, path_);
    if (v.type != KV::Type::I64_ARRAY) fail(path_ + ": key '" + std::string(key) + "' is not an integer array");
    return v.arr;
}

const TensorInfo * NaviFile::find(std::string_view name) const {
    const auto it = index_.find(std::string(name));
    return it == index_.end() ? nullptr : &tensors_[it->second];
}

const TensorInfo & NaviFile::get(std::string_view name) const {
    const TensorInfo * t = find(name);
    if (!t) fail(path_ + ": missing tensor '" + std::string(name) + "'");
    return *t;
}

const void * NaviFile::data(const TensorInfo & t) const {
    if (!map_) fail(path_ + ": file already closed");
    return map_ + data_offset_ + t.offset;
}

} // namespace navi
