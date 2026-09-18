#pragma once

// Reader for NumPy .npy (version 1.0/2.0, little-endian, C order). Tests only.

#include "runtime/common/error.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace navi {

struct Npy {
    std::string dtype;                  // "<f4", "<i4", "<i8", ...
    std::vector<std::size_t> shape;
    std::vector<std::uint8_t> data;

    std::size_t numel() const { std::size_t n = 1; for (auto d : shape) n *= d; return n; }

    template <class T> const T * as() const { return reinterpret_cast<const T *>(data.data()); }

    std::vector<float> as_f32() const {
        std::vector<float> out(numel());
        if (dtype == "<f4") std::memcpy(out.data(), data.data(), out.size() * 4);
        else if (dtype == "<f8") { const double * p = as<double>(); for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<float>(p[i]); }
        else fail("npy: cannot convert " + dtype + " to f32");
        return out;
    }
    std::vector<std::int32_t> as_i32() const {
        std::vector<std::int32_t> out(numel());
        if (dtype == "<i4") std::memcpy(out.data(), data.data(), out.size() * 4);
        else if (dtype == "<i8") { const std::int64_t * p = as<std::int64_t>(); for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<std::int32_t>(p[i]); }
        else fail("npy: cannot convert " + dtype + " to i32");
        return out;
    }

    static void save_f32(const std::string & path, const float * data, const std::vector<std::size_t> & shape) {
        std::string dims;
        for (std::size_t i = 0; i < shape.size(); ++i) dims += std::to_string(shape[i]) + (shape.size() == 1 || i + 1 < shape.size() ? ", " : "");
        if (shape.size() == 1) dims = std::to_string(shape[0]) + ",";
        std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + dims + "), }";
        while ((10 + header.size() + 1) % 64 != 0) header += ' ';
        header += '\n';
        std::ofstream f(path, std::ios::binary);
        if (!f) fail("npy: cannot write " + path);
        const std::uint16_t hlen = static_cast<std::uint16_t>(header.size());
        f.write("\x93NUMPY\x01\x00", 8);
        f.write(reinterpret_cast<const char *>(&hlen), 2);
        f.write(header.data(), static_cast<std::streamsize>(header.size()));
        std::size_t n = 1;
        for (auto d : shape) n *= d;
        f.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(n * 4));
    }

    static Npy load(const std::string & path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) fail("npy: cannot open " + path);
        char magic[6];
        f.read(magic, 6);
        if (std::memcmp(magic, "\x93NUMPY", 6) != 0) fail("npy: bad magic in " + path);
        unsigned char ver[2];
        f.read(reinterpret_cast<char *>(ver), 2);
        std::uint32_t hlen = 0;
        if (ver[0] == 1) { std::uint16_t h16; f.read(reinterpret_cast<char *>(&h16), 2); hlen = h16; }
        else { f.read(reinterpret_cast<char *>(&hlen), 4); }
        std::string header(hlen, '\0');
        f.read(header.data(), hlen);
        Npy n;
        auto field = [&](const char * key) {
            const auto p = header.find(key);
            if (p == std::string::npos) fail("npy: header lacks " + std::string(key));
            return p + std::strlen(key);
        };
        {
            auto p = field("'descr':");
            p = header.find('\'', p) + 1;
            n.dtype = header.substr(p, header.find('\'', p) - p);
        }
        if (header.find("'fortran_order': True") != std::string::npos) fail("npy: fortran order not supported");
        {
            auto p = header.find('(', field("'shape':")) + 1;
            const auto e = header.find(')', p);
            std::stringstream ss(header.substr(p, e - p));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (tok.find_first_of("0123456789") != std::string::npos) n.shape.push_back(std::stoul(tok));
            }
        }
        std::size_t item = 0;
        if (n.dtype == "<f4" || n.dtype == "<i4") item = 4;
        else if (n.dtype == "<f8" || n.dtype == "<i8") item = 8;
        else if (n.dtype == "<f2" || n.dtype == "<i2") item = 2;
        else fail("npy: unsupported dtype " + n.dtype);
        n.data.resize(n.numel() * item);
        f.read(reinterpret_cast<char *>(n.data.data()), static_cast<std::streamsize>(n.data.size()));
        if (!f) fail("npy: short read in " + path);
        return n;
    }
};

} // namespace navi
