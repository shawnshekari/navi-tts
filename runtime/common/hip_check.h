#pragma once

#include "runtime/common/error.h"

#include <hip/hip_runtime.h>

#include <string>

namespace navi {

inline void hip_check(hipError_t e, const char * what, const char * file, int line) {
    if (e != hipSuccess) {
        fail(std::string(what) + " failed: " + hipGetErrorString(e) + " (" + hipGetErrorName(e) + ") at " +
             file + ":" + std::to_string(line));
    }
}

} // namespace navi

#define NAVI_HIP(call) ::navi::hip_check((call), #call, __FILE__, __LINE__)
