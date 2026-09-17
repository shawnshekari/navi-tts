#pragma once

#include <stdexcept>
#include <string>

namespace navi {

// Setup-time failures (device, weights, config) throw; the request path
// reports status instead so a bad request never unwinds the serving thread.
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

[[noreturn]] inline void fail(const std::string & msg) { throw Error(msg); }

} // namespace navi
