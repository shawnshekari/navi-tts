cmake_minimum_required(VERSION 3.28)

# The ROCm clang compiles host and device alike; it is selected by the preset
# (CMAKE_CXX_COMPILER) before project() so the ABI check runs on the right one.
project(navi-tts LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)   # no C++20 modules; CMake 3.28 would otherwise want clang-scan-deps
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- ROCm ------------------------------------------------------------------
# No hipcc, no CMake HIP language: HIP sources are plain CXX sources compiled
# with `-x hip --offload-arch=...` (docs/DESIGN.md 8b). Host-only files that
# call the HIP runtime API only need the headers and __HIP_PLATFORM_AMD__.
set(NAVI_ROCM_ROOT "$ENV{ROCM_PATH}" CACHE PATH "ROCm install (therock tarball)")
if(NOT NAVI_ROCM_ROOT)
    set(NAVI_ROCM_ROOT "$ENV{HOME}/tools/therock-tarball/install")
endif()
if(NOT EXISTS "${NAVI_ROCM_ROOT}/include/hip/hip_runtime.h")
    message(FATAL_ERROR "NAVI_ROCM_ROOT=${NAVI_ROCM_ROOT} has no hip/hip_runtime.h")
endif()
# ROCm floor (DESIGN 9): at least 10.1, whatever is newer is fine.
set(NAVI_ROCM_MIN_VERSION "10.1" CACHE STRING "Oldest ROCm this project builds against")
if(NOT EXISTS "${NAVI_ROCM_ROOT}/.info/version")
    message(FATAL_ERROR "NAVI_ROCM_ROOT=${NAVI_ROCM_ROOT} has no .info/version; cannot check the ROCm floor")
endif()
file(STRINGS "${NAVI_ROCM_ROOT}/.info/version" NAVI_ROCM_VERSION LIMIT_COUNT 1)
string(REGEX MATCH "^[0-9]+\\.[0-9]+(\\.[0-9]+)?" _rocm_ver "${NAVI_ROCM_VERSION}")
if(NOT _rocm_ver OR _rocm_ver VERSION_LESS NAVI_ROCM_MIN_VERSION)
    message(FATAL_ERROR "ROCm ${NAVI_ROCM_VERSION} at ${NAVI_ROCM_ROOT}; navi-tts needs ${NAVI_ROCM_MIN_VERSION} or newer")
endif()
message(STATUS "ROCm ${NAVI_ROCM_VERSION} (floor ${NAVI_ROCM_MIN_VERSION}) at ${NAVI_ROCM_ROOT}")
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR NOT CMAKE_CXX_COMPILER MATCHES "${NAVI_ROCM_ROOT}")
    message(FATAL_ERROR "CMAKE_CXX_COMPILER must be the ROCm clang under ${NAVI_ROCM_ROOT} "
                        "(got ${CMAKE_CXX_COMPILER}); use a preset from CMakePresets.json")
endif()

# gfx1100 only until M3 adds gfx1151 (docs/DESIGN.md 8). Semicolon list.
set(NAVI_GPU_ARCHS "gfx1100" CACHE STRING "AMD GPU targets to compile device code for")
set(_offload_flags "")
foreach(arch IN LISTS NAVI_GPU_ARCHS)
    list(APPEND _offload_flags "--offload-arch=${arch}")
endforeach()
string(REPLACE ";" "," NAVI_GPU_ARCHS_CSV "${NAVI_GPU_ARCHS}")

find_library(NAVI_HIP_LIB amdhip64 PATHS "${NAVI_ROCM_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_library(navi_rocm INTERFACE)
target_include_directories(navi_rocm SYSTEM INTERFACE "${NAVI_ROCM_ROOT}/include")
target_compile_definitions(navi_rocm INTERFACE __HIP_PLATFORM_AMD__)
target_link_libraries(navi_rocm INTERFACE "${NAVI_HIP_LIB}")
target_link_options(navi_rocm INTERFACE "-Wl,-rpath,${NAVI_ROCM_ROOT}/lib")

# Mark a target's .hip sources for device compilation.
function(navi_hip_sources target)
    foreach(src IN LISTS ARGN)
        set_source_files_properties(${src} PROPERTIES
            LANGUAGE CXX
            COMPILE_OPTIONS "-x;hip;${_offload_flags};--rocm-path=${NAVI_ROCM_ROOT}")
    endforeach()
    target_sources(${target} PRIVATE ${ARGN})
endfunction()

# --- host flags (docs/DESIGN.md 8b) ------------------------------------------
# -O3 -march=znver3 -flto, no -ffast-math until the parity gates pass without it.
add_compile_options(-Wall -Wextra -Wno-unused-parameter)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=znver3 -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -march=znver3 -flto -g -DNDEBUG")
add_link_options(-fuse-ld=lld)
if(CMAKE_BUILD_TYPE MATCHES "Rel")
    add_link_options(-flto)
endif()

# build_info.h: git hash (regenerated at every build, see cmake/build_info.cmake),
# ROCm version, GPU targets.
set(_build_info "${CMAKE_BINARY_DIR}/generated/navi/build_info.h")
add_custom_target(navi_build_info ALL
    COMMAND "${CMAKE_COMMAND}"
        "-DSRC=${CMAKE_SOURCE_DIR}" "-DOUT=${_build_info}"
        "-DNAVI_ROCM_VERSION=${NAVI_ROCM_VERSION}" "-DNAVI_ROCM_MIN_VERSION=${NAVI_ROCM_MIN_VERSION}"
        "-DNAVI_GPU_ARCHS_CSV=${NAVI_GPU_ARCHS_CSV}"
        "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        -P "${CMAKE_SOURCE_DIR}/cmake/build_info.cmake"
    BYPRODUCTS "${_build_info}"
    COMMENT "build_info.h"
    VERBATIM)

# --- targets -----------------------------------------------------------------
add_library(navi_runtime STATIC
    runtime/common/json.cpp
    runtime/device/device.cpp
    runtime/text/unicode.cpp
    runtime/text/bpe.cpp
    runtime/weights/navi_file.cpp
    runtime/weights/device_weights.cpp
    runtime/audio/wav.cpp
    runtime/bench/bench.cpp
)
target_include_directories(navi_runtime PUBLIC "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/generated")
target_link_libraries(navi_runtime PUBLIC navi_rocm)
add_dependencies(navi_runtime navi_build_info)

add_library(navi_qwen3tts STATIC
    model/qwen3tts/params.cpp
    model/qwen3tts/tokenizer.cpp
    model/qwen3tts/graph.cpp
)
navi_hip_sources(navi_qwen3tts model/qwen3tts/vocoder.hip model/qwen3tts/talker.hip model/qwen3tts/frame.hip)
target_link_libraries(navi_qwen3tts PUBLIC navi_runtime)

add_executable(navi-tts runtime/cli/main.cpp)
target_link_libraries(navi-tts PRIVATE navi_qwen3tts navi_runtime)

# --- tests -------------------------------------------------------------------
# Parity gates against tests/reference (DESIGN 6). They need the converted model.
set(NAVI_TEST_MODEL "${CMAKE_SOURCE_DIR}/models/qwen3-tts-0.6b-f16.navi" CACHE FILEPATH ".navi used by the tests")
enable_testing()
add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE navi_qwen3tts navi_runtime)
add_test(NAME tokenizer COMMAND test_tokenizer "${NAVI_TEST_MODEL}" "${CMAKE_SOURCE_DIR}/tests/reference/tokenizer_cases.json")
add_executable(test_vocoder tests/test_vocoder.cpp)
target_link_libraries(test_vocoder PRIVATE navi_qwen3tts navi_runtime)
add_test(NAME vocoder COMMAND test_vocoder "${NAVI_TEST_MODEL}" "${CMAKE_SOURCE_DIR}/tests/reference" "${CMAKE_BINARY_DIR}/vocoder_out.wav")
add_executable(test_prefill tests/test_prefill.cpp)
target_link_libraries(test_prefill PRIVATE navi_qwen3tts navi_runtime)
add_test(NAME prefill COMMAND test_prefill "${NAVI_TEST_MODEL}" "${CMAKE_SOURCE_DIR}/tests/reference")
add_executable(test_frame tests/test_frame.cpp)
target_link_libraries(test_frame PRIVATE navi_qwen3tts navi_runtime)
add_test(NAME frame COMMAND test_frame "${NAVI_TEST_MODEL}" "${CMAKE_SOURCE_DIR}/tests/reference")
