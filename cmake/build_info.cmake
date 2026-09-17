# Regenerates build_info.h at every build so the git hash in bench JSON is
# never stale. Invoked by the navi_build_info target; copy_if_different keeps
# incremental builds cheap.
execute_process(COMMAND git -C "${SRC}" rev-parse --short=12 HEAD
                OUTPUT_VARIABLE NAVI_GIT_HASH OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT NAVI_GIT_HASH)
    set(NAVI_GIT_HASH "unknown")
endif()
execute_process(COMMAND git -C "${SRC}" status --porcelain --untracked-files=no
                OUTPUT_VARIABLE _dirty OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(_dirty)
    set(NAVI_GIT_HASH "${NAVI_GIT_HASH}-dirty")
endif()
configure_file("${SRC}/runtime/common/build_info.h.in" "${OUT}.tmp" @ONLY)
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${OUT}.tmp" "${OUT}")
