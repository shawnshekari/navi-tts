# ROCm toolchain notes (from the fork era)

Copied from ~/.local/share/rocm-clangrt-overlay/README:

    Workaround for the TheRock ROCm nightly in the navi31-llama toolbox: its clang 23
    ships no host libclang_rt.builtins.a, and /opt/rocm/lib/cmake/hip-lang forces
    --rtlib=compiler-rt, so every HIP link fails. This dir mirrors the ROCm clang
    resource dir (symlinks) and adds Fedora compiler-rt 22's x86_64 builtins.
    Use with:  -DCMAKE_HIP_FLAGS=-resource-dir=$HOME/.local/share/rocm-clangrt-overlay
    Durable alternative (root in the container):
      dnf install -y compiler-rt && ln -s /usr/lib/clang/22/lib/x86_64-redhat-linux-gnu/libclang_rt.builtins.a \
         /opt/rocm/lib/llvm/lib/clang/23/lib/x86_64-unknown-linux-gnu/

## Strix Halo mini PC (M3, 2026-09-18)

The Strix mini PC: Fedora 43, Ryzen AI MAX+ 395 (gfx1151, 20 WGPs), 124 GB unified.
ROCm is the therock gfx1151 tarball extracted to `~/tools/therock-tarball/install`
(same layout as the workstation, so the presets' compiler path holds); cmake and
ninja are `uv tool install cmake ninja` (user-space, `~/.local/bin`).

The host has no `libstdc++-devel`, so the ROCm clang finds no C++ headers and
no `libstdc++.so` to link. Homebrew's gcc 15 has both; until the package is
installed (`sudo dnf install libstdc++-devel` makes the flags unnecessary):

    B=/home/linuxbrew/.linuxbrew
    cmake --preset strix \
      "-DCMAKE_CXX_FLAGS=-isystem $B/include/c++/15 -isystem $B/include/c++/15/x86_64-pc-linux-gnu" \
      "-DCMAKE_EXE_LINKER_FLAGS=-L$B/lib/gcc/15"

The binary resolves `libstdc++.so.6` from `/lib64` at run time (system GCC 15,
same major). The BIOS VRAM carve-out is 512 MiB, so every allocation is GTT:
`drm-resident-gtt` is the number to read there, not a spill.
