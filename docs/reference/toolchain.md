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
