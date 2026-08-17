# Standing LJ bench — not ctest. The CUDA binary lives with the other
# tools (`cuda_nonbonded_style_bench`) so a CUDA-off CPU build still
# configures. This directory exists so ATV_BUILD_BENCHMARKS has a home.
message(STATUS "benchmarks/lj: see README.md (cuda_nonbonded_style_bench)")
