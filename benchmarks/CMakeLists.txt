# Native performance harnesses live under perf/. Deliberately NOT registered
# with ctest — tests/ is unit-only and "if it is registered, it runs"
# (see docs/developer/project-contract.md). The Python framework
# (`python -m benchmarks`) and the compare/ scripts need no CMake.
add_subdirectory(perf)
