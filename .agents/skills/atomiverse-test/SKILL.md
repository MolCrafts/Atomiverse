---
name: atomiverse-test
description: Writes or changes Atomiverse unit tests (gtest, pytest) and runs them through the CMake presets and ctest. Use when adding, fixing or running a test under tests/ or python/tests/.
---

# atomiverse-test

Tests are unit tests and nothing else; AGENTS.md `## Testing` owns what may
be a test, and `## Build and gates` owns presets and domain flags.

## Steps

1. **Decide whether it is a unit test.** Check the change against AGENTS.md
   `## Testing`: no full SCF/MD/vib/MC solve, no timing, no external-engine
   comparison, no reading outside the repository, no source-tree scans.
2. **Place it by the source layout.** C++ tests mirror `src/` under
   `tests/` (`core/`, `cpu/`, `cuda/`, `mpi/`) and link into `atv_tests`
   (or `atv_tests_cuda` / `atv_tests_mpi`); Python tests go in
   `python/tests/`, benchmark-framework units in `tests/benchmarks/`.
3. **Pick the preset for the domain.** Use `cpu-unit` for classical code,
   `cpu-unit-hf` or `cpu-unit-all` otherwise (`CMakePresets.json`); keep a
   single domain's closure only, the rest OFF.
4. **Build and run.** `cmake --workflow --preset cpu-unit`, or
   `cmake --preset cpu-unit-hf && cmake --build --preset cpu-unit-hf`
   then `ctest --preset cpu-unit-hf -R <regex>` for one test; each test
   must finish under the 10 s preset timeout.
5. **Rebuild after layout changes.** After a class-layout change rebuild
   every consuming translation unit before trusting a result.
6. **Run CUDA tests on the GPU node only.** Use the `atomiverse-cluster`
   skill for `atv_tests_cuda`.
7. **Prove the test bites.** Break the code on purpose, watch it fail,
   restore it.

## Done when

- the test fails on broken code and passes on the fix, within 10 s
- the matching `ctest --preset` run is green
- no skip-mark, weakened assert or registered opt-in switch was added
