# cuda-pw-large-cell — first large-cell CPU-vs-CUDA PW wall measurement

Parity-gap slug `cuda-pw-large-cell-measure` (ES order step 4), sanctioned by
`.claude/notes/cuda-rhf-device-only.md` ("the first large-cell measurement …
is the only place any of these residency waves can actually be shown to
pay"). Oracle: cpu-parity (wall) — the CPU leg at identical settings in the
same binary is both the correctness reference and the wall twin. **No perf
claim from the small sizes** (8 atoms is correctness-only); the CUDA-wins
story is stated only where the CPU wall exceeds 60 s.

## Methodology

- **System ladder**: 8-atom conventional diamond-Si cube (a = 5.43 Å)
  replicated n×n×n → 8 / 64 / 216 atoms (16 / 128 / 432 bands).
- **Settings (identical on both legs, inside the CUDA LSDA envelope)**:
  UPF `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC-LDA), `ecut_ry = 15`,
  `k_mesh = 1` (Γ), `tol = 1e-6` (density-change L2), `max_iter = 100`,
  `mix_beta = 0.7`, `pulay_depth = 5`, `xc = XCType::LSDA`, no smearing,
  restricted fixed occupations → both backends solve exactly n_e/2 bands
  (`src/cpu/planewave/pw_solver.cpp:227` — `extra_bands` is smearing-only).
- **Timed region**: `run(frame)` only, `std::chrono::steady_clock` inside
  the harness — geometry-dependent setup + the full SCF on both legs.
  `build()` (UPF parse) and CUDA context creation (`cudaFree(0)`) excluded.
- **Harness**: `tools/bench_cuda_pw_cell.cu` (public API only:
  `cpu::pw::PW` / `cuda::pw::PW` + `Frame`), compiled by
  `tools/gpu_job_cuda_pw_large_cell.sh` with the exact compile/dlink/link
  commands CMake generates for `reg_pw-cuda-lda-si` (harvested via
  `make -n` + `link.txt`/`dlink.txt`), so no tracked CMake file changes —
  the in-flight variable-cell tree has `regressions/CMakeLists.txt` and
  `tests/CMakeLists.txt` dirty.
- **Host/GPU**: GH200 node n420 — Grace Neoverse-V2 (288 CPUs visible,
  job `-c 16`) + NVIDIA GH200 120GB; nvcc 13.0.88, `sm_90`, GCC 14.3,
  OpenBLAS 0.3.32. CPU legs `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = 16`
  (plus a 1-thread reference at n=1); the CPU PW pipeline itself is serial
  (`src/cpu/planewave/` has no OpenMP) — threading enters via OpenBLAS.
  CUDA legs run with the same host-thread setting.
- **Tree**: `wf/cuda-pw-large-cell-measure` @ dev `3d025c35`.

## Jobs

| Job | What | Log |
| --- | --- | --- |
| 1035136 | FAILED at configure — worktree had no molrs submodule checkout | `.claude/cuda-pw-large-cell-1035136.out` |
| 1035238 | build + slice sanity + ladder n=1,2,3 | `.claude/cuda-pw-large-cell-1035238.out` |

## Results

PENDING — job 1035238 running; this section is filled from its log
(`BENCH_RESULT` / `BENCH_PARITY` lines) before the branch is closed.

## Gate

Queue gate: scale until the same-settings CPU wall exceeds 60 s, report
CPU-vs-CUDA walls with job ids + methodology. Filled with the results above.
