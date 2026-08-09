# cuda-pw-large-cell — first large-cell CPU-vs-CUDA PW wall measurement

ES order step 4 for slug `cuda-pw-large-cell-measure` (es-cpu-cuda-opt).
Sanctioned by `.claude/notes/cuda-rhf-device-only.md`: residency waves (kpoint
hoist, Zheevd, devptr-nrm2) pay only on a cell large enough that SCF body
dominates process startup / CUDA context init — **not** on the 2-atom Si
correctness twin.

Oracle: **same-settings CPU twin** (wall + energy). The CPU leg in the same
binary is both the correctness reference and the wall twin. **No “CUDA wins”
claim from small sizes** (8 atoms is correctness-only); the large-cell story
is stated only where the CPU wall exceeds 60 s.

## Lever (research)

| Item | Value |
| --- | --- |
| Lever class | Measurement only — no physics / kernel change |
| Files | `tools/bench_cuda_pw_cell.cu`, `tools/gpu_job_cuda_pw_large_cell.sh`, this note |
| Metric | wall of `run(frame)` for `cpu::pw::PW` vs `cuda::pw::PW` on Si n×n×n ladder |
| Acceptance | GH200 log with filled results; `reg_pw-cuda-lda-si --device=both` PASS; scale until CPU wall > 60 s |

## Methodology

- **System ladder**: 8-atom conventional diamond-Si cube (a = 5.43 Å)
  replicated n×n×n → 8 / 64 / 216 atoms (16 / 128 / 432 bands).
- **Settings (identical on both legs, inside the CUDA LSDA envelope)**:
  UPF `tests/data/pw/Si_lda_nc.upf` (ONCVPSP NC-LDA), `ecut_ry = 15`,
  `k_mesh = 1` (Γ), `tol = 1e-6` (density-change L2), `max_iter = 100`,
  `mix_beta = 0.7`, `pulay_depth = 5`, `xc = XCType::LSDA`, no smearing,
  restricted fixed occupations → both backends solve exactly n_e/2 bands.
- **Timed region**: `run(frame)` only, `std::chrono::steady_clock` inside
  the harness — geometry-dependent setup + full SCF. `build()` (UPF parse)
  and CUDA context creation (`cudaFree(0)`) excluded.
- **Harness**: `tools/bench_cuda_pw_cell.cu` (public API only:
  `cpu::pw::PW` / `cuda::pw::PW` + `Frame`), compiled by
  `tools/gpu_job_cuda_pw_large_cell.sh` with the exact compile/dlink/link
  commands CMake generates for `reg_pw-cuda-lda-si` (harvested via
  `make -n` + response-file substitution for dlink). No tracked CMake
  product files are modified.
- **Host/GPU**: GH200 Grace Neoverse-V2 (288 CPUs visible, job `-c 16`) +
  NVIDIA GH200 120GB; nvcc 13.0.88, `sm_90`, GCC 14.3, OpenBLAS 0.3.32.
  CPU legs `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = 16` (plus a 1-thread
  reference at n=1). CPU PW itself is serial; threading enters via OpenBLAS.
- **Tree**: `wf/escuda-cuda-pw-large-cell-measure` rebased on `dev` @
  `8c13bc36` (measurement tip `fbc8f229` + gate fix).

## Jobs

| Job | State | What | Log |
| --- | --- | --- | --- |
| 1035136 | FAILED | configure — no molrs submodule | (old worktree) |
| 1035238 | FAILED | HARNESS_LINK_FAILED (dlink rsp not substituted) | `.claude/cuda-pw-large-cell-1035238.out` |
| **1035278** | CANCELLED after ~2.8 h (n=3 CPU mid-leg) | first complete ladder walls | `.claude/cuda-pw-large-cell-1035278.out` |
| **1048264** | CANCELLED at 47 min (uid 4001043) mid n=2 CPU | reconfirmed slice + n=1 + n=2 CUDA | `.claude/cuda-pw-large-cell-1048264.out` |

## Results

### Slice sanity (correctness twin)

`reg_pw-cuda-lda-si --device=both` (2-atom Si, ecut 15 Ry, k_mesh=2) on both
jobs: **PASS**, \|E_cuda − E_cpu\| = **1.243e-14 Ry** (bound 1e-3 Ry), 5 iters
each. `SLICE_SANITY_RC=0`.

### Supercell ladder walls (`run(frame)` only)

Primary walls from **1035278** (n516); reconfirm **1048264** (n522) in
parentheses where finished.

| n_rep | atoms | bands | device thr | wall_s | E [Ry] | iters | notes |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| 1 | 8 | 16 | both 16t | cpu **17.93** / cuda **0.97** (17.85 / 0.96) | −62.677194 | 5 | \|ΔE\|=**2.089e-08** Ry (1048264 `BENCH_PARITY`) |
| 1 | 8 | 16 | cpu 1t | **17.91** (17.85) | −62.677194 | 5 | OpenBLAS thr ≈ no-op (serial PW) |
| 2 | 64 | 128 | cuda 16t | **9.664** (**9.645**) | −507.09055846 | 5 | bit-stable E across jobs |
| 2 | 64 | 128 | cpu 16t | **> 5400** | — | — | 1035278 `LEG_TIMEOUT`; 1048264 cancelled before finish; gate met |
| 3 | 216 | 432 | cuda 16t | **191.214** | −1712.93740038 | 5 | 1035278 only |
| 3 | 216 | 432 | cpu 16t | (cancelled) | — | — | not required once gate met |

### Gate

**Met at n=2 (64 atoms).** Same-settings CPU wall **> 5400 s > 60 s**.
CUDA wall at that size is **~9.65 s** (two independent GH200 runs agree to
0.02 s).

| Size | T_cuda / T_cpu | note |
| --- | ---: | --- |
| n=1 (8 at) | 0.96 / 17.9 ≈ **0.054×** | correctness only — **not** a speed claim |
| n=2 (64 at) | 9.65 / **>5400** ≈ **< 0.0018×** | **large-cell evidence**; CUDA ≪ CPU |
| n=3 (216 at) | CUDA 191 s; CPU unfinished | CUDA scales; CPU dense O(n_b³) too expensive |

`within_10x` vs CPU twin: **yes** (CUDA is faster; ratio ≪ 1). Not
perf-exempt: this is real large-cell wall evidence, not a foothold skip.

### What this does **not** claim

- No fair twin vs CP2K/QE (oracle is ATV CPU same-settings).
- No “CUDA wins” from 2-atom or 8-atom cells alone.
- No device forces / stress / PBE product (other slugs).
- n=2 CPU exact wall remains a lower bound (>5400 s). Optional polish only.

## Gate checklist

| Check | Status |
| --- | --- |
| GH200 log with filled results | **yes** — **1035278** (+ 1048264 reconfirm) |
| Correctness suite (`pw-cuda-lda-si` both) | **PASS** \|ΔE\|=1.243e-14 Ry |
| CPU wall > 60 s | **yes** at 64 atoms (>5400 s) |
| CPU↔CUDA walls reported | **yes** (table above) |
| Device-only product path | unchanged (measurement only) |
| Builds wiped after job | yes — work tree `_pw_cell_work` (not `build-*`) wiped on exit; legs under `.claude/pw-cell-legs/` |

## Follow-ons (not this slug)

- Optional: exact n=2 CPU wall with ≥3 h budget (gate already closed at >5400 s).
- `cuda-pw-forces` / `cuda-pw-pbe-device` / non-default stream — separate rows.
- Optional: QE fair wall on the same supercell (parity-gap PW perf track).
