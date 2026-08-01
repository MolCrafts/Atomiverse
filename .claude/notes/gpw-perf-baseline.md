<!-- mol:note:topic:gpw-perf-baseline -->
## GPW fair-wall baseline (`p0-profile-baseline`)

Standing **ATV_GPW_PROF** table for the four performance twins. Later
`wf/perf-*` workers compare against these numbers (same host class, OMP=4).

| | |
| --- | --- |
| Date | 2026-08-01 |
| Host | `arrhenius1` — AMD EPYC 9255 24-Core ×2 (48 cores), OMP=4, 1 MPI |
| Build | Release, `-DATV_BUILD_HF/PW/GPW=ON`, CUDA OFF, OpenBLAS |
| Env | `ATV_GPW_PROF=1`; OT twin also `ATV_GPW_OT_PROF=1` |
| Branch tip | `wf/perf-p0-profile-baseline` |

### How to reproduce

```bash
export OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores
export ATV_GPW_PROF=1          # [gpw prof], [gpw prof k], [gpw prof uks], [gpw setup*]
export ATV_GPW_OT_PROF=1       # [OT prof] rebuild counts (Γ OT only)
# build reg_* binaries (see docs/developer/gpw-perf-loop.md CMake)
/usr/bin/time -f 'WALL_SEC %e' ./build/regressions/reg_<twin>
```

Stderr tags:

| Tag | Path |
| --- | --- |
| `[gpw setup]` | Γ setup (STVpp, ionic density, cbg, …) |
| `[gpw setup k]` | multi-k setup (STVnl_k, orth_k, kbg_build / products_mg) |
| `[gpw prof]` | Γ RKS KS rebuild: collocate / Poisson / XC / integrate |
| `[gpw prof uks]` | Γ UKS dual-spin KS rebuild (same four stages) |
| `[gpw prof k]` | multi-k RKS KS rebuild + `nk` / `mg` / `build=#` |
| `[OT prof]` | Γ OT: H_grad / precond / linesearch (+ Armijo rebuild count) |
| `[mg coll]` | multigrid product count (first 5 calls) |

### Fair walls (single twin SCF, not multi-gate reg total)

| Twin | ATV fair wall [s] | Source | CP2K (REFERENCE) [s] | ratio ATV/CP2K |
| --- | ---: | --- | ---: | ---: |
| `gpw-uks-o2-gamma` | **17.6** | reg internal `wall=` (26 DIIS iters) | 21.3 | **0.83** |
| `gpw-kpoints-si` | **~3.7** (est.) | MAIN MP 2×2×2 150 Ry: setup STVnl≈1.25 + kbg≈1.37 + 6×KS≈1.04 | ≈3.3 | **~1.1** |
| `gpw-multigrid-kpoints` | **65.6** | reg `MAIN: wall` (7 DIIS iters, NGRIDS 4) | 23.6 | **2.8** |
| `ot-minimizer-water` | **~10.8** (OT only) | Σ`[OT prof]` + Γ setup; full OT+DIIS reg **13.3** | (see REFERENCE ~2.7 OT) | OT-only ~4× class |

Multi-gate **regression** walls (all gates in one binary; do **not** use for speed gates):

| Twin | Whole-reg WALL_SEC [s] |
| --- | ---: |
| `gpw-uks-o2-gamma` | 17.6 (single job) |
| `gpw-kpoints-si` | 18.9 (A1–A7 identities + MAIN) |
| `gpw-multigrid-kpoints` | 92.6 (M2/M3 + MAIN) |
| `ot-minimizer-water` | 13.3 (OT + DIIS) |

### Per-step KS breakdown (mean ms / SCF energy rebuild)

| Twin | ngrid | nk | mg | builds | collocate | Poisson | XC | integrate | sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `gpw-uks-o2-gamma` | 3 112 136 | Γ | 0 | 26 | **146** | 57 | 34 | **355** | 592 |
| `gpw-kpoints-si` MAIN | 37 800 | 4 | 0 | 6 | **100** | 1 | 5 | 66 | 173 |
| `gpw-multigrid-kpoints` MAIN | 37 800 | 4 | 1 | 7 | 1026 | 1 | 5 | **7123** | 8156 |
| `ot-minimizer-water` OT | 884 736 | Γ | 0 | 59 | 17 | 11 | **102** | 45 | 174 |
| `ot-minimizer-water` DIIS | 884 736 | Γ | 0 | 9 | 17 | 10 | 102 | 45 | 174 |

Setup (geometry-fixed, once per SCF job on the fair twin):

| Twin | Dominant setup |
| --- | --- |
| Γ UKS O₂ | `cbg_build` ≈ **2.1 s** (STVpp+ionic+locsr ≈ 0.08 s) |
| multi-k dense χ (MAIN class) | `STVnl_k` ≈ **1.25 s**, `kbg_build` ≈ **1.2–1.4 s** |
| multi-k MG MAIN | `STVnl_k` ≈ **1.24 s**, `products_mg` ≈ **0.08 s** (no kbg) |
| OT water | `cbg_build` ≈ **0.65 s** |

### OT rebuild economics (`ot-minimizer-water`)

| Metric | Value |
| --- | --- |
| OT outer iters (profiled) | 25 (reg reports 26 incl. last) |
| Armijo `ls_rebuilds` sum | **33** (mean **1.32** / iter) |
| Energy rebuilds on OT job | **59** ≈ outer + Armijo + final H |
| Mean `[OT prof]` step | H_grad≈200 ms, linesearch≈200 ms, total≈400 ms |

### Hotspot ranking confirmed by this baseline

1. **MG multi-k integrate** — ~7.1 s / step (87% of MAIN KS rebuild); product restrict+integrate thrash.
2. **Γ UKS integrate + collocate** on large MT grid — 355 + 146 ms/step; cbg setup 2.1 s once.
3. **Dense multi-k χ^k collocate** — ~100 ms/step of 173 ms; plus kbg setup ~1.3 s.
4. **OT energy rebuild multiplicity** — ~2.3 full KS builds per outer iter (Armijo).
5. Poisson/XC small on these twins except OT PBE GGA XC ~100 ms/step.

Maps to queue P0: collocate buffers, FFT plan cache, zgemm, OT double-energy;
P1: workspace / products-default-k / OT economics.

### Correctness snapshot (same build)

- `ctest -R '^test_cpu_(gpw|xc)_'` — 12/12 pass (light unit set; HEAVY off).
- Four baseline regs PASS with existing pins (no tol change).

### Instrumentation landed this slug

Thin `ATV_GPW_PROF` marks only (no algorithm change):

- multi-k RKS: `[gpw prof k]` in `build_k_energy` + `[gpw setup k]` STVnl/orth/kbg/products
- Γ UKS: `[gpw prof uks]` in `build_uks_H`

Γ RKS `[gpw prof]` / OT `[OT prof]` / `[mg coll]` pre-existed.
