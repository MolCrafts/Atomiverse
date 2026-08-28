# CUDA OpenMM MD — GH200 timing + LAMMPS Kokkos A/B

**Date:** 2026-07-23  
**Node:** n146 GH200 120GB (`hpc gpu -bg` JOBID=574524)  
**ATV engine:** OpenMM cluster/tile path, Release, `CMAKE_CUDA_ARCHITECTURES=90`  
**LAMMPS:** develop d3b14f6, serial `lmp-kk-serial`, Kokkos CUDA HOPPER90

## OPEN RED — cluster LJ force disagrees with brute force (2026-08-28)

`MultiGpu.LjForceParityNlistVsSingleGpu`
(`tests/cuda/dynamics/test_multi_gpu.cu:750`) fails on a **single** GH200:
the cluster/tile `compute_lj_nlist` path and the `compute_lj_bruteforce`
oracle differ by **9.50** where the tolerance is 5e-2, on 512 atoms at
cut=5.0 / spacing 3.0 / ghost_headroom 2.0. This is the production CUDA MD
force path against its own oracle, one GPU, no halo involved — so it is a
cluster-build or tile-mask defect, not a multi-GPU communication one.

Timing numbers below were taken before this was noticed; treat them as
unvalidated until the parity gate is green. The two `driver build: missing
required Integrator` failures alongside it were test-side (the force-pass
tests had no Integrator after the preset validation tightened) and are fixed.

Reproduce: `sbatch tools/job_cuda_fail_detail.sh` (GH200, ~2 min).

## Methodology (operator rules)

| Axis | Mechanism | How |
|------|-----------|-----|
| **Precision** | **One CMake mode**, aligned with LAMMPS `KOKKOS_PREC` | `-DATV_CUDA_PREC_MODE=double\|mixed\|single` → maps to `(real_t≡work_t, prec_t)` = `(KK_FLOAT, KK_ACC_FLOAT)`. **No free `CUDA_WORK` axis.** Build: `build_atv_bench.sh double\|mixed\|single` |
| **Neighbor FULL vs half** | **Script / LMP API params** (not hardcoded in ATV) | `run_compare.py --lmp-neigh {half\|full} --lmp-newton {on\|off}` |
| **LMP “best”** | Coarse package sweep at N≈1e6 | `run_compare.py --sweep-lmp --sweep-n 1000000` → log `logs/lmp_kk_sweep_1m.txt` |

### Precision mode map (ATV ↔ LMP KK)

| mode | ATV `real_t`/`work_t` | ATV `prec_t` | LMP `KK_FLOAT` | LMP `KK_ACC_FLOAT` |
|------|----------------------|--------------|----------------|--------------------|
| **double** (default both) | double | double | double | double |
| **mixed** | float | double | float | double |
| **single** | float | float | float | float |

**Historical note (pre-alignment):** ATV used to ship `REAL=float WORK=double PREC=double` as “float default”. That is **not** any LMP mode (LMP mixed = float math + double acc). Tables below tagged `float` are that old hybrid; re-run with `mixed` for a fair same-mode A/B (and rebuild LMP with `-D KOKKOS_PREC=mixed` if comparing mixed-to-mixed).

Protocol (both engines): LJ NVE, reduced units, ρ\*=**0.8**, cut=**2.5**, skin=**0.3**, dt=**0.005**, ε=σ=**1**, cubic SC lattice. Warmup then timed steps.

Harness: `tools/bench_lj_vs_lammps/`  
Logs: `logs/lj_atv_vs_lammps_kk_{float,double,double_vs_lmp_half}.txt`, `logs/lmp_kk_sweep_1m.txt`

---

## 1. LAMMPS coarse package sweep (N≈1e6)

| neigh | newton | comm | sort | ms/step | Matom·s |
|-------|--------|------|------|---------|---------|
| half | on | – | – | 2.30 | 435 |
| half | on | device | device | 2.28 | 438 |
| half | off | – | – | 2.36 | 424 |
| full | on | * | * | **FAIL** (KK requires `newton off` with `neigh full`) |
| **full** | **off** | **–** | **–** | **1.87** | **536** |
| full | off | device | device | 1.88 | 531 |

**BEST LMP (this sweep):** `neigh full` + `newton off` ≈ **1.87 ms/step** at 1M  
(~**18% faster** than default `half`+`newton on` ≈ 2.28–2.30).  
`comm`/`sort device` did not help further.

---

## 2. ATV float vs LMP best (`full` + `newton off`)

ATV build: `ATV_CUDA_REAL=float`, `ATV_CUDA_WORK=double`, `ATV_CUDA_PREC=double`  
(`build-cuda-bench` / product default storage)

| N | ATV ms | ATV Matom·s | LMP best ms | LMP Matom·s | **ATV/LMP** |
|---|--------|-------------|-------------|-------------|-------------|
| 103 823 | **0.138** | 752 | 0.419 | 248 | **0.33** |
| 250 047 | **0.218** | 1149 | 0.926 | 270 | **0.24** |
| 512 000 | **0.393** | 1303 | 1.383 | 370 | **0.28** |
| 1 000 000 | **0.719** | 1392 | 1.864 | 537 | **0.39** |
| 2 000 376 | **1.339** | 1494 | 2.905 | 689 | **0.46** |
| 5 000 211 | **3.640** | 1374 | 4.818 | 1038 | **0.76** |
| 10 077 696 | 8.873 | 1136 | **8.358** | 1206 | **1.06** |

Up to ~5M: ATV float still faster than best LMP. At **~10M**, LMP best edges ahead (~6%).

---

## 3. ATV double vs LMP best (`full` + `newton off`)

ATV build: `ATV_CUDA_REAL=double`, `WORK=double`, `PREC=double`  
(`build-cuda-bench-double` via `build_atv_bench.sh double`)

| N | ATV ms | ATV Matom·s | LMP best ms | LMP Matom·s | **ATV/LMP** |
|---|--------|-------------|-------------|-------------|-------------|
| 103 823 | **0.151** | 687 | 0.424 | 245 | **0.36** |
| 250 047 | **0.246** | 1015 | 0.926 | 270 | **0.27** |
| 512 000 | **0.450** | 1137 | 1.391 | 368 | **0.32** |
| 1 000 000 | **0.802** | 1247 | 1.867 | 536 | **0.43** |
| 2 000 376 | **1.445** | 1385 | 2.916 | 686 | **0.50** |
| 5 000 211 | **3.484** | 1435 | 4.808 | 1040 | **0.73** |
| 10 077 696 | **7.251** | 1390 | 8.379 | 1203 | **0.87** |

**Double–double fairer match:** ATV still faster at all sizes in this harness  
(~**1.15–3.7×** depending on N; gap narrows toward 10M).

Note: at 10M float ATV regressed vs double ATV (8.87 vs 7.25 ms) — treat as  
**measure once** / possible BW or rebuild noise; re-run before claiming float
slower at large N.

---

## 4. Reference: double ATV vs LMP **default** (`half` + `newton on`)

| N | ATV ms | LMP half/on ms | ATV/LMP |
|---|--------|----------------|---------|
| 103 823 | 0.151 | 0.482 | 0.31 |
| 1 000 000 | 0.801 | 2.286 | 0.35 |
| 5 000 211 | 3.475 | 8.279 | 0.42 |
| 10 077 696 | 7.255 | 16.276 | 0.45 |

Default LMP is **~1.2–2× slower** than best LMP at large N — old “ATV 2–3×”
headlines were inflated by this default.

---

## Interpretation / caveats

1. **Precision is compile-global** — compare only binaries built with matching  
   `ATV_CUDA_REAL`. Do not mix float ATV vs double LMP as “architecture wins.”
2. **Neighbor is a run param** on LMP (`--lmp-neigh` / `--lmp-newton`).  
   ATV classical CUDA path is FULL cluster CSR + ½ energy; fairer arithmetic  
   match = LMP `full` + `newton off` (and KK enforces that pair).
3. **Single-rank** serial KK; no multi-rank domain decomp, no binsize / hybrid  
   host-device deep tuning.
4. ATV headroom: nsys historically shows `lj_force_tile` ~55% GPU; fixed-box  
   `setup_cell` already skipped on rebuild.

---

## Reproduce

```bash
hpc gpu -bg -t 04:00:00 -c 16 && hpc gpu wait

hpc gpu run -- bash -lc '
  module load GPU/buildenv-gcccuda/2026.03-cu13.0
  export LD_LIBRARY_PATH="${LIBRARY_PATH:-}:$LD_LIBRARY_PATH"
  export LMP_BIN=$HOME/.local/aarch64/bin/lmp-kk-serial
  export OMP_NUM_THREADS=1
  cd /path/to/atomiverse

  # 1) precision builds (one mode knob = LMP KOKKOS_PREC)
  ./tools/bench_lj_vs_lammps/build_atv_bench.sh double
  ./tools/bench_lj_vs_lammps/build_atv_bench.sh mixed

  # 2) coarse LMP package sweep
  python3 tools/bench_lj_vs_lammps/run_compare.py --sweep-lmp --sweep-n 1000000

  # 3) fair A/B vs best LMP (full / newton off); same prec mode both sides
  python3 tools/bench_lj_vs_lammps/run_compare.py \
    --atv-bench build-cuda-bench-double/bin/cuda_nonbonded_style_bench \
    --tag double --lmp-neigh full --lmp-newton off
'

```

## LAMMPS build notes

- Nested `srun` + MPI LAMMPS → **PMI_Init** fail: use **`lmp-kk-serial`**  
  (`BUILD_MPI=OFF`) under `hpc gpu run`.
- Source: `/nobackup/proj/disk/teoroo/personal/jicli594/work/lammps`  
  build: `build-kokkos-cuda-serial`.
