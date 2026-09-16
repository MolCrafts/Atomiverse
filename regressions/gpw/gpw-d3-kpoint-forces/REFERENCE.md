# Reference — multi-k GPW PBE-D3 (zero damping) analytic ionic forces

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with additive **DFT-D3 zero ("original") damping**, PBE reference
parameters, two-body only (`CALCULATE_C9_TERM .FALSE.`).

**Product ionic forces** = multi-k electronic Hellmann–Feynman + Pulay
(`assemble_gpw_forces_kpoints`) **plus** geometry-only

\[
F_{\mathrm{disp}} = -\nabla E_{\mathrm{disp}}^{\mathrm{D3}}
\]

from `cpu::gpw::dftd3_pbe` with `kD3Cp2kPairCutoffBohr` = 2 × CP2K
`R_CUTOFF 20.0` Å (same as energy / stress).

Public path: `cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `d3 = true`,
`compute_forces = true`.

## Why the pin is a *difference* of two CP2K runs

DFT-D3 is geometry-only: it adds \(E_{\mathrm{disp}}(\{R\})\) to the total
energy and does not enter the Kohn–Sham potential, so the converged density is
identical with and without it. Therefore

\[
F_A^{\mathrm{CP2K}}(\mathrm{D3}) - F_A^{\mathrm{CP2K}}(\mathrm{plain})
  = -\nabla_A E_{\mathrm{disp}}^{\mathrm{D3}}
\]

isolates exactly the quantity this multi-k fold adds, with the (much larger)
electronic force cancelling — the multi-k analogue of `gpw-d3-forces`.

Because D3 is independent of the electronic k mesh, the multi-k ΔF pin matches
the periodic Γ force difference to print residual; this slug still **re-runs**
the multi-k twin so the pin is provenance-matched to the claimed algorithm
class (multi-k Quickstep + D3 ENERGY_FORCE).

## System

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → **16** valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Electronic k mesh | MONKHORST-PACK **1×1×2** (CP2K `FULL_GRID .FALSE.`; ATV TR-folded) |
| Dispersion | DFT-D3 zero damping, PBE: \(s_6=1.0\), \(s_8=0.722\), \(s_{r,6}=1.217\), \(s_{r,8}=1.0\); ATM off |
| Pair cutoff | `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20.0` Å |
| CP2K cutoff | 300 Ry (`REL_CUTOFF 40`, `NGRIDS 4`) |
| ATV cutoff | 120 Ry (dispersion gates geometry-only; dual gate grid-consistent) |
| SCF | CP2K `EPS_SCF 1e-7`, DIAGONALIZATION + DIRECT_P_MIXING α=0.4; ATV product `tol=1e-6` |

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, source build `…/cp2k/install/env.sh`, git:**5e54ba2**) |
| Host | Arrhenius login (`arrhenius1`), offline only |
| Parallelism | **1 MPI × OMP 8** (`mpirun -np 1`, `OMP_NUM_THREADS=8`) |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3-kpoint-forces/water_dimer_d3_kpoint_forces.inp` and `water_dimer_plain_kpoint_forces.inp` |
| Note | `tools/cp2k_oracle/inputs/gpw-d3-kpoint-forces/water_dimer_d3_kpoint_forces.note` |
| Wall (D3 ENERGY_FORCE twin) | **3.741 s** (CP2K internal `CP2K` timer TOTAL TIME maximum) |
| Date | 2026-08-01 |

### Energies [Hartree]

| Quantity | CP2K |
| -------- | ---- |
| Total (plain) | −34.338251978086042 |
| Total (D3) | −34.339072872762657 |
| **Dispersion energy** | **−0.00082089467662** |

(Matches the Γ / multi-k periodic D3-zero energy pin — geometry-only.)

### Atomic forces [Hartree/Bohr] (CP2K `FORCES|` table)

D3 run:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.18063746E−02 | 4.01494971E−10 | 3.72648379E−10 |
| H0 | 4.07355091E−02 | 1.13136310E−11 | 1.31298079E−01 |
| H1 | 4.07355091E−02 | 1.06384166E−11 | −1.31298079E−01 |
| O1 | −8.01907903E−02 | 3.58191574E−10 | 3.28465632E−10 |
| H2 | 3.75462068E−02 | 1.35483328E−11 | 1.35127326E−01 |
| H3 | 3.75462068E−02 | 1.32676062E−11 | −1.35127326E−01 |

Plain run:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.19227603E−02 | 4.01487087E−10 | 3.72642534E−10 |
| H0 | 4.08225175E−02 | 1.13134271E−11 | 1.31264301E−01 |
| H1 | 4.08225175E−02 | 1.06385615E−11 | −1.31264301E−01 |
| O1 | −8.04237550E−02 | 3.58190301E−10 | 3.28468534E−10 |
| H2 | 3.76338737E−02 | 1.35481357E−11 | 1.35133744E−01 |
| H3 | 3.76338737E−02 | 1.32675490E−11 | −1.35133744E−01 |

Derived dispersion force \(F_{\mathrm{disp}} = F(\mathrm{D3}) -
F(\mathrm{plain})\); largest component **2.329647×10⁻⁴ Ha/Bohr** (O1, \(x\)).

### Re-run (offline only — never ctest)

```bash
source /nobackup/proj/disk/teoroo/personal/jicli594/work/cp2k/install/env.sh
cd tools/cp2k_oracle/inputs/gpw-d3-kpoint-forces
export OMP_NUM_THREADS=8
mpirun -np 1 cp2k.psmp -i water_dimer_d3_kpoint_forces.inp \
    -o water_dimer_d3_kpoint_forces.out
mpirun -np 1 cp2k.psmp -i water_dimer_plain_kpoint_forces.inp \
    -o water_dimer_plain_kpoint_forces.out
# Record: Dispersion energy; FORCES| Atomic forces; CP2K timer TOTAL TIME
```

## Dual gate

1. Standalone `dftd3_pbe` energy + forces vs offline CP2K multi-k pins
   (\(E_{\mathrm{disp}}\), \(F_{\mathrm{disp}}=F(\mathrm{D3})-F(\mathrm{plain})\)).
2. Product multi-k SCF (`d3`, `k_mesh={1,1,2}`): `e_disp` pin; multi-k
   `C_occ_k` payload; wiring
   \(F(\mathrm{d3})-F(\mathrm{plain})=F_{\mathrm{standalone}}\).
3. Fair wall: product path vs CP2K D3 multi-k ENERGY_FORCE twin
   (\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\)).

## Tolerances (physical justification)

| Gate | Tol | Why |
| ---- | --- | --- |
| \|E_disp − pin\| | **1×10⁻⁸ Ha** | Same closed-form two-body sum / PBE params / 2×R_CUTOFF as `gpw-d3-kpoint-stress`. |
| max\|F_disp − CP2K Δ\| | **5×10⁻⁶ Ha/Bohr** | Difference of two independently converged SCFs (`EPS_SCF 1e-7`); signal ~2.3e−4. Same class as Γ `gpw-d3-forces`. |
| max\|(F_on − F_plain) − F_standalone\| | **1×10⁻⁷ Ha/Bohr** | Same multi-k density; only SCF residual enters. |
| product wall | **≤ 60 s** | e2e tier budget. |
| fair wall ratio | **≤ 10** | vs real CP2K D3 multi-k ENERGY_FORCE twin only. |

## Measured results (2026-08-01, green on worktree tip)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \|E_disp − pin\| | 1e−8 Ha | **5.53e−11** |
| max\|F_disp − CP2K Δ\| | 5e−6 Ha/Bohr | **7.858e−07** (signal 2.330e−4) |
| multi-k wiring | 1e−7 Ha/Bohr | **1.3e−17** |
| Product wall | ≤ 60 s | **16.284 s** (cold; warm re-run 12.416 s) |
| Fair wall ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | ≤ 10 | **4.353** (16.284 / 3.741) |

Product multi-k SCF: 11 iters; TR-folded 1×1×2 reports `n_kpts=1` with
`C_occ_k` payload (same collapse note as `gpw-mbd-kpoint-forces`); force
assembly still takes the k-path (`assemble_gpw_forces_kpoints`).

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| `oracle` | **cp2k** (multi-k ENERGY_FORCE D3 + FORCE Δ; fair wall) |
| \(T_{\mathrm{cp2k}}\) | **3.741 s** (internal timer, 1 MPI × 8 OMP) |
| \(T_{\mathrm{atv}}\) | **16.284 s** (product multi-k SCF + forces, OMP 8, 120 Ry) |
| ratio | **4.353** |

## What is checked

1. Multi-k product path with `d3` converges and carries `C_occ_k`.
2. Geometry-only \(E_{\mathrm{disp}}\) matches CP2K.
3. Standalone \(F_{\mathrm{disp}}\) matches CP2K multi-k FORCE difference.
4. Multi-k force assembly wiring equals standalone D3.
5. Fair wall ratio ≤ 10 vs real CP2K.

## Notes

- Sibling Γ force pin: `regressions/gpw/gpw-d3-forces`.
- Sibling multi-k stress pin: `regressions/gpw/gpw-d3-kpoint-stress`.
- Sibling multi-k dispersion force pins: `gpw-d4-atm-kpoint-forces`,
  `gpw-mbd-kpoint-forces`.
