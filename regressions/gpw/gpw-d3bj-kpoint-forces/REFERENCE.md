# Reference — multi-k GPW PBE-D3(BJ) analytic ionic forces

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with additive **DFT-D3 Becke–Johnson (rational) damping**, PBE
parameters, two-body only (`CALCULATE_C9_TERM .FALSE.`).

**Product ionic forces** = multi-k electronic Hellmann–Feynman + Pulay
(`assemble_gpw_forces_kpoints`) **plus** geometry-only

\[
F_{\mathrm{disp}} = -\nabla E_{\mathrm{disp}}^{\mathrm{D3(BJ)}}
\]

from `cpu::gpw::dftd3_pbe_bj` when `d3bj` is set. Same lattice-image sum and
pair cutoff as Γ `gpw-d3bj-forces`.

Public path: `cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `d3bj = true`,
`compute_forces = true`.

## Why the pin is a *difference* of two CP2K runs

DFT-D3 is geometry-only — it never enters the Kohn–Sham potential, so the
converged density is identical with and without it. Hence

\[
F_A^{\mathrm{CP2K}}(\mathrm{D3BJ}) - F_A^{\mathrm{CP2K}}(\mathrm{plain})
  = -\nabla_A E_{\mathrm{disp}},
\]

which isolates exactly the quantity this capability adds; the electronic
force (including any k-mesh residual) cancels.

## System

| Item | Value |
|------|--------|
| System | water dimer, **6 atoms**, order O H H O H H |
| Cell | **12 × 12 × 12 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (0,0,0); H (0.758602,0,±0.504284); O (3,0,0); H (3.758602,0,±0.504284) |
| Basis | `DZVP-MOLOPT-SR-GTH` |
| PP | O `GTH-PBE-q6`, H `GTH-PBE-q1` → 16 valence e⁻ (`n_occ=8`) |
| XC | PBE |
| Electronic k mesh | MONKHORST-PACK **1×1×2** (CP2K `FULL_GRID .FALSE.`; ATV TR-folded) |
| Dispersion | DFT-D3(BJ), PBE: \(s_6=1.0\), \(s_8=0.7875\), \(a_1=0.4289\), \(a_2=4.4407\) Bohr; CP2K CN; ATM off |
| Pair cutoff | `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20.0` Å |
| CP2K cutoff | 300 Ry |
| ATV cutoff | 120 Ry (dispersion gates geometry-only; dual gate grid-consistent) |

## Offline CP2K pin

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Install | `/nobackup/.../cp2k/install` (`source .../env.sh`) |
| Host | Arrhenius epyc login, offline only |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d3bj-kpoint-forces/` |
| Note | `water_dimer_d3bj_kpoint_forces.note` |
| Fair twin | D3BJ multi-k ENERGY_FORCE, **1 MPI × 8 OMP** |
| Wall (internal timer) | **3.931 s** |
| Date | 2026-08-01 |

### Energies [Hartree]

| Quantity | CP2K multi-k |
| -------- | ------------ |
| Total (plain) | −34.33825197808605 |
| Total (D3BJ) | −34.33962825578086 |
| **Dispersion energy** | **−0.00137627769481** |

Geometry-only: matches Γ `gpw-d3bj-forces` bit-for-bit.

### Atomic forces [Hartree/Bohr] (CP2K `FORCES|`)

D3BJ multi-k:

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −8.18231441E−02 | 4.01492980E−10 | 3.72641618E−10 |
| H0 | 4.08872726E−02 | 1.13134904E−11 | 1.31270241E−01 |
| H1 | 4.08872726E−02 | 1.06384740E−11 | −1.31270241E−01 |
| O1 | −8.05124309E−02 | 3.58188268E−10 | 3.28458012E−10 |
| H2 | 3.75636484E−02 | 1.35480669E−11 | 1.35137199E−01 |
| H3 | 3.75636484E−02 | 1.32671690E−11 | −1.35137199E−01 |

Plain multi-k: see `.note`. Derived dispersion force largest component
**9.96158×10⁻⁵** Ha/Bohr.

## Dual gate

1. Standalone `dftd3_pbe_bj` energy + forces vs offline CP2K multi-k FORCE Δ.
2. Product multi-k SCF (`d3bj`, `k_mesh={1,1,2}`): `e_disp` pin; multi-k
   payload; wiring \(F(\mathrm{d3bj})-F(\mathrm{plain})=F_{\mathrm{standalone}}\).
3. Fair wall: product path vs CP2K D3BJ multi-k ENERGY_FORCE twin
   (\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\)).

Unit twin: `GpwForces.KpointProductForcesFoldD3Bj` (cheap H₂ wiring).

## Tolerances

| Check | Tol |
| ----- | --- |
| \|E_disp − CP2K pin\| | 1×10⁻⁸ Ha |
| max\|F_disp − CP2K Δ\| | 5×10⁻⁶ Ha/Bohr |
| max\|(F_on − F_plain) − F_standalone\| | 1×10⁻⁷ Ha/Bohr |
| product wall | ≤ 60 s |
| fair wall ratio | ≤ 10 |

## Measured results (2026-08-01, green on worktree tip)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| A \|E_disp − pin\| | 1e−8 Ha | **7.63e−12** |
| A max\|F_disp − CP2K Δ\| | 5e−6 Ha/Bohr | **1.211e−06** |
| A max\|F_disp\| (non-vacuous) | > 1e−5 | **9.962e−05** |
| B product E_disp / n_kpts / iters | — | pin OK; n_kpts=1 (TR-folded); 11 iters |
| C multi-k wiring | 1e−7 Ha/Bohr | **6.5e−18** |
| Product wall | ≤ 60 s | **13.933 s** |
| Fair wall ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | ≤ 10 | **3.544** (13.933 / 3.931) |

Product multi-k SCF: 11 iters; TR-folded 1×1×2 reports `n_kpts=1` with
`C_occ_k` payload (same collapse note as `gpw-mbd-kpoint-forces`); force
assembly still takes the k-path (`assemble_gpw_forces_kpoints`).

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| `oracle` | **cp2k** (multi-k ENERGY_FORCE D3BJ + FORCE Δ; fair wall) |
| \(T_{\mathrm{cp2k}}\) | **3.931 s** (internal timer, 1 MPI × 8 OMP) |
| \(T_{\mathrm{atv}}\) (product SCF+force) | **13.933 s** |
| ratio | **3.544** |

## Siblings

- Γ D3(BJ) forces: `gpw-d3bj-forces`
- multi-k D3(BJ) stress: `gpw-d3bj-kpoint-stress`
- multi-k D4 ATM forces: `gpw-d4-atm-kpoint-forces`
- multi-k plain forces: `gpw-kpoint-forces`
