# Reference — multi-k GPW PBE-D4+ATM analytic ionic forces

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with additive **DFT-D4** including the Axilrod–Teller–Muto
**three-body** term (\(s_9 = 1\)).

**Product ionic forces** = multi-k electronic Hellmann–Feynman + Pulay
(`assemble_gpw_forces_kpoints`) **plus** geometry-only

\[
F_{\mathrm{disp}} = -\nabla\bigl(E_2 + E_{\mathrm{ATM}}\bigr)
\]

from `cpu::scf::dftd4` with `s9 = 1` when `d4_atm` is set. The ATM gradient
is analytic (same as Γ `gpw-d4-atm-forces`).

Public path: `cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `d4 = true`,
`d4_atm = true`, `compute_forces = true`.

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
| Dispersion | DFT-D4 BJ (PBE params) + **ATM** \(s_9=1\) |
| CP2K D4 cutoffs | `D4_CUTOFF 30.0` Å, `R_CUTOFF 20.0` Å; ATM: `CALCULATE_C9_TERM .TRUE.`, `FACTOR_S9_TERM 1.0` |
| CP2K cutoff | 300 Ry |
| ATV cutoff | 120 Ry (dispersion gates geometry-only; dual gate grid-consistent) |

## Offline pins

### Hard: E_disp + standalone forces (dftd4 3.7.0)

Geometry-only — independent of electronic k mesh. Same numbers as
`gpw-d4-atm-forces`:

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{disp}}\) (s9=1) | **−9.9554475834240310×10⁻⁴** Ha |
| Force pin | \(F = -g\) from `pin_periodic_atm_grad.json` (see Γ REFERENCE) |
| ATM increment | max component **~4.05×10⁻⁷** Ha/Bohr vs `--mbdscale 0` |

### Hard: multi-k ATM force increment (CP2K 2026.1 libdftd4)

Two offline ENERGY_FORCE runs (identical except ATM on/off). D4 is
geometry-only, so

\[
F^{\mathrm{CP2K}}(\mathrm{D4+ATM}) - F^{\mathrm{CP2K}}(\mathrm{D4\ 2b})
 = F_{\mathrm{ATM}}.
\]

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) **with `libdftd4`** |
| Binary | private Arrhenius rebuild `/nobackup/.../cp2k-dftd4/install` |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d4-atm-kpoint-forces/` |
| Note | `water_dimer_d4_atm_kpoint_forces.note` |
| Wall (fair twin) | **15.806 s** CP2K internal timer on the D4+ATM multi-k ENERGY_FORCE job |
| Date | 2026-08-01 |

Printed dispersion:

| Run | \(E_{\mathrm{disp}}\) [Ha] |
| --- | --- |
| ATM | −0.00099567884677 |
| 2b | −0.00099470623400 |

ΔF tables (max |ΔF| ≈ 4.19×10⁻⁷ Ha/Bohr) are tabulated in the `.note` file
and hard-coded in the regression.

### Why dftd4 *and* CP2K

- ATV's D4 implementation is the in-tree `dftd4` path (matches CLI dftd4
  3.7.0 to ~1e−14 Ha on E_disp). Hard E_disp / standalone force gates use
  that oracle.
- Multi-k product + fair wall use real CP2K ENERGY_FORCE with the private
  libdftd4 build (same policy as `gpw-d4-kpoint-stress`). Residual between
  CP2K ΔF and dftd4 ATM increment is ~2.4×10⁻⁸ Ha/Bohr (periodic ATM
  gradient defect / cutoff parity — same floor as Γ).

## Dual gate

1. Standalone `dftd4(s9=1)` energy + forces vs offline pins.
2. Standalone ATM force increment vs CP2K multi-k FORCE Δ (and vs dftd4 Δ).
3. Product multi-k SCF (`d4_atm`, `k_mesh={1,1,2}`): `e_disp` pin; multi-k
   payload; wiring \(F(\mathrm{d4\_atm})-F(\mathrm{plain})=F_{\mathrm{standalone}}\).
4. Fair wall: product path vs CP2K D4+ATM multi-k ENERGY_FORCE twin
   (\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\)).

## Tolerances

| Check | Tol |
| ----- | --- |
| \|E_disp − dftd4 pin\| | 1×10⁻⁸ Ha |
| max\|F_d4atm + g_pin\| | 5×10⁻⁸ Ha/Bohr |
| max\|dF_ATM − CP2K Δ\| | 5×10⁻⁸ Ha/Bohr |
| max\|(F_on − F_plain) − F_standalone\| | 1×10⁻⁷ Ha/Bohr |
| product wall | ≤ 60 s |
| fair wall ratio | ≤ 10 |

## Measured results (2026-08-01, green on worktree tip)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| A max\|F + g_dftd4\| | 5e−8 Ha/Bohr | **2.394e−08** |
| B max\|dF_ATM − CP2K Δ\| | 5e−8 Ha/Bohr | **1.356e−08** |
| B max\|dF\| (non-vacuous) | > 1e−8 | **4.054e−07** |
| C \|E_disp − pin\| | 1e−8 Ha | **2.76e−14** |
| D multi-k wiring | 1e−7 Ha/Bohr | **1.3e−17** |
| Product wall | ≤ 60 s | **9.682 s** |
| Fair wall ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | ≤ 10 | **0.613** (9.682 / 15.806) |

Product multi-k SCF: 11 iters; TR-folded 1×1×2 reports `n_kpts=1` with
`C_occ_k` payload (same collapse note as `gpw-mbd-kpoint-forces`); force
assembly still takes the k-path (`uses_kpath` / `assemble_gpw_forces_kpoints`).

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| `oracle` | **cp2k** (multi-k ENERGY_FORCE D4+ATM + FORCE Δ; fair wall) |
| E_disp / standalone F | dftd4 3.7.0 cross-check |
| \(T_{\mathrm{cp2k}}\) | **15.806 s** (internal timer, 1 MPI × 8 OMP) |
| ratio | **0.613** |

## What is checked

1. Multi-k product path with `d4` + `d4_atm` converges and carries `C_occ_k`.
2. Geometry-only \(E_{\mathrm{disp}}\) matches dftd4.
3. ATM force increment matches CP2K multi-k FORCE difference.
4. Multi-k force assembly wiring equals standalone D4+ATM.
5. Fair wall ratio ≤ 10 vs real CP2K.
