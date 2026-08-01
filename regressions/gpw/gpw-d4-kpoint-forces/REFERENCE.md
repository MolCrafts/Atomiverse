# Reference — multi-k GPW PBE-D4 two-body analytic ionic forces

## Method

**Monkhorst–Pack multi-k GPW** (PBE, electronic mesh 1×1×2) on a periodic
water dimer with additive **DFT-D4 two-body BJ** (ATM off, \(s_9 = 0\)).

**Product ionic forces** = multi-k electronic Hellmann–Feynman + Pulay
(`assemble_gpw_forces_kpoints`) **plus** geometry-only

\[
F_{\mathrm{disp}} = -\nabla E_2
\]

from `cpu::scf::dftd4` with `s9 = 0` when `d4` is set and `d4_atm` is off.

Public path: `cpu::gpw::GPWEvaluator` with `k_mesh = {1,1,2}`, `d4 = true`,
`d4_atm = false`, `compute_forces = true`.

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
| Dispersion | DFT-D4 BJ two-body only (\(s_9=0\)); PBE params \(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\) |
| CP2K D4 cutoffs | `D4_CUTOFF 30.0` Å, `R_CUTOFF 20.0` Å; `CALCULATE_C9_TERM .FALSE.`, `FACTOR_S9_TERM 0.0` |
| CP2K cutoff | 300 Ry |
| ATV cutoff | 120 Ry (dispersion gates geometry-only; dual gate grid-consistent) |

## Offline pins

### Hard: E_disp + standalone forces (dftd4 3.7.0)

Geometry-only — independent of electronic k mesh. Same numbers as
`gpw-d4-forces` / `gpw-d4-water`:

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{disp}}\) (s9=0) | **−0.00099458842745846774** Ha |
| Force pin | \(F = -g\) from dftd4 `--mbdscale 0 --grad` (O H H O H H) |
| Gradient norm | ≈ **1.719×10⁻⁴** Ha/Bohr |

### Hard: multi-k D4 force increment (CP2K 2026.1 libdftd4)

Two offline ENERGY_FORCE runs (identical except D4 on/off). D4 is
geometry-only, so

\[
F^{\mathrm{CP2K}}(\mathrm{D4}) - F^{\mathrm{CP2K}}(\mathrm{plain})
 = F_{\mathrm{disp}}.
\]

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) **with `libdftd4`** |
| Binary | private Arrhenius rebuild `/nobackup/.../cp2k-dftd4/install` |
| Inputs | `tools/cp2k_oracle/inputs/gpw-d4-kpoint-forces/` |
| Note | `water_dimer_d4_kpoint_forces.note` |
| Wall (fair twin) | **4.117 s** CP2K internal timer on the D4 multi-k ENERGY_FORCE job |
| Date | 2026-08-01 |

Printed energies:

| Run | \(E_{\mathrm{disp}}\) [Ha] | \(E_{\mathrm{total}}\) [Ha] |
| --- | --- | --- |
| D4 | −0.00099470623400 | −34.339246684320045 |
| plain | — | −34.338251978086035 |

ΔF tables (max |ΔF| ≈ 1.361×10⁻⁴ Ha/Bohr) are tabulated in the `.note` file
and hard-coded in the regression. Cross-check vs dftd4 3.7.0:
max |ΔF − F_dftd4| ≈ **4.2×10⁻⁹** Ha/Bohr.

### Why dftd4 *and* CP2K

- ATV's D4 implementation is the in-tree `dftd4` path (matches CLI dftd4
  3.7.0 to ~1e−14 Ha on E_disp). Hard E_disp / standalone force gates use
  that oracle.
- Multi-k product + fair wall use real CP2K ENERGY_FORCE with the private
  libdftd4 build (same policy as `gpw-d4-atm-kpoint-forces` /
  `gpw-d4-kpoint-stress`).

## Dual gate

1. Standalone `dftd4(s9=0)` energy + forces vs offline pins.
2. Standalone D4 forces vs CP2K multi-k FORCE Δ (D4 − plain).
3. Product multi-k SCF (`d4`, `k_mesh={1,1,2}`): `e_disp` pin; multi-k
   payload; wiring \(F(\mathrm{d4})-F(\mathrm{plain})=F_{\mathrm{standalone}}\).
4. Fair wall: product path vs CP2K D4 multi-k ENERGY_FORCE twin
   (\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\)).

## Tolerances

| Check | Tol |
| ----- | --- |
| \|E_disp − dftd4 pin\| | 1×10⁻⁸ Ha |
| max\|F_d4 + g_pin\| | 1×10⁻⁷ Ha/Bohr |
| max\|F_d4 − CP2K Δ\| | 5×10⁻⁷ Ha/Bohr |
| max\|(F_on − F_plain) − F_standalone\| | 1×10⁻⁷ Ha/Bohr |
| product wall | ≤ 60 s |
| fair wall ratio | ≤ 10 |

## Measured results (2026-08-01, green on worktree tip)

| Gate | Tol | Measured |
| ---- | --- | -------- |
| A max\|F + g_dftd4\| | 1e−7 Ha/Bohr | **2.906e−11** |
| B max\|F − CP2K Δ\| | 5e−7 Ha/Bohr | **4.210e−09** |
| B max\|F\| (non-vacuous) | > 1e−6 | **1.361e−04** |
| C \|E_disp − pin\| | 1e−8 Ha | **2.75e−14** |
| D multi-k wiring | 1e−7 Ha/Bohr | **1.1e−17** |
| Product wall | ≤ 60 s | **9.987 s** |
| Fair wall ratio \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | ≤ 10 | **2.426** (9.987 / 4.117) |

Product multi-k SCF: 11 iters; TR-folded 1×1×2 reports `n_kpts=1` with
`C_occ_k` payload (same collapse note as `gpw-d4-atm-kpoint-forces`); force
assembly still takes the k-path (`uses_kpath` / `assemble_gpw_forces_kpoints`).

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| `oracle` | **cp2k** (multi-k ENERGY_FORCE D4 + FORCE Δ; fair wall) |
| E_disp / standalone F | dftd4 3.7.0 cross-check |
| \(T_{\mathrm{cp2k}}\) | **4.117 s** (internal timer, 1 MPI × 8 OMP) |
| ratio | **2.426** |

## What is checked

1. Multi-k product path with `d4` (two-body) converges and carries `C_occ_k`.
2. Geometry-only \(E_{\mathrm{disp}}\) matches dftd4.
3. Standalone D4 forces match CP2K multi-k FORCE difference (D4 − plain).
4. Multi-k force assembly wiring equals standalone D4.
5. Fair wall ratio ≤ 10 vs real CP2K.
