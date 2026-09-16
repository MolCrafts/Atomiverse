# Reference — water-dimer GPW PBE-MBD@rsSCS ionic forces

## Method

**Many-body dispersion MBD@rsSCS** ionic forces on top of Γ-point **GPW**
(PBE), free-atom volume ratios \(V/V_{\mathrm{free}}=1\). Same water dimer as
`gpw-mbd-ice` / `gpw-ts-ice`.

Product path:

1. `mbd_rsscs_pbe` evaluates \(E_{\mathrm{disp}}\) matching libMBD
   (`cmbd_mbd_scs_energy` / pure-Python).
2. Forces \(F_A = -\partial E_{\mathrm{disp}}/\partial R_A\) [Ha/Bohr] from the
   same geometry-only energy (central FD product kernel, \(h=10^{-4}\) Bohr).
3. `assemble_gpw_forces` with `GpwDft::mbd = true` **adds** these forces to the
   electronic Hellmann–Feynman / Pulay terms (no longer energy-only).

Public API: `cpu::gpw::GpwDft::mbd = true` (and `GPWEvaluator::mbd`) with
`compute_forces=true` / `gpw_forces_*`.

PBE: \(\beta = 0.83\), Fermi \(a = 6\). Free-atom \(\alpha_0\), \(C_6\),
\(R_{\mathrm{vdW}}\) from `basis_data/ts_tables.h` (libMBD `ts_vdw_params`).

## Reference values (hard gates)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp** | libMBD Fortran `cmbd_mbd_scs_energy` (git `88069c6`, ATV Å→Bohr) | **−1.4928804171407961×10⁻³ Ha** |
| **Force dual-gate** | product \(F\) vs independent central FD \(h=10^{-5}\) Bohr of \(E_{\mathrm{disp}}\) | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| Translation | \(\|\sum_A F_A\|\) | ≤ **1×10⁻⁸** Ha/Bohr |

### Offline force sample (isolated free-atom, product kernel)

Printed by the regression (geometry-only; not a CP2K absolute pin):

| Atom | \(F_x\) | \(F_y\) | \(F_z\) [Ha/Bohr] |
| ---- | ------- | ------- | ----------------- |
| O0 | (runtime) | ~0 | ~0 |
| … | … | … | … |

Cross-check: pure-Python `pymbd` / libMBD analytic gradients agree with ATV
product forces to \(O(h^2)\) of the FD step (offline — never in ctest).

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD is via external libMBD when linked. Hard
pin is **libMBD energy** + **force dual-gate**, same policy as `gpw-mbd-ice` /
`gpw-ts-ice`. Twin notes under `tools/cp2k_oracle/inputs/gpw-mbd-forces/`.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (SCF only): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
MBD pin is **isolated free-atom** on the same nuclear coordinates
(`mbd_k_mesh={0,0,0}`).

Basis / PP (GPW short SCF; E_disp and F_MBD are basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

## Tolerance

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| | **1×10⁻⁸** Ha |
| max\|F_product − F_fd\| | **1×10⁻⁶** Ha/Bohr |
| \|Σ F\| | **1×10⁻⁸** Ha/Bohr |

## Perf gate

MBD energy is \(O(N^3)\) on a \(3N\times 3N\) matrix; force FD is \(6N\) energy
evals (still microseconds on 6 atoms). Full GPW SCF wall dominates if present.
**perf_exempt = true**: primary deliverable is E_disp + force dual-gate parity
(same policy as `gpw-mbd-ice` / `gpw-ts-ice`).
