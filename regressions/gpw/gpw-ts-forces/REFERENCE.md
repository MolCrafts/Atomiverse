# Reference — water-dimer GPW PBE-TS ionic forces

## Method

**Tkatchenko–Scheffler (TS)** ionic forces on top of Γ-point **GPW** (PBE),
free-atom volume ratios \(V/V_{\mathrm{free}}=1\). Same water dimer as
`gpw-ts-ice` / `gpw-mbd-forces`.

Product path:

1. `ts_pbe` evaluates \(E_{\mathrm{disp}}\) matching libMBD
   (`cmbd_ts_energy` / pure real-space).
2. Forces \(F_A = -\partial E_{\mathrm{disp}}/\partial R_A\) [Ha/Bohr] from the
   same geometry-only energy (analytic product kernel; dual-gated vs central
   FD, \(h=10^{-5}\) Bohr).
3. `assemble_gpw_forces` with `GpwDft::ts = true` **adds** these forces to the
   electronic Hellmann–Feynman / Pulay terms.

Public API: `cpu::gpw::GpwDft::ts = true` (and `GPWEvaluator::ts`) with
`compute_forces=true` / `gpw_forces_*`.

PBE: \(s_R = 0.94\), Fermi \(d = 20\). Free-atom \(\alpha_0\), \(C_6\),
\(R_{\mathrm{vdW}}\) from `basis_data/ts_tables.h` (libMBD `ts_vdw_params`).
Pair cutoff (periodic): **50 Å**.

## Reference values (hard gates)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp isolated** | libMBD Fortran `cmbd_ts_energy` (git `88069c6`, ATV Å→Bohr) | **−5.9764872295816794×10⁻⁴ Ha** |
| **E_disp periodic** | libMBD Fortran `get_ts_energy` pure-RS (`ewald_on = false`, rcut 50 Å; `gpw-ts-stress` pin driver) | **−6.0996307733860822×10⁻⁴ Ha** (= `gpw-ts-ice` pin to 7×10⁻¹⁸) |
| **Force dual-gate** | product \(F\) vs independent central FD \(h=10^{-5}\) Bohr of \(E_{\mathrm{disp}}\) | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| Translation | \(\|\sum_A F_A\|\) | ≤ **1×10⁻⁸** Ha/Bohr |
| Wiring | \(F(\mathrm{ts})-F(\mathrm{plain})\) vs standalone | max\|Δ\| ≤ **1×10⁻⁷** Ha/Bohr |

### Offline force sample (isolated free-atom, product kernel)

Printed by the regression (geometry-only; not a CP2K absolute pin).

Cross-check: pure-Python / libMBD analytic gradients agree with ATV product
forces to \(O(h^2)\) of the FD step (offline — never in ctest).

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE TS`. Hard pin is
**libMBD energy** + **force dual-gate**, same policy as `gpw-ts-ice` /
`gpw-mbd-forces`. Twin notes under `tools/cp2k_oracle/inputs/gpw-ts-forces/`.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (SCF + periodic TS pin): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
Isolated force dual-gate uses the same nuclear coordinates without lattice
images.

Basis / PP (GPW short SCF; E_disp and F_TS are basis-independent):
`DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).

## Tolerance

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| (isolated / periodic) | **1×10⁻⁸** Ha (reg) |
| max\|F_product − F_fd\| | **1×10⁻⁶** Ha/Bohr |
| \|Σ F\| | **1×10⁻⁸** Ha/Bohr |
| max\|F(ts)−F(plain)−F_geom\| | **1×10⁻⁷** Ha/Bohr |

## Perf gate

TS evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Full GPW SCF wall dominates if present.
**perf_exempt = true**: primary deliverable is E_disp + force dual-gate parity
(same policy as `gpw-ts-ice` / `gpw-mbd-forces`). CP2K has no stock TS twin for
a fair wall ratio.
