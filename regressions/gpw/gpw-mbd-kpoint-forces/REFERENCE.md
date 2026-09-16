# Reference — multi-k GPW analytic forces with MBD@rsSCS

## Method

**Monkhorst–Pack multi-k GPW** ionic forces with additive **MBD@rsSCS**
(geometry-only free-atom volumes). Closes the residual that Γ
`assemble_gpw_forces` already folded MBD (`gpw-mbd-forces`) while
`assemble_gpw_forces_kpoints` counted MBD for mutual exclusion but **skipped**
adding \(F_{\mathrm{MBD}}\).

Product path:

1. Multi-k SCF (`GpwDft::k_mesh`) supplies electronic \(P_k\), \(W_k\).
2. Analytic k-summed Hellmann–Feynman / Pulay assembly (same as
   `gpw-kpoint-forces`).
3. Geometry-only \(F_{\mathrm{MBD}} = -\nabla E_{\mathrm{MBD}}\) from
   `mbd_rsscs_pbe` is **added** on the multi-k path (identical switch as Γ:
   isolated when `mbd_k_mesh={0,0,0}`; Ewald + MBD k-mesh when non-zero).

Public API: `GpwDft::mbd = true` + multi-k mesh + `gpw_forces_*` /
`GPWEvaluator(compute_forces=true)`. No new symbols.

PBE MBD: \(\beta = 0.83\), Fermi \(a = 6\). Free-atom \(\alpha_0\), \(C_6\),
\(R_{\mathrm{vdW}}\) from `basis_data/ts_tables.h` (libMBD `ts_vdw_params`).

## Reference values (hard gates)

### A — Geometry-only MBD (water dimer, isolated free-atom)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp** | libMBD Fortran `cmbd_mbd_scs_energy` (git `88069c6`, ATV Å→Bohr) | **−1.4928804171407961×10⁻³ Ha** |
| **Force dual-gate** | product \(F\) vs independent central FD \(h=10^{-5}\) Bohr | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| Translation | \(\|\sum_A F_A\|\) | ≤ **1×10⁻⁸** Ha/Bohr |

Same nuclei / pin policy as `gpw-mbd-forces` / `gpw-mbd-ice`.

### B — Multi-k product wiring (H₂, k-mesh 1×1×2)

Compressed H₂ (same geometry as `gpw-kpoint-forces`): PADE, DZV-GTH-PADE,
`k_mesh={1,1,2}`, free-atom MBD (`mbd_k_mesh` default zero).

| Quantity | Gate |
| -------- | ---- |
| SCF converged; multi-k payload (`n_kpts ≥ 1`, `C_occ_k` non-empty; Γ has `n_kpts==0`) | hard |
| \(F^{k+\mathrm{MBD}}-F^{k,\mathrm{plain}}\approx F_{\mathrm{MBD}}^{\mathrm{geom}}\) | max\|Δ\| ≤ **1×10⁻⁸** Ha/Bohr |
| \(\|F_{\mathrm{MBD},0x}^{\mathrm{geom}}\|\) nontrivial | > **1×10⁻⁸** Ha/Bohr |

Geometry-only MBD does not change the SCF density, so the force difference
must match the standalone `mbd_rsscs_pbe` forces bit-for-bit within tol.

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD is via external libMBD when linked. Hard
pin is **libMBD energy** + **force dual-gate** + **multi-k wiring**, same
policy as sibling MBD slugs. Twin notes under
`tools/cp2k_oracle/inputs/gpw-mbd-kpoint-forces/`.

## Geometry

### Water dimer (Å) — MBD pin

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

### H₂ (Bohr) — multi-k wiring

| Atom | x | y | z |
| ---- | - | - | - |
| H | 5.5 | 6.0 | 6.0 |
| H | 6.5 | 6.0 | 6.0 |

Cell (H₂ SCF): **12 × 12 × 12 Bohr**, periodic; electronic **k = 1×1×2**.

## Tolerance

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| (water dimer) | **1×10⁻⁸** Ha |
| max\|F_product − F_fd\| | **1×10⁻⁶** Ha/Bohr |
| \|Σ F_MBD\| | **1×10⁻⁸** Ha/Bohr |
| max\|F_k+MBD − F_k − F_geom\| | **1×10⁻⁸** Ha/Bohr |

## Perf gate

MBD is \(O(N^3)\) geometry-only; multi-k H₂ force assembly is a single
gradient pass after one SCF. Full CP2K QS wall is not a fair twin (no stock
MBD pair type). **`perf_exempt`** — same policy as `gpw-mbd-forces` /
`gpw-kpoint-forces`.
