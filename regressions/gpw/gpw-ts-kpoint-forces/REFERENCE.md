# Reference — multi-k GPW analytic forces with free-atom TS

## Method

**Monkhorst–Pack multi-k GPW** ionic forces with additive **Tkatchenko–Scheffler
(TS)** free-atom volume ratios \(V/V_{\mathrm{free}}=1\). Closes the dedicated
multi-k **force** pin for free-atom TS: Γ `assemble_gpw_forces` already folds
TS (`gpw-ts-forces`); multi-k stress is a sibling (`gpw-ts-kpoint-stress`);
this regression asserts `assemble_gpw_forces_kpoints` adds the same
geometry-only \(F_{\mathrm{TS}}\).

Product path:

1. Multi-k SCF (`GpwDft::k_mesh`) supplies electronic \(P_k\), \(W_k\).
2. Analytic k-summed Hellmann–Feynman / Pulay assembly (same as
   `gpw-kpoint-forces`).
3. Geometry-only \(F_{\mathrm{TS}} = -\nabla E_{\mathrm{TS}}\) from `ts_pbe`
   is **added** on the multi-k path (periodic pure real-space lattice sum,
   rcut **50 Å**, free-atom volumes when Hirshfeld is off).

Public API: `GpwDft::ts = true` + multi-k mesh + `gpw_forces_*` /
`GPWEvaluator(compute_forces=true)`. No new symbols.

PBE TS: \(s_R = 0.94\), Fermi \(d = 20\). Free-atom \(\alpha_0\), \(C_6\),
\(R_{\mathrm{vdW}}\) from `basis_data/ts_tables.h` (libMBD `ts_vdw_params`).

## Reference values (hard gates)

### A — Geometry-only TS (water dimer, isolated free-atom)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp** | libMBD Fortran `cmbd_ts_energy` (git `88069c6`, ATV Å→Bohr) | **−5.9764872295816794×10⁻⁴ Ha** |
| **Force dual-gate** | product \(F\) vs independent central FD \(h=10^{-5}\) Bohr | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| Translation | \(\|\sum_A F_A\|\) | ≤ **1×10⁻⁸** Ha/Bohr |

Same nuclei / pin policy as `gpw-ts-forces` / `gpw-ts-ice`.

### B — Multi-k product wiring (H₂, k-mesh 1×1×2)

Compressed H₂ (same geometry as `gpw-kpoint-forces` / `gpw-mbd-kpoint-forces`):
PADE, DZV-GTH-PADE, `k_mesh={1,1,2}`, free-atom TS (no Hirshfeld).

| Quantity | Gate |
| -------- | ---- |
| SCF converged; multi-k payload (`n_kpts ≥ 1`, `C_occ_k` non-empty; Γ has `n_kpts==0`) | hard |
| \(F^{k+\mathrm{TS}}-F^{k,\mathrm{plain}}\approx F_{\mathrm{TS}}^{\mathrm{geom}}\) (periodic pure-RS, rcut 50 Å) | max\|Δ\| ≤ **1×10⁻⁸** Ha/Bohr |
| \(\|F_{\mathrm{TS},0x}^{\mathrm{geom}}\|\) nontrivial | > **1×10⁻⁸** Ha/Bohr |

Geometry-only TS does not change the SCF density, so the force difference
must match the standalone `ts_pbe` forces within tol.

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** stock `TYPE TS`. Hard pin is
**libMBD energy** + **force dual-gate** + **multi-k wiring**, same policy as
`gpw-ts-forces` / `gpw-ts-ice` / `gpw-mbd-kpoint-forces`. Twin notes under
`tools/cp2k_oracle/inputs/gpw-ts-kpoint-forces/`. Never label this pin
`oracle: cp2k`.

## Geometry

### Water dimer (Å) — TS pin

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Å→Bohr: `1/0.52917720859` (ATV).

### H₂ (Bohr) — multi-k wiring

| Atom | x | y | z |
| ---- | - | - | - |
| H | 5.5 | 6.0 | 6.0 |
| H | 6.5 | 6.0 | 6.0 |

Cell (H₂ SCF): **12 × 12 × 12 Bohr**, periodic; electronic **k = 1×1×2**.

## Tolerance

| Check | Tol |
| ----- | --- |
| \|E_disp − pin\| (water dimer isolated) | **1×10⁻⁸** Ha |
| max\|F_product − F_fd\| | **1×10⁻⁶** Ha/Bohr |
| \|Σ F_TS\| | **1×10⁻⁸** Ha/Bohr |
| max\|F_k+TS − F_k − F_geom\| | **1×10⁻⁸** Ha/Bohr |

## Perf gate

TS evaluation is \(O(N^2\times N_{\mathrm{images}})\) geometry-only; multi-k
H₂ force assembly is a single gradient pass after one SCF. Full CP2K QS wall
is not a fair twin (no stock TS pair type). **`perf_exempt = true`** — same
policy as `gpw-ts-forces` / `gpw-mbd-kpoint-forces`.

## Reproduce offline libMBD energy (never from ctest)

See `tools/cp2k_oracle/inputs/gpw-ts-stress/water_dimer_ts_stress.note` and
`pin_ts_strain.f90` for the libMBD Fortran pin driver (git `88069c6`). The
isolated \(E_{\mathrm{disp}}\) line is the A-gate pin here.
