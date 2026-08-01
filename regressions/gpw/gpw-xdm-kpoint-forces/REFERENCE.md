# Reference — multi-k GPW analytic forces with free-atom XDM

## Method

**Monkhorst–Pack multi-k GPW** ionic forces with additive **free-atom XDM**
pairwise dispersion (geometry-only multipole moments + CRC \(\alpha\)).
Closes the dedicated multi-k force pin residual: Γ
`assemble_gpw_forces` and multi-k `assemble_gpw_forces_kpoints` both fold
XDM; this regression hard-gates the multi-k wiring against dual pure-Python
energy/force pins (same policy as `gpw-xdm-forces` / `gpw-xdm-kpoint-stress`).

Product path:

1. Multi-k SCF (`GpwDft::k_mesh`) supplies electronic \(P_k\), \(W_k\).
2. Analytic k-summed Hellmann–Feynman / Pulay assembly (same as
   `gpw-kpoint-forces`).
3. Geometry-only \(F_{\mathrm{XDM}} = -\nabla E_{\mathrm{XDM}}\) from
   `xdm_pbe` is **added** on the multi-k path (identical switch as Γ:
   pure real-space lattice sum, rcut 50 Å, QE NC-PBE BJ damping).

Public API: `GpwDft::xdm = true` + multi-k mesh + `gpw_forces_*` /
`GPWEvaluator(compute_forces=true)`. No new symbols.

PBE BJ damping: QE NC-PBE \(a_1=0.4283\), \(a_2=2.4690\) Å. Free-atom
\(\alpha,M_1,M_2,M_3\) from in-tree `basis_data/xdm_tables.h`.

## Reference values (hard gates)

### A — Geometry-only free-atom XDM (water dimer)

| Quantity | Source | Value |
| -------- | ------ | ----- |
| **E_disp (isolated)** | dual pure-Python / `gpw-xdm-ice` | **−2.1020435998516220×10⁻³ Ha** |
| **E_disp (periodic)** | dual pure-Python / `gpw-xdm-ice` | **−2.1148157612879029×10⁻³ Ha** |
| **Force dual-gate** | product \(F\) vs independent central FD \(h=10^{-5}\) Bohr | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| Translation | \(\|\sum_A F_A\|\) | ≤ **1×10⁻⁸** Ha/Bohr |
| Isolated F sample | dual pure-Python ≡ ATV tables | max\|Δ\| ≤ **1×10⁻¹⁰** Ha/Bohr |

Same nuclei / pin policy as `gpw-xdm-forces` / `gpw-xdm-ice`. XDM is
geometry-only — independent of the electronic k mesh.

### Offline force sample (isolated free-atom)

| Atom | \(F_x\) | \(F_y\) | \(F_z\) [Ha/Bohr] |
| ---- | ------- | ------- | ----------------- |
| O0 | +1.6875829936003448×10⁻⁴ | 0 | 0 |
| H0 | +6.9736686381575340×10⁻⁵ | 0 | −1.3172272218329521×10⁻⁵ |
| H1 | +6.9736686381575340×10⁻⁵ | 0 | +1.3172272218329521×10⁻⁵ |
| O1 | −1.1378113807647229×10⁻⁴ | 0 | 0 |
| H2 | −9.7225267023356434×10⁻⁵ | 0 | −1.5448899346549854×10⁻⁵ |
| H3 | −9.7225267023356434×10⁻⁵ | 0 | +1.5448899346549854×10⁻⁵ |

### B — Multi-k product wiring (H₂, k-mesh 1×1×2)

Compressed H₂ (same geometry as `gpw-kpoint-forces` / `gpw-mbd-kpoint-forces`):
PADE, DZV-GTH-PADE, `k_mesh={1,1,2}`, free-atom XDM (periodic cell, rcut 50 Å).

| Quantity | Gate |
| -------- | ---- |
| SCF converged; multi-k payload (`n_kpts ≥ 1`, `C_occ_k` non-empty; Γ has `n_kpts==0`) | hard |
| \(F^{k+\mathrm{XDM}}-F^{k,\mathrm{plain}}\approx F_{\mathrm{XDM}}^{\mathrm{geom}}\) | max\|Δ\| ≤ **1×10⁻⁸** Ha/Bohr |
| \(\|F_{\mathrm{XDM},0x}^{\mathrm{geom}}\|\) nontrivial | > **1×10⁻⁸** Ha/Bohr |

Geometry-only XDM does not change the SCF density, so the force difference
must match the standalone `xdm_pbe` forces within tol.

### Why not CP2K QS?

**CP2K 2026.1** (`cp2k.psmp` Source code revision git:5e54ba2) stock
`PAIR_POTENTIAL TYPE` is only `DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is
**no** stock `TYPE XDM` (`input_cp2k_xc.F` usage string). Hard pin is
**dual pure-Python energy + force dual-gate + multi-k wiring**, same policy
as `gpw-xdm-forces` / `gpw-xdm-kpoint-stress`. Twin notes under
`tools/cp2k_oracle/inputs/gpw-xdm-kpoint-forces/`. Never fake `oracle: cp2k`.

## Geometry

### Water dimer (Å) — XDM pin

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (periodic E_disp pin): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.
Isolated force pin uses the same nuclear coordinates without lattice images.
Periodic E_disp / force dual-gate uses rcut **50 Å**.

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
| \|Σ F_XDM\| | **1×10⁻⁸** Ha/Bohr |
| max\|F − F_dual_python sample\| | **1×10⁻¹⁰** Ha/Bohr |
| max\|F_k+XDM − F_k − F_geom\| | **1×10⁻⁸** Ha/Bohr |

## Perf gate

XDM evaluation is \(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds). Multi-k H₂ force assembly is a single gradient pass after
one SCF. Full CP2K QS wall is not a fair twin (no stock XDM pair type).
**`perf_exempt = true`** — same policy as `gpw-xdm-forces` /
`gpw-xdm-kpoint-stress` / `gpw-mbd-kpoint-forces`. Oracle family:
`literature` / dual pure-Python (never fake `oracle: cp2k`).
