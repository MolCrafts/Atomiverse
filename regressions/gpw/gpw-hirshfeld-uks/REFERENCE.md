# Reference — O₂ triplet Γ-UKS GPW PBE + Hirshfeld-scaled MBD@rsSCS

## Method

**Hirshfeld volume ratios** \(V/V_{\mathrm{free}}\) from the final Γ-point
**UKS GPW PBE** total valence density \(\rho=\rho_\alpha+\rho_\beta\), used to
density-scale free-atom **MBD@rsSCS** parameters (Tkatchenko & Scheffler,
*Phys. Rev. Lett.* **102**, 073005 (2009); Ambrosetti et al., *J. Chem. Phys.*
**140**, 18A508 (2014)).

Public API:

```cpp
GpwDft gpw;
gpw.n_alpha = 7;
gpw.n_beta  = 5;
gpw.mbd = true;
gpw.hirshfeld = true;   // post-SCF volumes from ρ=ρα+ρβ → mbd_rsscs_pbe(…, vol_ratio)
```

Dispersion energy on `GpwResult::e_disp` [Hartree]; ratios on
`GpwResult::volume_ratios`. Free-atom path (`hirshfeld=false`) remains
geometry-only. RKS Hirshfeld: `reg_gpw-mbd-hirshfeld` / multi-k
`reg_gpw-hirshfeld-kpoints`.

Free-atom valence densities (in-tree exponential model,
`basis_data/hirshfeld_tables.h`):

\[
\rho_Z^{\mathrm{free}}(r)=A\exp(-\alpha_Z r),\quad
A=N_{\mathrm{val}}\alpha_Z^3/(8\pi),\quad
V_Z^{\mathrm{free}}=60\,N_{\mathrm{val}}/\alpha_Z^3
\]

O: \(N=6\), \(\alpha=2.2\) (Bohr⁻¹).

Scaling of free-atom α₀ / C₆ / R_vdW (libMBD tables): \(\alpha\propto v\),
\(C_6\propto v^2\), \(R\propto v^{1/3}\). MBD@rsSCS is **isolated** on the
nuclear coordinates.

## Reference value (hard gate)

**E_disp (PBE-MBD@rsSCS, Hirshfeld volumes, O₂ UKS, CUTOFF 200 Ry) =
−1.0379964698969957×10⁻⁴ Ha**

Cross-checks (offline — never in ctest):

| Source | Note | E_disp [Ha] |
| ------ | ---- | ----------- |
| **ATV product** (`GpwDft` UKS mbd+hirshfeld, CUTOFF 200, tol 1e-6) | dual gate | **−1.0379964698969957×10⁻⁴** |
| Free-atom MBD (volume_ratio = 1) on same O₂ geometry | `mbd_rsscs_pbe` dual | more attractive than Hirshfeld |
| Closed-shell water UKS≈RKS unit gate | `test_cpu_gpw_hirshfeld` | E_disp parity within 5e-6 Ha |

Hirshfeld scaling reduces |E_disp| vs free-atom (compressed molecular volumes
\(v_A\sim 0.67\)).

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no TS/MBD. Offline twin notes under
`tools/cp2k_oracle/inputs/gpw-hirshfeld-uks/`. Free-atom MBD already pinned vs
libMBD Fortran (`gpw-mbd-ice`); this slug pins the **UKS volume-ratio layer**.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 6.000000 | 6.000000 | 5.39625 |
| O | 6.000000 | 6.000000 | 6.60375 |

R(O–O) = **1.2075 Å**. Cell: **ABC 12.0 12.0 12.0 Å**, **PERIODIC NONE**
(Martyna–Tuckerman). Molecule **centered** to avoid MT edge truncation.

Basis / PP: `DZVP-MOLOPT-GTH` + `GTH-PBE-q6` (6 valence e⁻ / O → 12 total).
CUTOFF **200 Ry**, max_iter 150, tol 1e-6, mix 0.35, diis_depth 10, PBE,
**Nα=7, Nβ=5** (multiplicity 3).

Representative volume ratios at the pin (order O, O):

`0.6666, 0.6666`

## Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests: UKS product path physical
bounds; closed-shell water UKS vs RKS E_disp within 5e-6 Ha.

## Perf gate

**perf_exempt = true** — primary deliverable is E_disp + volume-ratio parity
on the UKS density path. Hirshfeld grid integral is \(O(N_{\mathrm{grid}} N_{\mathrm{atoms}})\);
SCF dominates wall (same policy as `gpw-mbd-hirshfeld` / `gpw-hirshfeld-kpoints`).
