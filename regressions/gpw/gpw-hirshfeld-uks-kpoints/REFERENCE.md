# Reference — multi-k collinear UKS GPW PBE + Hirshfeld-scaled MBD@rsSCS

## Method

**Hirshfeld volume ratios** \(V/V_{\mathrm{free}}\) from the final **multi-k**
(Monkhorst–Pack 2×2×1, time-reversal → 2 irreducible k) **collinear UKS**
GPW PBE total valence density
\(\rho=\sum_k w_k(\rho_k^\alpha+\rho_k^\beta)\), used to density-scale free-atom
**MBD@rsSCS** parameters (Tkatchenko & Scheffler, *Phys. Rev. Lett.* **102**,
073005 (2009); Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014)).

Extends Γ-UKS Hirshfeld (`reg_gpw-hirshfeld-uks`) and multi-k RKS Hirshfeld
(`reg_gpw-hirshfeld-kpoints`) to the combined path in
`fold_dispersion_k_uks`.

Public API:

```cpp
GpwDft gpw;
gpw.n_alpha = 7;
gpw.n_beta  = 5;
gpw.mbd = true;
gpw.hirshfeld = true;
gpw.k_mesh[0] = 2; gpw.k_mesh[1] = 2; gpw.k_mesh[2] = 1;
gpw.k_time_reversal = true;
```

Dispersion energy on `GpwResult::e_disp` [Hartree]; ratios on
`GpwResult::volume_ratios`. Free-atom path (`hirshfeld=false`) remains
geometry-only. MBD is **isolated** on the nuclear coordinates
(`mbd_k_mesh={0,0,0}`).

Free-atom valence densities (in-tree exponential model,
`basis_data/hirshfeld_tables.h`): same as `gpw-mbd-hirshfeld` /
`gpw-hirshfeld-uks`. O: \(N=6\), \(\alpha=2.2\) Bohr⁻¹.

Scaling of free-atom α₀ / C₆ / R_vdW: \(\alpha\propto v\), \(C_6\propto v^2\),
\(R\propto v^{1/3}\).

## Reference value (hard gate)

**E_disp (PBE-MBD@rsSCS, multi-k UKS Hirshfeld volumes, O₂, CUTOFF 120 Ry,
k = 2×2×1, TR fold) = −1.0427466367435301×10⁻⁴ Ha**

Cross-checks (offline — never in ctest):

| Source | Note | E_disp [Ha] |
| ------ | ---- | ----------- |
| **ATV product** (`GpwDft` UKS mbd+hirshfeld, k=2×2×1, CUTOFF 120) | dual gate | **−1.0427466367435301×10⁻⁴** |
| Free-atom MBD (volume_ratio = 1) same O₂ geometry | public fold dual | −1.1699749375715385×10⁻⁴ |
| Γ-UKS Hirshfeld O₂ (MT 12 Å, CUTOFF 200) | `reg_gpw-hirshfeld-uks` | −1.0379964698969957×10⁻⁴ |

Hirshfeld scaling reduces |E_disp| vs free-atom (compressed molecular volumes
\(v_A\sim 0.68\)).

### Dual gates

| Gate | Criterion | Tol |
| ---- | --------- | --- |
| A1 | \|E_disp product − E_disp free-atom\| and \|product\| < \|free\| | > 1e-6 Ha move |
| A2 | absolute E_disp pin | 1e-8 Ha |
| structural | n_kpts ≥ 2; two volumes in (0.2, 2.0); SCF converged | — |

### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no TS/MBD. Offline twin notes under
`tools/cp2k_oracle/inputs/gpw-hirshfeld-uks-kpoints/`. Free-atom MBD already
pinned vs libMBD Fortran (`gpw-mbd-ice`); Γ-UKS and multi-k RKS Hirshfeld
already productized. This slug pins the **multi-k UKS density → volume-ratio**
layer.

## Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 5.000000 | 5.000000 | 4.39625 |
| O | 5.000000 | 5.000000 | 5.60375 |

R(O–O) = **1.2075 Å**. Cell: **ABC 10.0 10.0 10.0 Å**, **PERIODIC XYZ**,
**k-mesh 2×2×1** (Monkhorst–Pack, time-reversal fold → 2 irreducible k).

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (6 valence e⁻ / O → 12 total).
CUTOFF **120 Ry**, max_iter 80, tol 1e-6, mix 0.45, diis_depth 10, PBE,
**Nα=7, Nβ=5** (multiplicity 3).

Representative volume ratios at the pin (order O, O):

`0.6823, 0.6823`

## Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit test: multi-k UKS path reports
volumes + attractive E_disp (`test_cpu_gpw_hirshfeld`).

## Perf gate

**perf_exempt = true** — primary deliverable is multi-k UKS E_disp +
volume-ratio parity. Stock CP2K has **no TS/MBD** algorithm twin (same policy
as `gpw-hirshfeld-uks` / `gpw-hirshfeld-kpoints` / `gpw-mbd-hirshfeld`).
Regression wall class ~**13 s** (OMP=8, Release) for the single product SCF.
