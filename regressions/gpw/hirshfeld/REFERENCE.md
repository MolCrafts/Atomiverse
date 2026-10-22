# gpw/hirshfeld

One case; every section below is run by `atv_regressions --case gpw/hirshfeld` and keeps its own oracle record.


## Reference — water-dimer Hirshfeld-scaled TS/MBD ionic forces

### Method

**Analytic ionic forces** (and fixed-\(P\) cell stress fold) with
**Hirshfeld volume ratios** \(v_A = V_A/V_A^{\mathrm{free}}\) scaling free-atom
**TS** and **MBD@rsSCS** parameters, on top of Γ-point **GPW PBE**.

Found rot (pre-fix): energy folded post-SCF volumes into `ts_pbe` /
`mbd_rsscs_pbe`, but `gpw_forces` / `gpw_stress` called the free-atom kernels
with empty `vol_ratio` — forces/stress ignored `GpwDft::hirshfeld`.

Product definition (density-chain \(\partial v/\partial R\) out of scope):

\[
F_A = -\frac{\partial E_{\mathrm{disp}}}{\partial R_A}\Big|_{v}
\quad\text{with \(v\) fixed at the SCF density},
\]

same fixed-\(v\) functional rebuilt under orthorhombic strain for analytic
stress.

Public API:

```cpp
GpwDft gpw{kinds, /*cutoff_ry=*/120.0, XcKind::PBE};
gpw.ts = true;          // or gpw.mbd = true
gpw.hirshfeld = true;   // post-SCF volumes → energy, forces, stress
gpw.build(frame);
const auto& scf = gpw.forces(frame);  // SCF + forces → kAtom_fx/fy/fz [eV/Å]
```

Scaling of free-atom \(\alpha_0\) / \(C_6\) / \(R_{\mathrm{vdW}}\):
\(\alpha\propto v\), \(C_6\propto v^2\), \(R\propto v^{1/3}\).

### Reference values (hard gates)

| Quantity | Source | Value / tol |
| -------- | ------ | ----------- |
| **E_disp isolated MBD@rsSCS + fixed \(v\)** | ATV product kernel = dual (same volumes as `gpw-mbd-hirshfeld`) | **−1.1479717148086300×10⁻³ Ha** (fixed dual volumes below), tol 1×10⁻⁸ |
| **Force dual-gate (TS, MBD)** | product \(F\) vs central FD \(h=10^{-5}\) Bohr of fixed-\(v\) energy | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| **Translation** | \(\|\sum_A F_A\|\) (MBD) | ≤ **1×10⁻⁸** Ha/Bohr |
| **Product wiring** | \(F(\mathrm{ts}+\mathrm{hirsh})-F(\mathrm{plain})\) vs `ts_pbe(…, scf.volume_ratios)` | max\|Δ\| ≤ **1×10⁻⁷** Ha/Bohr |
| **Not free-atom** | max\|F_hirsh − F_free\| and wiring residual vs free-atom | must exceed noise |
| **Stress fold** | analytic \(\sigma_{xx}(\mathrm{ts}+\mathrm{hirsh})\) vs free-atom TS at same SCF state | \|Δσ\| > 1×10⁻¹² Ha/Bohr³ |

Representative fixed volumes (order O,H,H,O,H,H), matching
`reg_gpw-mbd-hirshfeld`:

`0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784` (fixed dual table; product SCF ratios ≈ these)

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — **no** stock TS/MBD. Hard pin is **libMBD /
product dual** (force FD of the same fixed-\(v\) energy) plus product wiring.
Twin notes: `tools/cp2k_oracle/inputs/gpw-hirshfeld-forces/`.

**oracle:** literature / libMBD (volume-scaled gradient dual).  
**perf_exempt:** true — no fair CP2K twin algorithm (same policy as
`gpw-ts-forces` / `gpw-mbd-hirshfeld` / `gpw-mbd-forces`).

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (product SCF): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**.  
Isolated dual-gates use the same nuclear coordinates without lattice images.

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).  
CUTOFF **120 Ry** (product wiring + `GpwDft::stress` fold), n_occ 8, tol 1e-6, mix 0.4, PBE.

### Perf gate

**perf_exempt = true** — primary deliverable is fixed-\(v\) force dual-gate +
product wiring parity. CP2K has no matched TS/MBD ENERGY_FORCE twin for a fair
wall ratio. Product path wall budget ≤ 60 s on the regression host (four
single SCFs — ts+hirsh forces, plain forces, two `GpwDft::stress` — not
multi-SCF strain-FD).


## Reference — O₂ triplet Γ-UKS GPW PBE + Hirshfeld-scaled MBD@rsSCS

### Method

**Hirshfeld volume ratios** \(V/V_{\mathrm{free}}\) from the final Γ-point
**UKS GPW PBE** total valence density \(\rho=\rho_\alpha+\rho_\beta\), used to
density-scale free-atom **MBD@rsSCS** parameters (Tkatchenko & Scheffler,
*Phys. Rev. Lett.* **102**, 073005 (2009); Ambrosetti et al., *J. Chem. Phys.*
**140**, 18A508 (2014)).

Public API:

```cpp
GpwDft gpw{kinds, /*cutoff_ry=*/200.0, XcKind::PBE};
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

### Reference value (hard gate)

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

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no TS/MBD. Offline twin notes under
`tools/cp2k_oracle/inputs/gpw-hirshfeld-uks/`. Free-atom MBD already pinned vs
libMBD Fortran (`gpw-mbd-ice`); this slug pins the **UKS volume-ratio layer**.

### Geometry

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

### Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests: UKS product path physical
bounds; closed-shell water UKS vs RKS E_disp within 5e-6 Ha.

### Perf gate

**perf_exempt = true** — primary deliverable is E_disp + volume-ratio parity
on the UKS density path. Hirshfeld grid integral is \(O(N_{\mathrm{grid}} N_{\mathrm{atoms}})\);
SCF dominates wall (same policy as `gpw-mbd-hirshfeld` / `gpw-hirshfeld-kpoints`).


## Reference — water-dimer multi-k GPW PBE + Hirshfeld-scaled MBD@rsSCS

### Method

**Hirshfeld volume ratios** \(V/V_{\mathrm{free}}\) from the final **multi-k**
(Monkhorst–Pack 2×2×2, time-reversal) **GPW PBE** valence density, used to
density-scale free-atom **MBD@rsSCS** parameters (Tkatchenko & Scheffler,
*Phys. Rev. Lett.* **102**, 073005 (2009); Ambrosetti et al., *J. Chem. Phys.*
**140**, 18A508 (2014)). Extends Γ-only Hirshfeld (`reg_gpw-mbd-hirshfeld`).

Public API:

```cpp
GpwDft gpw{kinds, /*cutoff_ry=*/200.0, XcKind::PBE};
gpw.mbd = true;
gpw.hirshfeld = true;
gpw.k_mesh[0] = gpw.k_mesh[1] = gpw.k_mesh[2] = 2;
```

Dispersion energy on `GpwResult::e_disp` [Hartree]; ratios on
`GpwResult::volume_ratios`. Free-atom multi-k path (`hirshfeld=false`) remains
geometry-only pre-SCF. Isolated MBD (default `mbd_k_mesh={0,0,0}`).

Free-atom valence densities (in-tree exponential model,
`basis_data/hirshfeld_tables.h`): same as `gpw-mbd-hirshfeld`.

Scaling of free-atom α₀ / C₆ / R_vdW: \(\alpha\propto v\), \(C_6\propto v^2\),
\(R\propto v^{1/3}\).

### Reference value (hard gate)

**E_disp (PBE-MBD@rsSCS, multi-k Hirshfeld volumes, water dimer, CUTOFF 200 Ry,
k = 2×2×2, TR fold) = −1.1479849553026611×10⁻³ Ha**

Cross-checks (offline — never in ctest):

| Source | Note | E_disp [Ha] |
| ------ | ---- | ----------- |
| **ATV product** (`GpwDft` mbd+hirshfeld, k=2×2×2, CUTOFF 200, tol 1e-6) | dual gate | **−1.1479849553026611×10⁻³** |
| Γ Hirshfeld MBD (same geom, no k-mesh) | `reg_gpw-mbd-hirshfeld` | −1.1479852469005181×10⁻³ |
| Free-atom MBD (volume_ratio = 1) | `reg_gpw-mbd-ice` | −1.4928804171407961×10⁻³ |

On this 12 Å molecular cell the multi-k vs Γ density difference is tiny
(\(\Delta E_{\mathrm{disp}}\sim 3\times 10^{-10}\) Ha); the hard pin freezes the
multi-k product path (n_kpts = 4 with TR fold).

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no TS/MBD. Offline twin notes under
`tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoints/`. Free-atom / Γ Hirshfeld
already pinned; this slug pins the **k-summed density → volume-ratio** layer.

### Geometry

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell (SCF): **ABC 12.0 12.0 12.0 Å**, **PERIODIC XYZ**, **k-mesh 2×2×2**
(Monkhorst–Pack, time-reversal fold → 4 irreducible k).

Basis / PP: `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q1` (H) / `GTH-PBE-q6` (O).
CUTOFF **200 Ry**, max_iter 16, tol 1e-6, mix 0.4, PBE.

Representative volume ratios at the pin (order O,H,H,O,H,H):

`0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784`

### Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit tests: multi-k path reports
volumes + attractive E_disp; UKS + hirshfeld still throws.

### Perf gate

**perf_exempt = true** — primary deliverable is multi-k E_disp + volume-ratio
parity. SCF dominates wall (same policy as `gpw-mbd-hirshfeld` / `gpw-mbd-ice`).


## Reference — multi-k collinear UKS GPW PBE + Hirshfeld-scaled MBD@rsSCS

### Method

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
GpwDft gpw{kinds, /*cutoff_ry=*/120.0, XcKind::PBE};
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

### Reference value (hard gate)

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

#### Dual gates

| Gate | Criterion | Tol |
| ---- | --------- | --- |
| A1 | \|E_disp product − E_disp free-atom\| and \|product\| < \|free\| | > 1e-6 Ha move |
| A2 | absolute E_disp pin | 1e-8 Ha |
| structural | n_kpts ≥ 2; two volumes in (0.2, 2.0); SCF converged | — |

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no TS/MBD. Offline twin notes under
`tools/cp2k_oracle/inputs/gpw-hirshfeld-uks-kpoints/`. Free-atom MBD already
pinned vs libMBD Fortran (`gpw-mbd-ice`); Γ-UKS and multi-k RKS Hirshfeld
already productized. This slug pins the **multi-k UKS density → volume-ratio**
layer.

### Geometry

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

### Tolerance

**1 × 10⁻⁸ Ha** on E_disp (regression). Unit test: multi-k UKS path reports
volumes + attractive E_disp (`test_cpu_gpw_hirshfeld`).

### Perf gate

**perf_exempt = true** — primary deliverable is multi-k UKS E_disp +
volume-ratio parity. Stock CP2K has **no TS/MBD** algorithm twin (same policy
as `gpw-hirshfeld-uks` / `gpw-hirshfeld-kpoints` / `gpw-mbd-hirshfeld`).
Regression wall class ~**31 s** (OMP=8, Release, measured 2026-08-01 after rebase onto `dev`) for the single product SCF; still ≤ 60 s class.


## Reference — multi-k GPW forces with Hirshfeld-scaled TS/MBD volumes

### Method

**Monkhorst–Pack multi-k GPW** analytic ionic forces with **Hirshfeld volume
ratios** \(v_A = V_A/V_A^{\mathrm{free}}\) scaling free-atom **TS** and
**MBD@rsSCS** parameters.

Closes the residual that multi-k energy Hirshfeld
(`gpw-hirshfeld-kpoints`) and Γ force Hirshfeld (`gpw-hirshfeld-forces`)
were product-pinned while multi-k force assembly already folds
`ts_mbd_vol_ratio` / SCF `volume_ratios` into `ts_pbe` / `mbd_rsscs_pbe`
without a dedicated dual-gate pin.

Product definition (density-chain \(\partial v/\partial R\) out of scope):

\[
F_A = -\frac{\partial E_{\mathrm{disp}}}{\partial R_A}\Big|_{v}
\quad\text{with \(v\) fixed at the multi-k SCF density}.
\]

Public API:

```cpp
GpwDft gpw{kinds, /*cutoff_ry=*/100.0, XcKind::PADE};
gpw.ts = true;          // or gpw.mbd = true
gpw.hirshfeld = true;
gpw.k_mesh[0] = gpw.k_mesh[1] = 1;
gpw.k_mesh[2] = 2;
gpw.build(frame);
const auto& scf = gpw.forces(frame);  // SCF + forces → kAtom_fx/fy/fz [eV/Å]
```

Scaling of free-atom \(\alpha_0\) / \(C_6\) / \(R_{\mathrm{vdW}}\):
\(\alpha\propto v\), \(C_6\propto v^2\), \(R\propto v^{1/3}\).

### Reference values (hard gates)

#### A/B/C — Geometry-only fixed-\(v\) dual (water dimer, isolated)

| Quantity | Source | Value / tol |
| -------- | ------ | ----------- |
| **E_disp isolated MBD@rsSCS + fixed \(v\)** | product kernel = dual (same volumes as `gpw-hirshfeld-forces`) | **−1.1479717148086300×10⁻³ Ha**, tol 1×10⁻⁸ |
| **Force dual-gate (TS, MBD)** | product \(F\) vs central FD \(h=10^{-5}\) Bohr of fixed-\(v\) energy | max\|ΔF\| ≤ **1×10⁻⁶** Ha/Bohr |
| **Translation** | \(\|\sum_A F_A\|\) (MBD) | ≤ **1×10⁻⁸** Ha/Bohr |
| **Not free-atom** | max\|F_hirsh − F_free\| (TS) | must exceed **1×10⁻⁸** Ha/Bohr |

Representative fixed volumes (O,H,H,O,H,H):

`0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784`

#### D/E — Multi-k product wiring (H₂, k-mesh 1×1×2)

Compressed H₂ (same geometry as `gpw-mbd-kpoint-forces`):
PADE, DZV-GTH-PADE, `k_mesh={1,1,2}`, CUTOFF 100 Ry.

| Quantity | Gate |
| -------- | ---- |
| SCF converged; multi-k payload (`n_kpts ≥ 1`, `C_occ_k` non-empty) | hard |
| `volume_ratios` size 2; \(E_{\mathrm{disp}} < 0\) | hard |
| \(F^{k+\mathrm{ts}+\mathrm{hirsh}}-F^{k,\mathrm{plain}}\approx F_{\mathrm{TS}}(v_{\mathrm{SCF}})\) | max\|Δ\| ≤ **1×10⁻⁷** Ha/Bohr |
| same for MBD@rsSCS (isolated when `mbd_k_mesh` default zero) | max\|Δ\| ≤ **1×10⁻⁷** Ha/Bohr |
| residual vs free-atom forces nontrivial | max\|Δ_free\| > **1×10⁻⁹** Ha/Bohr |

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — **no** stock TS/MBD. Hard pin is **libMBD /
product dual** + multi-k product wiring. Twin notes:
`tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoint-forces/`.

**oracle:** literature / libMBD (volume-scaled gradient dual).  
**perf_exempt:** true — no fair CP2K twin algorithm (same policy as
`gpw-hirshfeld-forces` / `gpw-mbd-kpoint-forces`).

Host binary check (2026-08-01):

```
cp2k.psmp --version → CP2K version 2026.1 (git:5e54ba2)
```

### Geometry

#### Water dimer (Å) — fixed-\(v\) dual

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Isolated dual-gates use these nuclei without lattice images.

#### H₂ (Bohr) — multi-k wiring

| Atom | x | y | z |
| ---- | - | - | - |
| H | 5.5 | 6.0 | 6.0 |
| H | 6.5 | 6.0 | 6.0 |

Cell: **12 × 12 × 12 Bohr**, periodic; electronic **k = 1×1×2**;
basis `DZV-GTH-PADE` + `GTH-PADE-q1`; XC PADE; CUTOFF **100 Ry**.

### Perf gate

**perf_exempt = true** — primary deliverable is fixed-\(v\) force dual-gate +
multi-k product wiring. CP2K has no matched TS/MBD ENERGY_FORCE twin for a
fair wall ratio. Regression wall budget ≤ 60 s (two H₂ multi-k SCFs +
geometry duals).


## Reference — multi-k GPW cell stress with Hirshfeld-scaled TS/MBD volumes

### Method

**Monkhorst–Pack multi-k GPW** analytic cell stress with **Hirshfeld volume
ratios** \(v_A = V_A/V_A^{\mathrm{free}}\) scaling free-atom **TS** and
**MBD@rsSCS** parameters under orthorhombic diagonal strain.

Closes the residual that multi-k energy Hirshfeld
(`gpw-hirshfeld-kpoints`), multi-k force Hirshfeld
(`gpw-hirshfeld-kpoint-forces`), and Γ stress Hirshfeld (gate inside
`gpw-hirshfeld-forces`) were product-pinned while the multi-k stress path
already folds `scf.volume_ratios` into the fixed-\(P_k\) Lagrangian without a
dedicated dual-gate pin.

Product definition (density-chain \(\partial v/\partial\varepsilon\) out of
scope):

\[
\sigma_{aa}
  = -\Omega^{-1}
    \frac{\partial E_L[\{P_k\},\{W_k\};\varepsilon]}{\partial\varepsilon_{aa}}
\Big|_{v}
\quad\text{with \(v\) fixed at the multi-k SCF density}.
\]

Public API:

```cpp
GpwDft gpw{kinds, /*cutoff_ry=*/100.0, XcKind::PADE};
gpw.ts = true;          // or gpw.mbd = true
gpw.hirshfeld = true;
gpw.k_mesh[0] = gpw.k_mesh[1] = 1;
gpw.k_mesh[2] = 2;
gpw.stress_strain_h = 2e-3;
gpw.build(frame);
auto sig = gpw.stress(frame);          // one SCF + fixed-P stress [eV/Å³]
const auto& scf = gpw.last_result();   // volume_ratios, e_disp
```

Scaling of free-atom \(\alpha_0\) / \(C_6\) / \(R_{\mathrm{vdW}}\):
\(\alpha\propto v\), \(C_6\propto v^2\), \(R\propto v^{1/3}\).

### Reference values (hard gates)

#### A/B — Geometry-only fixed-\(v\) dual (water dimer)

| Quantity | Source | Value / tol |
| -------- | ------ | ----------- |
| **E_disp isolated MBD@rsSCS + fixed \(v\)** | product kernel = dual (same volumes as `gpw-hirshfeld-forces`) | **−1.1479717148086300×10⁻³ Ha**, tol 1×10⁻⁸ |
| **Geometry stress dual (TS, MBD)** | central FD of fixed-\(v\) \(E_{\mathrm{disp}}\) under orthorhombic strain \(h=2\times10^{-3}\) | \(\sigma_{\mathrm{disp}}\) finite and nonzero; self dual ≤ **1×10⁻¹²** Ha/Bohr³ |
| **Not free-atom** | max\|σ_hirsh − σ_free\| (TS) | must exceed **1×10⁻¹²** Ha/Bohr³ |

Representative fixed volumes (O,H,H,O,H,H):

`0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784`

#### C/D — Multi-k product wiring (H₂, k-mesh 1×1×2)

Compressed H₂ (same geometry as `gpw-hirshfeld-kpoint-forces` /
`gpw-mbd-kpoint-forces`): PADE, DZV-GTH-PADE, `k_mesh={1,1,2}`, CUTOFF 100 Ry.

| Quantity | Gate |
| -------- | ---- |
| SCF converged; multi-k payload (`n_kpts ≥ 1`, `C_occ_k` non-empty) | hard |
| `volume_ratios` size 2; \(E_{\mathrm{disp}} < 0\) | hard |
| \(\sigma^{k+\mathrm{ts}+\mathrm{hirsh}}-\sigma^{k,\mathrm{plain}}\approx\sigma_{\mathrm{TS}}(v_{\mathrm{SCF}})\) | max\|Δ\| ≤ **1×10⁻⁸** Ha/Bohr³ |
| same for MBD@rsSCS (isolated when `mbd_k_mesh` default zero) | max\|Δ\| ≤ **1×10⁻⁸** Ha/Bohr³ |
| residual vs free-atom stress nontrivial | max\|Δ_free\| > **1×10⁻¹²** Ha/Bohr³ |
| orthorhombic off-diagonals | exactly 0 |

#### Why not CP2K QS?

**CP2K 2026.1** stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — **no** stock TS/MBD. Hard pin is **libMBD /
product dual** + multi-k product wiring. Twin notes:
`tools/cp2k_oracle/inputs/gpw-hirshfeld-kpoint-stress/`.

**oracle:** literature / libMBD (volume-scaled strain dual).  
**perf_exempt:** true — no fair CP2K twin algorithm (same policy as
`gpw-hirshfeld-forces` / `gpw-hirshfeld-kpoint-forces` / `gpw-mbd-kpoint-stress`).

Host binary check (2026-08-01):

```
cp2k.psmp --version → CP2K version 2026.1 (git:5e54ba2)
```

### Geometry

#### Water dimer (Å) — fixed-\(v\) dual

| Atom | x | y | z |
| ---- | - | - | - |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Isolated MBD dual uses these nuclei without lattice images. Periodic TS dual
uses the same coordinates in a **12 × 12 × 12 Å** cube (`PERIODIC XYZ`).

#### H₂ (Bohr) — multi-k wiring

| Atom | x | y | z |
| ---- | - | - | - |
| H | 5.5 | 6.0 | 6.0 |
| H | 6.5 | 6.0 | 6.0 |

Cell: **12 × 12 × 12 Bohr**, periodic; electronic **k = 1×1×2**;
basis `DZV-GTH-PADE` + `GTH-PADE-q1`; XC PADE; CUTOFF **100 Ry**.

### Perf gate

**perf_exempt = true** — primary deliverable is fixed-\(v\) stress dual-gate +
multi-k product wiring. CP2K has no matched TS/MBD `STRESS_TENSOR` twin for a
fair wall ratio. Regression wall budget ≤ 60 s (two H₂ multi-k SCFs +
geometry duals).
