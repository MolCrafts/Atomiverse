# Nonlocal vdW-DF on the REL_CUTOFF multigrid — water dimer / PBE+DRSLL / NGRIDS 4

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-vdw-multigrid` |
| Regression | `regressions/gpw/gpw-vdw-multigrid/` |
| Public API | `cpu::gpw::GpwDft` (`vdw_df` / `vdw_df2` / `rvv10`, `ngrids`, `rel_cutoff`, `cutoff_ry`); lookup ctors `GaussianBasis` / `GthPotential` |
| Roadmap | `docs/roadmap/dft.md` §8 — nonlocal vdW × multigrid |
| Queue | CP2K capability loop (`docs/developer/cp2k-method-queue.md`) |

## Summary

**Self-consistent nonlocal vdW correlation (vdW-DF DRSLL, vdW-DF2 LMKLL,
rVV10) on the CP2K-style `CUTOFF`/`REL_CUTOFF` multigrid product path** — the
combination the Γ runner used to reject
(`"GPW vdw_df: multigrid not supported (single-grid only)"`). CP2K's own
default is `NGRIDS 4`, so every stock CP2K vdW input is a multigrid input;
this page opens the product of the shipped SC nonlocal potential
([`gpw-vdw-scf-potential`](gpw-vdw-scf-potential.md)) and the multigrid
product collocation ([`gpw-uks-multigrid`](gpw-uks-multigrid.md) /
[`gpw-multigrid-kpoints`](gpw-multigrid-kpoints.md)).

This page pins the H-bonded **water dimer (12 Å periodic cell, PBE + DRSLL,
CUTOFF 200 Ry, NGRIDS 4, REL_CUTOFF 40 Ry)** against offline **CP2K 2026.1**
(`ENERGY| Total` and `Dispersion energy`, NGRIDS 4 gate + NGRIDS 1 context).
Measured: ATV multigrid ↔ single-grid **4.28e-6 Ha** (CP2K's own gap
6.0e-7 Ha); E_disp parity vs CP2K **4.0e-5 Ha**. Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 0.83\) (both ATV solves vs
CP2K's one NGRIDS 4 run, same host, OMP=8). Product gate:
`reg_gpw-vdw-multigrid`.

Related: [Ch. 9 GPW](../ch09-gpw.md); nonlocal-vdW siblings
[`gpw-vdw-df-ice`](gpw-vdw-df-ice.md), [`gpw-vdw-df2-ice`](gpw-vdw-df2-ice.md),
`gpw-rvv10-ice`, [`gpw-optb88-vdw`](gpw-optb88-vdw.md).

## Method

**Nonlocal vdW is a functional of the finest-grid density.** On the product
path `build_H_and_energy` already prolongs every level's collocated density
spectrally to the finest grid before Poisson/XC and restricts the finest
`veff` back per level (exact adjoint of the prolongation). The Román–Pérez–
Soler \(V_{\mathrm{nl}}[\rho] = \delta E_c^{\mathrm{nl}}/\delta\rho\) is
evaluated on that fine ρ and added into `veff` at the same point as the
Hartree and semilocal XC potentials — **before** the per-level restriction —
so the integrated KS matrix stays the exact gradient of the collocated
\(E_H + E_{xc} + E_c^{\mathrm{nl}}\) and the SCF remains variational on both
paths. The final \(E_c^{\mathrm{nl}}\) re-collocates the SC density through
the same per-level products + spectral prolongation.

Citations: Dion et al., *PRL* **92**, 246401 (2004) (DRSLL); Lee et al.,
*PRB* **82**, 081101 (2010) (DF2); Sabatini et al., *PRB* **87**, 041108
(2013) (rVV10); Román-Pérez & Soler, *PRL* **103**, 096102 (2009) (RPS
kernel).

**Scope.** Γ-point closed-shell RKS. The k path keeps its fail-fast reject
(queued `gpw-vdw-kpoints`). Γ UKS + nonlocal vdW now rejects fail-fast at
`GpwDft::build` — it used to run and silently drop \(E_c^{\mathrm{nl}}\)
(`e_disp` stayed 0), found while opening this gate.

## Public API

```cpp
cpu::gpw::GpwDft gpw;
gpw.cell[0] = gpw.cell[1] = gpw.cell[2] = 12.0 * kA2B;
gpw.cutoff_ry = 200.0;
gpw.ngrids = 4;        // CP2K NGRIDS (its default is 4)
gpw.rel_cutoff = 40.0; // CP2K REL_CUTOFF [Ry]
gpw.n_occ = 8;
gpw.xc = cpu::gpw::XcKind::PBE;
gpw.vdw_df = true;     // or vdw_df2 / rvv10
gpw.build(atoms);
const cpu::gpw::GpwResult& r = gpw.run(atoms);
// r.e_total [Ha], r.e_disp = E_c^nl [Ha]
```

## Pin (offline CP2K 2026.1; never in ctest)

| Quantity | CP2K NGRIDS 4 (gate) | CP2K NGRIDS 1 (context) |
|----------|----------------------|-------------------------|
| E_tot [Ha] | −34.186333524093456 | −34.186334128002194 |
| Dispersion energy [Ha] | +0.15193559412669 | +0.15193559998706 |

CP2K's own NGRIDS 4 ↔ 1 gap is 6.0e-7 Ha — the multigrid is a cost
optimization, not a physics change; the regression pins the same statement on
the ATV side (single-grid ↔ multigrid internal consistency) plus the CP2K
E_disp / E_tot parity. Measured gates: `regressions/gpw/gpw-vdw-multigrid/REFERENCE.md`.
