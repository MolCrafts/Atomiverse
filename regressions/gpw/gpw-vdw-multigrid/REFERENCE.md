# Reference — Γ GPW PBE + vdW-DF (DRSLL) on the REL_CUTOFF multigrid (NGRIDS 4)

## Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on Γ-point **GPW PBE**
with **CP2K-style CUTOFF/REL_CUTOFF multigrid product collocation**
(`ngrids > 1 && rel_cutoff > 0`) — the combination the Γ runner used to
reject (`"GPW vdw_df: multigrid not supported (single-grid only)"`).

Why the gate simply opens: nonlocal vdW is a functional of the **finest-grid**
total density. The product path already prolongs every level's collocated
density to the finest grid before Poisson/XC and restricts the finest `veff`
back per level (exact adjoint). \(V_{\mathrm{nl}}[\rho]\) is evaluated on that
fine ρ and added to `veff` at the same point as Hartree + XC, **before** the
restriction — so the integrated KS matrix stays the exact gradient of the
collocated \(E_H + E_{xc} + E_c^{\mathrm{nl}}\) and the SCF is variational on
both paths. The final \(E_c^{\mathrm{nl}}\) re-collocates the SC density
through the same products + spectral prolongation.

Public API: `cpu::gpw::GpwDft` members `vdw_df` (also `vdw_df2` / `rvv10`),
`ngrids`, `rel_cutoff` — they merely become legal together. k-point vdW stays
fail-fast rejected (queued `gpw-vdw-kpoints`). **Found rot fixed in the same
cut:** Γ UKS + nonlocal vdW silently dropped \(E_c^{\mathrm{nl}}\)
(`e_disp` stayed 0 while the run "succeeded") — now rejected fail-fast at
`GpwDft::build`.

CP2K twin surface: `&MGRID NGRIDS 4 / REL_CUTOFF 40` (NGRIDS 4 is CP2K's
default) + `&VDW_POTENTIAL POTENTIAL_TYPE NON_LOCAL / &NON_LOCAL TYPE DRSLL`.

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, source revision `85b8a9b49c`),
  Arrhenius local install `~/work/cp2k/install` (same binary as the
  `gpw-vdw-scf-potential` pin).
- **Input:** `tools/cp2k_oracle/inputs/gpw-vdw-multigrid/water_dimer_vdw_df_mg.inp`
  (water dimer, 12 Å cell, PBE, DZVP-MOLOPT-SR-GTH + GTH-PBE, CUTOFF 200,
  EPS_SCF 1e-7, EPS_DEFAULT 1e-10, OT/DIIS; DRSLL kernel
  `vdW_kernel_table.dat`). NGRIDS 1 context = same input with `NGRIDS 1`.
- **Run (offline only; never in ctest):**
  ```bash
  ln -sf $CP2K_DATA/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  OMP_NUM_THREADS=8 cp2k.psmp -i water_dimer_vdw_df_mg.inp -o water_dimer_vdw_df_mg.out
  ```
- **Date:** 2026-07-31. Key lines (`grep 'ENERGY| Total\|Dispersion energy'`):

| Quantity | CP2K NGRIDS 4 (gate) | CP2K NGRIDS 1 (context) |
|----------|----------------------|-------------------------|
| E_tot [Ha] | **−34.186333524093456** | −34.186334128002194 |
| Dispersion energy [Ha] | **+0.15193559412669** | +0.15193559998706 |
| SCF | converged, 30 steps | converged, 30 steps |
| wall (OMP=8) | 24.7 s | 24.2 s |

CP2K's own NGRIDS 4 ↔ 1 gap: 6.0e-7 Ha (E_tot), 5.9e-9 Ha (E_disp) — the
multigrid is a cost optimization, not a physics change, which is what the
internal-consistency gate pins on the ATV side.

## Geometry

Same water dimer as `gpw-vdw-df-ice` / `gpw-vdw-scf-potential`:

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell **ABC 12.0 12.0 12.0 Å**, PERIODIC XYZ. Basis `DZVP-MOLOPT-SR-GTH`,
PP `GTH-PBE-q1`/`GTH-PBE-q6`. ATV: CUTOFF 200 Ry, `ngrids=4`,
`rel_cutoff=40`, DIIS, tol 1e-7 Ha.

## Gates (measured 2026-07-31, OMP=8, eps_default 1e-8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| \|E_tot(mg) − E_tot(sg)\| (ATV internal) | **4.28e-6 Ha** | 2e-5 Ha |
| \|E_disp(mg) − CP2K NGRIDS 4\| | **4.02e-5 Ha** | 1e-4 Ha |
| \|E_tot(mg) − CP2K NGRIDS 4\| | **7.93e-3 Ha** | 1e-2 Ha |

ATV values: E_tot sg = −34.178411658610, mg = −34.178407380745 Ha;
E_disp sg = +0.151895580634, mg = +0.151895426360 Ha.

The absolute ATV ↔ CP2K E_tot offset is the **pre-existing systematic of this
Γ dimer twin family** (never gated by the shipped single-grid vdW siblings):
it is identical on both grid paths (Δ_sg = 7.922e-3 vs Δ_mg = 7.926e-3 Ha —
they agree to the ATV internal-consistency level), and `eps_default = 1e-10`
moves E_tot by < 1e-8 Ha (measured), so it is not tail screening. The gate
pins that the multigrid total stays inside the single-grid envelope; the
capability statements are rows 1–2.

## Out of scope here

- k-point vdW (`gpw-vdw-kpoints`, queued) and UKS vdW (fail-fast reject).
- vdW forces/stress × multigrid: the analytic force assembly keeps its
  single-grid fine-cutoff collocation for **all** multigrid runs (pre-existing,
  functional-independent) — unchanged by this slug.

## Perf

CP2K NGRIDS 4 wall 24.7 s (OMP=8, same host). ATV regression wall **20.6 s
for BOTH solves** (sg + mg, OMP=8) →
\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \approx 0.83\) even charging both ATV
solves against CP2K's single multigrid run (≤ 10×; no exemption needed).
