# Reference — multi-k GPW PBE + vdW-DF (DRSLL) on the closed-shell RKS path

## Method

Self-consistent nonlocal vdW-DF (DRSLL) correlation on **Monkhorst-Pack
k-point** **GPW PBE** (closed-shell RKS) — the combination the multi-k runner
used to reject fail-fast (`"nonlocal vdW (vdw_df / vdw_df2 / rvv10)"` at
`GpwDft::build`). SCF \(V_{\mathrm{nl}} = \delta E_c^{\mathrm{nl}} / \delta\rho\)
is evaluated on the finest-grid k-summed density and added to `veff` every
iteration **before** `last_veff` / multigrid restriction; the post-SCF
\(E_c^{\mathrm{nl}}\) is reported via `e_disp` (never in the SCF convergence
energy). Placement mirrors the Γ runner (`gpw-vdw-scf-potential` /
`gpw-vdw-multigrid`).

Public API: `cpu::gpw::GpwDft` members `vdw_df` (also `vdw_df2` / `rvv10`)
together with `k_mesh` / explicit `k_points` become legal for closed-shell
RKS (single-grid or REL_CUTOFF multigrid). UKS + nonlocal vdW stays
fail-fast (queued `gpw-vdw-uks`).

CP2K twin surface: `&KPOINTS SCHEME MONKHORST-PACK 2 1 1` + `&MGRID NGRIDS 4 /
REL_CUTOFF 40` + `&VDW_POTENTIAL POTENTIAL_TYPE NON_LOCAL / &NON_LOCAL TYPE
DRSLL`. KPOINTS forces diagonalization SCF (no OT) on the CP2K side; ATV uses
the multi-k DIIS path.

## Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, source revision `85b8a9b49c`),
  Arrhenius local install (same binary as `gpw-vdw-multigrid` /
  `gpw-vdw-scf-potential`).
- **Input:** `tools/cp2k_oracle/inputs/gpw-vdw-kpoints/water_dimer_vdw_df_kp.inp`
  (water dimer, 12 Å cell, PBE, DZVP-MOLOPT-SR-GTH + GTH-PBE, CUTOFF 200,
  NGRIDS 4 / REL_CUTOFF 40, EPS_SCF 1e-7, EPS_DEFAULT 1e-10, DIAGONALIZATION
  + DIRECT_P_MIXING; DRSLL kernel `vdW_kernel_table.dat`; MP 2×1×1).
- **Run (offline only; never in ctest):**
  ```bash
  export LD_LIBRARY_PATH=$CP2K_INSTALL/lib:$LD_LIBRARY_PATH
  ln -sf $CP2K_DATA/{BASIS_MOLOPT,GTH_POTENTIALS,vdW_kernel_table.dat} .
  OMP_NUM_THREADS=8 cp2k.psmp -i water_dimer_vdw_df_kp.inp -o water_dimer_vdw_df_kp.out
  ```
- **Date:** 2026-07-31. Key lines
  (`grep 'ENERGY| Total\|Dispersion energy'`):

| Quantity | CP2K MP 2×1×1 NGRIDS 4 |
|----------|------------------------|
| E_tot [Ha] | **−34.186333514352548** |
| Dispersion energy [Ha] | **+0.15193559113836** |
| SCF | converged, 15 steps |
| wall (OMP=8) | **18.4 s** |

Context: the Γ multigrid twin (`gpw-vdw-multigrid`) is E_tot
−34.186333524093456 Ha / E_disp +0.15193559412669 Ha — the large 12 Å cell
makes MP 2×1×1 essentially Γ for this dimer (Δ ~1e-11 Ha class on the
oracle).

## Geometry

Same water dimer as `gpw-vdw-df-ice` / `gpw-vdw-multigrid`:

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0.000000 | 0.000000 | 0.000000 |
| H | 0.758602 | 0.000000 | 0.504284 |
| H | 0.758602 | 0.000000 | −0.504284 |
| O | 3.000000 | 0.000000 | 0.000000 |
| H | 3.758602 | 0.000000 | 0.504284 |
| H | 3.758602 | 0.000000 | −0.504284 |

Cell **ABC 12.0 12.0 12.0 Å**, PERIODIC XYZ. Basis `DZVP-MOLOPT-SR-GTH`,
PP `GTH-PBE-q1`/`GTH-PBE-q6`. ATV: CUTOFF 200 Ry, DIIS, tol 1e-7 Ha;
G1 single-grid; G2 `ngrids=4`, `rel_cutoff=40`, `k_mesh={2,1,1}`,
`k_time_reversal=false`.

## Gates (measured 2026-07-31, OMP=8)

| Gate | Measured | Tolerance |
|------|----------|-----------|
| G1 \|E_tot(kΓ) − E_tot(Γ)\| | **9.7e-10 Ha** | ≤ 1e-8 Ha |
| G1 \|E_disp(kΓ) − E_disp(Γ)\| | **9.7e-10 Ha** | ≤ 1e-8 Ha |
| G2 \|E_disp(MP mg) − CP2K\| | **4.02e-5 Ha** | ≤ 1e-4 Ha |
| G2 \|E_tot(MP mg) − CP2K\| | **7.93e-3 Ha** | ≤ 1e-2 Ha |

ATV G2 values: E_tot = −34.178407374593 Ha, E_disp = +0.151895422335 Ha.
The absolute E_tot offset is the **pre-existing Γ-dimer twin-family systematic**
(identical to `gpw-vdw-multigrid` Δ ≈ 7.93e-3 Ha); the capability statements
are the Γ-limit identity and the E_disp pin.

## Out of scope here

- UKS nonlocal vdW (`gpw-vdw-uks`).
- Analytic k-point vdW force/stress gates (V_nl already rides the k force
  assembly's grid-term `veff`; explicit gates are a follow-up).
- CUDA.

## Perf

| Run | Wall (OMP=8) |
|-----|--------------|
| CP2K MP 2×1×1 NGRIDS 4 | **18.4 s** |
| ATV G2 only (MP multigrid) | **13.7 s** |
| ATV full regression (G1×2 + G2) | **48.0 s** (≤ 60 s budget) |

\(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \approx 0.74\) on the G2 parity twin (≤ 10×;
no exemption needed).
