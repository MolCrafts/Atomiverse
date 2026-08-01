# Spec — gpw-vdw-uks-multigrid

## Goal

Product pin for the **Γ collinear-UKS × nonlocal vdW-DF (DRSLL) ×
REL_CUTOFF multigrid** triple already wired in `run_gpw_scf_impl` /
`build_uks_H` (each axis product alone via `gpw-vdw-uks`,
`gpw-vdw-multigrid`, `gpw-uks-multigrid`). Close the unpinned combo with an
offline CP2K twin where **both** sides use NGRIDS 4 / REL_CUTOFF 60.

## Capability claim

`cpu::gpw::GpwDft` / `GPWEvaluator` with `n_alpha`/`n_beta` + `vdw_df` +
`ngrids>1` + `rel_cutoff>0` runs a variational SCF: \(V_{\mathrm{nl}}[\rho]\)
of the **total** density \(\rho=\rho_\alpha+\rho_\beta\) is added to both
spin potentials on the fine grid **before** per-level restriction; \(E_c^{\mathrm{nl}}\)
lands on `GpwResult::e_disp`.

## Oracle

- **Family:** `oracle: cp2k`
- **Twin:** O₂ triplet LSD PBE + `&NON_LOCAL TYPE DRSLL`, PERIODIC 12 Å cubic,
  DZVP-MOLOPT-GTH + GTH-PBE-q6, CUTOFF 400 / REL_CUTOFF 60 / **NGRIDS 4**
- **Input:** `tools/cp2k_oracle/inputs/gpw-vdw-uks-multigrid/o2_uks_pbe_drsll_mg.inp`
- **Binary:** CP2K 2026.1 source build (`cp2k.psmp`, git:5e54ba2)
- **Fair wall:** same single twin job, 1 MPI × OMP=8; \(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\le 10\)

## Acceptance

1. ATV O₂ UKS PBE+DRSLL multigrid SCF converges; `|E_tot − CP2K| ≤ 1e−3 Ha`,
   `|E_disp − CP2K| ≤ 1e−4 Ha`.
2. `e_disp ≠ 0` (E_c^nl actually folded on the multigrid product path).
3. Fair wall within 10× of the offline CP2K twin (same NGRIDS 4 job).

## Out of scope

- Multi-k UKS + nonlocal vdW (`gpw-vdw-uks-kpoints`)
- Forces / stress × UKS × vdW × multigrid
- vdW-DF2 / rVV10 / optB88-vdW variants (same product path; DRSLL is the pin)

## Implementation note

No new algorithm surface is required if `build_uks_H` already adds V_nl under
`use_prod` (product path). Worker verifies the combo does not throw, pins
numbers, registers `reg_gpw-vdw-uks-multigrid`. If a residual reject is found,
lift it in the same change.
