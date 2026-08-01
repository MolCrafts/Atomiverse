# Spec — gpw-vdw-kpoint-forces

## Goal

Dedicated multi-k **analytic ionic force** pin for closed-shell RKS **GPW
PBE + self-consistent nonlocal vdW-DF (DRSLL)** on a Monkhorst–Pack mesh.
Energy product already closed (`reg_gpw-vdw-kpoints`); Γ SC potential + FD
force closed (`reg_gpw-vdw-scf-potential`). This slug hard-gates the k-path
force assembly that already folds \(V_{\mathrm{nl}}\) into `veff`.

## Public API (no new surface)

- `cpu::gpw::GpwDft` with `vdw_df = true` + `k_mesh` / explicit `k_points`
- `cpu::gpw::gpw_forces_from_scf` / `gpw_forces_analytic`
- Optional product: `GPWEvaluator` + `compute_forces = true`

No public `*Config`, no `make_*` / `build_*` factories. Frame remains sole I/O.

## Algorithm

1. Multi-k RKS SCF with SC nonlocal \(V_{\mathrm{nl}}[\rho]\) added to
   \(V_{\mathrm{eff}}\) every iteration (same as energy product).
2. Analytic forces: Hellmann–Feynman + Pulay on Bloch-phased operators;
   grid term uses `veff` that includes \(V_{\mathrm{nl}}\)
   (`assemble_gpw_forces_kpoints` in `gpw_forces.cpp`).
3. \(E_c^{\mathrm{nl}}\) reported as `e_disp` [Ha]; public Frame energy in eV
   when going through evaluators (internal Ha OK when cited).

## System (twin)

Periodic water dimer, same nuclei as `gpw-vdw-kpoints` / `gpw-vdw-df-ice`:

| Atom | x [Å] | y [Å] | z [Å] |
| ---- | ----- | ----- | ----- |
| O | 0 | 0 | 0 |
| H | 0.758602 | 0 | ±0.504284 |
| O | 3 | 0 | 0 |
| H | 3.758602 | 0 | ±0.504284 |

Cell **12³ Å**, `PERIODIC XYZ`, basis `DZVP-MOLOPT-SR-GTH`, PP `GTH-PBE`,
CUTOFF 200 Ry, MP **2×1×1**, DRSLL kernel.

## Offline oracle

- **Family:** `oracle: cp2k` (matched GTH-GPW + NON_LOCAL + KPOINTS)
- **Twin input:** `tools/cp2k_oracle/inputs/gpw-vdw-kpoint-forces/water_dimer_vdw_df_kp_forces.inp`
- **Run:** `ENERGY_FORCE`, 1 MPI × 8 OMP, record Total / Dispersion / FORCES / wall
- **Fair wall:** same twin job only — \(T_{\mathrm{atv,scf+force}}/T_{\mathrm{cp2k}}\le 10\)

## Gates

| Gate | What | Tol (initial) |
| ---- | ---- | ------------- |
| G1 | Multi-k SCF converges; `e_disp` vs CP2K Dispersion | ≤ 1e-4 Ha |
| G1 | \(E_{\mathrm{tot}}\) vs CP2K (Γ-dimer systematic allowed) | ≤ 1e-2 Ha |
| G2 | Analytic \(F\) vs CP2K `FORCES|` (absolute components) | measured; ≥ non-trivial signal |
| G3 | Analytic vs central FD on O0 \(x\) (k-path + vdW) | ≤ 5e-4 Ha/Bohr class |
| Perf | Fair wall ratio | ≤ 10 |

## Out of scope

- UKS multi-k vdW forces (other slugs)
- Analytic stress with nonlocal vdW
- CUDA
- Cross-representation (PW NC-UPF) twins

## Deps

`gpw-vdw-kpoints` ✓, `gpw-vdw-scf-potential` ✓
