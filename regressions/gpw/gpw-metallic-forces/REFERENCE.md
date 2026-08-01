# Reference — multi-k GPW free-energy analytic forces (FD smear)

## Method

Closed-shell **GPW** Kohn–Sham DFT with **Monkhorst–Pack k-point sampling**
and **Fermi–Dirac metallic occupation smearing**. Ionic forces are analytic
derivatives of the **Mermin free energy** \(F=E_{\mathrm{KS}}-TS\):

\[
P_k=\sum_b f_{kb}\,c_{kb}c_{kb}^\dagger,\qquad
W_k=\sum_b f_{kb}\varepsilon_{kb}\,c_{kb}c_{kb}^\dagger
\]

with fractional band occupations \(f_{kb}\in[0,2]\) from a shared \(\mu\).

**Payload fix:** `store_k_payload` fills `GpwResult::f_k` and rebuilds
fractional \(P_k\); force assembly uses full `C_k` + `f_k` (not integer
\(n_{\mathrm{occ}}\) truncation).

Public path: `GpwDft` members `smearing` / `degauss` / `extra_bands` +
`gpw_forces_analytic` / `gpw_forces_from_scf`. No public `*Config`, no factories.

## System (ATV correctness pin)

| | |
| --- | --- |
| Molecule | H₂, off-symmetry compressed bond |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.7, 6.1, 5.9) |
| Cell | cubic 12 Bohr, periodic XYZ |
| Basis | DZV-GTH-PADE (H) — in-tree |
| PP | GTH-PADE-q1 |
| XC | PADE |
| Cutoff | 100 Ry |
| k mesh | MONKHORST-PACK 1×1×2 (TR fold → nk=1) |
| Smear | Fermi–Dirac, `degauss = 0.02` Ry, `extra_bands = 2` |
| SCF | tol 1e-8 Ha, `local_pp_on_grid` |

## Offline correctness

### Analytic vs free-energy FD (decisive)

Central finite difference of \(F\) (`gpw_forces_fd`, \(h=10^{-3}\) Bohr) on the
same surface:

| Quantity | Value |
| --- | --- |
| max\|F_analytic − F_fd\| | **7.8×10⁻⁶ Ha/Bohr** |
| Free energy \(F\) | **−1.118469277632 Ha** |
| F₀ [Ha/Bohr] | **(−0.014404059091, −0.009867891977, +0.009866830655)** |
| F₁ [Ha/Bohr] | **(+0.21449155103, +0.10920852043, −0.10920745848)** |

Recorded 2026-08-01, Arrhenius x86_64, Release, `OMP_NUM_THREADS=4`–`8`.

Translational residual \|∑F\| is nonzero on this coarse grid (eggbox); the same
pattern appears in the CP2K H₂ twin below. **Gradient identity vs FD is the
product correctness gate**, not absolute Newton-3 on a coarse mesh.

### CP2K ENERGY_FORCE + SMEAR (algorithm twin)

| Field | Value |
| ----- | ----- |
| Engine | **CP2K 2026.1** (`cp2k.psmp`, git:5e54ba2) |
| Input (fair wall / H₂) | `tools/cp2k_oracle/inputs/gpw-metallic-forces/h2_kmesh_fd_forces.inp` |
| Input (Na metal class) | `tools/cp2k_oracle/inputs/gpw-metallic-forces/na_bcc_fd_forces.inp` |
| H₂ twin | DZVP-MOLOPT-GTH + GTH-PADE-q1, k 1×1×2, FD \(T_e=3157.75\) K, CUTOFF 100 |
| H₂ \(E\) example | −1.122793679 Ha (not ATV pin — basis differs) |
| H₂ F₀ / F₁ example | (−4.7e−3, −1.0e−2, +1.0e−2) / (+0.238, +0.127, −0.127) Ha/Bohr |
| Na twin | BCC \(a=4.225\) Å, Na2 +0.05 Å along x, DZVP-MOLOPT-SR-GTH + GTH-PBE-q9 |
| Na \(E\) / \(E_F\) | −93.509962456 Ha / 0.053784 Ha |
| Command | `source …/cp2k/install/env.sh; mpirun -np 1 cp2k.psmp -i …` |

## Tolerances (regression)

- \|F − pin\| ≤ 1e−5 Ha
- \|F_x − pin\| ≤ 1e−5 Ha/Bohr
- max\|F_an − F_fd\| ≤ 5e−5 Ha/Bohr
- Ne = 2 within 1e−5; finite \(E_F\); non-empty `f_k`

## Performance (fair wall)

Matched twin: H₂ ENERGY_FORCE + SMEAR, 1 MPI × OMP=4.

| Engine | Wall |
| ------ | ---- |
| Atomiverse SCF + analytic F (`reg` scf+F, OMP=8) | **~0.11 s** (scf 0.037 + F 0.069) |
| CP2K `h2_kmesh_fd_forces.inp` (OMP=4) | **1.43 s** |

\[
r = T_{\mathrm{atv}} / T_{\mathrm{cp2k}} \approx 0.077 \ll 10
\]

`within_10x = true`. Na ENERGY_FORCE twin (`na_bcc_fd_forces.inp`) wall
~0.7 s (hot) documents the metallic-class algorithm path; absolute Na forces
are not dual-gated (eggbox + basis/grid mismatch vs ATV).

## Payload / API notes

- `GpwResult::f_k` / `f_k_beta` — per-k fractional occupations after smear SCF
- `copy_gpw` in force path propagates smear members
- UKS store path also fills fractional `f_k` / `P_k` (product pin:
  `gpw-uks-metallic-forces`)
