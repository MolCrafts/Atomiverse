# Reference — Γ-UKS GPW analytic ionic forces with Dudarev DFT+U

## Method

Dudarev occupation \(n^\sigma=\tfrac12\mathrm{Herm}[(P^\sigma S)+(S P^\sigma)]\)
restricted to the correlated (atom, ℓ) block depends on geometry through \(S\),
so DFT+U contributes an **occupation-matrix force term**

\[
dE_U/dR=\sum_\sigma\mathrm{Tr}\big[M^\sigma\,dS/dR\big],
\qquad M^\sigma=\tfrac12\big(\hat V^\sigma P^\sigma+P^\sigma\hat V^\sigma\big),
\]

folded as \(W\leftarrow W-M\) into the existing Pulay \(W\!\cdot\!dS\)
contraction (`internal/dft_plus_u.h`, `subtract_dftu_pulay` on the Γ path).

Product path already packs `cfg.plus_u` in `gpw_forces.cpp` (Γ and multi-k).
This slug is the **dedicated Γ ENERGY_FORCE + PLUS_U pin** (multi-k dual gate
lives in `gpw-dftu-kpoint-forces`).

## Regression system A — Γ dual gate (decisive force)

| Item | Value |
|------|--------|
| Molecule | H₂ high-spin (Nα=2, Nβ=0), compressed bond 1.0 Bohr along x |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | cubic 12 Bohr, periodic |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1, PADE LSD, 100 Ry, `local_pp_on_grid` |
| k | Γ only |
| DFT+U | H L=0, **U−J = 2.0 eV** |

U−J = 2 eV deliberately (same as multi-k force pin): at 4 eV this DZV s-shell
toy is multi-branch under gross-Mulliken; see
`.claude/notes/gpw-dftu-projector-divergence.md`.

### Measured (first green 2026-08-01, Release, OMP=8)

| Gate | Value | Tol |
|------|-------|-----|
| A1 E(Γ-UKS+U, 100 Ry) | −0.611955472155 Ha (e_dftu −0.10209039) | payload sanity |
| A2 max\|F_an − F_fd\| (full 6N `gpw_forces_fd`, h=1e-3) | **6.9×10⁻⁶ Ha/Bohr** | 5×10⁻⁵ |
| A3 \|ΣF\| / Newton-3 | 2.2×10⁻¹³ | 1×10⁻⁶ |
| A4 max\|F(U)−F(U=0)\| | 5.4×10⁻² Ha/Bohr | ≥ 1×10⁻⁴ |

## Regression system B — CP2K ENERGY_FORCE pin (NiO d shell)

Periodic Γ NiO stretched molecule: 10 Å cubic box (PBC), Ni (5,5,4.1) Å,
O (5,5,5.9) Å (R = 1.8 Å), SZV-GTH-PADE(-q18) / GTH-PADE-q18/q6, PADE LSD,
Nα=13 / Nβ=11, PLUS_U Mulliken Ni d U−J = 6 eV.

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, git:**5e54ba2**), local install
  `/nobackup/.../work/cp2k/install`, OMP=8, `mpirun -np 1`.
- **Input:** `tools/cp2k_oracle/inputs/gpw-dftu-forces/nio_pbc_gamma_ot_force.inp`
  (OT FULL_SINGLE_INVERSE + OUTER_SCF, multigrid 250 Ry / REL 40 / NGRIDS 4,
  `RUN_TYPE ENERGY_FORCE`).
- **Date:** 2026-08-01. Wall **13.24 s** (`/usr/bin/time`, arrhenius1).
- **Host run dir:** `/tmp/atv-gpw-dftu-forces` (symlink BASIS_SET +
  GTH_POTENTIALS from `CP2K_DATA_DIR`).
- **Output lines:**

```
  *** SCF run converged in    14 steps ***
  DFT+U energy:                                                 0.04952309241681
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.929424563910430
 FORCES|      1  3.17871301E-07 -1.91561767E-06 -8.35837089E-03   8.35837111E-03
 FORCES|      2 -3.55176558E-08  2.19071246E-06 -7.27092413E-03   7.27092447E-03
 FORCES| Sum     2.82353646E-07  2.75094789E-07 -1.56292950E-02
```

### Asserted vs diagnostic

| Check | Status | Value / tol |
|-------|--------|-------------|
| \|E_ATV − E_CP2K\| | **asserted** ≤ 5×10⁻² Ha | **1.18×10⁻²** (single grid vs multigrid+OT) |
| transverse F ≈ 0 | **asserted** < 5×10⁻⁴ | ~1×10⁻⁸ |
| ΔF_z = F_z(Ni)−F_z(O) | **diagnostic** | ATV +8.97×10⁻², CP2K −1.09×10⁻³ |

**Projector-model divergence (not a bug):** CP2K `PLUS_U_METHOD MULLIKEN`
projects on the on-site **net** AO block \(q_{ab}=P_{ab}S_{ab}\)
(geometry-independent ⇒ no occupation force), while ATV projects on the
symmetrized **gross** product \(\tfrac12(PS+SP)|_\mathrm{block}\). Detail:
`.claude/notes/gpw-dftu-projector-divergence.md`. Decisive force correctness
is the A2 full-6N FD dual gate.

## Performance

| Item | Value |
|------|-------|
| CP2K ENERGY_FORCE twin (NiO, OMP=8, 1 MPI) | **13.24 s** |
| ATV NiO SCF + analytic forces (section B, OMP=8, Release) | **32.88 s** |
| ratio T_atv / T_cp2k | **2.483** (≤ 10 ✓) |
| ATV regression total (A1–A4 + B, OMP=8) | **37.6 s** |

Fair wall is the **NiO SCF + force assembly only** (same twin job class as
CP2K ENERGY_FORCE), not the multi-gate regression total (which includes
full-6N FD).

## Citations

- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — U−J functional.
- VandeVondele, Krack, Mohamed, Parrinello, Chassaing, Hutter, *Comput. Phys.
  Commun.* **167**, 103 (2005) — Quickstep GPW forces.
- CP2K 2026.1 `src/dft_plus_u.F` — Mulliken / Mulliken-charges +U forms.
