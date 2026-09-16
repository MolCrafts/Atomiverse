# Reference — multi-k GPW analytic ionic forces with Dudarev DFT+U

## Method

The Dudarev occupation \(n^\sigma=\sum_k w_k\,\tfrac12\mathrm{Herm}[(P_k^\sigma
S_k)+(S_k P_k^\sigma)]\) restricted to the correlated (atom, ℓ) block depends
on geometry through \(S_k\), so DFT+U contributes an **occupation-matrix force
term**

\[
dE_U/dR=\sum_\sigma\sum_k w_k\,\mathrm{Re\,Tr}\big[M_k^\sigma\,dS_k/dR\big],
\qquad M_k^\sigma=\tfrac12\big(\hat V^\sigma P_k^\sigma+P_k^\sigma\hat V^\sigma\big),
\]

folded as \(W_k\leftarrow W_k-M_k\) into the existing Pulay \(W\!\cdot\!dS_k\)
contraction (Γ and multi-k alike; `internal/dft_plus_u.h`,
`subtract_dftu_pulay[_k]`). This mirrors CP2K's own device for its
geometry-dependent Mulliken-charges +U (`dft_plus_u.F`: `matrix_w`
correction \(w\mathrel{+}= -\tfrac12(dE/dq_i+dE/dq_j)P_{ij}\)).

**Variational scatter fix (2026-07-31, this delivery):** the exact
\(\partial E_U/\partial P\) for this occupation is
\(\tfrac12(\hat VS+S\hat V)\) — the previous bare \(\hat V\) scatter left the
SCF fixed point non-stationary for the reported \(E_U[n(P,S)]\), which made
analytic and multi-SCF FD forces disagree at O(1) Ha/Bohr on the toy (and was
the ~9×10⁻⁴ Ha/Bohr³ residual in the +U stress dual gates). CP2K applies the
same S-weighted form (`h_block += ½(dEdq_i+dEdq_j)·S_ij`). All +U SCF fixed
points moved; the sibling +U regressions were re-pinned in the same change
(each pin cites its pre-fix value inline).

## Regression system A — multi-k dual gate (decisive)

| Item | Value |
|------|--------|
| Molecule | H₂ high-spin (Nα=2, Nβ=0), compressed bond 1.0 Bohr along x |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | cubic 12 Bohr, periodic |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1, PADE LSD, 100 Ry, `local_pp_on_grid` |
| k mesh | MONKHORST-PACK 1×1×2, TR-folded |
| DFT+U | H L=0, **U−J = 2.0 eV** |

U−J = 2 eV deliberately: at 4 eV this DZV s-shell toy sits in the
strong-overlap regime where the gross-Mulliken occupation is unphysical
(CP2K's own `PLUS_U` warns "unphysical Mulliken charges" there, see
`gpw-dftu-kpoints`) and the displaced multi-SCF FD oracle hops between SCF
branches — no dual gate can hold on a multi-valued surface. At 2 eV the SCF
is single-branch.

### Measured (first green 2026-07-31, Release, OMP=8)

| Gate | Value | Tol |
|------|-------|-----|
| A1 E(k-UKS+U 1×1×2, 100 Ry) | −0.611940650579 Ha (e_dftu −0.10209345) | payload sanity |
| A2 max\|F_an − F_fd\| (full 6N `gpw_forces_fd`, h=1e-3) | **6.4×10⁻⁶ Ha/Bohr** | 5×10⁻⁵ |
| A3 \|ΣF\| / Newton-3 | 1.1×10⁻¹⁴ | 1×10⁻⁶ |
| A4 max\|F(U)−F(U=0)\| | 5.4×10⁻² Ha/Bohr | ≥ 1×10⁻⁴ (U term live) |
| A5 max\|F_kΓ − F_Γ\| (explicit k=Γ ≡ real Γ fold) | 6.0×10⁻⁸ Ha/Bohr | 5×10⁻⁶ |

Pre-fold control: without the W-fold the same A2 gate reads ~1.6 Ha/Bohr
(wrong sign) — the occupation-matrix term is decisive, four orders above the
gate residual.

## Regression system B — CP2K ENERGY_FORCE pin (NiO d shell)

Periodic Γ NiO stretched molecule: 10 Å cubic box (PBC), Ni (5,5,4.1) Å,
O (5,5,5.9) Å (R = 1.8 Å), SZV-GTH-PADE(-q18) / GTH-PADE-q18/q6, PADE LSD,
Nα=13 / Nβ=11, PLUS_U Mulliken Ni d U−J = 6 eV.

### Offline CP2K pin

- **Engine:** CP2K **2026.1** (`cp2k.psmp`, local install
  `/nobackup/.../work/cp2k/install`), OMP=8.
- **Input:** `tools/cp2k_oracle/inputs/gpw-dftu-kpoint-forces/nio_pbc_gamma_ot_force.inp`
  (OT FULL_SINGLE_INVERSE + OUTER_SCF, multigrid 250 Ry / REL 40 / NGRIDS 4,
  `RUN_TYPE ENERGY_FORCE`).
- **Date:** 2026-07-31. Wall **12.84 s**.
- **Output lines:**

```
  *** SCF run converged in    14 steps ***
  DFT+U energy:                                                 0.04952309241684
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]           -184.929424563910374
 FORCES|      1  3.17871282E-07 -1.91561775E-06 -8.35837089E-03   8.35837111E-03
 FORCES|      2 -3.55176272E-08  2.19071254E-06 -7.27092413E-03   7.27092447E-03
 FORCES| Sum     2.82353655E-07  2.75094796E-07 -1.56292950E-02
```

(ΣF_z = −1.56×10⁻² Ha/Bohr is the multigrid eggbox background common to both
atoms at 250 Ry with the hard Ni q18 PP.)

### Asserted vs diagnostic

| Check | Status | Value |
|-------|--------|-------|
| \|E_ATV − E_CP2K\| | **asserted** ≤ 5×10⁻² Ha | 1.18×10⁻² (single grid vs multigrid+OT) |
| transverse F ≈ 0 | **asserted** < 5×10⁻⁴ | ~1×10⁻⁸ |
| ΔF_z = F_z(Ni)−F_z(O) vs CP2K | **diagnostic** | ATV +8.97×10⁻², CP2K −1.09×10⁻³ |

### Offline Γ d-shell FD validation (2026-07-31, not in ctest budget)

NiO +U DIIS needs `tol = 1e-8` (~480 iters/SCF, minutes) before FD noise
drops below the force signal — at the in-budget tol 1e-6 the ± states are
only consistent to ~1e-5 Ha and the 2h=4×10⁻³ divisor amplifies that to
~1e-2 Ha/Bohr of FD noise.

| Quantity (tol 1e-8, h = 2×10⁻³ Bohr, Ni z) | Value [Ha/Bohr] |
|---|---|
| analytic F_z(Ni) | +3.76808342×10⁻² |
| central FD | +3.80477507×10⁻² |
| \|analytic − FD\| | **3.7×10⁻⁴** |

(h = 4×10⁻³ left the −h SCF unconverged at 600 iters — reported for
completeness, not used.) The in-regression decisive force gate is the H₂
full-6N multi-k dual gate A2 (6.4×10⁻⁶ Ha/Bohr).

**Projector-model divergence (surfaced, out of scope here):** CP2K
`PLUS_U_METHOD MULLIKEN` projects on the on-site **net** AO block
\(q_{ab}=P_{ab}S_{ab}\) (elementwise; geometry-independent ⇒ **no**
occupation force term), while ATV projects on the symmetrized **gross**
product \(\tfrac12(PS+SP)\big|_\mathrm{block}\) (geometry-dependent ⇒ the
large ΔF_z above is the model's occupation force, confirmed against ATV's own
FD, not a bug). Aligning ATV's projector with CP2K's (or adding it as a
second method) is a census decision — it rewrites every +U SCF fixed point.

### Multi-k CP2K twin attempts (documented)

- `nio_pbc_kpoint_force.inp` (MP 1×1×2, diag + Broyden, ATOMIC guess,
  MAX_SCF 300): oscillates, `SCF run NOT converged` (closest approach 1.9×10⁻⁴
  at step 238). Restart from its own `.kp` (ALPHA 0.05, MAX_SCF 600): still
  oscillating — the degenerate minority-spin d manifold of the linear molecule
  defeats fixed-multiplicity diagonalization (OT is not available with
  k-points in CP2K). Same pathology class as documented for
  `gpw-dftu-kpoints` / `gpw-dftu-kpoint-stress`. Multi-k force correctness is
  therefore gated by the internal A2 full-6N FD dual gate + the A5 Γ-limit
  identity, with the Γ d-shell CP2K pin as the external anchor.

## Performance

| Item | Value |
|------|-------|
| ATV regression total (A1–A5 incl. full-6N FD + B, OMP=8, Release) | **38.6 s** |
| ATV B section alone (NiO SCF + analytic forces, 250 Ry single grid) | ≈ 13 s |
| CP2K ENERGY_FORCE twin (same system, multigrid, OMP=8) | 12.84 s |
| ratio T_atv/T_cp2k (comparable single-point + forces) | **≈ 1.0** (≤ 10 ✓) |

## Citations

- S. L. Dudarev et al., *Phys. Rev. B* **57**, 1505 (1998) — U−J functional.
- VandeVondele, Krack, Mohamed, Parrinello, Chassaing, Hutter, *Comput. Phys.
  Commun.* **167**, 103 (2005) — Quickstep GPW forces.
- CP2K 2026.1 `src/dft_plus_u.F` — Mulliken / Mulliken-charges +U forms
  (S-weighted KS contribution and `matrix_w` correction).
