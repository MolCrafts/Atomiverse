# Spec — periodic-wannier-gpw-uks

## Goal

Productize multi-k Marzari–Vanderbilt MLWF on **collinear UKS GPW**
(`cpu::gpw::PeriodicWannier::compute` must not throw on `is_uks()`).
Localize α and β occupied manifolds independently on multi-k Bloch orbitals
from `GpwResult::C_occ_k` / `C_occ_k_beta`.

## Capability

| Item | Value |
|------|-------|
| Method | GPW collinear UKS multi-k SCF + MV MLWF |
| Twin | CP2K 2026.1 Quickstep LSD + LOCALIZE / WANNIER_CENTERS |
| System | Diamond Si BCT 4-atom cell (same as `periodic-wannier-gpw`) |
| Occupations | M=0 closed-shell collinear: nα = nβ = 8 |
| Pin | Bond-midpoint geometry **per spin** + SCF energy vs offline CP2K |

## Public API

```cpp
cpu::gpw::GpwDft dft;
dft.n_alpha = 8; dft.n_beta = 8; // UKS path; n_occ ignored
// cell, cutoff, xc, build(atoms) …
cpu::gpw::PeriodicWannier mlwf;
auto r = mlwf.compute(dft, atoms);
// r.n_wann_up / n_wann_dn; r.centers = α then β; r.spin[i] ∈ {0,1}
```

No public `*Config`; no factories. RKS path unchanged.

## Acceptance

1. Unit: UKS no longer throws the foothold message; metallic/smear still reject.
2. Regression: SCF converges; n_up = n_dn = 8, n_wann = 16; each spin’s 8
   centers within 0.90 Å of BCT bond midpoints; Ω_final < 0.5 Ω_init.
3. Offline CP2K twin: energy parity within stated tol; fair wall
   T_atv / T_cp2k ≤ 10 on the same twin job.

## Out of scope

Disentanglement, metallic smear MLWF, open-shell (M≠0) geometry pin,
GPW Berry, CUDA.
