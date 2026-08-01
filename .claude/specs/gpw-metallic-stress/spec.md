# Spec — gpw-metallic-stress

## Goal

Multi-k RKS **free-energy analytic cell stress** under Fermi–Dirac /
Methfessel–Paxton metallic occupation smearing on GPW (Mermin free energy
\(F = E_{\mathrm{KS}} - TS\)).

## Capability gap (pre-landing)

- Energy product under smear: `reg_gpw-metallic-smearing` ✓
- Analytic multi-k stress (fixed integer \(n_{\mathrm{occ}}\)): `reg_gpw-kpoint-stress` ✓
- Residual: `store_k_payload` rebuilt \(P_k\) with fixed occ=2; stress/forces
  rebuilt \(P_k/W_k\) from integer \(C_{\mathrm{occ},k}\); `copy_gpw` dropped
  smear knobs so multi-SCF FD stress lost SMEAR.

## Product path

1. SCF stores full `C_k`, `eps_k`, fractional `f_k` and free-energy \(P_k\).
2. `gpw_stress` / `gpw_stress_at_state` rebuild
   \(P_k=\sum_b f_b c_b c_b^\dagger\),
   \(W_k=\sum_b f_b\varepsilon_b c_b c_b^\dagger\) under smear.
3. Hellmann–Feynman fixed-\(\{f_k\}\) orthorhombic diagonal strain FD of the
   non-self-consistent Lagrangian (same shape as insulator multi-k stress).
4. Dual-gate vs multi-SCF strain-FD of \(F\) (`gpw_stress_fd`) and offline
   CP2K `STRESS_TENSOR NUMERICAL` + `SMEAR` twin (BCC Na).

## Acceptance

See `acceptance.md`.
