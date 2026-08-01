# Spec — `gpw-uks-metallic-forces`

## Goal

Multi-k collinear **UKS** free-energy **analytic ionic forces** under
Fermi–Dirac / Methfessel–Paxton metallic occupation smearing (Mermin
\(F = E_{\mathrm{KS}} - TS\)).

## Gap

Energy path product: `reg_gpw-uks-smearing`. Force assembly rebuilt
\(P_k/W_k\) from **integer** \(n_\alpha/n_\beta\) columns of `C_occ_k`
(`store_k_payload` always used fixed-occ density). Free-energy Hellmann–Feynman
requires fractional \(f_{\sigma kb}\).

## Acceptance

1. `store_k_payload` (RKS + UKS) writes `f_k` / `f_k_beta` and full
   spectrum `C_occ_k` when `smearing != none`.
2. `gpw_forces_analytic` / `gpw_forces_from_scf` rebuild
   \(P=\sum f_b c c^\dagger\), \(W=\sum f_b\varepsilon_b c c^\dagger\).
3. Unit: multi-k UKS+FD analytic vs multi-SCF FD of \(F\) (H₂ toy).
4. Regression: BCC Na UKS+FD free-energy pin + force residual + closed-shell
   identity; CP2K ENERGY_FORCE + SMEAR twin for fair wall \(r\le 10\).

## Out of scope

Metallic cell stress (`gpw-metallic-stress`); BOMD under smear
(`gpw-bomd-smearing`); cross-representation PW twins.
