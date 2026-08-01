# Spec — `gpw-metallic-forces`

## Goal

Multi-k RKS **free-energy analytic ionic forces** under Fermi–Dirac /
Methfessel–Paxton metallic occupation smearing on the GPW product path.

## Problem

After a fractional-occupation SCF, `store_k_payload` rebuilt integer
\(n_{\mathrm{occ}}\) `C_occ_k` / `P_k`, and force assembly rebuilt
\(P_k,W_k\) with uniform occupancy 2. Free-energy forces \(-\partial F/\partial R\)
were therefore wrong for metals.

## Design

1. **Payload:** `GpwResult::f_k` (and UKS `f_k_beta`) hold per-k band
   occupations after SCF. Under smear, `store_k_payload` fills \(f_k\) from the
   shared-\(\mu\) spectrum and rebuilds \(P_k=\sum_b f_b\,c_b c_b^\dagger\).
2. **Forces:** `assemble_gpw_forces_kpoints` uses full `C_k` + fractional `f_k`
   when `cfg.smearing != none`:
   \(W_k=\sum_b f_b\varepsilon_b\,c_b c_b^\dagger\). Fixed-occupation path
   unchanged (`C_occ_k`, occ 2 / 1).
3. **`copy_gpw` in forces** propagates `smearing` / `degauss` / `mp_order` /
   `extra_bands` so `gpw_forces_analytic` runs the free-energy surface.
4. **FD oracle:** `gpw_forces_fd` already differentiates `e_total` (= Mermin
   \(F\) under smear).

## Public API

No new types. Members already on `GpwDft` (`smearing`, `degauss`, …) and
`gpw_forces_analytic` / `gpw_forces_from_scf`. New result fields: `f_k`,
`f_k_beta`.

## Acceptance

| # | Check |
| - | ----- |
| A1 | Unit: multi-k H₂ + FD smear, max\|F_an−F_fd\| < 1e−4 Ha/Bohr |
| A2 | Regression BCC Na (broken-symmetry), free energy + forces finite, Newton-3 |
| A3 | Offline CP2K ENERGY_FORCE + SMEAR twin run; fair wall \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\le 10\) |
| A4 | Fixed-occupation k-force path bit-compatible (no regression in unit H₂ k-mesh) |

## Out of scope

- Metallic **stress** (`gpw-metallic-stress`)
- UKS metallic forces product pin (`gpw-uks-metallic-forces`) — payload fixed here
- BOMD with smear (`gpw-bomd-smearing`)
