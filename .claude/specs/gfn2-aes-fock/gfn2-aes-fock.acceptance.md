---
title: GFN2-xTB AES in the SCC Fock — acceptance
slug: gfn2-aes-fock
oracle: xtb
---

# Acceptance

Dual gate. Offline pin only. SE-only cmake.

## Unit (`test_cpu_gfn2_params`)

- Analytic \(F\) vs central FD of the same total energy on off-eq H₂O
  (\(h=10^{-4}\) Å); AES-in-Fock residual \(\sim 0.19\) eV/Å on stretch
  (quadrupole \(\partial Q/\partial R\)).
- Net force \(\sum_i F_i = 0\) at round-off.

## Regression (`reg_gfn2-aes-fock`)

Offline xtb 6.7.1 `--gfn 2 --chrg 0 --uhf 0 --acc 1.0 --grad` on
`tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/mol.xyz`.

| Quantity | Pin | Tol |
| -------- | --- | --- |
| \(E\) | **−5.058625058504** Eh | \(1\times10^{-2}\) eV |
| \(F_{O,x}\) (O–H stretch) | from archived `gradient.txt` | \(0.35\) eV/Å |
| \(F_{H1,y}\) (angular) | from archived `gradient.txt` | printed; CAMM-vs-\(R\) hole |
| \(\|\sum_i F_i\|\) | 0 | \(1\times10^{-10}\) eV/Å |
| \(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\) | — | \(\le 10\) or `perf_exempt` |

`REFERENCE.md` states `oracle: xtb` and `oracle_version: 6.7.1`.
