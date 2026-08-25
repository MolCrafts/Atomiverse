# Acceptance — ipea-xtb-analytic-hessian

Status: **met** (dual gate green 2026-08-25).

## Correctness

- [x] `cpu::vib::Hessian` `build()` accepts `IPEA{with_forces=true}` as a
      Semiempirical force kind (unit `test_cpu_xtb_ipea_hessian`; no 6N
      `run`).
- [x] Regression `reg_ipea-xtb-analytic-hessian`: tetrahedral CH4 vs archived
      xtb 6.7.1 IPEA `--hess` (`tools/tb_oracle/inputs/xtb/ch4_ipea_hess/`).
- [x] `REFERENCE.md` states `oracle: xtb` and `oracle_version: 6.7.1`.
- [x] Hessian 15×15, finite, symmetric; nonlinear rotor; 9 vibrational
      wavenumbers, \(n_{\mathrm{imag}}=0\).
- [x] Polar water is **not** the pin (0.5 eV kernel hole). `--vip`/`--vea`
      not this slug.

## Speed

- [x] \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \approx 0.061 \le 10\) (1 thread;
      ATV 10.4 ms vs xtb printed 0.170 s).

## Scope-out (not this slug)

- `--vip` / `--vea` ΔSCC IP/EA (`ipea-xtb-vip`)
- PBC, open shell, ALPB, elements beyond CHNO
- Closed-form \(\partial^2 E/\partial R_i\partial R_j\)
- Polar-water ~0.5 eV energy hole vs xtb (energy-path kernel)
