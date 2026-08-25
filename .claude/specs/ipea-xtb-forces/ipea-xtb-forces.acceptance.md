# Acceptance — ipea-xtb-forces

Status: **met** (dual gate green 2026-08-25).

## Correctness

- [x] `IPEA::with_forces = true` writes analytic \(F=-\nabla E\) [eV/Å]; no throw.
- [x] `provides_force()` is true (kind default).
- [x] Unit `test_cpu_xtb_ipea`: analytic vs energy FD on off-eq H₂O
      (\(h=10^{-4}\) Å, \(O(h^2)\)); net force ~0.
- [x] Regression `reg_ipea-xtb-forces`: off-eq CH4 vs archived xtb 6.7.1
      IPEA `--grad` (`tools/tb_oracle/inputs/xtb/ch4_ipea_offeq_grad/`).
- [x] `REFERENCE.md` states `oracle: xtb` and `oracle_version: 6.7.1`.
- [x] `reg_ipea-xtb-small` energy pin unmoved (8.90 meV).

## Speed

- [x] \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} \approx 0.26 \le 10\) (1 thread).

## Scope-out (not this slug)

- `--vip` / `--vea` ΔSCC IP/EA (`ipea-xtb-vip`)
- PBC, open shell, ALPB, elements beyond CHNO
- IPEA Hessian (waits on this F)
- Polar-water ~0.5 eV energy hole vs xtb (energy-path kernel; FD-gated)
