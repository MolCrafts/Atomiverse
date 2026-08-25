# Acceptance — ipea-xtb-vip

Status: **met** (dual gate green 2026-08-25).

## Correctness

- [x] `IPEA::compute_vip` / `compute_vea` run ΔSCC at charge ±1 and store
      \(\mathrm{IP}_v\) / \(\mathrm{EA}_v\) [eV] including `kIpeaShiftEv`.
- [x] `kEnergy` remains the N-electron total; `last_vip()` / `last_vea()`
      are the property accessors.
- [x] Closed-shell energy / analytic \(G\) / FD Hessian paths unchanged
      (integer occupation; existing pins unmoved; CH4 energy 8.90 meV).
- [x] Unit `test_cpu_xtb_ipea`: finite VIP/VEA; flags default off;
      `kIpeaShiftEv ≈ 4.8455`.
- [x] Regression `reg_ipea-xtb-vip`: tetrahedral CH4 vs archived xtb 6.7.1
      IPEA `--vipea` (IP 13.0088 eV, EA −1.4559 eV; Δ 0.103 / 0.125 eV;
      ionized kernel residual, occupations \(2+1.6667\times 3\) / \(8+1\)).
- [x] `REFERENCE.md` states `oracle: xtb` and `oracle_version: 6.7.1`.

## Speed

- [x] \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} = 2.0 \le 10\) (1 thread;
      ATV 0.01 s, xtb `--vipea` clock 0.005 s). Not `perf_exempt`.

## Scope-out (not this slug)

- Extra-Z tables (`ipea-beyond-chno`)
- Polar-water ~0.5 eV energy hole vs xtb
- UHF / `spin-gfn-xtb`
- GFN `--etemp` product (`gfn-xtb-etemp`)
- ALPB, PBC, sTDA, `libxtb`
