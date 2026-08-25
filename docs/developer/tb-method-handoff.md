# TB / SE method loop — handoff

Last assign: **2026-08-25 wave 13** (this-shift wave 7, `theme=gfn`) on
parent `dev` @ `4ff28fab`.
Last GapCensus: **2026-08-25 wave 13** (`4ff28fab` = census commit).
Last harness: **wave 13** — `ipea-xtb-vip` merged
(`0a7e8b2a` / `5a140845`). Keep `gfn2-camm-r` `[~]`.
**incomplete_remaining=1. pending=31. blocked=5.**
**Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Harness wave 13 (this-shift wave 7, 2026-08-25)

| slug | Result |
| ---- | ------ |
| `ipea-xtb-vip` | **merged → done** — IPEA ΔSCC vertical IP/EA + `ipeashift` 4.8455 eV (`compute_vip` / `compute_vea`, degenerate-shell occ for Td t2); CH4 vs xtb 6.7.1 IPEA `--vipea` (`reg_ipea-xtb-vip`; IP **13.0088** eV / EA **−1.4559** eV, Δ 0.103 / 0.125 eV ionized kernel, \(T_{\mathrm{atv}}/T_{\mathrm{xtb}}=2.0\)). Twin `tools/tb_oracle/inputs/xtb/ch4_ipea_vip/` |
| `gfn2-camm-r` | **keep `[~]`** — unmerged `wf/gfn2-camm-r` @ `ad050eb7` (ahead=2 behind=2). Tip: two-centre Θ on H in Fock+G after frozen-P dΘ/dR; angular \(F_{H1,y}\) still **not** xtb `mmompop`. Spec lives on the tip |

Specs closed (deleted: `ipea-xtb-vip`).
Isolation worktrees: **none** (Clearup already ran).
Local TB tip: `wf/gfn2-camm-r` (incomplete — do **not** `-D`).

Roadmap ○→✓ **IPEA-xTB VIP/VEA** (CHNO ΔSCC IP/EA + ipeashift). Extra-Z
IPEA tables stay ○. CAMM-vs-\(R\) stays ○ (`[~]`). Foothold ≠ closed:
ALPB, CPCM-X, PBC, spin, sTDA, beyond-CHNO, extra-Z IPEA, `--etemp`,
CAMM-vs-\(R\).

## Start here next wave

**incomplete = 1** — Recover `gfn2-camm-r` (`wf/gfn2-camm-r` @
`ad050eb7`, ahead=2 behind=2) before new pending. Rebase onto current
`dev` @ `0a7e8b2a`. Spec lives on the tip. Without `theme=gfn`, ladder
returns to Phase B NDDO (`pm3-organic-energy`,
`nddo-closed-form-forces`) and DFTB0 analytic F / DFTB3/ThreeOb deepens
(`dftb2-threeob-periodic` before `dftb2-threeob-multi-k`). With
`theme=gfn`: wait for CAMM-R before ALPB / beyond-CHNO / sTDA / etemp
(same `gfn2.cpp`); IPEA extra-Z may staff on `ipea.cpp` (VIP landed). Do
**not** treat GFN CHNO energy + analytic G + family FD Hessian +
AES-in-Fock + IPEA E+G+H+VIP as the GFN product closed.

## Oracle / build (binding)

| Knob | Value |
| ---- | ----- |
| Family oracles | NDDO → **OpenMOPAC**; DFTB* → **DFTB+**; GFN-xTB → **xtb** |
| Forbidden | OpenMOPAC as DFTB pin; DFTB+ as PM6/PM7 pin; GFN as DFTB3 successor |
| Offline only | `tools/tb_oracle/`; never live oracle in ctest |
| SE-only cmake | SEMIEMPIRICAL+EVALUATE+TESTS+REGRESSIONS **ON**; HF/PW/GPW/MD/MC/MINIMIZER/VIB/PATH/KSPACE/POTENTIALS/NEIGHBOR/CUDA/**CLI** **OFF** |
| Hessian exception | landed Hessian slugs (`se-analytic-hessian`, `gfn0-xtb-analytic-hessian`, `gfn-xtb-analytic-hessian`, `ipea-xtb-analytic-hessian`) may add `-DATV_BUILD_VIB=ON` only; no staffed Hessian remaining |
| Domain suite (merge gate) | `ctest -R '^test_cpu_(am1\|rm1\|dftb0\|dftb2\|dftb3\|td\|xtb_gfn0\|xtb_gfn1\|xtb_ipea\|gfn2\|nddo_\|scc_)'` |
| Worker build | scoped `--target atomiverse <test_*> reg_<slug>` only |

**No oracle binary** on this host (`mopac` / `dftb+` / `xtb` absent).

## Next

1. Recover `gfn2-camm-r` (rebase onto current `dev` @ `0a7e8b2a`).
2. GapCensus owns refill. Do **not** staff from this harness.
3. **No push.** Wipe builds; Clearup worktrees same wave. `git worktree remove`
   of a tree with `molrs` needs `--force`.
