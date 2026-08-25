# TB / SE method loop — handoff

Last assign: **2026-08-25 wave 12** (this-shift wave 6, `theme=gfn`) on
parent `dev` @ `d0c49de0`.
Last GapCensus: **2026-08-25 wave 12** (`d0c49de0` = census commit).
Last harness: **wave 12** — `ipea-xtb-analytic-hessian` merged
(`c7db3d8f` / `f5e7c78a`). Keep `gfn2-camm-r` `[~]`.
**incomplete_remaining=1. pending=29. blocked=5.**
**Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Harness wave 12 (this-shift wave 6, 2026-08-25)

| slug | Result |
| ---- | ------ |
| `ipea-xtb-analytic-hessian` | **merged → done** — IPEA FD Hessian via `cpu::vib::Hessian` over shipped analytic G; CH4 vs xtb 6.7.1 IPEA `--hess` (`reg_ipea-xtb-analytic-hessian`; ω **1541.68×3 / 1705.89×2 / 3548.78 / 3576.13×3** cm⁻¹, Δω ≤ 1.30 cm⁻¹, \(E\) −4.670476737294 Eh / 8.90 meV kernel, \(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.061\)). Twin `tools/tb_oracle/inputs/xtb/ch4_ipea_hess/` |
| `gfn2-camm-r` | **keep `[~]`** — unmerged `wf/gfn2-camm-r` @ `4555d0de` (ahead=1 behind=2). Tip: frozen-P quadrupole dΘ/dR on O-centre CAMM; angular \(F_{H1,y}\) still **+0.11 vs xtb −0.80** eV/Å (naive two-centre Θ on H is not xtb `mmompop`). Spec lives on the tip |

Specs closed (deleted: `ipea-xtb-analytic-hessian`).
Isolation worktrees: **none** (Clearup already ran).
Local TB tip: `wf/gfn2-camm-r` (incomplete — do **not** `-D`).

Roadmap ○→✓ **IPEA-xTB analytic Hessian** (CHNO FD \(H\) via
`cpu::vib::Hessian`). IPEA VIP/VEA stays ○. CAMM-vs-\(R\) stays ○
(`[~]`). Foothold ≠ closed: ALPB, CPCM-X, PBC, spin, sTDA, beyond-CHNO,
IPEA `--vip`/`--vea`, CAMM-vs-\(R\).

## Start here next wave

**incomplete = 1** — Recover `gfn2-camm-r` (`wf/gfn2-camm-r` @
`4555d0de`, ahead=1 behind=2) before new pending. Rebase onto current
`dev` @ `c7db3d8f`. Spec lives on the tip. Without `theme=gfn`, ladder
returns to Phase B NDDO (`pm3-organic-energy`,
`nddo-closed-form-forces`) and DFTB3/ThreeOb deepens
(`dftb2-threeob-periodic` before `dftb2-threeob-multi-k`). With
`theme=gfn`: wait for CAMM-R before ALPB / beyond-CHNO / sTDA (same
`gfn2.cpp`); IPEA VIP may staff on `ipea.cpp` (Hessian landed). Do
**not** treat GFN CHNO energy + analytic G + family FD Hessian +
AES-in-Fock + IPEA E+G+H as the GFN product closed.

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

1. Recover `gfn2-camm-r` (rebase onto current `dev` @ `c7db3d8f`).
2. GapCensus owns refill. Do **not** staff from this harness.
3. **No push.** Wipe builds; Clearup worktrees same wave. `git worktree remove`
   of a tree with `molrs` needs `--force`.
