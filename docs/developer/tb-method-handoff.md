# TB / SE method loop — handoff

Last assign: **2026-08-25 wave 10** (this-shift wave 4, `theme=gfn`) on
parent `dev` @ `2a557824`.
Last GapCensus: **2026-08-25 wave 10** (refresh at assign HEAD).
Last harness: **wave 10** — `ipea-xtb-small` + `gfn-xtb-analytic-hessian`
merged (`50066df7` / `46c26afa`). Keep `gfn2-aes-fock` `[~]`.
**incomplete_remaining=1. pending=28. blocked=5.**
**Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Harness wave 10 (this-shift wave 4, 2026-08-25)

| slug | Result |
| ---- | ------ |
| `ipea-xtb-small` | **merged → done** — IPEA-xTB energy foothold (`cpu::semiempirical::IPEA`, extra 3s on C/N/O); CH4 vs xtb 6.7.1 IPEA-xTB (`reg_ipea-xtb-small`; \(E\) **−4.670476737294 Eh**, residual 8.90 meV, \(T_{\mathrm{atv}}/T_{\mathrm{xtb}}=2.0\)); `with_forces` throws; twin `tools/tb_oracle/inputs/xtb/ch4_ipea/` |
| `gfn-xtb-analytic-hessian` | **merged → done** — GFN1/2 FD Hessian via `cpu::vib::Hessian`; water ω vs xtb 6.7.1 `--gfn 1 --hess` (`reg_gfn-xtb-analytic-hessian`; GFN1 1474.96 / 3657.78 / 3761.56 cm⁻¹); GFN2 ω carries AES-in-Fock hole (Δω 121 cm⁻¹); twins `tools/tb_oracle/inputs/xtb/h2o_gfn{1,2}_hess/` |
| `gfn2-aes-fock` | **keep `[~]`** — unmerged `wf/gfn2-aes-fock` @ `0069724c` (ahead=1 behind=6); spec lives on the tip |

Specs closed (deleted: `ipea-xtb-small`, `gfn-xtb-analytic-hessian`).
Isolation worktrees: **none** (Clearup already ran).
Local TB tip: `wf/gfn2-aes-fock` (incomplete — do **not** `-D`).

Roadmap ○→✓ **IPEA-xTB** (CHNO energy foothold) + **GFN-xTB analytic Hessian**
(GFN0/1/2 FD \(H\) via `cpu::vib::Hessian`). AES-in-Fock stays ○.
Foothold ≠ closed: ALPB, CPCM-X, PBC, spin, sTDA, beyond-CHNO, GFN2
AES-in-Fock, IPEA forces / `--vip`.

## Start here next wave

**Recover `gfn2-aes-fock` first** (`incomplete ≫ pending`). Rebase
`wf/gfn2-aes-fock` onto current `dev` before gate. Skip same-file GFN2
dependents (`alpb-xtb`, `gfn2-beyond-chno`, sTDA) until AES-in-Fock
lands. Without `theme=gfn`, ladder returns to Phase B NDDO
(`pm3-organic-energy`, `nddo-closed-form-forces`) and DFTB3/ThreeOb
deepens (`dftb2-threeob-periodic` before `dftb2-threeob-multi-k`).

Do **not** treat GFN CHNO energy + analytic G + family FD Hessian + IPEA
energy as the GFN product closed.

## Oracle / build (binding)

| Knob | Value |
| ---- | ----- |
| Family oracles | NDDO → **OpenMOPAC**; DFTB* → **DFTB+**; GFN-xTB → **xtb** |
| Forbidden | OpenMOPAC as DFTB pin; DFTB+ as PM6/PM7 pin; GFN as DFTB3 successor |
| Offline only | `tools/tb_oracle/`; never live oracle in ctest |
| SE-only cmake | SEMIEMPIRICAL+EVALUATE+TESTS+REGRESSIONS **ON**; HF/PW/GPW/MD/MC/MINIMIZER/VIB/PATH/KSPACE/POTENTIALS/NEIGHBOR/CUDA/**CLI** **OFF** |
| Hessian exception | landed Hessian slugs (`se-analytic-hessian`, `gfn0-xtb-analytic-hessian`, `gfn-xtb-analytic-hessian`) may add `-DATV_BUILD_VIB=ON` only; no staffed Hessian remaining |
| Domain suite (merge gate) | `ctest -R '^test_cpu_(am1\|rm1\|dftb0\|dftb2\|dftb3\|td\|xtb_gfn0\|xtb_gfn1\|gfn2\|nddo_\|scc_)'` |
| Worker build | scoped `--target atomiverse <test_*> reg_<slug>` only |

**No oracle binary** on this host (`mopac` / `dftb+` / `xtb` absent).

## Next

1. Recover/merge `gfn2-aes-fock` (rebase onto current `dev` @ `46c26afa`).
2. Spec-first workers on `wf/<slug>`; SE-only cmake **including CLI=OFF**.
3. **No push.** Wipe builds; Clearup worktrees same wave. `git worktree remove`
   of a tree with `molrs` needs `--force`.
