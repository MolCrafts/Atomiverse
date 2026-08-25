# TB / SE method loop — handoff

Last assign: **2026-08-25 wave 9** (this-shift wave 3, `theme=gfn`) on
parent `dev` @ `7bc52489`.
Last GapCensus: **2026-08-25 wave 9** (refresh at assign HEAD; no new pending).
Last harness: **wave 8** — `gfn1-xtb-organic` + `gfn0-xtb-forces` +
`gfn2-quadrupole-aes` merged (`d214f365` / `e870f86d` / `a48f0ee3`).
**incomplete_remaining=4. pending=28. blocked=5.**
**Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Assign wave 9 (this-shift wave 3, 2026-08-25)

| slug | Kind | Action |
| ---- | ---- | ------ |
| `ipea-xtb-small` | resume | Rebase `wf/ipea-xtb-small` @ `01275c77` (ahead=1 behind=10). Spec on the tip. CH4 vs xtb IPEA-xTB **−4.670476737294 Eh**. |
| `gfn1-xtb-forces` | new | Analytic GFN1 G; replace `with_forces` throw. Oracle=xtb `--gfn 1 --grad`. Distinct file `gfn1.cpp`. |
| `gfn2-xtb-forces` | new | Analytic GFN2 G; AES already closed. Oracle=xtb `--gfn 2 --grad`. Twin `h2o_gfn2_offeq_grad` (−5.058625058504 Eh). |
| `gfn0-xtb-analytic-hessian` | new | GFN0 FD Hessian via `cpu::vib::Hessian` (G already ships). VIB=ON exception. |

Skipped same-file GFN2 ALPB/beyond-CHNO/sTDA; family Hessian/PBC/spin;
DFTB/NDDO (`theme=gfn`); `cpcm-x-xtb` (after ALPB). One staff slot empty
rather than collide on `gfn2.cpp`.

Workers: isolation worktree on nobackup; branch `wf/<slug>`; spec-first
`.claude/specs/<slug>/`. Resume rebases the existing spec. **No push.**

## Oracle / build (binding)

| Knob | Value |
| ---- | ----- |
| Family oracles | NDDO → **OpenMOPAC**; DFTB* → **DFTB+**; GFN-xTB → **xtb** |
| Forbidden | OpenMOPAC as DFTB pin; DFTB+ as PM6/PM7 pin; GFN as DFTB3 successor |
| Offline only | `tools/tb_oracle/`; never live oracle in ctest |
| SE-only cmake | SEMIEMPIRICAL+EVALUATE+TESTS+REGRESSIONS **ON**; HF/PW/GPW/MD/MC/MINIMIZER/VIB/PATH/KSPACE/POTENTIALS/NEIGHBOR/CUDA/**CLI** **OFF** |
| Hessian exception | `gfn0-xtb-analytic-hessian` may add `-DATV_BUILD_VIB=ON` only for its tree |
| Domain suite (merge gate) | `ctest -R '^test_cpu_(am1\|rm1\|dftb0\|dftb2\|dftb3\|td\|xtb_gfn0\|xtb_gfn1\|gfn2\|nddo_\|scc_)'` |
| Worker build | scoped `--target atomiverse <test_*> reg_<slug>` only |

**No oracle binary** on this host (`mopac` / `dftb+` / `xtb` absent).

## Next

1. Worktrees for the four `[~]` slugs (resume rebases onto current `dev` @ `7bc52489`).
2. Spec-first workers on `wf/<slug>`; SE-only cmake **including CLI=OFF**.
3. **No push.** Wipe builds; Clearup worktrees same wave. `git worktree remove`
   of a tree with `molrs` needs `--force`.
