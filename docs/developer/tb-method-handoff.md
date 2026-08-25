# TB / SE method loop — handoff

Last assign: **2026-08-25 wave 8** (this-shift wave 2, `theme=gfn`) on
parent `dev` @ `dd8b1459`.
Last GapCensus: **2026-08-25 wave 8** (refresh at assign HEAD).
Last harness: **wave 7** — `gfn0-beyond-toy` + `gfn2-xtb-water` merged
(`07f5c188` / `dee2e8ce`). Keep `gfn1-xtb-organic` `[~]`.
**incomplete_remaining=4. pending=27. blocked=5.**
**Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Assign wave 8 (this-shift wave 2, 2026-08-25)

| slug | Kind | Action |
| ---- | ---- | ------ |
| `gfn1-xtb-organic` | resume | Rebase `wf/gfn1-xtb-organic` @ `30510fc0` (ahead=1 behind=6). Spec on the tip. CH4 vs xtb `--gfn 1` **−4.274232224875 Eh**. |
| `gfn0-xtb-forces` | new | Analytic GFN0 G; replace central FD. Oracle=xtb `--gfn 0`. |
| `gfn2-quadrupole-aes` | new | Quadrupole AES on `GFN2`; close 1.45 meV H₂O residual. Must land before `gfn2-xtb-forces`. |
| `ipea-xtb-small` | new | IPEA-xTB energy foothold; own type/tables. Oracle=xtb. |

Skipped same-file GFN2 F/ALPB/beyond-CHNO; Hessian/sTDA/PBC/spin; DFTB/NDDO
(`theme=gfn`); `cpcm-x-xtb` (new pending after ALPB). One staff slot empty
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
| Domain suite (merge gate) | `ctest -R '^test_cpu_(am1\|rm1\|dftb0\|dftb2\|dftb3\|td\|xtb_gfn0\|gfn2\|nddo_\|scc_)'` |
| Worker build | scoped `--target atomiverse <test_*> reg_<slug>` only |

**No oracle binary** on this host (`mopac` / `dftb+` / `xtb` absent).

## Next

1. Worktrees for the four `[~]` slugs (resume rebases onto current `dev`).
2. Spec-first; SE-only cmake **including CLI=OFF**.
3. **No push.** Wipe builds; Clearup worktrees same wave. `git worktree remove`
   of a tree with `molrs` needs `--force`.
