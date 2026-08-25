# TB / SE method loop — handoff

Last assign: **2026-08-25 wave 6** on parent `dev` @ `cea5b9cc`.
Last GapCensus: **2026-08-25 wave 6**.
Last harness: **wave 6 partial** — `nddo-sharp-discriminator` merged
(`6f161e8d`). Remaining assign-w6 slugs were not started.
**incomplete_remaining=0. pending=17. blocked=9.** `new_slugs_added=3`.
**Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Harness wave 6 (partial, 2026-08-25)

| slug | Result |
| ---- | ------ |
| `nddo-sharp-discriminator` | **merged → done** — RM1 CO₂ ΔHf −81.5544 vs PARAM −81.55 kcal/mol, r=1.17296 vs 1.173 Å; CO −13.8686 / 1.13818; twin `tools/tb_oracle/inputs/mopac/rm1-co2/` |
| `pm3-organic-energy` | not started → pending |
| `nddo-closed-form-forces` | not started → pending |
| `dftb3-analytic-stress` | not started → pending |
| `dftb2-threeob-multi-k` | not started → pending |

Specs closed (deleted). TB `wf/*` and isolation worktrees: **none**.

**Found:** SE-only cmake now needs **`-DATV_BUILD_CLI=OFF`** —
`ATV_BUILD_CLI=ON` requires `ATV_BUILD_MD=ON` (`CMakeLists.txt:1252`). Loop
doc cmake block updated.

## Start here next wave

Ladder is still **NDDO PM**: `pm3-organic-energy` then
`nddo-closed-form-forces`, then PM6/PM7. Parallel DFTB deepens
(`dftb3-analytic-stress`, `dftb2-threeob-multi-k`, `dftb3-kpoints`) are
independent of NDDO files.

Do **not** staff Phase C GFN unless `theme=gfn|xtb`. Toy GFN0 is deleted;
INDEX GFN2-CHON claims never landed.

## Oracle / build (binding)

| Knob | Value |
| ---- | ----- |
| Family oracles | NDDO → **OpenMOPAC**; DFTB* → **DFTB+**; GFN-xTB → **xtb** |
| Forbidden | OpenMOPAC as DFTB pin; DFTB+ as PM6/PM7 pin; GFN as DFTB3 successor |
| Offline only | `tools/tb_oracle/`; never live oracle in ctest |
| SE-only cmake | SEMIEMPIRICAL+EVALUATE+TESTS+REGRESSIONS **ON**; HF/PW/GPW/MD/MC/MINIMIZER/VIB/PATH/KSPACE/POTENTIALS/NEIGHBOR/CUDA/**CLI** **OFF** |
| Domain suite (merge gate) | `ctest -R '^test_cpu_(am1\|rm1\|dftb0\|dftb2\|dftb3\|td\|xtb_gfn0\|nddo_\|scc_)'` |
| Worker build | scoped `--target atomiverse <test_*> reg_<slug>` only |

**No oracle binary** on this host (`mopac` / `dftb+` / `xtb` absent).

## Next

1. Assign remaining Phase B (`pm3-organic-energy`, `nddo-closed-form-forces`)
   + DFTB3/ThreeOb deepens.
2. Spec-first workers on `wf/<slug>`; SE-only cmake **including CLI=OFF**.
3. **No push.** Wipe builds; Clearup worktrees same wave. `git worktree remove`
   of a tree with `molrs` needs `--force`.
