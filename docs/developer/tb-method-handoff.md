# TB / SE method loop — handoff

Last assign: **2026-08-25 wave 6** on parent `dev` @ `cea5b9cc`.
Last GapCensus: **2026-08-25 wave 6**.
Last harness: **wave 4** (2026-07-29) — γ-fix / DFTB3-PBC / TD-f / multi-k
open-shell later landed into `## done` without a numbered harness wave.
**incomplete_remaining=5 (staffed this assign). pending=13. blocked=9.**
`new_slugs_added=3`. **Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Start here this wave

DFTB0 → DFTB2 depth → DFTB3 E+F+Γ-PBC → spin → **NDDO PM** is the ladder
head. γ Elstner is closed. GFN is **not** this wave (`theme` unset; GFN2
CHON never merged; toy GFN0 deleted).

Staffed (5):

| slug | Why |
| ---- | --- |
| `nddo-sharp-discriminator` | RM1 H₂O is 0.06 kcal/mol from experiment; CO₂ RM1 = −81.55 vs exp −94.05 (OpenMOPAC accuracy table error +12.50) |
| `pm3-organic-energy` | first PM* type; table pipeline for PM6/PM7 |
| `nddo-closed-form-forces` | product NDDO F is still fixed-P FD |
| `dftb3-analytic-stress` | DFTB3 has Γ F but no cell virial |
| `dftb2-threeob-multi-k` | ThreeOb still throws on `k_mesh≠Γ` |

## Recover

Nothing to rebase: no TB `wf/*`, no isolation worktrees, no live SE spec
files. INDEX's GFN2-CHON "code-complete" rows point at **missing files** —
not incomplete git tips. Do not treat them as Recover.

## Oracle / build (binding)

| Knob | Value |
| ---- | ----- |
| Family oracles | NDDO → **OpenMOPAC**; DFTB* → **DFTB+**; GFN-xTB → **xtb** |
| Forbidden | OpenMOPAC as DFTB pin; DFTB+ as PM6/PM7 pin; GFN as DFTB3 successor |
| Offline only | `tools/tb_oracle/`; never live oracle in ctest |
| SE-only cmake | `SEMIEMPIRICAL`+`EVALUATE`+TESTS+REGRESSIONS **ON**; HF/PW/GPW/MD/MC/MINIMIZER/VIB/PATH/KSPACE/POTENTIALS/NEIGHBOR/CUDA **OFF** |
| Domain suite (merge gate) | `ctest -R '^test_cpu_(am1\|rm1\|dftb0\|dftb2\|dftb3\|td\|xtb_gfn0\|nddo_\|scc_)'` |
| Worker build | scoped `--target atomiverse <test_*> reg_<slug>` only |

**No oracle binary exists on this host** — `mopac`, `dftb+` and `xtb` are all
absent. Pins come from published transcripts / accuracy tables / formula
identities. xtb 6.7.1 twins already sit on disk under
`tools/tb_oracle/inputs/xtb/` (gitignored `*.out`) for a later GFN wave.

## Next

1. Workers on `wf/<slug>`; spec-first; SE-only cmake; offline pins only.
2. Do **not** staff Phase C GFN this wave.
3. **No push.** Wipe builds after dual-gate; Clearup worktrees same wave.
