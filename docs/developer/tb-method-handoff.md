# TB / SE method loop — handoff

Last assign: **2026-08-25 wave 7** (this-shift wave 1, `theme=gfn`) on
parent `dev` @ `c05f6b45`.
Last GapCensus: **2026-08-25 wave 7**.
Last harness: **wave 7** — `gfn0-beyond-toy` + `gfn2-xtb-water` merged
(`07f5c188` / `dee2e8ce`). Keep `gfn1-xtb-organic` `[~]`.
**incomplete_remaining=1. pending=23. blocked=5.**
**Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Harness wave 7 (this-shift wave 1, 2026-08-25)

| slug | Result |
| ---- | ------ |
| `gfn0-beyond-toy` | **merged → done** — `cpu::semiempirical::GFN0` H/C/N/O energy; methanol vs xtb 6.7.1 **−7.597234498685 Eh** (`reg_gfn0-beyond-toy`; residual ~3.6e-8 eV); twin `tools/tb_oracle/inputs/xtb/ch3oh_gfn0/` |
| `gfn2-xtb-water` | **merged → done** — `cpu::semiempirical::GFN2` H/C/N/O energy; H₂O vs xtb 6.7.1 **−5.070383095219 Eh** (`reg_gfn2-xtb-water`; residual 1.45 meV; forces throw); twin `tools/tb_oracle/inputs/xtb/h2o_gfn2/` |
| `gfn1-xtb-organic` | **keep `[~]`** — unmerged `wf/gfn1-xtb-organic` @ `30510fc0` (CH4 vs xtb `--gfn 1`); Recover before new pending |

Specs closed (deleted: `gfn0-beyond-toy`, `gfn2-xtb-water`). Isolation
worktrees: **none** (Clearup already ran). Local TB tip:
`wf/gfn1-xtb-organic` (incomplete — do **not** `-D`).

Roadmap ○→✓ **GFN0-xTB** + **GFN2-xTB** energy (H/C/N/O). Foothold ≠ closed:
analytic GFN F, ALPB, IPEA, PBC, spin, sTDA, Hessian, elements beyond CHNO,
and GFN1 still unmerged.

## Start here next wave

**Recover `gfn1-xtb-organic` first** (`incomplete ≫ pending`). Then either
finish GFN1 merge or staff GFN2 dependents (`gfn2-xtb-forces`, `alpb-xtb`)
if `theme=gfn|xtb` still holds. Without theme, ladder returns to Phase B
NDDO (`pm3-organic-energy`, `nddo-closed-form-forces`) and DFTB3/ThreeOb
deepens.

Do **not** treat GFN energy footholds as the GFN product closed.

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

1. Recover/merge `gfn1-xtb-organic` (or rebase onto current `dev`).
2. Spec-first workers on `wf/<slug>`; SE-only cmake **including CLI=OFF**.
3. **No push.** Wipe builds; Clearup worktrees same wave. `git worktree remove`
   of a tree with `molrs` needs `--force`.
