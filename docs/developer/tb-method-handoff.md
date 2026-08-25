# TB / SE method loop — handoff

Last assign: **2026-08-25 wave 9** (this-shift wave 3, `theme=gfn`) on
parent `dev` @ `7bc52489`.
Last GapCensus: **2026-08-25 wave 9** (refresh at assign HEAD).
Last harness: **wave 9** — `gfn1-xtb-forces` + `gfn2-xtb-forces` +
`gfn0-xtb-analytic-hessian` merged (`276c7353` / `b8e86743` / `d4582d31`).
Keep `ipea-xtb-small` `[~]`.
**incomplete_remaining=1. pending=28. blocked=5.**
**Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Harness wave 9 (this-shift wave 3, 2026-08-25)

| slug | Result |
| ---- | ------ |
| `gfn1-xtb-forces` | **merged → done** — analytic GFN1 \(F=-\nabla E\) (HF+Pulay, CN/shell-poly, \(\partial\gamma/\partial R\), rep, D3); off-eq water vs xtb 6.7.1 `--gfn 1 --grad` (`reg_gfn1-xtb-forces`; \(E\) −5.768641651774 Eh, \(F_{O,x}\) 1.87e-3 eV/Å); twin `tools/tb_oracle/inputs/xtb/h2o_gfn1_offeq_grad/` |
| `gfn2-xtb-forces` | **merged → done** — analytic GFN2 \(F=-\nabla E\) (HF+Pulay, IES, rep, D4, AES kernel at frozen CAMM); off-eq water vs xtb 6.7.1 `--gfn 2 --grad` (`reg_gfn2-xtb-forces`; \(E\) **−5.058625058504 Eh**); AES still post-SCC (angular \(F_{H1,y}\) hole) |
| `gfn0-xtb-analytic-hessian` | **merged → done** — GFN0 FD Hessian via `cpu::vib::Hessian`; water ω vs xtb 6.7.1 `--gfn 0 --hess` (`reg_gfn0-xtb-analytic-hessian`; 1563.14 / 3497.87 / 3505.03 cm⁻¹); twin `tools/tb_oracle/inputs/xtb/h2o_gfn0_hess/` |
| `ipea-xtb-small` | **keep `[~]`** — unmerged `wf/ipea-xtb-small` @ `054f6c8e` (ahead=2 behind=6); spec lives on the tip |

Specs closed (deleted: `gfn1-xtb-forces`, `gfn2-xtb-forces`,
`gfn0-xtb-analytic-hessian`). Isolation worktrees: **none** (Clearup already ran).
Local TB tip: `wf/ipea-xtb-small` (incomplete — do **not** `-D`).

Roadmap ○→✓ **GFN-xTB analytic gradient** (GFN0/1/2 CHNO closed-form
\(F=-\nabla E\)). **GFN-xTB analytic Hessian** stays ○ (GFN0 FD H ships;
GFN1/2 is `gfn-xtb-analytic-hessian`). IPEA stays ○.
Foothold ≠ closed: ALPB, CPCM-X, PBC, spin, sTDA, family Hessian,
beyond-CHNO, IPEA, GFN2 AES-in-Fock.

## Start here next wave

**Recover `ipea-xtb-small` first** (`incomplete ≫ pending`). Rebase
`wf/ipea-xtb-small` onto current `dev` before gate. Then, if `theme=gfn|xtb`
still holds, staff GFN2 dependents (`alpb-xtb`, `gfn2-beyond-chno`) or
family Hessian (`gfn-xtb-analytic-hessian` — GFN1/2 F now exist). Without
theme, ladder returns to Phase B NDDO (`pm3-organic-energy`,
`nddo-closed-form-forces`) and DFTB3/ThreeOb deepens.

Do **not** treat GFN CHNO energy + analytic G + GFN0 Hessian as the GFN
product closed.

## Oracle / build (binding)

| Knob | Value |
| ---- | ----- |
| Family oracles | NDDO → **OpenMOPAC**; DFTB* → **DFTB+**; GFN-xTB → **xtb** |
| Forbidden | OpenMOPAC as DFTB pin; DFTB+ as PM6/PM7 pin; GFN as DFTB3 successor |
| Offline only | `tools/tb_oracle/`; never live oracle in ctest |
| SE-only cmake | SEMIEMPIRICAL+EVALUATE+TESTS+REGRESSIONS **ON**; HF/PW/GPW/MD/MC/MINIMIZER/VIB/PATH/KSPACE/POTENTIALS/NEIGHBOR/CUDA/**CLI** **OFF** |
| Hessian exception | `gfn-xtb-analytic-hessian` / `se-analytic-hessian` may add `-DATV_BUILD_VIB=ON` only for their tree |
| Domain suite (merge gate) | `ctest -R '^test_cpu_(am1\|rm1\|dftb0\|dftb2\|dftb3\|td\|xtb_gfn0\|xtb_gfn1\|gfn2\|nddo_\|scc_)'` |
| Worker build | scoped `--target atomiverse <test_*> reg_<slug>` only |

**No oracle binary** on this host (`mopac` / `dftb+` / `xtb` absent).

## Next

1. Recover/merge `ipea-xtb-small` (rebase onto current `dev` @ `d4582d31`).
2. Spec-first workers on `wf/<slug>`; SE-only cmake **including CLI=OFF**.
3. **No push.** Wipe builds; Clearup worktrees same wave. `git worktree remove`
   of a tree with `molrs` needs `--force`.
