# TB / SE method loop — handoff

Last assign: **2026-08-25 wave 11** (this-shift wave 5, `theme=gfn`) on
parent `dev` @ `a6164fe9`.
Last GapCensus: **2026-08-25 wave 11** (refresh at assign HEAD `a6164fe9`).
Last harness: **wave 11** — `gfn2-aes-fock` + `ipea-xtb-forces` merged
(`93c334cb` / `b7388af2`). No remaining `[~]`.
**incomplete_remaining=0. pending=29. blocked=5.**
**Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Harness wave 11 (this-shift wave 5, 2026-08-25)

| slug | Result |
| ---- | ------ |
| `gfn2-aes-fock` | **merged → done** — GFN2 AES (charge–dipole/quadrupole + AXC) in the SCC Fock (`cpu::semiempirical::GFN2`); off-eq H₂O vs xtb 6.7.1 `--gfn 2 --grad` (`reg_gfn2-aes-fock`; \(E\) **−5.058625058504 Eh**, \(F_{O,x}\) +3.869 eV/Å / 0.35 eV/Å). Angular \(F_{H1,y}\) is a **CAMM-vs-\(R\)** hole (xtb `mmompop` two-centre \(\Theta\) on H; ATV energy FD stays \(\sim+0.06\) eV/Å vs xtb \(-0.80\)) — not a missing Fock term. Twin `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/` |
| `ipea-xtb-forces` | **merged → done** — IPEA analytic \(F=-\nabla E\) (GFN1-shaped HF+Pulay, CN/shell-poly, \(\partial\gamma/\partial R\), rep, D3); off-eq CH4 vs xtb 6.7.1 `param_ipea` `--grad` (`reg_ipea-xtb-forces`; \(E\) **−4.662477307157 Eh**, \(F_{C,x}\) +1.801 eV/Å, \(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.26\)); `with_forces` no longer throws. Twin `tools/tb_oracle/inputs/xtb/ch4_ipea_offeq_grad/` |

Specs closed (deleted: `gfn2-aes-fock`, `ipea-xtb-forces`).
Isolation worktrees: **none** (Clearup already ran).
Local TB tips: **none** (do not `-D` non-TB `wf/*`).

Roadmap ○→✓ **IPEA-xTB analytic gradient** (CHNO closed-form \(F=-\nabla E\)).
AES-in-Fock **landed** (no table row to flip; angular \(F\) stays a CAMM
hole — GapCensus owns a deepen). IPEA VIP/VEA stays ○.
Foothold ≠ closed: ALPB, CPCM-X, PBC, spin, sTDA, beyond-CHNO, IPEA
`--vip`/`--vea`, IPEA Hessian, CAMM-vs-\(R\) AES.

## Start here next wave

**incomplete = 0.** Without `theme=gfn`, ladder returns to Phase B NDDO
(`pm3-organic-energy`, `nddo-closed-form-forces`) and DFTB3/ThreeOb
deepens (`dftb2-threeob-periodic` before `dftb2-threeob-multi-k`).
With `theme=gfn`: `ipea-xtb-vip` (same `ipea.cpp`) or ALPB / beyond-CHNO
/ sTDA (same `gfn2.cpp` — staff one). Do **not** treat GFN CHNO energy +
analytic G + family FD Hessian + AES-in-Fock + IPEA E+G as the GFN
product closed.

## Oracle / build (binding)

| Knob | Value |
| ---- | ----- |
| Family oracles | NDDO → **OpenMOPAC**; DFTB* → **DFTB+**; GFN-xTB → **xtb** |
| Forbidden | OpenMOPAC as DFTB pin; DFTB+ as PM6/PM7 pin; GFN as DFTB3 successor |
| Offline only | `tools/tb_oracle/`; never live oracle in ctest |
| SE-only cmake | SEMIEMPIRICAL+EVALUATE+TESTS+REGRESSIONS **ON**; HF/PW/GPW/MD/MC/MINIMIZER/VIB/PATH/KSPACE/POTENTIALS/NEIGHBOR/CUDA/**CLI** **OFF** |
| Hessian exception | landed Hessian slugs (`se-analytic-hessian`, `gfn0-xtb-analytic-hessian`, `gfn-xtb-analytic-hessian`) may add `-DATV_BUILD_VIB=ON` only; no staffed Hessian remaining |
| Domain suite (merge gate) | `ctest -R '^test_cpu_(am1\|rm1\|dftb0\|dftb2\|dftb3\|td\|xtb_gfn0\|xtb_gfn1\|xtb_ipea\|gfn2\|nddo_\|scc_)'` |
| Worker build | scoped `--target atomiverse <test_*> reg_<slug>` only |

**No oracle binary** on this host (`mopac` / `dftb+` / `xtb` absent).

## Next

1. GapCensus wave 12 owns discovery (CAMM-vs-\(R\) AES, IPEA Hessian,
   remaining Phase A/B/C). Do **not** staff from this harness.
2. Spec-first workers on `wf/<slug>`; SE-only cmake **including CLI=OFF**.
3. **No push.** Wipe builds; Clearup worktrees same wave. `git worktree remove`
   of a tree with `molrs` needs `--force`.
