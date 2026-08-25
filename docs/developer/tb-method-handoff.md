# TB / SE method loop — handoff

Last assign: **2026-08-25 wave 8** (this-shift wave 2, `theme=gfn`) on
parent `dev` @ `dd8b1459`.
Last GapCensus: **2026-08-25 wave 8** (refresh at assign HEAD).
Last harness: **wave 8** — `gfn1-xtb-organic` + `gfn0-xtb-forces` +
`gfn2-quadrupole-aes` merged (`d214f365` / `e870f86d` / `a48f0ee3`).
Keep `ipea-xtb-small` `[~]`.
**incomplete_remaining=1. pending=27. blocked=5.**
**Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Harness wave 8 (this-shift wave 2, 2026-08-25)

| slug | Result |
| ---- | ------ |
| `gfn1-xtb-organic` | **merged → done** — `cpu::semiempirical::GFN1` H/C/N/O energy; CH4 vs xtb 6.7.1 `--gfn 1` **−4.274232224875 Eh** (`reg_gfn1-xtb-organic`; residual 10.2 meV); twin `tools/tb_oracle/inputs/xtb/ch4_gfn1/`; forces throw |
| `gfn0-xtb-forces` | **merged → done** — analytic GFN0 \(F=-\nabla E\) (HF+Pulay, EEQ/CN, repulsion, D4, SRB); off-eq water vs xtb 6.7.1 `--gfn 0 --grad` (`reg_gfn0-xtb-forces`; \(F_{O,x}\) 2.6e-6 eV/Å); twin `tools/tb_oracle/inputs/xtb/h2o_gfn0_offeq_grad/` |
| `gfn2-quadrupole-aes` | **merged → done** — GFN2 charge–quadrupole AES + qkernel AXC; H₂O vs xtb 6.7.1 **−5.070383095219 Eh** (`reg_gfn2-quadrupole-aes`; residual **26 µeV**, was 1.45 meV); forces still throw |
| `ipea-xtb-small` | **keep `[~]`** — unmerged `wf/ipea-xtb-small` @ `01275c77` (ahead=1 behind=7); spec lives on the tip |

Specs closed (deleted: `gfn1-xtb-organic`, `gfn0-xtb-forces`,
`gfn2-quadrupole-aes`). Isolation worktrees: **none** (Clearup already ran).
Local TB tip: `wf/ipea-xtb-small` (incomplete — do **not** `-D`).

Roadmap ○→✓ **GFN1-xTB** energy already landed on the merge (`924d1327`);
harness does not flip further rows. **GFN-xTB analytic gradient** stays ○
(GFN0 analytic G ships; GFN1/2 `with_forces` still throw). IPEA stays ○.
Foothold ≠ closed: GFN2 F, ALPB, CPCM-X, PBC, spin, sTDA, Hessian,
beyond-CHNO, IPEA.

## Start here next wave

**Recover `ipea-xtb-small` first** (`incomplete ≫ pending`). Rebase
`wf/ipea-xtb-small` onto current `dev` before gate. Then, if `theme=gfn|xtb`
still holds, staff GFN2 dependents (`gfn2-xtb-forces`, `alpb-xtb`) — AES
before F is done. Without theme, ladder returns to Phase B NDDO
(`pm3-organic-energy`, `nddo-closed-form-forces`) and DFTB3/ThreeOb
deepens.

Do **not** treat GFN energy + GFN0 G as the GFN product closed.

## Oracle / build (binding)

| Knob | Value |
| ---- | ----- |
| Family oracles | NDDO → **OpenMOPAC**; DFTB* → **DFTB+**; GFN-xTB → **xtb** |
| Forbidden | OpenMOPAC as DFTB pin; DFTB+ as PM6/PM7 pin; GFN as DFTB3 successor |
| Offline only | `tools/tb_oracle/`; never live oracle in ctest |
| SE-only cmake | SEMIEMPIRICAL+EVALUATE+TESTS+REGRESSIONS **ON**; HF/PW/GPW/MD/MC/MINIMIZER/VIB/PATH/KSPACE/POTENTIALS/NEIGHBOR/CUDA/**CLI** **OFF** |
| Domain suite (merge gate) | `ctest -R '^test_cpu_(am1\|rm1\|dftb0\|dftb2\|dftb3\|td\|xtb_gfn0\|xtb_gfn1\|gfn2\|nddo_\|scc_)'` |
| Worker build | scoped `--target atomiverse <test_*> reg_<slug>` only |

**No oracle binary** on this host (`mopac` / `dftb+` / `xtb` absent).

## Next

1. Recover/merge `ipea-xtb-small` (rebase onto current `dev` @ `a48f0ee3`).
2. Spec-first workers on `wf/<slug>`; SE-only cmake **including CLI=OFF**.
3. **No push.** Wipe builds; Clearup worktrees same wave. `git worktree remove`
   of a tree with `molrs` needs `--force`.
