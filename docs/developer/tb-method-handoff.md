# TB / SE method loop — handoff

Last assign: **2026-07-29 wave 4** on parent `dev` @ `f8e8e432`.
Last GapCensus: **2026-07-29 wave 5** (post harness wave 4).
Last harness: **wave 4** (2026-07-29) — **all 8** assign-w4 staff merged.
**incomplete_remaining=0. pending=15. blocked=9.** `new_slugs_added=3`.
**Queue not exhausted.**

## Workflow order (binding)

Recover incomplete first → Census → Assign (`incomplete ≫ pending`) →
Worktrees (**spec-first**) → Merge → Clearup → Harness → Tutorial. See
`docs/developer/tb-method-loop.md` and `docs/developer/oracle-capability-loop.md`.

## Harness wave 4 land (2026-07-29)

| slug | Result |
| ---- | ------ |
| `dftb3-analytic-forces` | **merged → done** (closed-form DFTB3 gradient incl. ∂Γ/∂R; re-SCC FD deleted; mio SK kernel unified on `internal::MioSketchPair`) |
| `dftb2-multi-k-forces` | **merged → done** (Bloch analytic F **+ stress**; fixed the self-image virial) |
| `dftb2-dispersion-d3-periodic` | **merged → done** (`d3_periodic` lattice sum + `d3_cutoff` member; offline s-dftd3 periodic pin) |
| `dftb2-analytic-stress-threeob` | **merged → done** (closed-form ThreeOb virial; strain-FD deleted) |
| `dftb2-spin-analytic-stress` | **merged → done** (one spin-resolved `analytic_forces_stress`) |
| `td-dftb-h2` | **merged → done** (Casida foothold, `DFTB2::td_casida`; **no absolute pin exists**) |
| `am1-water-mopac-energy` | **merged → done** (first hard OpenMOPAC pin on NDDO) |
| `rm1-water-mopac-energy` | **merged → done** (RM1 pin + **repaired RM1 free-atom energies**) |

Specs closed (deleted): all 7 worker specs. TB `wf/*` tips and isolation
worktrees: **none**. Gate at close: SE unit suite **15/15**, SE regressions
**26/26**.

Net effect on DFTB2: **no strain-FD product path is left anywhere** — restricted,
collinear open shell, ThreeOb and multi-k all carry closed-form forces and a
closed-form virial.

## Start here next wave

**`dftb-gamma-units-fix` is the head of the ladder.** `γ_AB` adds Å² to eV⁻²
and omits `units::coulomb`; γ_AA = U_A is accidentally correct, γ_AB(R→∞) is
**14.4× low**. Every polar MioSketch pin in the tree is provisional until this
lands. Full analysis + blast radius:
`.claude/notes/dftb-vs-dftbplus-cleanup.md`.

Then: `dftb3-periodic-forces` (DFTB3 is still molecular min-image while DFTB2
has PBC / k-mesh / stress), `nddo-sharp-discriminator` (CO₂ or CO instead of
water, which sits only 0.06 kcal/mol from experiment for RM1), then Phase B
PM3/PM6/PM7 and Phase C GFN.

## Known red gate (not TB)

`test_no_factory_scrub` is red on `dev` from `src/cpu/scf/dftd4.cpp`
`build_eeq_system` (a free `build_*` in `cpu/scf`, which admits none),
introduced by the concurrent SCF/D4 session at `3ee28243`. Already red at
`bfe049f6`, before every TB commit this wave, and no TB commit touches
`src/cpu/scf/`. Owner: that loop.

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
absent. `s-dftd3` 1.2.1 **is** installed at
`/software/sse2/el9_epyc9005/easybuild/prefix/software/Simple-DFTD3/1.2.1-gfbf-2025b/bin`
and was used offline for the periodic D3 pin; it is the same D3 engine DFTB+
calls, so for a geometry-only dispersion term it is a family-correct pin, not
an oracle substitution.

Pins that can reach neither a binary nor a published number ship **structural /
formula identities** (analytic-vs-FD, extensivity, symmetry limits) and say so
in their REFERENCE `Gaps`. **Never invent a number.** Two wave-4 cases are in
that position and are labelled as such: `td-dftb-h2` (no published H₂/mio-1-1
excitation exists — H carries only 1s in mio-1-1, so H₂ is a two-level toy) and
the DFTB formula-level stress/force gates.

## Next

1. **Assign wave 5** — `dftb-gamma-units-fix` first (blocks pin fidelity),
   then `dftb3-periodic-forces`, `nddo-sharp-discriminator`.
2. Spec-first workers on `wf/<slug>`; SE-only cmake; offline pins only.
3. Do **not** staff Phase C GFN while the γ defect is open unless
   `theme=gfn|xtb`.
4. **No push.** Wipe builds after dual-gate; Clearup worktrees same wave.
