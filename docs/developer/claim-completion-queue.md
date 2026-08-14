# Claim-completion queue — CUDA / docs / Python for claimed domains

Operator directive (2026-08-14): for every domain that has reached its claim
bar on CPU, complete the three delivery surfaces — **CUDA**, **docs**,
**Python bindings**. The torch backend is removed; molrs/molpy alignment is
**latest** (molrs 0.13.1 submodule `95dde84`, molpy 0.13.x). Release
engineering is out of scope for this queue.

Every open row below has a live spec under `.claude/specs/<slug>/`
(see `.claude/specs/INDEX.md`, claim-completion campaign); the two Track D
rows `docs-python-examples` + `docs-python-nav` are covered by the single
spec `docs-python-surface`. CUDA specs are `deferred: true` (GH200 batch +
CUDA ES open gate).

## Claim bars (binding for this queue)

A domain claim requires all of:

1. ≥ 1 end-to-end regression anchored on an **external-code oracle or cited
   literature value** — self-freezes, cross-backend twins, and `isfinite`
   gates do not count.
2. **Forces** whenever the domain feeds dynamics; energy-only is a
   "single-point foothold" claim, never the domain claim.
3. A **CUDA** claim needs a device-principal implementation **and** an
   independent anchor (CPU-twin equivalence alone stays ◐).
4. A **roadmap chapter row** — a capability without a census row is invisible
   and cannot be claimed (electrodes/observe were in this state).
5. A **perf** claim (⚡) needs an in-repo, runnable benchmark; a pointer to a
   removed suite voids the mark.

## Track P — Python bindings

Binding shape stays: task-spec structs + handles in
`python/src/bindings/bind_*.cpp`, canonical deep namespaces, top-level
convenience re-exports, capability-gated via `ATV_HAS_*`. Every row lands
with mirrored `python/tests/` coverage and a `getting_started` section.

| Slug | Domain | Deliverable | Gate |
| ---- | ------ | ----------- | ---- |
| py-schema-molrs-0131 | all | **DONE 2026-08-14** — fixtures/tests/docs on `atomic_number` (uint) + plain-value meta; pins molrs==0.13.1, molpy==0.13.1; `element` column rides the C++ allow-list | python suite green (176 pass) |
| py-md-npt | MD | **first slice DONE 2026-08-14** — `Langevin`/`CSVR` thermostats + MTK `NoseHooverNPT` via `set_integrator`, full task-spec/codec/materializer mirror (wire IDs 11–13, bitwise materializer parity tests); residual: Berendsen / Parrinello–Rahman / per-axis couple + drag/chain knobs | python suite green (181); gate 223/223 |
| py-coulomb-pme | MD | bind `coul_cut`/`coul_long` + KSpace PME (`ATV_BUILD_KSPACE` wheel define + capability gate) | NaCl PME energy matches `md/npt_pme_nacl` pin from Python |
| py-pair-styles | MD | bind morse/buckingham/born/yukawa/wca/dpd + FENE bond, CHARMM dihedral, harmonic improper | ctor/energy smoke per style |
| py-recorder-xyz | MD | bind an XYZ trajectory Recorder — trajectory output from Python. **Design gate first:** `cpu::XYZ` is molrs-backed and `ATV_BUILD_IO`-gated while the task layer (`task_spec`/`materializer`) compiles unconditionally, so a task-mirrored XyzRecorder either moves behind the IO gate cleanly or stays a local-only (non-launcher) component — decide before binding | file written + row count |
| py-minimize-full | Minimize | bind CG + FIRE + cell-relax variants; evaluator dispatch beyond LJ (any bound Potential) | r_min invariant + cell smoke |
| py-mc | MC | bind `mc::MC` preset + movers (NVT/NPT/GCMC) | analytic ⟨N⟩/⟨V⟩ smoke vs `mc/` pins |
| py-vib | Vibration | bind `vib::Hessian` + RRHO thermochem read-back | frequency smoke vs `scf` freq pin |
| py-scf-single-point | SCF | RHF/UHF/RKS/UKS single-point + forces wrappers (the `driver.py` "future spec") | energy vs PySCF-pinned regression value |
| py-am1-forces | SE | expose AM1 gradients + AM1 evaluator in MD (AIMD from Python) | force vs CPU pin; NVE drift smoke |
| py-pw-forces | PW | PW forces/stress accessors; PW as MD evaluator | force accessor vs CPU pin |

Order: py-md-npt → py-coulomb-pme → py-recorder-xyz first (they complete the
"classical MD from Python" story); then py-scf-single-point / py-am1-forces;
py-mc / py-vib / py-pw-forces / py-minimize-full / py-pair-styles are
independent.

## Track D — docs

| Slug | Deliverable | Gate |
| ---- | ----------- | ---- |
| docs-roadmap-electrodes | `docs/roadmap/electrodes.md` — capability table for the MetalWalls map (ConstPot BO/CG/MaZe, TF, piston, dipole joint, observers), backend columns, ✓/○ per census discipline | registered in roadmap index |
| docs-roadmap-observe | observables census (RDF/MSD/VACF/density/profiles/charges/Voronoi/Widom/WHAM) with anchoring status per observable | registered in roadmap index |
| docs-multiscale-stale-rows | flip Löwdin / ESP(MK) / CHELPG rows ○→✓ (already shipped: `cpu/observe/lowdin.h`, `esp_charges.h`, gated by `regressions/scf/h2o_charges_pyscf`) | rows match tree |
| docs-roadmap-legend | define ◐ in the legend; re-point the LJ ⚡ evidence at a live benchmark (see cuda-md-lj-bench) | legend complete |
| docs-python-examples | `python/examples/` gets MD-LJ, minimize, AM1 scripts (today only `pw_qe_compare.py`, for the OFF-by-default backend) | scripts run under the dev build |
| docs-python-nav | register `python/docs/getting_started.md` in `zensical.toml` nav (currently outside `docs_dir`, unpublished) | page in site nav |

## Track C — CUDA

ES rows honor the **CUDA ES open gate** (CLAUDE.md): CPU regressions + oracle
pins green, public shape stable, CUDA↔CPU tolerances stated, perf claims only
on large systems. Classical MD CUDA is an independent track. CPU-only domains
by design (MC, SE, electrodes, bias, vibration, path, multiscale) have **no
CUDA row** — that is layout, not debt.

| Slug | Domain | Deliverable | Gate |
| ---- | ------ | ----------- | ---- |
| cuda-md-lj-bench | MD | restore `lj/performance/` benchmark so the roadmap's only ⚡ has runnable evidence (A/B methodology: `.claude/notes/cuda-openmm-md-perf.md`) | benchmark runs on GH200; ⚡ evidence path live |
| cuda-md-npt | MD | device NPH/PR barostat path (CPU parity + LAMMPS pin, tile engine unchanged) | CUDA↔CPU trajectory tolerance stated; LAMMPS pin green |
| cuda-scf-gradient | SCF | device RHF gradient kernels (unblocks GPU AIMD; today "CUDA SCF has no gradient kernel") — after the nested `cuda::Driver<scf::Phase>` fold residual | force vs CPU ≤ stated tol; CPU PySCF pins green |
| cuda-pw-spin-pbe | PW | device spin-PBE forces/stress (today rejects), then spin+smear | dual-gate vs CPU + QE-class anchor |
| cuda-gpw-product | GPW | continue the frozen public shape (`.claude/notes/cuda-gpw-public-shape.md`): water σ, multi-k F/σ, UKS F/σ, MG production; independent anchors per claim bar 3 (◐→✓ only then) | es-cpu-cuda queue gates |

## Non-goals

- No new claim attempts for below-bar domains (path/NEB gate tightening,
  metadynamics ΔF pin, GFN2 merge, periodic QM/MM) — those are capability
  work, queued elsewhere.
- No packaging/PyPI work.
