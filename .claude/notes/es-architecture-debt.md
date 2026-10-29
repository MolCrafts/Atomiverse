<!-- mol:note:topic:es-architecture-debt -->
## ES public-shape debt

Science factorization is sound (shared SCF/energy loop + concrete Hamiltonian;
PW / GPW / SE domain-split; method-as-nested-Driver for BOMD). Engineering does
**not** yet obey one public shape across families. The ledger below separates
source-confirmed architecture debt from historical numerical reports awaiting
recheck. It is not a fresh numerical acceptance verdict. Method/backend and
quantity status lives only in [the DFT roadmap](../../docs/roadmap/dft.md).

**Rule:** do not extend any dialect (no new Frame-bypass geometry, free-function
force entry, binding path or deeper inheritance). Close toward Frame method
entries and one outer bind rule. A genuine iterative solve owns or inherits
`Driver<*::Phase>`; a single-pass method remains a Component. Physical terms,
bases and parameterizations do not need Drivers. Design:
[es-extensibility.md](../specs/es-extensibility.md).

### Target

| Axis | Claim |
| --- | --- |
| Ownership | Concrete method owns Hamiltonian + parameters |
| Loop | A real self-consistency loop owns/inherits a typed-stage Driver; single-pass methods and model terms do not |
| I/O | `Frame` (`run` / `forces` / `stress`, docs/developer/project-contract.md table) |
| Kind | `HF` / `Semiempirical` / `PlaneWave` / `GPW` are energy-source tags only |
| Depth | Inheritance ≤ 2 |

### Post-SCF and response are not mean-field members

Post-HF, response, solvent and Hessians wrap a reference method, mirroring
PySCF: `Stability<RHF>`, `TDA` / `TDHF<RHF>`, `TDA` / `TDDFT<RKS|UKS>`,
`SfTDA` / `SfTDHF<UHF>`, `CISD<RHF>` (CIS(D)), `G0W0`, `ADC`, `DirectRPA`,
`BSE`, `GF2<RHF>`, `CCSD<RHF>::eom_{ee,ip,ea}_ccsd`, `CC2<RHF>::eom_cc2`,
`CCSD<UHF>::eom_sf_ccsd`, `MP2` / `CCSD` / `CASCI(mf)`,
`Hessian<RHF|UHF|RKS|UKS|ROHF>` (KS/ROHF differentiate `forces`; nuclear CPHF
in `Hessian<RHF>`), `Polarizability<RHF>`, `WithSolvent<Method>`; FD skeleton
`central_difference_nuclear_hessian`. Density fitting stays an SCF JK selector
(`ri`).

### Live dialects (do not grow)

1. **Method-as-Driver** — CPU Gaussian mean-field (`MeanFieldMethod`).
2. **Method + `*Evaluator`** — `GPWEvaluator` only, kept for the CPMD/XL/
   second-gen propagator state and legacy plain users. Gaussian SCF, AM1/RM1,
   PW and BO `GpwDft` mount directly as nested nodes on the parent Runtime.
3. **Owned/composed loops** — DFTB2 now owns persistent SCC stages as a
   Driver; DFTB3/GFN1/GFN2/IPEA compose `SccDriver` from Components. Both
   shapes are legitimate for real SCC loops; orchestration/lifecycle should
   share one implementation. DFTB0/GFN0 remain single-pass Components.
   CPU NDDO self-consistency is `NddoMethod` (`Driver<nddo::Phase>`);
   `NddoConverged` judges the residuals `scf_check` publishes, and `drive_`
   writes `result_.converged` from the Driver verdict.

### Ledger

| Sev | ID | Where | Debt |
| --- | --- | --- | --- |
| 🔴 | `es-bind-dialects` | wrapper, direct nesting and composed-loop lifecycle | Target: mount the method itself. Gaussian SCF, CPU NDDO, PW, BO GPW and DFTB0/2 mount directly. Component composition is not itself a defect. Open: GPWEvaluator migration, matching SCC lifecycle/orchestration consolidation, GPW OT stages, PW post-processors and older SE Frame/request surfaces. |
| 🔴 | `es-se-extension-boundary` | CPU/CUDA semiempirical fronts, model lookup and SCC helper | Five source-confirmed findings in [se-extensibility-audit.md](se-extensibility-audit.md): copied CUDA NDDO execution, closed model admission, parallel SCC orchestration, public Runtime helper and duplicate force-request controls. No blanket Driver conversion; implementation plan is `es-extensibility.md`. |
| 🟡 | `es-dftb-qmengine` | DFTB0/3, GFN0 vs DFTB2, GFN1/2 | Uneven `QmEngine` coverage for QM/MM `set_engine`. |
| 🟡 | `es-docs-drift` | `docs/tutorial/methods/gpw-{d3,dft-plus-u,mbd,xdm}.md` | General API/guide/ch09 now describe direct BO mounting and the remaining wrapper. Method-specific examples still use `GPWEvaluator` and must migrate with their regression callers; do not describe wrapper removal as complete. |
| 🟡 | `es-reg-internal` | `regressions/gpw/{d3,tkatchenko-scheffler,mbd,hirshfeld,xdm}` | 17 CPU section sources across five method cases still include private headers (D3: 5, TS: 2, MBD: 3, Hirshfeld: 3, XDM: 4). Twelve archived CUDA GPW consumers now use public Kind/Frame on both backends; syntax, original settings and numerical gates are preserved. Their archived runtime marks are not fresh acceptance. |
| 🟡 | `es-gpw-thread-determinism` | `tests/cpu/gpw/test_cpu_gpw_thread_reproducibility.cpp:123` | Primitive layout/contraction/XC tests do not validate full UKS+U/OT solve repeatability. Historical 8-thread differences need fresh checks on surviving sections listed below; no current failure is asserted. |
| 🟡 | `es-pw-stress-contract` | `src/cpu/planewave/internal/pw_stress.cpp:579`, `src/cuda/planewave/internal/pw_stress.cu:601` | Both production paths use one-SCF frozen-density NSCF strain FD (h=2e-3), despite `analytic` internal names. Roadmap and public/private header contracts now distinguish the algorithms. Closed-form derivatives remain a separate implementation task; numerical acceptance of the current FD path does not close it. |
| — | `es-kind-model` | `cpu/component/hf.h` | Decided: `HF` stays a tag on the Driver node (Gaussian HF and hybrid KS); `Semiempirical` / `PlaneWave` / `GPW` stay Component kinds; the slot copies `provides_*`. |
| — | `es-ghost-atoms` | `MeanFieldMethod::ghost_atoms` | Named exception. |

### Numerical recheck targets

These are surviving registered sections, not freshly reproduced failures.
Run with their declared thread widths, then record convergence, numerical
gates and time per section independently. A build with HF/PW/GPW or regressions
disabled cannot validate these targets.

| Case | Sections | Historical concern / next evidence |
| --- | --- | --- |
| `gpw/rks` | `gpw-kpoints-si`, `gpw-kpoint-stress`, `gpw-multigrid-kpoints` | Si convergence/runtime; absolute energy and stress have different oracle depth. |
| `gpw/periodic-mp2` | `periodic-ri-mp2-si`, `periodic-scs-mp2-si` | Γ-Si reference SCF convergence and per-section time; preserve the genuine CP2K correlation-energy pins. |
| `response/phonons` | `gpw-fd-phonon-si` | Open-shell cluster SCF and total section time. |
| `gpw/dft-plus-u` | `gpw-dftu-forces`, `gpw-dftu-kpoint-forces`, `gpw-dftu-kpoint-stress` | Convergence and repeated 8-thread solves; CP2K projector mismatch is a separate model/validation issue, not a tolerance fix. |
| `dynamics/second-gen-cpmd` | `second-gen-cpmd-ot`, `second-gen-cpmd-uks-ot` | OT convergence/time and UKS repeatability. |
| `gpw/uks`, `gpw/d3`, `gpw/mbd`, `gpw/wannier` | `gpw-uks-forces`, `gpw-uks-stress`, `gpw-d3bj-forces`, `gpw-mbd-stress`, `periodic-wannier-gpw` | Historical >60 s sections; remeasure before choosing smaller systems. |

Old `periodic-{gw,ugw,rpa,urpa,ump2,scs-ump2}` and multi-k correlated
self-pinned cases, `gpw-dftu-kpoints`, and standalone GPW functional-swap
cases are absent from current registration. Do not schedule deleted paths as
failing tests. Their missing independent validation remains in the roadmap;
deletion does not establish numerical correctness. GFN2 reports are owned by
the semiempirical track and were not revalidated by this DFT census.

### Source-confirmed closures

- `es-cuda-gpw-frame-io`: typed Driver and direct parent mount, private
  atom/cell and derivative helpers, public Frame geometry. GH200 allocation
  4522779 passes 41 units, four complete Driver cases and four PBE-family
  public twins. The initial failed diagnostic and its corrected lifecycle
  expectation are retained, with production archive byte-identical. Host
  setup/AO Pulay/stress and device handoff remain separate residency defects.
- `es-cuda-pw-driver`: native post-refactor acceptance passes 72 units,
  16 fixed/FD model cases, four physical nested branches and the existing
  QE Si₂ energy/full-force gate on GH200 allocation 4522779.
- `es-cuda-es-tests-e2e`: three complete-SCF stress cases moved unchanged
  to opt-in `pw-cell-stress-baselines`, outside CTest, and all three native
  cases pass. Other unit files are not certified by this targeted migration.
- `es-gpw-ot-freeloop`: all six CPU OT outer paths and Γ-RKS polish now
  share the existing typed Driver iteration loop. Internal line searches and
  bounded fixed-H optimizers remain numerical kernels. Total nested budgets,
  live bindings, reentry, exception cleanup and delayed derivative publication
  are checked on actual solves. CPU allocation 4522778 passes 180 units,
  11 Driver cases, four MPI comparisons, eight FD preflights and the existing
  CP2K water gate; the initial UKS virial failures remain archived. The final
  diagnostic tightens Γ SCF precision at unchanged comparison gates.
- CUDA PW now has one typed Driver iteration loop for all four SCF branches;
  run/force/stress share the accepted device state and nested execution borrows
  the parent's owner. CPU/CUDA plan refresh preserves user stages/criteria.
  The CPU fix passes 39 units and 16 fixed/FD public cases with actual stage
  measurements; the frozen old CPU archive fails eight lifecycle probes.
  Final NVCC targets, 29 host guards and the scoped post-refactor GH200
  nested, lifecycle, thermal and existing QE checks pass (allocation 4522779);
  see [scoped evidence](../../tools/diagnostics/pbe-family/results/2026-10-27-pw-driver/README.md).
- `PWEvaluator` is deleted. `src/cpu/planewave/pw_solver.cpp:1070`
  (`PW::run_nested_erased`) calls `evaluate_`, which runs SCF once and passes
  the resulting state into force/stress assembly. The old three-SCF evaluator
  debt is closed structurally; this is not a fresh numerical run.
- BO `GpwDft` mounts directly (`src/cpu/gpw/gpw_solver.cpp:5933`);
  remaining `GPWEvaluator` callers and propagation state still need migration.
- CUDA GPW full-product and kpoint e2e checks are sections of `gpw/rks`;
  no architecture-freeze regression case remains. CPU/CUDA shape conformance
  must be reviewed on the actual public headers.

### Conformance

| Method family | Driver? | Frame `run`/`forces`? | Conforms? |
| --- | --- | --- | --- |
| CPU RHF/UHF/RKS/UKS/ROHF/ROKS/GHF | Yes | Yes | Pass |
| CUDA RHF / UHF | Yes | Yes (UHF: STO-3G l ≤ 1, else throws) | Pass |
| AM1 / RM1 | Yes (`NddoMethod`, nested on parent Runtime) | Yes (both `QmEngine`) | Pass |
| CPU PW | Yes | Yes (+ `stress`) | Pass; nested force/stress assembly shares one SCF |
| CUDA PW | Yes | Yes (+ `stress`) | Unified; scoped native nested/lifecycle acceptance passes |
| CPU GpwDft | Yes (DIIS and OT) | Yes (+ `stress`) | Unified loop/mount; propagator-wrapper migration remains separate |
| CUDA GpwDft | Yes | Frame run/forces/stress; atom/cell helpers private | Unified; scoped native acceptance passes; residency debt remains |
| `*Evaluator` wrappers | Component | launch → Frame | Bind shell only |
| `CarParrinelloMethod` / `XlBomdMethod` | Composed `DriverNode` | owned method via `method()` | Pass |
| DFTB0 / GFN0 | No SCC Driver required; single-pass Components | DFTB0 Frame entries; GFN0 request/surface debt | No artificial Driver conversion |
| DFTB2 | Yes, persistent real SCC loop | Frame run/forces/stress + nested accumulation | Correct loop boundary; provider and shared orchestration work remains |
| DFTB3, GFN1/2, IPEA | Composed real SCC helper | Older Component/request surface | Composition is valid; lifecycle, public boundary and Frame/request migration remain |

### Close order

1. `es-bind-dialects`
2. GPW/PW device residency and complete stress algorithms
3. QmEngine coverage
4. Hygiene: docs and `es-reg-internal`
