# Electronic-structure extensibility

Status: staged architecture specification; capability/evidence status is tracked in the roadmap.
Scope: Gaussian KS, PW, GPW and semiempirical electronic structure.
Execution migration: [es-unification.md](es-unification.md).
SE source review: [se-extensibility-audit.md](../notes/se-extensibility-audit.md).
Implementation/evidence status remains exclusively in `docs/roadmap/`.

## 1. Decision: execution topology and physical models are separate

A Driver is the staged execution/iteration framework. Introduce a loop node
because an algorithm has an independently controlled iteration, budget,
residual and termination policy. Do not introduce one because a physical
formula, parameterization, basis or public class has a name.

| Responsibility | Owner | Driver? |
| --- | --- | --- |
| MD, geometry optimization, a genuine SCF/SCC solve | Execution node with its own stages/status | Yes, where the algorithm has that loop |
| XC, local/nonlocal ionic potential, DFT+U contribution, repulsion | Physical model / operator / single-step Component | No |
| Basis coefficients, pseudopotential parameters, SK tables | Immutable physical data | No |
| Fock build, density build, grid integration, force contraction | Single-step stage or numerical kernel | No |
| DFTB0 / another non-self-consistent evaluation | Single-pass method Component using its parent's pass | No artificial one-iteration electronic Driver |
| A parameter variant of an existing SCF/SCC method | Existing loop implementation + model identity | No copied loop implementation |
| Post-SCF analysis | Operation over an accepted reference state | Only a genuinely iterative algorithm needs a loop; never an implicit repeat SCF |

An iterative method may **inherit** its family's Driver or **own** a private
Driver node. This choice is internal execution plumbing, not the extension
contract. One solve has one owner of its electronic loop/status/history.
Different method instances retain independent workspaces and histories even
when they reuse the same loop implementation. Do not share one mutable Driver
instance between concurrent calculations.

This does not turn every inner numerical iteration into a framework node.
LAPACK/cuSOLVER eigensolves, quadrature and bounded internal numerical kernels
remain numerical operations. Driver owns the orchestration loop whose state,
budget and termination the method/workflow exposes.

Mounting remains `outer.on(phase).add(method)` / `add_ref(method)`. Single-pass
methods mount as Components; iterative methods mount their actual nested loop
on the parent's Runtime/Backend. Composition must not add a public forwarding
`*Evaluator` or a second Runtime. No universal public `SCFSolver` / `DFTBase`
that owns all representations, physical models and special-case switches.

This refines older wording that every concrete method must be a Driver.
DFTB0 must remain single-pass; DFTB2's real SCC Driver is legitimate. A
functional, GTH parameter set or AM1 core-repulsion term never owns a Driver.

## 2. Ownership and public boundary

- Frame remains the sole public system input/output: geometry, species, box,
  charges, energies, forces and virial. No parallel `System`, `AtomList`,
  geometry-bearing configuration or user-created Context.
- Concrete methods own model selection and numerical workspace. Physical
  providers carry immutable parameters; runtime caches, density, mixer history,
  streams and device allocations belong to the method/backend.
- Standalone `run(Frame&)` overwrites energy/properties, preserves forces;
  `run(const Frame&)` leaves Frame unchanged; `forces(Frame&)` overwrites
  requested physical results; periodic `stress(Frame&)` returns intensive
  stress and writes extensive `kStress = sigma * volume`.
- Nested evaluation accumulates energy/forces/virial, overwrites the owning
  method's property columns, and obeys the parent's force/stress requests.
  Multiple unrelated electronic methods writing the same charge column require
  explicit composition ownership; never silently add their charges.
- Failed solves and failed later components do not commit electronic history.
  Initial-force preparation and trials compute without accepting history.
  KS publication uses the physical Fock from the accepted SCF iteration,
  with density-preserving occupied/virtual rotations. DIIS extrapolates and
  level-shift work matrices do not supply Pulay multipliers.
  Host geometry/cell/identity edits invalidate the relevant method caches.
- Mathematical callbacks may receive call-scoped spans of physical quantities
  such as density samples, radial functions or coefficients. These are not a
  new public Context or a borrowed workspace object. They cannot expose
  `internal::IntegralView`, `GpwAtom`, solver storage or a Runtime.
- Callers own constructor inputs until transferred; methods retain immutable
  providers by owning value or `shared_ptr<const Provider>`. Output/scratch
  spans are preallocated by the consumer and cannot be retained by providers.
  Backend execution state is not stored on a shared provider.

## 3. Extension contracts

Names and operation contracts in this section are normative design targets;
the exact span/matrix spellings must reuse existing public core/linalg types.
There is no requirement that every interface share one base class.

### 3.1 XC functional: a batched mathematical interface

Public mathematical contract: `include/atomiverse/electronic/xc_functional.h`, namespace
`ATV::electronic`. The header contains physical descriptors and the CPU
callback contract only, with no backend implementation headers. Its build
ownership must follow existing HF/PW/GPW dependencies, not enable all ES domains.

```cpp
class XcFunctional {
public:
    virtual ~XcFunctional() = default;
    virtual XcIngredients ingredients() const noexcept = 0;
    virtual bool supports(XcSpin spin, XcOrder order) const noexcept = 0;
    void evaluate(const XcBatch& in, XcOrder order, XcOutput& out) const;
protected:
    virtual void evaluate_batch(const XcBatch& in, XcOrder order,
                                XcOutput& out) const = 0;
};
```

- `XcIngredients` identifies rho, sigma, tau and/or Laplacian requirements.
  It is a physical input declaration, not a mutable algorithm configuration.
- `XcBatch` holds equally sized non-owning sample spans: rho-alpha/beta
  [Bohr^-3], sigma-aa/ab/bb [Bohr^-8], and, when requested, tau [Bohr^-5]
  and Laplacian rho [Bohr^-5]. Tau convention is explicitly
  `1/2 sum_n occupation_n |grad psi_n|^2` for each spin channel.
- Native mathematical kernels and `XcOutput` use energy density **per volume**
  [Ha/Bohr^3]. Migrate callers directly; do not preserve a per-electron output
  API or add a compatibility conversion layer. `XcOutput` also contains the
  requested partial derivatives with respect to those inputs. It never mixes
  epsilon-per-electron and energy-per-volume conventions. Sigma-ab denotes
  `grad rho_alpha dot grad rho_beta`, with no hidden factor of two.
- The method computes only required ingredients. It validates spin, derivative
  order, shapes and supported representation before starting a solve.
  Missing tau infrastructure rejects a meta-GGA provider; declaring tau does
  not implement the kinetic-density channel.
- `evaluate` is a checked nonvirtual entry point: shapes and finite inputs are
  validated before output mutation; every requested output must be written and
  finite. Providers override `evaluate_batch`, once per batch, with no virtual
  call per grid point.
  Built-in enum choices select implementations of this same contract, not a
  second numerical pipeline. Third-party functionals need no new enum member.
- Local/semi-local XC does not encode exact exchange in a fake density-only
  callback. Hybrid/range-separated exchange is a separate orbital-dependent
  operator plus its matched energy functional.
- A backend that cannot execute a provider rejects it before mutating Frame.
  CPU callbacks are not invoked inside a claimed CUDA product iteration.

### 3.2 Basis and pseudopotential: data first, operators when physics changes

**Gaussian basis data.** Reuse the explicit `GaussianBasis(element, name,
sets)` constructor. Specify angular convention (Cartesian/real spherical),
normalization, coefficient ordering, exponent units and supported angular
momentum. A new parameter table needs neither a subclass nor a Driver.
The molecular `BasisSet` selector must eventually accept an equivalent public
data object; GPW support alone does not establish that extension on Gaussian KS.

**GTH data.** Reuse the explicit `GthPotential` constructor for new parameters
in the same analytic form. Validate valence, channel sizes, angular limits,
symmetry/finite coefficients and element consistency with the basis.

**Norm-conserving potential physics.** The exported contract is
`electronic/norm_conserving_potential.h`. `NormConservingPotential` has explicit
`atomic_number()` and integral `valence()`, checked nonvirtual
`local_r(radius, value, derivative)` and immutable `projectors()` views.
Providers override only `evaluate_local` and physical metadata/views.

- Radius is Bohr, V is Ha, dV/dr is Ha/Bohr. The origin is finite with zero
  derivative; the asymptotic tail is -Z_val/r.
- `NcProjectorSetView` supplies radius, positive dr/di Jacobian, radial
  projectors, complete row-major coupling matrix and optional atomic guess
  density/orbitals. The header states the QE Simpson index-grid quadrature,
  including even meshes and each projector's exact integration prefix.
- beta is r*p_l(r) [Bohr^-1/2], D is Ha; no automatic normalization or hidden
  Ry conversion is performed by the operator. Spherical harmonics are
  normalized real spherical functions; different l blocks cannot couple.
  CPU PW admits scalar l=0,1,2. The header fixes Fourier normalization,
  atomic Bloch phases and the finite Coulomb-cancelled G=0 convention.
- `cpu::pw::GthPseudopotential(element,name)` or an explicit `GthPotential`
  supplies the analytic GTH form; `UpfPseudopotential(path)` parses once.
  Both derive the same public physical interface. UPF converts Ry to Ha at
  ingestion. Unknown classification/identity, fractional valence, NLCC,
  spin-orbit, USPP and PAW reject; no unsupported channel is approximated.
- CPU `PW` takes `shared_ptr<const NormConservingPotential>` as its sole
  potential constructor argument. Every Frame atom must match the explicit
  element; current PW is single-species. There are no source-string slots,
  source-selector overloads or compatibility aliases.
- PW setup snapshots provider data into one immutable `NcPotentialData`.
  Local transforms, nonlocal operators, forces, frozen-density stress,
  k-table rebuilds and DFPT consume that same physical snapshot. Providers
  receive no FFT/workspace/history, and are not called per grid point,
  matrix element or electronic iteration. Parameter copies retain the same
  immutable provider while owning independent execution state.
- Built-in UPF radial interpolation preserves every table node and provides a
  matched C1 derivative. Its small final-node Coulomb residual is continued
  C1 to the exact tail over one final mesh interval; operator quadrature uses
  the unchanged supplied nodes.

GPW retains its qualified Gaussian-projector integrals until a general
radial-potential adapter is implemented and independently validated. A generic
provider is not proof that every representation supports it. CUDA's migrated
source accepts built-in providers only; third-party CPU providers reject
before device execution, and numerical CUDA acceptance is a separate gate.

USPP/PAW need nontrivial overlap, augmentation density and associated energy/
geometry derivatives. They are a separate representation/operator contract,
not additional values pretending to satisfy the NC interface.

### 3.3 Electronic energy terms and independent geometry terms

Extend at an existing physical boundary, with separate local-density,
AO/projector and orbital-dependent contracts where their semantics differ.
Do not publish one `Term::compute_everything(Context&)` interface.

| Term kind | Required contribution | Consumer |
| --- | --- | --- |
| Local density functional | Energy density and matched density/gradient derivatives | Grid XC/integration stage |
| AO/projector correction, e.g. a specified +U model | Energy and matching dE/dP on declared correlated projectors | Representation-owned Hamiltonian build/apply |
| Orbital-dependent exchange | Energy and exchange-operator action, including spin/k conventions | Compatible hybrid/GKS method stages |
| Density-independent dispersion/geometry potential | Energy, force and, if supported, cell derivative from Frame | Ordinary Potential beside the electronic method |

For each density/orbital-dependent term, the energy and electronic variation
must derive from the **same** model. The representation owns overlap/Pulay
and basis response; a term provides its explicit geometry/projector/overlap
derivatives needed by that contraction. State signs, double-counting,
occupations and k weights. A blanket `forces() = 0` default is forbidden.
External force and stress support are separately qualified; absent channels
reject before a force/stress solve. Exact-zero terms may declare a documented
zero derivative when that is a mathematical identity.

DFT+U admission records the actual population/projector definition. A
gross-Mulliken implementation cannot claim a net-Mulliken oracle by widening
tolerances. Existing reference/roadmap limitations remain until re-pinned.

### 3.4 Representation and numerical algorithms

Keep the representation-specific engines private to CPU/CUDA domains. The
mathematical seams are:

1. overlap action S*X and metric/orthogonality operations;
2. kinetic and Hamiltonian action H*X (dense matrix is optional);
3. occupied-state to density/population construction;
4. density/potential transforms and local-potential projection;
5. derivatives contracted with the accepted electronic state.

Do not require PW to materialize an AO matrix or require GPW to emulate a PW
FFT layout. Gaussian Pulay, Bloch phases and augmentation remain explicit
representation responsibilities. A new representation may require new kernels.

Reuse current linalg/DIIS/charge-mixing/criteria primitives when semantics
match. Export a narrowly documented mathematical primitive only when the
out-of-tree consumer actually needs it; do not expose engine storage or
introduce a generic public SCF context. A third-party representation implements
its novel stages and, if it really has SCF, composes the existing Driver
framework. It need not reproduce scheduling, convergence machinery or the
applicable public numerical algorithms. Full third-party representation
support is a later gate, not implied by the initial XC/data examples.

### 3.5 Semiempirical model/provider contracts

**NDDO:** retain one canonical parameter payload and backend-specific numerical
kernels. Introduce constructor-injected immutable model data plus an explicit
core-repulsion/integral prescription, not a new enum branch per plugin. The
data contract includes supported elements/orbitals, one-center parameters,
Gaussian core corrections, valence and heat-of-formation reference conventions.
Model changes beyond that prescription implement the corresponding small
physical operation and its derivatives. They do not supply SCF iteration code.
Result identity/provenance must represent external models without adding
`SemiEmpiricalMethod` enumerators or misleadingly labelling them AM1.

Proposed small provider operations (physical view types, not configuration bags):

```cpp
class NddoModel {
public:
    virtual ~NddoModel() = default;
    virtual ModelIdentity identity() const = 0;
    virtual NddoElementView element(int atomic_number) const = 0;
    virtual void core_repulsion(const AtomPairBatch& in,
                                PairEnergyDerivativeBatch& out) const = 0;
};
```

Reuse/rename the existing `AM1Parameter` payload at its common owner instead
of adding a parallel table schema. Element views borrow immutable provider
storage. Pair batches contain species and distances [Angstrom]; outputs are
E [eV] and dE/dr [eV/Angstrom]. The stage turns these radial derivatives into
Cartesian forces. Built-in AM1/RM1/MNDO/PM3 select the appropriate prescription
without changing loop code. Numerical integral operations outside the supported
NDDO prescription require a separately declared model operation.

**DFTB:** an immutable `SlaterKosterProvider` supplies ordered-pair H/S blocks
and radial derivatives, repulsive energy/derivative, onsite occupations,
Hubbard data and cutoffs. Use one explicit unit/convention scheme at admission;
convert mio/3ob once at the boundary. Missing pairs/derivatives throw. A
DFTB3 model adds its defined third-order energy/variation; it does not copy
DFTB2 SCC orchestration. Spin and periodic support remain separately stated.

```cpp
class SlaterKosterProvider {
public:
    virtual ~SlaterKosterProvider() = default;
    virtual SkElementView element(int atomic_number) const = 0;
    virtual void evaluate(const OrderedPairRadialBatch& in,
                          SkRadialOutput& out) const = 0;
};
```

Inputs contain ordered species pairs and r [Angstrom]. Outputs give labelled
angular H channels [eV], S channels [dimensionless], their radial derivatives
[eV/Angstrom, 1/Angstrom], repulsion [eV], derivative [eV/Angstrom] and cutoffs
[Angstrom]. Element views provide valence/orbital metadata and Hubbard values
[eV]; third-order/spin fields declare their conventions explicitly. Direction
cosine assembly and Bloch/image sums remain in the representation. Evaluate
batches at geometry setup, then reuse admitted tables/H0/S during SCC; do not
introduce virtual calls per matrix element or pair in an iteration hot loop.

**GFN/xTB:** retain distinct Hamiltonian physics (EEQ, shell charges,
anisotropic multipoles, density-dependent dispersion). Reuse a charge-loop
implementation only where update order, metric and convergence semantics
match. Do not disguise GFN2 as a DFTB parameter set. Export additional model
operations when a real external GFN model is demonstrated, not speculatively.

None of these providers inherits Driver, mutates Frame, publishes convergence
or owns a stream/mixer/history. CPU and CUDA consume one parameter source;
backend-specific kernels and independent per-method workspaces remain owned.

## 4. Migration order and deletion gates

Each step preserves existing science bounds. Baseline failures block that
slice until understood; no relaxed assertions, self-goldens or implied CUDA
support. Existing SE-backend milestones continue without an artificial
Driver conversion for DFTB0. Each migration step is scoped independently; implementing the initial XC
slice does not qualify the remaining potential, basis, SE or CUDA providers.

| Step | Change / main locations | Completion and deletion gate |
| --- | --- | --- |
| 0 | Record loop/model distinction in CLAUDE and existing ES specs; review current DFT/SE baselines | No requirement that every method is a Driver; preserve existing user changes and numerical evidence boundaries |
| 0a | Public consumer build: `CMakeLists.txt`, `cmake/AtvInstallHeaders.cmake` | Add a proper installed CMake config/export with transitive dependencies and module-selected public headers; prove an external consumer builds without the source tree |
| 1 | CPU Gaussian KS: `scf/xc_functional.h`, `xc_grid_engine.h`, RKS/UKS build path | Built-in LSDA/PBE/BLYP use the batched contract; external XC example A passes; remove old enum-dispatch numerical fork after callers migrate |
| 2 | CPU PW: `planewave/internal/{upf_reader,gth_upf,pw_setup}` and XC consumer | GTH/NC-UPF share the existing canonical NC operator data; external example B passes; no second PP parser or SCF loop |
| 3 | CPU GPW: Kind admission, `ks_grid_build`, DIIS/OT and propagator composition | Reuse XC contract, external basis/GTH example C passes; finish existing OT/wrapper migration; delete GPWEvaluator only after propagators/callers move; general radial GPW PP is a separate gated task |
| 4 | CPU NDDO: parameter lookup + NddoMethod; CPU DFTB2 + shared SCC helper | External SE example D passes; model injection replaces closed enum dependency for external models; preserve NDDO shared loop; one SCC orchestration for matching DFTB/xTB paths, no extra Driver on model terms |
| 5 | DFTB0, DFTB3, GFN0/1/2, IPEA public/request surfaces | Keep single-pass methods Components; SCC methods may own a loop; remove duplicate with_forces/with_stress controls; migrate QmEngine/Python via same method instance; GFN2/DFTB3 science gates stay independent |
| 6 | CUDA per qualified CPU slice; first consolidate four NDDO lifecycle/loop copies | One NDDO execution implementation per backend, per-instance state; CPU shape, explicit unsupported providers, GH200 numerical/residency evidence; delete copied loops, never host-fallback to arbitrary CPU callbacks |
| Deferred | New basis families, meta-GGA, hybrids, USPP/PAW | Outside this spec's initial completion gates; separate specs after physical channels/representation adapters and independent pins exist |

Step 4's removal of parallel SCC orchestration is about execution reuse, not
merging different Hamiltonians or forcing NDDO and DFTB into one phase enum.
Changes to public Frame, unit and derivative contracts migrate C++, Python,
QM/MM, MD/Minimize and regressions in the same slice. No compatibility shims.
The table is the default interface migration sequence, not a global freeze:
independent SE alignment can proceed on its proven family contracts. A CUDA
slice depends on the corresponding CPU contract/evidence, not on completion
of every unrelated DFT feature.

## 5. Out-of-tree extension acceptance

The extension is a separate CMake project linked to the proposed installed
`Atomiverse::atomiverse` target (`find_package(Atomiverse CONFIG REQUIRED)`),
using **only public headers**. The installed package must export all required
dependencies and module-selected headers, and remain usable after moving its
installation directory. The extension builds without editing
Atomiverse headers/sources, enum lists, switches, registries or source lists.
The plugin's own build may of course add its sources. No source-tree include
path, friend declaration, private workspace access or forked SCF code.
This is source-level C++ extensibility; stable binary ABI, dlopen registries,
Python-authored GPU kernels and untrusted-code isolation are out of scope.

| Example | Implementation outside Atomiverse | Required evidence |
| --- | --- | --- |
| A: custom XC | Derive XcFunctional for scaled Slater exchange, with E and first derivatives; plug into existing RKS, then compatible PW/GPW slices | Batched point values/derivatives vs analytic formula, SCF E and full nonzero force vector vs an offline same-functional PySCF pin; UKS/spin conventions where claimed; unsupported derivative order rejects |
| B: NC potential | Supply an existing published NC/GTH potential through the public potential protocol, not an enum identity | Same-Hamiltonian baseline energy and projector/local operator action; nonzero force vector and stress when claimed; no copied reader/kernel/SCF; reject USPP/PAW payloads at admission |
| C: basis/GTH data | Supply a published basis and GTH parameterization through public constructors, including an existing in-tree reference parameterization re-expressed as external data | Existing pinned case accepts external data without a new code path; check normalization/order, missing species, malformed channels and supported angular limits; no new Driver |
| D: SE model | Supply published AM1/RM1-compatible model data through the NDDO provider, plus a mio-compatible SK provider as the DFTB follow-up | Same archived MOPAC/SQM or DFTB+ model-specific E/charges/full forces; custom identity survives results; no new enum/switch or electronic iteration code; unsupported orbitals/spin/derivatives reject |

For A/B/D, reuse published models to establish the extension boundary first;
do not invent parameters and certify them with ATV's own outputs. The scaled
exchange example's off-engine reference must actually be run with the same
functional, geometry, basis and spin. Record source/version/input/output,
quantities, units and independently justified tolerances **before** acceptance.
No generic numerical tolerance is granted by this architecture spec.

Across every supported example:

- Standalone and nested Evaluate/MD/Minimize use the same extension instance
  and accepted electronic state; no second SCF for force/stress assembly.
- Verify overwrite versus accumulation, energy-only force preservation,
  same-object failure/retry, trial rejection/initial-force history, continuation
  and geometry/cell/species invalidation. Two method instances using one
  immutable provider cannot contaminate each other's histories.
- Atomic/cell derivatives are tested separately. Finite differences and
  CPU twins supplement genuine external/analytic pins, never replace them.
- A new physical quantity/system is a section of its existing method case;
  a parameter re-expression on one path is not a new case/architecture test.
  Full solves stay in regressions, <=60 s per section at stated thread width.
  Primitive batch/derivative tests stay in tests. No source-grep gates.
- Prove no extra virtual dispatch/allocation in per-point/per-pair hot loops
  through implementation review and appropriate measurement. Timing lives in
  benchmarks; report cold/warm state, iteration counts and transfer boundaries.
- Test domain module closures: SE-only does not require HF/PW/GPW, and a
  disabled domain excludes its extension examples. Python binds the actual
  method/provider lifetime when Python support is included in the slice.
- CUDA examples require compiled device execution, GH200 numerical logs and
  observed transfer/residency evidence. A CPU plugin's success grants no CUDA
  support; unsupported CUDA providers fail before output mutation.

## 6. Review and completion

Approve each public interface by its external consumer and two real compatible
in-tree implementations. Keep a family-specific operation private when that
proof is missing. Do not add an unused all-purpose plugin registry.

This spec closes only when the initial XC, NC potential, data and SE provider
examples have passed their declared scopes, matching loop copies have been
deleted, Frame/lifecycle contracts remain green, and public docs plus roadmap
state the verified envelopes. Broader representation/USPP/PAW work remains a
separate spec if not implemented here. Delete this live spec on completion;
retain the durable contracts in the harness/developer manual.
