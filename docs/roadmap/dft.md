# Density-Functional Theory

Tutorial: [KS](../tutorial/ch05-kohn-sham-dft.md) · [hybrids](../tutorial/ch06-hybrid-dft.md) · [PW](../tutorial/ch08-plane-wave.md) · [GPW](../tutorial/ch09-gpw.md).

Molecular KS shares the Gaussian SCF solver with
[Hartree–Fock](hartree-fock.md). Periodic DFT (plane-wave + GPW) lives in
§4–6. Legend: [Roadmap index](index.md).

The tables retain the historical **`dev`** inventory and its stated scope.
The [scientific regression runner was removed](../../regressions/README.md);
the linked case sources and reference outputs are now archives, not executable
checks of the current checkout. Current molecular acceptance runs through
[`benchmarks/scf/bench_rks.py`](../../benchmarks/scf/bench_rks.py) and
[`benchmarks/scf/bench_uks.py`](../../benchmarks/scf/bench_uks.py), using live
PySCF and the public Python methods. Historical marks below do not substitute
for current benchmark results.

Historical CPU validation on 2026-10-25 passed 1785
unit tests and all 68 enabled regression cases (301 sections at their declared
thread widths), plus the relocated installed XC extension project. Affected
PW cases were rechecked after the final repair; unchanged cases retain their
accepted results. CUDA was not rerun. Energy, forces and stress have separate
coverage; a ✓ remains limited to the systems, quantities and settings in its
cited case. Passing an existing check does not close the qualification gaps below.

The CPU NC-extension repair was rechecked on 2026-10-26: 1817 unit tests and
14 affected regression cases (63 sections, maximum 26.5 s per section) passed.
The 50 Python PW tests and all five modes of the relocated independent NC
project passed, along with its three required modes on a fresh minimal
PW+Evaluate installation. This is a scoped recheck, not a new run of all 68 cases.
CUDA was not rerun.

## Execution checklist

This is the working queue for autonomous design, implementation, validation
and optimization. Work on the highest-priority item whose dependencies are
ready; CPU and CUDA proceed independently when a hardware queue is pending.
The capability tables below retain their narrower evidence scopes.

An item is checked only for its stated backend and envelope after its required
checks pass. Source implementation, compilation, native numerical acceptance
and external-engine qualification are recorded separately. A queued job is
unfinished. Preserve failed runs and original numerical gates; extend a check
when a physical boundary is missing, rather than dropping the failed setting.

For each item: inspect the public contract and accepted state ownership;
implement the smallest complete vertical slice; run primitive and public-method
checks appropriate to it; compare energy, full forces and supported stress
against an independent reference; record convergence, settings, native outputs
and source/binary hashes. Profile accepted calculations before optimizing;
remeasure with the same physical settings and retain their numerical gates.
Complete-method solves stay outside unit tests: archived cases go to
`regressions/`, external-engine comparisons to `benchmarks/compare/`.

**P0 — finish current numerical acceptance**

- [x] `nc-projector-normalization`: twenty normalization-dependent amplitude
  skips now use exact-zero semantics across setup, CPU/CUDA Hamiltonians,
  forces and response. The original beta -> a beta, D -> D/a^2 controls
  dropped a 0.187354647692 Ha nonlocal term to zero at a=1e9 or1e-35;
  the repaired finite controls recover H*C, full forces and strain terms.
  Actual CPU/CUDA H*C, Sternheimer and selected finite-q response checks
  pass. Five of thirteen response screen sites have direct numerical
  coverage; the remaining edits retain compilation/algebraic coverage,
  not a new Hessian qualification. Arbitrary exponent ranges remain limited
  by representable intermediates.
- [x] `pbe-cpu`: shared PBE-family parameters, point/gradient checks and
  public fixed/FD PW checks; final CPU run passes 39 units and 16 cases.
- [x] `gpw-stage-reference`: correct periodic images and Gaussian-tail
  support without changing production kernels or gates; all ten native
  GH200 stage tests pass, including the five previously failed comparisons.
- [x] `pw-cuda-acceptance`: native post-refactor fixed/FD, move/reuse,
  standalone and nested lifecycle/accumulation, original stress baselines,
  and existing actual-QE Si₂ energy/full-force pin. The unused queued job
  4521991 is superseded by combined PW/GPW acceptance directly on reusable
  GH200 allocation 4522779. The scoped envelope passes 72 units, three moved
  full-SCF cases, 16 fixed/FD model cases, four nested branches and the existing
  actual-QE Si₂ energy/full-force pin.
- [x] `pbe-pw-periodic-oracles-cpu`: six actual QE 7.6 references for
  PBEsol/revPBE/RPBE × Si₂ RKS/skew H₃ UKS at Γ with FD smearing. Absolute
  Mermin energy, every force and six numerical strain-FD stress components
  pass the original gates.
- [x] `pbe-pw-periodic-oracles-cuda`: the same six independent references
  pass on GH200, including nonzero thermal and collinear-spin states.
- [x] `pbe-gpw-periodic-oracles-cpu`: qualify the actual CP2K 2026.2
  PBEsol/revPBE/RPBE Γ RKS references after repairing Gaussian-core overlap.
  Absolute energy, every force and supported diagonal stress pass their
  unchanged gates after the core correction.
- [x] `pbe-gpw-periodic-oracles-cuda`: the same independent GPW gates
  pass natively on GH200.
- [x] `pbe-gpw-uks-periodic-oracles-cpu`: four actual CP2K 2026.2
  PBE/PBEsol/revPBE/RPBE H3 Gamma UKS references (alpha2/beta1), with
  matched grids, basis and CORE_PPL GRID. Absolute energy, all nine forces
  and three supported diagonal stresses pass unchanged independent gates;
  the current CPU envelope also passes 233 units and existing Driver/MPI
  checks. This does not qualify empty-beta boundary derivatives or shear.
- [x] `pbe-gpw-uks-periodic-oracles-cuda`: the same four actual CP2K
  Gamma H3 UKS references pass on GH200 with unchanged absolute E/full-F/
  diagonal-stress gates, after coupled-spin DIIS and residual control.
- [x] `gpw-gaussian-core-overlap`: include the missing short-range correction
  between Gaussian ionic cores in every supported CPU/CUDA energy, full-force
  and cell-strain path. Actual CP2K 2026.2 H₂ references expose a
  `5.338928237e-6` Ha energy term and an `8.0368611e-5` Ha/Bohr x-force
  correction; the original independent gates currently fail. Preserve those
  runs, periodic/self-image counting and controlled tails, then verify ionic
  and strain derivatives and repeat the unchanged independent gates.
  The repaired scoped envelope passes 188 CPU units, 46 native CUDA units,
  29 x86 host guards, existing Driver/MPI/physical differences and both new
  three-case independent CP2K matrices. Maximum E/full-F errors are below
  `7e-10` Ha / `8e-9` Ha/Bohr; original failed runs remain failed.

**P1 — unify GPW execution and complete the periodic foundation**

- [x] `gpw-cpu-ot-driver`: route all six existing OT outer paths
  through the method's typed-stage Driver; preserve line searches, DIIS
  polish and total iteration budget. Preserve user stages/criteria across
  paths and moves. Require Γ RKS/UKS and supported multi-k physical checks,
  force/energy consistency, reuse and nested lifecycle checks. Allocation
  4522778 passes 180 units, the 11-case Driver matrix, four one/two-rank
  k-point comparisons, eight physical preflights and the retained CP2K water
  gates.
- [x] `gpw-cuda-driver-frame`: use `Driver<gpw::Phase>` for Γ RKS
  SG/MG, Γ UKS and multi-k RKS; remove public atom-list/cell/free-function
  entry points. Require one reference SCF per derivative call, real standalone
  Frame, borrowed parent Backend/Frame/stream, request preservation, nested
  accumulation, move/exception/reuse checks and unchanged physical gates.
  GH200 allocation 4522779 passes 41 units, all four complete Driver cases
  and four public PBE-family twins. The initial lifecycle-check count failure
  is retained; only its diagnostic expectation changed, with the production
  archive byte-identical.
- [ ] `gpw-propagator-mount`: migrate remaining `GPWEvaluator` consumers
  together with CPMD/XL/second-generation electronic propagation state;
  preserve each propagator's accepted-history and zero-SCF contracts.
- [x] `gpw-cuda-gamma-uks-pbe`: single-grid PBE/PBEsol/revPBE/RPBE
  through the existing typed Driver and Frame. Coupled alpha/beta DIIS
  matches the CPU algorithm; explicit positive finite `residual_tol` retains
  its old 1e-5 production default. All eight strict H3/empty-beta complete
  Driver cases pass at residual 1e-9 and original H3 100/H2 200 budgets,
  including nested request/stream/exception/retry checks. The scoped current
  envelope passes 233 CPU units, 131 distinct native CUDA units, 30 x86 host
  guards, original public gates and both actual CP2K UKS matrices. Five new
  coupled-DIIS primitives also pass memcheck with zero errors/leaks. UKS
  forces remain host-assembled; empty-beta parity does not qualify XC floor
  derivatives.
- [ ] `gpw-cuda-semilocal-depth`: UKS multigrid and multi-k UKS beyond
  the accepted Gamma single-grid slice; qualify multi-k/MG energy, every force
  and supported stress on asymmetric multi-atom systems.
- [ ] `xc-spin-domain-derivatives`: define the admitted spin domain and
  regularization/pullback contract before qualifying negative-spin, floor
  and fully polarized boundaries. The retained negative-beta probe returns
  PBE `v_beta≈99.42` while differentiating the raw composite energy gives
  `≈-0.03354` Ha. Research against LibXC 7 distinguishes its admitted
  nonnegative-density analytic endpoint convention from derivatives through
  caller-side clipping. A global raw-energy Jacobian replacement changes
  published empty-beta virtual spectra by hundreds of Ha and is therefore
  not an accepted repair. Preserve the established physical point law;
  qualify any discrete caller map and its derivative as a separate contract.
  Positive-density finite differences and CPU/device parity alone do not
  close these boundary questions. [Contract and compatibility study](../developer/xc-spin-domain.md).
- [ ] `gpw-cuda-residency`: retain accepted SCF grid/density state for force
  assembly without re-upload, implement device AO Pulay and cold start, then
  close multi-k setup/payload and stress transfers. Native consistency and
  Driver conformance do not close these device-residency defects.
- [x] `gpw-cuda-gamma-force-handoff`: Γ RKS single-grid force assembly
  consumes accepted device P, physical raw-H W and resident potentials, with
  cold immutable derivative tables and no host AO contraction or field
  re-upload. Native acceptance passes 45 units, four Driver cases, four
  public PBE-family twins and five p-shell water cases (90 force lanes).
  Captured H2D is zero; only the 4-byte eigensolver status and independently
  proven 8-byte vendor matrix-norm scalar return to the host. Memcheck reports
  zero errors/leaks. This does not close UKS/MG/k force, cold setup or stress
  residency.
- [x] `pw-analytic-stress-cpu-components`: closed-form Ewald and XC
  six-component virials plus NC radial/Cartesian projector and local-potential
  derivatives. All 37 component tests pass, including independent discrete
  energy strain differences, shear and rotations. Existing energy/potential
  laws and the public numerical total-stress path are preserved. The broader
  current CPU envelope passes 233 units and actual QE/CP2K comparisons.
- [x] `pw-analytic-stress-cpu-gamma-fixed-lsda`: the public Gamma RKS,
  fixed-occupation LSDA path now assembles all six closed-form tensor entries
  from one final-band ensemble, with no strained band solves. Force/stress
  combined requests share that one unstrained sweep and derivative Ewald
  object, and preserve accepted SCF metadata. Public run/forces/stress and a
  nested combined call pass the actual QE VWN5 E/full-F/six-stress gates,
  real Frame/metrics/criterion checks and one SCF lifecycle per call.
  The affected envelope passes 268 CPU units, 106 native CUDA units,
  three opt-in SCF stress baselines, 31 x86 guards and nine memcheck cases
  with no errors/leaks. Existing PW and actual QE matrices pass unchanged.
- [x] `pw-analytic-stress-cuda-operator-components`: device kinetic,
  Hartree, local and full-D nonlocal energy/W9 operators with borrowed
  stream/BLAS/scratch. Eight native tests cover signed f, full/unequal radial
  prefixes, Gamma/nonzero k, rotations, all strain lanes and normalization
  controls. Cold radial setup remains on the host; this is not CUDA total
  analytic stress or a device-only complete workflow.
- [x] `pw-analytic-stress-cpu-gamma-semilocal`: extend the same public tensor
  assembly to Gamma LSDA/PBE-family RKS/UKS with fixed or Fermi–Dirac
  occupations. Six actual-QE thermal/spin references pass unchanged E/full-F/
  six-stress gates; measured internal calls prove one SCF and one all-spin
  unstrained sweep per standalone stress or nested combined request. The
  original 16-family numerical stress comparison also passes. Final-library
  acceptance passes 278 CPU units, the new public/general/family checks on
  x86 and ARM, 14 affected native CUDA units and existing physical matrices;
  all 1426 captured source fingerprints agree. Fixed empty-spin, Gaussian/MP
  and multiple-k dispatch retain their earlier paths. This wave adds no new
  MPI or auxiliary-eigenstate completion qualification.
- [x] `pw-final-state-completion`: CPU final bands now require the existing
  Davidson target `tol*.1` in the actual Hpsi residual and norm-squared error
  below 1e-8, within the unchanged 40-expansion budget. Local auxiliary-work
  failures are synchronized before spectrum/density collectives. Real short
  budgets reject without publishing outputs; the same parent can retry.
  Prepared false reference verdicts remain false. Serial and MPI2 Gamma/
  eight-k checks cover empty ranks and remote failures. This guard does not
  cover every earlier admission/allocation error.
- [x] `pw-analytic-stress-cpu-k-fixed-lsda`: restricted fixed LSDA now uses
  the same tensor on general k meshes. Distributed T/nonlocal E+W and weighted
  electron count share one 21-scalar SUM; replicated terms are not multiplied
  by rank count. Actual QE eight-k Si2, serial/MPI2 public and nested calls,
  rotation and fixed-Miller full-SCF strain checks pass their original gates.
  Separate zero-SCF admission checks reject remote bad counts/shapes/weights
  consistently. Native acceptance includes 280 CPU units and ARM components.
- [x] `pw-analytic-stress-cuda-gamma-fixed-lsda`: the public Gamma RKS fixed
  LSDA derivative path now owns one final-band ensemble, requested device
  force/virial scratch and one parent publication kernel. Force and virial
  share a raw full-D projection; energy-only allocates no derivative workspace.
  Actual auxiliary Hpsi/norm and device finite/reference/raw-antisymmetry
  checks precede publication. Nine complete modes, 15 affected GTests (one
  with eight certificate controls), original physical matrices and two fresh
  zero-error/leak memchecks pass. Three initial ARM MPI startup failures remain
  failed and are closed by explicit unchanged-binary supplemental executions.
  Cold radial/Ewald work, auxiliary metadata and requested host outputs remain
  explicit boundaries; PBE-family, smear, spin and multi-k total tensors are
  separate open capabilities.
- [x] `pw-xc-energy-precision`: compensate the two existing padded-grid
  XC energy reductions without changing the point law, potentials or W.
  The retained Si2 tiny-strain failure is reproduced by the old accumulator;
  the actual candidate passes all original RKS/UKS component gates with
  active density/potential/force/W payloads bit-identical. Ten uniform-density
  exact-integral cases fail on the old library and pass the unchanged
  precision bound on both x86 and ARM. The scoped wave passes 278 CPU units,
  ARM precision/public checks and the original physical matrices.
- [x] `pw-analytic-stress-cpu-k-fd-semilocal`: reuse the final-band tensor
  for semilocal Fermi–Dirac RKS/UKS on supported k layouts. Ten matched actual
  QE cases cover full 2x2x2 meshes, five models, nonzero UKS magnetization and
  nontrivial entropy. Serial/MPI2 energy, full forces and six stress entries
  pass unchanged gates, with exact local band budgets and one publication.
  ARM repeats all ten cases. General-k admission does not imply independent
  qualification of every k layout; Gaussian/MP retain their earlier path.
- [x] `pw-analytic-stress-cuda-gamma-fixed-pbe-family`: extend the existing
  certified owner to PBE/PBEsol/revPBE/RPBE with the shared XC point law,
  distinct raw-gradient/flux buffers and 3x quadrature. Four matched actual
  QE references pass E/full-F/six-stress gates; 36 complete request/retry modes,
  four pure model controls and two current memchecks pass. Force-only skips
  tensor quadrature; energy-only has no owner. Current native acceptance
  binds 1453 source files, 18 CUDA artifacts, 16 affected GTests and the old
  physical matrices. The private missing-return diagnostic exit 139 remains
  false; its corrected current binary passes. Spin, smear and multiple-k
  CUDA total tensors remain open.
- [x] `pw-analytic-stress-cpu-gaussian`: the common final-band tensor now
  admits semilocal Gaussian RKS/UKS at Gamma and supported k layouts. Twelve
  matched actual-QE primary cases cover two Gamma and ten full-eight-k cases;
  serial/MPI2 E/full-F/six-stress, Ne/M/generalized entropy, exact band budgets
  and shared C/f/Ewald checks pass. The original near-integer .02-Ry control
  keeps its failed entropy qualification; an independent .04-Ry case supplies
  nontrivial broadening. Nominally empty fixed/Gaussian spin channels and MP
  retain their earlier paths. Gaussian entropy is not physical temperature.
- [x] `pw-analytic-stress-cuda-gamma-fd`: Gamma RKS LSDA/PBE-family final
  spectra now determine consistent mu/f/S and exact orbital density. Five
  matched QE references, 45 FD and 45 retained fixed method modes pass; full
  reference metadata remains unchanged on normal and exceptional exits.
  Auxiliary internal energy/free energy are separate; no entropy pressure is
  added. Actual Hpsi1e-6/norm1e-8 and the 20-round budget are unchanged.
  Native acceptance binds 1463 sources and 21 CUDA artifacts, with 17 GTests
  and four zero-error/leak memchecks. Spin, other smear kinds and k meshes
  remain separate CUDA total-tensor work.
- [x] `pw-final-occupation-tails`: CPU final nonlocal force and CUDA final
  density retain every nonzero occupation. Prepared positive/signed tiny-tail
  controls fail old screening and pass the scaled original gates; ordinary
  CUDA SCF density remains bit-identical and retains its prior cutoff policy.
  CPU/ARM NC and native GPU controls pass. This does not qualify signed-XC
  SCF, a density-floor extension or MP response.
- [x] `pw-analytic-stress-cpu-k-fixed-semilocal`: general-k fixed band-index
  semilocal RKS/populated UKS shares the final-band tensor. Ten matched actual-QE
  cases pass E/full-F/six-stress on serial/MPI2 and ARM, with original budgets,
  Ne/M/S and one publication. This is a per-k occupation constraint; the observed
  negative indirect-gap control remains unqualified as a global insulator.
- [x] `pw-analytic-stress-cuda-spin-xc`: strictly positive-spin LSDA/PBE-family
  device XC tensors pass all nine strain lanes at two steps, CPU/point/primal
  checks, rotation and nontrivial negative cross-gradient controls. One GTest
  covers five models with zero SCF/band solves; current memcheck has zero
  errors/leaks. This component does not qualify a public UKS owner or empty,
  clipped or signed-density XC domains.
- [x] `pw-analytic-stress-cpu-empty-beta`: Gamma fixed LSDA with Na>0/Nb=0
  uses a producer declaration, canonical zero-row beta blocks and collective
  validation. A matched actual-QE stretched-triplet reference passes E/full-F/
  six-stress on serial/MPI2 and ARM, including rotation; all nine frozen strain
  lanes pass at two steps. Zero-band/full-D and malformed/empty-rank controls
  pass. Short-bond and GGA failures remain independently unqualified. Other
  empty-spin/filling/k domains retain their earlier paths.
- [x] `pw-cuda-joint-spin-pulay`: alpha/beta histories reuse the restricted
  Pulay engine and retain the unmixed joint residual. Depth-one/history-wrap,
  opposite-spin and CPU parity checks pass, together with original public
  fixed/FD/Gaussian/MP1 and request/retry scopes. Asynchronous coefficient and
  cold upload sources have explicit lifetime ownership; the separate linear
  kernel is removed. An isolated same-input Si2 control converges in 105 rounds within the
  unchanged 200-round budget; its different QE electronic state remains false.
  This is no wall-time speedup or new signed-XC variational qualification.
- [x] `pw-cuda-packed-spin-operators`: independent spin blocks share one
  band-sum projection/force/tensor implementation. Seventeen zero-SCF controls
  cover full-D projectors, signed/tiny weights, scale gauges, all-nine strain
  and ionic force differences, collapsed RKS and packed bands exceeding G.
  The shared header owns the project → force → virial scratch contract;
  nonfinite expected oracles reject. Current native unit and memcheck pass.
  This is a pure operator component, separate from the public UKS owner.
- [x] `pw-analytic-stress-cpu-gamma-mp1-pbe`: the integrated MP1 PBE tensor
  for actual Gamma and initially populated UKS channels passes matched QE
  E/full-F/six-stress on serial/MPI2, together with independent
  all-nine frozen strains and twelve independent full-SCF deformations.
  Its actual negative occupations, signed density and generalized entropy
  are retained. The 1487-source CPU wave covers 84 groups/126 rank records,
  281 units and 16 private artifact identities, including relocated replay.
  Three original parser-helper exits remain failed; seven same-log strict
  recollections pass without another SCF. The LSDA compressed-strain failure
  and broader MP/response gaps remain open.
- [x] `pw-analytic-stress-cpu-gamma-rks-mp1-pbe`: restricted MP1 PBE at
  actual Gamma passes the Si2 12-Ry/.02-Ry/six-band matched-QE envelope.
  Actual occupations below zero and above two retain degeneracy two,
  total charge eight, zero moment and generalized entropy. Nonzero nonlocal
  energy/tensor, all forces/six stresses, all-nine frozen strains at two
  steps and twelve independent full-SCF deformations pass their original
  gates. The fresh 1508-source main wave completes 88 groups/132 rank
  records, 281 units and 18 artifact identities with unchanged input hashes.
  Its current-library mathematical oracle is separately linked against the
  normal build's actual Rust/vendor/bridge dependencies. Independent review
  and relocated parser replay hide the original repository, molpy and SDK.
  Earlier declaration/tool refusals remain recorded; other MP domains and
  the LSDA signed-density boundary remain open.
- [x] `pw-analytic-stress-cuda-gamma-uks`: two immutable final spin ensembles,
  shared total/spin densities, nonlocal force/tensor accumulation and one
  publication for fixed occupations and nonempty spin channels. The current
  joint-mixer source passes 162 native GH200 records: 135 request cases,
  33 unit tests, independent frozen strain/force differences, finite-domain
  fallback/retry and three zero-error/zero-leak memchecks. Eighteen separate
  artifact copies and a repository-independent relocated parser replay are
  sealed and independently reviewed. The original gate-representation refusal
  and archive-tool failures remain recorded; physical gates are unchanged.
  Finite unsupported spin-XC fields retain numerical epilogues and explicit
  host transfers. FD, multiple k points and empty endpoints remain separate.
- [ ] `pw-analytic-stress-cuda-gamma-uks-fd`: cache both fresh spin spectra
  and coefficient blocks, fill once at a shared chemical potential, and build
  both orbital densities from their corresponding cached coefficients through
  one density implementation. Preserve accepted result metadata and the
  per-call publication/fallback contract. Require shared-mu and tiny-tail
  component controls, matched-QE H3 plus a nonzero-projector spin case,
  full force/stress and all-nine frozen strains, request/lifecycle checks
  and native memcheck. Fixed-UKS or hydrogen-only coverage is insufficient.
- [x] `pw-cuda-nl-force-stream`: both legacy nonlocal-force kernel launches
  now explicitly use the solver's borrowed stream. Captured before/after
  execution moves the twelve custom launches from the default stream to the
  same stream as all twelve cuBLAS GEMVs; original force and public gates pass.
  The old numerical pass is retained as insufficient proof of stream
  ownership. This does not remove legacy per-call force scratch allocation.
- [ ] `pw-analytic-stress-cpu`: implement closed-form kinetic, ionic,
  nonlocal, Hartree and XC strain derivatives with correct fixed-basis and
  thermal/free-energy conventions. Require six-component reference stress,
  full-SCF deformation checks and rotation checks, fixed/FD/Gaussian/MP
  and restricted/collinear envelopes recorded separately.
  [Algorithm and operator-derivative design](../developer/pw-analytic-stress.md)
  is available; this does not mark its implementation complete.
- [ ] `pw-analytic-stress-cuda`: device counterpart with the same physical
  definitions and independent six-component gates; do not qualify it from
  the existing frozen-density strain-FD implementation.
- [ ] `gpw-general-cell-stress`: general cells and physical shear, followed
  by analytic derivatives and device assembly. Current orthorhombic fixed-P
  diagonal differences do not close a six-component tensor.
- [ ] `pw-mp-density-boundary`: resolve the observed LSDA×MP near-zero
  signed-density SCF failure with a variationally consistent treatment;
  retain the failed compressed-H₃ case and independent energy derivatives.

**P2 — expand physical models through the existing methods**

- [ ] `molecular-ks-cuda`: RKS/UKS LSDA/PBE energy and analytic full forces,
  then broader basis/GGA and hybrid coverage; require live matched PySCF
  references, open-shell/empty-spin cases and actual device execution.
- [ ] `xc-provider-periodic`: extend the public batched mathematical XC
  contract to CPU PW/GPW; add device implementations with explicit admission.
  Preserve energy-density, spin/sigma, derivative-order and ownership units.
- [ ] `meta-gga`: kinetic-density τ ingredients and matched operator/force/
  stress derivatives before adding r2SCAN/TPSS models; qualify each backend.
- [ ] `periodic-hybrids`: orbital-dependent exact/screened exchange for
  periodic PBE0/HSE, with separate convergence, energy, force and stress
  qualification. Do not encode exchange in a density-only XC callback.
- [ ] `dispersion-depth`: public-consumer D3/D4/TS/MBD energy/full-F/stress
  coverage and device counterparts; replace XDM's reimplementation pins
  with an accepted independent engine reference.
- [ ] `dftu-matched-model`: align projector/occupation definitions with an
  independent engine before promoting DFT+U comparisons; qualify Γ/multi-k,
  spin, OT and forces/stress separately.

**P3 — response, long trajectories and scale**

- [ ] `response-depth`: MP signed-weight DFPT projectors, metallic/spin
  response boundaries and analytic CPKS/Hessian; retain existing CPU scopes.
- [ ] `analysis-cuda`: device bands/DOS, phonons, Berry polarization and
  MLWF over an accepted reference state, with no implicit duplicate SCF.
- [ ] `aimd-reuse`: density/orbital prediction and cache reuse across geometry
  edits; preserve trial rejection and accepted history. Check long-trajectory
  energy drift, cell changes and thread/MPI repeatability.
- [ ] `pw-device-synchronization`: reduce PW band-side scalar synchronization
  after profiling; GPW scientific-state transfers remain P1 above.
- [ ] `large-engine-extensions`: PAW/USPP/GAPW, noncollinear/SOC and multi-GPU,
  each as a separately designed and independently validated capability.

## Current executable evidence

The current empty-beta and mixing wave
covers CPU Gamma/supported general-k semilocal fixed/FD/Gaussian derivatives
with populated spins, plus the narrow Gamma fixed-LSDA empty-beta tangent.
The ten fixed, ten FD and twelve Gaussian references retain their original
E/full-F/six-stress gates; one new stretched-triplet LSDA reference adds
serial/MPI2, rotation and ARM checks. All 102 primary per-rank records pass.
Fixed band indices per k do not imply a global insulating gap.
CUDA retains its Gamma RKS fixed/FD tensor and positive-spin pure XC component;
joint-spin Pulay now passes its own density and public filling/mode checks.
The public CUDA UKS tensor owner remains open.
Current acceptance records 281 CPU units, 82 groups/123 rank records/15 identities,
183 GH200 command leaves (121 GPU and62 CPU-only ARM),31 component GTests,
90 RKS method modes,31 x86 guards, five separate ARM primitives and six
zero-error/leak memchecks. ARM overlapping steps use disjoint cores, live PMI
and no GPU GRES, with observed peak eight. Three CPU and one ARM entropy
controls retain raw exit1 and false scientific qualification.
All 1480 captured source identities match in an actual read-only execution
view. Shared-workspace PPPM updates caused the first host-view precheck to
reject before science; this attempt and a launcher/parser rejection remain
separate. The accepted view preserves the captured source without overwriting
host work. Exact source bytes and50 native archive/ELF copies are archived;
relocated replay passes through the same source and artifact views. Prior
profiles, negative-gap and unmatched-state failures retain their attribution.
Broader source/build/repository qualification flags remain false.

The PBE-family follow-up
passes 39 distinct CPU units, eight public PW RKS/UKS cases and four public
GPW RKS cases across PBE, PBEsol, revPBE and RPBE. Independent ionic force
differences are within 2.705e-7 eV/Å on PW and 2.249e-5 eV/Å on GPW. Both
backends share immutable model parameters and GPW probe admission. CUDA GPW
now has `stress(Frame&)`, using one device solve plus host fixed-density
strain assembly, with orthorhombic shear still zero. The CUDA library and
both diagnostics compile, and 20 host admission checks pass. Actual GH200
job 4519883 passed all eight PW and four GPW public CPU/CUDA cases, but five
GPW stage comparisons failed among 74 units; its overall record remains
failed. The separate frozen-library recheck
passes all ten corrected periodic/windowed stage references at the original
gates (GH200 job 4521262). The subsequent CUDA PW Driver refactor still needs
its own native acceptance. These consistency results supply no new
independent periodic qualification.

The subsequent PW Driver recheck
unifies CUDA PW under `Driver<pw::Phase>` and fixes CPU/CUDA preservation
and execution of user-mounted stages/criteria. The final CPU run passes 39
units and 16 fixed/FD public cases with actual stage measurements; a frozen
pre-fix CPU archive fails all eight intended lifecycle probes. Reusable
GH200 allocation 4522779 now passes the complete PW post-refactor envelope:
72 units, three moved SCF baselines, 16 fixed/FD model cases, four nested
branches and the existing QE Si₂ energy/full-force pin. The separate
GPW Driver envelope
passes 41 units, four complete Driver cases and four PBE-family public twins;
overlapping unit cases are counted once. Final x86 CUDA compilation and
29 host admission checks also pass. CPU allocation 4522778 supplies direct build/test/MPI
steps; each allocation lasts two hours and avoids a new queue per iteration.

The follow-up public molecular matrix (campaign `2026-10-26-dft-wave2`)
passed 67 workloads / 134 required CPU-oracle cells, retaining the preceding
35-workload envelope and adding standard CAM-B3LYP, PBEsol, revPBE and RPBE.
Maximum errors were 1.287e-9 Ha and 1.208e-9 Ha/Bohr. The
standard CAM record
also passed 62 native units and six additional live molecular energy/full-force
comparisons; three extreme scalar spin/density points retain explicitly
recorded B88/LYP/VWN numerical differences. The earlier molecular diagnostics
recorded 28 affected native units, 24 comparisons and the initial CAM failures.
The GH200 record separately
passed 37 actual units, public PW result/reuse checks, and a matched builtin
GTH Si₂ PBE/Γ/smearing QE energy/full-force pin. These are scoped numerical
results; dirty-source/build-attestation and broader capability gaps remain.

The shared smearing repair
passes 65 local points and 20 spectra from actually compiled QE 7.6 routines,
plus nine native units. The archived production implementation retains 61
failed comparisons, including one NaN. MP quasi-occupations are now signed,
the generalized entropy has the correct degeneracy, and chemical-potential
search stops on electron count instead of drifting to a gap edge. Six public
DFPT admission tests separately verify that MP response is rejected before
SCF at all three q paths in restricted and collinear settings; response
projectors still require a signed-weight implementation.

The CPU PW follow-up
records LSDA/PBE × FD/Gaussian/MP1 at Γ on skew H₃. All six reference-cell
energies and six-component stress tensors pass matched actual QE 7.6 pins:
maximum errors 1.193e-6 Ha and 1.984e-5 eV/Å³. Five independent full-SCF
simple-deformation checks pass within 4.162e-7 eV/Å³, and both FD rotation
checks within 3.365e-7 eV/Å³. The LSDA-MP compressed sample does not converge;
the complete matrix remains explicitly failed. Stress now retains the
reference integer basis and FFT shape, accounts for both symmetric-shear
lanes, and refills the two spin spectra at one chemical potential. This is
still a numerical frozen-density strain derivative, not closed-form stress.
The same CPU record compares one and two MPI ranks on a 2×2×2 k mesh:
energy and stress differences are 3.047e-12 Ha and 2.039e-9 eV/Å³. Its original
source and cutoff-reselection negative controls retain their expected failures.

<!-- benchmark-evidence:start -->

Campaign `9d6af9a65069414b83d739d432387d79`; profile `linux-x86_64-cpu1-acceptance`; tier `acceptance`.

Numerical acceptance and roadmap qualification are separate. A local dirty or unverified build cannot promote ✓/⚡. Unmigrated capabilities retain explicit gaps.

| Capability ID | Capability | CPU | CUDA | Evidence |
| --- | --- | :---: | :---: | --- |
| `dft.rks.energy` | RKS molecular energy | ◐ | ○ | CPU numerical pass; CUDA numerical coverage pending; official source/build qualification incomplete |
| `dft.rks.gradient` | RKS molecular gradient | ◐ | ○ | CPU numerical pass; CUDA numerical coverage pending; official source/build qualification incomplete |
| `dft.uks.energy` | UKS molecular energy | ◐ | ○ | CPU numerical pass; CUDA numerical coverage pending; official source/build qualification incomplete |
| `dft.uks.gradient` | UKS molecular gradient | ◐ | ○ | CPU numerical pass; CUDA numerical coverage pending; official source/build qualification incomplete |

<!-- benchmark-evidence:end -->

**Blocked on periodic engines** (no CP2K-comparable twin in tree): hybrids,
meta-GGA, GAPW/PAW, stock VV10. rVV10 ships.

## 1. Molecular Kohn–Sham

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| RKS / UKS | ✓ | ○ | [scf/rks][scf-rks], [scf/uks][scf-uks]; no Gaussian KS on device |
| ROKS | ✓ | ○ | [scf/roks][scf-roks]: energy + analytic gradient |
| RKS / UKS analytic gradient | ✓ | ○ | Current live matrix (campaign `2026-10-26-dft-wave2`): standard CAM and PBE-family full forces, plus the previous LSDA/PBE/PBE0 envelope; spherical def2-SVP for PBE/PBE0/CAM, including non-symmetric water and empty-spin H₂⁺. The archived external-provider pins cover scaled Slater only; other models remain open |
| External CPU XC provider | ✓ | ○ | [Independent installed project](../../examples/extensions/xc/README.md); batched rho/sigma contract on existing RKS/UKS, energy + first derivatives; the cited force pins exercise rho-only exchange. No injected PW/GPW, ROKS, meta-GGA, hybrid or response kernel support |
| ROKS analytic gradient | ✓ | ○ | [scf/roks][scf-roks], `roks-grad-o2-sto3g`: PySCF full gradient |
| RKS / UKS Hessian | ✓ | ○ | [scf/rks][scf-rks], [scf/uks][scf-uks]: `Hessian<Method>::run`, FD of analytic forces, PySCF FD-gradient pins (not full CPKS) |
| Molecular DFT+U / ΔSCF-DFT / GKS | ○ | ○ | Periodic DFT+U is GPW (§4) |

## 2. Functionals (molecular + shared)

Hand-coded; libxc is an optional test oracle only (`ATV_BUILD_XC_ORACLE`).

| Rung / functional | CPU | CUDA | Notes |
| ----------------- | :-: | :--: | ----- |
| LSDA (Slater+VWN5) | ✓ | ✓ | CPU molecular: [scf/roks][scf-roks]; CUDA PW LDA/spin: [pw/rks][pw-rks], [pw/collinear-spin][pw-spin]; PW LDA has a broad ABINIT-class energy band |
| PW92 correlation base | ◐ | ◐ | Used inside PBE correlation on both backends; no standalone PW92 `XCType` or registered PW92-functional energy pin; `XCType::LSDA` selects VWN5 |
| PBE | ✓ | ✓ | CPU molecular energy: [scf/tddft][scf-td]; GPW: [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]; CUDA PW: [pw/rks][pw-rks], [pw/collinear-spin][pw-spin]; see PW limits below |
| BLYP | ✓ | ○ | [scf/rks][scf-rks], `blyp-water-sto3g`; GPW reachable, not separately pinned |
| PBEsol / revPBE / RPBE | ◐ | ◐ | Current live matrix (campaign `2026-10-26-dft-wave2`): CPU molecular RKS/UKS energies/full forces against live PySCF. Periodic actual-QE/CP2K pins: [pw/pbe-family](../../regressions/pw/pbe-family/REFERENCE.md), [gpw/pbe-family](../../regressions/gpw/pbe-family/REFERENCE.md). Shared-model follow-up: native PW fixed/FD RKS/UKS and GPW RKS energy/full-force/supported-stress twins now pass, including post-Driver acceptance. The five original stage-reference failures are retained and separately repaired. Matched periodic PW follow-up adds CPU FD eight-k RKS/UKS and CUDA Γ fixed-RKS E/full-F/six-stress pins; broader periodic ranges, GPW independent pins and molecular CUDA KS remain open |
| optB88 | ◐ | ○ | GPW (with vdW-DF); no end-to-end pin (cases removed) |
| BP86 / other GGA catalog | ○ | ○ | |
| Meta-GGA (TPSS, r2SCAN, …) | ○ | ○ | No τ channel in grid engine |
| B3LYP | ✓ | ○ | Gaussian only; [scf/rks][scf-rks], `b3lyp-fused-jk-h2o`: energy |
| PBE0 | ◐ | ○ | Current live matrix (campaign `2026-10-26-dft`): public RKS/UKS energies and full analytic forces, STO-3G/def2-SVP. Scoped numerical acceptance passes; formal source/build qualification remains open |
| CAM-B3LYP | ◐ | ○ | Current standard-model evidence: short-range ITYH-B88 and VWN5 replace the old B3LYP semi-local mixture; RKS/UKS energy/full-force pins include spherical def2-SVP. Extreme scalar spin/density parity limitations remain recorded; no periodic or device CAM path |
| Other RSH / screened hybrids | ○ | ○ | |
| B2PLYP | ✓ | ○ | [scf/rks][scf-rks], `b2plyp-h2o`: hybrid KS + scaled RMP2 energy |
| Other double hybrids | ○ | ○ | |
| HF-3c / PBEh-3c / … | ○ | ○ | |

## 3. Dispersion and nonlocal correlation

| Method | Molecular | GPW | CUDA | Notes |
| ------ | :-------: | :-: | :--: | ----- |
| D3 zero | ○ | ✓ | ○ | [gpw/d3][gpw-d3]: E, stress and ATM pins; force/ATM sections still include private headers. GPW ATM stress uses the complete energy strain derivative; the standalone internal D3 virial lacks the direct ATM term |
| D3-BJ | ✓ | ✓ | ○ | Molecular E: [scf/dispersion][scf-disp]; GPW E: [gpw/d3][gpw-d3]; GPW force/stress sections still include private headers |
| D4 | ✓ | ✓ | ○ | Molecular E: [scf/dispersion][scf-disp]; GPW E/F/σ + ATM: [gpw/d4][gpw-d4], dftd4 pins |
| TS | ○ | ✓ | ○ | [gpw/tkatchenko-scheffler][gpw-ts], [gpw/hirshfeld][gpw-hirshfeld]; free-atom periodic E pinned to libMBD; force/stress public-consumer coverage incomplete |
| MBD@rsSCS | ○ | ✓ | ○ | [gpw/mbd][gpw-mbd], [gpw/hirshfeld][gpw-hirshfeld]; libMBD E pins; force/stress public-consumer coverage incomplete |
| XDM | ○ | ◐ | ○ | [gpw/xdm][gpw-xdm]: free-atom path; numerical pins are independent Python reimplementations, not an accepted external oracle; private-header leaks also remain |
| vdW-DF / rVV10 | ○ | ✓ | ○ | [gpw/vdw-df][gpw-vdw]: SC nonlocal potential + F/σ in the recorded envelope |
| vdW-DF2 / optB88-vdW | ○ | ◐ | ○ | Same kernel path; no end-to-end pin (cases removed) |
| VV10 (stock) | ○ | ○ | ○ | **Blocked** — no CP2K `TYPE VV10` |

GPW dispersion has Γ and multi-k, RKS and UKS paths, with uneven evidence.
An energy pin does not close force/stress coverage, and an independent Python
implementation does not qualify as a numerical oracle. See the cases above and
[tutorial/methods catalog](../tutorial/methods/catalog.md).

## 4. Periodic Kohn–Sham

Two engines: **plane-wave** (`cpu::pw::PW` / `cuda::pw::PW`) and **GPW**
(`cpu::gpw::GpwDft` / `GPWEvaluator`). CUDA GPW is a partial product slice
(◐), not full CPU parity.

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Plane-wave DFT | ✓ | ✓ | [pw/rks][pw-rks]; CPU GTH and scalar NC-UPF share one operator path; the independent NC project below has a matched Si₂ energy/full-force pin. CUDA ⊂ CPU |
| External scalar NC provider | ✓ | ○ | [Independent installed project](../../examples/extensions/nc/README.md): immutable radial data/local callback on existing PW; published Si GTH-PBE-q4, absolute Mermin energy and all six nonzero forces against actual QE. Single species, real scalar l≤2; no USPP/PAW/NLCC/spin-orbit |
| GPW DFT | ✓ | ◐ | [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]; CUDA first product slices |
| Γ-point RKS | ✓ | ◐ | [gpw/rks][gpw-rks]; CUDA CPU-twin slice |
| k-point RKS | ✓ | ✓ | [pw/rks][pw-rks] on both; [gpw/rks][gpw-rks] CPU; CUDA GPW remains ◐ |
| Γ / k-point UKS | ✓ | ◐ | CPU: [pw/collinear-spin][pw-spin], [gpw/uks][gpw-uks]; CUDA PW collinear subset, GPW Γ energy/force slices; GPW multi-k UKS ○ |
| Periodic LDA / GGA | ✓ | ✓ | Cases above; no periodic hybrid / meta-GGA |
| Periodic DFT+U | ◐ | ○ | [gpw/dft-plus-u][gpw-u]: Γ/multi-k E/F/σ paths; ATV gross-Mulliken and CP2K net-Mulliken differ, so the recorded loose comparisons do not qualify a matched-model pin |
| Periodic dispersion | ◐ | ○ | See §3: method-dependent E/F/σ evidence; XDM oracle and public-consumer gaps remain |
| Periodic RI-MP2 / SCS-MP2 (Γ) | ✓ | ○ | [gpw/periodic-mp2][gpw-mp2]: CP2K correlation-energy pins; declared CPU sections rechecked |
| Periodic UMP2 / multi-k MP2 | ◐ | ○ | GPW; no external pin, self-pinned cases removed |
| Periodic dRPA / URPA | ◐ | ○ | Γ, multi-k, finite-q reachable; no external pin, self-pinned cases removed |
| Periodic G0W0 / UGW | ◐ | ○ | Γ, multi-k, finite-q reachable; no external pin, self-pinned cases removed |
| GAPW / PAW / USPP | ○ | ○ | NC-UPF or GTH only |

### Plane-wave depth (CPU)

| Capability | Status | Notes |
| ---------- | :----: | ----- |
| Analytic ionic forces | ✓ | [pw/rks][pw-rks]: local HF + Ewald symmetry gate; [external NC project](../../examples/extensions/nc/README.md): local + full off-diagonal KB, all six nonzero Si₂ force components against QE at 1e-4 Ha/Bohr. Broader elements/potentials remain unqualified |
| Cell stress (one-SCF frozen-density strain FD) | ✓ | Current public diagnostic: fixed reference basis, physical shear and shared-μ spin/smearing free energy at strain step 2e-3. Six Γ reference-cell E/σ QE pins pass; five independent full-SCF deformation checks pass, with the LSDA-MP compressed sample explicitly failed. Not a closed-form tensor |
| Closed-form analytic cell stress | ◐ | Fixed/FD/Gaussian scoped wave: Γ and supported general-k semilocal RKS/populated UKS share one final-band tensor, with serial/MPI2 and ARM checks. MP1 PBE Γ populated UKS and restricted Si2 additionally pass their declared signed-occupation envelopes. Fixed per-k band indices do not imply a global insulating gap. Other MP and nominally empty fixed/Gaussian channels retain numerical paths; Γ fixed-LSDA Na>0/Nb0 has its own zero-channel tangent qualification |
| True GGA PBE + collinear spin-GGA | ✓ | [pw/rks][pw-rks]: symmetry/activation gates; [pw/collinear-spin][pw-spin]: O₂ absolute energy pinned to an actual QE 7.5 run, alongside magnetization and GGA activation |
| Metallic smearing | ✓ | Current shared implementation: FD/Gaussian/MP occupation, entropy and chemical-potential pins from real QE 7.6; nine variational/endpoint units. Full-SCF stress and response have separate qualification limits |
| Bands / DOS | ✓ | [pw/bands-dos][pw-bands]; includes spin + metallic sections |
| CELL_OPT | ✓ | [pw/variable-cell][pw-cell]; reference states the geometry/energy comparison limits |
| BOMD | ✓ | [dynamics/bomd][bomd], `pw-bomd-h`: foothold |
| Berry / MLWF | ✓ | [pw/berry-phase][pw-berry], [pw/wannier][pw-wannier]; includes spin |
| DFPT phonons | ◐ | [response/dfpt][dfpt]: historical Γ / finite-q / incommensurate × GGA × spin × smear. The public interface now rejects MP response before SCF: signed quasi-occupations are not supported by its occupation thresholds/projectors. Fixed occupations, FD and Gaussian retain their existing scope; historical MP checks do not qualify the corrected model |

### Plane-wave depth (CUDA)

| Capability | Status | Notes |
| ---------- | :----: | ----- |
| LSDA + PBE energy | ✓ | Current GH200 diagnostic: shared CPU/device padded XC quadrature; builtin GTH Si₂ PBE/Γ/FD-smearing Mermin energy pinned to actual QE 7.5. Broader LSDA/PBE evidence remains the archived CPU-twin/symmetry envelope |
| Analytic forces + cell stress (closed-shell) | ◐ | Current scoped wave: Γ fixed/FD LSDA/PBE-family has certified device force/tensor assembly and matched actual-QE full-force/six-stress pins. The new FD scope has five matched actual-QE references; broader spin/smear/k settings remain separate |
| Collinear spin forces / cell stress | ✓ | [pw/collinear-spin][pw-spin]; includes spin-PBE CPU twins; stress uses NSCF strain FD; general external-vector qualification remains open |
| Closed-form analytic cell stress | ◐ | Γ RKS fixed/FD LSDA/PBE-family and fixed nonempty UKS pass their matched-QE/public/native scopes. The UKS owner shares independent spin band sums and one publication with the current joint mixer; finite unsupported XC fields retain numerical epilogues. UKS FD, Gaussian/MP, multiple k points and empty endpoints remain open |
| Metallic smear energy + F/σ | ◐ | Current Γ RKS FD LSDA/PBE-family has matched actual-QE E/full-F/six-stress pins and closed-form tensors. Earlier spin/Gaussian/MP settings retain their separate numerical-stress and qualification limits |
| Host Ritz + SCF control | — | Hot path (H·ψ, FFT, Poisson, V_eff) on device |

### GPW depth (CPU)

RKS/UKS, Γ/Monkhorst–Pack and DIIS/OT paths exist. The registered sections
validate specific combinations; they do not establish every Cartesian product
of solver, functional, spin, grid, dispersion and k-mesh.

| Area | Status | Notes |
| ---- | :----: | ----- |
| Functionals | ✓ | [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]: PADE/LSD, PBE; other reachable functionals lack GPW pins (§2) |
| DFT+U | ◐ | [gpw/dft-plus-u][gpw-u], [gpw/orbital-transformation][gpw-ot]: projector mismatch; OT reachability does not close external validation |
| Smearing / multigrid / OT | ✓ | [gpw/rks][gpw-rks], [gpw/uks][gpw-uks], [gpw/orbital-transformation][gpw-ot]; only the recorded combinations |
| Analytic forces + fixed-P strain-FD stress | ◐ | Public stress is an orthorhombic diagonal strain difference with zero shear; it is not a closed-form tensor. [gpw/rks][gpw-rks], [gpw/uks][gpw-uks] retain their quantity-specific historical pins; broader k/spin/metallic depth remains open |
| Dispersion / vdW | ◐ | §3: energy pins for several methods; XDM oracle and public force/stress coverage remain open |
| Bands / DOS / Berry / MLWF | ✓ | [gpw/bands-dos][gpw-bands], [gpw/berry-phase][gpw-berry], [gpw/wannier][gpw-wannier]; recorded spin/metallic sections |
| BOMD / CPMD / SGCP / XL-BOMD | ✓ | [dynamics/bomd][bomd], [dynamics/car-parrinello][cpmd], [dynamics/second-gen-cpmd][sgcp], [dynamics/xl-bomd][xlbomd]; see [MD § AIMD](molecular-dynamics.md#5-ab-initio-md) |
| Periodic correlated | ◐ | RI-MP2 / SCS-MP2 at Γ pinned; RPA, GW, UMP2, multi-k not (§4 table) |

**Public shape:** `GpwDft` is a `Driver<gpw::Phase>`, with kinds/cutoff/XC as
constructor identity and Frame `run` / `forces` / `stress`. Born–Oppenheimer
work mounts the method directly on the parent Driver; forces/stress follow
the parent's requests and use the converged state. `GPWEvaluator` remains for
electronic propagators and unmigrated callers. DIIS and all six OT outer
paths now share the method's typed Driver loop; bounded line searches and
fixed-H optimizers remain internal numerical kernels. Keep
`local_pp_on_grid` consistent between energy and gradient calls.

### CUDA GPW

| Capability | Status | Notes |
| ---------- | :----: | ----- |
| Γ RKS DIIS energy + F/σ slice | ◐ | [gpw/rks][gpw-rks]: CPU twins; host AO Pulay / stress residual; no independent CP2K qualification |
| Γ UKS energy / force slice | ◐ | [gpw/uks][gpw-uks]: Padé LSD CPU twins; full F/σ and PBE depth open |
| Multi-k RKS energy | ◐ | [gpw/rks][gpw-rks]: H₂ foothold + device mid-SCF pieces; setup still hybrid |
| Full method parity with CPU | ○ | Table stays ◐ until production depth + external pins |

## 5. Periodic response and analysis

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Periodic analytic forces | ✓ | ◐ | [pw/rks][pw-rks], [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]; quantity limits in §4 |
| Periodic cell stress | ◐ | ◐ | PW has scoped closed-form CPU semilocal fixed/FD/Gaussian and CUDA Γ fixed/FD LSDA/PBE-family tensors (§4); other PW domains retain numerical paths. GPW public stress remains fixed-P orthorhombic diagonal strain FD; CUDA GPW uses host assembly |
| DFPT | ✓ | ○ | [response/dfpt][dfpt], PW only |
| FD phonons / ω(q) | ✓ | ○ | [response/phonons][phonons]: classical SW supercell/dispersion pins; isolated GPW Si₂ energy/full forces, stationary geometry and stretch pinned to actual CP2K 2026.2 outputs, with five unprojected near-zero modes |
| Berry polarization | ✓ | ○ | [pw/berry-phase][pw-berry], [gpw/berry-phase][gpw-berry]; spin |
| MLWF | ✓ | ○ | [pw/wannier][pw-wannier], [gpw/wannier][gpw-wannier]; spin |

## 6. Periodic infrastructure

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Monkhorst–Pack k-mesh | ✓ | ✓ | [pw/rks][pw-rks], [gpw/rks][gpw-rks] |
| Block Davidson | ✓ | ✓ | [pw/rks][pw-rks]; exercised inside the method, no standalone product claim |
| NC-UPF (KB) | ✓ | ✓ | [pw/rks][pw-rks]; energy path; KB nonzero-force external pin open |
| GTH PP | ✓ | ◐ | CPU PW uses the public NC protocol and canonical radial operators; [external Si₂ project](../../examples/extensions/nc/README.md) pins E/full F. CUDA builtin Si₂ now has a matched QE PBE/Γ/smearing E/full-F pin. CPU GPW retains its Gaussian-projector path; CUDA GPW remains partial |
| FFT Poisson | ✓ | ✓ | [pw/rks][pw-rks], [gpw/rks][gpw-rks] |
| Martyna–Tuckerman Poisson | ✓ | ○ | [gpw/rks][gpw-rks], `gpw-water-pbe-mt` |
| OT minimizer | ✓ | ○ | [gpw/orbital-transformation][gpw-ot]; CPU DIIS/OT now share typed Driver execution, with current scoped lifecycle/MPI checks above. CUDA OT remains unsupported |
| Multigrid collocation | ✓ | ◐ | [gpw/rks][gpw-rks], [gpw/uks][gpw-uks]; CUDA MG energy CPU-twin slice exists |
| Metallic smearing | ✓ | ◐ | [pw/smearing][pw-smear], [gpw/rks][gpw-rks]; current CUDA PW Γ fixed/FD restricted/collinear E/F/strain-FD stress passes. Broader settings and CUDA GPW metallic support remain open |
| Bands / DOS post-process | ✓ | ○ | [pw/bands-dos][pw-bands], [gpw/bands-dos][gpw-bands] |

## 7. Validation gaps

The entries distinguish observed numerical failures from qualification and
coverage gaps.

| State | Scope | Evidence / remaining check |
| ----- | ----- | -------------------------- |
| Historical CPU baseline (2026-10-25) | Registered Gaussian, PW, GPW and composed workflows | 1785 units and all 68 enabled cases passed; all 301 reported section times were ≤60 s (maximum 52.4 s). The independent installed XC project passed RKS/UKS energies, full forces and lifecycle checks after relocation. CPU only; CUDA still requires GH200 validation. |
| Pending broader qualification | Molecular functional/basis/system range; GPW functional variants | The current live matrix closes specific RKS/UKS energy/full-force combinations, including PBE0 and spherical def2-SVP. Other systems, functionals, providers and GPW variants do not inherit those passes. |
| Pending matched-model qualification | DFT+U | Gross-Mulliken ATV occupation differs from CP2K's net-Mulliken block. Loose NiO comparisons and ATV finite differences do not close this gap. |
| Verified model repair, limited scalar boundary parity | Standard CAM-B3LYP | New standard-model evidence passes molecular E/full F and normal-domain scalar/ITYH checks. The original four failed standard-CAM comparisons remain historical records. Three extreme scalar points still differ in underlying B88/LYP/VWN components; they are not marked passed. |
| Pending oracle replacement | XDM | Literature supplies formulas; the numerical pins are Python reimplementations. Obtain numbers printed by an independent established engine for the same free-atom model. |
| Partial algorithm coverage | Closed-form PW cell stress, CPU and CUDA | Scoped waves qualify CPU Γ/supported general-k semilocal fixed/FD/Gaussian RKS/populated UKS, Γ fixed-LSDA empty-beta tangent and Γ PBE MP1 RKS/populated UKS; CUDA Γ RKS fixed/FD and fixed nonempty UKS LSDA/PBE-family have their own native evidence. Other MP, further empty-spin domains and CUDA UKS FD/Gaussian/MP/k remain open; numerical NSCF strain gates do not qualify closed-form implementations. |
| Missing response algorithm | Methfessel–Paxton DFPT | Correct MP ground-state quasi-occupations can be negative or exceed their spin degeneracy. DFPT occupation thresholds and occupied-space projectors assume probability weights; the public phonon interface rejects MP at all q and spin settings before SCF. Ground-state E/F/σ qualification does not extend to response. |
| Unclosed convergence boundary | LSDA × MP signed density near zero | The H₃ MP1/.04-Ry simple −xx strain sample fails joint density tol 1e-9 at both 250 and 1000 iterations. A quadrature spin density crosses zero and its clipped-functional potential jumps between about −.0444 Ha and zero. Inner Ritz residuals meet 1e-10; 716 exact-spectrum QE replays and tighter-root checks exclude a chemical-potential branch/precision defect. This independent full-SCF FD row remains failed. Removing the potential clip would break the existing energy derivative; a consistent XC extension or constrained solver is still required. |
| Verified repaired slice | Molecular spherical XC forces and range-separated integrals | CPU evidence: correct Cartesian density dual, erf Rys-root mapping and analytic derivatives, and the missing UKS long-range term. The follow-up standard-CAM repair has its own separate evidence; the earlier native-composition FD checks do not substitute for standard-functional pins. |
| Verified repaired slice | CUDA PW XC quadrature and derivative results | GH200 evidence: 37 actual units, public result/reuse checks, builtin Si₂ QE E/full-F pin, original-source negative control and zero-error padded-stage memcheck. Private LSDA/PBE quadrature now matches CPU; force/stress results carry their own SCF status. Spin×smearing stress rejects the missing occupation derivative path. No GPW or general device-residency qualification follows from these checks. |
| Pending public-consumer coverage | GPW dispersion forces/stress | 17 CPU section sources across D3, TS, MBD, Hirshfeld and XDM still include `internal/`. Their energy pins do not certify the whole F/σ surface. |
| Verified repaired slice | GPW finite-displacement Si₂ phonons | [response/phonons][phonons], `gpw-fd-phonon-si`: actual MT Poisson and matching Gaussian-product density/force contractions; CP2K stationary input replaces the nonstationary frequency fixture. Original energy/full-force, five raw modes below 80 cm⁻¹, stretch and 60 s bounds retained. |
| Verified repaired slice | GPW UKS forces / O₂ stress | Coupled-spin DIIS, energy-only XC for strained energies and reuse of the accepted SCF state remove redundant work. Both existing sections pass their numerical gates within 60 s at the declared 8 threads; this is section-runtime evidence, not a matched-engine speedup claim. |
| Rechecked declared CPU envelope | Γ RI/SCS-MP2, UKS+U/OT repeatability, second-gen CP OT | Surviving registered sections pass their original numerical and runtime gates. This does not close the DFT+U projector-definition mismatch or qualify unregistered settings. |
| Verified PW XC/response repair | GGA quadrature, joint-spin SCF and finite-q C(q) | Historical captures and current scope: fixed 3/2 LDA and 3x GGA quadrature with adjoint projection, shared Pulay mixing for ground-state/response solves, true Fourier C(q), and exact-zero empty-channel density. Current 14-case recheck retains the original science and 60 s section bounds, with no force/ASR projection. |
| Partial device product | CUDA GPW; broader CUDA PW metallic/spin settings | GPW public methods now use Frame/Kind and typed Driver execution; remaining scientific-state transfers and host force/stress assembly have separate residency tasks. CPU-twin sections do not qualify every physical combination. |
| Blocked by missing engine work | Periodic hybrids, meta-GGA, GAPW/PAW | No matched implemented periodic path; meta-GGA also lacks a τ channel. Stock VV10 has no CP2K `TYPE VV10` twin. |

The PW references also contain matched Quantum ESPRESSO timing/output records.
Those output values are not automatically correctness gates: e.g.
`pw-pbe-gga-si` still asserts symmetry and GGA activation. GPW timing records
affected by OpenMP binding require an unbound re-measurement before quoting a
current performance gap. No new performance qualification is implied here.

[scf-rks]: ../../regressions/scf/rks/REFERENCE.md
[scf-uks]: ../../regressions/scf/uks/REFERENCE.md
[scf-roks]: ../../regressions/scf/roks/REFERENCE.md
[scf-td]: ../../regressions/scf/tddft/REFERENCE.md
[scf-disp]: ../../regressions/scf/dispersion/REFERENCE.md
[pw-rks]: ../../regressions/pw/rks/REFERENCE.md
[pw-spin]: ../../regressions/pw/collinear-spin/REFERENCE.md
[pw-smear]: ../../regressions/pw/smearing/REFERENCE.md
[pw-bands]: ../../regressions/pw/bands-dos/REFERENCE.md
[pw-cell]: ../../regressions/pw/variable-cell/REFERENCE.md
[pw-berry]: ../../regressions/pw/berry-phase/REFERENCE.md
[pw-wannier]: ../../regressions/pw/wannier/REFERENCE.md
[gpw-rks]: ../../regressions/gpw/rks/REFERENCE.md
[gpw-uks]: ../../regressions/gpw/uks/REFERENCE.md
[gpw-u]: ../../regressions/gpw/dft-plus-u/REFERENCE.md
[gpw-ot]: ../../regressions/gpw/orbital-transformation/REFERENCE.md
[gpw-d3]: ../../regressions/gpw/d3/REFERENCE.md
[gpw-d4]: ../../regressions/gpw/d4/REFERENCE.md
[gpw-ts]: ../../regressions/gpw/tkatchenko-scheffler/REFERENCE.md
[gpw-mbd]: ../../regressions/gpw/mbd/REFERENCE.md
[gpw-hirshfeld]: ../../regressions/gpw/hirshfeld/REFERENCE.md
[gpw-xdm]: ../../regressions/gpw/xdm/REFERENCE.md
[gpw-vdw]: ../../regressions/gpw/vdw-df/REFERENCE.md
[gpw-mp2]: ../../regressions/gpw/periodic-mp2/REFERENCE.md
[gpw-bands]: ../../regressions/gpw/bands-dos/REFERENCE.md
[gpw-berry]: ../../regressions/gpw/berry-phase/REFERENCE.md
[gpw-wannier]: ../../regressions/gpw/wannier/REFERENCE.md
[bomd]: ../../regressions/dynamics/bomd/REFERENCE.md
[cpmd]: ../../regressions/dynamics/car-parrinello/REFERENCE.md
[sgcp]: ../../regressions/dynamics/second-gen-cpmd/REFERENCE.md
[xlbomd]: ../../regressions/dynamics/xl-bomd/REFERENCE.md
[dfpt]: ../../regressions/response/dfpt/REFERENCE.md
[phonons]: ../../regressions/response/phonons/REFERENCE.md
