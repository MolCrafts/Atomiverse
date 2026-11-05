# Semiempirical and Tight-Binding Methods

Tutorial: [Semiempirical models](../tutorial/ch17-semiempirical-am1.md).
This chapter records the current `dev` implementation and executed evidence as
of 2026-10-25; it is not a claim about the released `master` snapshot.

CPU AM1/RM1/MNDO/PM3 provide molecular closed-shell energy, charges, heat of
formation and analytic forces. Native CUDA versions of all four methods
pass external E/q/ΔHf/full-force and CPU comparisons for neutral closed-shell
CHNO molecules in FP64. CUDA Evaluate, short BOMD and three-step steepest-descent
composition are verified within the limits below. This is a method-by-method
implementation, not complete AmberTools SQM feature or input-file compatibility. Oracles:
**OpenMOPAC** (NDDO), **DFTB+** 24.1 (DFTB), **xtb 6.7.1** (GFN/IPEA). Legend: [Roadmap index](index.md).

## 1. NDDO

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| AM1 | ✓ | ✓ | CPU molecular closed-shell E/q/ΔHf/analytic F; CUDA E/q/ΔHf/analytic F restricted to neutral closed-shell CHNO in FP64. Broader existing CPU s/p coverage is retained; external full-force pins cover CHNO |
| RM1 | ✓ | ✓ | CPU molecular closed-shell E/q/ΔHf/analytic F; shipped table is H/C/N/O/P/S, not all published RM1 elements. CUDA E/q/ΔHf/analytic F is neutral closed-shell CHNO FP64 only |
| NDDO one-center primitives | ✓ | ✓ | CHNO AM1/RM1 FP64: shared parameters, Hcore/Fock, charges and electronic-energy reduction; primitive checks and complete `semiempirical/am1` external/CPU comparisons pass on GH200. Native SCF SE-only linkage passes without HF/PW/GPW/MD/Minimize |
| NDDO two-center primitives | ✓ | ✓ | CHNO AM1/RM1 FP64: analytic-STO Hcore, core repulsion, ERI, direct Fock and analytic pair derivatives without a global ERI cache; CPU comparisons, GH200 memcheck and racecheck pass |
| MNDO | ✓ | ✓ | `semiempirical/mndo`: neutral closed-shell CHNO E/q/ΔHf/analytic F, native composition and Python pass CPU/GH200 external and cross-backend gates; no heavier-element pair corrections |
| PM3 | ✓ | ✓ | `semiempirical/pm3`: neutral closed-shell CHNO E/q/ΔHf/analytic F, native composition and Python pass CPU/GH200 external and cross-backend gates; no heavier-element pair corrections |
| PDDG/MNDO, PDDG/PM3, PDDG/PM3-08 | ○ | ○ | Method-specific fitted parameters, corrections and external E/q/full-force qualification remain open; base MNDO/PM3 acceptance does not cover these variants |
| PM3-CARB1 / ZNB / MAIS | ○ | ○ | No native variant-specific parameter/correction implementation or acceptance |
| AM1/d / MNDO/d | ○ | ○ | Required d-orbital parameter and integral/derivative paths remain open |
| PM6 / PM7 | ○ | ○ | No public method or validated parameter/gradient path |
| OM3 | ○ | ○ | **Blocked** — no OpenMOPAC primary surface |
| Nuclear forces (CHNO molecules) | ✓ | ✓ | `semiempirical/am1`, `semiempirical/mndo`, `semiempirical/pm3`: all four methods' distorted H₂O, tilted CO and NH₃ full vectors against executed OpenMOPAC 23.2.5 |
| Analytic integral-derivative forces | ✓ | ✓ | `semiempirical/am1`: analytic pair derivatives with independent CHNO force pins; broader s/p elements have derivative consistency tests only |
| Evaluate composition | ✓ | ✓ | Standalone/mounted E/q/F agreement, additive contributions and failure/retry; CUDA restricted to the molecular CHNO envelope |
| Short BOMD / steepest-descent composition | ✓ | ✓ | `semiempirical/am1`: both methods pass initial-force preparation, CUDA 2+2 versus 4 steps and CPU/CUDA four-step agreement at 0.01 fs; three SD steps lower energy, without a minimization-convergence claim |
| NVE energy conservation and time-step refinement | ✓ | ✓ | Each method case has `nddo-nve`: 10 fs distorted-water trajectories at 0.1/0.05 fs, sampled every 0.5 fs; external initial energy, absolute E+K deviation, approximately fourfold improvement and CPU/CUDA short-trajectory checks pass |
| Accepted electronic-history transactions | ✓ | ✓ | CPU/CUDA AM1/RM1 actual Armijo accept/reject/exception, final-step commit and later-parent-component failure/retry pass; CUDA complete composed force checks also pass memcheck |
| Native Python AM1/RM1/MNDO/PM3 | ✓ | ✓ | Same-instance run/forces/Evaluate/MD/Minimize, input edits/retry and twelve independent external E/q/ΔHf/full-force fixtures pass on CPU and GH200. All 53 interface checks pass in the CUDA build. Python CUDA exposes VelocityVerlet and steepest descent; the CPU workflow gate uses LBFGS |
| Host edits between MD chunks | ✓ | ✓ | Direct AM1/RM1/MNDO/PM3+VelocityVerlet continuation passes coordinate/velocity and count/type edits, plus failed-refresh retry. CPU box refresh and CUDA unsupported-box rejection/restore pass. MPI change/error propagation passes two-rank units. The same-owner Python checks pass on both backends |
| FD Hessian via `vib::Hessian` | ✓ | ○ | Existing CPU AM1 workflow; new MNDO/PM3 and CUDA Hessian workflows have no separate acceptance |
| Heat of formation (MOPAC-style) | ✓ | ✓ | All four methods' executed OpenMOPAC pins; CUDA restricted to neutral closed-shell CHNO |
| NDDO UHF / COSMO / Γ-PBC | ○ | ○ | Molecular AM1 under wrap unwraps an isolated fragment |
| CPU NDDO SCF acceleration and input guard | ✓ | ○ | Pulay DIIS (six vectors, residual FP−PF), two-center blocks cached once per geometry (OpenMP build, 2 GiB budget), BLAS-3 pseudo-diagonalization once the commutator is below 0.05 eV, BLAS `dsyrk` density, OpenMP Fock and force pair loops; compensated energy sum. Atom pairs closer than 0.6 × the covalent-radius sum are refused before the SCF. 205-atom PBIX-Br trimer and 674-atom chain vs AmberTools SQM: [`benchmarks/compare/am1-sqm`](../../benchmarks/compare/am1-sqm/README.md); CUDA NDDO keeps its own loop and has no contact guard |

## 2. DFTB

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| DFTB0 | ✓ | ○ | Analytic nuclear forces (HF + Pulay + repulsive); no SCC γ |
| DFTB2 (SCC) | ✓ | ○ | mio-1-1 + 3ob-3-1 (H/C/N/O); `lattice` γ is Ewald |
| DFTB3 | ◐ | ○ | Energy + analytic F + Γ-PBC F are implemented, but present gates are internal snapshots/FD consistency, not an external DFTB3 Hamiltonian match; no k / stress / spin / 3ob / D3 |
| DFTB3 γ^h (DampXH) | ○ | ○ | ζ = 4 X–H damping; MioSketch DFTB3 runs undamped |
| DFTB3 Ewald γ | ○ | ○ | Periodic γ cut at the SK sphere; DFTB2 γ is Ewald |
| Collinear spin-DFTB2 | ✓ | ○ | MioSketch HOAO (`ShellResolvedSpin = No`); ThreeOb open shell throws |
| Shell-resolved collinear spin | ○ | ○ | DFTB+ `ShellResolvedSpin = Yes` throws |
| DFTB+U | ✓ | ○ | Equal-filling energy; analytic F and per-shell \(U_\ell\) still ○ |
| D3 dispersion (mol + Γ-PBC) | ✓ | ○ | DFTB2 two-body BJ, orthorhombic PBC. ATM, triclinic cells, and D4 are ○ |
| DFTB-D3 three-body (ATM) | ○ | ○ | `calculate_c9=false`; DFTB+ / s-dftd3 ATM |
| DFTB-D3 triclinic cell | ○ | ○ | `d3_periodic` throws unless the box is orthorhombic |
| SK beyond H/C/N/O | ○ | ○ | mio-1-1 ships P and S; in-tree tables are H/C/N/O and throw |
| On-site correction | ○ | ○ | DFTB+ `OnSiteCorrection` |
| Γ-point periodic DFTB2 | ✓ | ○ | Ewald γ energy pinned vs DFTB+; analytic F/σ on MioSketch (periodic σ unit-gated); ThreeOb is molecular only |
| Multi-k periodic DFTB2 | ◐ | ○ | Reachable; no external pin (self-pinned k-point cases removed) |
| DFTB2 analytic gradient / stress | ✓ | ○ | Supported molecular/periodic variants; +U derivatives remain unavailable; external multi-k and periodic-stress evidence is limited as above |
| TD-DFTB | ✓ | ○ | Closed-shell Γ MioSketch excitation energies; analytic zero-coupling/transition-charge gates, no absolute external excitation pin. Oscillator strengths, excited-state gradients, 3ob and collinear-reference Casida remain open. See [Excited States](excited-states.md) |
| LC-DFTB / Fermi filling / GBSA | ○ | ○ | |

## 3. Extended tight binding (xTB)

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| GFN0 / GFN1 | ✓ | ○ | H/C/N/O; analytic G; FD Hessian; integer occ; existing xtb cases pass |
| GFN2 | ◐ | ○ | H/C/N/O energy, analytic G and FD Hessian implemented; existing xtb cases pass, but tight external full-force-vector acceptance remains open |
| GFN2 AES (quadrupole) + self-consistent D4 | ◐ | ○ | AES and D4 are present in SCC/Fock and gradient paths; current component/FD checks do not close the full external gradient evidence gap |
| IPEA-xTB | ✓ | ○ | Energy + G + FD Hessian + ΔSCC VIP/VEA; CHNO only |
| Periodic / spin / ALPB / GBSA / CPCM-X / Fermi / sTDA-xTB | ○ | ○ | `--gbsa` is not ALPB. Extra-Z tables too |
| GFN-FF | ○ | ○ | Classical FF scope-out |

### Verification snapshot and remaining acceptance gates

The 2026-10-24 CPU SE+MD+Minimize build passed 1,299 unit checks and all
16 selected semiempirical/minimizer/MD regressions. Python passed 49 interface
checks (four CUDA-only skips) and the AM1/RM1/MNDO/PM3 external/composition
scripts, covering twelve independent molecular fixtures. The isolated
two-rank MPI build passed the new input-change and existing Criterion gates.
GH200 passed 113 SE-only CUDA checks, then 229 checks in the combined build
with thirteen multi-GPU skips. The complete composed AM1/RM1 force section
and each new MNDO/PM3 method section passed memcheck with zero errors.
Affected MD, RHF and minimizer regressions passed. CUDA Python passed all
53 interface checks and all twelve external fixtures,
plus each method's same-owner MD input edits/retry, initial forces, split
continuation and short Minimize.
The earlier SE-disabled HF/MD/CUDA build also passed build/link, thirteen
shared execution checks and the RHF/MD initial-force sections. A passing case
means its current assertions passed, not that every capability of the named
reference program is implemented.

| Method / case | Executed evidence | Main remaining gate |
| --- | --- | --- |
| AM1 | `semiempirical/am1` passed, including six AM1/RM1 CHNO energy/q/ΔHf/full-force fixtures; CPU/CUDA transaction and direct MD input boundaries pass, with CPU/CUDA Python composition | Broader-element external forces, larger-system performance and full host-allocator coverage |
| RM1 | `semiempirical/rm1` plus the RM1 fixtures in `semiempirical/am1` passed; CPU/CUDA transaction and direct MD input boundaries pass, with CPU/CUDA Python composition | Remaining parameter coverage, larger-system performance and full host-allocator coverage |
| MNDO | `semiempirical/mndo`: three external E/q/ΔHf/full-force pins, Frame/Evaluate, history/line-search, MD inputs and Python composition pass | Broader elements/variants and resource/performance qualification beyond the measured clusters |
| PM3 | `semiempirical/pm3`: three external E/q/ΔHf/full-force pins, Frame/Evaluate, history/line-search, MD inputs and Python composition pass | Broader elements/variants and resource/performance qualification beyond the measured clusters |
| AM1-BCC / ABCG2 | `semiempirical/am1-bcc` and Python acetate checks passed | A charge-correction workflow, not a separate SQM Hamiltonian or CUDA implementation |
| DFTB0 | `semiempirical/dftb0` passed | Native CUDA implementation and CPU/CUDA/oracle acceptance |
| DFTB2 | `semiempirical/dftb2` passed; concrete SCC Driver and Frame entry points are present | Native CUDA implementation; multi-k external pins and the variant limits above |
| DFTB3 | `semiempirical/dftb3` passed | Standard parameter/Hamiltonian closure and external E/F pins; current mixed-set snapshots are not DFTB+ values |
| GFN0 | `semiempirical/gfn0-xtb` passed | CUDA and further element/variant coverage |
| GFN1 | `semiempirical/gfn1-xtb` passed | CUDA and further element/variant coverage |
| GFN2 | `semiempirical/gfn2-xtb` passed | Resolve earlier failure records against reproducible inputs; close external full-vector force accuracy before refactoring or CUDA promotion |
| IPEA | `semiempirical/ipea-xtb` passed | CUDA and further element/variant coverage |
| TD-DFTB | `semiempirical/td-dftb` passed | Absolute external excitation evidence and response properties beyond excitation energies |

GFN2's existing `gfn2-xtb-forces` section compares energy at `1e-2 eV`, one
external force component at `0.35 eV/Å`, and two energy-FD components at
`0.25 eV/Å`. Passing these unchanged gates does not establish tight agreement
of the complete force vector. DFTB3's current mio-style/3ob-derivative mixed
parameter check is explicitly an internal snapshot. These evidence limits,
rather than absence of the code paths, account for the partial markers.

For NDDO, the [executed OpenMOPAC fixture record](../../tools/tb_oracle/inputs/mopac/live-23.2.5/README.md)
contains inputs, outputs, versions, hashes, extraction and independent oracle
gradient qualification. CPU and CUDA full forces pass for distorted H₂O,
tilted CO and NH₃ for both methods, together with energy, charge and heat of
formation comparisons. CUDA standalone and mounted Evaluate E/q/F agree;
additive contributions, input changes and failure/retry are covered. The
public force regression also passed memcheck with zero reported errors. Pair
derivative checks passed 31 targeted unit checks and memcheck; 18 two-center
checks passed racecheck. Native SCF and public-force linkage passed with HF/PW/GPW/MD/Minimize
disabled. A separate combined build passed both methods' initial-force
preparation, CUDA MD 2+2 versus four-step continuation and CPU/CUDA four-step
agreement at `dt = 0.01 fs`. Three steepest-descent steps lowered AM1 energy
from `-348.562007416316` to `-348.562025788489 eV` and RM1 energy from
`-345.569010474777` to `-345.569076813789 eV`. This demonstrates short workflow
composition, not converged optimization or NVE energy conservation.
The separate `nddo-nve` sections now verify 10 fs energy conservation and
time-step refinement for all four methods; longer-time stability is not established.
CPU/CUDA AM1/RM1 now pass actual Armijo acceptance, rejection and exception recovery,
including a later parent component failing after the method succeeds. The
transaction protects accepted electronic guesses; a method's `last_result`
still describes its most recent child evaluation, independently of whether
the complete parent operation subsequently succeeds.

MNDO and PM3 each have their own method case and three
[executed OpenMOPAC fixtures](../../tools/tb_oracle/inputs/mopac/mndo-pm3-23.2.5/README.md).
Both pass E/q/ΔHf/full-force pins, rigid transformations, Frame overwrite versus
mounted accumulation, normal/trial/initial history behavior, failed-parent retry,
short MD continuation with host input edits, and actual native line-search
composition on CPU and CUDA. Their complete method sections pass memcheck;
Python independently checks the same external fixtures and workflow boundaries.
The unique fitted tables feed the existing backend-owned NDDO kernels and
workspace; the new public methods expose physical results and keep solver
indexing/matrices private. This does not extend MNDO/PM3 beyond neutral
closed-shell CHNO or establish performance, long NVE stability or optimization
convergence.

The four `nddo-nve` sections pass locally and on GH200 with the same nonzero,
zero-total-momentum initial velocities, O/H masses of 16/1 amu and SCF residual
limits of `1e-10` (density) and `1e-12 eV` (energy). Both time steps integrate
10 fs and sample `E + K` every 0.5 fs. The independently chosen limits are
1 meV per molecule maximum sampled absolute deviation and a fine/coarse ratio
at most 0.4. Initial potential energy remains externally pinned at `1e-6 eV`;
kinetic energy is recomputed from host velocities at every sample.

| Method | CPU maximum deviation at 0.05 fs (eV) | CUDA maximum deviation at 0.05 fs (eV) | Fine/coarse ratio (CPU / CUDA) |
| --- | ---: | ---: | ---: |
| AM1 | 2.53708026e-6 | 2.53708015e-6 | 0.249820 / 0.249820 |
| RM1 | 2.43066472e-6 | 2.43066472e-6 | 0.249794 / 0.249794 |
| MNDO | 7.36875677e-6 | 7.36875688e-6 | 0.250000 / 0.250000 |
| PM3 | 3.02258991e-6 | 3.02259002e-6 | 0.249691 / 0.249691 |

Matching-time E+K/kinetic energies and final positions, velocities, charges and
forces pass the stated CPU/CUDA tolerances in each method's `REFERENCE.md`.
Each combined CPU/CUDA NVE section completes in 0.7–1.5 seconds on the tested
GH200 allocation. These are acceptance runtimes for tiny systems, not a
performance comparison. The local 16-case affected regression selection also
passes. No continuous-time error bound, long-time trajectory agreement or
force-path device-residency claim follows from these sampled trajectories.

### Current force resources and performance

The [four-method force benchmark](../../benchmarks/perf/semiempirical/nddo_forces/README.md)
now records CPU/CUDA complete forces at 3/12/24/48 atoms, cold and independently
warmed fixed/moving geometry, with exact executable/source hashes, controls,
per-call SCF counts, full standalone E/q/F and final resident vectors. Six
alternating before/after runs used one GH200 allocation and one CPU thread;
profiled runs are separate. This is prescribed geometry, not BOMD throughput.

CPU reuses the existing two density buffers. On both the local CPU and GH200,
720 allocation-instrumented calls preserve E/HOF/q/full forces and iteration
counts exactly; all 52 derivative-primitive rows remain allocation-free under
the C++ counter. Complete calls save `2 * iterations` allocations and
`16 * packed_size * iterations` requested bytes. At 48 atoms, moving calls
reduce cumulative requested bytes by 43.5–44.8%, while still making
1,232–2,688 C++ allocations per complete call. Direct library malloc is not
covered. Six-round elapsed-time measurements establish **no reliable CPU
end-to-end speedup**; lower allocation traffic is the accepted benefit.

| Method | 48-atom moving CPU median, before → after (ms) | CUDA public force after (ms) | CUDA resident force after (ms) |
| --- | ---: | ---: | ---: |
| AM1 | 40.871 → 40.736 | 29.790 | 29.801 |
| RM1 | 40.057 → 40.071 | 31.692 | 31.753 |
| MNDO | 39.569 → 39.328 | 29.801 | 29.462 |
| PM3 | 40.029 → 39.985 | 34.994 | 35.612 |

Medians pool three runs per revision, twenty warm calls per mode/run. Backend
SCF counts differ; these are complete-call measurements, not a kernel-only
speedup or a portable GPU advantage. The benchmark report contains the full
3–48 atom cold/fixed/moving table and iteration statistics. CUDA production
remains unchanged: a candidate library eigensolver failed complete AM1
acceptance and was rejected without changing scientific tolerances.

For each method, a full-process Nsight capture verifies the **48-atom,
twenty-call moving resident force interval**: no H2D copies, only
4/8/24/56-byte D2H status blocks, no scientific-array retrieval, and zero
observed device allocation/release events or runtime allocator calls.
Device matrix copies remain D2D. Each trace reconstructs 169,924,475 bytes
peak/live dynamic-device allocation, with matched frees by capture end;
56 pinned bytes and 3,044 device-static bytes are separate. This includes
library allocations and is not a measure of reserved VRAM or method ownership.
The source and trace checks do not establish zero allocations at arbitrary
sizes, all host allocations, or absence of other-device faults.

Current acceptance also passed 1,301 local units, 16 affected regressions,
49 local Python checks (four CUDA-only skips), 113 SE-only and 32 targeted
combined CUDA unit checks, all four external/native composition and NVE cases,
and all 53 CUDA-build Python checks plus external/composition scripts.
AM1/RM1 combined forces and MNDO/PM3 cases passed memcheck with zero errors.
Existing external tolerances are unchanged. Scientific arrays are validated
against the CPU at `1e-6 eV / 1e-7 e / 1e-5 eV/Å`, separately from the
OpenMOPAC oracle gates. No unverified eigensolver optimization is retained.

The older [energy-only report](../../benchmarks/perf/semiempirical/nddo_energy/README.md)
remains historical evidence for the revision it names; current force residency
and memory claims come from the force captures above.

### Gaps

- Elements beyond H/C/N/O in shipped SK / xTB tables (mio-1-1 P/S and
  xtb's all-element tables; AM1 NDDO rows are broader).
- DFTB3 is not a DFTB+ Hamiltonian until γ^h and, under PBC, Ewald γ land.
- D3 is two-body and orthorhombic; the ATM three-body term and a general
  cell are still open. On-site correction (`OnSiteCorrection`) is open.
- TD-DFTB Casida is closed-shell MioSketch at Γ; 3ob and a collinear
  reference are open.
- Analytic (closed-form) Hessians anywhere in this chapter — all FD via
  `cpu::vib::Hessian`. xtb `--hess` is numerical too.
- CUDA AM1/RM1/MNDO/PM3 molecular E/q/ΔHf/analytic F and Evaluate composition are
  externally validated within neutral closed-shell CHNO FP64. Short MD,
  steepest descent, native line search, direct MD input edits and Python
  workflows and 10 fs NVE/time-step refinement pass the checks above.
  Longer-time stability, broader system/resource scaling and complete host
  allocator coverage remain open; current force measurements are scoped above.
  Stress, charged/open-shell, non-CHNO and periodic requests reject.
