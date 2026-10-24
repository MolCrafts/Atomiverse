# Semiempirical and Tight-Binding Methods

Tutorial: [Semiempirical models](../tutorial/ch17-semiempirical-am1.md).
This chapter records the current `dev` implementation and executed evidence as
of 2026-10-24; it is not a claim about the released `master` snapshot.

CPU AM1/RM1 provide molecular closed-shell energy, charges, heat of formation
and analytic forces. Native CUDA AM1/RM1 energy, charges, heat of formation
and analytic forces pass external and CPU comparisons for neutral closed-shell
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
| MNDO / PM3 / PM6 / PM7 | ○ | ○ | No public method or validated parameter/gradient path |
| OM3 | ○ | ○ | **Blocked** — no OpenMOPAC primary surface |
| Nuclear forces (CHNO molecules) | ✓ | ✓ | `semiempirical/am1`: AM1/RM1 distorted H₂O, tilted CO and NH₃ full vectors against executed OpenMOPAC 23.2.5 |
| Analytic integral-derivative forces | ✓ | ✓ | `semiempirical/am1`: analytic pair derivatives with independent CHNO force pins; broader s/p elements have derivative consistency tests only |
| Evaluate composition | ✓ | ✓ | Standalone/mounted E/q/F agreement, additive contributions and failure/retry; CUDA restricted to the molecular CHNO envelope |
| Short BOMD / steepest-descent composition | ✓ | ✓ | `semiempirical/am1`: both methods pass initial-force preparation, CUDA 2+2 versus 4 steps and CPU/CUDA four-step agreement at 0.01 fs; three SD steps lower energy, without a minimization-convergence claim |
| FD Hessian via `vib::Hessian` | ✓ | ○ | CUDA Hessian workflow not validated |
| Heat of formation (MOPAC-style) | ✓ | ✓ | AM1/RM1 executed OpenMOPAC pins; CUDA restricted to neutral closed-shell CHNO |
| NDDO UHF / COSMO / Γ-PBC | ○ | ○ | Molecular AM1 under wrap unwraps an isolated fragment |

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

The 2026-10-24 CPU SE build passed all 711 registered checks, including all
11 semiempirical method cases; the separate AM1/AM1-BCC Python suite passed
26 checks. The GH200 SE-only build passed 106 CUDA checks, with one multi-GPU
skip and two disabled-HF skips; the combined build passed 221 with thirteen
multi-GPU skips.
Affected MD, RHF/UHF and Minimize regressions passed. With SE disabled and
HF/MD/CUDA enabled, build/link, 13 shared execution checks and the RHF/MD
initial-force sections also passed. A passing case means its
current assertions passed, not that every capability of the named reference
program is implemented.

| Method / case | Executed CPU evidence | Main remaining gate |
| --- | --- | --- |
| AM1 | `semiempirical/am1` passed, including six AM1/RM1 CHNO energy/q/ΔHf/full-force fixtures | Longer NVE/time-step convergence, line-search trial isolation, chunk-boundary host edits; broader-element external forces and Python CUDA binding |
| RM1 | `semiempirical/rm1` plus the RM1 fixtures in `semiempirical/am1` passed | Longer NVE/time-step convergence, line-search trial isolation, chunk-boundary host edits; remaining parameter coverage and Python RM1 surface |
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
composition, not converged optimization or NVE energy conservation. Actual
line-search trial/rejection, host edits between chunks, and longer NVE with a
halved-time-step comparison remain acceptance gaps.

The [energy/residency benchmark](../../benchmarks/semiempirical/nddo_energy/README.md)
records the public-force revision with an executable hash, numerical-source
hashes, CMake cache and source-archive provenance. All eight AM1/RM1 size
combinations (3/12/24/48 atoms) pass CPU/CUDA energy and charge comparisons.
Complete-process traces verify the restricted 48-atom, twenty-call warm energy
path: only 8/56-byte D2H blocks, no H2D and no visible CUDA allocator/free calls
inside the synchronized interval. Measured peak memory is unavailable. These
are energy-path measurements; force performance and force residency remain
unmeasured. No broad small-system GPU speedup is claimed.
CPU allocation measurements are recorded in the
[force benchmark](../../benchmarks/semiempirical/nddo_forces/README.md);
they establish lower C++ allocation traffic, not a controlled latency gain.

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
- CUDA AM1/RM1 molecular E/q/ΔHf/analytic F and Evaluate composition are
  externally validated within neutral closed-shell CHNO FP64. Short MD and
  steepest-descent composition pass the checks above. Line-search trial/rejection,
  host edits between chunks, longer NVE/time-step convergence, Python parity
  and current-revision force-performance/residency evidence remain open.
  Stress, charged/open-shell, non-CHNO and periodic requests reject.
