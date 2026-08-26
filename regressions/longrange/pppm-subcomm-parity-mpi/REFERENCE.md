# Reference — PPPM FFT subgroup: a smaller FFT communicator is the same sum

oracle: literature
oracle_version: n/a (no oracle binary is run). The NaCl rock-salt Madelung
constant M = 1.747564594633182 is a hard-coded literal in the regression
source; the arm-vs-arm criteria are an analytic identity evaluated in-file. No
third-party engine is imported, spawned or read at build or run time, and no
engine snapshot is stored.
perf_exempt: yes — this slug makes **no** performance claim. The reason the
subgroup arm exists is wall-clock, but not one wall-clock number is asserted
here: the three-arm × np ∈ {64, 125, 200, 256} matrix lives in the dated
snapshot block of `tools/bench_gaff_lammps/README.md` and is explicitly not an
acceptance criterion (spec `kspace-subcomm`, "Out of scope").

## Invariant under test

Two things, in two layers.

**Layer 1 — physics vs literature (np = 1).** The Ewald/SPME split of a
neutral rock-salt lattice must reproduce the exact lattice sum

    E = −½ · N · M · C / d,   M = 1.747564594633182,
    C = 14.399645 eV·Å/e²,    d = 2.82 Å,   N = 216

up to the **grid discretization** of the mesh it is evaluated on. This is a
window, not an identity: a K = 12 / order-5 SPME mesh is an approximation of
the lattice sum and is allowed to differ from it by the mesh truncation error.

**Layer 2 — decomposition invariance (np = 1 vs np = 8, and arm vs arm).**
The total energy and every per-atom force are functionals of the Hamiltonian.
Which MPI communicator the FFT slabs live on is *not* a physical parameter, so
moving the FFTW-MPI slab from the whole Cartesian communicator onto a
`pppm_fft_ranks(K0, nprocs)`-sized **subgroup** of it may change nothing but
the floating-point association of the summands. This is an identity, not a
window.

| Layer | Quantity | Must equal | Tolerance |
| --- | --- | --- | --- |
| 1 | E(np = 1) | −½ N M C / d | 2e-3 relative (grid) |
| 2 | E(np = 8, subgroup) | E(np = 1) | 1e-9 relative |
| 2 | E(np = 8, subgroup) | E(np = 8, world arm) | 1e-9 relative |
| 2 | every per-atom force, tag-aligned | F(np = 8, world arm) | 1e-9 eV/Å |
| 2 | Σᵢ Fᵢ, per leg | 0 | 1e-10 eV/Å |
| 2 | maxᵢ \|Fᵢ\|, per leg | 0 (rock-salt symmetry) | 5e-3 eV/Å |
| — | `mesh_arm()` / `fft_ranks()` self-report, per leg | the arm the leg is *for* | exact |

The last row is not decoration. On this system the two np = 8 arms are
**bitwise identical** in every physical number (see "Why the arms agree
exactly" below), so no physics gate can tell them apart — an implementation
that quietly resolved `fft_ranks` to `nprocs` and stayed on the world arm would
print PASS on rows 1–6. Falsification **D1** below is exactly that defect, and
gate 5 is the only thing that catches it.

### Citations

- C. Kittel, *Introduction to Solid State Physics*, 8th ed., Wiley (2005),
  ch. 3 — the rock-salt Madelung constant and the ionic-crystal lattice-energy
  expression `E = −½ N M q² / (4πε₀ d)` used as the oracle.
- J. Sherman, *Crystal Energies of Ionic Compounds and Thermochemical
  Applications*, Chem. Rev. **11**, 93 (1932), DOI 10.1021/cr60038a002 —
  the tabulated NaCl-structure Madelung constant; source of the 16-digit
  literal `M = 1.747564594633182` hard-coded in the regression.
- U. Essmann, L. Perera, M. L. Berkowitz, T. Darden, H. Lee & L. G. Pedersen,
  *A smooth particle mesh Ewald method*, J. Chem. Phys. **103**, 8577 (1995),
  DOI 10.1063/1.470117 — the SPME influence function H(m) and the B-spline
  charge interpolation that `KSpacePPPM` implements, and the source of the
  mesh-resolution / splitting-parameter error balance the 2e-3 window is
  derived from.
- S. Plimpton, *Fast Parallel Algorithms for Short-Range Molecular Dynamics*,
  J. Comput. Phys. **117**, 1–19 (1995), DOI 10.1006/jcph.1995.1039 — spatial
  decomposition as a **partition of one and the same sum**. This is the whole
  rationale for the 1e-9 windows: two decompositions of one Hamiltonian differ
  only in double-precision summation order, so anything above reassociation
  noise is a defect, not a tolerance.
- M. Frigo & S. G. Johnson, *The Design and Implementation of FFTW3*,
  Proc. IEEE **93**, 216 (2005), DOI 10.1109/JPROC.2004.840301 — FFTW's MPI
  dim-0 (slab) decomposition: a rank has work only if it owns at least one
  plane. That is the origin of the `P_fft ≤ K0` bound behind
  `pppm_fft_ranks`, and of the two idle ranks the subgroup arm removes here.

## System (child, np = 1 and np = 8)

216 rock-salt NaCl ions — a 6 × 6 × 6 simple-cubic array of alternating ±1 e
point charges at nearest-neighbour distance **d = 2.82 Å** — in a cubic
periodic box of edge **L = 6 d = 16.92 Å**, Σq = 0 (108 cations, 108 anions).

| Parameter | Value |
| --- | --- |
| Ions / box | 216 / cubic L = 16.92 Å, fully periodic |
| Lattice | site (i,j,k) at origin + (i,j,k)·2.82 Å, q = (−1)^(i+j+k) |
| Lattice origin | (1.41, 1.41, 1.41) Å = **exactly one mesh cell** (see below) |
| Masses / types | 22.9898 amu (type 0, +1 e) / 35.453 amu (type 1, −1 e) |
| Topology | none — an ionic crystal has no bonds, so no special-bond lists |
| Real space | `cpu::CoulLong{α = 0.40 Å⁻¹, r_c = 7.0 Å}` |
| Reciprocal space | `cpu::KSpacePPPM{α = 0.40 Å⁻¹, spacing = 1.45 Å, order = 5}` (`fft_ranks` left at the deriving default 0) |
| Mesh | `good_fft_size(⌈16.92/1.45⌉) = good_fft_size(12) = 12` per dimension |
| Verlet skin | 0.3 Å |
| Driver | `cpu::Evaluate` — one static pass, no time advance, no convergence loop |
| Decomposition | forced `ATV_FORCE_DIMS=2,2,2` (ignored at np = 1) |

Pipeline — all defaults, the FFT communicator being the feature under test:
`Evaluate::install_defaults()` injects `comm::Migrate` (−50),
`comm::HaloSend{max(1.1 · 7.0, 7.0 + 0.3) = 7.7 Å}` (−30), `comm::HaloRecv`
(+10) and `md::mpi::Reduce` (+20) into the `evaluate` phase, with the neighbor
list at −10 and the two evaluators at 0. Nothing registers a halo component or
a hop count by hand.

### Why this geometry

| Choice | Why it must be this |
| --- | --- |
| L = 16.92 Å, spacing 1.45 Å | ⌈16.92/1.45⌉ = 12 and `good_fft_size(12) = 12` (12 = 2²·3 is {2,3,5,7}-smooth), order + 1 = 6 ≤ 12, and dim 2 is even, so no clamp moves it. **12 is the point**: `pppm_fft_ranks(12, 8) = 6` — the largest divisor of 12 that fits 8 ranks — so the FFT group is a *proper* subset and two ranks are genuine non-members. A mesh with no proper divisor ≤ 8 would silently give the world arm. |
| np = 8, dims 2×2×2 | 8.46 Å slabs. The auto-injected 7.7 Å shell needs `⌈7.7 · 2 / 16.92⌉ = 1` hop, which is the deepest legal hop at `dims = 2` (`SwapList::derive_hops` refuses more). |
| r_c = 7.0 Å | ≤ L/2 = 8.46 Å (minimum image well defined) **and** max(1.1 r_c, r_c + skin) = 7.7 Å ≤ 8.46 Å (the shell fits one slab). r_c = 8.0 Å would give a 8.8 Å shell and be refused outright at dims = 2. |
| α = 0.40 Å⁻¹ | Balances the split at this mesh: α·r_c = 2.8 ⇒ real-space truncation erfc(2.8) ≈ 7.5e-5; k_max = πK/L = 2.229 Å⁻¹ ⇒ reciprocal truncation ≈ exp(−(k_max/2α)²) = exp(−7.76) ≈ 4.3e-4. Both sit two decades inside the 2e-3 window; measured total deviation 7.3e-6. |
| origin = 1.41 Å = L/12 | One **exact mesh cell**, so d = 2.82 Å is exactly two cells and every ion still sits on a mesh node — while no ion sits on the periodic boundary or on the 8.46 Å domain cut (nearest ion 1.41 Å away). |

#### Why the ions sit on mesh nodes

With every ion on a mesh node the B-spline stencil is inversion-symmetric about
each ion, so the mesh force cancels to the ULP and the ‖Σᵢ Fᵢ‖ / maxᵢ |Fᵢ| rows
become *sharp* (measured 5e-15 and 2e-15) instead of approximate. Any mesh
pipeline that lost a plane, mis-addressed an FFT owner or skipped a halo breaks
that cancellation and shows up as a force — falsifications D2b and D3 below
move maxᵢ |Fᵢ| to 3.4e-2 and 6.4e-1 eV/Å respectively.

Off-node ions would **not** do. `cpu::KSpacePPPM` interpolates forces by pure
analytic differentiation (`src/cpu/longrange/pppm.cpp:1178`) with no self-force
correction (the LAMMPS `PPPM::compute_sf_precoeff` / `fieldforce_ad` `sf_coeff`
term). An ion at a generic sub-cell fraction therefore feels a spurious force
∝ q_i², i.e. the **same sign on every ion**. Measured while developing this
case, with the lattice displaced to (0.37, 0.53, 0.29) Å: 1.385e-3 eV/Å per ion
and ‖Σᵢ Fᵢ‖ = 2.991e-1 eV/Å over 216 ions, identical on all three arms and at
np = 1. That is a pre-existing accuracy artifact of the AD scheme, unrelated to
this spec and not what this case is about — but it is why the origin is a whole
mesh cell rather than an arbitrary offset, and it is reported as a
found-not-fixed item rather than absorbed into a loosened tolerance.

### The three legs

The parent re-launches **the same frozen binary** through the compiled-in
`ATV_MPIEXEC` three times, so an arm difference can never be a build
difference:

| Leg | Launch | Selector path | Expected self-report |
| --- | --- | --- | --- |
| serial control | `mpiexec -n 1` | no DD ⇒ replicated | `Replicated`, `fft_ranks = 1` |
| feature | `mpiexec -n 8` | `pppm_select_mesh_arm(12, 8, pppm_fft_ranks(12, 8) = 6)` | `BrickSubgroup`, `fft_ranks = 6` |
| control arm | `ATV_PPPM_BRICK=1 mpiexec -n 8` | debug forcing, preserved from the two-arm era | `BrickWorld`, `fft_ranks = 8` |

`ATV_PPPM_BRICK` travels through the environment (Hydra exports the launching
environment to every rank) and is unset again immediately after the leg.

The report is assembled **in this file**: an out-of-library `cpu::Component` at
`evaluate::Phase::finalize @ +900` — after `md::mpi::Reduce` (evaluate @ +20)
has made the Frame scalars system totals and after `HaloRecv` (evaluate @ +10)
folded every ghost force onto its owner — allgathers every rank's owned rows by
tag and min/max-reduces the energy, the arm and the group size across ranks.

## Pinned numbers (hard-coded gates in the regression)

Measured **2026-08-26** on the x86_64 login node `arrhenius1`
(AMD EPYC 9255, GCC 14.3.0, MPICH 4.3.2 / Hydra, FFTW 3.3.11 with
`libfftw3_mpi`, Release build `build-gaff`, `OMP_NUM_THREADS=1`).
Total wall for all three legs together: **1.51 s** (budget ≤ 60 s).

| Gate | Want | Tolerance | Measured |
| --- | --- | --- | --- |
| `good_fft_size(⌈L/spacing⌉)` | 12 | exact | **12** |
| `pppm_fft_ranks(12, 8)` | 6 | exact | **6** |
| `pppm_select_mesh_arm(12, 8, 6)` | `BrickSubgroup` | exact | **BrickSubgroup** |
| `kCoulomb` vs `ATV::units::coulomb` | equal | exact | **0.000e+00** |
| \|E(np1) / E_Madelung − 1\| | 0 | 2e-3 | **7.338e-06** |
| \|E(np8,sub) − E(np1)\| / \|E(np1)\| | 0 | 1e-9 | **3.539e-16** |
| \|E(np8,sub) − E(np8,world)\| / \|E\| | 0 | 1e-9 | **0.000e+00** |
| max \|Fᵢ(sub) − Fᵢ(world)\| (tag-aligned) | 0 | 1e-9 eV/Å | **0.000e+00** |
| ‖Σᵢ Fᵢ‖, np = 1 | 0 | 1e-10 eV/Å | **9.062e-15** |
| ‖Σᵢ Fᵢ‖, np = 8 subgroup | 0 | 1e-10 eV/Å | **5.102e-15** |
| ‖Σᵢ Fᵢ‖, np = 8 world | 0 | 1e-10 eV/Å | **5.102e-15** |
| maxᵢ \|Fᵢ\|, np = 1 | 0 | 5e-3 eV/Å | **2.364e-15** |
| maxᵢ \|Fᵢ\|, np = 8 (both arms) | 0 | 5e-3 eV/Å | **2.064e-15** |
| owned rows assembled / lost-or-duplicated tags / passes, per leg | 216 / 0 / 1 | exact | **216 / 0 / 1** |
| reduced-energy spread across ranks, per leg | 0 | 1e-10 rel | **0.000e+00** |
| np = 1 self-report | `Replicated` / 1 | exact | **Replicated / 1** |
| np = 8 default self-report | `BrickSubgroup` / 6 | exact | **BrickSubgroup / 6** |
| np = 8 `ATV_PPPM_BRICK=1` self-report | `BrickWorld` / 8 | exact | **BrickWorld / 8** |

### Pinned energies

| Quantity | Value [eV] |
| --- | --- |
| **E(np = 1), replicated arm — the pin** | **−963.7465953900127715** |
| E(np = 8), subgroup arm (`fft_ranks = 6`) | −963.7465953900124305 |
| E(np = 8), world arm (`ATV_PPPM_BRICK=1`) | −963.7465953900124305 |
| E_Madelung = −½ · 216 · 1.747564594633182 · 14.399645 / 2.82 | −963.7395233854490471 |

The Madelung value is computed in-source from the four literals; the engine
energies are provenance for this environment. The gates the case enforces are
the relative ones above, so the engine pins do not themselves have to survive a
retune of α / spacing / order.

### Why the arms agree *exactly*

`|E(np8,sub) − E(np8,world)|` and `max|ΔF|` are both **identically zero**, not
merely below 1e-9, and that is expected rather than suspicious. FFTW's MPI
dim-0 split of 12 planes over 8 ranks uses its default block ⌈12/8⌉ = 2, giving
`local_n0 = 2,2,2,2,2,2,0,0` — the same six two-plane slabs, on the same six
ranks, as the six-member subgroup's ⌈12/6⌉ = 2. (The measured bitwise equality
is what confirms this: any other split would put a different number of nonzero
partials into the remap and the energy reduction and could not land on 0.0.)
The two arms therefore perform *bitwise the same arithmetic*; what differs is
that on the world arm ranks 6 and 7 still
enter every `fftw_mpi` collective and the full-communicator Alltoallv while
owning nothing. Removing that is the entire point of the third arm — and it is
also precisely why gate 5 (the arm self-report) is load-bearing: on this system
physics alone cannot distinguish the arms.

The np = 8 vs np = 1 row is *not* zero (3.5e-16, one ulp of the total) because
the serial leg takes a genuinely different code path — replicated grid,
single serial FFT, no remap.

### Tolerance rationale

- **2e-3 relative for the Madelung window** — this is the size of the
  **physical** discretization error of a K = 12 / order-5 SPME mesh at
  α = 0.40 Å⁻¹, not a float tolerance. The two error channels of the Ewald
  split (Essmann et al.) are real-space erfc(α r_c) ≈ 7.5e-5 and reciprocal
  exp(−(πK/L / 2α)²) ≈ 4.3e-4; the measured deviation is 7.3e-6. The window is
  ~270× the observed error — wide enough that re-tuning α or the mesh inside
  this resolution class cannot flip the case, narrow enough that a wrong
  Hamiltonian (a lost mesh plane costs 1.1e-2, falsification D3) fails it. It
  matches the window the unit-level companion uses on the same crystal
  (`tests/mpi/test_pppm_brick_parity.cpp:42`).
- **1e-9 relative energy and 1e-9 eV/Å force for every arm-vs-arm and
  np-vs-np row** — reassociation only. Both sides evaluate the same
  Hamiltonian on the same configuration; the only admissible difference is the
  order in which the summands are added (which rank owns which mesh plane,
  which Alltoallv reassociates which brick, which reduction tree the energy
  goes through). For 216 atoms and a 12³ mesh in double precision that is
  O(N · ε_mach) ≈ 1e-15 relative. Measured 3.5e-16 and 0.0 — at least six
  orders inside the gate, wide enough to survive a future reorder (rank count,
  thread count, xsimd), narrow enough that the smallest realistic defect in the
  falsification table (2.6e-4, D2b) lands **five orders past it**.
- **1e-10 eV/Å for ‖Σᵢ Fᵢ‖** — Newton's third law across the cuts, plus
  rock-salt symmetry: the exact value is zero, and with the ions on mesh nodes
  the mesh contribution cancels to the ULP too. Measured 5e-15 ≈ 216 · ε_mach
  on the force scale.
- **5e-3 eV/Å for maxᵢ \|Fᵢ\|** — the same rock-salt |F| residual bound the
  unit-level companion uses (`tests/mpi/test_pppm_brick_parity.cpp:136`), kept
  identical so the two gates are comparable. Measured 2e-15.

The reassociation windows are deliberately stricter than the harness's generic
1e-6 numerical-energy tier: these are identities, not converged approximations.

## Falsification (2026-08-26)

Each defect was **actually injected** into `src/cpu/longrange/pppm.cpp`, the
library and the case rebuilt, the case re-run, and the file then restored from
a pristine copy (md5 `ef39c497ae85392630ea1a4e3a77d146` verified back). No gate
was loosened and no tolerance was moved. The committed tree is the unperturbed
one and re-runs to the pinned numbers above.

| # | Injected defect | Observed failure |
| --- | --- | --- |
| **D1** | `pppm.cpp:388` — derive the group as `fft_ranks_resolved_ = mesh_nprocs;` instead of `pppm_fft_ranks(K0, mesh_nprocs)`, i.e. a selector that silently falls back to the world arm | **Gate 5 only.** np = 8 default leg reports `BrickWorld / 8` instead of `BrickSubgroup / 6`. Every physics gate stays green — Madelung 7.338e-06, np8-vs-np1 3.539e-16, arm-vs-arm 0.0, forces 0.0. `FAIL (1 gate)`, exit 1. This is the false-green the arm self-report exists to stop. |
| **D2a** | `pppm.cpp:803` — after the ownership Allgather, zero the last FFT-group member's plane count (`fft_m0_counts_[fft_ranks_resolved_ - 1] = 0`), breaking the membership restriction of the m0 table | **`fft_owner` throws**, exactly as ac-004 requires: `KSpacePPPM: mesh plane m0=10 of K0=12 is owned by no rank of the FFT group (fft_ranks=6 of 8 ranks). Reproduce with the same mesh and rank count; the FFTW-MPI dim-0 slabs must tile [0, K0).` The child aborts (`BAD TERMINATION OF ONE OF YOUR APPLICATION PROCESSES`); the parent prints `FAIL: np=8 child exited nonzero`, exit 1. |
| **D2b** | D2a **plus** the deleted `return nprocs - 1;` fallback restored in place of the throw at the end of `fft_owner`
(`pppm.cpp:821`), i.e. the pre-spec behaviour: no throw, planes routed to rank 7 — a **non-member** — which drops them | **Gates 2, 3 and 4.** Silent wrong answer: E(np8,sub) = −963.4938424404770103. np8-vs-np1 **2.623e-04** (want 1e-9) FAIL; arm-vs-arm energy **2.623e-04** FAIL; tag-aligned max\|ΔF\| **3.403e-02 eV/Å** FAIL (worst tag 202); np = 8 subgroup maxᵢ\|Fᵢ\| **3.403e-02** FAIL. `FAIL (4 gates)`, exit 1. Note the size: 2.6e-4 is **inside** the 2e-3 Madelung window, so a literature-window-only case would have printed PASS. This row is why the 1e-9 identity gates exist next to the physics window. |
| **D3** | `pppm.cpp:602` — skip the mesh `reverse_comm` (`for (int dim = 0; dim < 3; ++dim)` → `dim < 0`), so ghost density is never folded into its owners | **Gates 2 and 4.** E(np8) = −953.3114763849750943, np8-vs-np1 **1.083e-02** FAIL; both np = 8 legs break the symmetry rows: ‖Σᵢ Fᵢ‖ **3.244e-01 eV/Å** and maxᵢ\|Fᵢ\| **6.370e-01 eV/Å**, both FAIL. Gate 3 (arm vs arm) stays **green**, as it must — the defect is arm-independent — which is exactly why gate 2 against the serial control exists beside it. `FAIL (5 gates)`, exit 1. |
| **D4** | `pppm.cpp:758` — invert the subgroup membership rule in `MPI_Comm_split` (`cart_rank >= nprocs - fft_ranks_resolved_` instead of `cart_rank < fft_ranks_resolved_`), so cart rank 0 is no longer a member and the `first_rank` reciprocal-energy accumulation misses the subgroup reduction | **Gates 2 and 3.** E(np8,sub) = −969.3166544588441411; np8-vs-np1 **5.780e-03** FAIL, arm-vs-arm energy **5.780e-03** FAIL. Forces unaffected (max\|ΔF\| 0.0, ‖Σ F‖ 5.1e-15) — a pure energy-accounting defect that only an energy identity can see. `FAIL (2 gates)`, exit 1. |

Read together the table shows the gates are non-overlapping: D1 is invisible to
every physical gate, D3 is invisible to the arm-vs-arm gate, D4 is invisible to
the force gates, and D2b is invisible to the literature window. Removing any
one gate would let one of these defects through.

## Scope note

This case pins the **correctness** of the third mesh arm: an FFT slab confined
to a proper subgroup of the Cartesian communicator, with real non-member ranks
that still spread, halo, remap and interpolate, produces the same energy and
the same forces as the world-communicator brick arm and as the serial control,
and it self-reports the arm it actually took.

Out of scope, by spec: the wall-clock case for the third arm
(`tools/bench_gaff_lammps/README.md`, three-arm × np ∈ {64, 125, 200, 256}
snapshot); the 24048-atom / 256-rank real-machine parity window
(`tools/bench_gaff_lammps/widebox_parity.sh`, ac-010); the pure-function
behaviour of `pppm_fft_ranks` / `pppm_select_mesh_arm` (unit-tested serially in
`tests/cpu/longrange/test_pppm_coul.cpp`); the n = 1/2/4 arm gates
(`tests/mpi/test_pppm_subcomm_parity.cpp`); the Alltoallv shape; migrating the
subgroup communicator to `DomainRuntime`; and the CUDA backend.
