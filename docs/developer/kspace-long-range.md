# KSpace long-range electrostatics (CPU)

This note is the durable contract for `CoulLong` + `KSpaceEwald` /
`KSpacePME` composition, cell refresh, stress, and MPI scope. It closes
cpu-breadth-03 **ac-009** documentation.

## Layout (after hf-capability-gaps-19-longrange)

| Layer | Path | Owns |
|---|---|---|
| Facade math | `include/atomiverse/longrange/` | `EwaldSum`, PME B-spline / moduli / influence; HE green tables (`pppm_green.h`) for future IK path — no Frame/MPI/FFT |
| CPU wrappers | `include/atomiverse/cpu/longrange/`, `src/cpu/longrange/` | `KSpaceEwald` / `KSpacePME` (complex c2c SPME) / **`KSpacePPPM` (real brick + r2c/c2r, SPME H(m))** |
| Build flag | `ATV_BUILD_KSPACE` | Gates FFTW3 / mesh reciprocal |
| Science gate | `tests/cpu/longrange/test_pppm_coul.cpp` | Madelung, Ewald/PME parity, α-independence, FD forces |

SCF and future consumers include only `atomiverse/longrange/*`, never
`cpu/longrange/*`. Citations travel with the math:
Ewald 1921 (DOI: 10.1002/andp.19213690304); Essmann et al. 1995
(DOI: 10.1063/1.470117).

## Pairing (fail closed)

Real-space and reciprocal-space halves are ordinary components that implement
`LongRangeParticipant` and declare:

| Field | Meaning |
|---|---|
| `role` | `RealSpace` (`CoulLong`) or `ReciprocalSpace` (`KSpaceEwald` / `KSpacePME` / `KSpacePPPM`) |
| `alpha` | Ewald splitting α (1/Å) — **must match** between the pair |
| cutoff / accuracy | cold metadata only |

`Driver` validation at `build()` accepts **exactly one** real-space plus
**exactly one** reciprocal partner with matching `alpha`. Missing, duplicate,
or mismatched partners throw before launch with component names and α values.
There is no short-range or energy-only fallback.

NPT / any component with `requires_system_stress()` further needs the
reciprocal style to **write** complete reciprocal stress into system meta.
PME does so after the Essmann influence-function virial implementation.

## Stress (system virial)

Stress is the **system** 6-component virial in `frame.meta` under key
`"stress"` (`f.meta.get(kStress)`), Voigt order xx,yy,zz,xy,xz,yz, units eV
(LAMMPS metal / force·distance). It is **not** a per-atom column.

- Real-space: `CoulLong` accumulates pair virial `r·F`.
- Reciprocal: Ewald and PME write the influence-function geometric factor
  `V_ab = ug · (δ_ab + vt · k_a k_b)` (m-space form for PME).
- Kinetic: `Virial` observer → `kStressKinetic`.
- Barostat pressure: `P = tr(σ_virial + σ_kinetic) / (3V)`, positive compressive.

## Cell refresh vs capacity rebuild

`KSpace` owns a lifecycle ledger (`lifecycle_stats()`):

| Event | When | Cost |
|---|---|---|
| `builds` | first `build()` | allocate atom/grid scratch, create FFT plans (PME) |
| `cell_refreshes` | Box identity or revision changes, **same** FFT/grid shape | update reciprocal basis / influence; overwrite persistent spline scratch (`fill`, not re-`assign`) |
| `capacity_rebuilds` | atom capacity growth **or** FFT mesh dimensions change | replan FFT, reallocate grid |

Caches key on `(Box*, revision)`. Replica frames with equal revisions but
distinct Box identities still refresh.

Instrumentation gate: `PmeLifecycle.StableGridRefreshIsReplanFreeAndMemoryStable`
and related unit tests.

## MPI: PME vs PPPM FFT

| Style | Multi-rank (default) | Optional / debug |
|---|---|---|
| **PME** | Replicated-grid: `MPI_Allreduce` full charge mesh; rank-0 FFT; `MPI_Bcast` potential | — |
| **PPPM** | **Subgroup arm**: brick + reverse/forward + Alltoallv brick↔FFT + FFTW-MPI slabs on a derived communicator of `pppm_fft_ranks(K0, nprocs)` cart ranks `[0, P_fft)` (largest divisor of K0 within the rank budget; the tree's first real `MPI_Comm_split`, fingerprint-guarded, aliased un-split when `P_fft == nprocs`). Non-members spread/halo/remap/interpolate only — no FFT collective. Selection: `pppm_select_mesh_arm`; the old `mesh ≤ 64³ && ranks ≥ 16 → replicated` rule is gone (falsified at np=64: brick 4.78 s vs replicated 5.43 s). Gates: `test_mpi_pppm_subcomm_parity` n=1/2/4, `test_mpi_pppm_brick_parity` n=1/2. | `ATV_PPPM_BRICK=1` forces the world-comm brick arm (all ranks in the FFT collective); `ATV_PPPM_ALLREDUCE=1` forces the replicated arm (dens_full Allreduce, every rank runs the serial FFT). Both are the debug/compare paths the three-arm bench matrix is measured with. |

Self-energy partitions over owned atoms; reciprocal energy/net-charge on
rank 0; `Reduce` all-reduces frame energy/stress.

Halo still only exchanges immediate neighbour ranks — the charged NPT MPI
fixture uses a box large enough that `cutoff + skin` fits in each subdomain
(see `test_mpi_pme_nacl_parity`, `test_mpi_pppm_brick_parity`, n=6 rock-salt).

## Atom spatial sort (not kspace, but cache locality)

LAMMPS `atom_modify sort`: **exchange → sort → borders** (not domain
decomposition). Atomiverse:

| Mode | Where sort runs |
|---|---|
| **Multi-rank** | `HaloSend` full rebuild, *before* ghost slots / the swap-table sendlists |
| **Serial** | `NeighborList` reneighbor before cell-list rebuild |

After a successful reorder, `notify_sorted` bumps `neighbor_generation`
so `BondedTupleCache` re-resolves tag→local (serial MD still has a `dd`).
Without that bump, nfreq=1 sort leaves bonded forces on stale local
indices and the trajectory explodes.

Cadence: first sort after `nfreq` steps (default 1000), matching LAMMPS
`nextsort` init — not on step 0.

## Accepted scope

| In scope | Out of scope (explicit) |
|---|---|
| Isotropic / orthorhombic NPT | Triclinic / anisotropic NPT |
| Neutral ionic crystals | 1-2/1-3/1-4 reciprocal exclusion correction (link 04) |
| Serial + multi-rank replicated grid | Distributed FFT PME |
| Soft core + CoulLong + PME composition | CUDA KSpace (deferred) |

## Public consumers

| Consumer | Path |
|---|---|
| Facade math | `tests/cpu/longrange/test_ewald_sum.cpp`, `test_pme_math.cpp` |
| Unit + lifecycle | `tests/cpu/longrange/test_pme_coul.cpp` |
| 1/2-rank parity | `tests/mpi/test_pme_nacl_parity.cpp` |
| Literature regression | `regressions/md/npt_pme_nacl/` |
| LAMMPS validation | `/* benchmarks removed; re-add later */ npt_pme/validation/bench_pme_nacl_lammps` |

Build with `-DATV_BUILD_KSPACE=ON` (FFTW3). MPI tests also need `-DATV_USE_MPI=ON`.
