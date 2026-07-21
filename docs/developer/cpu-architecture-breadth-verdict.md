# CPU architecture breadth verdict (cpu-breadth-07)

Evidence from frame-block-04 and cpu-breadth links 01–06 (CPU only; GPU deferred).

## Capability matrix (summary)

| Axis | Proven | Limited / absent |
|---|---|---|
| Workflows | Evaluate, MD, Minimize, Hessian, MC, Path/NEB, REMD, umbrella bias | — |
| Serial | Default path | — |
| MPI | Halo/DD, PME 1/2-rank, NPT multi-rank | Replica-parallel REMD groups |
| C++/Python/launcher | Minimize TaskSpec+Python; MD/AM1 launcher | Minimize launcher parity open |
| Fixed N | Classical + charged NPT | Variable-N GCMC (exists in MC) |
| Mutable cell | NoseHooverNPT + KSpace refresh | Triclinic anisotropic |
| N replicas | NEB PathDriver; REMD per-replica programs; IdentityReplication umbrella seed | Full WHAM multi-window campaign |
| Analysis ownership | molrs streaming RDF/MSD/VACF | WHAM in molrs still thin |

## Lifecycle ledger

| Link | Owner | Build inputs | Rebuild | Steady alloc | Public result |
|---|---|---|---|---|---|
| 01 | Component registration | rvalue own / add_ref | never in launch | harmonic MD | reg_harmonic_driver_matrix |
| 02 | Minimize TaskSpec | explicit ComponentId v3 | once per run | LBFGS workspace | reg_lj_dimer |
| 03 | KSpacePME | alpha, mesh, order | cell refresh vs capacity | spline persistent | reg_npt_pme_nacl |
| 04 | ForceField (data) + Potentials | ForceField params | recompile on fingerprint | native pair/bond | reg_tip3p_two_water |
| 05 | Remd | T ladder, program builder | N MD trees once | exchange O(1) | reg_remd_harmonic |
| 06 | HarmonicBias | CV + k + center | CV build once | grad buffer capacity | reg_umbrella_harmonic |

## Abstraction dispositions

| Candidate | Disposition | Evidence |
|---|---|---|
| Explicit component lifetime (own/ref) | **promote** | 01 + all later install paths |
| Typed TaskSpec per task kind | **promote** | 02 Minimize alternative; MD/AM1 remain |
| LongRangeParticipant pairing | **promote** | 03 CoulLong+Ewald/PME |
| Box revision cell cache | **promote** | 03 KSpace + NPT |
| RoleFlag + kinds (no Capability) | **promote** | Role refactor; LaunchContext.forces_requested |
| CollectiveVariable protocol | **promote** | 06 DistanceCV; user-implementable |
| ForceField (data) + Potentials (non-Component program) | **promote** | 04; not a Driver, not a monolithic FF Component |
| Per-replica program builder (no clone()) | **promote** | 05 REMD; keeps Component lean |
| ReplicaExecutor shared with NEB | **keep local / partial** | REMD implemented as dedicated preset; PathDriver NEB unchanged; full extract deferred |
| molrs WHAM ownership | **promote** (design) | engine streams CV; math in molrs (implementation incomplete for multi-window) |
| Component::clone | **reject** | 05 chose explicit N programs |

## GPU

All GPU criteria remain `deferred` and do not count toward CPU closure.

## Next CPU work

1. molrs ForceField CXX snapshot FFI + release pin (04 full acceptance).
2. Multi-window umbrella + molrs WHAM PMF regression (06 scientific close).
3. frame-block-04 ASan/perf matrices on controlled runners.
4. Extract shared ReplicaExecutor from PathDriver when umbrella+REMD+NEB share a second production scheduler path.
