# CPU component and execution contracts

Status: implementation; six-step migration approved by the user.

## Ordered implementation

1. Driver/Runtime: unique object lifecycle independent of phase bindings;
   exception-safe nested state; refresh compiled contracts on live mutations.
2. Neighbor consumers: half/full requirements replace the mutable global mode.
3. Evaluation: configured outputs and per-call requests; whole-system,
   spatially distributed and replicated execution capabilities.
4. Identity: tag-based persistent state; invalidate row/geometry caches after
   sorting, migration and replica configuration changes.
5. Trial state: commit/rollback before observation; electronic propagation
   history advances only at its declared time-step boundary.
6. Analysis: safe snapshot ownership, changing capacity, and explicit input
   distribution for scientific observers.

Reuse Plan, GhostContract, Frame/TagIndex, Transaction, TimestepConsumer and
existing reduction utilities. Do not add a public configuration aggregate.
Different physical Hamiltonians remain distinct. Unsupported combinations must
fail explicitly rather than silently select a different execution algorithm.

## Implemented contracts

- Runtime owns object reference counts, borrowed nested plans, dispatch mutation
  locks and physical child communication isolation. Live nested additions and
  removals refresh the parent contracts without rebuilding unchanged objects.
  Nested DriverNode identity has one lifecycle across registration-private
  stage adapters; each binding retains its own budget. Cross-Runtime live
  attachment remains rejected.
  Core execution is a protected backend primitive; public shutdown dispatches
  to the CPU owner and clears continuation state. Driver-managed Runtime unwinds
  successfully built components on escaping exceptions and rebuilds on retry;
  direct Runtime callers retain manual lifecycle and nested retry semantics.
- CPU convergence requirements declare their role. All requirements on every
  participating spatial rank must succeed; any stop wins. Stop-only guards do
  not suppress Minimize's default force requirement. Illegal guard success is
  reduced before all ranks throw. CUDA termination semantics are unchanged.
- Neighbor consumers request half/full and ghost-centred rows independently.
  EAM uses half-list density reverse and embedding-derivative forward exchange;
  SW uses full owned rows for angles and once-per-pair two-body ownership.
  OpenMP scratch follows the actual team size, including serialized nesting.
  PairList publishes which axes carry resolved halo images; consumers preserve
  those images rather than folding far skin-band ghosts into duplicate pairs.
  Linear workflows decide rebuilding at the force boundary after position updates.
- Classical pair/bonded and Ewald/PME/PPPM outputs honor independent force and
  stress requests. PPPM publishes reciprocal virial. PairDPD uses tag-pair noise
  and the dissipative sign consistent with its displacement convention.
- Brownian noise is tag-keyed. Bonded tuples invalidate on Frame identity,
  block generation, owner count and tag ordering, and reject missing partners
  consistently across spatial ranks. FixMove origins follow tags across sorting;
  spatial migration is explicitly unsupported until those origins can migrate.
- Joint cell relaxation uses owned rows, global force normalization/RMS and
  measured displacements in Angstrom. FIRE/CG/LBFGS/Joint reject nonfinite forces
  collectively. Isotropic/anisotropic cell relaxation uses owned rows, physical
  force/displacement units and explicit communication invalidation after scaling.
  Invalid pressure cannot be hidden by scale clamping; pressure/stress criteria
  return Stop for nonfinite selected observables, including collective polling.
  Thermostats, barostats, deform and time-dependent fields consume
  the integrator's timestep through the existing protocol.
- MC applies commit/rollback before observations, including nested MC, and
  restores accepted energy/charges/outputs after rejection. Trial evaluations
  preserve electronic predictor/history and metadynamics state. Unsupported
  reversible CP/shadow-mode combinations are rejected.
- Async analysis pins snapshot readers, resizes owned slots and drops a new
  snapshot when every slot is busy. Each subscriber observes FIFO order; explicit
  drain reports worker exceptions. Scientific observers declare full-frame
  requirements; MSD/VACF validate identities and sampling intervals.
- PW requests share one PWProblem/SCF result for energy and derivatives;
  stress-only updates last_result. Direct PW/GPW generic nesting is rejected
  until their stack-bound stage closures have a proper borrowed lifecycle.
  Post-HF restores temporary compute_eri after exceptions.
- SCF, GPW and CC reuse cpu/linalg/pulay.h for the constrained coefficient solve;
  real AO commutator math is shared there. Invalid history depths are rejected.
  KDiis accepts an explicit reducer; GPW retains its documented standalone WORLD
  collective scope. GPWEvaluator rejects multi-rank independent replicas until
  an explicit electronic execution communicator is available.
- FMM subdivision terminates for coincident/unresolvable coordinates; PCG
  handles zero RHS and rejects invalid preconditioners or curvature.

## Validation

Single-method tests belong in tests; trajectories and external-engine comparisons
belong in regressions/benchmarks. Expanded CPU build: /tmp/atv-architecture-pw.
Build/test/performance workloads run in separate windows. Preserve all other
worktree changes. CUDA is not validated on this host.

The triclinic migration passes 105 rebuilt non-MPI executables, 18 MPI
registrations, 34 tilted LAMMPS checks and unchanged
800-step PR-NPT, MTK NPT and NPH regressions. The materializer checks compiled
plan equivalence; full trajectories remain in regressions. The earlier 300
non-MPI / 58 MPI full-suite result belongs to the previous orthorhombic snapshot,
not the current Box ABI. Current scope and evidence are in
../notes/cpu-lammps-alignment.md.

The separate orthorhombic benchmark snapshot passes 68 LJ/EAM/SW initial/one-step checks and all
60 timed initial-state checks. Complete paired measurements, source/binary
hashes and representation limitations are in benchmarks/lj/cpu-contracts-m4.json;
historical snapshots remain explicitly separate. The newer flat-stencil snapshot is in benchmarks/lj/cpu-triclinic-m4.json:
neighbor construction improves 9–24% relative to its paired baseline but remains
1.54–2.01× LAMMPS. Universal performance parity is not established.

## Remaining review coverage

933 CPU files were inventoried; entry points across all pillars were reviewed,
but not every numerical implementation body. Do not claim a complete kernel
audit or universal LAMMPS parity.

Open work includes the red `regressions/mc/gcmc_lj` (⟨N⟩ below the activity
expansion on the rewritten trial loop; see the specs index), explicit
electronic execution communicator ownership and
subcommunicator propagation (including GPW DFT+U), full PW/GPW borrowed nesting,
shared XC placement currently under cpu/scf, duplicate electrode/metadynamics
implementations, FixMove origin migration, and broader quantitative performance
coverage beyond the measured LJ/EAM/SW cases. LAMMPS algorithm/reference details
are maintained in ../notes/cpu-lammps-alignment.md.
