# CUDA MD — device-controlled step loop (MPS packing)

Goal: a single-GPU `cuda::md::MD` run whose steady step never waits on the
device, computes only what a consumer asked for, and issues as few launches as
the algorithm needs — so several packed processes under NVIDIA MPS keep the GPU
busy instead of waiting on their own host round trips. Algorithmic reference is
OpenMM (cluster/tile engine, device-side list maintenance); the thermo /
`ev_set` surface is LAMMPS. The user authorized the full plan; work it in phase
order and do not report it done until every phase is.

Scope: single-GPU `cuda::md::MD` through `cuda::Backend`. Out of scope:
`MultiGpu` loops (retired by [md-unification.md](md-unification.md) item 5),
minimizers, ES methods. This spec carries md-unification item 4 for the
single-GPU MD loop.

## Law (P0, lands first)

**CUDA-only = no device→host synchronization inside a CUDA run loop.**

| | Rule |
| --- | --- |
| Forbidden on the stepping thread between setup and the end-of-run retrieve | `cudaStreamSynchronize`, `cudaEventSynchronize`, `cudaDeviceSynchronize`, synchronous `cudaMemcpy`/`cudaMemset`, D2H into pageable memory, `cudaMalloc`/`cudaFree`/`cudaMallocHost`/`cudaFreeHost`, any host branch on a device-computed value of the current step |
| Allowed | `cudaMemcpyAsync` into pinned memory on a stream; non-blocking `cudaEventQuery`; reads of pinned/mapped host words whose producer completion was observed by a query; host workers that wait off the stepping thread |
| Control decisions | Made on the device for the current step (rebuild, overflow, fault, stop). The host learns them **late and asynchronously**; the device state freezes at the deciding step so a late host is never wrong |
| Errors | The error path may synchronize once to report (throw) — never the success path |

Harness: iron law 10 in `docs/developer/project-contract.md`, the MD section of
`.claude/notes/cuda-only-no-dh.md`, the corrected skin-gate bullet in
`.claude/notes/cuda-openmm-md-perf.md`. Evidence (not a ctest, per iron law 1):
`tools/diagnostics/md-residency/` — an Nsight capture of steady MD steps
between `cudaProfilerStart/Stop`, audited by `trace.py` for zero forbidden
runtime calls on the stepping thread and zero D2H except declared async pinned
mirrors.

## Status (2026-10-29, snapshot p5 on GH200, CUDA 13.2.1)

- Verified: 358/358 unit tests; md-residency audit 0 forbidden calls in the
  step loop for lj-nve / lj-nh / lj-npt / ion-pme; lj-nve bitwise equal to the
  pre-change tree, NH / NPT / PME within the baseline's own FP64 atomic
  run-to-run noise, identical rebuild counts. Results:
  `tools/diagnostics/md-residency/results/p4/` (audits),
  `benchmarks/cuda/mps_pack/results/p5.txt` (MPS table).
- Launches: LJ NVE/NH and LJ/coul/long + PME run as ~1.3 host launches and
  ~2 runtime calls per step (baseline 6–10 launches, 20–30 calls): segment
  graphs with the neighbor IF node and the PME graph spliced in, the tile
  repack as a neighbor rebuild hook. NoseHooverNPT is not capturable yet
  (23 eager launches, still sync-free).
- Single process (µs/step, base → p5): lj-nve 157.7 → 152.9, lj-nh 162.1 →
  158.0, lj-npt 290.5 → 274.3, ion-pme 102.5 → 98.2.
- MPS K=8 aggregate (base → p5): lj-nve +5–7 % at 2k/8k/32k atoms; ion-pme
  2k +4 %, 8k −4 %, 32k −3 %. GPU-saturated ion-pme still pays ~2.6 µs/step
  more device work: the device skin gate (+1.2 µs vs the old flag kernel) and
  the device step advance node (+1.4 µs).
- CUDA 13.2 facts: a graph with conditional nodes instantiates once (capture
  per GraphRing slot); re-launching one exec blocks the host until it
  retires (hence GraphRing); every separately launched conditional graph
  costs ~16 µs idle (hence splicing into the step graph).
- Open: P3.2 (shared live cluster positions), P3.3 (kick + KE fusion), NPT
  capturable (device ramp/step), cheaper skin gate and step advance, Python
  exposure of thermo keywords, CPU last-step thermo line.
- Pre-existing bugs fixed on the way: `CudaPairLJSpecialBonds` expected
  step-0 energy after one step; `ZeroEnergy` skipped `energy_improper`;
  `stress_kinetic` was never cleared without a `Virial` observer;
  `PressureSeries` never sampled for `every > 1`.

## Current violations (audit, 2026-10-29)

| Site | Per step? | Fix phase |
| --- | --- | --- |
| `neighbor/neighbor_list.cu` skin gate `cudaEventSynchronize` | every step | P1 |
| `neighbor/neighbor_list.cu` CSR count D2H + `cudaStreamSynchronize` | rebuild steps | P1 |
| `neighbor/neighbor_list.cu` flat export sync `cudaMemcpy` + `cudaEventSynchronize` | every step (flat consumers) | P1 |
| `potentials/pair_lj_cut_coul_long.cu` `pack_tiles` count D2H + sync | rebuild steps | P1 |
| host `rebuild_count` / `rebuild_generation` bookkeeping | rebuild steps | P1 |
| `record_base.cuh` / `async_bridge.cu` worker events spin (no `BlockingSync`) | sampled steps | P4 |

Build-time syncs (`build()`, `InitVelocity`, warmups) are allowed by the law.

## P1 — device-resident control

1. **Skin gate on device.** `atom_displacement_max` writes one device flag
   (replace the mapped per-block flag array). The rebuild pipeline is a graph
   instantiated at build whose body sits under a CUDA conditional IF node; the
   displacement kernel sets the handle with `cudaGraphSetConditional`. One
   `cudaGraphLaunch` per step covers check + optional rebuild. Same decision
   law as today (current-step minimum-image displacement vs half skin, cell
   change forces a rebuild) → identical rebuild steps, bitwise-identical
   trajectories.
2. **Count-free rebuild.** No host read of the CSR total. Kernels after the
   scan (`fill_cluster_pairs`, `compute_csr_masks`, exclusion bake) size their
   grid from the build-time capacity and read the device total (early exit /
   grid-stride). Rebuild generation and rebuild count are device counters,
   read once at the end of the run.
3. **Overflow as a device fault.** Total > capacity → set
   `ForceFault::kNeighborCapacity`, skip writes. The integrators already refuse
   faulted forces; the run reports at its end with the faulting step and the
   required reserve (today's message). Auto-regrow + replay is not in scope.
4. **Fault freeze is complete.** Every kernel that mutates x, v, box or
   extended state guards on the fault word, not only the integrators:
   NoseHoover NVT scaling, Berendsen barostat, Deform, ZeroMomentum, SLLOD
   thermostat, NHC chains. A frozen run stays at its faulting step.
5. **Async fault/stop observation.** Each step appends an async copy of the
   fault word into a pinned ring on the defer stream; the host polls it with
   `cudaEventQuery` and stops issuing steps when set. `retrieve()` is the only
   synchronization of the run. `StopOnNaN` reads a metric already brought back
   asynchronously.
6. **Packed tiles on device.** `PairLJCutCoulLong` repacks inside its own IF
   graph keyed by the neighbor's device "rebuilt this step" flag; the fused
   tile kernel sizes its grid by packed capacity and reads the device count.
7. **Flat export without host count.** Prune/export grids from capacity with
   the device counter; no sync, no legacy-stream copy.

Acceptance: md-residency audit shows zero forbidden calls for LJ NVE, LJ +
NoseHoover, GAFF PEO (`PairLJCutCoulLong` + PME + bonded) NVE and NPT;
trajectories and rebuild steps bitwise equal to the pre-P1 tree on GH200
double and mixed; an overflow case reports the faulting step.

## P2 — on-demand energy and virial (LAMMPS `thermo` / `ev_set`)

1. **Interface.** `set_thermo(int every, std::vector<std::string> keywords)`
   on both backends (one entry; the default keyword list prints today's
   columns). Keywords follow LAMMPS `thermo_style custom`: `step temp pe ke
   etotal econserve epair ebond eangle edihed eimp press pxx pyy pzz pxy pxz
   pyz vol`. Unknown keywords throw at configuration. The last step of each run
   is printed, as in LAMMPS.
2. **Demand.** Components declare what they consume and when
   (`Demand{energy, virial, every}`; thermo derives it from its keywords, NPT
   and barostats require the virial every step, pressure recorders at their
   cadence). The backend computes per-step `eflag`/`vflag` as the union over
   consumers whose cadence hits the step (LAMMPS `Integrate::ev_set`) and
   publishes them on `LaunchContext`. The build-time `set_stress_required`
   mechanism is removed — one mechanism.
3. **Producers obey.** Pair, Coulomb, bonded, PME/Ewald kernels take
   `<Energy, Virial>` template flags and skip accumulation, block reductions
   and global atomics when off. Zeroing clears only the requested scalars.
4. **CPU parity** for the interface and printed columns; CPU energy gating is
   not required.

Acceptance: printed values equal the always-on path at every printed step
(FP64 reduction order tolerance 1e-12 relative); forces and trajectories
bitwise unchanged; `lj_force_tile` time drops on non-thermo steps.

## P3 — fewer launches, whole-step graphs

1. **Zeroing.** One allocation for fx|fy|fz (stride = capacity) → one memset.
   Energy and virial scalars zeroed by one memset over the requested range
   (also clears `energy_improper`, which `ZeroEnergy` currently skips).
   `ZeroKinetic` disappears: the KE kernel writes block partials and the last
   block stores the result.
2. **Live cluster positions.** The per-step displacement kernel also scatters
   current positions into the cluster order (a rebuild step's scatter
   overwrites them). Pair styles drop their own per-step gather;
   `PairLJCutCoulLong` gathers parameters only at rebuild.
3. **Kick + KE fusion.** When no velocity writer is bound between
   `final_integrate` and the KE slot, the integrator's final kick computes KE
   and kinetic stress in the same kernel and the separate KE launch is
   skipped.
4. **Whole-step graphs.** The backend captures each distinct step signature
   (active-op mask × eflag × vflag) once and replays it. Launch purity is
   required: no host-side per-step state baked into kernel arguments — the
   step index lives in a device counter advanced by the graph tail; ramp
   targets, Langevin keys, Deform strain and SLLOD flow derive from it on the
   device. The neighbor IF graph and PME graph become child nodes. Components
   whose launch needs a host worker (async recorders) declare themselves not
   capturable; steps where they fire run eagerly, still sync-free.

Acceptance: steady LJ NVE step = one `cudaGraphLaunch`; GAFF PEO non-sampled
step = one launch; trajectories bitwise equal to P2.

## P4 — remaining host costs and the MPS measurement

1. Events waited on by worker threads use `cudaEventBlockingSync`, so packed
   processes with one core do not have two threads spinning.
2. `benchmarks/cuda/mps_pack/`: K = 1, 2, 4, 8 processes on one GPU under
   `nvidia-cuda-mps-control`, LJ 2k / 8k / 32k atoms and GAFF PEO 1,840 atoms;
   aggregate ns/day and per-process host API time per step, before (current
   `dev`) and after each phase.

## Validation

- GH200 via `molq`, double and mixed: CUDA MD primitives, neighbor, LJ,
  LJ/coul/long, PME, NH/NPT/SLLOD/Langevin/Deform unit targets; memcheck on
  the neighbor graph; racecheck on the captured step.
- Bitwise trajectory comparison scripts live with the md-residency diagnostic.
- No "MPS faster" claim without the P4 table.
