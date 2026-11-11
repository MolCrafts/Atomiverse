# General GAFF MD vs LAMMPS (CPU) and OpenMM (CUDA)

Harness: `benchmarks/compare/gaff-liquid` (results not in git). Open work:
[`../specs/gaff-m1-open-work.md`](../specs/gaff-m1-open-work.md). Machines:
[arrhenius-workbench.md](arrhenius-workbench.md). System B is the general-GAFF
liquid (toluene / NMA / ethanol / EC, 24,840 atoms, fourier torsions, cvff
impropers, X–H RATTLE, 2 fs). System A (PEO/LiTFSI) stays in
`benchmarks/compare/gaff-peo-lammps/`.

## Fairness rules

- Constraints compare as RATTLE (LAMMPS `fix rattle`), never `fix shake`: ATV
  `FixRattle` and OpenMM also constrain velocities.
- Thermostat chain length 1 in all three codes (ATV `NoseHoover` is
  single-chain; LAMMPS `tchain 1`; OpenMM `NoseHooverIntegrator` chain 1).
- LJ unshifted and untailed everywhere; `run_atv_gaff` takes `shift` from the
  ff, otherwise the LAMMPS default (off).
- PME/PPPM mesh, order and α explicit in every code. At 64³ / order 5 / α 0.35
  all three reach 3–4e-5 relative RMS force error vs Ewald, so same-settings
  timing is equal-accuracy timing.
- ATV timing tables are cumulative per driver; subtract the warmup table
  (`summarize_cpu.py` does).

## Milestone-1 acceptance

G1 and G2 green in all codes; CPU steps/s ≥ LAMMPS at np = 1 and at the best
full-node MPI×OMP layout; CUDA ns/day ≥ OpenMM on one GH200 for ≥ 100k atoms,
double and mixed separately (small cells reported, not gated).

## CPU findings (M4)

- G1 green: bonded terms match LAMMPS to 1e-14, nonbonded to 3e-6 (mesh
  method), forces 6e-5 of RMS.
- One rank: ATV at or ahead of LAMMPS overall (Kspace, Neigh faster; Pair at
  parity). Slower: bonded kernels (~1.8×, `AngleHarmonic` acos and per-kernel
  thread scratch), RATTLE (~1.4× vs `fix rattle`), one-rank Comm (self-image
  swaps).
- Threaded: the Runtime owns one set of chunk-owned force slots and partial
  sums (`md::ChunkForces`, reached through `md::ForcePass`) reduced in slot
  order once per slot walk, as LAMMPS `package omp`; every pair, bonded and
  many-body style uses it, so a run is bit-identical at a fixed thread count
  (not across thread counts: the chunk count follows the team). No OpenMP
  floating-point `reduction` in a force kernel. Bonded caches key on
  `Frame::row_revision()`. Pair styles share `md::PairLoop`; PPPM/PME spread,
  Green's sum and gather are threaded (x-plane slabs) with FFTW threads;
  their energy / virial fold is a fixed 64-chunk `device::ChunkedRange`
  (thread-count independent);
  SETTLE is analytic and threaded. Timings: wip/cpu-md-perf commit messages
  (LAMMPS bench data.rhodo through run_atv_gaff, `--settle` for waters).
- Every CPU pair style walks `md::PairLoop` (lj/cut and coul/cut keep
  their xsimd ket loops as the loop's row `walk`; soft-core dU/dλ and DPD's
  conservative-only virial ride `PairTermX`).
- Open: no CPU energy-request gate; PME still Allreduces a c2c mesh to
  rank 0.

## CUDA

- Present: `PairLJCutCoulLong` (per-type LJ, Amber exceptions restored by a
  sparse pass, reciprocal exclusion correction), `DihedralMultiHarmonic`
  (fourier phase 0/π, n ≤ 4), PME; `reg_cuda-gaff-fragment` pins the force pass
  to OpenMM CUDA.
- Missing for system B: impropers and constraints (no CUDA RATTLE/SETTLE),
  CHARMM torsions. Method: [cuda-openmm-md-perf.md](cuda-openmm-md-perf.md).
