# General GAFF MD vs LAMMPS (CPU) and OpenMM (CUDA)

Harness and protocol: `benchmarks/gaff/README.md`. System B is the
general-GAFF liquid (toluene / NMA / ethanol / EC, 24,840 atoms, fourier
torsions, cvff impropers, X–H RATTLE, 2 fs). System A (PEO/LiTFSI) stays in
`tools/bench_gaff_lammps/`.

## Fairness rules that are easy to get wrong

- Constraints are compared as RATTLE (LAMMPS `fix rattle`), never against
  `fix shake`: ATV `FixRattle` and OpenMM also constrain velocities.
- Thermostat chain length is 1 in all three codes (ATV `NoseHoover` is
  single-chain; LAMMPS `tchain 1`; OpenMM `NoseHooverIntegrator` chain 1).
- LJ is unshifted and untailed everywhere; `run_atv_gaff` takes `shift`
  from the ff and otherwise uses the LAMMPS default (off).
- The PME/PPPM mesh, order and α are explicit in every code. At 64³ /
  order 5 / α 0.35 all three reach 3–4e-5 relative RMS force error against
  Ewald, so same-settings timing is equal-accuracy timing.
- Atomiverse timing tables are cumulative per driver; subtract the warmup
  table (`summarize_cpu.py` does).

## Current state

Numbers live in `benchmarks/gaff/README.md` (G1 table, preliminary G2,
OpenMM GH200 ns/day). Arrhenius has the OpenMM GPU baseline only; the ATV vs
LAMMPS CPU matrix and the 100 ps G2 run are still open
([`../specs/gaff-m1-open-work.md`](../specs/gaff-m1-open-work.md)). Machine
paths: [arrhenius-workbench.md](arrhenius-workbench.md).

Acceptance for milestone 1: G1 and G2 green in all codes; CPU steps/s ≥
LAMMPS at np = 1 and at the best full-node MPI×OMP layout; CUDA ns/day ≥
OpenMM on one GH200 for ≥ 100k atoms, double and mixed separately (small
cells reported, not gated).

M4 workbench findings:

- G1 green on CPU: bonded terms agree with LAMMPS to 1e-14, nonbonded to
  3e-6 (mesh-method difference), forces 6e-5 of RMS.
- One rank: ATV is at or ahead of LAMMPS overall (Kspace and Neigh faster,
  Pair at parity). Slower sections, by measured cost: bonded kernels
  (~1.8× LAMMPS, ~1.5 ms/step, dominated by `AngleHarmonic` acos and
  per-kernel thread scratch), RATTLE (~1.4× vs `fix rattle`), Comm on one
  rank (self-image swaps, ~0.3 ms/step).
- Threaded arm: every bonded kernel owns a full-length thread-private force
  buffer and reduces it on its own; LAMMPS `package omp` reduces once per
  step for all styles. This is the structural reason Bond scales badly with
  threads.
- Mac timings vary ±30% between runs (thermal state, efficiency cores); use
  them for direction only.

## CUDA

Nothing CUDA-side is GAFF-capable yet: single-type LJ, special_bonds only
with equal weights, no exclusion correction in `CoulLong`, no periodic
torsion / improper kernels, no constraints. The cluster-vs-brute-force red
in `cuda-openmm-md-perf.md` blocks every CUDA timing claim.
