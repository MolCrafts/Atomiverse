# Targeted validation on Arrhenius

All builds, numerical regressions and performance runs were submitted to
Arrhenius GPU batch allocations. No scientific thresholds were relaxed.
This is a focused validation of the changed paths, not the whole repository.

- CUDA mixed/default 64-bit tags, single/32-bit tags and double: focused
  Frame conversion, motion residual, boundary images, atom permutation,
  scalar reset, minimizer rollback and asynchronous snapshot units.
- CPU: fresh-force preparation, actual initial NPT kinetic energy and
  independent Langevin force-sample units. Two-rank MPI checks cover the
  kinetic/stress reduction and reverse-force communication.
- Two GPUs: state/force communication units, NVE trajectories with and
  without neighbor lists, migrating atoms and bonded topology/image transfer.
- Scientific pins: CPU/CUDA first velocity-Verlet step; NPT volume parity;
  analytic first-step Langevin velocity variance; complete GAFF fragment;
  NaCl PME/Ewald; even/odd and skew PME against an independent full Fourier
  energy and force oracle. The final PME mapping and packed pair changes
  also pass their focused checks in single, mixed and double modes.
- Mixed HF compatibility: density/capture/header units, H2 RHF, H2O RHF
  energy/gradient and H2+, H2 and H2O+ UHF pins. H2O maximum gradient error
  is `1.959e-8 Ha/Bohr` against the original `5e-6` limit; net gradient is
  `7.928e-16 Ha/Bohr`. Integral/gradient accumulation stays FP64 until the
  final Frame force cast. No force-evaluation allocation is introduced.
- Scaling: all eight precision/size comparisons pass initial energy and
  forces, final energy and periodic positions. Throughput and memory runs
  are separate. Exact inputs, binary identities, process durations and
  resource sampling metadata are retained in each summary.

Focused Slurm records: 4148365/4148366 (state and scientific checks),
4149435 (two-GPU migration), 4172747 (single/double Langevin), 4172858
(PME mapping), 4173342 (packed-pair A/B and science), 4173894 (final scaling
and single-mode pair checks), 4173979 (mixed HF compatibility).
