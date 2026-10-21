# Multi-GPU velocity-Verlet trajectory and topology transport

This public-library consumer checks that domain decomposition preserves the
single-GPU velocity-Verlet trajectory and transports the complete bonded
topology when atoms migrate. It requires at least two visible CUDA GPUs and
returns failure if that hardware is unavailable.

The integration reference is W. C. Swope, H. C. Andersen, P. H. Berens and
K. R. Wilson, *A computer simulation method for the calculation of equilibrium
constants for the formation of physical clusters of molecules: Application to
small water clusters*, J. Chem. Phys. **76**, 637–649 (1982),
[doi:10.1063/1.442716](https://doi.org/10.1063/1.442716).
The decomposition invariant is the same physical trajectory under a partition
of the atoms, as described by S. Plimpton, *Fast Parallel Algorithms for
Short-Range Molecular Dynamics*, J. Comput. Phys. **117**, 1–19 (1995),
[doi:10.1006/jcph.1995.1039](https://doi.org/10.1006/jcph.1995.1039).

The four cases retain the original fixtures, integration steps and assertions:

| Case | Atoms | Steps | Position tolerance |
|---|---:|---:|---:|
| `NveTrajectoryParity` | 64 | 10 | 0.005 Å per component |
| `NveTrajectoryParityNlist` | 512 | 12 | 0.005 Å per component |
| `NveMigrationParity` | 512 | 20 | 0.02 Å per component |
| `MigratePreservesBondedTopology` | 64 | 20 | Exact equality of bond-tag sets |

Run `reg_cuda-multi-gpu-trajectory CASE` to check one case, or omit `CASE` to
run all four. The migration fixture uses the original +x drift and requires
the complete topology after every-step migration; no tolerance is applied to
the bond-tag set. The position checks use the serial GPU as their numerical
reference, rather than a frozen trajectory from another engine.
