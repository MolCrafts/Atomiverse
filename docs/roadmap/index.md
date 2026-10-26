# Scientific Methods Roadmap

Honest inventory of what Atomiverse can compute today: method × backend ×
status. Rows are marked implemented only when the method is reachable from the
public API **and** pinned by an end-to-end pinned case. Snapshot
against **`dev`** (the development branch; `master` is the released snapshot).

Each roadmap chapter has a **tutorial twin** (theory + practice). The tables
below are the same spine as the Tutorial nav — status here, pedagogy there.

| Roadmap | Tutorial twin |
| ------- | ------------- |
| [Hartree–Fock](hartree-fock.md) | [HF](../tutorial/ch02-rhf.md), [open shell](../tutorial/ch03-open-shell.md), [basis](../tutorial/ch04-basis-sets.md), [SCF](../tutorial/ch18-scf-techniques.md) |
| [DFT](dft.md) | [KS DFT](../tutorial/ch05-kohn-sham-dft.md), [hybrids](../tutorial/ch06-hybrid-dft.md), [PW](../tutorial/ch08-plane-wave.md), [GPW](../tutorial/ch09-gpw.md) |
| [Post-Hartree–Fock](post-hf.md) | [Correlation](../tutorial/ch07-correlation.md) |
| [Excited States](excited-states.md) | [Electronic excitations](../tutorial/electronic-excitations.md) |
| [Semiempirical](semiempirical.md) | [AM1 / SE](../tutorial/ch17-semiempirical-am1.md) |
| [Force Fields](force-fields.md) | [Many-body potentials](../tutorial/ch16-many-body-potentials.md), [classical MD](../tutorial/ch10-classical-md.md) |
| [Molecular Dynamics](molecular-dynamics.md) | [MD](../tutorial/ch10-classical-md.md), [ensembles](../tutorial/ch11-ensembles.md), [vibrations](../tutorial/ch14-vibrations.md) |
| [Electrodes](electrodes.md) | [Electrodes](../tutorial/electrodes.md) |
| [Geometry Optimization](optimization.md) | [Geometry optimization](../tutorial/ch12-geometry-optimization.md) |
| [Sampling](sampling.md) | [Sampling](../tutorial/sampling.md) |
| [Multiscale](multiscale.md) | [Environment](../tutorial/molecular-environment.md), [charges](../tutorial/ch20-hf-properties-environment.md) |
| [Observables](observables.md) | [Trajectory analysis](../tutorial/ch13-observables.md) |

Tutorial-only foundations (no status table): [systems](../tutorial/ch01-foundations.md),
[math](../tutorial/mathematical-tools.md), [workflows](../tutorial/ch00-architecture.md),
[comparing models](../tutorial/ch15-cross-method.md).

Pin / API detail for one regression: [tutorial/methods](../tutorial/methods/catalog.md).
Usage: [User Guide — methods](../user-guide/methods.md).

## Status

| Symbol | Meaning |
| ------ | ------- |
| ○ | Not implemented on that backend |
| ◐ | Partial: foothold, restricted envelope, or incomplete external pin |
| ✓ | Public API + end-to-end pin (external oracle or analytic) |
| ⚡ | ✓ plus a standing performance harness vs a production twin |

**✓** requires a cited literature / oracle number in a ≤ 60 s out-of-library
consumer. Unit tests alone are not enough.

**⚡** is scarce. Today it is only the Lennard–Jones 12-6 pair style
(CPU xsimd + CUDA cluster/tile, vs LAMMPS/Kokkos in `benchmarks/lj/` and
`benchmarks/cuda/`). Correct-but-unbenchmarked code stays ✓.

**◐** never jumps to ✓ until the chapter’s Gaps row is closed.

**○** does not mean “planned” or “rejected” — it is a census, not a schedule.

### Dominance rule

A method that a strictly better algorithm dominates (same purpose, same or
lower cost, no winning regime) is **not listed**. Dominated code may still
ship for protocol reproduction (e.g. Berendsen) — that is a docstring concern,
not a roadmap row.

## Backends

| Backend | Role |
| ------- | ---- |
| **CPU** | Primary portable path (OpenMP / MPI; xsimd where used) |
| **CUDA** | Device path for selected MD and SCF/PW/GPW kernels (GH200) |
| **Kokkos** | Serial (CPU) execution-space foothold only. Not a method column. |

A CUDA ✓ requires the **principal** work of that method to run on device.
A CPU fallback does not count. Kokkos is not a table column — tables stay
**CPU \| CUDA \| Notes**. `ATV_BUILD_KOKKOS` builds one target,
`atomiverse_kokkos`: the shifted LJ 12-6 pair kernel on `Kokkos::Serial`
(LAMMPS `lj/cut/kk` host `parallel_reduce`). That kernel is ◐, not a ✓,
and it is not an MD backend.
