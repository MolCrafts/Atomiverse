# SCF Reading Map

A short map into the **Tutorial** electronic-structure arc — not a methods
manual. Install and concepts stay in [Getting started](getting-started.md)
and [Core concepts](concepts.md).

All SCF tutorials share the same computational pattern:

1. start from a density guess
2. build an effective matrix from that guess
3. diagonalize the matrix
4. rebuild the density
5. stop when the density stops changing

What changes from chapter to chapter is not the loop itself, but what the
matrix means physically.

## Recommended Reading Order

Read these chapters in order:

1. [Chapter 2: Atomic Orbitals to Single-Point RHF](../tutorial/ch02-rhf.md)
2. [Chapter 3: Unrestricted Hartree-Fock (UHF)](../tutorial/ch03-open-shell.md)
3. [Chapter 5: Unrestricted Kohn-Sham DFT (UKS)](../tutorial/ch05-kohn-sham-dft.md)
4. [Chapter 8: Plane-Wave DFT for Periodic Systems](../tutorial/ch08-plane-wave.md)

Each chapter adds exactly one new idea.

Chapter 2 teaches feedback and convergence with a matrix model that has almost
no chemistry in it.

Chapter 2 replaces the toy matrix with real molecular integrals and introduces
the RHF closed-shell approximation.

Chapter 3 removes the "same spatial orbital for both spins" restriction and
shows how open-shell systems require separate alpha and beta channels.

Chapter 5 keeps the spin-resolved structure of UHF but replaces explicit
Hartree-Fock exchange with exchange-correlation functionals evaluated on a grid.

Chapter 8 moves from molecules to periodic solids, replacing atom-centered
Gaussian basis sets with plane waves and introducing pseudopotentials, FFT
acceleration, k-point sampling, and Ewald summation.

## Which Example Matches Which Tutorial?

| Tutorial | Primary example | Why it matters |
|---|---|---|
| SCF introduction | `examples/01_scf.cpp` | Learn the loop before learning chemistry |
| RHF | `examples/02_rhf.cpp` | First physically meaningful calculation |
| UHF | `examples/03_uhf.cpp` | Open-shell systems and spin contamination |
| UKS | `examples/04_uks.cpp` | DFT, grids, and exchange-correlation potentials |
| Plane-Wave DFT | `examples/08_pw.cpp` | Periodic systems, pseudopotentials, FFT acceleration |

Note: tutorial numbers and example numbers do **not** line up. The example
files are numbered in their own pedagogical order (so plane-wave DFT, which
comes last among the example programs, is `examples/08_pw.cpp` even though it is
Chapter 8). Always follow the "Primary example" column above rather than
assuming the numbers match. Read the theory and the matching code side by side.

## When To Use CPU Solvers Versus CUDA SCF Components

If you are learning the method or validating a result, start with the CPU
solvers. The CPU code is easier to read because the linear algebra and the SCF
logic sit close together.

If you are learning how Atomiverse expresses workflows in its staged execution
model, read the CUDA SCF component path next. That path is more about software
architecture: the SCF loop is split into `build_hamiltonian`, `diagonalize`,
`build_density`, and `convergence_check` phases, and each phase is implemented
by one or more components.

## What To Read Alongside The Tutorials

Pair each tutorial with one code-reading target:

- after Chapter 2, read [`cpu::scf::solve`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/solve.h)
- after Chapter 2, read [`IntegralPack`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/integrals.h), [`SCFSolver`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/scf_solver.h), and [`scf_types.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/scf_types.h)
- after Chapter 3, revisit `SCFSolver` with `uhf_config`
- after Chapter 5, revisit `SCFSolver` with `uks_config` and read [`xc_functional.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/xc_functional.h)
- after Chapter 8, read [`pw_solver.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/pw/pw_solver.h) and [`internal/pw_workspace.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/pw/internal/pw_workspace.h)

If you want to understand how the same ideas map onto the runtime system, then
read the [Developer Manual](../developer/index.md) after finishing Chapter 2.
That is usually the point where `Driver`, `Plan`, and `Component` start to make
sense instead of feeling abstract.
