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

1. [Chapter 1: The bare SCF loop](../tutorial/ch01-foundations.md)
2. [Chapter 2: Atomic Orbitals to Single-Point RHF](../tutorial/ch02-rhf.md)
3. [Chapter 3: Unrestricted Hartree-Fock (UHF)](../tutorial/ch03-open-shell.md)
4. [Chapter 5: Unrestricted Kohn-Sham DFT (UKS)](../tutorial/ch05-kohn-sham-dft.md)
5. [Chapter 8: Plane-Wave DFT for Periodic Systems](../tutorial/ch08-plane-wave.md)

Each chapter adds exactly one new idea.

Chapter 1 teaches feedback and convergence with a matrix model that has almost
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

## Which Tutorial Teaches What

| Tutorial | Focus |
|---|---|
| [Ch. 1](../tutorial/ch01-foundations.md) | The bare SCF loop, no chemistry |
| [Ch. 2](../tutorial/ch02-rhf.md) | Closed-shell RHF single point |
| [Ch. 3](../tutorial/ch03-open-shell.md) | UHF, spin contamination, ROHF/GHF |
| [Ch. 5](../tutorial/ch05-kohn-sham-dft.md) | Kohn–Sham DFT, grids, XC |
| [Ch. 8](../tutorial/ch08-plane-wave.md) | Plane-wave DFT, pseudopotentials, k-points |

Each chapter embeds the Practice snippet; runnable demos under `examples/` are
optional and self-documented.

## When To Use CPU Solvers Versus CUDA SCF Components

If you are learning the method or validating a result, start with the CPU
solvers. The CPU code is easier to read because the linear algebra and the SCF
logic sit close together.

If you are learning how Atomiverse expresses workflows in its staged execution
model, read the staged SCF next. That path is more about software
architecture: a concrete method such as `cpu::scf::RHF` **is** a
`cpu::Driver<scf::Phase>`, and its loop is five typed stages — `fock`, `mix`,
`diagonalize`, `density`, `check` — each a Component owned as a direct member
under `cpu/scf/stages/`.

## What To Read Alongside The Tutorials

Pair each tutorial with one code-reading target:

- after Chapter 1, read [`cpu::scf::solve`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/solve.h)
- after Chapter 2, read [`methods.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/methods.h) (the concrete `RHF` / `UHF` / `RKS` / `UKS`) and the loop they share, [`mean_field_method.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/mean_field_method.h)
- after Chapter 3, revisit `UHF` — the same class, with `n_alpha` / `n_beta` set as members
- after Chapter 5, revisit `UKS` and read [`xc_functional.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/xc_functional.h)
- after Chapter 8, read [`pw_solver.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/planewave/pw_solver.h) and [`internal/pw_workspace.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/planewave/internal/pw_workspace.h)

If you want to understand how the same ideas map onto the runtime system, then
read the [Developer Manual](../developer/index.md) after finishing Chapter 2.
That is usually the point where `Driver`, `Plan`, and `Component` start to make
sense instead of feeling abstract.
