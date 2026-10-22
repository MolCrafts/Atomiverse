# Principles and Practice

This manual begins with physical questions and mathematical foundations,
then develops small, interpretable Atomiverse calculations. The original
Markdown files are the only authoritative textbook source.

Each **method-family** chapter has a **roadmap twin** — the status tables
(CPU / CUDA / Notes). Foundations below have no roadmap page.

| Tutorial | Roadmap twin |
| -------- | ------------ |
| [HF](ch02-rhf.md), [open shell](ch03-open-shell.md), [basis](ch04-basis-sets.md), [SCF](ch18-scf-techniques.md) | [Hartree–Fock](../roadmap/hartree-fock.md) |
| [KS DFT](ch05-kohn-sham-dft.md), [hybrids](ch06-hybrid-dft.md), [PW](ch08-plane-wave.md), [GPW](ch09-gpw.md) | [DFT](../roadmap/dft.md) |
| [Correlation](ch07-correlation.md) | [Post-Hartree–Fock](../roadmap/post-hf.md) |
| [Electronic excitations](electronic-excitations.md) | [Excited States](../roadmap/excited-states.md) |
| [Semiempirical](ch17-semiempirical-am1.md) | [Semiempirical](../roadmap/semiempirical.md) |
| [Many-body](ch16-many-body-potentials.md), [classical MD](ch10-classical-md.md) | [Force Fields](../roadmap/force-fields.md) |
| [MD](ch10-classical-md.md), [ensembles](ch11-ensembles.md), [vibrations](ch14-vibrations.md) | [Molecular Dynamics](../roadmap/molecular-dynamics.md) |
| [Electrodes](electrodes.md) | [Electrodes](../roadmap/electrodes.md) |
| [Geometry optimization](ch12-geometry-optimization.md) | [Optimization](../roadmap/optimization.md) |
| [Sampling](sampling.md) | [Sampling](../roadmap/sampling.md) |
| [Environment](molecular-environment.md), [charges](ch20-hf-properties-environment.md) | [Multiscale](../roadmap/multiscale.md) |
| [Trajectory analysis](ch13-observables.md) | [Observables](../roadmap/observables.md) |

Read the [preface](preface.md) for scope and evidence conventions, and
[maintenance guidance](MAINTENANCE.md) before extending the manual.

## Reading order

The old `chNN` filenames remain stable identifiers, not printed chapter
numbers. New subjects use descriptive filenames.

<!-- reading-order-start -->

### Foundations (tutorial only)

| Order | Chapter |
| --- | --- |
| 1 | [Systems and Units](ch01-foundations.md) |
| 2 | [Mathematical Tools](mathematical-tools.md) |
| 3 | [Simulation Workflows](ch00-architecture.md) |

### Hartree–Fock

| Order | Chapter |
| --- | --- |
| 4 | [Basis Sets](ch04-basis-sets.md) |
| 5 | [Hartree–Fock Theory](ch02-rhf.md) |
| 6 | [Open-Shell Systems](ch03-open-shell.md) |
| 7 | [Self-Consistency](ch18-scf-techniques.md) |

### Density-functional theory

| Order | Chapter |
| --- | --- |
| 8 | [Density Functional Theory](ch05-kohn-sham-dft.md) |
| 9 | [Hybrid Functionals](ch06-hybrid-dft.md) |
| 10 | [Plane Waves](ch08-plane-wave.md) |
| 11 | [Gaussian and Plane Waves](ch09-gpw.md) |

### Post-Hartree–Fock and response

| Order | Chapter |
| --- | --- |
| 12 | [Electron Correlation](ch07-correlation.md) |
| 13 | [Electronic Excitations](electronic-excitations.md) |
| 14 | [Semiempirical Models](ch17-semiempirical-am1.md) |

### Classical motion and force fields

| Order | Chapter |
| --- | --- |
| 15 | [Molecular Dynamics](ch10-classical-md.md) |
| 16 | [Statistical Ensembles](ch11-ensembles.md) |
| 17 | [Many-Body Potentials](ch16-many-body-potentials.md) |
| 18 | [Electrodes](electrodes.md) |

### Structure, sampling, measurement

| Order | Chapter |
| --- | --- |
| 19 | [Geometry Optimization](ch12-geometry-optimization.md) |
| 20 | [Sampling and Free Energy](sampling.md) |
| 21 | [Trajectory Analysis](ch13-observables.md) |
| 22 | [Molecular Vibrations](ch14-vibrations.md) |
| 23 | [Charges and Multipoles](ch20-hf-properties-environment.md) |
| 24 | [Molecules and Environment](molecular-environment.md) |
| 25 | [Comparing Models](ch15-cross-method.md) |
<!-- reading-order-end -->

## Learning routes

- A first trajectory: systems → mathematics → workflows → molecular dynamics
  → ensembles → trajectory analysis.
- A first molecular electronic calculation: systems → mathematics → basis
  sets → HF → open shells → self-consistency → geometry optimization.
- A comparative project: validate a small force model, then use the final
  model-comparison chapter to separate geometry, Hamiltonian, and sampling effects.

## Reference material

- [CPU workflow reference](../developer/cpu-workflow-reference.md)
- [Many-body potentials (EAM, SW)](methods/md-manybody-potentials.md)
- [Method catalog](methods/catalog.md)
- [Method report policy](methods/README.md)
- [External references](appendix-pyscf-oracle.md)
- [User guide](../user-guide/index.md)
- [Developer guide](../developer/index.md)
- [Roadmap](../roadmap/index.md): capability × backend status (1:1 with method chapters above)
