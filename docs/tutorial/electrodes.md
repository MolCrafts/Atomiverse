# Electrodes

Tutorial twin of the [Electrodes roadmap](../roadmap/electrodes.md).

Constant-potential and constant-charge electrodes are **MD Components**, not
a separate driver. This page is the short physical entry; the status tables
and MetalWalls / LAMMPS anchors live on the roadmap.

## What problem this solves

An electrode in a classical simulation must hold a prescribed electrostatic
potential (or charge) while the electrolyte and solvent move. That means the
electrode atom charges are unknowns solved each step (or every few steps),
not fixed inputs.

## Roadmap spine

| Topic | Roadmap |
| ----- | ------- |
| Const-pot BO / CG / MaZe | [Electrodes §1](../roadmap/electrodes.md#1-charge-solvers) |
| Piston, finite field, D-field | [Electrodes §2](../roadmap/electrodes.md#2-fields-and-mechanics) |
| PIM ⊕ electrode | [Electrodes §3](../roadmap/electrodes.md#3-polarizable-coupling) |
| Observers (Q(t), capacitance, Ψ(z)) | [Electrodes §4](../roadmap/electrodes.md#4-observers) |

Related pedagogy: [Molecules and environment](molecular-environment.md)
(fields, continuum solvent, QM/MM) and [classical MD](ch10-classical-md.md).

## Practice pointers

- Compose under `cpu::md::MD` with `add_electrode(type, psi)` on the Frame.
- Worked regression reports: [method catalog](methods/catalog.md) (search
  `const-pot`, `maze`, `pim`, `electrode`).
- Domain is CPU-only today.
