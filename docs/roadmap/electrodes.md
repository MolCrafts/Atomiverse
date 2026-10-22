# Electrodes and Classical Electrochemistry

Tutorial: [Electrodes](../tutorial/electrodes.md).

Constant-potential electrodes are **Components on `cpu::md::MD`**, not a
driver category. Membership is atom type on the Frame
(`add_electrode(type, psi)`); solved charges write `kAtom_charge`. Domain is
CPU-only. Legend: [Roadmap index](index.md).

Anchors: MetalWalls (`c8ad175d`), LAMMPS ELECTRODE, or literature closed forms
under `md`.

## 1. Charge solvers

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Const-pot BO (Cholesky) | ✓ | ○ | `ConstPotBO` |
| Const-pot CG | ✓ | ○ | Large electrodes |
| MaZe (symplectic / adiabatic) | ✓ | ○ | |
| Per-electrode Ψ + η | ✓ | ○ | |
| Constant charge (conq) | ✓ | ○ | |
| Electroneutrality (electrode / global) | ✓ | ○ | |
| Gaussian electrode charge | ✓ | ○ | |
| Thomas–Fermi metallicity | ✓ | ○ | + Voronoi volumes |
| Charge ASPC predictor | ✓ | ○ | |
| Electrode atom forces / charging work | ✓ | ○ | |
| Frozen electrode atoms | ✓ | ○ | |
| QEq (bulk, ≠ electrode) | ✓ | ○ | See also [Force Fields](force-fields.md) |
| MPI charge halo | ✓ | ○ | `const-pot-charge-halo-mpi` |

## 2. Fields and mechanics

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Piston electrode (P_z) | ✓ | ○ | |
| Finite-field + single electrode | ✓ | ○ | Dufils |
| Displacement D-field | ✓ | ○ | |
| Constant-D polarization feedback | ✓ | ○ | |

General E-fields: [MD § Nonequilibrium](molecular-dynamics.md#4-nonequilibrium).

## 3. Polarizable coupling

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| PIM + XL dipoles | ✓ | ○ | |
| Dipole ⊕ electrode joint SCF | ✓ | ○ | No Ewald yet |
| AIM / DAIM | ○ | ○ | Blocked |

## 4. Observers

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Electrode Q(t) | ✓ | ○ | |
| Capacitance from ⟨δQ²⟩ | ✓ | ○ | Limmer closed form |
| Potential profile Ψ(z) | ✓ | ○ | |
| Voronoi volume / classical P | ✓ | ○ | |
| C_QQ(τ) / elastance | ◐ | ○ | No standing regression; self-consistency cases removed |

## Gaps / out of scope

| Item | Status |
| ---- | ------ |
| Full 2D-PBC Ewald electrochemical cell | ○ (Yeh–Berkowitz slab only) |
| ML heterogeneous electrodes | Blocked |
| PLUMED bridge | In-tree `cpu/bias/` instead |
