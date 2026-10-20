# Molecular Dynamics

Tutorial: [MD](../tutorial/ch10-classical-md.md) · [ensembles](../tutorial/ch11-ensembles.md) · [vibrations](../tutorial/ch14-vibrations.md).

`cpu::md::MD` / `cuda::md::MD`. NVE/NVT/NPT on both backends. Legend:
[Roadmap index](index.md). Electrode Components:
[Electrodes](electrodes.md).

## 1. Integrators and thermostats

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Velocity Verlet | ✓ | ✓ | |
| Langevin (Schneider–Stoll) | ✓ | ✓ | Not BAOAB/GJF |
| Brownian dynamics | ✓ | ◐ | CUDA: device kernel; CPU-only build throws `brownian_integrate`. Einstein ⟨r²⟩=6Dt pin stays the CPU regression |
| Nosé–Hoover (single chain) | ✓ | ✓ | Standalone NVT |
| Nosé–Hoover chain (M-chain) | ✓ | ✓ | Inside MTK NPT only |
| Bussi / CSVR | ✓ | ○ | |
| DPD thermostat | ✓ | ○ | |
| NVELimit | ✓ | ✓ | Displacement cap |

## 2. Barostats and ensembles

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| NVE / NVT / NPT | ✓ | ✓ | |
| NPH (isotropic) | ✓ | ○ | `NphIntegrator` |
| MTK NPT (diagonal) | ✓ | ✓ | `NoseHooverNPT` |
| Parrinello–Rahman NPT (full cell) | ✓ | ○ | CPU full-cell LJ regression; general-cell neighbor lists and spatial decomposition |
| Surface-tension NPT | ✓ | ○ | Semi-isotropic |
| Berendsen barostat | ✓ | ✓ | Protocol reproduction — not for sampling |
| Temperature ramp | ✓ | ✓ | |

CPU `Box` uses lattice vectors in columns (`r = H s`) and supports general
triclinic cells. Neighbor construction, fractional spatial decomposition and
periodic halo images support both single-rank and MPI execution, including
shear changes and multiple images of the same atom. LJ, EAM and SW+LJ tilted
initial/one-step states are compared with LAMMPS; general cell matrices and
self-image multiplicity have direct tests. The 800-step PR-NPT LJ regression
also passes. These checks do not establish every potential/barostat combination;
GPW and slab Ewald retain orthorhombic restrictions.

CPU MD uses one Runtime rebuild sequence and explicit periodic ghost rows for
both serial and MPI execution. Frame owns row extents; neighbor consumers
declare interaction coverage independently of skin and communication reach.
The [LJ CPU benchmark](../../benchmarks/lj/README.md#cpu-md-architecture-apple-m4-2026-10-19)
records the same-input LAMMPS comparison at 10,648 and 103,823 atoms: the tested
Apple M4 loop meets the performance target, while communication category
parity remains open. This does not establish parity for other potentials or
cluster hardware.

## 3. Constraints and virtual sites

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| RATTLE | ✓ | ○ | Distance + 1–3; per-cluster solve (X–H clusters converge independently); MPI |
| SETTLE | ✓ | ○ | Three-site rigid water |
| Virtual sites | ✓ | ○ | Weighted3 (TIP4P), OutOfPlane3 (TIP5P) |
| LINCS / general rigid bodies | ○ | ○ | |

## 4. Nonequilibrium

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Constant / oscillating E-field | ✓ | ○ | |
| Constant-force / constant-velocity pull | ✓ | ○ | |
| Uniaxial Deform | ✓ | ✓ | CUDA: `erate` only |
| SLLOD / Lees–Edwards / heat flux | ○ | ○ | |

## 5. Ab initio MD

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| HF BOMD | ✓ | ◐ | CUDA: STO-3G s/p RHF only |
| KS BOMD (Gaussian) | ✓ | ○ | |
| GPW BOMD | ✓ | ○ | Γ / multi-k / UKS / smear |
| PW BOMD | ✓ | ○ | Foothold |
| AM1 BOMD | ✓ | ○ | |
| DFTB / xTB MD | ✓ | ○ | Via forces → MD; no dedicated AIMD pin family |
| CPMD (GPW) | ✓ | ○ | eNH; multi-k / UKS |
| Second-gen CPMD | ✓ | ○ | ASPC + partial SCF; OT variants |
| XL-BOMD | ✓ | ○ | Incl. linearized F/σ |

## 6. Nuclear quantum / nonadiabatic

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| PIMD skeleton | ✓ | ○ | Ring springs + IdentityReplication |
| RPMD / PILE / contraction | ○ | ○ | |
| FSSH / Ehrenfest | ○ | ○ | NAC vectors ship; no dynamics consumer |

## 7. Vibrations

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| FD Hessian (`cpu::vib::Hessian`) | ✓ | ○ | 6N central; any force evaluator |
| Normal modes / RRHO thermochem | ✓ | ○ | |
| Method analytic Hessian | — | — | See [HF](hartree-fock.md) / [DFT](dft.md); not this generic path |
| FD phonons / DFPT | — | — | See [DFT §5](dft.md#5-periodic-response-and-analysis) |
