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

CUDA mixed precision retains FP32 main positions plus residuals and FP64
velocities, masses, integration and canonical cell; force geometry and FFT
work remain FP32. Import/export, atom permutation, communication, minimizer
rollback and asynchronous position snapshots preserve the full motion state.
Global energy and stress scalars remain FP64 in every CUDA mode.

A fresh CPU or CUDA velocity-Verlet run evaluates the initial forces before
its first half-kick. Initial force preparation does not advance time, integrate
positions or emit records. Langevin draws a distinct sample for every force
application, including preparation and the first step; its temperature ramp
still follows the physical step. Focused regressions pin the first Verlet
step and the analytic one-step Langevin velocity variance on both backends.
MTK NPT uses the actual initial kinetic energy, with CPU/CUDA volume-parity
and MPI reduction checks.

## 2. Barostats and ensembles

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| NVE / NVT / NPT | ✓ | ✓ | |
| NPH (isotropic) | ✓ | ○ | `NphIntegrator` |
| MTK NPT (diagonal) | ✓ | ✓ | `NoseHooverNPT` |
| Parrinello–Rahman NPT (full cell) | ✓ | ○ | |
| Surface-tension NPT | ✓ | ○ | Semi-isotropic |
| Berendsen barostat | ✓ | ✓ | Protocol reproduction — not for sampling |
| Temperature ramp | ✓ | ✓ | |

## 3. Constraints and virtual sites

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| RATTLE | ✓ | ○ | Distance + 1–3; MPI |
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
