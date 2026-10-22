# Molecular Dynamics

Tutorial: [MD](../tutorial/ch10-classical-md.md) · [ensembles](../tutorial/ch11-ensembles.md) · [vibrations](../tutorial/ch14-vibrations.md).

`cpu::md::MD` / `cuda::md::MD`. NVE/NVT/NPT on both backends. Legend:
[Roadmap index](index.md). Electrode Components:
[Electrodes](electrodes.md).

## 0. CUDA precision modes

| Mode | Force math / mesh / FFT | Coordinates | Integration, v, m | Energy, virial | Forces |
| ---- | ----------------------- | ----------- | ----------------- | -------------- | ------ |
| `double` (default) | FP64 | FP64 | FP64 | FP64 | 64-bit fixed point |
| `mixed` | FP32 | FP32 + FP32 correction | FP64 | FP64 | 64-bit fixed point |
| `single` | FP32 | FP32 | FP32 | FP64 | 64-bit fixed point |

`ATV_CUDA_PREC_MODE` selects the mode; `mixed` follows OpenMM's mixed
precision. A force outside the fixed-point range (|F| ≥ 2.2·10⁶ eV/Å, or
NaN) stops integration and is reported with atom tag and step.

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
| MTK NPT (diagonal) | ✓ | ✓ | `NoseHooverNPT`; CUDA orthorhombic, with PME reciprocal virial |
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
