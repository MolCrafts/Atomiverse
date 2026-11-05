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
| DPD thermostat | ◐ | ○ | No external pin; self-pinned case removed |
| NVELimit | ✓ | ✓ | Displacement cap |

## 2. Barostats and ensembles

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| NVE / NVT / NPT | ✓ | ✓ | |
| NPH (isotropic) | ✓ | ○ | `NphIntegrator` |
| MTK NPT (diagonal) | ✓ | ✓ | `NoseHooverNPT`; CUDA orthorhombic, with PME or Ewald reciprocal virial |
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

Image flags follow LAMMPS `Domain::pbc` on both backends: every wrap of atom
state (CPU rebuild wrap, SLLOD and the MC translate move; CUDA velocity
Verlet, NVE limit, MTK NPT, SLLOD, Brownian and the `sd` / `cg` / `lbfgs`
minimizer steps) adds the lattice translation to `image`, so
\(r + H\,\mathrm{image}\) is continuous and is what a retrieved Frame carries
(unwrapped transport, `FixEfield` energy). A rejected MC move and an L-BFGS
Armijo rollback restore the images they changed. Each image component is an
`ImageInt` (int32) on both backends, so a run may cross the cell ±2³¹ times
per axis; LAMMPS data files and molpy frames carry `ix/iy/iz` in and out
exactly.

On CUDA a plan whose integrator tilts the cell (SLLOD yz/xz/xy) dispatches the
general-cell pair, neighbor and mask kernels from the first step. Both
backends decide a Verlet rebuild with LAMMPS `Neighbor::check_distance`'s box
check: the half-skin trigger shrinks by the motion of the cell corners since
the last build (`core/neighbor_check.h`), and a cell change that alone
consumes the skin rebuilds; on CUDA the decision is the device skin gate, so a
deforming cell no longer rebuilds every step.

CPU MD uses one Runtime rebuild sequence and explicit periodic ghost rows for
both serial and MPI execution. Frame owns row extents; neighbor consumers
declare interaction coverage independently of skin and communication reach.
The [LJ CPU benchmark](../../benchmarks/compare/lj-lammps/README.md) compared
the same input against LAMMPS at 10,648 and 103,823 atoms (Apple M4,
2026-10-19): the tested loop meets the performance target, while communication category
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
| Constant E-field | ✓ | ◐ | `FixEfield` (LAMMPS `fix efield`, constant path): \(F = qE\), field in V/Å; \(U = -\sum q\,E\cdot r^u\) through the image flags (orthorhombic and triclinic), no virial (LAMMPS `fix_modify virial` default off). CUDA `cuda::md::FixEfield`: device-only, graph-capturable; unit tests vs analytic and CPU on a charged pair, no external pin. Wire ID 38; Python `FixEfield` via `MD.set_evaluator` on CPU and CUDA |
| Oscillating E-field | ✓ | ○ | |
| Constant-force pull | ✓ | ○ | |
| Constant-velocity pull (SMD) | ◐ | ○ | No external pin; self-pinned case removed |
| Uniaxial Deform | ✓ | ✓ | `Deform(Mode::erate` or `Mode::trate, rate, dt, axis)` on both backends: one axis, remap x (the axis row of the cell and the coordinates scale together, so tilted cells keep fractional coordinates), no shear; CUDA unit test vs CPU on a tilted cell; neither backend's `Deform` is exposed to Python |
| NVT SLLOD (`fix nvt/sllod` + `fix deform erate`, remap v) | ◐ | ◐ | `NvtSllod`, one direction per run (x y z yz xz xy), constant engineering rate, upper-triangular cell, Nosé–Hoover chain of length `tchain` (default 1, the LAMMPS `fix nvt/sllod` default; FixNH order with the carried peculiar temperature), LAMMPS `flip yes` box flip when a tilt passes half its box length (decided at end_of_step, executed at the next pre_exchange with `Domain::image_flip`; on CUDA both on the device). Same scalar laws on both backends (`core/sllod_flow.h`); unit tests: the chain against a transcription of `FixNH::nhc_temp_integrate`, CUDA vs CPU one step and an eight-step `tchain = 3` run (chain variables, unwrapped state), the flip on both backends, and an LJ shear across the flip whose per-step energy and forces equal a freshly built list (CPU) and the CPU run (CUDA). `PressureSeries` writes the six-component −P (lab-frame kinetic, LAMMPS default thermo) for one continuous run. Python: `NvtSllod(..., tchain=1)` + `MD.set_pressure_series` (CUDA). No external pin |
| Lees–Edwards / heat flux | ○ | ○ | |

## 5. Ab initio MD

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| HF BOMD | ✓ | ◐ | CUDA: STO-3G s/p RHF only |
| KS BOMD (Gaussian) | ✓ | ○ | |
| GPW BOMD | ✓ | ○ | Γ pinned; multi-k / UKS / smear reachable, not separately pinned |
| PW BOMD | ✓ | ○ | Foothold |
| AM1 BOMD | ◐ | ○ | AM1 forces pinned; no external-pinned MD case (self-pinned case removed) |
| DFTB / xTB MD | ◐ | ○ | Via forces → MD; forces pinned, no external-pinned MD case (self-pinned DFTB2 MD cases removed) |
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
