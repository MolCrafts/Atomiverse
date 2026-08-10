# Spec — `cuda-pw-forces`

| Field | Value |
| --- | --- |
| slug | `cuda-pw-forces` |
| method | cuda |
| branch | `wf/cuda-pw-forces` |
| oracle | same-settings CPU twin `cpu::pw::pw_forces_analytic` (+ optional FD class) |
| status | in progress |

## Problem

`cuda::pw::PW` is **energy-only**. CPU has `pw_forces_analytic` (local HF +
Ewald + KB nonlocal). Energy foothold ≠ forces closed — roadmap §10 + queue
P0. No public force surface on CUDA; no device force kernels.

## Lever

| | |
| --- | --- |
| Class | Device analytic ionic forces (local Hellmann–Feynman + KB nonlocal) + host Ewald |
| Files | `src/cuda/planewave/internal/pw_forces.{cu,cuh}`, wire `pw_solver.{cu,cuh}`; atom-index column on `DeviceBandSolver`; `DeviceDensity::device_rho_G` accessor |
| Metric | max\|F_cuda − F_cpu\| [eV/Å] on stretched Si dimer (GTH-PBE-q4, ecut 8 Ry, Γ) |
| Non-goal | Stress product; spin/smear; energy-path change; classical MD |

## Acceptance

1. Public free function `cuda::pw::pw_forces_analytic(frame, method)` mirrors CPU
   shape; returns forces [eV/Å].
2. Local HF + KB nonlocal force kernels are **device-only** (no host G-loops in
   the force kernels). Ewald remains the shared host classical sum (same as CPU
   electronic path using host Ewald for energy).
3. Unit: local-force kernel vs host twin on synthetic ρ(G); single-atom
   symmetry |F| small via public API where cheap.
4. Regression `reg_cuda-pw-forces`: dual-gate `--device=both` — CUDA↔CPU force
   parity on stretched Si dimer, stated tol; energy path of `run()` unchanged.
5. GH200 log evidence (units + regression PASS). No claim green without GPU log.
6. Build dirs wiped after job / worker exit.

## Dual gate

| Gate | How |
| --- | --- |
| Correctness | same-settings CPU `pw_forces_analytic`; max\|ΔF\| ≤ **5e-2 eV/Å** (re-pin GH200 **1049792** measured ~2e-2; cuFFT/FFTW + Zheevd/Jacobi + independent SCF) |
| Wall | **perf_exempt** — force foothold / correctness slice on small cell (same class as `cuda-pw-pbe-device`) |

## Stated tols

| Quantity | Tol |
| --- | --- |
| max\|F_iα^cuda − F_iα^cpu\| | 5e-2 eV/Å (measured ~2e-2 on Si dimer, job 1049792) |
| local kernel vs host synthetic | 1e-12 (unit) |
| energy path `run()` | bit-stable / no intentional change |
