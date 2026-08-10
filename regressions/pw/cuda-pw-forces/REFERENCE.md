# Reference — CUDA plane-wave analytic ionic forces

## Method

Device twin of `cpu::pw::pw_forces_analytic`:

- local Hellmann–Feynman from ρ(G) × v(G) (device kernel)
- Kleinman–Bylander nonlocal from β, ψ (device kernels)
- Ewald ion–ion (shared host classical sum)

Public API: `cuda::pw::pw_forces_analytic(frame, method)` [eV/Å].

## System

2-atom Si FCC primitive cell, lattice \(a = 5.43\) Å, second atom stretched
by \(+0.08\) Å along \(x\) so forces are nonzero and exercise KB projectors.

Settings (both backends):

| Member | Value |
| --- | --- |
| `gth_name` | `GTH-PBE-q4` |
| `ecut_ry` | 8.0 |
| `k_mesh` | 1 (Γ) |
| `tol` | 1e-6 |
| `max_iter` | 40 |
| `xc` | `XCType::LSDA` |

## Oracle

Same-settings CPU `cpu::pw::pw_forces_analytic` (not CP2K; force twin is the
landed CPU analytic path). Optional class: CPU FD gate in
`tests/cpu/planewave/test_pw_forces_fd.cpp` already covers analytic↔FD.

## Pin / gate

\[
\max_{i,\alpha}\lvert F^{\mathrm{cuda}}_{i\alpha} - F^{\mathrm{cpu}}_{i\alpha}\rvert
  \le 5\times 10^{-2}\ \mathrm{eV/Å}
\]

and \(\max |F^{\mathrm{cpu}}| > 0.05\) eV/Å (non-vacuous).

**Re-pin GH200 job 1049792:** measured \(\max|\Delta F|\approx 2.0\times 10^{-2}\)
eV/Å (CPU \(\max|F|\approx 2.54\), CUDA \(\approx 2.52\)); design-time 5e-3 was
below the independent-SCF floor. Bound = 2.5× measured residual.

Tolerance absorbs independent SCF trajectories (cuFFT vs FFTW, Zheevd vs
Jacobi eigenvectors, tree vs serial reductions). Not bit-exact by design.

## Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU force parity as above |
| Wall | **perf_exempt** — correctness foothold on small cell |

## Non-goals

Stress product, collinear spin, metallic smear, energy-path changes.
