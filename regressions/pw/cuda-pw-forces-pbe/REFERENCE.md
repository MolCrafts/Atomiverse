# Reference — CUDA plane-wave analytic ionic forces under PBE

## Method

Same device twin as `cuda-pw-forces` (`cuda::pw::pw_forces_analytic`):

- local Hellmann–Feynman from ρ(G) × v(G) (device kernel)
- Kleinman–Bylander nonlocal from β, ψ (device kernels)
- Ewald ion–ion (shared host classical sum)

SCF and force-path V_eff rebuild use **device PBE-GGA** (`XCType::PBE`:
spectral ∇ρ / −∇·flux in `DeviceDensity`, kernels in `xc_pbe.cu`).

Public API: `cuda::pw::pw_forces_analytic(frame, method)` [eV/Å].

## System

2-atom Si FCC primitive cell, lattice \(a = 5.43\) Å, second atom stretched
by \(+0.08\) Å along \(x\) (nonzero forces + KB projectors).

Settings (both backends):

| Member | Value |
| --- | --- |
| `gth_name` | `GTH-PBE-q4` |
| `ecut_ry` | 8.0 |
| `k_mesh` | 1 (Γ) |
| `tol` | 1e-6 |
| `max_iter` | 40 |
| `xc` | **`XCType::PBE`** |

## Oracle

Same-settings CPU `cpu::pw::pw_forces_analytic` (not CP2K). Analytic↔FD
class coverage remains on CPU (`tests/cpu/planewave/test_pw_forces_fd.cpp`).

## Pin / gate

\[
\max_{i,\alpha}\lvert F^{\mathrm{cuda}}_{i\alpha} - F^{\mathrm{cpu}}_{i\alpha}\rvert
  \le 5\times 10^{-3}\ \mathrm{eV/Å}
\]

and \(\max |F^{\mathrm{cpu}}| > 0.05\) eV/Å (non-vacuous).

**GH200 job 1050253** (`f28cb2df`): units **1/1 PASS**; dual-gate
\(\max|\Delta F| = 2.615\times 10^{-13}\) eV/Å (tol \(5\times 10^{-3}\));
CPU = CUDA \(\max|F| = 2.486453\) eV/Å; force-path SCF both **iter=8**,
\(E = -13.89672028\) Ry. Bound kept at design-time \(5\times 10^{-3}\).

**Root cause of first fail (job 1050212, \(\max|\Delta F|\sim 14\) eV/Å):**
`pw_forces_analytic` constructed `DeviceBandSolver` without
`den.stream()`, so density and band halves raced on separate non-default
streams. LSDA often survived; PBE (many more density-side FFTs) failed to
converge (hit `max_iter=40`). Fixed: borrow `den.stream()` as in `PW::run`;
local HF launches on the same stream.

## Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU force parity as above under PBE |
| Wall | **perf_exempt** — correctness foothold on small cell |

## Non-goals

Stress, collinear spin, metallic smear, energy-path changes, PBE-family
device variants (PBEsol/revPBE/RPBE).
