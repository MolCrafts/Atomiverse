# Reference — CUDA plane-wave Nielsen–Martin cell stress

## Method

Device twin of `cpu::pw::pw_stress`:

1. One device SCF at the reference geometry.
2. Hellmann–Feynman cell stress by central finite differences of the
   *non-self-consistent* total energy under symmetric strain (density
   remapped \(\rho\to\rho\cdot\Omega_0/\Omega\), potentials rebuilt, one
   Davidson diagonalization per strained cell).

Sign: mechanical / CP2K \(\sigma=-(1/\Omega)\,\partial E/\partial\varepsilon\).

Public API: `cuda::pw::pw_stress(frame, method[, energy_ry_out])` → Voigt
σ [eV/Å³].

## System

2-atom Si FCC primitive cell, lattice \(a = 5.43\) Å (compressed vs
\(a_0\approx 5.47\) → positive hydrostatic stress).

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

Same-settings CPU `cpu::pw::pw_stress` (not CP2K; stress twin is the landed
CPU Nielsen–Martin analytic path). Offline multi-SCF FD class covered by
`reg_pw-analytic-stress-si` / `pw_stress_fd` on CPU.

## Pin / gate

\[
\max_{v}\lvert \sigma^{\mathrm{cuda}}_{v} - \sigma^{\mathrm{cpu}}_{v}\rvert
  \le 5\times 10^{-2}\ \mathrm{eV/Å^3}
\]

plus cubic symmetry (off-diag and diag equality within \(2.5\times 10^{-2}\)).

## Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU stress parity as above |
| Wall | **perf_exempt** — correctness foothold (1 SCF + 12 NSCF; small cell) |

## Non-goals

Collinear spin, metallic smear, multi-SCF FD production path, variable-cell
driver product, energy-path changes.
