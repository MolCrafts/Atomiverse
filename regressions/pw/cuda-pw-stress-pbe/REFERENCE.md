# Reference — CUDA plane-wave Nielsen–Martin cell stress under PBE

## Method

Device twin of `cpu::pw::pw_stress` with **`XCType::PBE`**:

1. One device SCF at the reference geometry (device PBE-GGA on the FFT grid:
   spectral ∇ρ / −∇·flux in `DeviceDensity`, kernels in `xc_pbe.cu`).
2. Hellmann–Feynman cell stress by central finite differences of the
   *non-self-consistent* total energy under symmetric strain (density
   remapped \(\rho\to\rho\cdot\Omega_0/\Omega\), potentials rebuilt with the
   same PBE path, one Davidson diagonalization per strained cell).

Sign: mechanical / CP2K \(\sigma=-(1/\Omega)\,\partial E/\partial\varepsilon\).

Public API: `cuda::pw::pw_stress(frame, method[, energy_ry_out])` → Voigt
σ [eV/Å³] with `method.xc = XCType::PBE`.

Sibling of `cuda-pw-stress` (LSDA). Closes the closed-shell PBE stress
dual-gate left open after `cuda-pw-pbe-device` + `cuda-pw-stress` (LSDA).

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
| `xc` | **`XCType::PBE`** |

## Oracle

Same-settings CPU `cpu::pw::pw_stress` under PBE (not CP2K). Offline multi-SCF
FD class covered by `reg_pw-analytic-stress-si` / `pw_stress_fd` on CPU (LSDA
class); PBE stress twin is this dual-gate.

## Pin / gate

\[
\max_{v}\lvert \sigma^{\mathrm{cuda}}_{v} - \sigma^{\mathrm{cpu}}_{v}\rvert
  \le 5\times 10^{-2}\ \mathrm{eV/Å^3}
\]

plus cubic symmetry (off-diag and diag equality within \(2.5\times 10^{-2}\)).

Design-time bound matches LSDA `cuda-pw-stress` / CPU analytic↔FD window;
re-pin from the first GH200 log without tightening at design time.

**GH200 job 1122143** (`wf/cuda-pw-pbe-stress`): unit
`test_cuda_pw_cell_stress` **1/1 PASS** (H₂ LSDA + H₂ PBE twins);
dual-gate \(\max|\Delta\sigma| = 5.585\times 10^{-12}\) eV/Å³ (tol
\(5\times 10^{-2}\)); \(\mathrm{mean}|diag_{cpu}| = 0.273236\) eV/Å³;
CPU = CUDA \(E = -13.90403280\) Ry, stress-SCF **iter=7**,
\(|\Delta E| = 4.441\times 10^{-14}\) Ry. Bound kept at design-time
\(5\times 10^{-2}\).

**Re-verify GH200 job 1122705** after rebase onto `dev@d701a1cd`: unit
1/1 PASS; dual-gate max|Δσ| = \(5.585\times 10^{-12}\) eV/Å³; same E and
|ΔE|; `BUILDS_WIPED_OK`.

## Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU PBE stress parity as above |
| Wall | **perf_exempt** — correctness foothold (1 SCF + 12 NSCF; small cell) |

## Non-goals

Collinear spin, metallic smear, multi-SCF FD production path, variable-cell
driver product, energy-path changes beyond stress-path SCF.
