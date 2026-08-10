# cuda-pw-spin-pbe

## What

Collinear fixed-occupation UKS with **true spin-GGA PBE** on `cuda::pw::PW`
(O₂ triplet). Dual-gate against same-settings `cpu::pw::PW`.

## Settings

| Member | Value |
| --- | --- |
| system | O₂, bond 1.2075 Å, cubic box 8 Å |
| UPF | `tests/data/pw/O_ONCV_PBE-1.2.upf` |
| ecut_ry | 12 |
| k_mesh | 1 (Γ) |
| spin_polarized | true |
| multiplicity | 3 → M = 2 |
| xc | `XCType::PBE` |
| tol | 5e-4 |
| max_iter | 60 |
| mix_beta | 0.3 |
| pulay_depth | 5 |

## Oracle

- **CPU twin** (same settings): energy dual-gate, not a CUDA self-golden.
- Magnetization identity: M = multiplicity − 1 = 2 (literature / fixed occupations).
- GGA activation: \|E_PBE − E_LSDA\| > 1e-4 Ha on the CPU twin (or CUDA LSDA
  leg when available) so the path is not stuck in the σ=0 LSDA limit.
- Offline method pin class: `regressions/pw/pw-spin-gga` (CPU energy
  −28.7585210635 Ha at these settings; dual-gate uses live CPU twin, not that
  absolute pin alone).

## Tolerances

| Check | Bound |
| --- | --- |
| mag | 1e-9 vs 2.0 |
| \|E_cuda − E_cpu\| | **2e-3 Ha** (cuFFT/FFTW, tree/serial, libdevice/libm, Davidson) |
| \|E_PBE − E_LSDA\| | > 1e-4 Ha |

## Gate (GH200)

| Job | Result |
| --- | --- |
| **1122874** (n416, exit 0) | CPU/CUDA PBE both **−28.7585210635 Ha**, mag=2; \|ΔE\|=**1.14e-13 Ha**; \|E_PBE−E_LSDA\|_cpu=**0.129 Ha** |
| prior **1122357** | first green on pre-rebase tip |

Log path on staff tree: `.claude/cuda-pw-spin-pbe-1122874.out`.

## Not claimed

Forces, stress, spin+smear, PBEsol/revPBE/RPBE, fair CP2K wall.
