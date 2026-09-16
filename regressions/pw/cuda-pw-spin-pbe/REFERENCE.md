# cuda-pw-spin-pbe

## What

Collinear UKS with **true spin-GGA PBE** on `cuda::pw::PW` (O₂ triplet):
fixed-occupation energy / F / σ, plus collinear **spin×smear** energy and
analytic forces (shared μ, \(f\in[0,1]\)). Dual-gate against same-settings
`cpu::pw::PW`.

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
| mag (fixed occ) | 1e-9 vs 2.0 |
| \|E_cuda − E_cpu\| | **2e-3 Ha** (cuFFT/FFTW, tree/serial, libdevice/libm, Davidson) |
| \|E_PBE − E_LSDA\| | > 1e-4 Ha |
| max\|ΔF\| | **5e-3 eV/Å** (stretched bond 1.30 Å; non-vacuous max\|F\| > 0.05) |
| max\|Δσ\| (fixed occ) | **5e-2 eV/Å³** |

## Gate (GH200)

| Job | Result |
| --- | --- |
| **1357414** (n575, exit 0) | spin×smear **energy + forces**: \|ΔE\|=**1.10e-13 Ha**, mag=1.6682; force-path SCF energy_ry=**−57.57643373** smear=1 (same Mermin F as `run()`); max\|ΔF\|=**1.92e-12** eV/Å (max\|F\|=3.79); spin-PBE F/σ reconfirm max\|ΔF\|=**1.98e-12** eV/Å, max\|Δσ\|=**4.86e-12** eV/Å³ |
| **1336653** | spin×smear **energy** only: \|ΔE\|=**1.10e-13 Ha**, mag=1.6682 |
| **1335945** | spin-PBE **F/σ** first green |
| **1124645** (n40, exit 0) | wave-4 reconfirm after rebase onto `dev`@`44bd64bb`; CPU/CUDA PBE both **−28.7585210635 Ha**, mag=2; \|ΔE\|=**1.14e-13 Ha**; \|E_PBE−E_LSDA\|_cpu=**0.129 Ha**; BUILD/UNIT/REG RCs 0 |
| **1123961** (n454, exit 0) | wave-3 reconfirm after rebase onto `dev`@`4913b33b`; same dual-gate numbers |
| **1122874** (n416, exit 0) | same dual-gate numbers on pre-rebase tip |
| prior **1122357** | first green on pre-rebase tip |

Log path on staff tree: `.claude/cuda-claim-md-es-1357414.out` (gitignored `*.out`).

## Not claimed

spin×smear analytic **stress**, multi-k spin, PBEsol/revPBE/RPBE, fair CP2K wall.
