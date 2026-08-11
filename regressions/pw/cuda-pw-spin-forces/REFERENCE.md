# Reference — CUDA plane-wave collinear LSDA spin ionic forces

## Method

Device twin of `cpu::pw::pw_forces_analytic` under collinear fixed-occupation
UKS (nspin=2, `XCType::LSDA`):

- local Hellmann–Feynman from **total** \(\rho=\rho_\uparrow+\rho_\downarrow\)
- Kleinman–Bylander nonlocal with \(f=1\) per occupied spin-orbital
  (\(N_\uparrow,N_\downarrow\) from multiplicity)
- Ewald ion–ion (shared host classical sum)

Public API: `cuda::pw::pw_forces_analytic(frame, method)` [eV/Å] with
`spin_polarized=true`, `multiplicity=3`, `xc=LSDA`.

## System

O₂ molecule in an 8 Å cubic cell, bond stretched to 1.30 Å (equilibrium
~1.2075 Å) so forces are non-vacuous. Triplet multiplicity 3 (\(M=2\)).

Settings (both backends):

| Member | Value |
| --- | --- |
| `upf_path` | `O_ONCV_PBE-1.2.upf` |
| `ecut_ry` | 12.0 |
| `k_mesh` | 1 (Γ) |
| `tol` | 5e-4 |
| `max_iter` | 60 |
| `mix_beta` | 0.3 |
| `pulay_depth` | 5 |
| `spin_polarized` | true |
| `multiplicity` | 3 |
| `xc` | `XCType::LSDA` |

## Oracle

Same-settings CPU `cpu::pw::pw_forces_analytic` after its collinear spin lift
(not CP2K; not a self-golden of CUDA).

## Pin / gate

\[
\max_{i,\alpha}\lvert F^{\mathrm{cuda}}_{i\alpha} - F^{\mathrm{cpu}}_{i\alpha}\rvert
  \le 5\times 10^{-3}\ \mathrm{eV/Å}
\]

and \(\max |F^{\mathrm{cpu}}| > 0.05\) eV/Å (non-vacuous).

**GH200 job 1130077:** measured \(\max|\Delta F| = 8.51\times 10^{-13}\) eV/Å
(CPU = CUDA \(\max|F| = 12.491950\) eV/Å; CUDA force-path SCF energy
\(-56.33636038\) Ry, iter=10). Bound kept at design-time \(5\times 10^{-3}\).

**Re-verify GH200 job 1130145** (tip rebased onto Assign wave-4 `dev`
@ `e13cdb91`, feat HEAD `87f08dec`): \(\max|\Delta F| = 9.25\times 10^{-13}\)
eV/Å; same \(\max|F|=12.491950\) eV/Å; energy \(-56.33636038\) Ry, iter=10;
`test_cuda_pw_force_local` + dual-gate PASS; BUILD/UNIT/REG_RC=0.

## Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU force parity as above |
| Wall | **perf_exempt** — correctness foothold on small cell |

## Non-goals

spin-PBE forces, spin stress, metallic smear forces, multi-k spin forces.
