# Reference — CUDA plane-wave collinear LSDA spin cell stress

## Method

Device twin of `cpu::pw::pw_stress` under collinear fixed-occupation UKS
(nspin=2, `XCType::LSDA`):

- one SCF at the reference geometry
- Hellmann–Feynman cell stress by central finite differences of the
  *non-self-consistent* total energy under symmetric strain
- density remapped as \(\rho_\sigma'(r)=\rho_{\sigma0}(r)\cdot\Omega_0/\Omega\)
  for both spin channels (fixed fractional ionic coordinates)

Public API: `cuda::pw::pw_stress(frame, method)` [eV/Å³] Voigt with
`spin_polarized=true`, `multiplicity=3`, `xc=LSDA`.

Sign (mechanical / CP2K):
\(\sigma=-(1/\Omega)\partial E/\partial\varepsilon\).

## System

O₂ molecule in an 8 Å cubic cell, bond stretched to 1.30 Å (equilibrium
~1.2075 Å). Triplet multiplicity 3 (\(M=2\)). Same geometry class as
`reg_cuda-pw-spin-forces`.

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

Same-settings CPU `cpu::pw::pw_stress` after its collinear spin lift
(not CP2K; not a self-golden of CUDA).

## Pin / gate

\[
\max_{v}\lvert \sigma^{\mathrm{cuda}}_{v} - \sigma^{\mathrm{cpu}}_{v}\rvert
  \le 5\times 10^{-2}\ \mathrm{eV/Å^3}
\]

and \(\mathrm{mean}|\sigma^{\mathrm{cpu}}_{\mathrm{diag}}| > 10^{-4}\) eV/Å³
(non-vacuous).

**GH200 job 1133857** (original tip on Assign wave-1 `dev` @ `00dc6a51`)
and **re-verify job 1135756** (rebased tip `0c13f656` on Assign wave-3
`dev` @ `0f56cbca`):
\(\max|\Delta\sigma| = 1.435\times 10^{-11}\) eV/Å³;
\(\mathrm{mean}|diag_{cpu}|=0.077790\) eV/Å³;
\(\sigma_{cpu}\approx(-0.081821,-0.080685,-0.070865)\) eV/Å³;
CUDA stress-path SCF energy \(-56.33636038\) Ry, iter=10 (matches
`reg_cuda-pw-spin-forces` force-path energy); `run_reg_cuda-pw-spin-stress`
PASS; BUILD/REG_RC=0; **perf_exempt** small cell.

## Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU stress parity as above |
| Wall | **perf_exempt** — correctness foothold on small cell |

## Non-goals

spin-PBE stress, spin∧smear stress, multi-k spin stress, soft tols.
