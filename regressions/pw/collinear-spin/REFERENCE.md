# pw/collinear-spin

One case; every section is run by `atv_regressions --case pw/collinear-spin`
and keeps its own oracle record.

## Reference — PW collinear spin-polarized O₂ triplet

### Method and system

`cpu::pw::PW` uses true collinear spin-GGA PBE, spectral spin-density gradients
and the corresponding divergence contribution to each spin potential. The
periodic O₂ molecule has positions (0, 0, 0) and (0, 0, 1.2075) Å in a cubic
8 Å cell, Γ sampling, and fixed occupations N↑ = 7, N↓ = 5 (12 valence
electrons, multiplicity 3).

| Setting | Value |
| --- | --- |
| Pseudopotential | SG15 NC ONCV PBE oxygen, `tests/data/pw/O_ONCV_PBE-1.2.upf` |
| Wavefunction / density cutoff | 12 / 48 Ry |
| Density FFT grid | 36 × 36 × 36 |
| CPU GGA XC quadrature | 54 × 54 × 54 (private 3/2 padding) |
| Native SCF convergence | Unmixed two-spin density residual, `tol = 1e-7` |
| Native iteration budget | `max_iter = 200` |
| Native mixing | `mix_beta = 0.3`, `pulay_depth = 5` |
| Thread width | OpenMP 1, OpenBLAS 1 |

The CPU NC-UPF operator uses the supplied `PP_RAB` mesh weights with QE's
Simpson quadrature for radial transforms. Its β-projector implementation
supports s/p/d angular channels and diagonalizes each real symmetric per-l
D block with LAPACK, including multiple coupled radial projectors. The UPF
Ry-to-Ha conversion is applied once. These are the CPU operator semantics
validated here; they do not establish a fresh CUDA validation.

### External oracle and pins

The energy oracle is an actual **Quantum ESPRESSO 7.5** run from source
[`770a0b2d12928a67048e2f3da8d10d057e52179e`](https://github.com/QEF/q-e/tree/770a0b2d12928a67048e2f3da8d10d057e52179e),
run on 2026-10-25. The matching energy-only input is
[`o2_pbe.in`](../../../tools/qe_oracle/inputs/pw-collinear-spin/o2_pbe.in);
it uses `conv_thr = 1e-10`, `electron_maxstep = 200`, fixed triplet occupations,
PBE, the same NC-UPF, geometry and cutoffs. QE selects the same 36³ density
grid and converges in 24 iterations. Its SCF criterion differs from the
native density-residual criterion.

The [raw QE output](../../../tools/qe_oracle/inputs/pw-collinear-spin/o2_pbe.qe.txt)
prints:

```text
!    total energy              =     -57.46535365 Ry
```

Thus the external energy pin is **−28.7326768250 Ha**. Acceptance preserves
the original bounds:

- SCF converges and its energy is finite.
- `|E_native − E_QE| ≤ 1e-3 Ha`.
- `|M − 2| ≤ 1e-9`, the fixed-occupation identity N↑ − N↓ = 2.
- The entire section completes in **≤60 s** at the declared thread width.

The UPF SHA-256 is
`212d0bcacf75a66f5e086100069a3ddf5d754afd335f054ddfb4ca4b9e730400`;
the input SHA-256 is
`b1022660b57d154394ab0ca593fd66ebc228afef60690bfe58174e1c59c66242`.
Binary/build details, raw-output hashes and the independent 72³ grid
diagnostic are in [reference.json](../../../tools/qe_oracle/inputs/pw-collinear-spin/reference.json)
and the [reproduction README](../../../tools/qe_oracle/inputs/pw-collinear-spin/README.md).
The 72³ result differs by 3.18e-6 Ha, below the unchanged 1e-3 Ha gate.

### Current CPU validation

The current native run
converges the PBE triplet in 12 iterations and prints −57.46535257 Ry, M=2.
Both registered O₂ sections pass their original energy, magnetization,
activation and whole-section runtime gates. The density FFT and wavefunction
cutoffs above match QE; CPU GGA XC now uses its private 54³ quadrature, while
the archived QE run uses its 36³ grid. No refreshed fair wall ratio is claimed.
The captures below precede the XC-quadrature and joint-Pulay changes.

### Historical native validation and timing

The [captured native section](../../../tools/qe_oracle/inputs/pw-collinear-spin/native-pw-spin-polarized-o2.txt)
prints E = −28.7326751302 Ha, M = 2 and convergence in 35 iterations;
its absolute error against QE is 1.6948e-6 Ha. Its complete section boundary,
including construction/build/run/checks, is **16.865870 s**. Debug iteration
logging was enabled.

The energy-only QE subprocess wall is **1.882814625 s** (QE's internal
`PWSCF` wall is 1.85 s), also at OpenMP/OpenBLAS 1/1. The observed quotient
of native section wall to QE process wall is **8.96**. These boundaries differ
and the native run includes debug logging; this is an observation, not an
independent final performance measurement. A final batch without debug
logging is required for such a claim. There is no runtime exemption: the
registered native section enforces the 60 s bound.

## Reference — true collinear spin-GGA PBE on O₂ triplet (plane-wave)

### Method and oracle

`pw-spin-gga` uses the same O₂ geometry, NC-UPF, 12/48 Ry cutoffs, 36³ FFT
grid, fixed occupations and native SCF controls listed above. Its PBE solve
uses the same actual QE 7.5 energy pin **−28.7326768250 Ha** and the same
1e-3 Ha energy and 1e-9 magnetization bounds. It then solves the same system
with `XCType::LSDA` to exercise the GGA contribution.

Spectral spin gradients form the GGA invariants. Each spin potential contains
its density derivative and the divergence of
`2 * v_sigma_same * grad(rho_spin) + v_sigma_cross * grad(rho_other)`.
The additional activation gate remains
`|E_PBE − E_LSDA| > 1e-4 Ha`; it supplements the external PBE pin and does
not supply an absolute LSDA oracle. Both solves must converge with finite
energies. The entire two-solve section must finish in **≤60 s** at OpenMP /
OpenBLAS 1/1.

### Historical native validation and timing

The [captured native section](../../../tools/qe_oracle/inputs/pw-collinear-spin/native-pw-spin-gga.txt)
prints PBE E = −28.7326751302 Ha (35 iterations), LSDA E = −28.6078529734 Ha
(34 iterations), M = 2 and `|E_PBE − E_LSDA| = 1.248222e-1 Ha`.
The complete section wall is **26.585936 s**, with debug iteration logging
enabled. This includes **two SCF solves** and is not compared as a wall ratio
to the single energy-only QE PBE calculation. The native measurements and
raw-segment hashes are recorded separately in
[native-validation.json](../../../tools/qe_oracle/inputs/pw-collinear-spin/native-validation.json);
none of these native values is an oracle pin.

The CPU energy reference and radial/projector implementation above have been
updated. The CUDA records below retain their historical job data; they are
not fresh validation of the changed CPU operator or its current QE reference.
The affected CUDA slices require a new same-settings GH200 check.

## cuda-pw-collinear-spin

### What

Dual-gate collinear fixed-occupation UKS (O2 triplet, multiplicity 3 → M = 2)
on `cuda::pw::PW` with **`XCType::LSDA`**. Same geometry and occupations as
`pw-spin-polarized-o2`, restricted to the device-supported surface (no PBE
spin-GGA, no smear).

### Oracle

1. **Magnetization identity** — M = multiplicity − 1 = 2 (integer fill).
2. **Same-settings CPU twin** — `cpu::pw::PW` at identical members is the
   correctness oracle for CUDA energy (ES CUDA open-gate law).

Not a freeze of CUDA’s own energy. Not the PBE literature pin from
`pw-spin-polarized-o2` (that functional has no collinear device path yet).

### Settings

| Member | Value |
| --- | --- |
| PP | `tests/data/pw/O_ONCV_PBE-1.2.upf` |
| ecut | 12 Ry |
| k_mesh | 1 (Γ) |
| box | 8 Å cubic |
| R(O–O) | 1.2075 Å |
| spin_polarized | true |
| multiplicity | 3 |
| xc | LSDA |
| mix_beta | 0.3 |
| tol | 5e-4 |

### Tolerances

| Quantity | Tol |
| --- | --- |
| mag | 1e-9 |
| \|E_cuda − E_cpu\| | 2e-3 Ha |

Non-bit-exact: cuFFT vs FFTW, tree vs serial reductions, libdevice vs libm,
Davidson trajectory.


## Reference — CUDA plane-wave collinear LSDA spin ionic forces

### Method

Device twin of `cpu::pw::pw_forces_analytic` under collinear fixed-occupation
UKS (nspin=2, `XCType::LSDA`):

- local Hellmann–Feynman from **total** \(\rho=\rho_\uparrow+\rho_\downarrow\)
- Kleinman–Bylander nonlocal with \(f=1\) per occupied spin-orbital
  (\(N_\uparrow,N_\downarrow\) from multiplicity)
- Ewald ion–ion (shared host classical sum)

Public API: `cuda::pw::pw_forces_analytic(frame, method)` [eV/Å] with
`spin_polarized=true`, `multiplicity=3`, `xc=LSDA`.

### System

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

### Oracle

Same-settings CPU `cpu::pw::pw_forces_analytic` after its collinear spin lift
(not CP2K; not a self-golden of CUDA).

### Pin / gate

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

**Re-verify GH200 job 1130196** (tip rebased onto Assign wave-5 `dev`
@ `7f8da1f0`, HEAD `e927d822`): \(\max|\Delta F| = 1.56\times 10^{-12}\)
eV/Å; same \(\max|F|=12.491950\) eV/Å; energy \(-56.33636038\) Ry, iter=10;
`test_cuda_pw_force_local` + dual-gate PASS; BUILD/UNIT/REG_RC=0.

### Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU force parity as above |
| Wall | **perf_exempt** — correctness foothold on small cell |

### Non-goals

spin-PBE forces, spin stress, metallic smear forces, multi-k spin forces.


## cuda-pw-spin-pbe

### What

Collinear UKS with **true spin-GGA PBE** on `cuda::pw::PW` (O₂ triplet):
fixed-occupation energy / F / σ, plus collinear **spin×smear** energy and
analytic forces (shared μ, \(f\in[0,1]\)). Dual-gate against same-settings
`cpu::pw::PW`.

### Settings

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

### Oracle

- **CPU twin** (same settings): energy dual-gate, not a CUDA self-golden.
- Magnetization identity: M = multiplicity − 1 = 2 (literature / fixed occupations).
- GGA activation: \|E_PBE − E_LSDA\| > 1e-4 Ha on the CPU twin (or CUDA LSDA
  leg when available) so the path is not stuck in the σ=0 LSDA limit.
- Offline method pin class: `regressions/pw/collinear-spin` (CPU energy
  −28.7585210635 Ha at these settings; dual-gate uses live CPU twin, not that
  absolute pin alone).

### Tolerances

| Check | Bound |
| --- | --- |
| mag (fixed occ) | 1e-9 vs 2.0 |
| \|E_cuda − E_cpu\| | **2e-3 Ha** (cuFFT/FFTW, tree/serial, libdevice/libm, Davidson) |
| \|E_PBE − E_LSDA\| | > 1e-4 Ha |
| max\|ΔF\| | **5e-3 eV/Å** (stretched bond 1.30 Å; non-vacuous max\|F\| > 0.05) |
| max\|Δσ\| (fixed occ) | **5e-2 eV/Å³** |

### Gate (GH200)

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

### Not claimed

spin×smear analytic **stress**, multi-k spin, PBEsol/revPBE/RPBE, fair CP2K wall.


## Reference — CUDA plane-wave collinear LSDA spin cell stress

### Method

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

### System

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

### Oracle

Same-settings CPU `cpu::pw::pw_stress` after its collinear spin lift
(not CP2K; not a self-golden of CUDA).

### Pin / gate

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
`pw/cuda-pw-spin-forces` force-path energy); `pw/cuda-pw-spin-stress`
PASS; BUILD/REG_RC=0; **perf_exempt** small cell.

### Dual gate

| Gate | Criterion |
| --- | --- |
| Correctness | CUDA↔CPU stress parity as above |
| Wall | **perf_exempt** — correctness foothold on small cell |

### Non-goals

spin-PBE stress, spin∧smear stress, multi-k spin stress, soft tols.
