# pw/collinear-spin

One case; every section below is run by `atv_regressions --case pw/collinear-spin` and keeps its own oracle record.


## Reference — PW collinear spin-polarized O₂ triplet

### Method

Collinear spin-polarized plane-wave KS (`cpu::pw::PW` with
`spin_polarized = true`, `multiplicity = 3`, `xc = XCType::PBE`). Fixed
occupations N↑ = 7, N↓ = 5 for 12 valence electrons (oxygen NC-PP Z_val = 6 × 2).
XC: **true collinear spin-GGA PBE** via spectral \(\nabla\rho_\sigma\) in
`XCBuilder::step_spin` (productized by `pw-spin-gga`; no longer the historical
\(\sigma=0\) LSDA-limit path). Γ-only Monkhorst–Pack mesh.

### System

| Quantity | Value |
| -------- | ----- |
| Molecule | O₂ |
| R(O–O) | 1.2075 Å (same as molecular SCF O₂ regressions) |
| Cell | cubic L = 8.0 Å |
| PP | SG15 ONCV PBE oxygen `tests/data/pw/O_ONCV_PBE-1.2.upf` (Hamann ONCVPSP; SG15 library) |
| E_cut | 12 Ry |
| k-mesh | 1×1×1 (Γ) |
| Multiplicity | 3 (triplet ³Σg⁻) |
| SCF tol (density L2) | 5×10⁻⁴ |
| max_iter | 60 |
| mix_beta | 0.3 |

### Pins

#### Magnetization (literature + fixed occupations)

O₂ ground state is the triplet ³Σg⁻ (Herzberg, *Molecular Spectra and
Molecular Structure* I; standard open-shell test). With fixed integer
occupations and multiplicity 3:

\[
M = N_\uparrow - N_\downarrow = 2.
\]

**Gate:** `|M − 2| ≤ 1×10⁻⁹` (occupation identity on `PWResult::magnetization`).

#### Total energy (offline method pin)

\[
E_{\mathrm{tot}} = -28.7585210635~\mathrm{Ha}
\quad (| \Delta E | \le 1\times 10^{-3}~\mathrm{Ha}).
\]

**Re-pin 2026-07-26 (`pw-spin-gga`):** true collinear spin-GGA replaces the
prior PBE@\(\sigma=0\) pin (−28.6267426458 Ha). Magnetization (literature hard
gate) is unchanged. Same settings as `reg_pw-spin-gga`.

**Re-pin 2026-07-25:** previous frozen value −28.3392565801 Ha was produced
by a Davidson subspace step that truncated the (complex Hermitian) subspace
Hamiltonian to its real part, corrupting the Ritz vectors, and by G-sphere
enumeration bounds that clipped sphere-edge plane waves. Both fixed with the
`berry-polarization` case.

**Oracle note:** Host CP2K 2026.1 module is listed but unloadable here
(dependency chain / `hwloc`); no `pw.x` (Quantum ESPRESSO) on PATH. Absolute
e_tot is therefore a **fixed-settings offline method pin** for regression
stability (same PP, ecut, cell, geometry). Re-pin against QE with this UPF
when an offline twin is available; magnetization remains the literature hard
gate. Units: public energy on `PWResult` is also reported in Ry
(`energy = 2 × energy_ha`).

### Perf

| Field | Value |
| ----- | ----- |
| perf_exempt | true |
| reason | CP2K binary not loadable on host (module dep); no fair single-rank CP2K PW/NC-UPF twin for wall ratio |
| T_atv (this host, single rank) | ~40 s (order of magnitude; not for ratio) |
| ratio T_atv/T_cp2k | n/a |

### Dual gate

1. Correctness: mag + e_tot gates above; SCF must converge.
2. Perf: exempt (no CP2K twin wall).


## Reference — true collinear spin-GGA PBE on O₂ triplet (plane-wave)

### Method

Collinear spin-polarized plane-wave KS (`cpu::pw::PW` with
`spin_polarized = true`, `multiplicity = 3`, `xc = XCType::PBE`) with
**true collinear spin-GGA**: spectral \(\nabla\rho_\uparrow\),
\(\nabla\rho_\downarrow\) on the FFT grid enter \(\sigma_{\sigma\sigma'}\)
in `evaluate_xc`, and the spin potentials include the divergence term

\[
v_\sigma = v_{\rho_\sigma}
- \nabla\cdot\big(
  2\,v_{\sigma_{\sigma\sigma}}\,\nabla\rho_\sigma
  + v_{\sigma_{\uparrow\downarrow}}\,\nabla\rho_{\bar\sigma}
\big)
\]

(`XCBuilder::step_spin`; same form as GPW `xc_grid_gga_spin`). This is **not**
the historical \(\sigma=0\) LSDA-limit call per spin channel.

Fixed occupations N↑ = 7, N↓ = 5 for 12 valence electrons (oxygen NC-PP
\(Z_{\mathrm{val}}=6\times 2\)). Γ-only Monkhorst–Pack mesh.

### System

| Quantity | Value |
| -------- | ----- |
| Molecule | O₂ |
| R(O–O) | 1.2075 Å |
| Cell | cubic L = 8.0 Å |
| PP | SG15 ONCV PBE oxygen `tests/data/pw/O_ONCV_PBE-1.2.upf` |
| E_cut | 12 Ry |
| k-mesh | 1×1×1 (Γ) |
| Multiplicity | 3 (triplet ³Σg⁻) |
| XC | PBE true collinear spin-GGA |
| SCF tol (density L2) | 5×10⁻⁴ |
| max_iter | 60 |
| mix_beta | 0.3 |

### Pins

#### Magnetization (literature + fixed occupations)

O₂ ground state is the triplet ³Σg⁻ (Herzberg). With fixed integer occupations
and multiplicity 3:

\[
M = N_\uparrow - N_\downarrow = 2.
\]

**Gate:** \(|M - 2| \le 1\times 10^{-9}\) on `PWResult::magnetization`.

#### Total energy (offline method pin)

\[
E_{\mathrm{tot}} = -28.7585210635~\mathrm{Ha}
\quad (| \Delta E | \le 1\times 10^{-3}~\mathrm{Ha}).
\]

**Pin 2026-07-26** (`wf/pw-spin-gga`, Release, single rank): first green with
true spin-GGA (spectral \(\sigma\)). The prior spin foothold
`pw-spin-polarized-o2` used PBE@\(\sigma=0\) and sat near −28.63 Ha (LSDA-like);
true spin-GGA lowers the energy by ~0.13 Ha at these settings.

**Oracle note:** Host has no loadable CP2K binary and no `pw.x` (Quantum
ESPRESSO) on PATH. Absolute \(e_{\mathrm{tot}}\) is a **fixed-settings offline
method pin** for regression stability (same PP, ecut, cell, geometry). Re-pin
against QE with this UPF when an offline twin is available
(`tools/cp2k_oracle/inputs/pw-spin-gga.note`). Magnetization remains the
literature hard gate. Units: public energy on evaluators is eV; `PWResult`
reports Ha/Ry as elsewhere.

#### GGA activation pin

Same geometry/occupations with `xc = XCType::LSDA`:

\[
|E_{\mathrm{PBE}} - E_{\mathrm{LSDA}}| > 10^{-4}~\mathrm{Ha}.
\]

Observed 2026-07-26: \(|E_{\mathrm{PBE}}-E_{\mathrm{LSDA}}| \approx 0.129\) Ha
(\(E_{\mathrm{LSDA}}\approx -28.62946347\) Ha). Documents that the production
spin path is not stuck in a pure density-only LDA limit of a single functional
call.

### Perf

| Field | Value |
| ----- | ----- |
| perf_exempt | true |
| reason | No fair CP2K/QE NC-UPF wall twin on pin host (module dep / no `pw.x`) |
| T_atv (this host, single rank, PBE+LSDA) | ~60 s order of magnitude |
| ratio T_atv/T_oracle | n/a |

### Dual gate

1. Correctness: mag + e_tot + GGA activation gates above; both SCFs converge.
2. Perf: exempt (no oracle wall).

### Provenance

- Spec: `.claude/specs/pw-spin-gga/spec.md`
- Functional: PBE (1996); `cpu::scf::evaluate_xc` + spectral spin gradients
- Sibling: closed-shell PW GGA `reg_pw-pbe-gga-si`; spin foothold
  `reg_pw-spin-polarized-o2` (now also true spin-GGA when `xc=PBE`)


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
