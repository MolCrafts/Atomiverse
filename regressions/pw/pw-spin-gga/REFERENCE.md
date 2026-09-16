# Reference — true collinear spin-GGA PBE on O₂ triplet (plane-wave)

## Method

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

## System

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

## Pins

### Magnetization (literature + fixed occupations)

O₂ ground state is the triplet ³Σg⁻ (Herzberg). With fixed integer occupations
and multiplicity 3:

\[
M = N_\uparrow - N_\downarrow = 2.
\]

**Gate:** \(|M - 2| \le 1\times 10^{-9}\) on `PWResult::magnetization`.

### Total energy (offline method pin)

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

### GGA activation pin

Same geometry/occupations with `xc = XCType::LSDA`:

\[
|E_{\mathrm{PBE}} - E_{\mathrm{LSDA}}| > 10^{-4}~\mathrm{Ha}.
\]

Observed 2026-07-26: \(|E_{\mathrm{PBE}}-E_{\mathrm{LSDA}}| \approx 0.129\) Ha
(\(E_{\mathrm{LSDA}}\approx -28.62946347\) Ha). Documents that the production
spin path is not stuck in a pure density-only LDA limit of a single functional
call.

## Perf

| Field | Value |
| ----- | ----- |
| perf_exempt | true |
| reason | No fair CP2K/QE NC-UPF wall twin on pin host (module dep / no `pw.x`) |
| T_atv (this host, single rank, PBE+LSDA) | ~60 s order of magnitude |
| ratio T_atv/T_oracle | n/a |

## Dual gate

1. Correctness: mag + e_tot + GGA activation gates above; both SCFs converge.
2. Perf: exempt (no oracle wall).

## Provenance

- Spec: `.claude/specs/pw-spin-gga/spec.md`
- Functional: PBE (1996); `cpu::scf::evaluate_xc` + spectral spin gradients
- Sibling: closed-shell PW GGA `reg_pw-pbe-gga-si`; spin foothold
  `reg_pw-spin-polarized-o2` (now also true spin-GGA when `xc=PBE`)
