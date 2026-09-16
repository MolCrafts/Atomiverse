# Reference — PW collinear spin-polarized O₂ triplet

## Method

Collinear spin-polarized plane-wave KS (`cpu::pw::PW` with
`spin_polarized = true`, `multiplicity = 3`, `xc = XCType::PBE`). Fixed
occupations N↑ = 7, N↓ = 5 for 12 valence electrons (oxygen NC-PP Z_val = 6 × 2).
XC: **true collinear spin-GGA PBE** via spectral \(\nabla\rho_\sigma\) in
`XCBuilder::step_spin` (productized by `pw-spin-gga`; no longer the historical
\(\sigma=0\) LSDA-limit path). Γ-only Monkhorst–Pack mesh.

## System

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

## Pins

### Magnetization (literature + fixed occupations)

O₂ ground state is the triplet ³Σg⁻ (Herzberg, *Molecular Spectra and
Molecular Structure* I; standard open-shell test). With fixed integer
occupations and multiplicity 3:

\[
M = N_\uparrow - N_\downarrow = 2.
\]

**Gate:** `|M − 2| ≤ 1×10⁻⁹` (occupation identity on `PWResult::magnetization`).

### Total energy (offline method pin)

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

## Perf

| Field | Value |
| ----- | ----- |
| perf_exempt | true |
| reason | CP2K binary not loadable on host (module dep); no fair single-rank CP2K PW/NC-UPF twin for wall ratio |
| T_atv (this host, single rank) | ~40 s (order of magnitude; not for ratio) |
| ratio T_atv/T_cp2k | n/a |

## Dual gate

1. Correctness: mag + e_tot gates above; SCF must converge.
2. Perf: exempt (no CP2K twin wall).
