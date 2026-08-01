# Acceptance — `gpw-dos-si`

## Capability

Density of states on the **GPW multi-k mesh**: after multi-k GTH-GPW RKS SCF
on diamond Si, Gaussian-broaden \(\varepsilon_{nk}\) into \(D(E)\) (closed-shell
spin degeneracy 2). Matched twin class: CP2K 2026.1 Quickstep GPW +
`MONKHORST-PACK 2 2 2` + `&PRINT &DOS` / PDOS.

## Public API

- `cpu::gpw::Dos` members: `k_mesh`, `n_bands`, `sigma_ha`, `n_bins`, …
- `DosResult compute(GpwDft&, atoms)` — multi-k SCF + DOS
- Free math: `gap_from_spectrum`, `gaussian_dos`, `integrate_dos_to`
- No public `*Config`, no `make_*` / `build_*` factories

## Gates

| Id | Gate | Tol |
| --- | --- | --- |
| A1 | \(\lvert E_{\mathrm{atv}}-E_{\mathrm{cp2k}}\rvert\) | ≤ 1e-3 Ha |
| A2 | \(\int^{E_{\mathrm{vbm}}} D(E)\,dE \approx 16\); open gap ≥ 0.1 eV; \(D\ge 0\) | n_elec ± 1.5 |
| Perf | \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) on same twin SCF | ≤ 10 |

## Oracle

- Input: `tools/cp2k_oracle/inputs/gpw-dos-si/si_gpw_dos.inp`
- Engine: CP2K 2026.1 (`cp2k.psmp`, git:5e54ba2)
- Offline only; never live oracle in ctest
