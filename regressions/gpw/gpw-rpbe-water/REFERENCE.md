# Reference — isolated H₂O / RPBE / DZVP-MOLOPT-GTH (Martyna–Tuckerman)

## Method

Γ-point closed-shell GPW Kohn–Sham DFT with the **RPBE** GGA functional
(Hammer, Hansen, Nørskov, *Phys. Rev. B* **59**, 7413 (1999)): exponential
exchange enhancement

\[
F_x^{\mathrm{RPBE}}(s) = 1 + \kappa\bigl(1 - e^{-\mu s^2/\kappa}\bigr)
\]

with PBE \(\kappa = 0.804\), \(\mu \approx 0.21951\), and **PBE correlation**
(libxc `GGA_X_RPBE` + `GGA_C_PBE` / CP2K `XC_FUNCTIONAL RPBE`). **Not** the
same as **revPBE** (Zhang & Yang κ-revision of the rational PBE form).

Shared pointwise engine: `cpu::scf::evaluate_xc(XCType::RPBE, …)` on the GPW
grid (`xc_grid_rpbe`).

Public API: `cpu::gpw::GPWEvaluator` + `cpu::single_point` (Frame in/out) with
`xc = XcKind::RPBE`. Kinds use lookup constructors
`GaussianBasis(Element, name)` / `GthPotential(Element, name)` — no public
`*Config` bag, no factories.

## System

Isolated water, CP2K getting-started geometry (Å), cubic box \(L = 10\) Å
(same twin as `gpw-water-pbe-mt` / `gpw-blyp-water`):

| Atom | x | y | z |
| ---- | --: | --: | --: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H | 5.7586 | 5.0000 | 5.5043 |
| H | 4.2414 | 5.0000 | 5.5043 |

- Basis: **DZVP-MOLOPT-GTH** (O, H)
- Pseudopotentials: **GTH-PBE-q6** (O), **GTH-PBE-q1** (H)
- XC: **RPBE** (and PBE twin for dual-gate)
- Density cutoff: **400 Ry**
- Valence electrons: 8 → \(n_\mathrm{occ} = 4\)
- SCF: mix 0.3, max_iter 200, tol \(10^{-7}\)
- BC: isolated Martyna–Tuckerman (`nonperiodic = true`)

## Offline oracle status

**No CP2K binary / module** on the worker host at pin time (`cp2k`,
`cp2k.psmp`, `cp2k.sopt` not on PATH; `module avail cp2k` empty). Absolute
total-energy digits from CP2K are therefore **not** invented from memory and
**not** taken from a live third-party process at ctest time.

Pin strategy (binding, mirrors `gpw-blyp-water` / `gpw-revpbe-si`):

1. **RPBE activation dual-gate.** Same geometry with `XcKind::PBE` must shift
   the total energy by at least
   \[
   |E_{\mathrm{RPBE}} - E_{\mathrm{PBE}}| > 10^{-4}~\mathrm{Ha}.
   \]
2. **Convergence / finiteness.** SCF converges; \(E_{\mathrm{RPBE}}\) finite;
   Frame `kEnergy` finite [eV].
3. **Absolute first-green ATV band.** Frozen after first green for setup
   integrity (not a third-party oracle). When CP2K is available offline, re-run
   `tools/cp2k_oracle/inputs/gpw-rpbe-water.inp` and replace the absolute pin
   with the twin \(E_{\mathrm{tot}}\) (cite CP2K version), tighten tol.

### Atomiverse numbers (verified 2026-07-25, first green)

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{RPBE}}\) | **−17.2435550328 Ha** |
| \(E_{\mathrm{PBE}}\) | −17.2127580657 Ha |
| \(\lvert\Delta E\rvert\) | \(3.080 \times 10^{-2}\) Ha (\(> 10^{-4}\)) |
| SCF iters (RPBE) | 9 |
| Wall RPBE / PBE (OMP=4, Release) | ~20.8 s / ~5.9 s |

PBE twin matches the established `gpw-water-pbe-mt` residual
(\(E_{\mathrm{PBE}} \approx -17.212758\) Ha vs offline CP2K
−17.212337 Ha).

## Tolerance

- Dual-gate floor: \(10^{-4}\) Ha (**hard**).
- Absolute: \(1 \times 10^{-3}\) Ha vs first-green ATV pin until offline CP2K
  replaces it.

## What is checked

1. SCF converges for RPBE.
2. Finite \(E_{\mathrm{RPBE}}\) and Frame energy.
3. Dual-gate \(|E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha.
4. Absolute soft band vs first-green ATV.
5. No external process spawn.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary on pin host for fair wall twin. |
| \(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | n/a (exempt) |
| `within_10x` | n/a (exempt) |
| \(T_{\mathrm{atv}}\) (RPBE SCF, OMP=4) | ~20.8 s |

When CP2K is available, time both single-rank runs (same OMP) and replace this
section with measured \(r\). Target \(r \le 10\).

## Provenance

- Spec: `.claude/specs/gpw-rpbe-water/`
- Functional: RPBE (Hammer, Hansen, Nørskov 1999); shared `evaluate_xc`
- GTH: Krack, TCA **114**, 145 (2005); MOLOPT: VandeVondele & Hutter, JCP
  **127**, 114105 (2007)
- Twin input: `tools/cp2k_oracle/inputs/gpw-rpbe-water.inp`
- Sibling PBE pin: `regressions/gpw/gpw-water-pbe-mt/REFERENCE.md`
