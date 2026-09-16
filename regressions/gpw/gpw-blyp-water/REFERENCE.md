# Reference — isolated H₂O / BLYP / DZVP-MOLOPT-GTH (Martyna–Tuckerman)

## Method

Γ-point closed-shell GPW Kohn–Sham DFT with the **BLYP** GGA functional
(Becke exchange, *Phys. Rev. A* **38**, 3098 (1988) + Lee–Yang–Parr correlation,
*Phys. Rev. B* **37**, 785 (1988); libxc `GGA_X_B88` + `GGA_C_LYP` / CP2K
`XC_FUNCTIONAL BLYP`). Shared pointwise engine:
`cpu::scf::evaluate_xc(XCType::BLYP, …)` on the GPW grid (`xc_grid_blyp`).

Public API: `cpu::gpw::GPWEvaluator` + `cpu::single_point` (Frame in/out) with
`xc = XcKind::BLYP`. Kinds use lookup constructors
`GaussianBasis(Element, name)` / `GthPotential(Element, name)` — no public
`*Config` bag, no factories.

## System

Isolated water, CP2K getting-started geometry (Å), cubic box \(L = 10\) Å
(same twin as `gpw-water-pbe-mt`):

| Atom | x | y | z |
| ---- | --: | --: | --: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H | 5.7586 | 5.0000 | 5.5043 |
| H | 4.2414 | 5.0000 | 5.5043 |

- Basis: **DZVP-MOLOPT-GTH** (O, H)
- Pseudopotentials: **GTH-PBE-q6** (O), **GTH-PBE-q1** (H)
- XC: **BLYP** (and PBE twin for dual-gate)
- Density cutoff: **400 Ry**
- Valence electrons: 8 → \(n_\mathrm{occ} = 4\)
- SCF: mix 0.3, max_iter 200, tol \(10^{-7}\)
- BC: isolated Martyna–Tuckerman (`nonperiodic = true`)

## Offline oracle status

**No CP2K binary / module** on the worker host at pin time (`cp2k`,
`cp2k.psmp`, `cp2k.sopt` not on PATH; `module avail cp2k` empty). Absolute
total-energy digits from CP2K are therefore **not** invented from memory and
**not** taken from a live third-party process at ctest time.

Pin strategy (binding, mirrors `gpw-pbesol-si` / `gpw-water-pbe-mt`):

1. **BLYP activation dual-gate.** Same geometry with `XcKind::PBE` must shift
   the total energy by at least
   \[
   |E_{\mathrm{BLYP}} - E_{\mathrm{PBE}}| > 10^{-4}~\mathrm{Ha}.
   \]
2. **Convergence / finiteness.** SCF converges; \(E_{\mathrm{BLYP}}\) finite;
   Frame `kEnergy` finite [eV].
3. **Absolute first-green ATV band.** Frozen after first green for setup
   integrity (not a third-party oracle). When CP2K is available offline, re-run
   `tools/cp2k_oracle/inputs/gpw-blyp-water.inp` and replace the absolute pin
   with the twin \(E_{\mathrm{tot}}\) (cite CP2K version), tighten tol.

### Atomiverse numbers (verified 2026-07-25, first green)

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{BLYP}}\) | **−17.1982170663 Ha** |
| \(E_{\mathrm{PBE}}\) | −17.2127580657 Ha |
| \(\lvert\Delta E\rvert\) | \(1.454 \times 10^{-2}\) Ha (\(> 10^{-4}\)) |
| SCF iters (BLYP) | 12 |
| Wall BLYP / PBE (OMP=4, Release) | ~34.4 s / ~19.1 s |

PBE twin matches the established `gpw-water-pbe-mt` residual
(\(E_{\mathrm{PBE}} \approx -17.212758\) Ha vs offline CP2K
−17.212337 Ha).

## Tolerance

- Dual-gate floor: \(10^{-4}\) Ha (**hard**).
- Absolute: \(1 \times 10^{-3}\) Ha vs first-green ATV pin until offline CP2K
  replaces it.

## What is checked

1. SCF converges for BLYP.
2. Finite \(E_{\mathrm{BLYP}}\) and Frame energy.
3. Dual-gate \(|E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha.
4. Absolute soft band vs first-green ATV.
5. No external process spawn.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary on pin host for fair wall twin. |
| \(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | n/a (exempt) |
| `within_10x` | n/a (exempt) |
| \(T_{\mathrm{atv}}\) (BLYP SCF, OMP=4) | ~34.4 s |

When CP2K is available, time both single-rank runs (same OMP) and replace this
section with measured \(r\). Target \(r \le 10\).

## Provenance

- Spec: `.claude/specs/gpw-blyp-water/`
- Functional: B88 (1988) + LYP (1988); shared `evaluate_xc`
- GTH: Krack, TCA **114**, 145 (2005); MOLOPT: VandeVondele & Hutter, JCP
  **127**, 114105 (2007)
- Twin input: `tools/cp2k_oracle/inputs/gpw-blyp-water.inp`
- Sibling PBE pin: `regressions/gpw/gpw-water-pbe-mt/REFERENCE.md`
