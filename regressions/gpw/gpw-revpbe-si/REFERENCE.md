# Reference — Γ-GPW revPBE on Si₂ (DZVP-MOLOPT-GTH)

## Method

Closed-shell GPW Kohn–Sham DFT with the **revPBE** GGA functional
(Zhang & Yang, *Phys. Rev. Lett.* **80**, 890 (1998)): same analytic form as
PBE with revised exchange enhancement \(\kappa = 1.245\) (PBE uses
\(\kappa = 0.804\)); \(\mu\) and correlation \(\beta\) match PBE
(libxc `GGA_X_PBE_R` + `GGA_C_PBE` / CP2K `XC_FUNCTIONAL revPBE`). PW92 LDA
correlation base.

Public API: concrete `cpu::gpw::GpwDft` with members (`xc = XcKind::RevPBE`,
`cutoff_ry`, …); Frame-compatible kinds via lookup constructors
`GaussianBasis(Element::Si, "DZVP-MOLOPT-GTH")` /
`GthPotential(Element::Si, "GTH-PBE-q4")` — no public `*Config`, no factories.
Shared pointwise engine: `cpu::scf::evaluate_xc(XCType::RevPBE, …)`.

## System

Silicon **2-atom diamond fragment** in cubic cell \(a = 5.43\) Å:

| | |
| --- | --- |
| Lattice | cubic \(a = 5.43\) Å |
| BC | **isolated** Martyna–Tuckerman (`nonperiodic = true`; CP2K `PERIODIC NONE` / `PSOLVER MT`) — periodic MOLOPT image sums exceed the ≤ 60 s e2e budget for this twin; dual-gate still proves revPBE on the GPW GGA path |
| Positions [Å] | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Basis | **DZVP-MOLOPT-GTH** (Si) |
| Pseudopotential | **GTH-PBE-q4** (zion 4) |
| Valence | 8 e⁻ → \(n_{\mathrm{occ}} = 4\) |
| XC | **revPBE** (and PBE twin for dual-gate) |
| Density cutoff | **200 Ry** |
| SCF | mix 0.4, max_iter 60, tol \(10^{-6}\) |

## Offline oracle status

**No CP2K binary / module** on the worker host at pin time (`cp2k`,
`cp2k.psmp`, `cp2k.sopt` not on PATH; `module avail cp2k` empty). Absolute
total-energy digits from CP2K are therefore **not** invented from memory and
**not** taken from a live third-party process at ctest time.

Pin strategy (binding, mirrors `gpw-pbesol-si` / `gpw-blyp-water`):

1. **revPBE activation dual-gate.** Same geometry with `XcKind::PBE` must shift
   the total energy by at least
   \[
   |E_{\mathrm{revPBE}} - E_{\mathrm{PBE}}| > 10^{-4}~\mathrm{Ha}.
   \]
2. **Convergence / finiteness.** SCF converges; \(E_{\mathrm{revPBE}}\) finite.
3. **Absolute first-green ATV band.** Frozen after first green for setup integrity
   (not a third-party oracle). When CP2K is available offline, re-run
   `tools/cp2k_oracle/inputs/gpw-revpbe-si.inp` and replace the absolute pin
   with the twin \(E_{\mathrm{tot}}\) (cite CP2K version), tighten tol.

### Atomiverse numbers (verified 2026-07-25, first green)

| Quantity | Value |
| --- | --- |
| \(E_{\mathrm{revPBE}}\) | **−13.0900394799 Ha** |
| \(E_{\mathrm{PBE}}\) | −13.0743742822 Ha |
| \(\lvert\Delta E\rvert\) | \(1.567 \times 10^{-2}\) Ha (\(> 10^{-4}\)) |
| SCF iters (revPBE) | 10 |
| Wall revPBE / PBE (OMP=4, Release) | ~14.8 s / ~9.2 s |

PBE twin matches the established `gpw-pbesol-si` residual
(\(E_{\mathrm{PBE}} \approx -13.074374\) Ha).

## Tolerance

- Dual-gate floor: \(10^{-4}\) Ha (**hard**).
- Absolute: \(1 \times 10^{-3}\) Ha vs first-green ATV pin until offline CP2K
  replaces it.

## What is checked

1. SCF converges for revPBE.
2. Finite \(E_{\mathrm{revPBE}}\).
3. Dual-gate \(|E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha.
4. Absolute soft band vs first-green ATV.
5. No external process spawn.

## Performance

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| reason | No CP2K binary on pin host for fair wall twin. |
| \(r = T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) | n/a (exempt) |
| `within_10x` | n/a (exempt) |
| \(T_{\mathrm{atv}}\) (both SCFs, OMP=4) | ~24 s |

When CP2K is available, time both single-rank runs (same OMP) and replace this
section with measured \(r\). Target \(r \le 10\).

## Provenance

- Spec: `.claude/specs/gpw-revpbe-si/`
- Functional: revPBE (Zhang & Yang 1998); shared `evaluate_xc`
- GTH: Krack, TCA **114**, 145 (2005); MOLOPT: VandeVondele & Hutter, JCP
  **127**, 114105 (2007); transcribed from CP2K `GTH_POTENTIALS` /
  `BASIS_MOLOPT` (v2024.1 data files)
- Twin input: `tools/cp2k_oracle/inputs/gpw-revpbe-si.inp`

## Re-pin 2026-07-26 — the previous value was unphysical

The earlier pin **-13.0900394799 Ha** was taken with an atom at the **cell origin** under
Martyna–Tuckerman cluster boundaries. That path does not wrap the collocation, so
the Gaussian density is clipped at the cell faces: the energy still looks
plausible but is badly wrong, and the forces are catastrophically so (31.9 Ha/Bohr
measured on a water dimer — see `../gpw-d3-forces/REFERENCE.md` "Placement").

With the fragment **centred** in the same cell the energy is **-7.2202596767 Ha** — the
old pin was ≈ 5.9 Ha (77 %) too low. The centred magnitude is consistent with the
independent offline **CP2K 2026.1** Si₂ reference taken this wave
(UKS triplet, isolated, −7.613 Ha; see `../gpw-fd-phonon-si/REFERENCE.md`), which
is the sanity anchor the old value lacked. SCF now converges in 9 iterations
(0.4 s) instead of grinding against the clipped potential.

`GpwDft::build` now **throws** on off-centre nonperiodic systems, so this class of
pin cannot be taken again.

**Still a self-golden pin.** CP2K cannot converge this closed-shell Si₂ in a
5.43 Å MT box — verified 2026-07-26 with CP2K 2026.1 (`SCF run NOT converged`),
after fixing the stored oracle input, which used the invalid keyword
`&XC_FUNCTIONAL PBEsol` / `revPBE` and had therefore never run. The correct CP2K
spelling is `&PBE / PARAMETRIZATION PBESOL` (resp. `REVPBE`); the fixed decks are
in `tools/cp2k_oracle/inputs/`. Consequently this case gates **functional
activation and GPW grid-path wiring** (PBEsol/revPBE vs PBE separation), not
absolute Si energetics. A true twin needs an open-shell treatment of Si₂ or a
different, better-conditioned system.
