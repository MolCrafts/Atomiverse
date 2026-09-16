# Pin notes — pim-npt-polarizable-il

Slug: `pim-npt-polarizable-il` (metalwalls-method-loop wave 5;
re-pinned 2026-07-29 against the real MetalWalls binary).
Regression: `regressions/md/pim-npt-polarizable-il/`.

## Oracle — corrected

**Was** (waves 1–5, no binary on the host): `oracle=metalwalls`, provenance
cited as "MetalWalls JOSS **5**, 2373 (2020) bulk PIM + Thole,
*Chem. Phys.* **59**, 341 (1981) damping".

**Now**: `oracle=literature (Applequist/Thole), MetalWalls-form-validated`,
`mw_expressible=false`. The Thole half of the old citation was wrong —
**MetalWalls implements no Thole damping** — and the pinned numbers are not
MetalWalls numbers. Full evidence, raw MW output and the unit conversions
are in `regressions/md/pim-npt-polarizable-il/REFERENCE.md`; the short form:

- MW binary `git c8ad175d` (release branch 2026-06-18) reproduces its own
  shipped `tests/pim/3D-ishii-nodisp/energies_breakdown.ref` to 1e-14
  relative, so the tool is trustworthy.
- **Form check passes.** A +1(α=0)/−1(α=3 ų) dimer at d = 2.35 Å in boxes of
  30/45/60/90 Bohr: MW's `E_pot(pol) − E_pot(nopol)` converges as 1/L³ onto
  the exact isolated-pair value −0.708225558187 eV; 1/L³ Richardson gives
  −0.708234375674 eV (1.2e-5 relative, the extrapolation residual).
  ⇒ ATV's μ = αE, E_pol = −½ Σμ·E^perm (self term included) and
  k = e²/4πε₀ are **confirmed correct against real MetalWalls**.
- **Lattice number differs by 5.17×.** On the exact regression geometry MW
  gives E_pol = −1.682169167852 eV vs the pinned −8.695749870556 eV
  (relative discrepancy 4.169). Cause isolated with the same numpy oracle:
  Thole on/off changes the rc = 4.0 Å number by 0.9 %; rc 4.0 → 4.7 Å
  (= L/2) changes it by 1.8×; the reciprocal-space tail supplies the
  remaining ~2.9×. The truncated charge–dipole field in a rock-salt lattice
  is a cancelling alternating sum and is not a converged quantity.
- **Why MW cannot express the pin** (two independent gaps):
  1. no Thole damping anywhere (`grep -ri thole` over `src/` + `doc/` → 0
     hits); MW's only dipole damping is Tang–Toennies on the *charge–dipole*
     term (`src/damping.f90`, applied at
     `src/coulomb_sr_pim_3DPBC.inc:514–536`), and its dipole–dipole tensor
     is undamped;
  2. no truncated-Coulomb mode: `num_pbc` is 2 or 3 only and electrostatics
     is always Ewald-split — `coulomb_rcut` is the erfc splitting radius,
     not a truncation of the physical interaction.

Pin values and tolerances therefore **kept unchanged**. Nothing was
loosened, weakened, or adjusted to match.

## Inputs

- MW twin generator + generated cases:
  `tools/metalwalls_oracle/inputs/metalwalls/pim-npt-polarizable-il/`
  (`gen_mw_twin.py`, `lattice64/{pol,nopol}`, `dimer_L{030,045,060,090}/`,
  `npt250/run/`).
- Offline numpy pin (unchanged):
  `tools/metalwalls_oracle/inputs/pim_npt_polarizable_il_pin.py`.
  Never executed at build/ctest/regression time.

## MetalWalls bugs found while building the twin

1. **Segfault when a PIM run omits the `damping` section.** `system%tt%b /
   %order / %c` are allocated only inside `read_damping_parameters`
   (`src/configuration.F90:3264`), but the PIM short-range kernel
   dereferences them unconditionally (`src/coulomb_sr_pim_3DPBC.inc:514`).
   Workaround in the twins: an explicit `damping` block with intensity
   `c = 0` (identity damping).
2. **Division by zero on `tt_pair … 0 …`.** `src/configuration.F90:3307`
   validates integrality with `tt_order/int(tt_order)`, which traps for
   order 0. Workaround: order 4 with `c = 0`.

## Configuration (must match the regression exactly)

- 4×4×4 simple-cubic rock-salt toy polarizable IL, d = 2.35 Å,
  cubic box L = 9.4 Å (V = 830.584 ų), index order ix→iy→iz.
- Charges (−1)^(ix+iy+iz): +1 cation (type 0, α = 0),
  −1 anion (type 1, α = 3.0 ų).
- Deterministic displacement per atom index i, axis a ∈ {0,1,2}:
  `disp = 0.15 · sin(1.7·(i+1) + 2.3·a)` Å (breaks the centro-symmetry
  that would make E_pol ≡ 0 on the perfect lattice).
- PIM: real-space cutoff 4.0 Å (minimum image, hard truncation),
  thole_a = 0.39, tt_b = 0, Jacobi SCC to max|Δμ| < 1e-13 e·Å (25 iters).
- Coulomb prefactor k = 14.399645 eV·Å/e² (`units::coulomb`).

## Pinned values (hard-coded in `reg_pim-npt-polarizable-il.cpp`)

| Observable | Value | Units |
| --- | --- | --- |
| E_pol | −8.695749870555906 | eV |
| W_xx | −10.66184373345355 | eV |
| W_yy | −12.125914211727187 | eV |
| W_zz | −9.657852793291474 | eV |
| W_xy | −0.2325271422922529 | eV |
| W_xz | −2.2469237837238 | eV |
| W_yz | +1.375774440282161 | eV |
| P_pol = tr(W)/(3V) | −0.013021203851134545 | eV/ų |
| F0 | (+1.3910743837872235, −0.5347228671088384, +0.3580748823060393) | eV/Å |

Tolerances: |ΔE| ≤ 1e-8 eV, |ΔW| ≤ 1e-7 eV, |ΔP_pol| ≤ 1e-10 eV/ų
(covers std::sin vs np.sin position ulps, summation order, and the 1e-13
SCC tail).

## Performance (re-pin 2026-07-29 — no longer exempt)

Twin `npt250/run/`: same 64-ion box as the NPT product leg (d = 2.9 Å,
L = 11.6 Å), 250 steps, dt = 1 fs, T = 800 K, τ_T = 100 fs, P = 0,
τ_P = 500 fs, LJ 0.01 eV / 3.0 Å / rc 4.0 Å, MW at its own production
Ewald/dipole tolerances. `mpirun -n 1`, `OMP_NUM_THREADS=1`, same node.

- MetalWalls: 0.83 s wall (0.330 s MD loop).
- ATV `reg_pim-npt-polarizable-il`: 0.06 s wall (all three legs).
- ratio ATV/MW = **0.072** wall, 0.17 loop-only.

Caveat: MW does a full Ewald over charges + charge–dipole + dipole–dipole
with a CG dipole solve; ATV does a 4.0 Å-truncated O(N²) Jacobi SCC. ATV is
cheaper largely because it is doing less electrostatics.

## Capability gap reported upstream (out of slug scope)

`cpu::PimInducedDipole` has **no long-range path for the induced-dipole
field** — its `cutoff` is a hard multipole truncation, and pairing it with
`CoulLong`/PME extends only the permanent charge–charge term. For a bulk
periodic ionic system that makes E_pol and the polarization virial
cutoff-dependent by factors of ~5 (measured above). A dipole-capable
Ewald/PME (MW's `coulomb_lr_pim.f90` / `coulomb_keq0_pim.F90` are the
reference) is the missing capability. Queue row material, not a defect in
this pin.

Note (found gap, still open): `PairBorn` / `PairBuck` do not write
`kStress`, so a Fumi–Tosi NPT twin would barostat blind to the Born term —
which is why the product leg uses the LJ + coul/cut virial-correct stack.
