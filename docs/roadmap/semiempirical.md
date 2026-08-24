# Semiempirical and Tight-Binding Methods

Legend and caveats: [Roadmap index](index.md).

Atomiverse ships two **concrete NDDO methods**: **AM1** and **RM1**
(`cpu::semiempirical::AM1` / `RM1`; evaluators `AM1Evaluator` / `RM1Evaluator`
of kind `cpu::Semiempirical`). The shared NDDO engine is **internal**
(`NDDOSolver` is not a public bag). Methods compute energies, Mulliken
charges, the density matrix, and **nuclear forces** via fixed-density central
finite difference of the SCF energy (product path; matches re-SCF FD at SCF
stationarity). Closed-form multipole / integral derivatives remain ○ (see
queue `nddo-closed-form-forces`). AM1 and RM1 now carry **absolute OpenMOPAC
ΔHf pins** at each method's own optimized geometry
(`reg_am1-water-mopac-energy`, `reg_rm1-water-mopac-energy`, 2026-07-29); the
RM1 free-atom energies were AM1-derived and were repaired to MOPAC's `calpar`
identity (a 19 kcal/mol ΔHf error), now gated for every parameter row. Residual
against MOPAC is ~1e-3 eV and is a **physical-constant convention** floor (ATV
uses CODATA-2018; MOPAC froze a₀ = 0.529167 Å, 1 Ha = 27.21 eV), not a defect.
Evaluators write forces for Evaluate / MD / Minimize by default. Launch unwraps an isolated fragment for molecular AM1
under PBC wrap (same pattern as Gaussian SCF). Regressions: `am1_water_force`,
`am1_water_minimize`, `am1_water_bomd`, `rm1_water_energy`. Tutorial:
[Chapter 17](../tutorial/ch17-semiempirical-am1.md) + `examples/15_am1.cpp`.
Heat of formation is MOPAC-style. Hard OpenMOPAC energy pins and PM6/PM7 remain
open. AM1-BCC charge assignment still uses the energy/charge path (see
[Multiscale and Properties](multiscale.md)).

**SCC-DFTB2** multi-element H/C/N/O (`DFTB2`) ships with static in-tree
**mio-style sketch** SK tables (`reg_h2_dftb2`, `reg_h2o_dftb2_mio`),
**mio-1-1 H–H** radial grids for the absolute DFTB+ energy pin
(`dftb_mio_sk_tables.h`, `reg_dftb2-dftbplus-energy-pin`), and production
**3ob-3-1** organic CHNO tables (`DftbSkSet::ThreeOb`, `reg_dftb2-3ob-organic`).
**Nuclear forces** are analytic on MioSketch for closed shell **and collinear
open shell** (Hellmann–Feynman + Pulay + SCC γ + repulsive;
`reg_dftb2-analytic-forces`, `reg_dftb2-spin-analytic-forces`) and on the
**ThreeOb** organic path (`reg_dftb2-analytic-forces-threeob`). **Collinear
spin-DFTB2** ships (`multiplicity`, dual α/β, Mulliken m; `reg_spin-dftb2-o2`).
**Γ-point periodic** SCC-DFTB2 lattice sums (H⁰/S, screened γ, repulsive) ship
on MioSketch (`reg_dftb2-periodic-gamma`) with **lattice-image-consistent
nuclear forces** under PBC (`reg_dftb2-periodic-forces`). **Monkhorst–Pack
k-mesh** beyond Γ ships for closed-shell MioSketch **energy**
(`k_mesh` / `k_time_reversal` members; `reg_dftb2-kpoints`) **and, since
2026-07-29, for multi-k analytic nuclear forces and cell stress** via the
Bloch-sum image-resolved densities (`reg_dftb2-multi-k-forces`). Opt-in
**Grimme D3(BJ)** dispersion ships both as an additive **molecular** correction
and as a **Γ-PBC lattice sum** (`DFTB2::d3`, `d3_periodic`, `d3_cutoff`;
`reg_dftb2-dispersion-d3`, `reg_dftb2-dispersion-d3-periodic`), each with its
own analytic gradient and virial. Non-SCC **DFTB0** ships (`reg_dftb0-h2`).
**GFN0-xTB toy** was deleted (`XtbGfn0` / `reg_h2o_gfn0` gone as of
`bc14dabd`); production GFN stays ○. **DFTB3** ships energy
(`reg_dftb3-water-mio`) — after its third-order potential was corrected to be
the derivative of its own energy — **and closed-form nuclear forces** including
∂Γ/∂R (`reg_dftb3-analytic-forces`), plus **Γ-PBC lattice-image forces**
(`reg_dftb3-periodic-forces`); DFTB3 still has no k-mesh, cell stress, open
shell, ThreeOb or D3. **Analytic stress**
ships as the closed-form pair virial on **every** DFTB2 product path —
restricted MioSketch, collinear open shell, ThreeOb 3ob-3-1, and multi-k
(`reg_dftb2-analytic-stress`, `reg_dftb2-spin-analytic-stress`,
`reg_dftb2-analytic-stress-threeob`, `reg_dftb2-multi-k-forces`); **no
multi-SCC strain-FD product path is left on DFTB2**, and the virial now carries
the self-image A–(A+T) bonds it previously dropped (a 14% σ error on a dense
periodic cell). Wave-4 TB staff closed 2026-07-29.

**γ_AB (2026-07-29):** MioSketch DFTB2/DFTB3 use the DFTB+ **Elstner**
short-range kernel (`internal/dftb_gamma.h`, `1/R − expGamma`). History:
`.claude/notes/dftb-vs-dftbplus-cleanup.md`. Loop queue:
`docs/developer/tb-method-queue.md`. Family oracles: OpenMOPAC (NDDO) /
DFTB+ (DFTB) / xtb (GFN) — see `tools/tb_oracle/README.md`.

## 1. Semiempirical Quantum Methods

| Method                          | CPU | CUDA | Kokkos |
| ------------------------------- | :-: | :--: | :---: |
| AM1                             |  ✓  |   ○  |   ○  |
| RM1                             |  ✓  |   ○  |   ○  |
| PM3                             |  ○  |   ○  |   ○  |
| PM6                             |  ○  |   ○  |   ○  |
| PM7                             |  ○  |   ○  |   ○  |
| OM3                             |  ○  |   ○  |   ○  |
| Semiempirical nuclear forces (product path) |  ✓  |   ○  |   ○  |
| Semiempirical closed-form integral-derivative forces |  ○  |   ○  |   ○  |
| Semiempirical analytic Hessian  |  ✓  |   ○  |   ○  |
| AM1 heat of formation (MOPAC-style ΔHf) |  ✓  |   ○  |   ○  |

The NDDO core (overlap, one-electron core Hamiltonian, two-electron integrals,
Fock build, density, SCF, forces) is parameterization-agnostic in structure.
**AM1** and **RM1** (H/C/N/O/P/S; Rocha et al. 2006 / MOPAC tables) are
transcribed. PM3/PM6/PM7 remain tables away (oracle = **OpenMOPAC**). OM3 is
not in the OpenMOPAC primary surface (blocked in the TB queue until a
family-locked pin exists). Product-path forces are fixed-density FD; closed-form
multipole derivatives stay ○. Analytic Hessian remains finite-difference via
`cpu::vib::Hessian` over SE forces (VIB off under SE-only loop cmake).

## 2. Density-Functional Tight Binding

| Method                    | CPU | CUDA | Kokkos |
| ------------------------- | :-: | :--: | :---: |
| Non-SCC DFTB / DFTB0      |  ✓  |   ○  |   ○  |
| SCC-DFTB / DFTB2          |  ✓  |   ○  |   ○  |
| DFTB3                     |  ✓  |   ○  |   ○  |
| Spin-polarized DFTB       |  ✓  |   ○  |   ○  |
| Noncollinear-spin DFTB    |  ○  |   ○  |   ○  |
| DFTB+U                    |  ✓  |   ○  |   ○  |
| Dispersion-corrected DFTB |  ✓  |   ○  |   ○  |
| Spin-orbit DFTB           |  ○  |   ○  |   ○  |
| Periodic DFTB             |  ✓  |   ○  |   ○  |
| DFTB analytic gradient    |  ✓  |   ○  |   ○  |
| DFTB analytic stress      |  ✓  |   ○  |   ○  |
| TD-DFTB                   |  ✓  |   ○  |   ○  |

DFTB2 is multi-element H/C/N/O with static SK tables (`dftb_mio_tables.h` sketch
+ `dftb_mio_sk_tables.h` mio-1-1 H–H + `dftb_3ob_tables.h` 3ob-3-1 organics).
Collinear spin (`multiplicity`) and **analytic nuclear forces**
(HF+Pulay+γ+rep; **open shell included** on MioSketch; **ThreeOb** closed-form
F ships) are product paths. **Analytic stress** ships as the closed-form pair
virial on restricted MioSketch (`reg_dftb2-analytic-stress`) **and on ThreeOb**
(`reg_dftb2-analytic-stress-threeob`; all six Voigt components vs strain-FD at
5e-13 eV/Å³, plus the isotropic tr(σ)/3 = −dE/dΩ identity); only collinear
open shell still falls back to multi-SCC affine strain-FD. Offline **DFTB+** absolute energy
identity on mio-1-1 H–H ships (`reg_dftb2-dftbplus-energy-pin`); 3ob CH4 has a
DFTB+ pin under pure SCC-DFTB2. **Γ-point PBC** lattice-sum energy **and**
lattice-image nuclear forces ship; **multi-k energy** (Monkhorst–Pack) ships
for closed-shell MioSketch. **TD-DFTB** ships as a linear-response Casida
foothold on the restricted MioSketch ground state
(`DFTB2::td_casida`; Niehaus et al., PRB **63**, 085108 (2001);
`reg_td-dftb-h2`): **energies + oscillator strengths** (length form; triplets
dark), no excited-state gradients, singlet + triplet, molecular Γ only. It has
**no absolute energy oracle** for H₂/mio-1-1 excitations, so it is gated on the
λ→0 single-particle limit, Ω symmetry, the transition-charge sum rule, analytic
two-level identities, and \(f\geq0\); see its `REFERENCE.md` `Gaps`.
**γ_AB** is DFTB+ Elstner (`internal/dftb_gamma.h`). Wave 4 also landed
**multi-k analytic forces + stress**, **DFTB3 analytic forces**, **periodic
lattice-sum D3**, and **ThreeOb / open-shell analytic stress**. **Multi-k
collinear open shell** (dual α/β H(k) + dual-channel forces) ships 2026-07-29.
Still ○: **TD-DFTB excited-state gradients**, **long-range-corrected TD-DFTB**,
**ThreeOb multi-k**, and **DFTB3 beyond molecular min-image** (no k-mesh /
stress / open shell / ThreeOb / D3 on the DFTB3 path — Γ-PBC forces do ship).
Oracle for all DFTB rows: **DFTB+** (never OpenMOPAC). Ladder: DFTB0 → DFTB2
depth → DFTB3 F → TD → NDDO PM → GFN late.

## 3. Extended Tight-Binding Methods

| Method                    | CPU | CUDA | Kokkos |
| ------------------------- | :-: | :--: | :---: |
| GFN0-xTB                  |  ○  |   ○  |   ○  |
| GFN1-xTB                  |  ○  |   ○  |   ○  |
| GFN2-xTB                  |  ○  |   ○  |   ○  |
| Spin-polarized GFN-xTB    |  ○  |   ○  |   ○  |
| IPEA-xTB                  |  ○  |   ○  |   ○  |
| Periodic GFN-xTB          |  ○  |   ○  |   ○  |
| GFN-xTB analytic gradient |  ○  |   ○  |   ○  |
| GFN-xTB analytic Hessian  |  ○  |   ○  |   ○  |
| GFN-FF                    |  ○  |   ○  |   ○  |
| sTDA-xTB                  |  ○  |   ○  |   ○  |

GFN0/1/2 product types are **absent** (the H/C/O `XtbGfn0` sketch was deleted;
INDEX GFN2-CHON rows are stale — sources never landed on `dev`). GFN-xTB is
**not** a DFTB3 successor; schedule after the NDDO PM ladder (oracle = **xtb**). GFN-FF is classical force-field territory (scope-out
of the electronic SE/TB capability loop).
