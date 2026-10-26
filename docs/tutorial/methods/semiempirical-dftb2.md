# SCC-DFTB2

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Case: `ctest -R '^semiempirical/dftb2$'` — sources and oracle record in [`regressions/semiempirical/dftb2/`](../../../regressions/semiempirical/dftb2/). One section of the case per `##` heading below: `atv_regressions --case semiempirical/dftb2 --section <name>`.

Self-consistent-charge **DFTB2** (Elstner et al., *Phys. Rev. B* **58**, 7260
(1998)): minimal valence LCAO with two-centre Slater–Koster hoppings and
overlaps, second-order charge self-consistency on Mulliken fluctuations
\(\Delta q_A\) through the Elstner \(\gamma_{AB}\) kernel, and a short-range
pair repulsive; optionally collinear spin polarization (Köhler / DFTB+ spin
channel), D3(BJ) dispersion, DFTB+U, Γ-point or Monkhorst–Pack periodic
sums, analytic nuclear forces and analytic cell stress. The concrete method
is `cpu::semiempirical::DFTB2{sk_set, charge, multiplicity}` (constructor
arguments; `DftbSkSet::MioSketch` by default) with public members
(`boundary`, `scc_tol`, `max_scc`, `mix_beta`,
`spin_w_ha`, `d3`, `d3_periodic`, `d3_cutoff`, `k_mesh`, `k_time_reversal`,
`u_eff`, …), bound into `cpu::Evaluate` at `evaluate::Phase::evaluate`. No
`*Config` bag, no `make_*` / `build_*` factory. `run(frame)` overwrites energy
[eV] and charges [e], preserving forces; `forces(frame)` also overwrites
forces [eV/Å]. `stress(frame)` returns intensive stress [eV/Å³] and writes
the extensive virial on `kStress` [eV, Voigt `(xx,yy,zz,xy,xz,yz)`]. When
mounted in Evaluate, the method accumulates the requested energy, force and
supported stress contributions. +U has no cell-stress implementation; direct
`stress(frame)` rejects it. Mulliken data are also read back
through `last_mulliken_dq()` / `last_mulliken_spin()`. Linear-response
excitations on the same object are on [TD-DFTB](semiempirical-td-dftb.md#td-dftb-h2);
the third-order extension is [DFTB3](semiempirical-dftb3.md#dftb3-water-mio).

The case is CPU-only. Sections documented below:
[`dftb2-dftbplus-energy-pin`](#dftb2-dftbplus-energy-pin) (H₂ energy vs
DFTB+), [`dftb2-3ob-organic`](#dftb2-3ob-organic) (CH₄ on 3ob-3-1),
[`dftb2-analytic-forces`](#dftb2-analytic-forces) (H₂O closed-form forces),
[`spin-dftb2-o2`](#spin-dftb2-o2) (O₂ triplet energy and spin) and
[`dftb2-spin-analytic-forces`](#dftb2-spin-analytic-forces) (O₂ open-shell
forces). The case also runs these sections, whose oracle records are in the
case `REFERENCE.md`: `h2o_dftb2_mio` (H₂O mio-1-1 energy vs DFTB+ 24.1,
−110.9590 eV, tol 0.05 eV), `dftb2-analytic-stress` (H₂O pair-virial stress
vs strain-FD, 1×10⁻⁸ eV/Å³), `dftb2-ewald-gamma` (8-water 7.0 Å cell, Ewald γ
energy, SCC energy and forces vs DFTB+), `dftb2-scc-guess` (ASPC charge
extrapolation: energy unchanged, ≥ 1.25× fewer SCC iterations),
`dftb2-dispersion-d3` and `dftb2-dispersion-d3-periodic` (D3(BJ) cluster and
lattice sums vs s-dftd3 1.2.1), `dftb2-spin-analytic-stress` (O₂ triplet and
NO doublet stress vs strain-FD, `scc_tol = 1e-12`) and `dftb-plus-u` (Dudarev \(E_U\) identity).
The variants under [`dftb2-kpoints`](#dftb2-kpoints),
[`dftb2-periodic-forces`](#dftb2-periodic-forces) and
[`dftb2-periodic-gamma`](#dftb2-periodic-gamma) have no section in the case.

## Common setup

**SK sets.** `DftbSkSet::MioSketch` is DFTB.org **mio-1-1** H/C/N/O — onsite
energies, Hubbard \(U\), the full two-centre SK grids and the spline
repulsive for every ordered pair, transcribed from the released `.skf` by
`tools/gen_dftb_mio_tables.py` into `basis_data/dftb_mio_sk_tables.h` (H–H also
in `basis_data/dftb_mio_hh_sk.h`); despite the name, no pair is a sketch.
`DftbSkSet::ThreeOb` is DFTB.org **3ob-3-1** organic CHNO
(`basis_data/dftb_3ob_tables.h`). There is no runtime `.skf` parser; tables
store length in Bohr and energy in Ha, converted at the use site. Elements
outside H/C/N/O throw.

**Boundary.** `boundary = DftbBoundary::cluster` is an isolated molecule
(DFTB+ `C` geometry: no lattice images in \(H^0\), \(S\), \(\gamma\) or the
repulsive); `lattice` (the default) image-sums \(H^0\)/\(S\)/repulsive inside
the SK cutoff and Ewald-sums \(\gamma\) (DFTB+ `S`/`F` geometry). Every
molecular section below sets `cluster` explicitly; a large box is not a
substitute.

**SCC and energy.** Anderson charge mixing (`mix_beta`, default 0.3) until
\(\max|\Delta q|\) (and \(|\Delta m|\) when open shell) is below `scc_tol`;
exhausting `max_scc` throws. The total energy is assembled as
\(E = \mathrm{Tr}(\rho H^0) + E^{(2)} (+ E_{\mathrm{spin}}) + E_{\mathrm{rep}}\),
which equals the band-sum form
\(\sum_i n_i\varepsilon_i - \tfrac12\sum_{AB}\Delta q_A\gamma_{AB}\Delta q_B + E_{\mathrm{rep}}\)
once the SCC shift is double-count subtracted.

**Analytic forces and stress.** Closed form at the converged SCC density —
Hellmann–Feynman on \(\partial H^0\), Pulay energy-weighted density on
\(\partial S\) plus the SCC (and spin) shift on \(\partial S\), the
fixed-\(\Delta q\) \(\gamma\) gradient, and the repulsive derivative — in one
spin-resolved assembly (`analytic_forces_stress` in
`src/cpu/semiempirical/dftb2.cpp`) that also emits the pair virial. A
converged SCC-DFTB2 density is stationary, so no coupled-perturbed term is
needed and analytic vs re-SCC central FD is exact up to truncation. This is
the DFTB+ force-module boundary (Hourahine et al., *J. Chem. Phys.* **152**,
124101 (2020)).

**Units.** Public surface eV, Å, eV/Å, eV/Å³, Mulliken populations in e.
Masses (H 1.008, C 12.011, O 15.999) are Frame columns only, not in the
Hamiltonian.
1 Ha = 27.211386245988 eV (`kHartreeInEv`, DFTB+ printout / CODATA);
1 Ha/Bohr = 51.42206747467393 eV/Å.

**Build.** SE-only closure (`-DATV_BUILD_SEMIEMPIRICAL=ON
-DATV_BUILD_EVALUATE=ON -DATV_BUILD_REGRESSIONS=ON`, every other domain and
CUDA OFF) in the CPU environment `buildenv-gcc/2026.03-mpich`.

## Oracle

Family oracle for every DFTB row is **DFTB+** (24.1 unless stated) — never
OpenMOPAC, never xtb, and GFN-xTB is not a DFTB3 successor. Pins are offline
transcripts; ctest never launches DFTB+. Offline inputs live under
`tools/tb_oracle/inputs/dftbplus/<case>/` (re-run:
`tools/tb_oracle/run_dftbplus_pin.sh`). Where no DFTB+ number exists for a
geometry, the gate is a formula-level identity (analytic vs re-SCC FD of the
same ATV energy) and the oracle line says `literature`.

## dftb2-dftbplus-energy-pin — SCC-DFTB2 H₂ total energy vs DFTB+ / mio-1-1 {#dftb2-dftbplus-energy-pin}

```text
oracle: dftb+
oracle_version: 24.1
```

### Summary

SCC-DFTB2 on isolated H₂ at \(r(\mathrm{H–H}) = 0.74\) Å is pinned to an
offline **DFTB+ 24.1** total energy obtained with the published **mio-1-1**
`H-H.skf`, on the public Evaluate surface (eV on `frame.meta[kEnergy]`). This
is the absolute-energy hygiene pin for H–H.

### Method

\[
E = E_{\mathrm{band}} + E_{\mathrm{SCC}} + E_{\mathrm{rep}},\quad
E_{\mathrm{band}} = \sum_{i\in\mathrm{occ}} n_i\,\varepsilon_i
  \ (\text{or }\operatorname{Tr}(P H^0)),\quad
E_{\mathrm{SCC}} = \tfrac12\sum_{A,B}\Delta q_A\,\gamma_{AB}(R_{AB})\,\Delta q_B,\quad
E_{\mathrm{rep}} = \sum_{A<B} V_{\mathrm{rep}}^{AB}(R_{AB}).
\]

\(H^0_{μν}\) and \(S_{μν}\) come from the H \(s\) SK radial tables;
\(\gamma_{AB}\) is the DFTB+ Elstner screened kernel with atomic \(U_A\). At
homonuclear H₂ symmetry \(\Delta q_A = 0\), so \(E_{\mathrm{SCC}} = 0\) and the
pin reduces to band + repulsive.

| Piece | Source in ATV | Oracle twin |
| ----- | ------------- | ----------- |
| H–H \(H_{ss}\), \(S_{ss}\) radial grids | `dftb_mio_hh_sk.h` (mio-1-1) | DFTB+ `H-H.skf` |
| H onsite \(E_s\), Hubbard \(U\) | same header | mio-1-1 |
| Repulsive | mio-1-1 spline (static) | DFTB+ spline |

| Setting | Value |
| ------- | ----- |
| System | H₂, \(Z=1\), closed shell, charge 0 |
| \(r(\mathrm{H–H})\) | **0.74 Å** |
| Box | cubic 30 Å, `boundary = cluster` |
| SCC | `scc_tol = 1e-8`, `max_scc = 50`; forces off |

```cpp
cpu::Frame frame = make_h2(0.74);   // box = cpu::Box::cubic(30.0)

cpu::semiempirical::DFTB2 dftb{cpu::semiempirical::DftbSkSet::MioSketch};
dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
dftb.scc_tol = 1e-8;
dftb.max_scc = 50;
dftb.run(frame);

const double E_eV = frame.meta.get(kEnergy);
// gate: |E_eV − (−18.36635052)| ≤ 1e−3 eV
```

### Validation

DFTB+ 24.1, mio-1-1, `SCC = Yes`, `SCCTolerance = 1e-8`,
`MaxAngularMomentum { H = "s" }`, Type2 SK names; SK: dftbparams/mio
`H-H.skf` (operator host; not vendored as binary).

| Quantity | Value | Unit |
| -------- | ----: | ---- |
| Total energy | −0.6749509324 | Ha |
| Total energy (pin) | **−18.36635052** | eV |
| Band energy | −0.6807883427 | Ha |
| Repulsive | +0.0058374104 | Ha |
| SCC energy | 0 (\(\Delta q = 0\)) | Ha |

| Quantity | Atomiverse / pin | DFTB+ 24.1 | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E_{\mathrm{tot}}\) | `kEnergy` | **−18.36635052 eV** | **1×10⁻³ eV** | `kTol_eV = 1.0e-3` |
| Host residual | \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert\) | — | — | ~**3×10⁻⁹ eV** |

The tolerance is intentionally loose against the residual (mHa-scale hygiene
for SK interpolation / unit-conversion drift); the measured residual is near
machine identity for the transcribed tables + DFTB+ "new" SK interpolation +
spline repulsive.

### Performance

Informative only — not asserted in ctest.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----------------------------------------: | ---- | ------ |
| H₂ SCC-DFTB2 Evaluate (forces off) | ~0.15 (DFTB+ 24.1 cold, incl. I/O) | ~0.06 | **≈ 0.4** | ✓ | Cold DFTB+ includes SK/I/O; ATV tables are static |

### Limitations

This pin establishes absolute oracle identity for H–H only, where
\(\Delta q = 0\) and \(\gamma_{AB}\) never enters the energy; polar mio-1-1
identity is the `h2o_dftb2_mio` section, 3ob identity is
[`dftb2-3ob-organic`](#dftb2-3ob-organic).

## dftb2-3ob-organic — SCC-DFTB2 — CH₄ / 3ob-3-1 organic SK {#dftb2-3ob-organic}

```text
oracle: dftb+
oracle_version: 24.1
```

### Summary

SCC-DFTB2 total energy for isolated methane using the DFTB.org **3ob-3-1**
organic H/C/N/O tables, via `DFTB2{DftbSkSet::ThreeOb}`. The section asserts

\[
\bigl|E_{\mathrm{ATV}} - (-87.86856831\,\mathrm{eV})\bigr| \le 0.05\,\mathrm{eV}
\]

against an offline DFTB+ 24.1 pin (pure SCC-DFTB2, **no** ThirdOrder). This is
**SK fidelity under DFTB2** for production 3ob tables — not a DFTB3 claim (3ob
was trained with third-order corrections; the twin is deliberately DFTB2).
`perf_exempt`.

### Method

```text
ParameterSet → SK → H/S → eigen → P → SCC(Hubbard, γ) → repulsive → E
```

**SK set — 3ob-3-1 organic CHNO** (Gaus, Goez, Elstner, *J. Chem. Theory
Comput.* **9**, 338 (2013); DOI
[10.1021/ct300849w](https://doi.org/10.1021/ct300849w)). The in-tree header
is a CC-BY-SA 4.0 transcription of the DFTB.org `.skf`
(`tools/gen_dftb_3ob_tables.py`, `Dftb3obSkPoint` stored as `double` at 17
significant digits).

| Item | Setting |
| ---- | ------- |
| Spin | Closed shell (`multiplicity = 1`), charge 0 |
| SCC | Yes, `scc_tol = 1e-10`, `max_scc = 80` |
| ThirdOrder | **No** (pure DFTB2) |
| SK | 3ob-3-1 (`H-H`, `C-C`, `C-H`, `H-C`) |
| System | CH₄, tetrahedral, C at origin; H at \(\pm s,\pm s,\pm s\) with \(s=1.09/\sqrt{3}\), \(r(\mathrm{C{-}H})=1.09\) Å |
| Cell | cubic \(L = 20\) Å, `boundary = cluster` |
| Forces | off (`run(frame)`) |

```cpp
cpu::Frame frame = make_ch4();   // s = 1.09 / std::sqrt(3.0); box = cpu::Box::cubic(20.0)

cpu::semiempirical::DFTB2 dftb{cpu::semiempirical::DftbSkSet::ThreeOb};
dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
dftb.scc_tol = 1e-10;
dftb.max_scc = 80;
dftb.run(frame);

const double e_eV = frame.meta.get(kEnergy);  // [eV]
```

### Validation

DFTB+ 24.1 total energy −3.2291103259 Ha. Decomposition (documentation):

| Term | Ha | eV |
| ---- | --: | --: |
| Electronic | −3.1685644194 | −86.2210 |
| SCC | 0.0011039499 | 0.0300 |
| Repulsive | −0.0605459065 | −1.6475 |
| **Total** | **−3.2291103259** | **−87.8686** |

| Quantity | Atomiverse / pin | DFTB+ 24.1 | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E_{\mathrm{tot}}\) | **−87.86856831 eV** | **−87.86856831 eV** (−3.2291103259 Ha) | **0.05 eV** | `\|E − pin\| ≤ 0.05` |
| Finite energy | required | — | finite | |

The **50 meV** band covers linear SK interpolation and Elstner \(\gamma\) vs
DFTB+ poly-interp / \(\gamma\) implementation differences; it fails gross
SK-set / geometry / method mistakes. Offline re-pin (human only): DFTB+ 24.1,
`SCC = Yes`, no ThirdOrder, 3ob-3-1 SK, CH₄ \(r_{\mathrm{CH}}=1.09\) Å; record
Total Energy [H] and convert via 27.211386245988.

### Performance

`perf_exempt: true` — energy pin only. DFTB+ wall for this molecule ~0.12 s
(single core, release); ATV expected well under 10× (small dense
eigenproblem).

### Limitations

- **DFTB2 + 3ob SK**, not DFTB3 accuracy for the 3ob training set.
- Energy-only gate. ThreeOb forces and cell stress are closed-form on the
  product path, with no standing ThreeOb regression; ThreeOb is molecular
  only (`k_mesh` beyond Γ throws) and closed shell only.
- Organic CHNO only; other 3ob extensions (S, P, …) are not in-tree.
- Related narrative: [ch17-semiempirical-am1](../ch17-semiempirical-am1.md)
  (NDDO/AM1; different family — OpenMOPAC oracle).

## dftb2-analytic-forces — SCC-DFTB2 analytic forces — H₂O (mio-1-1) {#dftb2-analytic-forces}

```text
oracle: literature
oracle_version: Hourahine et al., J. Chem. Phys. 152, 124101 (2020); Elstner
  et al., Phys. Rev. B 58, 7260 (1998) — closed-form SCC-DFTB2 nuclear gradient
family_oracle: dftb+
```

### Summary

Closed-shell SCC-DFTB2 nuclear forces on off-equilibrium water. After one SCC
solve, forces are the closed-form gradient at the SCC density, written to
`frame.span(kAtom_f*)` in eV/Å — **not** a \(6N\) re-SCC finite difference of
\(E\). No DFTB+ run produced an absolute energy or force for this geometry:
the gate is analytic \(F\) vs re-SCC central FD of the same ATV energy.
`perf_exempt: true`.

### Method

Ionic forces, pairwise accumulation of antisymmetrized bond contributions:

\[
\mathbf{F}_A = -\frac{\partial E}{\partial \mathbf{R}_A}
= \mathbf{F}_A^{H_0} + \mathbf{F}_A^{\mathrm{Pulay}}
+ \mathbf{F}_A^{\mathrm{SCC}\,\gamma} + \mathbf{F}_A^{\mathrm{rep}},
\]

| Term | Contraction (schematic) |
| ---- | ----------------------- |
| Electronic \(H_0\) | \(\mathrm{Tr}\!\bigl(P\,\partial H_0/\partial R\bigr)\) |
| Pulay + shift on \(S\) | \(-\mathrm{Tr}\!\bigl(W\,\partial S/\partial R\bigr) + \mathrm{Tr}\!\bigl(P\,(\tfrac12(V_A+V_B)\,\partial S/\partial R)\bigr)\) with \(V_A=\sum_B\gamma_{AB}\Delta q_B\), \(W=\sum_i n_i\varepsilon_i\,c_ic_i^\top\) |
| SCC \(\gamma\) (fixed \(\Delta q\)) | \(+\Delta q_A\Delta q_B\,(\partial\gamma_{AB}/\partial r)\,\hat r\) with bond \(= R_B - R_A\) |
| Repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) from the mio-1-1 pair spline |

SK block derivatives \(\partial H_0/\partial R\), \(\partial S/\partial R\)
combine the radial derivative of the SK grids with directional factors for
\(s\)/\(p\) blocks (`fill_sk_block_deriv`).

| Knob | Value |
| ---- | ----- |
| System | Isolated H₂O, **off-equilibrium** so \(\|F\|\neq 0\): O (0, 0, 0), H1 (1.05, 0, 0), H2 (−0.26, 1.00, 0) Å |
| Cell | Cubic 20 Å, `boundary = cluster` |
| Method | SCC-DFTB2, closed shell (`multiplicity = 1`, `charge = 0`), `DftbSkSet::MioSketch` |
| SCC | `scc_tol = 1e-10` |
| Forces | a parent force request |
| FD protocol | Central re-SCC energy-only, \(h=10^{-4}\) Å |

```cpp
cpu::Frame frame = make_water();   // geometry above, box = cpu::Box::cubic(20.0)

cpu::semiempirical::DFTB2 dftb{cpu::semiempirical::DftbSkSet::MioSketch};
dftb.boundary = cpu::semiempirical::DftbBoundary::cluster;
dftb.scc_tol = 1e-10;

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add(std::move(dftb));
eval.build();
eval.run(frame);
// Energy: frame.meta.get(kEnergy) [eV]; forces: kAtom_fx / fy / fz [eV/Å]
```

### Validation

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E\) [eV] | finite, \(\|E\| > 10^{-3}\) | hard | ATV twin, not an absolute DFTB+ \(E\) |
| \(\|F\|^2\) [eV²/Å²] | \(> 10^{-8}\) | hard | Off-eq geometry must produce force |
| \(\|\sum_i F_{i,d}\|\) [eV/Å] | \(\le 10^{-6}\) | hard | Translation invariance, each Cartesian |
| \(F_{\mathrm{O},x}\) vs re-SCC FD [eV/Å] | identity | \(1\times10^{-5}\) | measured 2.5e-07 |
| \(F_{\mathrm{H1},y}\) vs re-SCC FD [eV/Å] | identity | \(1\times10^{-5}\) | measured 4.9e-09 |

Measured energy −91.70909683 eV (ATV twin snapshot, not gated). The
analytic path replaces \(6N\) full re-SCC force evaluations with **one**
post-SCC contraction of \(P\), \(W\), \(\partial H_0\), \(\partial S\),
\(\partial\gamma\), and \(\partial E_{\mathrm{rep}}\).

### Limitations

- Not an absolute DFTB+ force pin; the absolute mio-1-1 energy on a
  polar water geometry is the `h2o_dftb2_mio` section.
- Molecular cluster only here; periodic and open-shell analytic forces use
  the same assembly (see [`dftb2-periodic-forces`](#dftb2-periodic-forces),
  [`dftb2-spin-analytic-forces`](#dftb2-spin-analytic-forces)).

## spin-dftb2-o2 — Collinear spin-polarized SCC-DFTB2 — O₂ triplet {#spin-dftb2-o2}

```text
oracle: dftb+
oracle_version: 24.1
family_oracle: dftb+
```

### Summary

Collinear unrestricted SCC-DFTB2 on gas-phase **O₂** in the triplet ground
multiplet \(^3\Sigma_g^-\). `DFTB2` owns the SCC loop, dual \(\alpha/\beta\)
Fock matrices and Mulliken charge/spin populations. Collinear spin constants
\(W\) are **not** in the SK tables: they are explicit `spin_w_ha` input in Ha
(as in DFTB+ HubbardDerivs / SpinConstants — NOT eV), assigned from
`basis_data::kDftbSpinConstantsPbeHa` (`basis_data/dftb_spin_constants.h`,
included by `dftb2.h`). `multiplicity = 3` with empty `spin_w_ha` throws.
Gates: total energy vs DFTB+ 24.1, \(E_{\mathrm{spin}}\), and total Mulliken
spin \(\sum_A m_A \approx 2\).

### Method

- Charge SCC: DFTB+ Elstner \(\gamma_{AB}\) + onsite Hubbard \(U\).
- Spin: `launch()` converts `spin_w_ha` Ha→eV and collapses the shell matrix
  to the highest-occupied atomic orbital (HOAO) diagonal (H \(W_{ss}\), O
  \(W_{pp}\)), matching DFTB+ `ShellResolvedSpin = No`. Published PBE table for
  O \(\{W_{ss},W_{sp},W_{ps},W_{pp}\}=\{-0.035,-0.030,-0.030,-0.028\}\) Ha (DFTB+
  manual appendix; Köhler et al. 2007). Dual \(\alpha/\beta\) generalized
  eigenproblems per SCC cycle.

\[
\varepsilon_A^\sigma = \sum_B \gamma_{AB}\,\Delta q_B + \sigma\, W_A\, m_A,
\qquad \sigma \in \{+1,-1\}\ (\alpha/\beta),
\]

\[
E = \mathrm{Tr}(\rho H^0)+E^{(2)}+E_{\mathrm{spin}}+E_{\mathrm{rep}}
  = \sum_{\sigma i} f_{i\sigma}\,\varepsilon_{i\sigma}
  - \tfrac12\sum_{AB}\gamma_{AB}\,\Delta q_A\Delta q_B
  - \tfrac12\sum_A W_A\, m_A^2
  + E_{\mathrm{rep}}.
\]

**Occupations.** Integer fillings from charge \(q\) and multiplicity:
\(N_\alpha - N_\beta = m-1\), \(N_\alpha + N_\beta = N_{\mathrm{val}} - q\).
Oxygen \(Z_{\mathrm{val}}=6\) → \(N_{\mathrm{val}}=12\); multiplicity 3 →
\(N_\alpha=7\), \(N_\beta=5\), \(\sum_A m_A = 2\).

**Why \(\Delta E_{\mathrm{total}} = \Delta E_{\mathrm{spin}}\) under a change
of \(W\).** O₂ is homonuclear, so \(\Delta q_A\equiv 0\) and the spin shift
\(\sigma W_A m_A\) is one constant \(c\) on both atoms; \(H^\sigma=H^0\pm cS\)
has the same eigenvectors as \(H^0\), so \(\rho^\alpha,\rho^\beta\),
\(\mathrm{Tr}(\rho H^0)\), \(E^{(2)}\) and \(E_{\mathrm{rep}}\) are unchanged and
only \(E_{\mathrm{spin}}=W_{\mathrm{pp}}(\mathrm{O},\mathrm{PBE})\) moves
(\(m_A=1\)). Against a \(W=-1\) eV reference energy −177.2425823576 eV, the
published \(W=-0.028\) Ha \(=-0.761918814888\) eV shifts the total by exactly
\(+0.238081185112\) eV; the section gates that identity at 1e-4 eV.

| Observable | Definition | Unit | Access |
| ---------- | ---------- | ---- | ------ |
| \(E\) | total energy | eV | `frame.meta.get(kEnergy)` / `last_energy()` |
| \(\Delta q_A\) | Mulliken charge fluctuation | e | `last_mulliken_dq()` |
| \(m_A\) | Mulliken spin \(N_{\alpha,A}-N_{\beta,A}\) | e | `last_mulliken_spin()` |

### Implementation

```cpp
cpu::Frame frame(2);   // O (0,0,0), O (1.21,0,0) Å; box = cpu::Box::cubic(30.0)

cpu::semiempirical::DFTB2 dftb{cpu::semiempirical::DftbSkSet::MioSketch,
                               /*charge=*/0, /*multiplicity=*/3};
dftb.boundary = cpu::semiempirical::DftbBoundary::cluster; // DFTB+ `C` geometry
dftb.spin_w_ha = cpu::semiempirical::basis_data::kDftbSpinConstantsPbeHa;
dftb.scc_tol = 1e-10;
dftb.max_scc = 80;
// default shell_resolved_spin = false (DFTB+ ShellResolvedSpin = No)

dftb.run(frame);

const double e = frame.meta.get(kEnergy);          // [eV]
const auto& m = dftb.last_mulliken_spin();         // m_A [e]
const auto& dq = dftb.last_mulliken_dq();          // Δq_A [e]
```

`add_ref(lvalue)` keeps the component with the caller after `run` (Mulliken
accessors); `add(rvalue)` transfers ownership into the plan.

### Validation

DFTB+ 24.1 twin: mio-1-1 `O-O.skf`, `SpinConstants { ShellResolvedSpin = No;
O = { -0.028 } }`, `UnpairedElectrons = 2`, cluster geometry; O₂ at
\(R(\mathrm{O{-}O}) = 1.21\) Å (near experimental), charge 0, multiplicity 3,
`scc_tol = 1e-10`, `max_scc = 80`.

| Quantity | DFTB+ | Converted |
| -------- | ----- | --------- |
| Total energy | −6.5047954387 H | **−177.004501133607 eV** |
| Energy SPIN | −0.0280000000 H | **−0.761918814888 eV** |
| Energy SCC | 0 (Δq = 0) | 0 |
| Gross atomic spins | 1.000, 1.000 | e |

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E\) [eV] | **−177.004501172488** (measured −177.004501172479) | DFTB+ 24.1 **−177.004501133607** | **\(10^{-4}\)** | \(\lvert\Delta\rvert=3.9\times10^{-8}\) eV |
| \(E_{\mathrm{spin}}\) [eV] | **−0.761918814888** | DFTB+ Energy SPIN \(=-0.028\) Ha | **\(10^{-9}\)** | \(W_{pp}(\mathrm{O},\mathrm{PBE})\); \(m_A=1\) |
| \(E-E_{\mathrm{ref}}\) [eV] | +0.238081185121 | +0.238081185112 | \(10^{-4}\) | \(W=-1\) eV reference −177.2425823576 eV |
| \(\sum_A m_A\) [e] | **2.0** (observed \([1.0, 1.0]\)) | 2 (triplet occupations) | **0.25** | Mulliken spin sum |
| Per-atom \(\lvert m_A\rvert\) | \(> 0.3\) each | both atoms spin-polarized | — | asymmetry gate \(\lvert m_0-m_1\rvert < 0.2\) |
| \(\Delta q_A\) | observed \([0, 0]\) | homonuclear | finite | not hard-pinned beyond finiteness |

### Performance

`perf_exempt: true` — DFTB+ 24.1 wall is a cold process + I/O (~0.3 s); ATV
Evaluate+DFTB2 is ~0.02 s. Not a fair kernel-vs-kernel wall.

### Limitations

- **MioSketch only for spin.** ThreeOb open shell throws.
- **`spin_w_ha` is required.** There is no silent \(W\) default, and \(W\)
  does not live in the SK tables.
- **HOAO collapse only.** `shell_resolved_spin = true` throws.
- **No noncollinear / SOC.**
- Multiplet reference: Herzberg, *Molecular Spectra and Molecular Structure*
  I (O₂ \(^3\Sigma_g^-\) ground multiplet).

## dftb2-spin-analytic-forces — Collinear spin-DFTB2 analytic forces — O₂ {#dftb2-spin-analytic-forces}

```text
oracle: dftb+
oracle_version: 24.1
```

### Summary

Closed-form nuclear forces for collinear unrestricted SCC-DFTB2 on triplet
O₂ at a stretched bond: dual \(\alpha/\beta\) Hellmann–Feynman + Pulay with
spin-dependent onsite shifts on \(\partial S\) + fixed-\(\Delta q\) SCC
\(\gamma\) + repulsive. \(W_A\) is geometry-independent and the fixed-\(m\)
spin energy contributes no extra nuclear term. Same public surface as
[`spin-dftb2-o2`](#spin-dftb2-o2) with a parent force request.

### Settings and gates

O₂ at (0, 0, 0) and (1.30, 0, 0) Å, cubic 30 Å box, `boundary = cluster`,
`multiplicity = 3`, `spin_w_ha = kDftbSpinConstantsPbeHa`,
`scc_tol = 1e-10`, `max_scc = 80`.

| Gate | Tol |
| ---- | --- |
| \(E\) finite, \(\lvert E\rvert > 10^{-3}\) eV | hard |
| \(\lvert F\rvert^2 > 10^{-8}\) at the off-eq geometry | hard |
| \(\lvert\sum_i F_{i,d}\rvert\) per Cartesian | \(10^{-6}\) eV/Å |
| \(F_{0,x}\), \(F_{1,x}\) vs re-SCC central FD (\(h=10^{-4}\) Å) | 0.05 eV/Å |

Measured: \(E = -176.77425642\) eV, \(F_{0,x} = 4.91147291\) eV/Å
(re-SCC FD 4.91147232, \(\lvert d\rvert = 5.9\times10^{-7}\) eV/Å). Recorded
DFTB+ 24.1 comparison (archived `dftb.out.txt`, not gated): total energy
−6.4963341184 H = −176.774256878772 eV, \(F_{0,x} = +0.095512870310\) Ha/Bohr
= +4.9114692618 eV/Å; \(\lvert\Delta E\rvert = 4.6\times10^{-7}\) eV,
\(\lvert\Delta F_{0,x}\rvert = 3.6\times10^{-6}\) eV/Å. Homonuclear O₂ has
\(\Delta q\equiv 0\), so this does not exercise the charge-dependent
\(\gamma\) gradient; the NO doublet in `dftb2-spin-analytic-stress` does.
`perf_exempt: true`.

## dftb2-kpoints — Multi-k SCC-DFTB2 — 1-D carbon chain (4×1×1 Monkhorst–Pack) {#dftb2-kpoints}

*No section of the case runs this variant.*

Roadmap: multi-k periodic DFTB2 ◐ (reachable; no external pin).

### Method

Restricted closed-shell SCC-DFTB2 with a Monkhorst–Pack k-mesh beyond Γ
(`k_mesh`, `k_time_reversal`; requires `boundary = lattice`). With k-point
weights \(w_k\) (\(\sum_k w_k = 1\)), complex Bloch matrices \(H^{0k}\),
\(S^k\) and density \(P^k\),

\[
E = \sum_k w_k\,\mathrm{Re}\,\mathrm{Tr}\bigl(P^k H^{0k}\bigr)
  + E^{(2)}[\{\Delta q\}] + E_{\mathrm{rep}},
\qquad
E^{(2)} = \tfrac12\sum_{AB}\Delta q_A\,\gamma_{AB}\,\Delta q_B,
\]

with \(\Delta q_A\) the **k-weighted** Mulliken fluctuation; the SCC shift
\(\varepsilon_A = \sum_B \gamma_{AB}\,\Delta q_B\) is real, k-independent, and
added onsite at every \(\mathbf{k}\). For fractional wavevector
\(\boldsymbol{\kappa}\) and \(\mathbf{T}=n_0\mathbf{a}_1+n_1\mathbf{a}_2+n_2\mathbf{a}_3\),

\[
H^{0k}_{\mu\nu}
  = \sum_{\mathbf{T}} e^{2\pi i\,\boldsymbol{\kappa}\cdot\mathbf{n}}\,
    H^{0,\mathrm{SK}}_{\mu\nu}\big(|\mathbf{R}_\nu+\mathbf{T}-\mathbf{R}_\mu|\big),
\quad
S^{k}_{\mu\nu}
  = \sum_{\mathbf{T}} e^{2\pi i\,\boldsymbol{\kappa}\cdot\mathbf{n}}\,
    S^{\mathrm{SK}}_{\mu\nu}\big(|\mathbf{R}_\nu+\mathbf{T}-\mathbf{R}_\mu|\big),
\]

with the Γ-path shell enumeration (`monkhorst_pack_mesh`,
`fill_sk_block_phased`). Mesh (CP2K / DFTB+ convention, `k_mesh = {N_0, N_1, N_2}`):
\(\kappa_d = (2r - N_d - 1)/(2 N_d)\), \(r = 1,\ldots,N_d\), uniform weight
\(1/(N_0 N_1 N_2)\) before folding. `k_time_reversal = true` (default) folds
\(\pm\mathbf{k}\) pairs with doubled weight — exact for real-Hamiltonian
closed-shell densities. Default `k_mesh = {1,1,1}` keeps the real Γ solver.
DFTB+ twin: `KPointsAndWeights = SupercellFolding { 4 0 0  0 1 0  0 0 1  0.0 0.0 0.0 }`.

Multi-k MioSketch also carries analytic nuclear forces and pair-virial cell
stress (Bloch-sum gradient contracting the image-resolved densities) and
collinear open shell; ThreeOb throws on `k_mesh ≠ {1,1,1}`.

### Recorded check

Two C atoms at (0, 0, 0) and (1.40, 0, 0) Å, orthorhombic cell
\(L_x=2.8\) Å, \(L_y=L_z=12.0\) Å (masses 12.011), `k_mesh = {4, 1, 1}`,
`scc_tol` \(1\times10^{-10}\), `max_scc = 50`, energy only.

```cpp
cpu::semiempirical::DFTB2 dftb{cpu::semiempirical::DftbSkSet::MioSketch};
dftb.scc_tol = 1e-10;
dftb.k_mesh = {4, 1, 1};        // Monkhorst–Pack along chain
dftb.k_time_reversal = true;
// frame.box = cpu::Box::orthorhombic(2.8, 12.0, 12.0); boundary = lattice (default)
```

| Quantity | Recorded value | Tol | Notes |
| -------- | -------------: | --- | ----- |
| Total energy \(E\) (4×1×1) [eV] | **−98.9944775364** | ±0.05 eV | ATV Hamiltonian twin (oracle `literature`, family dftb+) |
| \(\lvert E - E_\Gamma\rvert\) [eV] | \(> 0.01\) | — | multi-k must separate from Γ |
| \(E_\Gamma\) (1×1×1 control) [eV] | **−113.5976598145** | ±0.05 eV | equals [dftb2-periodic-gamma](#dftb2-periodic-gamma) |

These values come from the removed end-to-end case and were recorded on
exponential-sketch C–C tables; they are not reproduced on the current mio-1-1
C–C tables and are not a DFTB+ digit match. `perf_exempt: true` (no DFTB+
wall).

## dftb2-periodic-forces — Γ-PBC SCC-DFTB2 analytic forces — off-eq 1-D C chain {#dftb2-periodic-forces}

*No section of the case runs this variant.*

Roadmap: Γ-point periodic DFTB2 ✓ with analytic F/σ on MioSketch.

### Method

Closed-form nuclear forces for Γ-point periodic SCC-DFTB2 under the **same
lattice-image walk** used for the energy (\(H^0\), \(S\), \(\gamma\),
repulsive), so \(F=-\nabla E\) of the lattice-summed energy rather than a
min-image contraction. Matrices walk lattice translations with \(|\mathbf{T}|\)
up to the maximum pair cutoff (element `r_cut`; `enumerate_lattice` /
`lattice_images_for`):

\[
H^0_{\mu\nu} = \sum_{\mathbf{T}} H^{0,\mathrm{SK}}_{\mu\nu}\big(|\mathbf{R}_\nu+\mathbf{T}-\mathbf{R}_\mu|\big),
\quad
S_{\mu\nu} = \sum_{\mathbf{T}} S^{\mathrm{SK}}_{\mu\nu}\big(|\mathbf{R}_\nu+\mathbf{T}-\mathbf{R}_\mu|\big).
\]

The force terms are those of [`dftb2-analytic-forces`](#dftb2-analytic-forces),
accumulated over every lattice image of \(B\) within \(r_\mathrm{cut}\) (with
the symmetric \(A\leftarrow B+\mathbf{T}\) bookkeeping, matching `build_h0_s` /
`gamma_pair` / `repulsive_energy`). Self-images (\(A=A+\mathbf{T}\),
\(\mathbf{T}\ne\mathbf{0}\)) carry **zero** nuclear force (\(|\mathbf{T}|\) is
geometry-independent) but enter the virial.

### Recorded check

Two C atoms at (0, 0, 0) and (1.20, 0, 0) Å, orthorhombic \(L_x=2.8\),
\(L_y=L_z=12.0\) Å, so left/right image contacts sit at 1.20 Å and 1.60 Å
(plus farther shells inside \(r_\mathrm{cut}\approx 4.5\) Å) and
\(F_x\neq 0\); a parent force request, `scc_tol = 1e-10`, `max_scc = 50`,
`d3 = false`. Oracle `literature` (formula level: Elstner 1998; Hourahine et
al. 2020), family dftb+.

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E\) [eV] | finite, \(\|E\| > 10^{-3}\) | hard | ATV twin |
| \(\|F\|^2\) [eV²/Å²] | \(> 10^{-8}\) | hard | off-eq geometry |
| \(\|\sum_i F_{i,d}\|\) [eV/Å] | \(\le 10^{-6}\) | hard | each Cartesian |
| \(F_{0,x}\), \(F_{1,x}\) vs re-SCC FD [eV/Å] | identity | **\(1\times 10^{-5}\)** | \(h=10^{-4}\) Å, lattice-summed \(E\) |
| \(\lvert E_{\mathrm{PBC}}-E_{\mathrm{mol}}\rvert\) [eV] | \(> 0.05\) | hard | same dimer in a 20 Å cubic box must separate |

The equilibrium \(r=1.4\) chain of [dftb2-periodic-gamma](#dftb2-periodic-gamma)
has vanishing force by symmetry and is not used. The same geometry with
`DFTB3` is the `dftb3-periodic-forces` section of
[DFTB3](semiempirical-dftb3.md#dftb3-water-mio). `perf_exempt: true`.

## dftb2-periodic-gamma — Γ-point periodic SCC-DFTB2 — 1-D carbon chain {#dftb2-periodic-gamma}

*No section of the case runs this variant.*

Γ-point periodic DFTB2 energy is pinned against DFTB+ by the
`dftb2-ewald-gamma` section.

### Method

Γ-point SCC-DFTB2 with `boundary = lattice`: lattice-summed two-centre
\(H^0\) and \(S\) and pair repulsive within the static SK cutoffs (element
`r_cut`), and the Ewald-summed \(\gamma\). When all \(L_d \gg r_\mathrm{cut}\)
only \(\mathbf{T}=\mathbf{0}\) contributes to \(H^0\)/\(S\). On this
homonuclear chain \(\Delta q\equiv 0\), so \(\gamma\) does not enter the
energy.

### Recorded check

Two C atoms at (0, 0, 0) and (1.40, 0, 0) Å, orthorhombic \(L_x=2.8\) Å,
\(L_y=L_z=12.0\) Å: each carbon has nearest-neighbour contacts along \(\pm x\)
from the partner **and** its periodic images. `scc_tol = 1e-10`,
`max_scc = 50`, forces off, charge 0, multiplicity 1. Oracle `literature`
(ATV Hamiltonian twin), family dftb+.

| Quantity | Recorded value | Tol | Notes |
| -------- | -------------: | --- | ----- |
| Total energy \(E\) [eV] | **−113.5976598145** | ±0.05 eV | removed end-to-end case, exponential-sketch C–C tables |
| \(\lvert E-E_{\mathrm{mol}}\rvert\) [eV] | \(> 0.05\) | — | same dimer in a 20 Å cubic box; lattice images must matter |

Not a DFTB+ digit match. `perf_exempt: true`.

## References

- M. Elstner, D. Porezag, G. Jungnickel, J. Elsner, M. Haugk, Th. Frauenheim,
  S. Suhai, G. Seifert, *Phys. Rev. B* **58**, 7260 (1998),
  DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)
- B. Hourahine et al., *J. Chem. Phys.* **152**, 124101 (2020),
  DOI 10.1063/1.5143190 (DFTB+ force module)
- M. Gaus, A. Goez, M. Elstner, *J. Chem. Theory Comput.* **9**, 338 (2013),
  DOI [10.1021/ct300849w](https://doi.org/10.1021/ct300849w) (3ob-3-1)
- C. Köhler, Th. Frauenheim, B. Hourahine, G. Seifert, M. Sternberg,
  *J. Phys. Chem. A* **111**, 5622 (2007),
  DOI [10.1021/jp068802p](https://doi.org/10.1021/jp068802p)
  (collinear spin-DFTB / DFTB+ spin constants)
- DFTB.org / dftbparams mio-1-1 and 3ob-3-1; DFTB+ 24.1
- Roadmap: [`docs/roadmap/semiempirical.md`](../../roadmap/semiempirical.md) §2
