# rm1-water-mopac-energy — RM1 heat of formation of water, pinned to OpenMOPAC

```text
oracle: mopac
oracle_version: OpenMOPAC PARAM survey tables (github.com/openmopac/param @ 9859905,
                Apache-2.0); cross-calibrated against MOPAC2016 Version 21.298M
                reference output tests/keywords/RM1.out
family_oracle: mopac
```

NDDO family lock: **OpenMOPAC only**. No DFTB+ number appears anywhere in this
case. No `mopac` binary exists on the Atomiverse build host and ctest never
shells out to one — every value below is hard-coded from a published source.

## Method

**RM1** (Recife Model 1) — G. B. Rocha, R. O. Freire, A. M. Simas and
J. J. P. Stewart, *RM1: A reparameterization of AM1 for H, C, N, O, P, S, F,
Cl, Br and I*, **J. Comput. Chem. 27, 1101–1111 (2006)**,
DOI [10.1002/jcc.20425](https://doi.org/10.1002/jcc.20425). Same functional
form and parameter count as AM1; different parameter values.

Public API — concrete method, parameters are members:

```cpp
cpu::semiempirical::RM1 rm1;
rm1.charge = 0;
rm1.multiplicity = 1;
rm1.density_tol = 1.0e-10;
rm1.build(frame);
const cpu::semiempirical::RM1Result& r = rm1.run(frame);
r.heat_of_formation_kcal_mol;   // the pinned quantity
r.total_energy_ev;
```

MOPAC's heat of formation is

    ΔHf = (E_elec + E_core − Σ_A EISOL(A)) · (eV→kcal/mol) + Σ_A ΔHf_exp(A)

with `EISOL` the free-atom NDDO energy that MOPAC **derives** in
`src/models/calpar.F90` from the active parameter set (see *Finding*, below).

## Geometry convention — why this case relaxes

A MOPAC ΔHf belongs to the **method's own stationary geometry**, not to an
arbitrary one. OpenMOPAC states this for its whole reference-data set
([`data/README.md`](https://github.com/openmopac/mopac/blob/main/data/README.md)):

> Because most of the reference data is heats of formation and most geometries
> are not fully constrained by experiment, the reference geometries are relaxed
> to the local minimum of the semiempirical model being trained.

So the published −57.86 kcal/mol **cannot** be pinned at the experimental
geometry, and this is not a technicality: at r(OH) = 0.957 Å / ∠HOH = 104.500°
RM1 gives **−57.83875** kcal/mol, **0.0213** kcal/mol away — it *fails* this
case's ±0.02 gate. The case therefore relaxes water's two
symmetry-unique internal coordinates (r_OH under C2v, ∠HOH) on the RM1 surface
with a pattern search over `RM1::run` and pins **both** the relaxed geometry and
the ΔHf at it. Start geometry (Herzberg 1966, MOPAC tag `GR=HERZ1966`, the same
start OpenMOPAC's own `h2or.mop` deck uses):

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O | 0.000000 | 0.000000 | 0.0 |
| H | 0.957000 | 0.000000 | 0.0 |
| H | −0.239614 | 0.926517 | 0.0 |

r(OH) = 0.957 Å, ∠HOH = 104.500°, charge 0, singlet, 20 Å cubic box
(isolation / unwrap only — RM1 is molecular). Search terminates at Δr < 10⁻⁶ Å
and Δθ < 10⁻⁴ deg.

## Pin

| Quantity | Published | Source | ATV | Tol |
|----------|-----------|--------|-----|-----|
| ΔHf(RM1, relaxed) | **−57.86** kcal/mol | OpenMOPAC PARAM survey `RM1.heats` | **−57.8553** | ±0.02 kcal/mol |
| ΔHf(RM1, relaxed) | **−242.04** kJ/mol (= −57.8489 kcal/mol) | Scientia Plena 12(6), 069903 (2016), Tabela 1 | **−57.8553** | ±0.02 kcal/mol |
| r(OH) at the RM1 minimum | **0.956** Å | OpenMOPAC PARAM survey `RM1.geos` | **0.95637** | ±0.002 Å |
| ∠HOH at the RM1 minimum | **103.610°** | OpenMOPAC PARAM survey `RM1.geos` | **103.599** | ±0.10° |

Reported but not gated: E(NDDO) = −345.571986 eV (electronic + core; MOPAC
does not print a comparable absolute in its default output).

### Sources, verbatim

**1 — OpenMOPAC PARAM survey (primary).** `Analysis/Survey organic/RM1.heats`
line 546 in [github.com/openmopac/param @ `9859905`](https://github.com/openmopac/param/blob/9859905d8bb88ee219e8900dc1129ef558d77193/Analysis/Survey%20organic/RM1.heats#L546)
(Apache-2.0; J. J. P. Stewart is an RM1 co-author and PARAM is the environment
the PMx/RM1 surveys are produced in — the sibling `RM1.dat` is literally
`SURVEY  RM1`):

```text
 Formula        Molecule                              Heat of Formation   Diff. Ref.
                                                       Exp.     Calc.
 H2O           Water                                 -57.80    -57.86    -0.06     5    HO          water
```

and `Analysis/Survey organic/RM1.geos` lines 165–166 (experimental column
first, RM1-calculated second):

```text
 H2O           Water          O-H       0.957   0.956     2 -0.001    1  8  0  0  h2or
                              H-O-H   104.500 103.610       -0.890    1  8  1  0  h2or
```

**2 — independent peer-reviewed publication.** L. S. Martins and A. H. Lima,
*Estudo da entalpia de combustão de hidrocarbonetos por métodos semiempíricos*,
**Scientia Plena 12(6), 069903 (2016)**, DOI
[10.14808/sci.plena.2016.069903](https://doi.org/10.14808/sci.plena.2016.069903).
Tabela 1, caption "Valores em **kJ/mol**", MOPAC2012 with
`RM1 OPT GNORM=0.01 PRECISE XYZ CHARGE=0` (EF optimiser — a relaxed geometry):

```text
  AM1        -36,73             -113,26                -334,13            -247,90
  RM1        -58,07             -86,60                 -341,20            -242,04
  EXP        -74,80                 0                  -393,50            -241,75
```

H₂O column: RM1 −242.04 kJ/mol = **−57.8489 kcal/mol** (÷4.184 exactly).
Two controls in the same table that ATV also reproduces: AM1 water −247.90
kJ/mol = −59.2495 kcal/mol (ATV AM1: −59.2507) and RM1 CO₂ −341.20 kJ/mol =
−81.5487 kcal/mol (ATV RM1: −81.55435).

**3 — calibration against a distributed MOPAC output.** OpenMOPAC ships an RM1
reference output for CO₂ at
[`tests/keywords/RM1.out`](https://github.com/openmopac/mopac/blob/main/tests/keywords/RM1.out)
(MOPAC2016 Version 21.298M):

```text
CYCLE:  1 ... HEAT: -81.09054                       (input geometry, d(CO) = 1.16 Å)
FINAL HEAT OF FORMATION = -81.55398 KCAL/MOL
  C 0.000000000   O ±1.173109199 Å   (EF-optimised, GRADIENT NORM 0.58121)
```

Atomiverse RM1 at those two exact geometries: **−81.09085** and **−81.55429**
kcal/mol — a residual of **3.1 × 10⁻⁴ kcal/mol** at *both*, i.e. a pure
constant, not a geometry-dependent error. That is the direct evidence that the
RM1 parameter set, the free-atom energies and the ΔHf assembly agree with
OpenMOPAC to the precision MOPAC prints.

### Tolerance argument — why ±0.02 kcal/mol

- The primary source quotes 2 decimals, so it carries ±0.005 kcal/mol of
  rounding; the kJ/mol source likewise (±0.005 kJ/mol = ±0.0012 kcal/mol).
- MOPAC version drift is ~4 × 10⁻⁴ kcal/mol (MOPAC2016 21.298M vs 23.2.5 on
  CO₂), and ATV's own residual against a distributed MOPAC output is
  3.1 × 10⁻⁴ kcal/mol.
- ±0.02 is ~3× the largest observed residual (0.0064 against the kJ/mol source)
  and covers both roundings with margin.
- **Upper bound, and the reason it is not looser:** |RM1 − experiment| = 0.06
  kcal/mol. A tolerance ≥ 0.06 would be satisfied by *any* code that merely
  reproduced the experimental ΔHf, and this would stop being a pin on RM1. The
  case asserts `|published RM1 − experiment| > tol` in code so that this
  argument cannot silently rot.
- For scale: the free-atom-energy defect this case was written to catch (below)
  moved ΔHf by **19 kcal/mol**, ~1000× the tolerance.

## Finding (iron law) — RM1 free-atom energies and quadrupole additives were wrong

Fixed on this branch, in `src/cpu/semiempirical/rm1_parameters.cpp`.

`eisol`, `dd`, `qq`, `am`, `ad`, `aq` are **not** published RM1 parameters:
MOPAC derives all six in `src/models/calpar.F90` from the method's own
parameter row, at `ev = 27.211386245988` (2018 CODATA, `conref_C.F90`
`fpcref(1,4)`). The shipped table instead built `eisol` as *"the AM1 free-atom
energy shifted by the one-electron parameter change (uss/upp)"*, leaving the
two-electron part at AM1 values even though RM1 rewrites `gss/gsp/gpp/gp2/hsp`:

| Z | eisol shipped | eisol per `calpar` | error [eV] | ΔHf error/atom [kcal/mol] |
|---|---|---|---|---|
| 6 C | −119.7956994 | −117.8673442 | +1.928 | −44.5 |
| 7 N | −202.8194020 | −205.0876425 | −2.268 | +52.3 |
| 8 O | −312.8526806 | −312.0403541 | +0.812 | −18.7 |
| 15 P | −125.0659220 | −123.1797789 | +1.886 | −43.5 |
| 16 S | −179.9286042 | −185.3861381 | −5.458 | +125.9 |

H was right by construction (`ios=1`, `iop=0` ⇒ no two-electron term). Water's
ΔHf was **−38.82** kcal/mol, 19 off. Two further defects in the same rows:
`aq` did not invert to RM1's own `hpp = (gpp−gp2)/2` (ratio 0.79…0.95, i.e. up
to 6% error in the quadrupole additive), and `am`/`ad` used `ev = 27.21`
(MOPAC's `OLDCON` value) where the AM1 table and `nddo_two_electron.cpp` use
27.211386245988. Separately, `kAm1EvToKcalMol` was the legacy 23.061
(`fpcref(2,9)`) rather than MOPAC's default 23.060547830619029 — that constant
alone was the entire 0.0076 kcal/mol residual against the CO₂ reference output.

The AM1 table satisfies the `calpar` identity exactly for all 42 of its rows,
which is how the RM1 rows were isolated. Gated from now on by
`tests/cpu/semiempirical/test_nddo_calpar_derived.cpp`, which re-derives all
six columns for **every** AM1 and RM1 row (RED on the pre-fix table, GREEN
after) and additionally pins that RM1's oxygen `eisol` is not AM1's shifted
value.

## Offline oracle twin

`tools/tb_oracle/inputs/mopac/rm1-water/` — `rm1-water-opt.mop`
(`RM1 PRECISE GNORM=0.001 XYZ`, full relaxation) and `rm1-water-1scf.mop`
(`RM1 1SCF PRECISE XYZ` at the ATV stationary geometry). Never executed by
ctest; present so the pin can be regenerated on a host that has OpenMOPAC.

## Gaps

- **No MOPAC-distributed reference *output* for RM1 water exists.** The primary
  pin is the PARAM survey **table**, not a `.out` transcript. The transcript-level
  calibration is done on CO₂ (source 3) instead. Re-pin from a real
  `rm1-water-opt.mop` run when a host with OpenMOPAC is available.
- **The RM1 paper's Supporting Information is unobtainable.** J. Comput. Chem.
  is closed access with no OA location (checked OpenAlex, Unpaywall, Semantic
  Scholar; Wiley returns 403 on both the article and `downloadSupplement`), so
  the per-molecule table in the original publication could not be read. The
  paper is cited for the *method*, not for the number.
- **Water is a mild discriminator on its own.** RM1 lands 0.06 kcal/mol from
  experiment here, so the ±0.02 window is what makes the case method-specific.
  A molecule where RM1 deviates strongly (CO₂: RM1 −81.55 vs experiment −94.05;
  CO: RM1 −13.87 vs experiment −26.42) would be a sharper NDDO gate and is the
  natural next slug.
- **RM1 coverage is H, C, N, O, P, S.** F, Cl, Br, I from the published RM1
  parameter set are not transcribed; those rows must be added with `calpar`-derived
  columns, not hand-shifted ones.
- **RM1 forces are still the fixed-density finite-difference path** (shared
  `NDDOSolver::compute_forces`); no closed-form RM1 gradient is pinned here.

## Perf

```text
perf_exempt: true
perf_exempt_reason: no offline OpenMOPAC binary on host; fair wall deferred
```

Wall: ~0.03 s at 1 thread.
