# OpenMOPAC offline twin — RM1 / water / heat of formation

Twin for `regressions/semiempirical/rm1-water-mopac-energy`.

> **Offline only. ctest never executes these decks.** No `mopac` binary exists
> on the Atomiverse build host; every number the regression checks is
> hard-coded from `REFERENCE.md`. These files exist so the pin can be
> regenerated on a host that does have OpenMOPAC, without guessing keywords or
> geometry.

- **family:** NDDO → **OpenMOPAC only** (never DFTB+ for an NDDO pin)
- **method keyword:** `RM1` — Rocha, Freire, Simas & Stewart,
  *J. Comput. Chem.* **27**, 1101 (2006)

## Decks

| File | Keywords | Purpose |
| ---- | -------- | ------- |
| `rm1-water-opt.mop` | `RM1 PRECISE GNORM=0.001 XYZ` | Full relaxation on the RM1 surface. `FINAL HEAT OF FORMATION` here is the quantity the regression pins — a MOPAC ΔHf belongs to the **method's own** stationary geometry (OpenMOPAC `data/README.md`: *"the reference geometries are relaxed to the local minimum of the semiempirical model being trained"*). |
| `rm1-water-1scf.mop` | `RM1 1SCF PRECISE XYZ` | Single point at the ATV RM1 stationary geometry (r(OH) = 0.95637 Å, ∠HOH = 103.599°), for a point-for-point comparison that removes any optimizer difference. |

Both start from the experimental geometry r(OH) = 0.957 Å, ∠HOH = 104.5°
(Herzberg 1966, MOPAC reference tag `GR=HERZ1966`).

## Expected

| Quantity | Published RM1 | ATV |
| -------- | ------------- | --- |
| ΔHf | **−57.86 kcal/mol** — OpenMOPAC PARAM survey [`Analysis/Survey organic/RM1.heats` L546](https://github.com/openmopac/param/blob/9859905d8bb88ee219e8900dc1129ef558d77193/Analysis/Survey%20organic/RM1.heats#L546) | **−57.8553 kcal/mol** |
| ΔHf | **−242.04 kJ/mol** — Scientia Plena 12(6), 069903 (2016), Tabela 1 | −57.8553 kcal/mol (= −241.98 kJ/mol) |
| r(OH) | **0.956 Å** — PARAM survey `RM1.geos` L165 | **0.95637 Å** |
| ∠HOH | **103.610°** — PARAM survey `RM1.geos` L166 | **103.599°** |
| E(NDDO) | — (MOPAC does not print a comparable absolute) | −345.571986 eV |

Experimental context: ΔHf(H₂O) = −57.800 kcal/mol, JANAF Third Edition 1986
(OpenMOPAC `data/PM6-ORG/Core/water.mop`, `H=-57.8000 HR=JANAF86`).

## Calibration of the ATV RM1 ΔHf path

There is no MOPAC-distributed RM1 output for water, but there **is** one for
CO₂ — `tests/keywords/RM1.mop` / `.out` in `openmopac/mopac` (Apache-2.0),
produced by MOPAC2016 Version 21.298M:

```text
RM1
comment1
comment2
C  0.00 0.0 0.0
O  1.16 0.0 0.0
O -1.16 0.0 0.0
```

```text
CYCLE:  1 ... HEAT: -81.09054            (input geometry, d(CO) = 1.16 Å)
FINAL HEAT OF FORMATION = -81.55398 KCAL/MOL
   C  0.000000000  O  ±1.173109199  (Å, EF-optimised, GRADIENT NORM 0.58121)
```

Atomiverse RM1 at those two geometries: **−81.09085** and **−81.55429**
kcal/mol — a residual of **3.1 × 10⁻⁴ kcal/mol** at both, i.e. the RM1
parameter set, the free-atom energies and the ΔHf assembly agree with
OpenMOPAC to the precision MOPAC prints. That is the evidence that the water
number above carries MOPAC's accuracy even though no MOPAC water output exists
offline.

## Temps

Scratch only: `/tmp/mopac_*`, `/tmp/tb_oracle_*`. Never commit MOPAC binaries
or run them from ctest.
