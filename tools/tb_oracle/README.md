# TB / SE offline oracle pins

Family-locked **offline** references for the Atomiverse semiempirical /
tight-binding loop (`cpu/semiempirical`). **Never** launch OpenMOPAC, DFTB+,
or xtb from ctest — pin numbers and input twins live here; regressions hard-code
values from `REFERENCE.md`.

## Family matrix (binding)

| Family | Primary oracle | Forbidden as energy pin |
| ------ | -------------- | ----------------------- |
| NDDO (MNDO / AM1 / PM3 / RM1 / PM6 / PM7) | **OpenMOPAC** | DFTB+, xtb |
| DFTB0 / DFTB2 / DFTB3 / spin / PBC / TD-DFTB | **DFTB+** | OpenMOPAC, xtb |
| GFN-xTB (GFN0/1/2, IPEA, …) | **xtb** | OpenMOPAC, DFTB+; **not** a DFTB3 successor |
| Unified NDDO+TB API shape (no energy) | SCINE Sparrow | — (architecture only) |
| Evaluate / MD embed patterns | CP2K | — (architecture only) |

`REFERENCE.md` for every SE regression must state:

```text
oracle: mopac|dftb+|xtb|literature
oracle_version: <package version or formula citation>
```

## DFTB+ install (Arrhenius / this clone, 2026-07-29)

Prebuilt **DFTB+ 24.1** + **mio-1-1** / **3ob-3-1** live outside git
(`molcrafts/software/`):

```bash
source tools/tb_oracle/setup_dftbplus_env.sh
# PATH → …/software/dftbplus-24.1/bin/dftb+
# ATV_DFTBPLUS_SK_MIO → …/software/dftb-params/mio-1-1
# ATV_DFTBPLUS_SK_3OB → …/software/dftb-params/3ob-3-1

tools/tb_oracle/run_dftbplus_pin.sh tools/tb_oracle/inputs/dftbplus/h2_mio_scc mio
```

Smoke: H₂ / mio-1-1 / SCC @ 0.74 Å → **−18.36635052 eV** (= ATV
`reg_dftb2-dftbplus-energy-pin`). Details: `molcrafts/software/README.md` and
`.claude/notes/dftb-vs-dftbplus-cleanup.md`.

## Layout

```text
tools/tb_oracle/
  README.md                 # this file
  inputs/
    dftbplus/               # DFTB+ HSD / geo.gen twins + notes
      ch4_3ob_dftb2/        # 3ob CH4 pure SCC-DFTB2 pin (wave-1)
      …                     # more twins land with worker tips / merges
    mopac/                  # NDDO OpenMOPAC pins
      am1-water/            # AM1 H2O absolute E / ΔHf / charge pin (wave-4)
      rm1-water/            # RM1 H2O ΔHf pin (wave-4)
    xtb/                    # GFN2 6.7.1 twins on disk; *.out gitignored
```

## Current offline twins (on `dev`)

| Path | Family | Notes |
| ---- | ------ | ----- |
| `inputs/dftbplus/ch4_3ob_dftb2/` | DFTB+ | CH4 3ob-3-1 under pure DFTB2 (`SCC=Yes`, no `ThirdOrder`) |
| `inputs/dftbplus/h2_mio_scc/` | DFTB+ | H₂ mio-1-1 absolute E pin twin (`reg_dftb2-dftbplus-energy-pin`) |
| `inputs/dftbplus/dftb2-periodic-gamma/` | DFTB+ | Γ-PBC C-chain geometry twin (ATV Hamiltonian pin) |
| `inputs/dftbplus/dftb2-periodic-forces/` | DFTB+ | Off-eq Γ-PBC C-chain force twin (analytic vs re-SCC FD) |
| `inputs/dftbplus/dftb2-kpoints/` | DFTB+ | 4×1×1 MP multi-k C-chain twin (`reg_dftb2-kpoints`; energy-only) |
| `inputs/dftbplus/dftb0_h2.note` | DFTB+ | H₂ non-SCC DFTB0 pin twin (`reg_dftb0-h2`; shares the mio-1-1 H₂ transcript) |
| `inputs/dftbplus/dftb3_h2o_mio_note.md` | DFTB+ | H₂O SCC-DFTB3 formula / module boundary (`reg_dftb3-water-mio`) |
| `inputs/dftbplus/dftb2_analytic_forces_note.md` | DFTB+ | Formula-level force module boundary (MioSketch) |
| `inputs/dftbplus/dftb2_spin_analytic_forces_note.md` | DFTB+ | Collinear open-shell force module boundary (O₂) |
| `inputs/dftbplus/dftb2_analytic_forces_threeob_note.md` | DFTB+ | ThreeOb closed-form F vs re-SCC FD (formula) |
| `inputs/dftbplus/dftb2_analytic_stress_note.md` | DFTB+ | Cell-stress module boundary (pair virial; formula) |
| `inputs/dftbplus/dftb2-dispersion-d3/` | s-dftd3 | Water-dimer **molecular** D3(BJ) geometry twin (`reg_dftb2-dispersion-d3`) |
| `inputs/mopac/am1-water/` | OpenMOPAC | AM1 H₂O deck + archived MOPAC 7.00 transcript (`reg_am1-water-mopac-energy`); **first mopac twin** |
| `inputs/xtb/{h2o,ch4,nh3,ch3oh}_gfn2/` + `h2o_gfn2_offeq_grad/` | xtb 6.7.1 | transcripts on disk; `*.out` gitignored — Phase C, not yet a product pin |
| `inputs/mopac/rm1-water/` | OpenMOPAC | RM1 H₂O ΔHf decks (`reg_rm1-water-mopac-energy`); the ΔHf path is calibrated against MOPAC2016 21.298M `tests/keywords/RM1.out` (CO₂) to 3.1e-4 kcal/mol |

Twins above are on `dev` (wave-3/4 dftbplus + mopac water). Wave-6 census:
xtb GFN2 transcripts exist on disk but are gitignored (`*.out`); PM3/PM6/PM7
and NDDO CO₂ mopac twins still to seed. Periodic-D3 / multi-k-force twins
already landed with those slugs.

## Pinning when the oracle binary is unavailable

`mopac` is not installed on the Atomiverse build hosts. A **published
transcript** of the oracle is then a valid pin source provided it is treated
like any other offline run:

1. Transcribe the transcript **verbatim** under `inputs/<family>/<slug>/`,
   with the reconstructed input deck beside it and a README that says which
   parts are verbatim and which are reconstructed.
2. Take geometry **and** energy from the *same* transcript. Splicing a
   published energy onto a geometry from elsewhere is not a pin.
3. State the package version in `REFERENCE.md` (`oracle_version:`) — an
   archived MOPAC 7.00 run is not a current OpenMOPAC run, and the difference
   belongs in the tolerance rationale, not in the pin.
4. Size the tolerance from the *transcript's* precision and any known
   convention offset (see the constant note below), never from the observed
   residual alone.

**Constant-convention floor (NDDO pins).** Atomiverse's NDDO integrals use
CODATA-2018 (`a₀ = 0.529177210903` Å, 1 Ha = `27.211386245988` eV);
MOPAC froze `a₀ = 0.529167` Å, 1 Ha = `27.21` eV, `e²·k = 14.399` eV·Å. That is
a 4.5×10⁻⁵ relative difference and it puts a ~10⁻³ eV floor under **every**
AM1/RM1/PM* pin against a MOPAC transcript. Measured for AM1 water: 6.2×10⁻⁴ eV
of a 6.2×10⁻⁴ eV residual, i.e. 82 % of the gap, disappears when ATV is rebuilt
with MOPAC's constants. Do not chase it per row, and do not tighten an NDDO
mopac tolerance below it.

## How to pin (operator / worker)

1. Run the family oracle **once** offline on the pin host (or cite a published
   formula when the binary is unavailable).
2. Record exact geometry, method flags, package version, and numbers in
   `regressions/semiempirical/<slug>/REFERENCE.md`.
3. Copy minimal input twins under `tools/tb_oracle/inputs/<family>/…` so the
   pin is reproducible without network or live oracle in CI.
4. Hard-code tolerances in `reg_<slug>.cpp` — ctest must never shell out to
   mopac / dftb+ / xtb.

## Temps

Scratch only: `/tmp/tb_oracle_*` `/tmp/mopac_*` `/tmp/dftbplus_*` `/tmp/xtb_*`
`/tmp/atv-*`. Wipe after pin generation; never commit oracle binaries.
