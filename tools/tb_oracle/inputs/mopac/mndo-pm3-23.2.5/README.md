# Live OpenMOPAC 23.2.5 MNDO/PM3 molecular pins

Six neutral closed-shell single points: distorted water, tilted CO and distorted
nonplanar ammonia for each method. The geometries are inputs, not optimized
structures. These are actual executions of the unmodified official macOS release,
not reconstructed literature transcripts. No legacy constants (`OLDCON`), SQM
parameter mode or changes to Atomiverse's model are involved.

`inputs/` contains the exact decks; `outputs/` contains unedited AUX, standard
output, archive and process output. `provenance.json` records the official release,
source commit, executable/library/archive hashes, controls and scientific-file
hashes. `pins.json` is mechanically extracted by `extract_mopac_pin.py`.
The binary reports 23.2.5 and ran through Rosetta on Apple silicon.

## Quantities and precision

- Coordinates are Å; atom order is unchanged and AUX input/final coordinates agree.
- AUX `ATOM_CHARGES` is in elementary charges (14 digits after the decimal point).
- AUX `GRADIENTS:KCAL/MOL/ANGSTROM` is **positive dE/dR**, with 13 digits after the
  decimal point. `forces_ev_angstrom = -gradient / 23.060547830619029`.
- AUX heat of formation is in kcal/mol, with 15 significant digits. It is a
  different energy convention from the electronic plus core total energy.
- `total_energy_ev` uses the `DISP` **kcal/mol** total-energy line, printed to five
  decimals, divided by the same conversion constant. Its rounding uncertainty is
  at most `2.1682e-7 eV`. The redundant five-decimal **eV** line has a larger
  `5e-6 eV` rounding bound and is not used for the precise energy pin.
- `DISP` only requests the energy breakdown. All six archived dispersion
  contributions are zero; it does not enable a dispersion correction.

These conventions follow the pinned upstream
[constants](https://github.com/openmopac/mopac/blob/052691223d19935a89f0fe18cd12301bd83e4201/src/conref_C.F90),
[AUX writer](https://github.com/openmopac/mopac/blob/052691223d19935a89f0fe18cd12301bd83e4201/src/output/to_screen.F90),
and [energy writer](https://github.com/openmopac/mopac/blob/052691223d19935a89f0fe18cd12301bd83e4201/src/output/writmo.F90).
See the official [GRADIENTS](https://openmopac.net/Manual/gradients.html),
[AUX](https://openmopac.net/Manual/auxiliary.html),
[SCFCRT](https://openmopac.net/Manual/scfcrt.html) and
[DISP](https://openmopac.net/Manual/disp.html) documentation.

## Gradient accuracy qualification

**This is an independent numerical-gradient oracle, not an analytic-gradient
oracle.** In upstream
[`dcart.F90`](https://github.com/openmopac/mopac/blob/052691223d19935a89f0fe18cd12301bd83e4201/src/forces/dcart.F90#L83),
`chnge = 1e-4 Å`; `PRECISE` selects the centered pair-energy difference at
`R ± 5e-5 Å` (lines 289–299). The pair density is held fixed at the SCF solution.
The printed 13 decimals do not imply 13-digit derivative accuracy.

`gradient-qualification.json` independently checks this limitation using only
OpenMOPAC: each Cartesian coordinate is displaced in both directions, with a new
SCF at steps `1e-3`, `5e-4` and `2.5e-4 Å`. Central differences of high-precision
HOF are Richardson-extrapolated at the two adjacent step pairs. The report stores
every displaced HOF, the extrapolated gradients, refinement differences and the
difference from the archived MOPAC gradient. The per-system Richardson refinement differences are recorded in the report; the
largest native-gradient difference is `4.69e-7 eV/Å` (PM3 water). No
Atomiverse result participates in this qualification. It supports a finite oracle error budget; it does not turn
finite differences into an analytic proof or broaden element/method coverage.

A `1e-6 eV` total-energy comparison accounts for the known print rounding; HOF
can separately be compared at `1e-6 kcal/mol`. Force acceptance of `1e-5 eV/Å`
is conservative relative to the independent gradient qualification, and is not
chosen from the Atomiverse–MOPAC residual. Charge acceptance remains a separate
SCF/population check (the archive preserves the unrounded AUX values).

## Reproduce

Obtain the [official release archive](https://github.com/openmopac/mopac/releases/download/v23.2.5/mopac-23.2.5-mac.zip)
and verify the hashes in `provenance.json`. From the repository root, use a scratch
directory; MOPAC creates its outputs beside each input:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p /tmp/mopac-replay
cp tools/tb_oracle/inputs/mopac/mndo-pm3-23.2.5/inputs/*.mop /tmp/mopac-replay/
for deck in /tmp/mopac-replay/*.mop; do
  /path/to/mopac "$deck"
done
python3 tools/tb_oracle/extract_mopac_pin.py \
  tools/tb_oracle/inputs/mopac/mndo-pm3-23.2.5 > /tmp/mopac-pins.json
python3 tools/tb_oracle/check_mopac_gradient.py /path/to/mopac \
  tools/tb_oracle/inputs/mopac/mndo-pm3-23.2.5 \
  --workdir /tmp/mopac-gradient-replay > /tmp/mopac-gradient-qualification.json
```

The last operation is an optional offline qualification (324 tiny single points),
not part of ctest. Raw displaced outputs remain in the scratch directory. No
MOPAC implementation or binary is included in this repository.
