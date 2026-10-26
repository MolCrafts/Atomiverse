# SQM offline reference

These inputs and outputs come from executing **AmberTools SQM**, not from
Atomiverse. `provenance.json` records the local package, binary hash, command
and separate source reference. The current input is an isolated, distorted
water single point (`maxcyc=0`); `verbosity=4` prints the full nuclear gradient.

The unmodified raw output is archived as `outputs/am1-water/sqm.out.txt`
so Git does not treat it as a compiled executable. Input and output hashes
are recorded in `provenance.json`.

To regenerate an output using the installed oracle:

```sh
mkdir -p /tmp/sqm-am1-water
cp tools/sqm_oracle/inputs/am1-water/sqm.in /tmp/sqm-am1-water/
cd /tmp/sqm-am1-water
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$AMBERHOME/bin/sqm" -O -i sqm.in -o sqm.out
```

The reported total SCF energy is electronic plus core-core energy; it is not
the heat of formation. SQM prints the nuclear **gradient** under the heading
“Forces on QM atoms from SCF calculation”. Its units are kcal/mol/Angstrom:
negate and divide by SQM's `EV_TO_KCAL=23.061` for forces in eV/Angstrom.
Mulliken charges are printed to three decimal places.

This is reference evidence for the port, **not yet an accepted parity test**.
Current ATV AM1 uses a different overlap approximation and numerical
convention. Its agreement with MOPAC does not imply exact SQM agreement.
The target is method/capability coverage with modern numerical algorithms,
not reproduction of historical SQM arithmetic. Explain these differences with
controlled comparisons; do not use a broad tolerance to conceal them.

Source reference:
[`Amber-MD/AmberClassic`](https://github.com/Amber-MD/AmberClassic/tree/76b00b2e9d7dc41ab1ce7c7aa2fe52b3608ec002/src/sqm).
The source revision is not claimed as the conda binary's build revision.
Relevant files are `constants.F90`, `qm2_setup_orb_exp.F90`, `qm2_h1elec.F90`,
`qm2_load_params_and_allocate.F90`, `qm2_get_qm_forces.F90` and `sqm.F90`.
