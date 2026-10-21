# CUDA GAFF neutral two-chain fragment

oracle: openmm

Force-field form: J. Wang, R. M. Wolf, J. W. Caldwell, P. A. Kollman and
D. A. Case, “Development and testing of a general amber force field”,
*J. Comput. Chem.* **25** (2004), 1157–1174,
[doi:10.1002/jcc.20035](https://doi.org/10.1002/jcc.20035).

`reference.txt` is the energy [eV] followed by all eight Cartesian force
vectors [eV/Å] printed by OpenMM 8.6.1, CUDA double, on Arrhenius GH200.
It is an engine-generated pin, not a number inferred from Atomiverse.
The exact oracle input is `tools/bench_gaff_openmm/fragment.data` and
`fragment.ff`. Reproduce inside a GPU allocation:

```sh
.openmm-env/bin/python tools/bench_gaff_openmm/compare.py \
  --data tools/bench_gaff_openmm/fragment.data \
  --ff tools/bench_gaff_openmm/fragment.ff --atv build-gaff-gpu/bin/bench_gaff \
  --out logs/gaff-fragment-pin --engine openmm --platform CUDA \
  --precision double --steps 1 --warmup 1
```

The first line of `logs/gaff-fragment-pin/openmm.txt` is the potential
energy. Copy that raw file to `reference.txt` only after actually running the
oracle. The standalone consumer uses the same explicitly listed positions,
charges, diagonal LJ parameters, harmonic bonds/angles, two Fourier torsion
terms, and Amber graph-distance specials. PME alpha is 0.35 Å⁻¹, mesh 24³,
order 5, cutoff 9 Å; LJ is unshifted without a dispersion correction.

This covers normal intermolecular nonbonded pairs, all three exception
distances, partial tiles, and a neutral PME system. Force and energy absolute
tolerances are 5e-5 in double and 5e-4 in mixed (eV/Å and eV respectively).
No trajectory or performance assertion is registered with ctest.
