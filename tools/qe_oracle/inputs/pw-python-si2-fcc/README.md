# QE reference for the Python FCC Si₂ fixture

`capture.py` produces `tests/data/pw/si2_fcc_qe75_reference.json`, which
`python/tests/test_pw.py` reads for its LSDA energy and six-component force
pins. The fixture is an Si fcc primitive cell with a = 5.43 Å, cell columns
(a/2)(0,1,1), (1,0,1), (1,1,0), atoms (0,0,0) and (1.4375,1.3575,1.3575) Å,
the published Si GTH-PBE-q4 (`../nc-extension-si2/Si_GTH_PBE_q4.gth`), Γ,
ecutwfc 8 Ry, ecutrho 32 Ry, eight electrons and four restricted
fixed-occupation bands.

```sh
python tools/qe_oracle/inputs/pw-python-si2-fcc/capture.py \
  --pw-x /path/to/qe-7.5/bin/pw.x --qe-source /path/to/qe-7.5-src --out OUT
cp OUT/reference.json tests/data/pw/si2_fcc_qe75_reference.json
```

`OUT` must not exist. The script writes each input, runs one serial QE
process at a time (`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) and records the
command, executable/source/input hashes, printed energy, total forces and
every printed force contribution. Runs use conv_thr 1e-12 Ry, Davidson
threshold 1e-13 Ry, at most 200 iterations and mixing 0.3; these stricter
external settings don't change the fixture physics or any Python bound.

CPU `XCType::LSDA` is Slater exchange plus VWN5 correlation, so QE uses
`input_dft='SLA VWN'` (printed IDs 1,2,0,0,0,0,0; unpolarized VWN parameters
a = .0310907, b = 3.72744, c = 12.9352, x0 = -.10498). The GTH-PBE potential
is kept for LSDA as in the existing fixture.

The current fixture is QE 7.5, commit
`770a0b2d12928a67048e2f3da8d10d057e52179e-dirty`, captured 2026-10-26:

| XC / FFT grid | E [Ry] | max absolute force component [eV/Å] | pre-projection net Fx [eV/Å] |
| --- | ---: | ---: | ---: |
| LSDA 14³, the native base quadrature (the pinned run) | -14.29760005 | 2.0847256917 | +0.0002517110 |
| LSDA 56³ | -14.29759746 | 2.0848976985 | +0.0000020569 |
| PBE 22³, the native GGA quadrature | -14.34498873 | 2.0626954496 | -0.0010844914 |
| PBE 44³ | -14.34499483 | 2.0622534769 | -0.0000658202 |
| PBE 88³ | -14.34499499 | 2.0622416498 | -0.0000215973 |

The fine-grid runs are convergence diagnostics, not other fixtures.

## Net force

QE's `PW/src/forces.f90` subtracts `sumfor/nat` before printing total forces,
so their zero sum can't validate the Python 1e-4 eV/Å net-force assertion.
The test therefore pins the sum of the printed contributions (nonlocal, ionic,
local, core, Hubbard and SCF correction) before that projection; printed
rounding contributes about 1e-6 eV/Å. Two further PBE 22³ runs translate both
atoms by ±0.001 Å along x: -14.34498881 / -14.34498865 Ry. Their negative
central derivative, -0.0010884555 eV/Å, matches the pre-projection force
-0.0010844914 eV/Å within the 6.8e-5 eV/Å print-rounding bound, so the coarse
quadrature has a real rigid-translation energy error.
