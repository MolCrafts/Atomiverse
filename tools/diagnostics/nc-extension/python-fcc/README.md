# QE adjudication of the Python FCC Si fixture

These are actual serial QE7.5 executions, not native freezes or replacement
Python expectations. The fixture is unchanged: Si fcc primitive cell with
lattice constant5.43 Å, cell columns(a/2)(0,1,1),(1,0,1),(1,1,0), atoms
(0,0,0) and(1.4375,1.3575,1.3575) Å, published Si GTH-PBE-q4, Γ, ecutwfc8 Ry,
ecutrho32 Ry, eight valence electrons, four restricted fixed-occupation bands.
The potential is the parent directory's native QE `.gth` input. No generated
UPF, copied reader/operator or native solver contributes a reference value.

CPU `XCType::LSDA` calls Slater exchange plus VWN5 correlation. QE explicitly
uses `input_dft='SLA VWN'` and prints IDs(1,2,0,0,0,0,0). Its unpolarized
VWN parameters a=.0310907,b=3.72744,c=12.9352,x0=-.10498 match CPU VWN5.
The GTH-PBE potential parameters are held fixed even for LSDA, exactly as in
the existing Python fixture; this is not a switch to another pseudopotential.

`capture.py` writes each input then runs only one real QE process at a time:

```sh
python tools/diagnostics/nc-extension/python-fcc/capture.py
```

It uses the already built `/tmp/atv-qe-7.5-build/bin/pw.x`, reported source
commit770a0b2d12928a67048e2f3da8d10d057e52179e-dirty. Every input/raw output,
source/executable hash, exact command and environment requests
OMP_NUM_THREADS=1/OPENBLAS_NUM_THREADS=1 are in `reference.json` and the
corresponding files. All seven runs converged at conv_thr1e-12 Ry and
Davidson threshold1e-13 Ry; max_iter200, mixing0.3. These stricter external
convergence settings do not change fixture physics or any Python bound.

| XC / FFT grid | E [Ry] | max absolute force component [eV/Å] | preprojection net Fx [eV/Å] |
| --- | ---: | ---: | ---: |
| LSDA14³, matching native base quadrature | -14.29760005 | 2.0847256917 | +0.0002517110 |
| LSDA56³, numerical diagnostic | -14.29759746 | 2.0848976985 | +0.0000020569 |
| PBE22³, matching native private GGA quadrature | -14.34498873 | 2.0626954496 | -0.0010844914 |
| PBE44³, numerical diagnostic | -14.34499483 | 2.0622534769 | -0.0000658202 |
| PBE88³, numerical diagnostic | -14.34499499 | 2.0622416498 | -0.0000215973 |

Full six-component total forces, individual contributions and their sums are
preserved in `reference.json`; `analysis.json` converts their units explicitly.
These values do not qualify a stress or a performance ratio. Fine-grid runs
are convergence diagnostics, not different Python fixtures or new goldens.

## Net-force interpretation

QE's `PW/src/forces.f90` explicitly subtracts `sumfor/nat` before printing its
final total forces. Their exactly zero sum cannot independently validate the
unchanged Python1e-4 eV/Å net-force assertion. Verbosityhigh also prints the
nonlocal, ionic, local, core, Hubbard and SCF-correction contributions before
that projection. Their actual sum gives the last column above (uncertainty
from printed rounding is about1e-6 eV/Å).

Two additional PBE22³ runs translate **both** atoms by±0.001 Å along x while
keeping the cell fixed. Their actual energies are-14.34498881/-14.34498865 Ry.
The negative central derivative is-0.0010884555 eV/Å, agreeing with the
preprojection force-0.0010844914 eV/Å within the QE energy print-rounding bound
6.8e-5 eV/Å. Thus this coarse quadrature has an actual rigid-translation
energy error; zeroing a force sum would hide it. Nothing in this evidence
changes or waives the original1e-4 force-sum bound.

## Native validation after the numerical fixes

`native-validation.json` and `native-validation.out.txt` record the actual
CPU run against the relocated installed SDK. They are native validation
records; only `reference.json` supplies the independent QE expectations.
The original 8/32 Ry, Gamma, occupations, potential, tolerance1e-6 and
max_iter40 remain unchanged. Native XC integration now uses a fixed22³ LDA
or42³ PBE private quadrature and the same weighted-adjoint projection back
to the14³ density grid.

| XC | Native E [Ry] | Native net Fx [eV/Å] |
| --- | ---: | ---: |
| LSDA | -14.297597372302373 | +0.0000216874 |
| PBE | -14.344992484196649 | -0.0000134912 |

The Python suite passes50/50 checks. It compares LSDA energy and **all six**
force components with the actual matched-grid QE14³ execution at the original
0.001 eV/0.005 eV/Å bounds; its independent native net-force bound remains
0.0001 eV/Å. The former CPU/CUDA self-frozen expectation has been removed.
The fine-grid QE runs remain convergence diagnostics.

The common energy/force discrepancy came from truncating the ion-ion Ewald
reciprocal sum with the electronic cutoff. Ewald now selects its own physical
precision domains and owns the charged-background term; DFPT uses the same
domains. Local and nonlocal HF forces now use one consistent set of orbitals
and occupations. Both LDA and GGA use oversampled XC integration with the
energy/potential adjoint relation preserved. No force-sum projection or
acceptance-bound relaxation was applied. Fixed oversampling is a numerical
integration policy, not a universal error guarantee for arbitrary densities.
