# Plane-wave smearing: QE routine pins

`capture_qe.py` generates `tests/cpu/planewave/smearing_qe_pins.h`, the
independent reference for the shared CPU/CUDA smearing translation unit
`src/cpu/planewave/internal/smearing.cpp` (unit target `test_cpu_pw_smearing`).
It compiles **unmodified Quantum ESPRESSO 7.6 routines**, including the actual
Fermi-level search, and records 65 local occupation/entropy points and 20
five-band spectra for Fermi–Dirac, Gaussian and Methfessel–Paxton orders 1–3,
with spin degeneracy 1 and 2, metallic fillings, insulating plateaus and a
nonmonotone MP number function.

The script downloads the tagged upstream sources, or takes a local copy with
`--source-dir`, and compiles
[wgauss](https://github.com/QEF/q-e/blob/qe-7.6/Modules/wgauss.f90),
[w1gauss](https://github.com/QEF/q-e/blob/qe-7.6/Modules/w1gauss.f90),
[w0gauss/w2gauss](https://github.com/QEF/q-e/blob/qe-7.6/Modules/w0gauss.f90),
[efermig](https://github.com/QEF/q-e/blob/qe-7.6/PW/src/efermig.f90) and
[sumkg](https://github.com/QEF/q-e/blob/qe-7.6/PW/src/sumkg.f90) with
`qe_driver.f90`. `qe_support.f90` defines constants, precision and identity
reductions for one serial pool; it doesn't reimplement any smearing or
root-search formula. It needs a Fortran compiler and Python 3.10 or newer; no
Atomiverse library is linked. Run from the repository root:

```sh
python tools/qe_oracle/inputs/pw-smearing/capture_qe.py --out "$OUT" --header "$OUT/pins.h"
cmp tests/cpu/planewave/smearing_qe_pins.h "$OUT/pins.h"
```

`$OUT` must not exist. It receives the downloaded sources, the build log, the
raw QE output and `reference.json` (source URLs and hashes, compiler identity,
commands, executable hash and any Slurm allocation); none of it is committed.

## Conventions

For reduced energy `x=(epsilon-mu)/width`, the spinless conventions are
`f(x)=QE.wgauss(-x,N)` and `s(x)=-QE.w1gauss(-x,N)`. MP occupation is signed
and can exceed one. With physicists' Hermite polynomials and
`A_N=(-1)^N/(4^N N! sqrt(pi))`, its entropy is
`s_N(x)=0.5 A_N H_(2N)(x) exp(-x*x)`. Band occupation and entropy each acquire
the same spin degeneracy. This satisfies `s'(x)=x f'(x)` and gives the free
energy correction `-width * sum(weight * s)`.

The unit gates are `2e-13` for local values, `2e-12` for fixed-mu spectrum
number/entropy and `1.5e-10` for the solved electron count. Metallic roots are
compared with QE within `1e-8` in mu and `5e-8` in entropy. Inside an
insulating gap mu isn't unique: both roots must have negligible entropy and
the native root must stay at least four widths from either gap edge. QE MP2/3
can stop on its step-size criterion with count residuals up to `2.9e-9` on
that plateau; the native search keeps its `1e-10` count criterion.

The 2026-10-26 repair that these pins qualified kept signed MP tails, fixed
the MP entropy normalization, stopped the Fermi search on electron count,
started the nonmonotone MP search from the Gaussian root and made the exactly
occupied FD entropy endpoint finite; the previous implementation failed the
same comparison. **MP DFPT response isn't qualified**: the public DFPT solver
rejects MP at admission.
