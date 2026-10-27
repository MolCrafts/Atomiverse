# Plane-wave smearing: independent QE qualification

The CPU and CUDA plane-wave solvers share
`src/cpu/planewave/internal/smearing.cpp`. This diagnostic compares that
production translation unit with **compiled, unmodified Quantum ESPRESSO 7.6
routines**, including its actual Fermi-level search. It covers Fermi–Dirac,
Gaussian, and Methfessel–Paxton orders 1–3: 65 local occupation/entropy points
and 20 five-band spectra, with spin degeneracy 1 and 2, metallic fillings,
insulating plateaus, and a nonmonotone MP number function.

`capture_qe.py` downloads the tagged upstream sources, or accepts their local
directory with `--source-dir`. It compiles
[wgauss](https://github.com/QEF/q-e/blob/qe-7.6/Modules/wgauss.f90),
[w1gauss](https://github.com/QEF/q-e/blob/qe-7.6/Modules/w1gauss.f90),
[w0gauss/w2gauss](https://github.com/QEF/q-e/blob/qe-7.6/Modules/w0gauss.f90),
[efermig](https://github.com/QEF/q-e/blob/qe-7.6/PW/src/efermig.f90), and
[sumkg](https://github.com/QEF/q-e/blob/qe-7.6/PW/src/sumkg.f90).
The support file defines constants, precision, and identity reductions for one
serial pool. It does not reimplement any smearing or root-search formula.
Source URLs and hashes, compiler identity, commands, executable hashes, and
Slurm allocation are recorded in `reference.json`.

For reduced energy `x=(epsilon-mu)/width`, the spinless conventions are
`f(x)=QE.wgauss(-x,N)` and `s(x)=-QE.w1gauss(-x,N)`. MP occupation is signed
and can exceed one. With physicists' Hermite polynomials and
`A_N=(-1)^N/(4^N N! sqrt(pi))`, its entropy is
`s_N(x)=0.5 A_N H_(2N)(x) exp(-x*x)`. Band occupation and entropy each acquire
the same spin degeneracy. This satisfies `s'(x)=x f'(x)` and produces the free
energy correction `-width * sum(weight * s)`.

The archived production implementation clipped MP occupation, doubled its
entropy, drove already-correct insulating electron-count plateaus toward a
band edge, and returned NaN for the exactly occupied FD entropy endpoint.
The repair retains signed MP tails, uses the correct entropy normalization,
stops Fermi search on electron count, starts nonmonotone MP search from the
Gaussian root, and evaluates endpoint/tail limits safely. A weighted Hermite
recurrence avoids separate large polynomials and factorials.

Run these commands inside a CPU batch allocation, from the repository root.
They need a Fortran compiler, a C++20 compiler, and Python 3.10 or newer; no
Atomiverse library or Python extension is linked.

```sh
python tools/diagnostics/pw-smearing/capture_qe.py \
  --out /tmp/qe-smearing-reference \
  --header /tmp/qe-smearing-pins.h
cmp tests/cpu/planewave/smearing_qe_pins.h /tmp/qe-smearing-pins.h
python tools/diagnostics/pw-smearing/compare.py \
  --qe-reference /tmp/qe-smearing-reference/reference.json \
  --before-source tools/diagnostics/pw-smearing/evidence-20261026/before-production.cpp \
  --out /tmp/qe-smearing-comparison
```

The comparator builds the archived and current **production** source separately
with the same probe and actual QE fixture. It retains both raw outputs, complete
failure lists, and maximum errors. It requires the archived implementation to
fail and the repaired implementation to pass. A nonfinite value is retained
as JSON `null`, never silently discarded. The separate registered unit target
`test_cpu_pw_smearing` also checks the variational identity, signed grand
potential derivatives, extreme tails, invalid inputs, and exact FD endpoints.

The scalar gates are `2e-13` for local values, `2e-12` for fixed-mu spectrum
number/entropy, and `1.5e-10` for the solved electron count. Metallic roots are
compared to QE within `1e-8` in mu and `5e-8` in entropy. Inside an insulating
gap mu is not unique: both QE and native roots must have negligible entropy,
and the native root must remain at least four widths from either gap edge.
QE MP2/3 can stop on its step-size criterion with count residuals up to
`2.9e-9` on this plateau; native search keeps its `1e-10` count criterion.
No full-SCF energy or stress gate is changed by these scalar checks.

The dated evidence directory contains the compiled-QE output, before/after
probe output, unit log, provenance, and checksums. Full CPU/CUDA spin-polarized
SCF energy and stress comparisons live in the sibling
`pw-spin-smear-stress` and `cuda-pw-spin-smear-stress` diagnostics. Their final
MP width is 0.04 Ry, with the original nonzero-entropy gate retained; the old
0.02 Ry PBE case had effectively zero true QE entropy.

MP ground-state occupations and stress are supported. **MP DFPT response is
not qualified**: the response implementation assumes nonnegative occupied
projectors and includes FD-specific occupation derivatives. The public DFPT
solver rejects MP at admission. Supporting MP response requires signed
projectors and the corresponding MP derivatives throughout that algorithm.

## A subsequent LSDA MP SCF failure

`mp-root-review-20261026/summary.json` retains a later **failed** full-SCF
strain sample: LSDA MP1, 0.04 Ry, `Fxx=-0.002`. Its joint density residual
stayed above `1e-8` with both 250 iterations at mixing 0.35 and 1000 iterations
at mixing 0.2. This failure is separate from the repaired scalar qualification.

`replay.py` sends all 716 recorded spectra through actual QE `efermig`.
Native and QE chemical potentials agree within `4.17e-16` Ha. In the failing
late-iteration segment, actual QE gives `|dN/dmu| >= 5.549` per Ha.
`analyze_replay.py` then tightens the same local MP1 root to a number residual
near `2e-15`: mu changes by at most `5.95e-12` Ha and any occupation by at most
`1.37e-11`. The observed mu fluctuation is `4.50e-6` Ha. Supplied eigenvalue
changes predict adjacent mu changes through the implicit-function derivative
within `5.16e-10` Ha. These measurements support a continuous root following
changing input eigenvalues; they do not support changing root tolerance or
branch selection to fix this SCF failure. Inner Davidson termination and
density/XC clipping remain candidate causes requiring separate evidence.

The replay accepts the diagnostic-only CPU solver's `TRACE_SPECTRUM`,
`TRACE_RESIDUAL`, and `STRAIN_SAMPLE` log records. This input format describes
MP1 with spin degeneracy one, as used in the joint collinear spectrum. Run
the Fortran compilation/replay within a CPU allocation:

```sh
python tools/diagnostics/pw-smearing/replay.py \
  --trace tools/diagnostics/pw-smearing/mp-root-review-20261026/source-trace-250.log \
  --qe-source-dir /tmp/qe-smearing-reference/sources \
  --out /tmp/qe-smearing-replay
python tools/diagnostics/pw-smearing/analyze_replay.py \
  --replay /tmp/qe-smearing-replay/replay.json \
  --out /tmp/qe-smearing-root-analysis.json
```
