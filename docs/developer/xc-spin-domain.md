# Spin-density domain and derivative contracts

The public analytic XC point evaluator and the derivative of a composed,
regularized grid energy need explicit, separate contracts at a density floor.
The current PBE-family point law is preserved. Its negative-density extension
and its exact floor/fully polarized boundaries do not acquire variational
qualification from CPU/CUDA parity or ordinary positive-density tests.

## Why the raw-energy derivative proposal was not adopted

A retained PBE probe at alpha density 0.2 and beta density -0.01 has analytic
beta potential about 99.42 Ha, while differentiating its literal raw composite
energy gives about -0.03354 Ha. A scratch chain-rule implementation preserves
the primal energies and differentiates that composition, including its smooth
piecewise branches. This mathematical property does not establish compatibility
with the existing public physical operator.

Actual LibXC 7.0.0 evaluations at alpha=0.2, beta=0 and zero gradient give
beta potential -0.2479485160 Ha with default thresholds, and -0.2479759008 Ha
with both exchange/correlation density floors set to 1e-15. The current CPU
value is -0.2479758895 Ha; the raw-composite proposal instead gives
-0.03090038974 Ha. These are distinct limiting conventions, even without a
minority-gradient singularity. With sigma_aa=0.02 the default LibXC, adjusted
LibXC and CPU values are 11.59022655, 118.65256358 and 118.21053158 Ha;
changing thresholds is not a monotone numerical convergence argument.

LibXC's wrapper sanitizes density/sigma arguments, then calls its generated
analytic evaluator; it does not apply the sanitization Jacobian to returned
potentials. This source observation explains why differentiating the whole
wrapper is a different operation. See the [LibXC 7 GGA wrapper](https://gitlab.com/libxc/libxc/-/raw/7.0.0/src/work_gga_inc.c),
[threshold setters](https://gitlab.com/libxc/libxc/-/raw/7.0.0/src/functionals.c)
and [manual](https://libxc.gitlab.io/manual/libxc-5.1.x/).

The difference is observable through the current GPW public method. On a
matched fixed-occupation H2 alpha2/beta0 fixture, the proposal changes energy
by at most 3.2e-15 Ha and alpha levels by about 2.6e-12 Ha, but changes PBE's
published beta canonical levels from approximately [238.647, 296.109,
675.437, 707.450] Ha to [-0.115, 0.311, 0.782, 1.583] Ha. This comparison
shows a compatibility change; it does not independently qualify either
virtual spectrum. GpwResult publishes beta virtual levels even with no
occupied beta orbitals. Current Bands/DOS admission requires both spins
occupied, and current unrestricted TDA/TDDFT is LSDA-only.

## Requirements for a future discrete extension

A grid caller needing a variational extension must specify its primal energy,
admitted density/gradient domain and pullback together. Density projection,
spin floors, zeta clipping and sigma guards have different derivatives;
masking an established potential cannot generally differentiate an unchanged
raw-density energy. Exact kinks require an explicit one-sided or subgradient
convention. A central finite difference across a kink is not an oracle for
one branch's derivative.

Retain positive, unsaturated-domain qualification separately from raw negative
compatibility behavior. Qualify occupied-density variations and published
virtual/response operators separately at an exactly inactive spin channel.
No global replacement is approved by the present study.

The retained local study is in `.molq/gpw-uks-pbe/`:
`emptybeta-compatibility.json` records 1920 actual LibXC points with read-back
thresholds and source/library hashes; `emptybeta-observable-compatibility.json`
records public spectra; `piecewise-validation.json` records the mathematical
proposal. These scratch artifacts do not replace a scoped release gate.
