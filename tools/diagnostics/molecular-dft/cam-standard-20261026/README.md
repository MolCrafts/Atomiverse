# Standard CAM-B3LYP repair, 2026-10-26

The native CPU implementation now uses the standard CAM-B3LYP semilocal
functional, with short-range B88 exchange and VWN5 correlation. The previous
B3LYP semilocal substitute remains documented in the
[failed historical comparisons](../cam-standard-mismatch.json).

The recipe follows [libxc 7.0.0's CAM-B3LYP initializer](https://gitlab.com/libxc/libxc/-/blob/7.0.0/src/hyb_gga_xc_cam_b3lyp.c):

```text
f_semilocal = .35 B88 + .46 ITYH-B88(mu=.33) + .19 VWN5 + .81 LYP
K_exact    = .19 K[1/r] + .46 K[erf(.33*r)/r]
```

Libxc expresses the same partition as long-range HF `.65` and a short-range
HF adjustment `-.46`. The native exact-exchange parameters remain
`alpha=.19`, `beta=.46`, `mu=.33` in inverse Bohr. Standard CAM uses VWN5,
whereas the existing B3LYP recipe uses VWN-RPA; that distinction is retained.

For one spin channel, the [ITYH definition](https://gitlab.com/libxc/libxc/-/blob/7.0.0/maple/gga_exc/gga_x_ityh.mpl)
uses the full B88 enhancement `F` to set
`a=mu*sqrt(2*C_x*F/(9*pi))/(2*rho^(1/3))`. Its short-range energy is
`f_SR=f_B88*A(a)`, with the
[upstream erf attenuation](https://gitlab.com/libxc/libxc/-/blob/7.0.0/maple/attenuation.mpl).
The implemented first derivatives are analytic:

```text
g          = A + a*A'/2
vrho_SR    = vrho_B88*g - f_B88*a*A'/rho
vsigma_SR  = vsigma_B88*g
```

The direct attenuation expression uses extended precision. Above `a=1.35`
the native implementation evaluates the expansion through `a^-16`, with the
same cutoff/order as the inspected upstream definition. It evaluates `g`
separately to preserve its small value when the leading `a^-2` contributions
cancel. No numerical derivative of a global XC call is used. Production
kernels do not link libxc.

## Actual molecular qualification

Slurm **4476898** completed successfully on **n213**, using one math thread,
the public Python Frame/RKS/UKS interface and live PySCF 2.14.0 / libxc 7.0.0.
Both solvers converged for all six combinations:

| Method/system | Basis | Energy error, Ha | Maximum full-force-component error, Ha/Bohr |
| --- | --- | ---: | ---: |
| RKS non-symmetric water | STO-3G | 9.17e-12 | 4.82e-10 |
| RKS non-symmetric water | def2-SVP | 4.78e-12 | 2.65e-10 |
| UKS H2+ | STO-3G | 7.58e-12 | 1.54e-13 |
| UKS H2+ | def2-SVP | 1.20e-12 | 2.70e-10 |
| UKS triplet O2 | STO-3G | 6.37e-11 | 4.74e-11 |
| UKS triplet O2 | def2-SVP | 5.97e-11 | 2.67e-10 |

The unchanged gates are **1e-8 Ha** and **1e-6 Ha/Bohr**, with every force
component checked, finite values and both convergence flags required.
[Complete observations and provenance](molecular-results.json) retain inputs,
full vectors, native/PySCF library hashes, source hashes, grid/solver controls
and allocation metadata; the [compressed log](molecular.log.gz) retains the
native solver output. The native binary SHA256 is
`867e7c257d325f31799c5e2fe95b341919587d0b849bed12b6a341b394e57874`.
Formal build attestation remains explicitly false.

Reproduce in a fresh output directory with the matching native environment:

```sh
PYTHONPATH="$PWD/python/src:$PWD/build-hf-auto:$PWD" \
  .claude/scratch/hf-auto/env/bin/python \
  tools/diagnostics/molecular-dft/validate.py \
  --xc CAMB3LYP --threads 1 --out /path/to/fresh/cam-after
```

## Local derivatives and retained boundary differences

The [62 passing native unit tests](unit.log) include ordinary XC checks,
all first derivatives against genuine libxc ID433 at 12 declared normal-domain
points, and 30 one-spin ITYH energy/rho/sigma samples. The local gate remains
`abs(delta) <= 1e-11*max(1,abs(reference))`. The ITYH oracle uses actual
spin-unpolarized libxc calls, transformed by the exact spin-scaling identity;
this exercises dilute spin channels without subtracting `1-zeta` in the
oracle. Auxiliary checks cover finite differences, spin exchange, zero mu,
empty spin and the analytic large-mu density limit.

[Scalar pins](scalar-pins.json) retain all **15 original full-CAM points**,
including three outside that qualified normal envelope. This envelope
requires positive spin densities and `min(rho_a,rho_b)/max(...) >= 1e-6`;
it is a qualification boundary, not a new production cutoff.
[All comparisons](scalar-comparisons.json) explicitly retain
`all_points_passed=false`:

- At the two vacant-spin points, the unused spin potential differs by
  3.80e-9 Ha and its independent sigma derivative by 8.28e-6. Genuine component
  calls localize these residuals to the existing VWN spin-floor potential
  and LYP vacant-spin sigma evaluation. The new ITYH component agrees at
  approximately 1e-15 absolute scale. H2+ full physical forces above pass.
- At densities `(0.5,1e-8)`, the minority-spin sigma derivative differs by
  1.092e-4, about 6e-11 relative. The retained component outputs localize this
  residual to the existing full B88 polarized comparison; the new ITYH
  component differs by only 2.30e-13. This point remains unqualified at the
  strict local gate. It was not silently promoted to a pass.

The normal-domain and short-range component gates pass. These boundary
comparisons remain numerical limitations of the broader local-field claim;
they do not change the standard CAM recipe or the molecular energy/force
gates. No self-generated reference replaces a libxc result.

[Unit build/link evidence](unit-build.json) identifies the compiler, flags,
source and binary hashes; [inspected upstream sources](inspected-sources.json)
record exact URLs and checksums. Jobs **4476946** and **4476978** compiled only
affected tests and scalar diagnostics against the existing native library.
The native extension was not relinked during molecular validation.
These checks qualify the recorded CPU energy/force envelope, not periodic
hybrid DFT, CUDA molecular KS, or range-separated response kernels.
