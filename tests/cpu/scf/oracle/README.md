# Native XC parity oracle

The test-only adapter calls genuine libxc through `xc_gga_exc_vxc` and
`xc_lda_exc_vxc`. Production kernels do not link libxc. Dependency selection
belongs to build configuration; tests contain no version gates or skips.
The original `XCvsLibxc` density points and absolute/relative 1e-12 bounds
remain unchanged.

LYP is qualified against libxc 7.0.0. An actually executed libxc 7.1.2 call
loses about 6.5e-12 relative accuracy in the beta potential at densities
(1e-6,1e-8) and equal sigma entries 0.01,0.1,1. The new upstream helper
reconstructs small spin powers as `1+expm1(p*log1p(z))`; its cancellation
is visible in both values and derivatives. Independent 90-digit evaluation
of the official symbolic LYP energy supports the native and 7.0.0 results.
That high-precision calculation is a diagnosis, not an external oracle or
a source of regression pins. The native kernel uses direct density ratios
for spin fractions and their derivatives, preserving the same functional,
spin floor and energy conventions.

The 2026-10-25 diagnosis ran the unchanged `XCvsLibxc` battery against
PySCF-bundled libxc 7.0.0 and Homebrew libxc 7.1.2. With
sigma_aa = sigma_ab = sigma_bb = 0.01, 0.1 and 1.0, the beta potential was:

| sigma | libxc 7.0.0 | libxc 7.1.2 | 90-digit symbolic derivative |
| --- | ---: | ---: | ---: |
| 0.01 | 9.5415919508346363 | 9.5415919507723252 | 9.5415919508344597 |
| 0.1 | 95.464578041955633 | 95.464578041332558 | 95.464578041953984 |
| 1.0 | 954.69443895316556 | 954.69443894693609 | 954.69443895314923 |

The native kernel passes all 120 comparisons against 7.0.0 (maximum scaled
error 1.77e-14); against 7.1.2 these three entries fail (6.51e-12). The
90-digit values differentiate the Miehlich LYP energy with mpmath 1.4.1. The
reference expression is the official
[libxc 7.1.2 symbolic LYP source](https://gitlab.com/libxc/libxc/-/raw/7.1.2/python/gga_exc/gga_c_lyp.py);
the inspected
[generated 7.1.2 kernel](https://gitlab.com/libxc/libxc/-/raw/7.1.2/src/maple2c/gga_exc/gga_c_lyp.c)
has SHA256 `38239ef2f216310780dda61bd6c1265fd78d8739c3e52c248eff86f21846be68`.
Its `opz_pow_n_m1_k` helper returns `expm1(p*log1p(z))` but rebuilds the power
by adding 1; for the minority spin fraction 0.019801980198019806 the 11/3
power then differs from the direct power by 4.74e-11 relative.
The 7.1.2 limitation must not be hidden by distorting the native mathematics,
changing points or relaxing the original assertions.
