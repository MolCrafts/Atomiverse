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

Actual 7.0.0/7.1.2 raw values, reproduction commands, the high-precision script
and output, exact upstream source URLs and inspected-source checksum are
archived in [the LYP precision evidence](../../../../tools/diagnostics/lyp-precision/README.md).
The 7.1.2 limitation must not be hidden by distorting the native mathematics,
changing points or relaxing the original assertions.
