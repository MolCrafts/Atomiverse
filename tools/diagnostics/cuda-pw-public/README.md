# CUDA PW derivative results

This opt-in diagnostic executes full public `cpu::pw::PW` / `cuda::pw::PW`
solves. It is not a CTest unit and does not use the archived regression runner.
Run configuration, compilation and execution on a CUDA compute node.

Add the target to an existing CUDA+PW repository configuration:

```sh
cmake -S "$repo" -B "$build" \
  -DCMAKE_PROJECT_Atomiverse_INCLUDE="$repo/tools/diagnostics/cuda-pw-public/register.cmake"
cmake --build "$build" -j8 --target cuda-pw-public
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 "$build/bin/cuda-pw-public"
```

`ATV_BUILD_CUDA=ON`, `ATV_BUILD_PW=ON` and a compatible local molpy checkout
must already be configured. GH200 uses `CMAKE_CUDA_ARCHITECTURES=90`.

The diagnostic checks:

- The actual QE7.5 Si₂ absolute Mermin energy and all six nonzero forces,
  using the original inputs and printed pins from
  [`examples/nc-extension`](../../../examples/nc-extension/README.md).
  The CUDA method uses the builtin GTH-PBE-q4 with the same published
  coefficients, 12 Ry, Γ, six bands, Fermi–Dirac smearing 0.02 Ry,
  tol 1e-9, max_iter 200 and mixing 0.2. Original bounds remain 1e-5 Ha
  and 1e-4 Ha/Bohr per component.

- PBE GTH Si₂ at 8 Ry, Γ, stretched by 0.08 Å: all six nonzero force
  components against the matching CPU method, max error below 5e-3 eV/Å,
  energy below 1e-4 eV and force amplitude above 0.05 eV/Å.
- PBE H₂ restricted / H collinear, each with fixed occupations and
  Fermi–Dirac smearing at 6 Ry, Γ: `forces()` reports the same SCF energy,
  convergence, iteration count, k-point counts, magnetization, Fermi level,
  electron count and entropy as an independent `run()` at identical settings.
- The supported stress combinations report their reference SCF metadata and
  write the returned intensive stress times cell volume into `kStress`.
- After a converged call, one-iteration force/stress calls replace the old
  status with `converged=false`, `iterations=1` and fresh metadata.

At the time of this archived run, spin with smearing had no consistent CUDA
strained-occupation evaluator and `stress()` rejected it before device setup.
The [subsequent spin/smearing stress diagnostic](../cuda-pw-spin-smear-stress/README.md)
tracks its replacement: shared-chemical-potential strained occupations,
fixed reference basis, physical shear normalization and a joint spin
convergence norm. Those changes and their validation have separate records;
the original 37-unit run below does not qualify them.

The stretched diamond Si comparison is a CPU twin check. The separate cubic
Si₂ case uses the actual independent QE pin. H comparisons check API
consistency; they are not independent external numerical qualification. The stress algorithm
remains a frozen-density NSCF strain finite difference, and the CUDA band solver
still has the host scalar/spectrum transfers documented in its source.

CPU and CUDA share the private XC grid rule: even dimensions at least 1.5×
the base dimensions for LSDA and 3× for PBE. The device interpolates the same
G-sphere onto this private quadrature and projects its functional derivative
back with the normalized adjoint gather. The band/density FFT dimensions are
unchanged. These stages reuse existing cuFFT, scatter/gather and pointwise
kernels, with per-instance scratch and no new scientific host transfers.
The high-frequency positive-density test exercises aliasing that the smooth
atomic seed alone misses; the spin test checks both projected channels.

Recorded GH200 results and their limits are in
[`results/2026-10-26-gh200/validation.json`](results/2026-10-26-gh200/validation.json).
The successful numerical run is job 4475642 on n207: 37 actual Google Tests,
all H metadata/reuse checks, and both Si checks passed. The Si CPU twin
residual remains 9.09e-5 eV/Å (SCF tolerance and stopping differences); this
is not a rounding-level equality claim. The QE energy error is 1.13e-6 Ha,
maximum force error 1.16e-5 Ha/Bohr. This qualifies only the stated builtin
Si/PBE/Γ/smearing energy-and-force envelope, not arbitrary NC providers,
k-points, elements or stress. It contains no performance claim.

The negative control in job 4475731 rebuilds the original density source from
commit `911457aef4f18f509fa91a2c05dab745803fdd71` under a distinct class name
(the computation is unchanged) and runs the exact new high-frequency test.
It exits 1: for example, the zero-mode potential differs by 0.0131 Ha at a
1e-10 Ha gate. The repaired test passes unchanged. Compute Sanitizer reports
zero errors for both repaired padded-stage tests. Original source/header,
commands and compressed raw logs are retained with the results. The first
attempt's duplicate-symbol link failure is also recorded; it is not a
numerical failure or a pass.

To recreate this optional negative-control target, extract
`before-density-source.tar.gz` into a separate directory and configure with
`ATV_BUILD_TESTS=ON` and
`ATV_PW_QUADRATURE_BASELINE=/absolute/path/to/that/directory`, then build
`cuda-pw-density-before`. Running it with
`--gtest_filter=CudaPWDensity.PaddedXcResolvesHighFrequencyDensity` is expected
to fail. No alternate reference values or tolerances are used.
