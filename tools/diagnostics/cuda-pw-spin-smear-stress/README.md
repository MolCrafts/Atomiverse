# Collinear PW smearing stress on CUDA

This opt-in diagnostic runs complete public CPU/CUDA PW methods on a GH200
allocation. It does not use the archived regression runner or register full
SCF solves as CTest units.

Configure a CUDA+PW build with the following additional CMake argument, then
build and run the diagnostic on the compute node:

```sh
cmake -S "$repo" -B "$build" \
  -DCMAKE_PROJECT_Atomiverse_INCLUDE="$repo/tools/diagnostics/cuda-pw-spin-smear-stress/register.cmake"
cmake --build "$build" -j8 --target cuda-pw-spin-smear-stress
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  "$build/bin/cuda-pw-spin-smear-stress" results.json .02
```

The default matrix is LSDA/PBE × Fermi–Dirac/Gaussian/MP1 at Γ. The width
argument specifies the FD width; Gaussian and MP1 use twice that width.
The accepted inputs use FD .02 Ry, Gaussian .04 Ry and MP1 .04 Ry. These
were selected from declared .12/.04/.02 candidates to retain both nonzero
spin and a nonzero free-energy correction. The original .12-Ry FD case
became effectively unpolarized; .02-Ry Gaussian and corrected MP1 did not
exercise a substantial correction. Those cases cannot substitute for a
nontrivial thermal-spin check.

The H₃ positions are (1.2,1.1,1.0), (2.1,1.6,1.3), (1.6,2.25,1.8) Å.
The cell matrix is `[[4.2,.3,.2],[0,4,.1],[0,0,3.8]]` Å, with lattice
vectors in columns. Both methods use builtin GTH-PBE-q1, 6 Ry wavefunction
cutoff, 24 Ry density sphere, five bands per spin, multiplicity 2 as an
initial seed, free final magnetization, density tolerance 1e-9, max_iter 250,
mixing .35 and MP order 1. Native PBE XC quadrature is 48×42×42; the actual
QE oracle uses 72³. Neither grid is claimed cutoff-converged.

The implementation differentiates the smearing free energy with one reference
SCF plus twelve frozen-density NSCF strain samples at h=.002. It retains the
reference density/wavefunction Miller labels, order, k mesh and FFT shape,
and recomputes metric-dependent operators. Runtime checks verify these
invariants in every sample. Both strained spin spectra are filled at one
shared chemical potential, with the matching smearing correction. MP
quasi-occupations may exceed the probability range; they must not be clipped.
This remains a numerical strain derivative, not a closed-form stress tensor.
Off-diagonal outputs are physical tensor entries, accounting for both lanes
of a symmetric strain probe.

Each Γ case requires:

- Actual independently printed QE7.6 absolute energy within 1e-5 Ha and each
  of six stress components within 2e-3 eV/Å³. The matched inputs, stdout and
  hashes are under [`tools/qe_oracle`](../../qe_oracle/inputs/pw-spin-smear-stress/).
- CPU/CUDA NSCF stress agreement within 5e-4 eV/Å³ and each engine's
  independent full-SCF simple-shear finite difference within 2e-3 eV/Å³.
  The independent deformation varies only `F_ab`, using no production strain
  helper and no inherited symmetric-shear factor.
- Proper rotation covariance within 2e-3 eV/Å³ on both FD functionals.
- Nonzero magnetization, correction, force and shear stress, complete reference
  SCF metadata and Frame virial publication, and fresh `converged=false`
  metadata after a one-iteration reuse.

The previous symmetric full-SCF probe at h=.002 reselected the cutoff basis:
its +yz sample dropped two wavefunction modes and produced a spurious stress
near -.203 eV/Å³ instead of the physical +.022 value. Its failing output is
retained as a cutoff diagnostic. It is not used as a physical stress oracle,
and its failed numerical threshold is not reported as a pass. Density-sphere
cutoff changes also limit finite-step comparisons; actual QE analytic stress
is the independent numerical qualification.

Spin SCF convergence uses the joint **unmixed** output-input norm of both
channels. The device computes it with existing cuBLAS copy/axpy/nrm2 routines
and a device result; no new kernel or scientific host transfer is added.
The pure-stage test applies equal-opposite constant spin changes: the total
charge residual vanishes, but the correct norm is sqrt(2)*delta independently
of the mixing fraction. A separately compiled original implementation must
fail this test. Another stage test verifies frozen-density volume scaling
retains every mode and rejects incompatible shapes instead of replacing a
nonuniform density with its mean.

Auxiliary commands:

```sh
# Input-selection evidence; these are probes, not qualification passes.
"$build/bin/cuda-pw-spin-smear-stress" --probe
# Small complete CUDA force/stress path, suitable for Compute Sanitizer.
"$build/bin/cuda-pw-spin-smear-stress" --stress-smoke
# Separate optional 2x2x2-k slice in a larger 6.2/6.0/5.8-A cell.
"$build/bin/cuda-pw-spin-smear-stress" multi-k.json .02 multi-k
```

The optional multi-k mode uses a different declared cell and a 1e-4 independent
FD step. It has no external QE pin. The original smaller-cell multi-k attempt
failed its unchanged 250-iteration cap and remains outside the qualified
convergence envelope. CUDA still uses linear spin mixing; it does not acquire
CPU joint DIIS or MPI k-point distribution from this repair. The existing
stress path retains its reference-density host handoff; no complete
CUDA-only residency claim is made.
