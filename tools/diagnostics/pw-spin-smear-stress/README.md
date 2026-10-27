# CPU plane-wave spin-smearing stress

This opt-in diagnostic exercises the public CPU `PW::run`, `forces`, `stress`
and `pw_stress_fd` methods. It is outside the CTest unit tier. Configure a
PW-enabled repository build with the opt-in target, then run on a CPU compute
allocation:

```sh
cmake -S "$repo" -B "$build" \
  -DCMAKE_PROJECT_Atomiverse_INCLUDE="$repo/tools/diagnostics/pw-spin-smear-stress/register.cmake"
cmake --build "$build" -j4 --target pw-spin-smear-stress
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$build/bin/pw-spin-smear-stress" --mode matrix --output /fresh/matrix.json
```

The matrix uses a triclinic H3 cell, builtin GTH-PBE-q1, a 6 Ry wavefunction
cutoff, Γ, and multiplicity 2. LSDA and PBE each use Fermi–Dirac at .02 Ry,
Gaussian at .04 Ry, and MP1 at .04 Ry. Those inputs have separate actual
[QE 7.6 references](../../qe_oracle/inputs/pw-spin-smear-stress/reference.json)
with the full printed Cartesian stress tensor; the
[pin header](../../qe_oracle/inputs/pw-spin-smear-stress/pins.h) is generated
only from those external outputs. The energy and every stress-component gates
are 1e-5 Ha and 2e-3 eV/Å³. The physical forces, all shear components,
magnetization and entropy must be nontrivial. Both spin spectra share one
chemical potential; their fractional occupations and matching entropy enter
each strained non-self-consistent free energy.

The stress implementation uses one reference SCF and 12 frozen-density NSCF
strain samples at 2e-3. The reference integer wavefunction/density modes and
FFT shapes stay fixed; only their Cartesian metric and ionic operators
change. Density coefficients scale directly by Ω0/Ω. The result is a numerical
finite-basis strain derivative, **not a closed-form stress algorithm**.
Symmetric shear deforms both off-diagonal entries and therefore requires an
extra factor of two when returning the physical σxy, σxz and σyz components.
`last_result()` and frame energy retain the complete reference SCF state,
including its occupations, Fermi level, entropy, magnetization and iterations.

Independent single-component deformations `F=I±h e_a⊗e_b` check all six
components against full SCF energies at the unchanged 2e-3 eV/Å³ gate.
Two generic rigid-rotation cases check the physical tensor transformation
`σ'=RσRᵀ`, also at 2e-3 eV/Å³. These native consistency checks supplement
the external QE numbers. Fixed-spin and restricted-smearing cases separately
check metadata and the frame virial writeback.

With `ATV_USE_MPI=ON`, `--mode mpi` uses a 2×2×2 k mesh and the same H3
spin-smearing model. Run it once locally and once with two MPI ranks and
compare the complete energy and stress. The qualification run uses 1e-6 Ha
and 2e-5 eV/Å³ gates for that comparison. This checks the all-rank spectrum
collection and prevents duplicating an already complete band/entropy sum.

## Cutoff-crossing diagnostic

`--mode cutoff` intentionally retains a failed physical-reference check. At
h=.002 the symmetric +yz strain changes the wavefunction basis from 103 to
101 modes, while the single-component shear keeps 103. The base density
sphere contains 851 modes and several samples change its mode set too,
despite keeping the same 16×14×14 FFT shape. Consequently, symmetric full-SCF
finite differences of the re-cut energy are discontinuous and cannot act as
a physical stress oracle. This mode records all native mode counts and FFT
shapes, then exits nonzero at the original 5e-5 eV/Å³ comparison gate. The
failed comparison is preserved separately from the reference-cell QE comparisons; its gate
was not widened or converted into a pass.

An optional `pw-spin-smear-stress-before` target can compile a separately
saved earlier `pw_stress.cpp`, supplied as `ATV_PW_STRESS_BASELINE`. Its
`--mode negative` check is expected to fail against the physical simple-shear
energy derivative. That earlier implementation includes the missing shear
normalization and cutoff jumps; it is never used to generate reference values.

These checks qualify the recorded builtin-H, XC, spin, smearing, cutoff and
k-mesh combinations. They do not qualify arbitrary NC providers, GPW,
closed-form stress, or MP response/DFPT. In particular, generalized MP
occupations may be signed; a passing ground-state stress check does not
validate response code that discards negative occupations.

## Recorded CPU result

The [2026-10-26 campaign record](results/2026-10-26-cpu/validation.json)
finished on 2026-10-27 in CPU Slurm job 4478398. All six reference energies
and stress tensors match their actual QE counterparts: maximum errors are
1.20e-6 Ha and 1.99e-5 eV/Å³. Five independent full-SCF derivative cases pass,
with maximum error 4.17e-7 eV/Å³; the two rotation cases pass at 3.37e-7 eV/Å³.
Single-rank versus two-rank MPI agrees to 3.05e-12 Ha and 2.04e-9 eV/Å³.
Both metadata baselines pass. Original-source and cutoff-crossing checks
remain expected failures with their original gates.

**The complete matrix is not passing.** LSDA MP1 at .04 Ry fails the
independent full-SCF `F_xx=1-.002` convergence check at the unchanged 1e-9
joint density criterion. Both the original mix=.35/250-iteration protocol
and mix=.2/1000 iterations fail. Its unstrained energy and numerical stress
still pass the separate QE gates; its full-SCF derivative is recorded as
`null`, with `passed:false`, and the final job exits 1 after collecting all
other evidence. Per-case failures do not suppress later matrix cases.

The [convergence investigation](results/2026-10-26-cpu/convergence/analysis.json)
records a spin-density quadrature point repeatedly crossing zero. Its XC
potential jumps between a finite negative value and zero under the existing
variational clipped-density prescription. All 500 inner eigensolves satisfy
their Ritz-residual tolerance, while an actual QE Fortran replay of 716
captured spectra agrees with the native chemical potentials to 4.17e-16 Ha.
No XC extension, outer convergence tolerance, or physical comparison gate
was changed. `--mode mp-strains`, `--mix`, and `--max-iter` reproduce the two
solver protocols.

Six separate MP DFPT admission units pass; their
[independent build receipt](results/2026-10-26-cpu/dfpt-guard/build.json)
explicitly does not attest the stress binary. A configure-only check also
confirms that a minimal PW=ON/VIB=OFF archive includes each required
vibration helper exactly once. The record contains selected source hashes,
executed commands, binary hashes and logs from a dirty worktree; it does
not claim a formal reproducible-build attestation.
