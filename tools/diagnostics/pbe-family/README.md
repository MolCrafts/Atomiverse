# Shared PBE-family CPU/CUDA validation

Current [PW Driver results](results/2026-10-27-pw-driver/README.md) and
[GPW Driver results](../gpw-driver/results/2026-10-27/README.md) record passing
CPU, x86 CUDA build and actual GH200 envelopes. Earlier
[follow-up results](results/2026-10-27-followup/README.md) retain the original
failed GPW stage comparisons and their separate ten-test native repair. The
[original results](results/2026-10-27/README.md) retain earlier CPU checks and
the two failed GH200 integration builds.

The CPU point evaluator and CUDA restricted/collinear grid kernels select the
same immutable `electronic::PbeParameters`. PBEsol changes exchange μ and
correlation β, revPBE changes κ, and RPBE uses the exponential exchange
enhancement. Model selection happens on the host; CUDA receives the parameters
by value and evaluates the functional on the device. There is one grid entry
per spin convention, with an explicit `XCType` argument.

CUDA PW carries that identity through energy, force and frozen-density NSCF
strain stress. CUDA GPW carries it through its existing RKS KS chain, including
the multigrid and multi-k callers. GPW UKS retains its Padé-only admission rule.
Adding these functionals does not extend the existing solver/backend limits.

## Primitive checks

- `XCFunctional.PbeFamilyDerivativesMatchEnergy`: all four CPU models, both
  density derivatives and three independent sigma derivatives, including
  negative cross-spin sigma.
- `PwXcQuadrature.*`: all four models in the spectral energy/potential
  adjoint, including high-frequency modes and clipped spin-density ringing.
- `test_cuda_pw_xc_pbe` and `test_cuda_pw_xc_pbe_spin`: all four device models
  versus the CPU point evaluator, including vacuum, empty spin, finite
  gradients and the PW92 zero-gradient limit. Independent finite differences
  of device energy check both local spin potentials and all six flux lanes.
- `test_cuda_pw_density`: actual padded FFT XC stages, restricted and
  collinear, high-frequency density and clipped spin-density ringing.
- `test_cuda_gpw_ks_gga`: each family's collocation/Poisson/XC/integration
  chain versus CPU, with and without ionic and local potentials. This is a
  stage check, not a complete GPW SCF qualification.
- `GpwAdmission.*` and `test_cuda_gpw_stress`: invalid strain steps, including
  NaN/infinity, and an empty axis mask are rejected before SCF and Frame
  publication. CPU and CUDA reuse the same GPW probe validator.

Existing tolerances are retained for the expanded model matrix.

## Public-method diagnostic

Configure a CPU or CUDA PW build with:

```sh
cmake -S "$repo" -B "$build" \
  -DCMAKE_PROJECT_Atomiverse_INCLUDE="$repo/tools/diagnostics/pbe-family/register.cmake"
cmake --build "$build" --target pbe-family -j8
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$build/bin/pbe-family"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$build/bin/pbe-family" --smearing fd
```

The eight complete calculations cover PBE/PBEsol/revPBE/RPBE × restricted H₂
and collinear H₃, at Γ in a skew cell. Positions are `(1.2,1.1,1.0)`,
`(2.1,1.6,1.3)` and `(1.6,2.25,1.8)` Å (H₂ uses the first two). Cell vectors
are the columns of `[[4.2,.3,.2],[0,4,.1],[0,0,3.8]]` Å. All models deliberately
use the same builtin GTH-PBE-q1 pseudopotential to isolate XC model selection.
The wavefunction cutoff is 6 Ry, with fixed occupations, three extra bands,
density tolerance `1e-9`, at most 400 iterations and mixing fraction `.3`.
These are small diagnostic settings, without a cutoff-convergence claim.
The separate FD invocation uses H₃ for both spin conventions, a width of
`.02` Ry and three extra bands. Each thermal case requires the correct
electron count and a nonzero entropy correction. The two invocations cover
all four restricted/collinear and fixed/thermal SCF branches for every model.

Every CPU case requires converged reference and displaced SCFs, correct Frame
virial publication, and agreement of one nonzero ionic force with a separate
full-SCF central energy derivative at `1e-4` Å (tolerance `2e-4` eV/Å).
CUDA builds additionally compare public CPU/CUDA energy (`1e-7` Ha), every
force (`2e-5` eV/Å) and all six stress components (`5e-4` eV/Å³), checking the
CUDA reference convergence and Frame virial. Stress remains a frozen-density
NSCF strain finite difference at the existing production step; this diagnostic
does not establish a closed-form stress implementation.

CUDA cases also solve before moving the method, destroy the moved-from
object, and reuse the destination for const-run, force and stress calls.
They check unrequested Frame channels, one-iteration exhaustion after a
successful solve, and changed-geometry reuse against a fresh method. These
are opt-in complete-method checks, outside CTest.

With CUDA and tests enabled, `pw-cell-stress-baselines` preserves the three
original complete-SCF stress cases moved out of the unit suite, including
their original gates. `pw-driver-baselines` checks direct nesting under
`cuda::Evaluate`; the separate native record lists its evaluated branches
and lifecycle assertions. `cuda-pw-public` retains the earlier public-result
checks and existing actual QE Si₂ energy/full-force pin. Passing new model
twins does not replace that independent pin.

These are consistency and CPU/device comparisons. They do not replace
independent QE/CP2K pins for the new periodic functionals, validate molecular
CUDA KS, or establish complete GPW product parity. Numerical results are kept
separate from roadmap qualification.

## GPW public-method diagnostic

With GPW enabled, the same configure hook adds `pbe-family-gpw`:

```sh
cmake --build "$build" --target pbe-family-gpw -j8
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$build/bin/pbe-family-gpw"
```

Four Γ RKS fixtures use public `Kind`, `Frame`, `build`, `run`, `forces` and
`stress` on each backend. H₂ uses the DZV-GTH-PADE basis and the same GTH-PBE-q1
potential for every functional. The orthorhombic cell is `(12,11,10)` Bohr;
positions are `(5.45,5.15,4.4)` and `(6.55,5.65,4.8)` Bohr. Parameters are
80 Ry density cutoff, one doubly occupied orbital, DIIS depth 6, SCF tolerance
`1e-10`, at most 100 iterations and Gaussian-tail threshold `1e-10`.

The fixed CPU/CUDA gates are `5e-6` Ha for energy, `5e-5` Ha/Bohr for every
force, and `5e-5` Ha/Bohr³ for each stress entry, converted to public units.
The independent CPU ionic difference uses `1e-4` Å and `1e-3` eV/Å tolerance.
Publication checks use `1e-10`; run and const-run must retain unrelated Frame
channels. CUDA `stress(Frame&)` uses one device SCF and the existing host
fixed-density strain assembly, with `stress_strain_h=.002`. It preserves forces
and writes `kStress=sigma*volume`. Only diagonal strains are implemented;
the three shear entries are checked as zero, without a shear-physics claim.

The GH200 collector compares each XML against the executable's unfiltered
`--gtest_list_tests` inventory and requires both complete public twin matrices.
Its `--envelope pw-driver` mode additionally requires the FD matrix, nested
Driver branches, all three moved SCF cases and the existing QE/public checks.
It records failure for missing, filtered or corrupt evidence, rather than
treating a partial run as the complete envelope.
