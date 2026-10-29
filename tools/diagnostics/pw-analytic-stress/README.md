# CPU PW closed-form stress: public and nested validation

`pw-analytic-stress` is an opt-in scientific executable, outside CTest. It
checks the implemented Γ or unshifted 2×2×2 restricted, fixed-occupation LSDA Si2 slice through
public `PW::run`, `forces`, `stress`, and one existing `cpu::Evaluate` parent
requesting force and stress together. It adds no Driver or public state hook.

After including `register.cmake` in the PW diagnostic configuration:

```sh
cmake --build build --target pw-analytic-stress
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/bin/pw-analytic-stress
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/bin/pw-analytic-stress --mesh 2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 build/bin/pw-analytic-stress --mesh 2 --rotate-z90
```

Each of the four calls must perform exactly one SCF stage/criterion lifecycle.
A user stage collects actual eV energy, unmixed density residual and energy
change on every iteration. A stop-only criterion checks those same
measurements and execution ownership, without replacing the method's stopping
rule. Returned iterations must match both collector samples and criterion
polls. Standalone stage and criterion Frames must carry the actual input
geometry snapshot; the nested execution must use the actual parent Frame and
restore its loop services and metrics. Finite default-zero metrics cannot pass.

The standalone derivative results retain all original run metadata. Public
energy-only and stress-only calls preserve unrequested force/virial fields;
public stress publishes the extensive Frame virial as σΩ. The nested combined
call must agree with standalone forces and virial within 2e-8 eV/Å and 1e-7 eV,
respectively. It is collected exactly once after the child returns. The parent
uses the existing whole-frame replica policy. `--mpi` explicitly initializes and
finalizes MPI; capture each MPI rank's stdout independently with the site's MPI
launcher. Running without `--mpi` keeps the serial behavior. MPI and final-library
qualification require their own native receipts.

The fixture is the locked slightly distorted diamond primitive Si2, GTH-PBE-q4
norm-conserving potential, 12 Ry wave cutoff, Γ, eight electrons/four occupied
restricted bands, Slater exchange plus VWN5 correlation, `tol=1e-10`,
`max_iter=200`, and `mix_beta=.2`. Native density/wave spheres contain 1393/171
full-complex modes; the density FFT is 16×18×16 and LSDA quadrature 24×28×24.
The cutoff is a finite-basis validation fixture, not a converged materials result.

The independent [QE reference](qe-2026-10-28/reference.json) includes complete
successful [input](qe-2026-10-28/successful/input.in),
[raw output](qe-2026-10-28/successful/qe.out), Hartree XML, binary/potential
hashes, environment and primary source hashes. QE7.6 actually prints
`SLA+VWN+NOGX+NOGC`, and converges in 63 steps at 1e-12 Ry. Official correlation
index 2 calls the VWN form with the same unpolarized parameters as CPU VWN5;
PZ and PW are distinct models. A separate fixed-charge six-band NSCF calculation
shows a 0.926044991745512 eV gap. It introduces no second SCF into the public
method or this executable. `pins.h` stores the raw XML values, independently
checked against printed energy, both total force rows and all stress entries.

Every available public and nested quantity must meet the unchanged external
gates: energy 1e-5 Ha, each force component 1e-4 Ha/Bohr, each of the six stress
components 2e-3 eV/Å³. Stress is compressive positive, ordered xx,yy,zz,xy,xz,yz.
The first proposed dimer-like Si2 geometry failed 250-step SCF convergence;
its [raw output](qe-2026-10-28/failed-dimer/qe.out) and explicit failure status
remain separate and are never counted as successful reference evidence.

Prior independent mathematical validation is retained in
[math provenance](qe-2026-10-28/math-provenance/validation-3-receipt.json): three
SCF tolerances, component derivatives, and 72 fixed-Miller full-SCF five-point
strain samples at three step sizes, plus 12 fresh-cutoff negative controls.
The [basis audit](qe-2026-10-28/math-provenance/basis-order-audit.json) verifies
ordered Miller labels and FFT sizes. Those receipts retain their own original
source/binary hashes, including the older cold helper; they are not relabeled
as executions of this new public integration. This executable does not repeat
the 72-SCF study or replace those independent mathematical oracles.

This target's scope is Γ/eight-k RKS fixed LSDA. Its public/nested
qualification must be run against the final library; compiling the source or
passing earlier scratch probes does not constitute that acceptance.

`pw-analytic-stress-general` extends public/nested validation to the six
existing actual-QE PBEsol/revPBE/RPBE × Si2 RKS/H3 UKS references with nonzero
Fermi–Dirac entropy. It reuses their original pins and E/full-F/six-stress
gates. Standalone stress and nested combined calls must each perform one
SCF, one all-spin final-band sweep, one tensor assembly and zero calls to the
legacy strained-band path. GNU link wrapping counts the real internal calls;
there are no counters or diagnostic callbacks in production code.

`pw-analytic-stress-family INDEX shadow` covers the original 16 PBE-family ×
RKS/UKS × fixed/FD hydrogen settings (`INDEX` from 0 through 15). `shadow`
selects the expected closed-form call budget in this diagnostic; it does not
change method dispatch. The preserved numerical-library runs provide the
comparison with the original 5e-4 eV/Å³ stress gate. Gaussian/MP, multiple k
points outside restricted fixed LSDA and fixed UKS with an empty spin channel retain their earlier public
path. The final-library runs and immutable isolated-candidate results must
retain separate source and binary attribution.


`collect.py` performs read-only validation of the native log, all four SCF audit
records, the two complete force rows and six stress entries, the offline QE
XML/input/raw/XC and complete SHA manifest, exact static pin tokens, equal source
inventories, and the supplied binary/hardware evidence. It never launches a
scientific calculation. Write its scoped report to a separate output path:

```sh
python3 tools/diagnostics/pw-analytic-stress/collect.py \
  --log /path/to/native/main.log \
  --source-before /path/to/native/source-before.json \
  --source-after /path/to/native/source-after.json \
  --binary build/bin/pw-analytic-stress \
  --hardware /path/to/native/hardware.txt \
  --binaries-sha256 /path/to/native/binaries.sha256 \
  --output /path/to/collected/validation.json
```

The binary SHA inventory argument is optional; when supplied its matching entry
must equal the executable's actual SHA. The report fixes `formal_source`,
`formal_build` and `formal_source_build` to false: this is scoped physics/Driver
collection, not the repository-wide formal-source/build gate. Collector code is
hashed separately, so adding it does not retrospectively change which source
built an earlier numerical executable. Old mathematical receipts remain
separately attributed to their original hashes.

The new `RKS_fixed_LSDA_Si2` PASS record carries `mesh=1|2` and
`rotated=0|1`. The collector also accepts the original `GammaRKS` record and
the explicitly named isolated `K8RKS` records. It selects the matching actual
reference automatically. The [eight-k raw package](qe-k8-2026-10-28/reference.json)
contains the executed QE7.6 input, output, XML, PP, binary/environment evidence
and complete manifest: scientific convergence in 32 steps, printed Slater/VWN,
and eight explicit crystal points. XML geometric weights sum to one; QE's
printed/output RKS band weights sum to two because they include the spin factor.
The checker verifies both conventions. Primary QE source and unit provenance
are reused from the unchanged Γ package and checked through its full manifest.

`--rotate-z90` rotates the locked cell and atoms exactly. The reference uses
`F' = (-Fy,Fx,Fz)` and `sigma' = (yy,xx,zz,-xy,-yz,xz)`; no additional rotated
QE run is claimed. External gates remain 1e-5 Ha, every force 1e-4 Ha/Bohr and
every stress 2e-3 eV/Å³ for all mesh/rotation choices.

When present, the four `BAND_AUDIT` records must match the SCF labels and verify
`Davidson = local_k × (iterations + derivative_request)`, one collective
extractor per derivative request, one tensor assembly per stress request, and
zero legacy strained solves. The global count must equal `mesh³`. Old logs
without these records keep their original SCF/physics scope and are explicitly
not upgraded to measured band budgets. Supply both `--rank` and `--ranks` for
each MPI log, including a Γ empty rank, whose local Davidson count is zero:

```sh
python3 tools/diagnostics/pw-analytic-stress/collect.py \
  --log /path/to/native/rank1.log --rank 1 --ranks 2 \
  --source-before /path/to/native/source-before.json \
  --source-after /path/to/native/source-after.json \
  --binary build/bin/pw-analytic-stress \
  --hardware /path/to/native/hardware.txt \
  --binaries-sha256 /path/to/native/binaries.sha256 \
  --output /path/to/collected/rank1-validation.json
```

Collect every rank separately. Concatenated logs fail the unique PASS and four
ordered audit requirements; the collector never deduplicates them into a
synthetic single execution. Keep per-rank reports alongside the complete raw
rank inventory and process exit statuses.


`pw-multi-k-fd [INDEX] [--mpi]` checks ten matched-grid actual-QE FD
references: LSDA/PBE/PBEsol/revPBE/RPBE × Si2 RKS/nonzero-moment H3 UKS
on the full eight-k mesh. Omit INDEX to run all ten; MPI logs remain per rank.
The expected local budget is local_k*spin_channels*(iterations+1). The new
observer checks actual SCF energy/residual/delta energy, one stage/criterion
lifecycle, reference metadata and parent context, plus exactly one E/F/W
contribution on top of nonzero parent sentinels. `rawrefs/manifest.json` and
`export_multi_k_fd_pins.py --check` bind the independent data and static pins.
Earlier coarse-grid/near-zero-moment controls retain their own narrower scope.

`pw-analytic-stress-cuda-family MODEL MODE` runs the shared CUDA diagnostic
adapter with one of pbe/pbesol/revpbe/rpbe. MODE is one of --force, --stress,
--energy, --nested, --nested-force, --nested-stress, --nested-energy,
--exhausted or --retry. The original LSDA executable uses the same adapter.
All three diagnostic C++ targets treat missing returns as build errors;
production owns no diagnostic sink. The shared owner admits Gamma RKS fixed or Fermi–Dirac filling.
Its raw gradients and fluxes use distinct owned
buffers on the actual GGA quadrature; force-only creates no derivative GGA
workspace. Requested host output transfers remain separate from orbital and
density residency and from the external oracle's readback.


`pw-gaussian-ensemble [INDEX] [--mpi]` reuses the FD fixture, observer and
execution harness for twelve matched actual-QE primary cases: two Gamma UKS
LSDA/PBE cases and ten full-eight-k RKS/UKS semilocal cases. Geometry, filling
width and FFT quadrature belong to each reference recipe. `--control` selects
the separate original PBE UKS .02-Ry near-integer case: it intentionally fails
the unchanged entropy qualification gate. The independent .04-Ry case is a
primary reference; it does not replace that failed control. Raw data and exact
pin checks live in `gaussian-rawrefs/` and `export_gaussian_pins.py --check`.

`pw-analytic-stress-cuda-fd MODEL MODE` runs the same CUDA adapter for
lsda/pbe/pbesol/revpbe/rpbe with six bands and .02-Ry FD filling. Five matched
QE inputs/outputs/XML records live beside `qe-gamma-rks-fd-2026-10-28/pins.h`.
Final lambda/mu/f/S and direct C/f density are checked independently. Full
reference SCF metadata is snapshotted before the derivative call and preserved
on successful and exceptional exits. Private candidate evidence remains
separate from acceptance of the current shared-library executables.


`pw-fixed-k-ensemble [INDEX] [--mpi]` uses the same observer/fixture/execution
harness for five semilocal models times RKS/populated UKS on full eight-k
meshes. Fixed occupations constrain band indices per k point; a positive
global insulating gap is not assumed. The accepted optional electron-count
field remains zero. Actual sum(w*f), orbital-density G0 charge, spin counts and
zero entropy are independently observed. Raw references and an independent
negative indirect-gap control are in `fixed-k-rawrefs/`;
`export_fixed_k_pins.py --check` verifies original raw hashes and exact pins.

`test_cuda_pw_frozen_spin_xc_virial` is a no-SCF, no-band component GTest
with five LSDA/PBE-family controls. Strictly positive spin fields and nonzero
negative gradient cross terms exercise all nine strain lanes at two steps,
rotation, production/independent primals, CPU tensor and point flux identities.
The integrated operator borrows its stream and scratch. This does not admit
public CUDA UKS total tensors or extend the regulated spin-density domain.
