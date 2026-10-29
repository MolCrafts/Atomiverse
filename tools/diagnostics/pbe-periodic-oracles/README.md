# Actual periodic PBE-family oracles

These opt-in diagnostics compare public CPU/CUDA methods with outputs from
independent installed QE 7.6 and CP2K 2026.2 binaries. The reference values are
parsed from real stdout, never generated from Atomiverse. Each case preserves
input, stdout, input/output SHA256, engine-binary SHA256, pseudopotential/basis
SHA256, actual printed XC labels, exit status, SCF acceptance, execution host,
and allocation/step identity in its `reference.json` and `execution.json`.

## PW / QE

`qe-2026-10-27/reference.json` contains six cases: Si2 RKS and H3 UKS, each with
PBEsol, revPBE and RPBE. All cases are periodic Γ, use the existing GTH-PBE
pseudopotentials unchanged across XC variants, and have Fermi–Dirac width
0.02 Ry. This compares the same Hamiltonian across programs; it does not claim
pseudopotential transferability for other materials.

Si2 uses the existing `nc-extension/si2.in` geometry and cutoffs (12/48 Ry,
40³ QE grid, 6 bands). H3 uses the existing `pw-spin-smear-stress` skew-cell
geometry (6/24 Ry, 72³ QE grid, 5 bands, initial magnetization 1/3). Original
public fixtures and hashes are recorded in the capture manifest. QE's `input_dft`
overrides the XC stated in the pseudopotential, and the actual printed XC label
must match the requested variant.

The compared energy is the finite-temperature Mermin free energy including the
printed entropy contribution, in Ha (QE Ry divided by two). Every Cartesian
ionic-force component is checked in Ha/Bohr. All six physical stress components
are checked, using compression-positive σ = −∂E/∂ε / Ω. QE's stress tensor is
printed in Ry/Bohr³, converted by division by two. Public `kStress` is extensive
virial σΩ in eV; returned physical stress is in eV/Å³.

Predeclared gates retain the existing QE public-product gates: 1e−5 Ha energy,
1e−4 Ha/Bohr per force lane, and 2e−3 eV/Å³ per stress lane. The public PW energy
publication deliberately uses its documented QE Rydberg-to-eV conversion, while
force/stress use their core-unit conversion. Mixing the energy constants creates
a false sub-micro-eV publication failure; the original failed native receipt is
retained under `.molq/pbe-periodic-oracles/cpu-4522778/`.

`pbe-periodic-oracles [--backend cpu|cuda|both]` emits six cases per requested
backend, with `PASS PBE_PERIODIC_QE <id> <backend>` and errors. The default on a
CUDA build runs both. `collect.py` belongs to the parent workflow and records
native-run evidence; capture success alone is not a native scientific pass.

## GPW / CP2K

`cp2k-2026-10-27/reference.json` contains H2 RKS PBEsol/revPBE/RPBE. The fixture
is exactly the public `pbe-family/gpw.cpp` periodic cell (12,11,10 Bohr), tilted
H–H geometry, DZV-GTH-PADE basis, GTH-PBE-q1, 80 Ry density cutoff, one grid,
and local pseudopotential on the grid (`CORE_PPL GRID`). CP2K uses named LibXC
exchange/correlation pairs so PBEsol/revPBE/RPBE parameters are explicit:

| Variant | Exchange | Correlation |
| --- | --- | --- |
| PBEsol | GGA_X_PBE_SOL | GGA_C_PBE_SOL |
| revPBE | GGA_X_PBE_R | GGA_C_PBE |
| RPBE | GGA_X_RPBE | GGA_C_PBE |

CP2K 2026.2 aborts analytical stress for `CORE_PPL GRID` with
`Virial not debuged for CORE_PPL`. The failed real inputs/stdout remain under
`attempts/cp2k-core-ppl-analytic-stress-abort/`. The accepted capture instead
uses CP2K's own `STRESS_TENSOR NUMERICAL`, performing independently converged
strained calculations without changing the Hamiltonian. Its output prints
compression-positive stress in bar, converted to GPa by 1e−4. Full external
stress is preserved; the public orthorhombic GPW method supports only diagonal
xx/yy/zz, so only these three lanes qualify. No GPW shear support is claimed.
Integer occupations make the GPW total energy a zero-temperature energy.

`pbe-periodic-cp2k-oracles [--backend cpu|cuda|both]` uses the fixed existing GPW
product gates: 5e−6 Ha energy, 5e−5 Ha/Bohr per force lane, and 5e−5 Ha/Bohr³ per
supported diagonal stress lane. It measures every lane and every variant before
returning failure, retaining the gaps for diagnosis. The original PBEsol
CP2K energy is −1.158385101258297 Ha; the previous native energy
−1.15839043956 Ha differs by almost exactly CP2K's printed Gaussian-core
overlap energy +5.33892824e−6 Ha. This is evidence of a production physics gap,
not a reason to relax a gate. Native qualification requires a separately
reviewed correction and fresh measured comparisons.

## Reproduction and provenance

External captures use one fresh `srun` step per engine invocation within the
existing persistent allocation, with four OpenMP threads and single-threaded
BLAS. Reusing a single PMI connection across separate MPI executables fails;
both original bootstrap failures are retained under `attempts/`. No extra
Slurm allocation is submitted by these scripts.

```
python3 tools/diagnostics/pbe-periodic-oracles/capture_qe.py /path/to/pw.x \
  --allocation --out /new/reference-directory
python3 tools/diagnostics/pbe-periodic-oracles/capture_cp2k.py \
  --out /new/reference-directory
python3 tools/diagnostics/pbe-periodic-oracles/export_pins.py \
  /path/to/qe/reference.json /new/pins.h
python3 tools/diagnostics/pbe-periodic-oracles/export_cp2k_pins.py \
  /path/to/cp2k/reference.json /new/cp2k_pins.h
```

The installed engine/data paths are recorded in the manifests. Exporters accept
only exact expected case sets and verify raw-file hashes. Keep original failed
receipts even after a later correction passes. Generated wavefunction/restart
and scratch files are excluded; raw stdout is intentionally included.

Primary implementation references: [QE 7.6 XC dispatch](https://github.com/QEF/q-e/blob/qe-7.6/Modules/funct.f90),
[CP2K XC input](https://github.com/cp2k/cp2k/blob/master/src/input_cp2k_xc.F),
and [CP2K XC-grid documentation](https://manual.cp2k.org/cp2k-2024_2-branch/CP2K_INPUT/FORCE_EVAL/DFT/XC/XC_GRID.html).

## Γ UKS H3: four independently captured models

`cp2k-uks-2026-10-28/reference.json` captures actual CP2K PBE, PBEsol,
revPBE and RPBE with three H atoms, integer Nα=2/Nβ=1, the same
DZV-GTH-PADE/GTH-PBE-q1 basis/potential, and an orthorhombic periodic
12×11×10 Bohr cell. The asymmetric tilted positions reuse the PW H3 geometry
in Å; their explicit Bohr conversions are recorded in the manifest. The single
80 Ry grid is 36×32×30. Each external numerical-stress calculation has twenty
converged SCFs and independently printed spin counts 2/1 throughout.

`pbe-periodic-uks-oracles [--backend cpu|cuda|both]` compares absolute energy,
all nine ionic-force components and only the three supported diagonal stress
components at the original 5e−6 Ha, 5e−5 Ha/Bohr and 5e−5 Ha/Bohr³ gates.
It checks electron counts from the public P/Pβ/S payload via Tr(PαS)=2 and
Tr(PβS)=1; optional Mulliken populations are checked when supplied. Register
this separate opt-in target through `register-uks.cmake`. The first scoped CPU
run passed all four external references; CUDA qualification requires its own
native build and measured results.

This fixture does **not** qualify a raw negative-spin-density extension or the
empty-beta endpoint. `cpu-raw-xc-domain-probe/` retains the actual current CPU
point probe and a `passed:false` evidence record: at α=.2, β=−.01, the returned
PBE beta potential is about +99.4215 while the stable central derivative of its
raw energy is about −.0335385. The empty-beta positive-side finite-step sweep
also crosses distinct floor/clip scales. These are separate derivative-domain
gaps, not reasons to change inputs or loosen a gate. CPU production XC remains
unchanged; piecewise derivative proposals are kept outside production.
