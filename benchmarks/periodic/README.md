# Periodic DFT acceptance

`pw` and `gpw` are methods in the scientific benchmark framework. Their
required participants are Atomiverse CPU and a live independent executable.
Both methods define a CUDA participant, required with `--require-cuda`.
The smoke tier runs the two restricted PBEsol cases. Acceptance adds the
remaining PBE-family cases, mixed PW H₂O, two GPW H₂ geometry cases, GPW H₂
multigrid, two GPW Si₄ k-point cases, and two GPW H₃ k-point cases. Each
participant runs one fresh cold calculation and at least one fresh warm
calculation per case.

| Method | Scope | Energy | Every force component | Stress |
| --- | --- | ---: | ---: | ---: |
| PW / QE 7.6 | Six Γ Si₂ RKS and skew H₃ UKS cases, 0.02 Ry Fermi–Dirac, plus skew Γ H₂O at fixed PBE occupations | 1e-5 Ha, including −TS for smeared cases | 1e-4 Ha/Bohr | 2e-3 eV/Å³, six Voigt components |
| GPW / CP2K 2026.2 | Nine Γ single-grid cases, Γ multigrid H₂, two-k-point Si₄ RKS single-grid and multigrid, and two-k-point H₃ UKS single-grid on CPU and CUDA. Two-k-point H₃ UKS multigrid runs on CPU. All use integer occupations. | 5e-6 Ha | 5e-5 Ha/Bohr | 5e-5 Ha/Bohr³, xx/yy/zz |

The original QE decks, GTH files and CP2K decks are under the
[PW reference](../../regressions/pw/pbe-family/REFERENCE.md) and
[GPW reference](../../regressions/gpw/pbe-family/REFERENCE.md). Both
participants use the geometry in those decks. CP2K reads the release's
basis data (`BASIS_SET` or `BASIS_MOLOPT`) and `GTH_POTENTIALS`. The adapter rewrites only their paths
for an isolated scratch directory. CP2K independently converges its numerical
strain calculations. The public GPW method supplies three diagonal stresses.
PW supplies xx, yy, zz, xy, xz and yz, with compression positive on both sides.
The [H₂O QE deck](fixtures/h2o-rks-pbe.in) lists O before H while the
coordinates are ordered H–O–H. QE reads the original O ONCV-PBE NC-UPF
and the archived H GTH-PBE-q1 file. Atomiverse uses the same UPF and
its published H GTH-PBE-q1 parameters. The row checks all nine force
components and all six stresses with the original PW tolerances.

The GPW periodic-image H₂ case reuses the archived 12 × 11 × 10 Bohr H₂
PBEsol deck. The native participant also evaluates the same system after
moving the first H by the lattice vector −a−b−c into a neighboring image.
The adapter compares the two native observations directly, then checks
both against the live CP2K reference at the standard GPW tolerances.
The geometry-reuse case starts with the first H displaced by −0.4 Bohr in
x. It then restores the reference coordinates on the same method and cell,
requires a change in each observable, and compares the restored geometry
with the live CP2K reference.
The multigrid H₂ case uses three grid levels at the same PBEsol geometry.
The native participant reads the cutoff and level controls from the same
deck. The CP2K adapter verifies that Gaussian products occupy multiple
levels before comparing total energy, every force component and the three
diagonal stresses.
The Si₄ cases displace one atom of the diamond BCT cell and use PBE with
DZVP-MOLOPT-GTH and GTH-PBE-q4. Atomiverse and CP2K use the full
Monkhorst–Pack 1 × 1 × 2 mesh. CP2K reports analytical stress for both cases.
The Si₄ multigrid case requires multiple populated CP2K grid levels.
The H₃ UKS cases use the same full mesh with two α and one β occupied
orbital at each k point. The single-grid row compares CPU and CUDA against
the same live CP2K reference for energy, every force component and xx/yy/zz
stress.

## Run locally

Configure a CPU Release build with HF, PW, GPW and benchmarks enabled, using
the project's usual molpy source and FFTW dependencies. Build the public C++
participant:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_CUDA=OFF \
  -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
  -DATV_BUILD_BENCHMARKS=ON
cmake --build build --target atv_periodic_dft --parallel 4
```

Use Python 3.12+ with NumPy and Pydantic, QE 7.6, and CP2K 2026.2 built with
LibXC. Set the executable locations if the programs are outside `PATH`.
The default native participant is `build/bin/atv_periodic_dft`. Override it
for another build directory.
`CP2K_DATA_DIR` defaults to `share/cp2k/data` in the executable's prefix.

```sh
export ATV_BENCH_PERIODIC="$PWD/build/bin/atv_periodic_dft"
export ATV_BENCH_PW_X=/path/to/qe-7.6/bin/pw.x
export ATV_BENCH_CP2K=/path/to/cp2k-2026.2/bin/cp2k.psmp
export CP2K_DATA_DIR=/path/to/cp2k-2026.2/share/cp2k/data

python -m benchmarks run --method pw --method gpw --tier acceptance \
  --threads 1 --warmup 0 --repeats 1 --timeout 600 --no-snapshot \
  --out build/periodic-acceptance
python -m benchmarks validate --input build/periodic-acceptance/results.json
```

To qualify GPW on a CUDA host, build the same target with
`ATV_BUILD_CUDA=ON` and `ATV_BUILD_GPW=ON`. Point both native participants
to the CUDA-enabled executable and require CUDA in the campaign:

```sh
export ATV_BENCH_PERIODIC="$PWD/build/bin/atv_periodic_dft"
export ATV_BENCH_PERIODIC_CUDA="$ATV_BENCH_PERIODIC"
python -m benchmarks run --method gpw --tier acceptance --require-cuda \
  --threads 4 --warmup 0 --repeats 1 --timeout 600 --no-snapshot \
  --out build/periodic-gpw-cuda
```

Use a fresh output directory for another run. The framework keeps engine logs,
observations, component comparisons and the report in that directory.
The observations record executable and input hashes, and hashes for the
pseudopotential and basis data. Missing executables, failed SCFs and incomplete
outputs fail required cells. Raw timing is reported without speedup ratios
because the engines perform different derivative work.
