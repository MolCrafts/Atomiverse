# Atomiverse

Atomiverse is a next-generation engine for multiscale, multiphysics atomistic
simulation. Dual backends (CPU + CUDA) share one **Driver / Component** API
across classical MD, Monte Carlo, geometry optimization, vibration, and
electronic structure (Gaussian SCF, plane-wave DFT, GPW, NDDO/AM1, DFTB).

Atomiverse is distributed under the **GNU General Public License version 2
(GPL-2.0)** — the same license as [LAMMPS](https://www.lammps.org) — see
[LICENSE](LICENSE). Parallel-decomposition machinery is deliberately ported
from LAMMPS (license-compatible by construction); ported files carry a
provenance header naming the originating LAMMPS source file and retain the
Sandia copyright notice.

## What it computes

- **Classical molecular dynamics** — Lennard-Jones, Morse, Buckingham,
  Born-Mayer-Huggins, Yukawa, DPD and bonded force fields; Ewald/PME/PPPM
  electrostatics; NVE/NVT/NPT/NPH with Nose-Hoover, Parrinello-Rahman, Langevin,
  CSVR and Berendsen; RATTLE/SETTLE constraints and general virtual sites;
  EAM and Stillinger-Weber many-body potentials.
- **Gaussian-basis electronic structure** — RHF, UHF, ROHF, GHF, RKS, UKS, ROKS
  with analytic gradients; CPHF Hessians and polarizabilities; MP2 through MP4,
  CCSD, CCSD(T), CASCI/CASSCF, NEVPT2; TDHF/TDA/full LR-TDDFT, spin-flip,
  ADC(2), EOM-CC, G0W0, RPA, BSE and GF2; implicit solvent (C-PCM / COSMO /
  IEF-PCM), ECPs, RI-J/RI-JK, CFMM and x2c-1e.
- **Plane-wave DFT** — norm-conserving pseudopotentials, LDA and GGA, collinear
  spin, k-points, metallic smearing, analytic forces and Nielsen-Martin stress,
  bands/DOS, Wannier functions, Berry-phase polarization and DFPT phonons.
- **Gaussian-and-plane-waves (GPW) DFT** — RKS and UKS at Gamma and on a
  k-mesh, diagonalization or orbital transformation, DFT+U, metallic smearing,
  a large dispersion/vdW set (D3, D3-BJ, D4, TS, MBD, XDM, vdW-DF, rVV10,
  optB88-vdW), analytic forces and stress, CELL_OPT, and periodic MP2/RPA/GW.
- **Semiempirical and tight binding** — NDDO (AM1, RM1), DFTB0/2/3 on mio-1-1
  and 3ob-3-1 tables with analytic forces and stress, TD-DFTB, and the
  extended tight-binding family GFN0/GFN1/GFN2/IPEA-xTB.
- **Ab-initio molecular dynamics** — Born-Oppenheimer MD, Car-Parrinello,
  second-generation CPMD and XL-BOMD, over any of the electronic methods above.
- **Sampling, paths and free energy** — Monte Carlo (NVT/NPT/uVT, Widom),
  umbrella sampling, well-tempered metadynamics, WHAM, thermodynamic
  integration, REMD and H-REMD, NEB, dimer, string, IRC and PIMD.
- **Electrochemistry** — constant-potential and constant-charge electrodes,
  MaZe charge dynamics, Thomas-Fermi metallicity and the electrode observers.
- **Multiscale and properties** — QM/MM (mechanical, electrostatic and
  polarizable, with H link atoms), population and ESP charges, vibrational
  analysis and thermochemistry, RDF/MSD/VACF and density cubes.

Not every method runs on both backends. The
[roadmap](docs/roadmap/index.md) states, method by method, what is
implemented on CPU, what is implemented on CUDA, and what is not implemented
at all; every claim there is backed by a case under `regressions/`.

Everything in this README and in `docs/` targets the **C++ API**. For the
Python package (pure `pip install` + on-demand native build via
`python -m atomiverse install`), see
[`docs/user-guide/python.md`](docs/user-guide/python.md).

## Quickstart: build and run your first calculation

These commands are verified on the project's CPU login node. The one
environment subtlety — loading the toolchain module and mirroring its runtime
library path — is explained in
[`docs/user-guide/getting-started.md`](docs/user-guide/getting-started.md);
if you are on a generic Linux box with GCC + OpenBLAS + CMake on `PATH`, drop
the `module load` / `LD_LIBRARY_PATH` lines.

```bash
# 1. Configure a CPU build with the Gaussian-SCF methods and the examples.
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DATV_BUILD_CUDA=OFF -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON

# 2. Build the first real-chemistry example.
cmake --build build --target 02_rhf -j

# 3. Run it.
./build/examples/02_rhf
```

You should see the restricted Hartree-Fock energy of H₂ converge to
**E_total ≈ -1.1168 Hartree**, matching the textbook value (Szabo & Ostlund,
Table 3.7). That is a complete electronic-structure calculation from atomic
coordinates to total energy.

## Documentation map

Read it in this order:

1. **[User guide](docs/user-guide/index.md)** — start at
   [Getting Started](docs/user-guide/getting-started.md): what to install, how
   to build, and how to run and read your first example.
2. **[Tutorials](docs/tutorial/index.md)** — a short book. Each chapter teaches
   one method: the theory behind it, the Atomiverse API that implements it, and
   the runnable example you should keep open beside the text.
3. **[API reference](docs/api/index.md)** — the generated Doxygen reference.
4. **[Developer manual](docs/developer/index.md)** — architecture and how to
   add new components (only needed if you are extending the code).

The docs site is built with [Zensical](https://zensical.org); `zensical.toml` at
the repository root holds the configuration and navigation:

```bash
uv run --with zensical --with molcrafts-zensical-theme zensical serve   # live preview
uv run --with zensical --with molcrafts-zensical-theme zensical build   # → site/
```

The C++ API reference is generated separately by Doxygen (output in
`build/doxygen/html/`) and is not yet part of the site:

```bash
cmake -B build/docs -DATV_DOCS_ONLY=ON
cmake --build build/docs --target docs
```

## Example programs

The `examples/` directory is a flat, numbered sequence — one concept per file,
read in order. Each tutorial chapter is written around the matching example.

| # | File | Method | Build flag |
|---|------|--------|------------|
| 01 | `examples/01_scf.cpp` | The bare SCF loop (no chemistry) | — |
| 02 | `examples/02_rhf.cpp` | Restricted Hartree-Fock (H₂) | `ATV_BUILD_HF` (CUDA twin `02_rhf.cu`: `ATV_BUILD_CUDA`) |
| 03 | `examples/03_uhf.cpp` | Unrestricted Hartree-Fock | `ATV_BUILD_HF` |
| 04 | `examples/04_uks.cpp` | Kohn-Sham DFT (UKS-LSDA) | `ATV_BUILD_HF` |
| 05 | `examples/05_geom_opt.cpp` | Geometry optimization (L-BFGS + DFT) | `ATV_BUILD_HF` |
| 06 | `examples/06_aimd.cpp` | Ab-initio MD (NVT) | `ATV_BUILD_HF` |
| 07 | `examples/07_md_lj.cpp` | Classical MD (argon, LJ) | — (CUDA twin `07_md_lj.cu`: `ATV_BUILD_CUDA`) |
| 08 | `examples/08_pw.cpp` | Plane-wave DFT (silicon) | `ATV_BUILD_PW` |
| 09 | `examples/09_gpw.cpp` | GPW DFT (water) | `ATV_BUILD_GPW` (needs `ATV_BUILD_PW`) |
| 11 | `examples/11_koopmans.cpp` | Koopmans IP/EA | `ATV_BUILD_HF` |
| 12 | `examples/12_delta_scf.cpp` | ΔSCF excited state | `ATV_BUILD_HF` |
| 13 | `examples/13_mp2.cpp` | MP2 correlation | `ATV_BUILD_HF` |
| 14 | `examples/14_fci.cpp` | Full CI | `ATV_BUILD_HF` |
| 15 | `examples/15_am1.cpp` | AM1 semiempirical | `ATV_BUILD_SEMIEMPIRICAL` |
| 16 | `examples/16_density_cube.cpp` | Density cube output | `ATV_BUILD_HF` |

See [`examples/README.md`](examples/README.md) for the full table and the rules
for adding new examples.

## Requirements

C++20, CMake ≥ 3.18, a C++ compiler, BLAS/LAPACK (OpenBLAS). The Gaussian-SCF
methods need `-march=native`. Plane-wave DFT (`ATV_BUILD_PW`) needs FFTW3. The
CUDA backend needs the CUDA toolkit and a GPU. A Rust toolchain is required to
build the molrs I/O dependency. molrs comes from one of two places:

- **CI and plain `cmake -B build`** use the git submodule at `./molrs` (after
  clone run `git submodule update --init --recursive molrs`); the gitlink pins
  the newest *committed* molrs `dev` commit Atomiverse was verified against.
- **Developer builds** (`cmake --preset cpu-unit` and every other preset in
  `CMakePresets.json`) set `ATV_MOLRS_PATH=${sourceDir}/../molrs` — the
  sibling MolCrafts checkout of molrs `dev`, whose **uncommitted** working
  tree (the streaming `*.mrec` trajectory writer behind `cpu::MrecRecorder`)
  no gitlink can capture. Clone `MolCrafts/molrs` next to this repo on `dev`.

Either source is held to the same CXX API contract
(`cmake/MolrsContract.cmake`); surfaces that only the dev tree has (the
`TrajectoryWriter` bridge) are detected per pin and compiled out otherwise.
xtensor is fetched automatically.
