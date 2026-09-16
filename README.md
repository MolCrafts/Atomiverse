# Atomiverse

Atomiverse is an engine for multiscale, multiphysics atomistic simulation.
Classical molecular dynamics, Monte Carlo, geometry optimization, vibrational
analysis and electronic structure are not separate codes bolted together: they
are one staged loop over one system object, so changing the physics of a run
means changing which energy source you attach, not rewriting the run.

Two backends — CPU and CUDA — sit behind that one API. C++20, with a Python
package on top.

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

## How far to trust it

A capability is listed above only when an end-to-end case under `regressions/`
proves it. Each case is an out-of-library consumer that links the public
headers and nothing else, runs in under a minute, and is checked against a
number some other code printed — PySCF, CP2K, Quantum ESPRESSO, DFTB+, xtb,
LAMMPS, MetalWalls — or against an analytic result. There are 564 of them.

**Coverage is uneven between the backends, and the CPU side is far ahead.**
The CUDA backend is a production classical-MD engine and a much narrower
electronic-structure one: `cuda::scf::RHF` is restricted Hartree-Fock only,
over STO-3G s/p shells, and it throws rather than falling back when you leave
that envelope. [`docs/roadmap/`](docs/roadmap/index.md) states this method by
method — what runs on CPU, what runs on CUDA, and what is not implemented at
all. Read it before planning work around a method.

## Quickstart

```bash
git clone --recursive <repo-url> && cd Atomiverse
# or, in an existing clone:  git submodule update --init --recursive molrs

cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DATV_BUILD_CUDA=OFF -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON
cmake --build build --target 02_rhf -j
./build/examples/02_rhf
```

That prints the restricted Hartree-Fock energy of H₂ at **E_total ≈ -1.1168
Hartree**, the textbook value (Szabo & Ostlund, Table 3.7) — a complete
electronic-structure calculation from coordinates to total energy.

On an HPC system with Lmod you must load the toolchain **and** mirror its
link-time path into the runtime path, or binaries fail on `libopenblas.so.0`;
[Getting Started](docs/user-guide/getting-started.md) gives the exact
incantation. On a machine with GCC, OpenBLAS and CMake already on `PATH`, the
block above is all you need.

Everything in this README targets the **C++ API**. The Python package
(`pip install`, then an on-demand native build via `python -m atomiverse
install`) is documented in [`docs/user-guide/python.md`](docs/user-guide/python.md).

## Documentation

| Read | For |
|---|---|
| [User guide](docs/user-guide/index.md) | Install, build, and the Driver / Component / Frame vocabulary every run uses |
| [Tutorials](docs/tutorial/index.md) | One method per chapter: the theory, the API that implements it, and the runnable example beside it |
| [Roadmap](docs/roadmap/index.md) | What is implemented, on which backend, with what still missing |
| [Developer manual](docs/developer/index.md) | Internals and how to extend the engine |
| [API reference](docs/api/index.md) | Map into the generated Doxygen reference |

The prose site is built with [Zensical](https://zensical.org) from
`zensical.toml`; the C++ reference is generated separately by Doxygen and is
not yet part of it.

```bash
uv run --with zensical --with molcrafts-zensical-theme zensical serve  # live preview
cmake -B build/docs -DATV_DOCS_ONLY=ON && cmake --build build/docs --target docs
```

## Repository layout

| Path | Contents |
|---|---|
| `include/atomiverse/`, `src/` | Mirrored header and implementation trees: `core/` (the CUDA-free contract both backends share), `cpu/`, `cuda/` |
| `tests/` | Unit tests only — one function or method each, the whole suite in seconds |
| `regressions/` | End-to-end cases, one directory per delivered capability, each citing its oracle |
| `benchmarks/` | Standing performance harnesses; never registered with ctest |
| `examples/` | The numbered tutorial programs |
| `docs/` | The manual |
| `app/`, `python/` | The `atomiverse` CLI and the nanobind wheel |
| `tools/` | Data generators, oracle pin scripts, batch-job templates |

## Example programs

`examples/` is a flat, numbered sequence — one concept per file, read in order.
Each tutorial chapter is written around the matching example; the table and the
rules for adding new ones are in [`examples/README.md`](examples/README.md).

| # | File | Method | Build flag |
|---|------|--------|------------|
| 01 | `01_scf.cpp` | The bare SCF loop (no chemistry) | — |
| 02 | `02_rhf.cpp` | Restricted Hartree-Fock (H₂) | `ATV_BUILD_HF` (CUDA twin `02_rhf.cu`) |
| 03 | `03_uhf.cpp` | Unrestricted Hartree-Fock | `ATV_BUILD_HF` |
| 04 | `04_uks.cpp` | Kohn-Sham DFT (UKS-LSDA) | `ATV_BUILD_HF` |
| 05 | `05_geom_opt.cpp` | Geometry optimization (L-BFGS + DFT) | `ATV_BUILD_HF` |
| 06 | `06_aimd.cpp` | Ab-initio MD (NVT) | `ATV_BUILD_HF` |
| 07 | `07_md_lj.cpp` | Classical MD (argon, LJ) | — (CUDA twin `07_md_lj.cu`) |
| 08 | `08_pw.cpp` | Plane-wave DFT (silicon) | `ATV_BUILD_PW` |
| 09 | `09_gpw.cpp` | GPW DFT (water) | `ATV_BUILD_GPW` |
| 11 | `11_koopmans.cpp` | Koopmans IP/EA | `ATV_BUILD_HF` |
| 12 | `12_delta_scf.cpp` | ΔSCF excited state | `ATV_BUILD_HF` |
| 13 | `13_mp2.cpp` | MP2 correlation | `ATV_BUILD_HF` |
| 14 | `14_fci.cpp` | Full CI | `ATV_BUILD_HF` |
| 15 | `15_am1.cpp` | AM1 semiempirical | `ATV_BUILD_SEMIEMPIRICAL` |
| 16 | `16_density_cube.cpp` | Density cube output | `ATV_BUILD_HF` |

## Requirements

C++20, CMake ≥ 3.18, BLAS/LAPACK (OpenBLAS), and a Rust toolchain for the
molrs I/O dependency. The Gaussian-SCF kernels need `-march=native`.
Plane-wave DFT (`ATV_BUILD_PW`) and reciprocal-space electrostatics
(`ATV_BUILD_KSPACE`) need FFTW3. The CUDA backend needs the CUDA toolkit and a
GPU. xtensor is fetched automatically.

Every method domain has its own `ATV_BUILD_<DOMAIN>` flag, and turning one off
excludes its sources, headers, tests and regressions rather than merely hiding
a symbol — so working on one method never means compiling the whole engine.

**molrs** is a hard dependency, vendored as a git submodule at `./molrs`; the
gitlink pins the commit this tree was verified against, and a CXX API version
gate (`cmake/MolrsContract.cmake`) rejects a mismatched one. Developers working
on both repositories at once can point `ATV_MOLRS_PATH` at a sibling checkout
instead — that is what the presets in `CMakePresets.json` do, and optional
bridge surfaces present only in that tree are detected per pin and compiled out
otherwise.

## License

**GPL-2.0-only** — the same license as [LAMMPS](https://www.lammps.org); see
[LICENSE](LICENSE).

That choice is deliberate rather than incidental. It makes Atomiverse
license-compatible with LAMMPS source, so the parallel machinery — domain
decomposition, halo and reverse communication, buffer pack/unpack, k-space FFT
decomposition — is **ported from LAMMPS rather than independently rederived**.
Every such file carries a provenance header naming the originating LAMMPS
source file and retains the Sandia copyright notice.
