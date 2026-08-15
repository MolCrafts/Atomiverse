# Atomiverse

Atomiverse is a next-generation engine for multiscale, multiphysics atomistic
simulation. Dual backends (CPU + CUDA) share one **Driver / Component** API
across classical MD, Monte Carlo, geometry optimization, vibration, and
electronic structure (Gaussian SCF, plane-wave DFT, GPW, NDDO/AM1, DFTB).

Methods include:

- **Classical molecular dynamics** — Lennard-Jones and bonded force fields, NVE/NVT/NPT.
- **Geometry optimization** — L-BFGS / conjugate-gradient on ab-initio forces.
- **Gaussian-basis SCF** — restricted/unrestricted Hartree-Fock (RHF/UHF) and
  Kohn-Sham DFT (UKS) for molecules.
- **Plane-wave DFT** — Kohn-Sham LDA for periodic crystals.
- **Gaussian-and-plane-waves (GPW) DFT** — Gaussian-basis Kohn-Sham with
  grid-based Hartree/XC and GTH pseudopotentials (experimental).
- **Semiempirical / tight-binding** — NDDO (AM1/RM1), SCC-DFTB2/3 (mio/3ob), TD-DFTB.
- **Ab-initio molecular dynamics** — the MD loop driven by SCF forces.

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
| 02 | `examples/02_rhf.cpp` | Restricted Hartree-Fock (H₂) | `ATV_BUILD_HF` |
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
build the molrs I/O dependency (git submodule at `./molrs`; after clone run
`git submodule update --init --recursive molrs`). xtensor is fetched
automatically.
