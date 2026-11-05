# Examples

Each example shows one concept of the public API and runs as a normal user
program. Examples demonstrate usage only: speed and accuracy comparisons
against other engines live in [`benchmarks/`](../benchmarks/README.md), and
pinned scientific checks live in `regressions/`.

| Directory | Audience | How it runs |
|-----------|----------|-------------|
| [`cpp/`](cpp/) | C++ API, numbered in reading order | `-DATV_BUILD_EXAMPLES=ON`, binaries in `<build>/examples/` |
| [`python/`](python/) | Python API (`import atomiverse`) | `python examples/python/<script>.py` |
| [`cli/`](cli/) | `atomiverse -in` input decks | `atomiverse -in examples/cli/in.lj` |
| [`extensions/`](extensions/) | Out-of-tree projects against an installed Atomiverse | their own CMake project (`find_package(Atomiverse)`) |

## C++ (`cpp/`)

| # | Example | Concept | Needs (`ATV_BUILD_*`) |
|---|---------|---------|------------------------|
| 01 | [`01_rhf.cpp`](cpp/01_rhf.cpp) | A method object and the Frame: RHF single point on H₂ | HF |
| 02 | [`02_open_shell.cpp`](cpp/02_open_shell.cpp) | α/β electron counts: UHF and UKS-LSDA on the H atom | HF |
| 03 | [`03_ionization.cpp`](cpp/03_ionization.cpp) | Koopmans vs ΔSCF ionization energy of water | HF |
| 04 | [`04_correlation.cpp`](cpp/04_correlation.cpp) | Post-HF wrappers: `MP2<RHF>` and `FCI<RHF>` on H₂ | HF |
| 05 | [`05_density_cube.cpp`](cpp/05_density_cube.cpp) | Result handles and recorders: density/orbital cube files | HF, EVALUATE |
| 06 | [`06_geometry_optimization.cpp`](cpp/06_geometry_optimization.cpp) | `Minimize` + L-BFGS + a convergence criterion (aspirin, RHF) | HF, MINIMIZER |
| 07 | [`07_aimd.cpp`](cpp/07_aimd.cpp) | Born–Oppenheimer MD: `MD` with an SCF force provider | HF, MD |
| 08 | [`08_md_lj.cpp`](cpp/08_md_lj.cpp) | Classical MD: neighbor list + pair potential, NVE argon | MD, POTENTIALS, NEIGHBOR |
| 09 | [`09_pw.cpp`](cpp/09_pw.cpp) | Plane-wave DFT for a periodic crystal (silicon) | PW, EVALUATE |
| 10 | [`10_gpw.cpp`](cpp/10_gpw.cpp) | Gaussian-and-plane-waves DFT (water) | GPW, EVALUATE |
| 11 | [`11_am1.cpp`](cpp/11_am1.cpp) | Semiempirical AM1: energy, ΔHf, forces, minimization | SEMIEMPIRICAL, MINIMIZER, EVALUATE |

`cpp/molecules.h` builds the small molecules the examples share. Build one
example with `cmake --build build --target 01_rhf` and run
`./build/examples/01_rhf`.

## Python (`python/`)

| Script | Concept |
|--------|---------|
| [`md_lj_argon.py`](python/md_lj_argon.py) | Classical MD from Python (same system as `cli/in.lj`) |
| [`minimize_lj_dimer.py`](python/minimize_lj_dimer.py) | Geometry minimization from Python |
| [`am1_ch4.py`](python/am1_ch4.py) | AM1 single point and Mulliken charges |
| [`gaff_parameterize.py`](python/gaff_parameterize.py) | GAFF2 typing with molpy + AM1-BCC charges from Atomiverse |

## Extensions (`extensions/`)

- [`xc/`](extensions/xc/README.md): a user-defined exchange-correlation
  functional plugged into RKS/UKS.
- [`nc/`](extensions/nc/README.md): a user-defined norm-conserving
  pseudopotential plugged into plane-wave DFT.

Copy one outside the checkout and build it against an installed Atomiverse.

## Adding an example

1. One concept per file; take the next number in reading order.
2. Gate the target in `cpp/CMakeLists.txt` on every module the source uses.
3. No comparisons against other engines and no recorded results here.
4. Add a row to the table above and link it from the tutorial chapter that
   explains the concept.
