# Getting Started

From a clean checkout to a built library and a first runnable example.
This page is **install and run** only. For Driver / Component vocabulary see
[Core concepts](concepts.md); for theory and literature numbers see the
[Tutorial](../tutorial/).

Requires **C++20** and **CMake ≥ 3.18**.

## 1. Dependencies

| Need | Why |
|---|---|
| C++20 compiler + CMake ≥ 3.18 | build |
| BLAS/LAPACK (OpenBLAS) | dense linear algebra in SCF |
| Rust toolchain | molrs (I/O) is built via corrosion |
| FFTW3 *(optional)* | plane-wave DFT / k-space (`ATV_BUILD_PW`) |
| CUDA toolkit + GPU *(optional)* | CUDA backend (`ATV_BUILD_CUDA`) |

### CMake options you will use

| Option | Turns on |
|---|---|
| `ATV_BUILD_HF` | Gaussian SCF (RHF / UHF / UKS) |
| `ATV_BUILD_PW` | Plane-wave DFT (needs FFTW3) |
| `ATV_BUILD_GPW` | GPW DFT (needs RHF + PW) |
| `ATV_BUILD_CUDA` | CUDA backend |
| `ATV_BUILD_EXAMPLES` | example programs under `examples/` |
| `ATV_BUILD_TESTS` | unit test suite |

## 2. Toolchain notes (HPC / Lmod)

On a generic Linux box with `gcc`, OpenBLAS and CMake already on `PATH`, skip
to step 3.

On clusters that use **Lmod**, load modules in a **login shell** and mirror
link-time paths into the runtime path — otherwise binaries fail with
`libopenblas.so.0` / missing `GLIBCXX_*`:

```bash
module load buildenv-gcc/2026.03-mpich      # example: GCC 14 + OpenBLAS + FFTW
export OPENBLAS_ROOT="$OPENBLASDIR"
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
# x86_64 login node only, if default rustup is wrongly aarch64:
export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
```

## 3. Configure and build

Smallest useful start — Gaussian SCF + examples, CPU only:

```bash
git submodule update --init --recursive molrs   # once per clone
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DATV_BUILD_CUDA=OFF -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON \
      -DBLA_VENDOR=OpenBLAS
cmake --build build --target 02_rhf -j
```

Add `-DATV_BUILD_PW=ON` for plane-wave, `-DATV_BUILD_TESTS=ON` for tests.

> **molrs path:** molrs is a hard dependency, vendored as a git submodule at
> `./molrs`, so `ATV_MOLRS_PATH` defaults to `${CMAKE_SOURCE_DIR}/molrs`. In a
> git worktree or non-standard layout, pass an absolute path:
> `-DATV_MOLRS_PATH=/abs/path/to/molrs`.

## 4. Run a first example

```bash
./build/examples/02_rhf
```

H₂ / STO-3G restricted Hartree–Fock. You should see a converged total energy
and a small iteration count. That proves the build chain works. Chapter 2
walks through the theory and shows **density and HOMO cubes** produced from
the same SCF — not a bare geometry canvas.

Theory and Practice snippets: [Chapter 2](../tutorial/ch02-rhf.md). Other
runnable demos live under `examples/` and document themselves; the handbook
does not catalogue them. `examples/CMakeLists.txt` shows which build flag each
one is gated behind (e.g. `08_pw` needs `ATV_BUILD_PW`, `09_gpw` needs
`ATV_BUILD_GPW`); a target that is not configured simply does not exist.

Full catalog and scientific context: [Tutorial index](../tutorial/).

## 5. Minimal user code shape

Every calculation looks like this — fill a **Frame**, pick a **preset**,
attach **components**, `build` + `run`:

```cpp
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/dynamics/velocity_verlet.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"
using namespace ATV;

cpu::md::MD sim;
sim.set_evaluator(cpu::PairLJCut{0.0104, 3.4, 6.8});
sim.set_neighbor(cpu::md::NeighborList<>{6.8, 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{0.01});
sim.build();
sim.run(frame, 1000);
```

Header paths follow the **directory** a type lives in, not its namespace:
`cpu::md::MD` is `atomiverse/cpu/dynamics/md.h`, `cpu::minimizer::Minimize` is
`atomiverse/cpu/minimize/minimize.h`.

Read energy from `frame.meta.get(ATV::kEnergy)` (eV). Details:
[Core concepts](concepts.md) · [Methods](methods.md).

## 6. Recommended path

1. This page — build and run something.
2. [Core concepts](concepts.md) — Driver / Component / Frame / Criterion.
3. [Methods](methods.md) — which preset for your job.
4. [Tutorial](../tutorial/) — the method you care about (MD, SCF, PW, …).
5. [Developer](../developer/) — only when you need internals or extensions.
