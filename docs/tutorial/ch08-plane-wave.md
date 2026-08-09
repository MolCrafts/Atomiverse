# Chapter 8. Plane-Wave DFT

API: concrete `cpu::pw::PW` (members) + `PWEvaluator` for Driver binding. Empty UPF path is a toy
\(V_\mathrm{loc}\); real Si UPF lives in the regression data tree.

## Live Result: Silicon Diamond

Tutorial settings: **ecut = 15 Ry**, **k_mesh = 2** (seconds, not production):

| | Atomiverse |
|---|---:|
| Converged | yes (5 iters) |
| \(E_\mathrm{total}\) | **−15.670898 Ry** |
| | **−7.835449 Ha** |
| | **−213.213424 eV** |

For reference, the PseudoDojo deltafactor **converged** \(E_\mathrm{min}\) for
this exact UPF is −16.916 Ry (−230.158 eV per cell at \(a_0\approx 5.47\) Å) —
not the same ecut/k-mesh as the tutorial. Absolute PW energies shift with basis
completeness; production acceptance in the regressions suite compares **EOS**
(\(a_0\), \(B_0\)) rather than raw total energy. See
the regression reference notes.

```cpp
cpu::pw::PW pw;
pw.upf_path = "…/Si_lda_nc.upf";
pw.ecut_ry = 15.0;   // tutorial; ~30 Ry for production Si
pw.k_mesh = 2;
cpu::pw::PWEvaluator pwe{std::move(pw)};
cpu::single_point(frame, pwe);          // frame.meta.get(kEnergy) (eV)
auto res = pwe.last_result();           // Ry / Ha on the result object
```

This tutorial assumes you have read
[Chapter 2](ch02-rhf.md) and
[Chapter 5](ch05-kohn-sham-dft.md).

The first four tutorials covered **molecular** electronic structure with
atom-centered Gaussian basis sets. This tutorial moves to **periodic** systems —
crystals, surfaces, bulk materials — where the natural basis is **plane waves**.
The Practice block below computes the energy of bulk silicon.

The guiding question:

> how does the SCF iteration change when we replace Gaussian basis functions
> with plane waves, and what new physics does that unlock?

## Why plane waves?

For a molecule in a large box, localised Gaussians are efficient: a handful per
atom. For a periodic crystal the wavefunction extends over the whole lattice,
and a plane-wave expansion

\[
\psi_{n\mathbf{k}}(\mathbf{r}) =
\sum_{\mathbf{G}} c_{n\mathbf{k}}(\mathbf{G})\,
e^{i(\mathbf{k}+\mathbf{G})\cdot\mathbf{r}}
\]

is natural because:

1. **Bloch's theorem** is satisfied automatically — each **k**-point labels an
   irreducible representation of the translation group.
2. **Systematic convergence** — raising the kinetic-energy cutoff
   \f$E_{\mathrm{cut}}\f$ adds more plane waves and converges the basis toward
   completeness. There is no basis-set superposition error.
3. **FFT acceleration** — switching between reciprocal space (where kinetic
   energy and the Hartree potential are diagonal) and real space (where the
   exchange-correlation potential is local) costs only
   \f$\mathcal{O}(N\log N)\f$.

The price: plane waves are inefficient for tightly bound core electrons. The Si
\f$1s\f$ orbital oscillates on the scale of 0.1 Å and would demand an enormous
cutoff. That is why we use pseudopotentials.

## Pseudopotentials: freezing the core

A **norm-conserving pseudopotential** replaces the bare nuclear Coulomb
potential with a softer effective potential that, outside a core radius
\f$r_c\f$, matches the all-electron potential, while inside \f$r_c\f$ it
produces nodeless, norm-conserving pseudo valence wavefunctions with the correct
scattering properties.

Atomiverse reads pseudopotentials from **UPF v1** (Unified Pseudopotential
Format) XML files. A UPF file supplies the valence charge \f$Z_{\mathrm{val}}\f$,
the local potential \f$V_{\mathrm{loc}}(G)\f$, the Kleinman-Bylander non-local
projectors \f$\beta_\ell\f$, and a reference atomic density. A silicon LDA
pseudopotential ships with the repository at
`tests/data/pw/Si_lda_nc.upf` (for Si,
\f$Z_{\mathrm{val}} = 4\f$).

## G-vectors and the energy cutoff

For lattice vectors \f$\mathbf{a}_i\f$ the reciprocal lattice is
\f$\mathbf{G} = \sum_i n_i \mathbf{b}_i\f$ with
\f$\mathbf{b}_i\cdot\mathbf{a}_j = 2\pi\delta_{ij}\f$. The basis is truncated by
keeping the \f$\mathbf{G}\f$ that satisfy

\[
\tfrac{1}{2}\,|\mathbf{k}+\mathbf{G}|^2 \le E_{\mathrm{cut}}.
\]

The number of plane waves scales as
\f$N_G \propto E_{\mathrm{cut}}^{3/2}\,\Omega\f$ (\f$\Omega\f$ = cell volume).
A typical cutoff for silicon is 30 Ry (~10³ plane waves for the 2-atom cell);
the example uses a smaller 15 Ry so it finishes in seconds.

## k-point sampling

Physical quantities are integrals over the Brillouin zone, approximated by a
weighted sum over a discrete **Monkhorst-Pack mesh**:

\[
\frac{1}{\Omega_{\mathrm{BZ}}}\int_{\mathrm{BZ}} f(\mathbf{k})\,d\mathbf{k}
\approx \sum_i w_i\, f(\mathbf{k}_i).
\]

A converged silicon calculation uses a \f$4\times4\times4\f$ mesh or finer; the
example uses \f$2\times2\times2\f$ for speed (`PW::k_mesh = 2`).

## The plane-wave SCF cycle

The self-consistent loop has the same *shape* as every other SCF in Atomiverse —
build an effective Hamiltonian, diagonalize, rebuild the density, check
convergence — but the ingredients are different. In the plane-wave code the loop
lives inside the concrete method `cpu::pw::PW`, which drives its step helpers
(`cpu::pw::internal::PoissonSolver`, `XCBuilder`, `VeffBuilder`,
`DavidsonSolver`, `DensityBuilder`, `ConvergenceCheck`) in a fixed order. These
helpers are owned by the solver; they are **not** Plan components and carry no
phase enum.

**1. Hartree potential (Poisson, reciprocal space).** Solve
\f$V_{\mathrm{H}}(\mathbf{G}) = 4\pi\,\rho(\mathbf{G})/G^2\f$ for
\f$\mathbf{G}\neq 0\f$; the \f$\mathbf{G}=0\f$ term is set to zero (it cancels
against the uniform neutralizing background).

**2. Exchange-correlation (real space, via FFT).**
\f$\rho(\mathbf{r}) = \mathrm{FFT}^{-1}[\rho(\mathbf{G})]\f$, evaluate
\f$V_{\mathrm{xc}}(\mathbf{r}) = \delta E_{\mathrm{xc}}/\delta\rho\f$
pointwise (the same LDA functional family as Chapter 5), then FFT back to
\f$V_{\mathrm{xc}}(\mathbf{G})\f$.

**3. Effective potential.**
\f$V_{\mathrm{eff}}(\mathbf{G}) = V_{\mathrm{loc}}(\mathbf{G}) +
V_{\mathrm{H}}(\mathbf{G}) + V_{\mathrm{xc}}(\mathbf{G})\f$. The kinetic term
\f$T(\mathbf{G}) = \tfrac{1}{2}|\mathbf{k}+\mathbf{G}|^2\f$ is diagonal and
constant across the SCF.

**4. Diagonalize (Davidson).** The Hamiltonian is \f$N_G \times N_G\f$ with
\f$N_G\sim 10^3\f$, and we only need the lowest few bands. The **Davidson
method** is a matrix-free iterative diagonalizer: it never forms the full
matrix, applying \f$H\psi = (T + V_{\mathrm{eff}})\psi\f$ with FFTs to switch
between the diagonal-kinetic (reciprocal) and local-potential (real) spaces.

**5. Density + mixing.** Build \f$\rho_{\mathrm{out}}\f$ from the occupied
states. Feeding it straight back diverges, so **Pulay mixing** (DIIS in density
space) extrapolates from the history of input/output densities; `PW`
exposes `mix_beta` and `pulay_depth`.

**6. Convergence.** Measured by the RMS change in density; a short warmup
suppresses false convergence from the initial guess.

## Ewald summation and the total energy

The ion-ion Coulomb sum is only conditionally convergent. The **3D Ewald
method** splits it into absolutely convergent real-space, reciprocal-space, and
self-energy pieces. The total energy per cell is

\[
E_{\mathrm{total}} =
\sum_n^{\mathrm{occ}} f_n\varepsilon_n
- E_{\mathrm{H}}[\rho]
- \int V_{\mathrm{xc}}\rho\,d\mathbf{r}
+ E_{\mathrm{xc}}[\rho]
+ E_{\mathrm{ewald}},
\]

i.e. the band-structure energy (sum of occupied eigenvalues), the
double-counting corrections to the effective potential, and the Ewald ion-ion
energy.

## Running it from C++

The user-facing entry point is the `cpu::pw::PWEvaluator` component — the
plane-wave energy component. It owns a `PW`, has
`role() == Evaluator` and `capabilities() == Energy`, and writes the total
energy to `frame.meta.get(kEnergy)` in eV (the Rydberg / Hartree breakdown stays on
`last_result()`). Run it through the one-pass `cpu::Evaluate` driver, most
concisely via the `cpu::single_point` helper:

```cpp
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/planewave/pw_evaluator.h"

cpu::pw::PW pw;
pw.upf_path = "tests/data/pw/Si_lda_nc.upf";
pw.ecut_ry  = 30.0;   // wavefunction cutoff [Ry]
pw.k_mesh   = 4;      // 4x4x4 Monkhorst-Pack mesh
pw.tol      = 1e-6;   // SCF tolerance on the density change

cpu::pw::PWEvaluator pwe{std::move(pw)};
cpu::single_point(frame, pwe);          // frame.meta.get(kEnergy) (eV)

const auto& r = pwe.last_result();
// r.energy [Ry], r.energy_ha (Hartree), r.converged, r.iterations
```

Low-level code uses the same explicit solver lifecycle:

```cpp
#include "atomiverse/cpu/planewave/pw_solver.h"
cpu::pw::PW& solver = pw;  // concrete method
solver.build(frame);
const cpu::pw::PWResult& r = solver.run(frame);
```

If `upf_path` is empty, the solver falls back to a toy local potential
(\f$V_{\mathrm{loc}}\propto -1/G^2\f$): the SCF loop still runs and returns a
finite number, useful for exercising the infrastructure, but the energy is not
physically meaningful. Use a real UPF for real numbers.

## Build and run the example

The plane-wave path needs FFTW3 at build time.

The example builds the 2-atom silicon diamond cell, configures `PW`
(15 Ry, \f$2\times2\times2\f$ mesh), runs it as a `PWEvaluator`, and prints the
total energy in Ry, Hartree, and eV.

## What the solver does internally

```
cpu::Frame (Si diamond, 2 atoms)  +  PW
  → PW::build(frame)         // parse the UPF
  → PW::run(frame):          // geometry-dependent setup rebuilt each run:
      Ewald sum, G-vectors, FFT plans, V_loc, KB projectors
      SCF loop (internal step helpers, fixed order):
        PoissonSolver   → V_H(G)
        XCBuilder       → V_xc(G)  via FFT + LDA
        VeffBuilder     → V_eff = V_loc + V_H + V_xc
        DavidsonSolver  → lowest N_occ bands per k-point
        DensityBuilder  → ρ(G) + Pulay mixing
        ConvergenceCheck→ ||Δρ|| < tol
  → PWResult{energy [Ry], energy_ha, converged, iterations}
```

`build()` does the one-time UPF parse; the geometry-dependent structures are
rebuilt inside each `run()`, so the same solver handles a moving geometry (AIMD,
optimization). Under MPI the solver distributes k-points round-robin across
`MPI_COMM_WORLD` and Allreduces the density and band-energy accumulations.

A GPU path exists as the same method shape: `cuda::pw::PW` has the same public
members as `cpu::pw::PW` plus `build(reference)` / `run(frame)` /
`last_result()`, and returns the same `PWResult`. Both halves of the SCF
iteration are device-resident: the Davidson hot path (cuFFT for \f$H\psi\f$,
cuBLAS for the subspace algebra) and the whole density side — \f$\rho(G) \to
V_H \to V_{xc} \to V_\mathrm{eff}(r)\f$, Pulay mixing and the energy/residual
reductions. Per SCF iteration **no \f$O(N_G)\f$ or \f$O(N_r)\f$ array crosses
PCIe**; what does is the \f$\le 8\times 8\f$ Pulay Gram matrix, its
coefficients, and five energy/residual scalars. The host keeps the geometry
work (problem assembly, per-k table binding), the Pulay coefficient solve
(bit-for-bit the CPU's `solve_pulay_coefficients`) and the energy report.

```cpp
cuda::pw::PW pw;
pw.upf_path = "Si_lda_nc.upf";
pw.ecut_ry = 15.0;
pw.k_mesh = 2;
pw.xc = cpu::scf::XCType::LSDA;   // required on this backend
pw.build(frame);
const auto& r = pw.run(frame);    // r.energy [Ry], r.energy_ha [Ha]
```

The device path is a strict subset of the CPU path, and a member it cannot
honour makes `run()` throw with a how-to instead of silently computing a
different Hamiltonian: only `XCType::LSDA`, only restricted
(`spin_polarized = false`, `multiplicity = 1`), only fixed closed-shell
occupations (`smearing = Smearing::none`). k-points are not distributed across
MPI ranks on this backend — every rank computes every k-point, so
`n_kpts_local == n_kpts`. Use `cpu::pw::PW` for GGA, collinear spin, metallic
smearing, or MPI k-point parallelism.

CPU↔CUDA agreement at identical settings is asserted end-to-end by
`regressions/pw/pw-cuda-lda-si/` (`--device=both`), which also carries the
ABINIT-class literature window.

## Where to read the code

1. [`include/atomiverse/cpu/planewave/pw_evaluator.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/planewave/pw_evaluator.h) — the user-facing component
2. [`include/atomiverse/cpu/planewave/pw_solver.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/planewave/pw_solver.h) — `PW`, `PWResult`
3. `include/atomiverse/cpu/planewave/internal/` — the step helpers and numerics
   (`g_vector.h`, `fft.h`, `ewald.h`, `kpoint.h`, `upf_reader.h`, …)

For a converged equation-of-state validation against CP2K, see
`regressions/ or a future benchmarks tree/`.

## References

1. M. C. Payne *et al.*, *Rev. Mod. Phys.* **64**, 1045 (1992).
2. D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013).
3. M. Schlipf and F. Gygi, *Comput. Phys. Commun.* **196**, 36 (2015).
4. E. R. Davidson, *J. Comput. Phys.* **17**, 87 (1975).
5. P. Pulay, *Chem. Phys. Lett.* **73**, 393 (1980).
