# Plane Waves

Status tables: [DFT roadmap](../roadmap/dft.md).

A crystal can be modeled by a unit cell repeated through space. Translational
symmetry then becomes a computational advantage: the orbitals can be
organized by their phase change from one cell to the next.

**Learning goals.** Interpret Bloch states, distinguish basis cutoff from
Brillouin-zone sampling, and define a reproducible silicon calculation.

**Before you begin.** [Density Functional Theory](ch05-kohn-sham-dft.md) and
the eigenproblem in [Mathematical Tools](mathematical-tools.md).

## Symmetry and reciprocal space

For a periodic potential, Bloch's theorem permits
$\psi_{n\mathbf k}(\mathbf r)=e^{i\mathbf k\cdot\mathbf r}u_{n\mathbf k}(\mathbf r)$,
where $u$ has lattice periodicity. The band index $n$ labels states and
$\mathbf k$ describes phase change between cells. Reciprocal vectors
$\mathbf G$ satisfy $e^{i\mathbf G\cdot\mathbf T}=1$ for lattice translations
$\mathbf T$. Expanding the periodic part in these vectors gives plane waves
with wavevectors $\mathbf k+\mathbf G$.

In atomic units the cutoff retains
$|\mathbf k+\mathbf G|^2/2\le E_{\mathrm{cut}}$. It controls spatial
resolution. A mesh of $\mathbf k$ points approximates a different object:
an integral over the Brillouin zone. Refining one does not replace refining
the other. Hartree and Rydberg are different units, with $1\,E_h=2$ Ry.

## Cores and electrostatics

Pseudopotentials replace selected core effects by a smoother valence
interaction. They must match element, valence partition, and functional.
Cutoff convergence solves that pseudopotential problem more accurately;
it does not recover a different all-electron Hamiltonian.

For nonzero reciprocal vector, Poisson's equation gives
$V_H(\mathbf G)=4\pi\rho(\mathbf G)/G^2$ in atomic units. The zero component
requires consistent neutrality and potential-reference conventions.
Charged periodic cells may need a compensating background and finite-size
corrections. Setting a coefficient to zero does not turn them into isolated ions.

## Periodic orbital expansions

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
   $E_{\mathrm{cut}}$ adds more plane waves and converges the basis toward
   completeness. There is no basis-set superposition error.
3. **FFT acceleration** — switching between reciprocal space (where kinetic
   energy and the Hartree potential are diagonal) and real space (where the
   exchange-correlation potential is local) costs only
   $\mathcal{O}(N\log N)$.

The price: plane waves are inefficient for tightly bound core electrons. The Si
$1s$ orbital oscillates on the scale of 0.1 Å and would demand an enormous
cutoff. That is why we use pseudopotentials.

## Pseudopotentials

A **norm-conserving pseudopotential** replaces the bare nuclear Coulomb
potential with a softer effective potential that, outside a core radius
$r_c$, matches the all-electron potential, while inside $r_c$ it
produces nodeless, norm-conserving pseudo valence wavefunctions with the correct
scattering properties.

CPU PW consumes an immutable `electronic::NormConservingPotential`.
`cpu::pw::UpfPseudopotential(path)` reads scalar norm-conserving UPF data once;
`cpu::pw::GthPseudopotential(element, name)` supplies published analytic GTH
parameters. Both use the same local and Kleinman–Bylander operator path.
A UPF file supplies the valence charge $Z_{\mathrm{val}}$, radial local
potential, nonlocal projectors and optional reference atomic density.
USPP, PAW, NLCC and spin-orbit payloads reject at admission. A silicon LDA
pseudopotential ships with the repository at
`tests/data/pw/Si_lda_nc.upf` (for Si,
$Z_{\mathrm{val}} = 4$).

## Basis cutoff

For lattice vectors $\mathbf{a}_i$ the reciprocal lattice is
$\mathbf{G} = \sum_i n_i \mathbf{b}_i$ with
$\mathbf{b}_i\cdot\mathbf{a}_j = 2\pi\delta_{ij}$. The basis is truncated by
keeping the $\mathbf{G}$ that satisfy

\[
\tfrac{1}{2}\,|\mathbf{k}+\mathbf{G}|^2 \le E_{\mathrm{cut}}.
\]

The number of plane waves scales as
$N_G \propto E_{\mathrm{cut}}^{3/2}\,\Omega$ ($\Omega$ = cell volume).
A typical cutoff for silicon is 30 Ry (~10³ plane waves for the 2-atom cell);
the example uses a smaller 15 Ry so it finishes in seconds.

## k-point sampling

Physical quantities are integrals over the Brillouin zone, approximated by a
weighted sum over a discrete **Monkhorst-Pack mesh**:

\[
\frac{1}{\Omega_{\mathrm{BZ}}}\int_{\mathrm{BZ}} f(\mathbf{k})\,d\mathbf{k}
\approx \sum_i w_i\, f(\mathbf{k}_i).
\]

A converged silicon calculation uses a $4\times4\times4$ mesh or finer; the
example uses $2\times2\times2$ for speed (`PW::k_mesh = 2`).

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
$V_{\mathrm{H}}(\mathbf{G}) = 4\pi\,\rho(\mathbf{G})/G^2$ for
$\mathbf{G}\neq 0$. The $\mathbf{G}=0$ term needs a consistent potential
reference and neutrality convention. Ionic and electronic contributions
must be combined consistently; a charged cell requires an explicitly
defined background and finite-size treatment, not just deletion of a coefficient.

**2. Exchange-correlation (real space, via FFT).**
$\rho(\mathbf{r}) = \mathrm{FFT}^{-1}[\rho(\mathbf{G})]$, evaluate
$V_{\mathrm{xc}}(\mathbf{r}) = \delta E_{\mathrm{xc}}/\delta\rho$
pointwise (the same LDA functional family as [Density Functional Theory](ch05-kohn-sham-dft.md)), then FFT back to
$V_{\mathrm{xc}}(\mathbf{G})$.

**3. Effective potential.**
$V_{\mathrm{eff}}(\mathbf{G}) = V_{\mathrm{loc}}(\mathbf{G}) +
V_{\mathrm{H}}(\mathbf{G}) + V_{\mathrm{xc}}(\mathbf{G})$. The kinetic term
$T(\mathbf{G}) = \tfrac{1}{2}|\mathbf{k}+\mathbf{G}|^2$ is diagonal and
constant across the SCF.

**4. Diagonalize (Davidson).** The Hamiltonian is $N_G \times N_G$ with
$N_G\sim 10^3$, and we only need the lowest few bands. The **Davidson
method** is a matrix-free iterative diagonalizer: it never forms the full
matrix, applying $H\psi = (T + V_{\mathrm{eff}})\psi$ with FFTs to switch
between the diagonal-kinetic (reciprocal) and local-potential (real) spaces.

**5. Density + mixing.** Build $\rho_{\mathrm{out}}$ from the occupied
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

## Build and run the example

The plane-wave path needs FFTW3 at build time.

The example builds the 2-atom silicon diamond cell, configures `PW`
(15 Ry, $2\times2\times2$ mesh), runs it through `cpu::single_point`, and prints the
total energy in Ry, Hartree, and eV.

## Practice plan

Use diamond silicon with explicit lattice vectors and fractional or Cartesian
coordinates. Record the pseudopotential file and checksum, valence electrons,
functional, cutoff units, mesh, and occupation convention. Replace placeholder
paths with real project-relative files.

Converge the cutoff at fixed mesh, then the mesh at fixed cutoff. Inspect
energy per atom and stress before constructing a volume–energy curve. Cell
changes alter the finite plane-wave space, so stress may need tighter
convergence than energy. Historical suggestions for a production cutoff are
not a substitute for this study.

## Silicon example

The user-facing entry point is the concrete method `cpu::pw::PW` itself — a
`Driver<pw::Phase>` node that a hosting driver mounts directly. It writes the
total energy to `frame.meta.get(kEnergy)` in eV (the Rydberg / Hartree
breakdown stays on `last_result()`). Run it through the one-pass
`cpu::Evaluate` driver, most concisely via the `cpu::single_point` helper:

```cpp
#include "atomiverse/cpu/evaluate/single_point.h"
#include "atomiverse/cpu/planewave/norm_conserving_potential.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

cpu::pw::PW pw{{std::make_shared<cpu::pw::UpfPseudopotential>("tests/data/pw/Si_lda_nc.upf")},
               30.0,  // wavefunction cutoff [Ry]
               4};    // 4x4x4 Monkhorst-Pack mesh
pw.tol      = 1e-6;   // SCF tolerance on the density change

cpu::single_point(frame, pw);           // frame.meta.get(kEnergy) (eV)

const auto& r = pw.last_result();
// r.energy [Ry], r.energy_ha (Hartree), r.converged, r.iterations
```

Low-level code uses the same explicit solver lifecycle:

```cpp
#include "atomiverse/cpu/planewave/pw_solver.h"
cpu::pw::PW& solver = pw;  // concrete method
solver.build(frame);
const cpu::pw::PWResult& r = solver.run(frame);
```

The constructor takes a list of NC providers and selects one per element by atomic number.
For a file-free silicon calculation, put
`std::make_shared<cpu::pw::GthPseudopotential>(molrs::Element::Si, "GTH-PBE-q4")`
in that list.
For different scalar NC physics, derive `NormConservingPotential` and supply
immutable radial tables plus the batched local value/derivative operation.
PW owns all transforms, force contractions and electronic iteration. The
[independent installed project](../../examples/extensions/nc/README.md)
demonstrates that extension without a new Driver or a copied solver.

## Implementation outline

```
immutable NC provider + cpu::Frame (Si diamond, 2 atoms) + PW
  → PW::build(frame)         // validate identity and snapshot physical data
  → PW::run(frame):          // geometry-dependent setup rebuilt each run:
      Ewald sum, G-vectors, FFT plans, V_loc, KB projectors
      SCF loop (internal step helpers, fixed order):
        PoissonSolver   → V_H(G)
        XCBuilder       → V_xc(G)  via FFT + LDA
        VeffBuilder     → V_eff = V_loc + V_H + V_xc
        DavidsonSolver  → lowest N_occ bands per k-point
        DensityBuilder  → ρ(G) + Pulay mixing
        ConvergenceCheck→ ||ρ_out − ρ_in|| < tol, before mixing
  → PWResult{energy [Ry], energy_ha, converged, iterations}
```

The provider parses or constructs its data once. The geometry-dependent structures are
rebuilt inside each `run()`, so the same solver handles a moving geometry (AIMD,
optimization). Under MPI the solver distributes k-points round-robin across
`MPI_COMM_WORLD` and Allreduces the density and band-energy accumulations.

A GPU path exists as the same method shape: `cuda::pw::PW` has the same constructor and public members as `cpu::pw::PW` plus `build(reference)` / `run(frame)` /
`last_result()`, and returns the same `PWResult`. Both halves of the SCF
iteration are device-resident: the Davidson hot path (cuFFT for $H\psi$,
cuBLAS for the subspace algebra) and the whole density side — $\rho(G) \to
V_H \to V_{xc} \to V_\mathrm{eff}(r)$, Pulay mixing and the energy/residual
reductions. Per SCF iteration **no $O(N_G)$ or $O(N_r)$ array crosses
PCIe**; what does is the $\le 8\times 8$ Pulay Gram matrix, its
coefficients, and five energy/residual scalars. The host keeps the geometry
work (problem assembly, per-k table binding), the Pulay coefficient solve
(bit-for-bit the CPU's `solve_pulay_coefficients`) and the energy report.

```cpp
cuda::pw::PW pw{{std::make_shared<cpu::pw::UpfPseudopotential>("Si_lda_nc.upf")}, 15.0, 2, cpu::xc::XCType::LSDA};
pw.build(frame);
const auto& r = pw.run(frame);    // r.energy [Ry], r.energy_ha [Ha]
```

CUDA accepts builtin GTH and scalar NC-UPF providers in the same list. The
SCF supports LSDA and the semilocal PBE family, with restricted or collinear
spin and fixed or smeared occupations. Each CUDA rank computes the full k mesh
and reports `n_kpts_local == n_kpts`. CPU PW distributes k-points across MPI
ranks.

CPU↔CUDA agreement at identical settings is asserted end-to-end by
`pw-cuda-lda-si` (`--device=both`), which also carries the
ABINIT-class literature window.

The following numerical material is retained from the source snapshot.
It was not rerun for this editorial revision; match its input and provenance
before using it as a validation target.

## Reference results: Silicon Diamond

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
cpu::pw::PW pw{{std::make_shared<cpu::pw::UpfPseudopotential>("…/Si_lda_nc.upf")}, 15.0, 2};  // tutorial; ~30 Ry for production Si
cpu::single_point(frame, pw);           // frame.meta.get(kEnergy) (eV)
auto res = pw.last_result();            // Ry / Ha on the result object
```

This tutorial assumes you have read
[Hartree–Fock Theory](ch02-rhf.md) and
[Density Functional Theory](ch05-kohn-sham-dft.md).

The first four tutorials covered **molecular** electronic structure with
atom-centered Gaussian basis sets. This tutorial moves to **periodic** systems —
crystals, surfaces, bulk materials — where the natural basis is **plane waves**.
The Practice block below computes the energy of bulk silicon.

The guiding question:

> how does the SCF iteration change when we replace Gaussian basis functions
> with plane waves, and what new physics does that unlock?

## Source references

1. `pw_solver.h` — `PW` (the user-facing method), `PWResult`
2. `internal` — the step helpers and numerics
   (`g_vector.h`, `fft.h`, `ewald.h`, `kpoint.h`, `upf_reader.h`, …)

For a converged equation-of-state validation against CP2K, see
the end-to-end validation cases (and standing benchmarks when present).

## References

1. M. C. Payne *et al.*, *Rev. Mod. Phys.* **64**, 1045 (1992).
2. D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013).
3. M. Schlipf and F. Gygi, *Comput. Phys. Commun.* **196**, 36 (2015).
4. E. R. Davidson, *J. Comput. Phys.* **17**, 87 (1975).
5. P. Pulay, *Chem. Phys. Lett.* **73**, 393 (1980).

## Questions for reflection

Which approximation changes when only the k mesh is refined? Why can a large periodic box still differ from a genuinely isolated electrostatic calculation?
