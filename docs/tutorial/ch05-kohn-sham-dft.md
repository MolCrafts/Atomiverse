# Chapter 5: Unrestricted Kohn-Sham DFT (UKS)

| | |
|---|---|
| **Example** | [`examples/04_uks.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/04_uks.cpp) |
| **Results** | [`examples/results/04_uks.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/04_uks.json) |

```bash
cmake --build build --target 04_uks && ./build/examples/04_uks
```

## Live atomic and H₂ numbers (UKS-LSDA)

| System | Method | \(E_\mathrm{total}\) (Ha) |
|--------|--------|--------------------------:|
| He | UKS-LSDA | **−2.771886** |
| He | UHF (same geometry) | −2.807784 |
| H | UKS-LSDA | **−0.435670** |
| Li | UKS-LSDA | **−7.221302** |

H₂ scan: UHF vs UKS-LSDA from the same binary (STO-3G).

```molplot preset="molplot" theme="auto" width="100%" aspect="16:9"
mark: {type: line, point: true, tooltip: true, strokeWidth: 2}
data:
  values:
    - {R: 0.50, method: UHF, E: -1.042996}
    - {R: 0.60, method: UHF, E: -1.101128}
    - {R: 0.74, method: UHF, E: -1.116759}
    - {R: 1.00, method: UHF, E: -1.066109}
    - {R: 1.50, method: UHF, E: -0.957707}
    - {R: 2.00, method: UHF, E: -0.937213}
    - {R: 2.50, method: UHF, E: -0.933867}
    - {R: 3.00, method: UHF, E: -0.933285}
    - {R: 0.50, method: UKS-LSDA, E: -1.038469}
    - {R: 0.60, method: UKS-LSDA, E: -1.099776}
    - {R: 0.74, method: UKS-LSDA, E: -1.121207}
    - {R: 1.00, method: UKS-LSDA, E: -1.084898}
    - {R: 1.50, method: UKS-LSDA, E: -0.965796}
    - {R: 2.00, method: UKS-LSDA, E: -0.874831}
    - {R: 2.50, method: UKS-LSDA, E: -0.822541}
    - {R: 3.00, method: UKS-LSDA, E: -0.796166}
encoding:
  x: {field: R, type: quantitative, title: "R (Å)", scale: {zero: false}}
  y: {field: E, type: quantitative, title: "E (Hartree)", scale: {zero: false, nice: true}}
  color: {field: method, type: nominal, title: null}
  tooltip:
    - {field: method, type: nominal}
    - {field: R, type: quantitative, format: ".2f"}
    - {field: E, type: quantitative, format: ".6f"}
```

Near equilibrium, LSDA sits slightly **below** UHF (correlation); at large
\(R\) the local functional fails to recover the exact radical limit that UHF
gets right — a known LDA limitation, not a bug.

The move from RHF to UHF changed the spin structure of the SCF problem but kept
the Hartree-Fock mean field. The move from UHF to UKS changes something deeper:
it keeps the same spin-resolved SCF skeleton, but it replaces explicit
Hartree-Fock exchange with an exchange-correlation functional of the density.

The example for this chapter is
[`examples/04_uks.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/04_uks.cpp). That
example runs a closed-shell helium atom, open-shell hydrogen and lithium atoms,
and a small $\mathrm{H}_2$ dissociation comparison against UHF. It is a good
bridge from the earlier chapters because it shows clearly what UKS keeps from
UHF and what it no longer shares with Hartree-Fock.

## The Idea Behind Kohn-Sham DFT

Hartree-Fock builds an explicit exchange operator from the occupied orbitals.
Kohn-Sham DFT takes a different route. It says that a large part of the missing
many-electron physics can be captured more economically through a density
functional. Instead of carrying an explicit exchange matrix
$K(P^\sigma)$, the method asks for the exchange-correlation energy
$E_{\mathrm{xc}}[\rho_\alpha, \rho_\beta]$ and for the corresponding
potentials obtained by functional differentiation with respect to the spin
densities.

This means that the UHF pair of Fock matrices

$$
F^\alpha = H_{\mathrm{core}} + J(P) - K(P^\alpha),
\qquad
F^\beta = H_{\mathrm{core}} + J(P) - K(P^\beta)
$$

becomes, in UKS,

$$
F^\alpha = H_{\mathrm{core}} + J(P) + V_{\mathrm{xc}}^\alpha,
\qquad
F^\beta = H_{\mathrm{core}} + J(P) + V_{\mathrm{xc}}^\beta.
$$

The SCF choreography is unchanged. You still guess densities, build effective
matrices, diagonalize, rebuild densities, and test convergence. The difference
is that one of the matrix-building steps now happens in real space on a
numerical grid rather than purely through analytic four-index exchange
contractions.

## From Density Matrices To Real-Space Density

The bridge between matrix algebra and DFT is the real-space density. In the
atomic-orbital basis, the spin densities at a point $\mathbf{r}$ are

$$
\rho_\sigma(\mathbf{r}) =
\sum_{\mu\nu} P^\sigma_{\mu\nu}\chi_\mu(\mathbf{r})\chi_\nu(\mathbf{r}),
\qquad \sigma \in \{\alpha,\beta\}.
$$

The total density is $\rho = \rho_\alpha + \rho_\beta$. For gradient-based
functionals such as PBE, the calculation also needs derivatives of the density
with respect to space. That is why UKS needs more than the quantities used in
Hartree-Fock alone. It must also evaluate basis functions and their gradients
at many quadrature points.

The important conceptual point is that UKS does not abandon the basis-set
picture you learned in the RHF and UHF chapters. It adds a second numerical
layer on top of it. The density matrices still live in the AO basis. The
exchange-correlation functional is simply evaluated in real space and then
projected back into matrix form.

## Why Grid Integration Appears

Analytic Hartree-Fock exchange can be expressed directly in terms of the ERI
tensor. General exchange-correlation functionals cannot. UKS therefore replaces
the formal real-space integral

$$
\int f(\mathbf{r})\, d\mathbf{r}
$$

with a quadrature sum over grid points $\mathbf{r}_g$ and weights
$w_g$:

$$
\int f(\mathbf{r})\, d\mathbf{r}
\approx
\sum_g w_g f(\mathbf{r}_g).
$$

At each grid point the code evaluates the basis, reconstructs
$\rho_\alpha$ and $\rho_\beta$, asks the chosen functional for
$\varepsilon_{\mathrm{xc}}$ and its derivatives, accumulates the scalar
$E_{\mathrm{xc}}$, and assembles the matrices
$V_{\mathrm{xc}}^\alpha$ and $V_{\mathrm{xc}}^\beta$.

## The Energy Formula Readers Most Often Misunderstand

A common beginner mistake is to treat the exchange-correlation energy as if it
were recoverable from a matrix trace, in exact analogy with Hartree-Fock
exchange. In general it is not. The UKS total energy used in Atomiverse is

$$
E_{\mathrm{KS}} =
\sum_\sigma \mathrm{Tr}\!\left(H_{\mathrm{core}} P^\sigma\right)
+ \frac{1}{2}\mathrm{Tr}\!\left(J(P) P\right)
+ E_{\mathrm{xc}}
+ V_{\mathrm{nuc}}.
$$

The scalar $E_{\mathrm{xc}}$ is accumulated separately from the grid energy
density. It is not, in general, equal to
$\mathrm{Tr}(V_{\mathrm{xc}} P)$. That point is easy to miss if you think only
in terms of matrix formulas, but it is central to understanding why DFT is not
just Hartree-Fock with a different label.

## Reading The Example

In [`examples/04_uks.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/04_uks.cpp), the
basic call looks like this:

```cpp
auto ints = /* IntegralPack is private — use RHF */;
cpu::scf::UKS uks(n_alpha, n_beta, XCType::LSDA);
config.n_radial = 50;
config.n_angular = 110;
config.max_iter = max_iter;
cpu::scf::UKS uks(n_alpha, n_beta, cpu::scf::XCType::LSDA);
solver.build(ints);
const auto& uks = solver.run(ints);
```

> **Element coverage:** like the other Gaussian-basis tutorials, STO-3G here
> supports elements **H through O (Z = 1–8)**; heavier atoms throw at
> `IntegralPack constructor (private)` by design rather than returning a wrong number.

The spin counts play the same role they did in UHF. The new ingredient is the
choice of exchange-correlation functional and the numerical grid used to
evaluate it. The example uses LSDA because it is the clearest first step.

## How To Read `examples/04_uks.cpp`

[`examples/04_uks.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/04_uks.cpp) is a compact
survey of what this method changes. The helium atom shows that a closed-shell
singlet fits naturally into the same framework. The hydrogen and lithium atoms
show that the same code also handles open-shell occupations. The
$\mathrm{H}_2$ scan then lets you compare UKS-LSDA against UHF on the same
minimal-basis molecular problem.

Read the printed decomposition carefully. It teaches a habit that is worth
keeping in every DFT calculation: when a total energy looks surprising, the
way to debug it is not to stare at the final scalar. It is to ask whether the
one-electron, Coulomb, and exchange-correlation pieces each make sense on their
own, and whether the grid-integrated electron counts are close to the intended
occupations.

## Running The Example

Build and run it with

```bash
cmake -B build -DATV_BUILD_CUDA=OFF -DATV_BUILD_HF=ON -DATV_BUILD_EXAMPLES=ON
cmake --build build --target 04_uks
./build/examples/04_uks
```

The printed helium, hydrogen, lithium, and $\mathrm{H}_2$ cases should make one
point especially clear: once you accept the density-functional point of view,
closed-shell and open-shell systems still fit into the same self-consistent
story.

## What Usually Goes Wrong

The first failure mode is numerical rather than conceptual. If the grid is too
coarse, the integrated electron counts drift and the exchange-correlation
contribution becomes noisy. The second failure mode is conceptual: carrying over
Hartree-Fock intuitions too literally and trying to reconstruct
$E_{\mathrm{xc}}$ from the matrix potential. The third is familiar from UHF:
the wrong `n_alpha` and `n_beta` values quietly define the wrong physical
problem.

If you keep the structure of this chapter in mind, UKS stops looking like an
entirely new field. It is still the same SCF loop you learned in the
introductory chapter, RHF,
and UHF chapters. The new ingredient is simply that one part of the effective
field is no longer an analytic exchange matrix but a numerically integrated
functional of the density.
