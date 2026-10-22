# Density Functional Theory

Status tables: [DFT roadmap](../roadmap/dft.md).

The many-electron wavefunction depends on all electron coordinates. The
electron density depends on just one position. Density functional theory
asks how much of the ground-state problem can be formulated using this
simpler variable without pretending that electron interactions disappear.

**Learning goals.** Explain a functional, identify the role of the density
theorems, and distinguish basis, grid, and functional errors.

**Before you begin.** [Hartree–Fock Theory](ch02-rhf.md) and the spin densities
of [Open-Shell Systems](ch03-open-shell.md).

## What the density theorems establish

A functional maps a whole function to a number. An energy functional thus
assigns energy to a density, not merely to a scalar density value. For the
usual nonrelativistic scalar-potential ground-state problem, the
Hohenberg–Kohn result relates ground-state density to external potential up
to a constant. The simplest argument assumes a nondegenerate ground state;
degeneracy and representability require additional care. The theorem does
not provide a practical approximate exchange–correlation functional.

The Kohn–Sham construction introduces auxiliary noninteracting orbitals to
reproduce the density. Its energy is

$$E=T_s+\int v_{\mathrm{ext}}(\mathbf r)\rho(\mathbf r)\,d\mathbf r
+E_H+E_{\mathrm{xc}}+E_{\mathrm{nuc}}.$$

$T_s$ is the auxiliary kinetic energy and $E_H$ the classical Coulomb energy.
$E_{\mathrm{xc}}$ includes exchange, correlation, and the kinetic-energy
difference from the interacting system. Functional differentiation produces
the potential $v_{\mathrm{xc}}=\delta E_{\mathrm{xc}}/\delta\rho$.
An energy and its derivative are not interchangeable:
$E_{\mathrm{xc}}$ is generally not $\operatorname{Tr}(PV_{\mathrm{xc}})$.

Local approximations use density at each point; gradient approximations
also use its spatial variation. Numerical quadrature replaces integrals
by weighted point samples. Refining that grid cannot supply a missing
diffuse orbital or remove a functional's self-interaction error.

## Kohn–Sham equations

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

## Real-space density

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

## Total energy

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

## Calculation settings

The helium atom shows that a closed-shell singlet fits the same framework.
Hydrogen and lithium show open-shell occupations. The H₂ scan compares
UKS-LSDA against UHF on the same minimal-basis problem. When a total energy
looks surprising, inspect the one-electron, Coulomb, and exchange-correlation
pieces separately, and check that grid-integrated electron counts match the
intended occupations.

## Diagnostics and limitations

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

## Practice plan

Use helium before a molecular scan: one alpha and one beta electron make
the electron count transparent. At fixed basis refine the radial and angular
grid, then tighten SCF. Only afterward compare functionals or enlarge the basis.

The retained local C++ fragment mixes undeclared solver objects and repeated
UKS declarations. It is historical interface material, **not a runnable
program**. The upstream `examples/04_uks.cpp`, inspected on 16 September
2026, uses a complete helium frame and the concrete UKS method. Start from
that complete example at a pinned revision rather than trying to infer
missing declarations from the fragment. No numerical result was rerun for
this prose revision.

## Live Atomic and H₂ Numbers

| System | Method | \(E_\mathrm{total}\) (Ha) |
|--------|--------|--------------------------:|
| He | UKS-LSDA | **−2.771886** |
| He | UHF (same geometry) | −2.807784 |
| H | UKS-LSDA | **−0.435670** |
| Li | UKS-LSDA | **−7.221302** |

H₂ scan: UHF vs UKS-LSDA, STO-3G.

```molplot preset="molplot" theme="auto" width="100%" aspect="16:10"
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

In this retained small-basis comparison, LSDA lies slightly below UHF near
equilibrium. The difference reflects a change of functional and density,
not an independently measured exact correlation energy. At large separation,
the approximate local functional does not recover the same separated-atom
limit as UHF. UHF's broken-symmetry energy must also be distinguished from
an exact spin-pure molecular wavefunction. Tightening SCF does not remove
either model limitation.

The move from RHF to UHF changed the spin structure of the SCF problem but kept
the Hartree-Fock mean field. The move from UHF to UKS changes something deeper:
it keeps the same spin-resolved SCF skeleton, but it replaces explicit
Hartree-Fock exchange with an exchange-correlation functional of the density.

The retained results concern closed-shell helium, open-shell hydrogen and
lithium, and an H₂ dissociation comparison. They illustrate how spin-resolved
KS changes the energy model while retaining a coupled SCF structure.
The local code fragment does not constitute a complete runnable reproduction;
the practice plan explains how to find the complete upstream helium example.

## Numerical quadrature

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

## Example configuration

In the scan, the
basic call looks like this:

```cpp
auto ints = /* use RHF{basis}.run(frame) */;
cpu::scf::UKS uks{cpu::scf::BasisSet::STO3G, n_alpha, n_beta, XCType::LSDA};
config.n_radial = 50;
config.n_angular = 110;
config.max_iter = max_iter;
cpu::scf::UKS uks{cpu::scf::BasisSet::STO3G, n_alpha, n_beta, cpu::scf::XCType::LSDA};
solver.build(ints);
const auto& uks = solver.run(ints);
```

> **Element coverage:** like the other Gaussian-basis tutorials, STO-3G here
> supports elements **H through O (Z = 1–8)**; heavier atoms throw at
> the method constructor path by design rather than returning a wrong number.

The spin counts play the same role they did in UHF. The new ingredient is the
choice of exchange-correlation functional and the numerical grid used to
evaluate it. The example uses LSDA because it is the clearest first step.

## Questions for reflection

How can a density theorem be exact while an LSDA result is approximate? Which study distinguishes functional error from an inadequate integration grid?
