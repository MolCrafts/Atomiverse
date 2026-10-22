# Many-Body Potentials

Status tables: [Force Fields roadmap](../roadmap/force-fields.md).

The energy of a bond can depend on its surroundings. Metals respond to
coordination, while covalent crystals prefer particular angles. A central
pair potential cannot describe every such effect, however carefully its
parameters are adjusted.

**Learning goals.** Identify environment dependence, differentiate an
embedding energy, and design separate metallic and covalent validation cases.

**Before you begin.** Pair forces in [Molecular Dynamics](ch10-classical-md.md)
and the chain rule in [Mathematical Tools](mathematical-tools.md).

## From pairs to environments

The embedded-atom method writes

$$U=\sum_A F(\rho_A)+\frac12\sum_A\sum_{B\ne A}\phi(r_{AB}),
\qquad \rho_A=\sum_{B\ne A}f(r_{AB}).$$

$\rho_A$ is a model of the local environment, not necessarily an observable
electron density. If $F$ were linear, its contribution could be rewritten
as a pair sum. Nonlinearity is what makes embedding genuinely many-body.
For the Sutton–Chen form $F(\rho)=-\epsilon c\sqrt\rho$, positive parameters
and $\rho>0$ give $F'<0$ and $F''>0$: this function is convex.

The derivative associated with a pair includes
$\phi'(r_{AB})+[F'(\rho_A)+F'(\rho_B)]f'(r_{AB})$. Both environments respond
when the pair separation changes. Omitting either response gives an
incorrect force even when the energy formula has been implemented correctly.

Stillinger–Weber models add angular interactions. A tetrahedral angular
penalty contains $(\cos\theta+1/3)^2$, minimized near $109.47$ degrees.
This expresses a preference for a local geometry; it does not make one
parameter set transferable to all tetrahedrally bonded materials.

## Limits of pair models

### Crystal structure

For equivalent sites in a perfect monatomic crystal, the pair energy per
atom is $E/N=\tfrac12\sum_{j\ne i}\phi(r_{ij})$. Neighbor distances and
their multiplicities both matter. A simple attractive-well model often
favors highly coordinated arrangements, but there is no theorem that every
isotropic pair potential must prefer close packing.

Indeed, deliberately designed isotropic pair potentials can stabilize
diamond-like open structures; the example by Rechtsman, Stillinger, and
Torquato is listed in the references. This does not make a radial pair model
a transferable description of covalent chemistry. It shows why a useful
physical motivation should not be mistaken for an impossibility theorem.

In silicon and diamond, tetrahedral coordination motivates interactions
that explicitly depend on local geometry. A Stillinger–Weber angular term
provides one such representation. Its suitability should be tested against
the structures and deformations relevant to the proposed calculation.

### Elastic response

Elastic constants describe the curvature of energy with respect to small
strains. For a monatomic cubic Bravais lattice with central pair interactions,
the zero-temperature, zero-pressure Born elastic constants satisfy

$$C_{12}=C_{44}.$$

This Cauchy relation has specific assumptions. Pressure corrections, thermal
fluctuations, and internal relaxation in crystals with a multi-atom basis
require separate treatment. Inversion symmetry alone is not sufficient to
justify the displayed equality for every measured elastic modulus.

The inherited room-temperature values below illustrate why independent
elastic measurements are useful when assessing a potential. They should
not be treated as a temperature-matched proof of a zero-temperature identity.

| Metal | $C_{12}$ (GPa) | $C_{44}$ (GPa) | $C_{12}/C_{44}$ |
|---|---|---|---|
| Cu | 121 | 75 | 1.6 |
| Ag | 93 | 46 | 2.0 |
| Au | 157 | 42 | 3.7 |

The numerical values are retained from the original manual, which attributes
them to Kittel. Their experimental conditions and provenance should be
checked before quantitative fitting. Matching a cohesive energy and a
lattice constant does not determine the full elastic response.

### Vacancy formation

Removing an atom requires a reference for where it goes. At zero pressure,
a common large-cell definition relative to a bulk reservoir is

$$E_v=E_{N-1}^{\mathrm{defect}}-\frac{N-1}{N}E_N^{\mathrm{bulk}}.$$

For an ideal monatomic pair crystal, let $e_b<0$ be the energy per atom
relative to separated atoms. The incident pair sum at one equivalent site
is $2e_b$. Removing that atom without relaxing any remaining coordinates
therefore gives $E_{N-1}^{\mathrm{unrelaxed}}=E_N-2e_b$, and the definition
above yields $E_v^{\mathrm{unrelaxed}}=-e_b=E_{\mathrm{coh}}$ in the
appropriate bulk limit.

The qualification *unrelaxed* matters. Atomic and cell relaxation, finite
cell interactions between defects, and reservoir conventions can change the
comparison. There is no universal relaxed-vacancy-to-cohesive-energy ratio
for all metals. In an embedding model, removal also changes the environment
energy of neighboring atoms. Vacancy formation therefore provides a useful
test beyond reproducing the bulk cohesive energy.

## Implementation scope

The theory of EAM and angular potentials is distinct from the capabilities
of a particular implementation. Element coverage, MPI, CUDA and parameter
loading are stated on the
[many-body potentials method page](methods/md-manybody-potentials.md#scope),
next to the regression case that pins both potentials.

## Practice plan

Use two different materials because they expose different model responses.

1. For Sutton–Chen copper, scan the lattice constant of a periodic fcc sample.
   Report energy per atom, the parameter set, and the cutoff. Distinguish
   quantities used in fitting from genuinely independent validation targets.
2. Remove one atom from a converged bulk cell. Evaluate the unrelaxed vacancy
   energy with the stated bulk reference, then relax the defect and repeat
   at larger cell sizes. Do not compare these two quantities as if they were
   the same observable.
3. For Stillinger–Weber silicon, perturb a bond length and a bond angle.
   Compare diamond and a competing structure at specified density; then
   distinguish that fixed-density test from a comparison of separately
   relaxed phases at equal pressure.
4. Check forces at nonsymmetric configurations using several finite-difference
   displacements. Symmetry can cancel errors and hide a missing contribution.

Record the original and perturbed configurations so that a failed test can
be reproduced. The short setup sketch is retained as historical interface
material; the later explicit force-stage examples show the intended binding
pattern for those examples, not a promise of compatibility with every revision.

## Setup fragments

These original fragments and build notes are retained for reference.
They may omit initialization; use the complete model example when running code.

```cpp
sim.set_evaluator(cpu::EAM{/* Sutton-Chen Cu lookup */});
// or
sim.set_evaluator(cpu::StillingerWeber{/* Si parameters */});
sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
```

## Embedding energy

### Model functions

EAM (Daw & Baskes, *Phys. Rev. B* **29**, 6443, 1984) starts from a result of
density-functional theory: the energy of an impurity embedded in a host is, to a
good approximation, a **function of the host electron density at the impurity site**.

Now treat *every* atom as an impurity embedded in the electron density created by
all the others:

$$
E = \underbrace{\sum_i F(\rho_i)}_{\text{embedding}}
  + \underbrace{\tfrac{1}{2}\sum_i\sum_{j\ne i}\phi(r_{ij})}_{\text{pair contribution}},
\qquad
\rho_i = \sum_{j \ne i} f(r_{ij}).
$$

The three supplied functions are the pair term $\phi(r)$, the environment
contribution $f(r)$, and the embedding term $F(\rho)$. The local quantity
$\rho_i$ is constructed by summing $f$ over neighbors; it is not a fourth
independent function or necessarily a directly measurable electron density.

**Everything hinges on $F$ being nonlinear.** If $F(\rho) = a\rho$ were linear, then
$\sum_i F(\rho_i) = a\sum_i\sum_j f(r_{ij})$ — a sum over pairs, and EAM would collapse
back into a pair potential. It is the **curvature** of $F$ that makes the energy
many-body: it makes the cost of a bond depend on how many other bonds the atom
already has.

$F=-\epsilon c\sqrt{\rho}$ is convex for positive parameters and density:
$F''>0$ while $F'<0$. Additional neighbors provide diminishing embedding
binding. This environment dependence removes some restrictions of central
pair models, but vacancy energies, elastic relations, and structural
stability still require validation for the chosen parameterization.

### The Sutton–Chen form

Atomiverse ships the analytic **Sutton–Chen** parameterization (*Philos. Mag. Lett.*
**61**, 139, 1990), a long-range Finnis–Sinclair form:

$$
\phi(r) = \varepsilon\left(\frac{a}{r}\right)^{n}, \qquad
f(r) = \left(\frac{a}{r}\right)^{m}, \qquad
F(\rho) = -\varepsilon\, c\, \sqrt{\rho}.
$$

The square root is the nonlinearity, and it has a real physical pedigree: in a
tight-binding picture the bond energy of an atom scales as the square root of its
coordination (the band width scales as $\sqrt{Z}$), which is precisely
$F \propto -\sqrt{\rho}$. Finnis and Sinclair derived it that way; Sutton and Chen
gave it a convenient long-range analytic form.

Copper (`cpu/potentials/eam_data/eam_tables.h`):

| $n$ | $m$ | $\varepsilon$ (eV) | $c$ | $a$ (Å) |
|---|---|---|---|---|
| 9 | 6 | 1.2382 × 10⁻² | 39.432 | 3.61 |

### Running it

`cpu::EAM` is a force component rather than a `Pair`. The detailed example
therefore binds it explicitly at the force phase instead of relying on the
short introductory `set_evaluator` sketch:

```cpp
#include "atomiverse/cpu/potentials/eam.h"

const double cutoff = 10.0;  // (Å)

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{cutoff, 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{1.0});           // fs
sim.on(md::Phase::force)
   .add(cpu::EAM{cpu::EAMPotential{molrs::Element::Cu, "sutton-chen"}, cutoff});
sim.build();
sim.run(frame, 1000);
```

`EAMPotential(molrs::Element, name)` is a **lookup constructor**: it reads the in-tree static
table and throws on an element or parameterization it does not have. There is no
`setfl` / `.eam.alloy` parser anywhere in the engine, and nothing reads a data file
at build, test, or run time.

A note on the cutoff. Sutton–Chen is a **power law** — it has no intrinsic range, so
the cutoff is a convergence parameter you own. It matters; see the results below.

### Results — copper

`eam_cu` scans the fcc lattice constant, refines the minimum
parabolically, and compares against the two properties Sutton and Chen *fitted* to:

| Quantity | Atomiverse | Reference | Error |
|---|---|---|---|
| Lattice constant $a_0$ | 3.6121 Å | 3.61 Å | **0.06 %** |
| Cohesive energy $E_{\text{coh}}$ | 3.483 eV/atom | 3.54 eV/atom | 1.6 % |

The reported lattice constant is a useful consistency check for this fitted
model. Agreement of one property does not establish that every parameter,
force, or implementation detail is correct.

The cutoff sequence below shows increasing binding for this example. In the
Sutton–Chen model, truncation removes both positive pair contributions and
negative embedding contributions. Their balance, rather than the sign of
one term alone, determines the net energy error. Study the actual cutoff
sequence instead of assuming a universal monotonic trend.

| Cutoff (Å) | $E_{\text{coh}}$ (eV) | Deficit |
|---|---|---|
| 5 | 3.3725 | −4.7 % |
| 8 | 3.4651 | −2.1 % |
| 10 | 3.4828 | −1.6 % |

These retained values approach the quoted bulk reference as the cutoff
increases. They motivate a convergence study, but three historical values
do not establish the infinite-cutoff limit. If a trend differs, investigate
the state, units, parameterization, truncation convention, and numerical
implementation before assigning a unique cause.

## Angular interactions

EAM fixes metals, where bonding is non-directional and the environment enters through
a scalar density. It does *not* fix silicon. A tetrahedral $sp^3$ network is held
together by bonds with **directions**, and no function of $\rho_i$ alone can encode
"109.47° between neighbours".

Stillinger and Weber (*Phys. Rev. B* **31**, 5262, 1985) took the direct route: add a
term that depends explicitly on the **angle** at each atom.

$$
E = \sum_{i<j}\phi_2(r_{ij}) \;+\; \sum_i \sum_{j<k} \phi_3(r_{ij}, r_{ik}, \theta_{jik})
$$

$$
\phi_2(r) = A\varepsilon\left[B\left(\frac{\sigma}{r}\right)^{p} - \left(\frac{\sigma}{r}\right)^{q}\right] \exp\!\left(\frac{\sigma}{r - a\sigma}\right)
$$

$$
\phi_3 = \lambda\varepsilon\,\bigl(\cos\theta_{jik} - \cos\theta_0\bigr)^{2}\,
        \exp\!\left(\frac{\gamma\sigma}{r_{ij} - a\sigma}\right)
        \exp\!\left(\frac{\gamma\sigma}{r_{ik} - a\sigma}\right)
$$

Read $\phi_3$ as an **energy penalty for having the wrong bond angle**. It is zero when
$\theta = \theta_0$ and positive otherwise, with $\cos\theta_0 = -1/3$, i.e.
$\theta_0 = 109.47°$ — the tetrahedral angle. The two exponentials switch the penalty
off smoothly as either leg leaves the cutoff.

Note the cutoff behaviour: both terms contain $\exp[\sigma/(r - a\sigma)]$, which
$\to 0$ as $r \to a\sigma$ from below. The potential and **all its derivatives** vanish
identically at $r = a\sigma$. Unlike Sutton–Chen, there is no truncation error at all —
the cutoff is exact, and $a\sigma = 3.7712$ Å *is* the model.

Silicon (`cpu/potentials/sw_data/sw_tables.h`):

| $\varepsilon$ (eV) | $\sigma$ (Å) | $A$ | $B$ | $p$ | $q$ | $a$ | $\lambda$ | $\gamma$ | $\cos\theta_0$ |
|---|---|---|---|---|---|---|---|---|---|
| 2.1683 | 2.0951 | 7.049556277 | 0.6022245584 | 4 | 0 | 1.80 | 21.0 | 1.20 | −1/3 |

### Running it

```cpp
#include "atomiverse/cpu/potentials/sw.h"

const cpu::SWPotential pot{molrs::Element::Si, "stillinger-weber"};

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{pot.cutoff(), 0.3});
sim.set_integrator(cpu::md::VelocityVerlet{1.0});
sim.on(md::Phase::force).add(cpu::StillingerWeber{pot});
sim.build();
sim.run(frame, 1000);
```

The cutoff comes from the potential (`pot.cutoff()` = $a\sigma$) rather than from you,
because for SW it is not a convergence knob — it is a defined property of the model.

### Building a diamond-cubic frame

The eight-atom conventional cell, which you will need for silicon and for any other
diamond-structure material:

```cpp
const double a0 = 5.431;  // (Å)
const double basis[8][3] = {
    {0.00, 0.00, 0.00}, {0.25, 0.25, 0.25},
    {0.50, 0.50, 0.00}, {0.75, 0.75, 0.25},
    {0.50, 0.00, 0.50}, {0.75, 0.25, 0.75},
    {0.00, 0.50, 0.50}, {0.25, 0.75, 0.75},
};

cpu::Frame frame(8 * nc * nc * nc);
auto x = frame.span(kAtom_x);
auto y = frame.span(kAtom_y);
auto z = frame.span(kAtom_z);
auto mass = frame.span(kAtom_mass);
auto type = frame.span(kAtom_type);
int idx = 0;
for (int ix = 0; ix < nc; ++ix)
  for (int iy = 0; iy < nc; ++iy)
    for (int iz = 0; iz < nc; ++iz)
      for (int k = 0; k < 8; ++k) {
          x[idx] = (ix + basis[k][0]) * a0;
          y[idx] = (iy + basis[k][1]) * a0;
          z[idx] = (iz + basis[k][2]) * a0;
          mass[idx] = 28.0855;
          type[idx] = 14;   // Z
          ++idx;
      }
frame.box = cpu::Box::cubic(a0 * nc);
```

Two atoms per lattice point, offset by ¼ along the body diagonal: that is what makes
the structure tetrahedral rather than close-packed.

### Results — silicon

`sw_si`:

| Quantity | Atomiverse | Reference | Error |
|---|---|---|---|
| Lattice constant $a_0$ | 5.4312 Å | 5.431 Å | **0.00 %** |
| Cohesive energy $E_{\text{coh}}$ | 4.3364 eV/atom | 4.3363 eV/atom | **0.00 %** |

Four to five significant figures — and it *should* be that good, precisely because SW
is strictly cutoff-terminated. There is no truncated tail to hide an error in, so the
agreement is a genuine check on all ten transcribed parameters.

Reproducing fitted quantities is a necessary consistency check, not a proof
of the implementation. A complementary test compares structures that differ
in their local coordination:

| Structure (same number density) | $E$ (eV/atom) |
|---|---|
| **Diamond** (coordination 4) | **−4.3364** |
| fcc (coordination 12) | −3.4520 |

In this retained comparison, diamond is lower by about 0.88 eV per atom at
the stated equal density. The result is evidence for this parameterized
model under those conditions, not an impossibility result for all radial
pair potentials. Separately testing the angular energy and its derivatives
is still necessary: a wrong derivative can coexist with a correct energy
ordering. Phase stability at equal pressure also requires volume relaxation
and, at finite temperature, free-energy information.

## Forces and neighbor data

### Two passes for EAM

You cannot compute an EAM force in one sweep. The force on $i$ is

$$
\mathbf{F}_i = -\sum_{j\ne i}\Bigl[\phi'(r_{ij}) + \bigl(F'(\rho_i) + F'(\rho_j)\bigr) f'(r_{ij})\Bigr]\hat{\mathbf{r}}_{ij},
$$

which needs $F'(\rho_j)$ — the embedding derivative **at the neighbour**. So $\rho$ must
be known everywhere before any force is evaluated:

1. **Pass 1** — for every atom, accumulate $\rho_i = \sum_j f(r_{ij})$, then $F'(\rho_i)$.
2. **Pass 2** — walk the pairs again and build the forces.

Both passes live inside one `launch()`. Note the structure of that force expression:
it is *not* pairwise, even though it is written as a sum over pairs, because the
coefficient of each pair term depends on the environments of both endpoints.

### The full neighbor list

The environment sum needs contributions from all neighbors. With a full
neighbor list, each atom's row can provide that sum directly. With a half
list, each stored pair must instead update the environments of both endpoints,
with the required communication in a distributed calculation. A half-list
algorithm is therefore possible; simply summing each incomplete row as if it
were complete is not correct.

An angular term also requires complete neighbor information around its
central atom so that neighbor pairs can be enumerated. This is a requirement
on the available information, not a universal requirement that every
many-body implementation use the same list representation.

The inherited implementation sketch requests a full list in `build()`:

```cpp
ctx.runtime.md.full_list_requested = true;   // -> CellList::refresh_full
```

When extending an implementation, check its neighbor and communication
contracts rather than copying this internal field assignment in isolation.
A stable-looking trajectory does not rule out a missing environment
contribution. Compare both energies and forces at explicit test geometries.

### Force verification

Both potentials are checked against a central difference of the energy the component
actually reports:

$$
F_{i\alpha} \stackrel{?}{=} -\frac{E(x_{i\alpha}+h) - E(x_{i\alpha}-h)}{2h}
$$

to 10⁻⁶ eV/Å (EAM) and 10⁻⁵ eV/Å (SW), in `tests/cpu/potentials/test_eam.cpp` and
`test_sw.cpp`.

This is not ceremony. An **energy-only** test cannot detect a force that is the
gradient of a *different* functional than the energy — the two are checked
independently and each looks fine on its own. Atomiverse shipped exactly that bug in
its GPW code once, and it survived seventeen commits. During the work in this chapter
the FD oracle caught a real error on its first run: the EAM force had been halved by
false analogy with the (correctly halved) energy.

For the illustrated owner-atom accumulation scheme, the counting rules are:

- The full list visits each **pair** twice. Pair energy and pair virial
  contributions accumulated on both visits therefore carry a factor of one
  half. Do not halve an embedding energy counted only once per central atom.
- The **force** is *not* halved: each visit writes only atom $i$'s force, so the
  $i$-visit must deposit the whole force that the pair exerts on $i$.
- A **triplet** is different again: it is enumerated once per **vertex**, so the
  three-body term is not halved at all. A triplet has a distinguished atom; a pair
  does not.

For a translation-invariant three-body term, differentiating the two leg
coordinates gives $\mathbf F_j$ and $\mathbf F_k$. The central force obeys
$\mathbf F_i=-(\mathbf F_j+\mathbf F_k)$. This relation can be used for
accumulation and checked independently against coordinate finite differences.

---

## Component responsibilities

`cpu::Pair` — and the backend-agnostic `PairKernel` behind it — models exactly one
thing: `fpair(r²)`, a scalar force magnitude as a function of a squared separation. It
holds **no per-atom state** and knows nothing of triplets.

The inherited design represents the environment-dependent computation with
a separate component rather than placing its state inside the stateless pair
kernel. This is an architectural choice, not a requirement on every possible
implementation of the physical model. The following schematic fragment
illustrates that separation:

```cpp
class EAM : public cpu::Component, public cpu::NeighborConsumer { ... };

sim.on(md::Phase::force).add(cpu::EAM{...});   // explicit, not set_evaluator
```

`set_evaluator` is a singular slot (a second pair style throws), and `Pair::launch()` is
`final`. Explicit `.on(phase).add()` is the general road, and it is not a workaround —
it is the API for anything that is not literally a pair style.

---

## What the model establishes

- A radial pair model has structural restrictions, but it is not universally
  incapable of stabilizing an open crystal. Assess its intended physical
  scope and the assumptions of any elasticity or defect argument.
- A nonlinear embedding function makes the energy depend on local
  environments. Both endpoints' environment derivatives contribute to the
  force associated with a pair separation.
- Angular terms provide an explicit way to favor tetrahedral geometry.
  A parameterization validated for silicon is not thereby validated for
  another tetrahedrally coordinated material.
- The algorithm must account for all required neighbor contributions and
  avoid double counting. Full- and half-list implementations may organize
  that work differently.
- Agreement with a fitted number, a plausible energy ordering, or a smooth
  trajectory is useful but incomplete evidence. Combine independent force,
  cutoff, size, structure, and defect checks appropriate to the question.

## References

- M. S. Daw and M. I. Baskes, "Embedded-atom method: Derivation and application to
  impurities, surfaces, and other defects in metals", *Phys. Rev. B* **29**, 6443 (1984).
- M. W. Finnis and J. E. Sinclair, "A simple empirical N-body potential for transition
  metals", *Philos. Mag. A* **50**, 45 (1984).
- A. P. Sutton and J. Chen, "Long-range Finnis-Sinclair potentials",
  *Philos. Mag. Lett.* **61**, 139 (1990).
- F. H. Stillinger and T. A. Weber, "Computer simulation of local order in condensed
  phases of silicon", *Phys. Rev. B* **31**, 5262 (1985).
- M. C. Rechtsman, F. H. Stillinger, and S. Torquato,
  ["Synthetic diamond and wurtzite structures self-assemble with isotropic pair interactions"](https://arxiv.org/abs/0709.3807),
  *Phys. Rev. E* **75**, 031403 (2007), DOI: 10.1103/PhysRevE.75.031403.
- C. Kittel, *Introduction to Solid State Physics*, 8th ed. — elastic constants
  (Table 3, Ch. 3) and cohesive energies (Ch. 3).

## Questions for reflection

Show how linear embedding reduces to pair interactions. What changes in the force on a pair when a third atom moves without changing that pair distance?
