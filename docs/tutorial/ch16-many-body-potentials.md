# Chapter 16. Many-Body Potentials

Pair potentials are qualitatively wrong for metals and covalent solids.
This chapter uses `cpu::EAM` and `cpu::StillingerWeber`.

```cpp
sim.set_evaluator(cpu::EAM{/* Sutton-Chen Cu lookup */});
// or
sim.set_evaluator(cpu::StillingerWeber{/* Si parameters */});
sim.set_neighbor(cpu::md::NeighborList<>{cutoff, skin});
```

Pair LJ NVE conservation ([Chapter 10](ch10-classical-md.md)) proves the MD
pipeline; many-body styles reuse the same neighbor list and force reduction.
Always re-check FD forces after parameter changes — the two-pass EAM force is a
common source of sign/density bugs.

Every classical potential in this book so far has been a **pair potential**: the
energy is a sum over separations,

$$
E = \tfrac{1}{2}\sum_i \sum_{j \ne i} \phi(r_{ij}),
$$

and the force between two atoms depends only on how far apart *they* are. Chapter 10
built a Lennard-Jones argon fluid this way, and for argon it is the right model —
a closed-shell noble gas really is held together by pairwise dispersion.

For a metal or a covalent solid it is not just inaccurate; it is **qualitatively
wrong**. This chapter explains why, then builds the two potentials that fix it:
the **embedded-atom method** (EAM) for metals and the **Stillinger–Weber** (SW)
potential for covalent silicon.

Runnable end-to-end cases, checked against the original papers:

- [`regressions/md/eam_cu/`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/md/eam_cu/reg_eam_cu.cpp) — EAM copper
- [`regressions/md/sw_si/`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/md/sw_si/reg_sw_si.cpp) — SW silicon

---

## 1. Why Pair Potentials Fail

### The close-packing catastrophe

Take a pair potential — *any* pair potential — and ask which crystal structure it
prefers. The energy per atom is

$$
\frac{E}{N} = \tfrac{1}{2}\sum_{j \ne i} \phi(r_{ij}),
$$

a sum of attractive contributions, one per neighbour. Adding another neighbour near
the minimum of $\phi$ *lowers* the energy. So a pair potential is always happiest
with as many neighbours as geometry allows: it favours **close packing** (fcc or hcp,
coordination 12).

Silicon has coordination **4**. Carbon (diamond) has coordination 4. These are
extremely *open* structures — most of the space is empty. **No pair potential can
explain them.** The reason silicon is tetrahedral is that its bonds are $sp^3$
hybrids pointing at 109.47° to each other, and bond *directions* matter. A pair
potential cannot see an angle: it only ever sees $r_{ij}$.

### The Cauchy relation

There is a sharper, quantitative version of the same failure. For any crystal with
inversion symmetry whose atoms interact through central pair forces, the elastic
constants must satisfy the **Cauchy relation**

$$
C_{12} = C_{44}.
$$

This is not an approximation — it follows from the pair form. And it is violated by
essentially every real metal:

| Metal | $C_{12}$ (GPa) | $C_{44}$ (GPa) | $C_{12}/C_{44}$ |
|---|---|---|---|
| Cu | 121 | 75 | 1.6 |
| Ag | 93 | 46 | 2.0 |
| Au | 157 | 42 | 3.7 |

(Room-temperature values; see Kittel, *Introduction to Solid State Physics*, Table 3.)
A pair potential fitted to a metal's cohesive energy and lattice constant will get
its elastic response badly wrong, no matter what functional form you choose.

### The vacancy paradox

A third symptom, and the one that historically drove the development of EAM. Remove
one atom from a crystal. In a pair model, the energy cost is exactly the bonds you
broke — so the vacancy formation energy should equal the cohesive energy,
$E_v/E_{\text{coh}} = 1$. Experimentally, for fcc metals, that ratio is about
**0.3**.

Why? Because when an atom leaves, its neighbours do not simply lose a bond; they
find themselves in a **thinner electron density**, and the bonds they still have
*get stronger*. A bond's strength depends on its environment. That is the physical
insight the embedded-atom method formalizes.

---

## 2. The Embedded-Atom Method

### The physics

EAM (Daw & Baskes, *Phys. Rev. B* **29**, 6443, 1984) starts from a result of
density-functional theory: the energy of an impurity embedded in a host is, to a
good approximation, a **function of the host electron density at the impurity site**.

Now treat *every* atom as an impurity embedded in the electron density created by
all the others:

$$
E = \underbrace{\sum_i F(\rho_i)}_{\text{embedding}}
  + \underbrace{\tfrac{1}{2}\sum_i\sum_{j\ne i}\phi(r_{ij})}_{\text{core–core repulsion}},
\qquad
\rho_i = \sum_{j \ne i} f(r_{ij}).
$$

Three functions: $f(r)$, the electron density one atom contributes at distance $r$;
$\rho_i$, the total density at atom $i$; and $F(\rho)$, the energy of embedding an
atom into density $\rho$.

**Everything hinges on $F$ being nonlinear.** If $F(\rho) = a\rho$ were linear, then
$\sum_i F(\rho_i) = a\sum_i\sum_j f(r_{ij})$ — a sum over pairs, and EAM would collapse
back into a pair potential. It is the **curvature** of $F$ that makes the energy
many-body: it makes the cost of a bond depend on how many other bonds the atom
already has.

$F$ is concave ($F'' > 0$, $F' < 0$): the first neighbours buy a lot of binding, later
ones buy less. This single fact fixes all three failures above. It gives $E_v/E_{\text{coh}} < 1$
(a neighbour that loses density becomes more tightly bound), it breaks the Cauchy
relation, and it correctly balances close packing against the core repulsion.

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

`cpu::EAM` is a force component like any other. It is **not** a `Pair` — see §5 —
so it is bound explicitly rather than through `set_evaluator`:

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

`regressions/md/eam_cu` scans the fcc lattice constant, refines the minimum
parabolically, and compares against the two properties Sutton and Chen *fitted* to:

| Quantity | Atomiverse | Reference | Error |
|---|---|---|---|
| Lattice constant $a_0$ | 3.6121 Å | 3.61 Å | **0.06 %** |
| Cohesive energy $E_{\text{coh}}$ | 3.483 eV/atom | 3.54 eV/atom | 1.6 % |

The lattice constant is the strong statement: it lands within 0.06 %, which says the
functional form and every parameter are right.

The cohesive energy is systematically ~1.6 % **low**, and the deficit is *understood*
rather than tolerated. Truncating a power law throws away density that is still
contributing; less density means less binding, because $F(\rho) = -\varepsilon c\sqrt{\rho}$
decreases monotonically. So the error must have this sign, and it must shrink as the
cutoff grows — which is exactly what happens:

| Cutoff (Å) | $E_{\text{coh}}$ (eV) | Deficit |
|---|---|---|
| 5 | 3.3725 | −4.7 % |
| 8 | 3.4651 | −2.1 % |
| 10 | 3.4828 | −1.6 % |

Converging upward toward 3.54, as the truncation argument requires. **When a number
is off, find out whether its error has the sign the physics predicts.** If it does,
you understand your model; if it does not, you have a bug.

---

## 3. Stillinger–Weber: Making Angles Matter

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

`regressions/md/sw_si`:

| Quantity | Atomiverse | Reference | Error |
|---|---|---|---|
| Lattice constant $a_0$ | 5.4312 Å | 5.431 Å | **0.00 %** |
| Cohesive energy $E_{\text{coh}}$ | 4.3364 eV/atom | 4.3363 eV/atom | **0.00 %** |

Four to five significant figures — and it *should* be that good, precisely because SW
is strictly cutoff-terminated. There is no truncated tail to hide an error in, so the
agreement is a genuine check on all ten transcribed parameters.

But the two fitted numbers are **not** the interesting test. Reproducing numbers a
model was fitted to only proves you implemented the model. The test that proves the
**three-body term is doing physics** is structural:

| Structure (same number density) | $E$ (eV/atom) |
|---|---|
| **Diamond** (coordination 4) | **−4.3364** |
| fcc (coordination 12) | −3.4520 |

Diamond wins by 0.88 eV/atom. Compare at *equal density*, so the contest is about
structure and not volume. **A pair potential cannot produce this row of the table** —
it would put fcc lower, every time. Delete the three-body term, flip its sign, or break
its angular derivative, and fcc wins and the regression fails. That is what makes it a
test of three-bodyness rather than of bookkeeping.

---

## 4. How the Force Is Computed

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

$\rho_i$ is a sum over **all** neighbours of $i$. A **half** neighbor list — the default,
and what a pair potential wants — stores each pair once, so atom $i$ only "sees" the
half of its neighbours that happened to land in its own row. Summing over that gives
you half a density and a wrong answer.

Likewise, SW's angular term at vertex $i$ enumerates every unordered pair $(j,k)$ of
$i$'s neighbours — impossible unless $i$'s row is complete.

Both components therefore request a full list, in `build()`:

```cpp
ctx.runtime.md.full_list_requested = true;   // -> CellList::refresh_full
```

If you write a new many-body style, this is the one line you must not forget. Symptom
if you do: energies and forces that are wrong by roughly a factor related to
coordination, and — because the error is smooth — an MD run that looks stable while
being physically meaningless.

### Every force is finite-difference gated

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

The subtlety, worth internalizing because it will bite you in any many-body kernel over
a full list:

- The full list visits each **pair** twice. The **energy** and the **virial** are
  therefore halved — both visits accumulate into one total.
- The **force** is *not* halved: each visit writes only atom $i$'s force, so the
  $i$-visit must deposit the whole force that the pair exerts on $i$.
- A **triplet** is different again: it is enumerated once per **vertex**, so the
  three-body term is not halved at all. A triplet has a distinguished atom; a pair
  does not.

For the three-body force, get the two leg forces $\mathbf{F}_j, \mathbf{F}_k$ by
differentiation and then take the vertex force from translation invariance,
$\mathbf{F}_i = -(\mathbf{F}_j + \mathbf{F}_k)$. Differentiating it independently just
gives you a third chance to get a sign wrong.

---

## 5. Why These Are Not `Pair` Components

`cpu::Pair` — and the backend-agnostic `PairKernel` behind it — models exactly one
thing: `fpair(r²)`, a scalar force magnitude as a function of a squared separation. It
holds **no per-atom state** and knows nothing of triplets.

A many-body potential is therefore not an *extension* of the pair abstraction; it is a
**sibling** of it. Trying to widen `Pair` to accommodate EAM would have deformed the
one abstraction that is currently clean and fast. Instead both styles follow the
pattern `CoulLong` already established:

```cpp
class EAM : public cpu::Component, public cpu::NeighborConsumer { ... };

sim.on(md::Phase::force).add(cpu::EAM{...});   // explicit, not set_evaluator
```

`set_evaluator` is a singular slot (a second pair style throws), and `Pair::launch()` is
`final`. Explicit `.on(phase).add()` is the general road, and it is not a workaround —
it is the API for anything that is not literally a pair style.

---

## 6. Current Limitations

- **Single-rank only.** Both components **throw** under MPI. EAM's second pass needs
  $F'(\rho_j)$ for a *ghost* $j$, and a ghost's density cannot be completed locally
  unless the halo reaches twice the cutoff. The engine performs exactly one forward
  halo exchange (before the neighbor build) and one reverse exchange (after the force
  phase, carrying only forces) — there is no communication point *inside* the force
  phase for a derived per-atom quantity. Rather than silently compute wrong forces at
  rank boundaries, the components refuse to build. This is being fixed properly, in the
  communication layer, rather than worked around here.
- **Single element.** Alloy cross-terms (the `setfl` mixing rules) are not implemented.
- **No CUDA mirror yet.**
- **Tersoff, MEAM, ReaxFF** are not implemented. SW establishes the three-body pathway
  they would build on.

---

## 7. What to Take Away

1. A pair potential cannot describe a metal or a covalent solid — not "less
   accurately", but **qualitatively wrongly**. It predicts close packing for silicon,
   $C_{12} = C_{44}$ for every cubic crystal, and a vacancy energy three times too
   large.
2. **EAM** makes bond strength depend on the local **electron density** through a
   nonlinear embedding function $F(\rho)$. The nonlinearity *is* the many-body physics;
   a linear $F$ collapses back to a pair potential.
3. **Stillinger–Weber** makes energy depend on bond **angles**, which is the only way
   to prefer an open tetrahedral network over close packing.
4. Many-body styles need the **full** neighbor list, and their forces need
   **finite-difference verification** — energy agreement alone can hide a wrong force
   indefinitely.
5. When your number is off, check whether the error has the sign the physics predicts.
   A 1.6 % cohesive-energy deficit that shrinks monotonically with cutoff is a
   truncated power-law tail. The same 1.6 % that does not move when you change the
   cutoff is a bug.

## References

- M. S. Daw and M. I. Baskes, "Embedded-atom method: Derivation and application to
  impurities, surfaces, and other defects in metals", *Phys. Rev. B* **29**, 6443 (1984).
- M. W. Finnis and J. E. Sinclair, "A simple empirical N-body potential for transition
  metals", *Philos. Mag. A* **50**, 45 (1984).
- A. P. Sutton and J. Chen, "Long-range Finnis-Sinclair potentials",
  *Philos. Mag. Lett.* **61**, 139 (1990).
- F. H. Stillinger and T. A. Weber, "Computer simulation of local order in condensed
  phases of silicon", *Phys. Rev. B* **31**, 5262 (1985).
- C. Kittel, *Introduction to Solid State Physics*, 8th ed. — elastic constants
  (Table 3, Ch. 3) and cohesive energies (Ch. 3).
