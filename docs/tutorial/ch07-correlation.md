# Chapter 7: Beyond Mean Field — Correlation Methods

| | |
|---|---|
| **Examples** | [`13_mp2.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/13_mp2.cpp), [`14_fci.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/14_fci.cpp) |
| **Results** | [`13_mp2.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/13_mp2.json), [`14_fci.json`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/results/14_fci.json) |

```bash
cmake --build build --target 13_mp2 14_fci
./build/examples/13_mp2
./build/examples/14_fci
```

## Live results

### H₂O / STO-3G MP2 (`13_mp2`)

| | (Ha) |
|---|---:|
| \(E_\mathrm{HF}\) | **−74.96302314** |
| \(E_\mathrm{corr}\) (MP2) | **−0.03554565** (≤ 0 by construction) |
| \(E_\mathrm{MP2}\) | **−74.99856879** |

### H₂ / STO-3G full CI (`14_fci`, \(R=0.7414\) Å)

```molvis {format="xyz" representation="ball-and-stick" controls="view" height="240px"}
2
name=H2 Connct="[0,1]"
H  0.0000  0.0000  0.0000
H  0.7414  0.0000  0.0000
```

| | Atomiverse (Ha) |
|---|---:|
| \(E_\mathrm{RHF}\) | −1.11668439 |
| \(E_\mathrm{FCI}\) | **−1.13727017** |
| \(E_\mathrm{corr}=E_\mathrm{FCI}-E_\mathrm{RHF}\) | **−0.02058579** |

FCI in STO-3G is the **exact** non-relativistic energy inside that two-function
space — the gold reference for how much correlation RHF misses on H₂.

Every electronic-structure method in the previous chapters — Hartree-Fock
([Chapter 2](ch02-rhf.md), [Chapter 3](ch03-open-shell.md)) and Kohn-Sham DFT
([Chapter 5](ch05-kohn-sham-dft.md), [Chapter 6](ch06-hybrid-dft.md)) — is a **mean-field**
theory. Each electron moves in the *average* field of all the others. That
averaging is what makes the SCF loop tractable, and it is also its central
approximation: real electrons avoid each other instantaneously, and mean-field
theory does not fully capture that avoidance.

This chapter explains what mean-field theories miss, sketches the hierarchy of
methods built to recover it, and then puts two of them to work: Atomiverse
implements **closed-shell MP2** (`RHF::mp2`) and an **exact tiny full
CI** (`RHF::fci`) as post-SCF steps on the concrete method. Both require a
converged SCF reference and return a correlation energy — real, runnable, and
validated against reference values. What remains out of scope is the expensive
tail of the hierarchy: coupled cluster, multi-reference, and large-scale CI. DFT
([Chapter 6](ch06-hybrid-dft.md)) stays the engine's affordable workhorse for
correlation at production scale; MP2 and full CI are the wavefunction reference
points that let you *measure* it.

You will learn:

1. **The concept** of electron correlation and the correlation energy.
2. **The hierarchy** — MP2, configuration interaction, coupled cluster — and the
   cost-versus-accuracy trade that orders them.
3. **What Atomiverse provides today**: DFT for production, plus MP2 and tiny full
   CI as the implemented wavefunction reference methods.
4. **Where correlation lives** in the Driver/Component architecture — as post-SCF
   pure functions, reinforcing the same pattern the rest of the book teaches.

## The Concept: What Mean-Field Theory Misses

For a given basis set, Hartree-Fock gives the best possible single Slater
determinant — the best wavefunction of the form "each electron in its own
orbital." The exact non-relativistic energy is lower. The difference is, by
definition, the **correlation energy**:

$$ E_\text{corr} = E_\text{exact} - E_\text{HF}. $$

It is a small fraction of the total energy — typically under 1% — but it is
comparable to, or larger than, the **chemical** energies we care about: reaction
enthalpies, barrier heights, binding energies. Getting total energies to 99%
and still being useless for chemistry is the recurring hazard of this field, and
correlation is why.

Two flavours are worth naming:

- **Dynamic correlation** — the short-range, instantaneous avoidance of
  electrons (the "Coulomb hole"). Captured by mixing in many determinants that
  each promote electrons into virtual orbitals.
- **Static (non-dynamic) correlation** — the near-degeneracy that appears when a
  single determinant is qualitatively wrong (bond breaking, diradicals,
  transition metals). This one breaks single-reference methods and needs
  multi-reference treatments.

Hartree-Fock captures **neither**. Kohn-Sham DFT ([Chapter 6](ch06-hybrid-dft.md))
folds dynamic correlation into the exchange-correlation functional — an
*approximate, implicit* treatment that is astonishingly cost-effective but not
systematically improvable. The wavefunction hierarchy below is the *explicit,
systematically improvable* alternative, bought at steeply rising cost.

## The Hierarchy: MP2, CI, Coupled Cluster

Post-Hartree-Fock methods start from the converged HF orbitals and add
correlation by allowing electrons to occupy the empty (virtual) orbitals. They
form a rough ladder of accuracy and cost, all scaling far worse than the
$\mathcal{O}(N^3)$–$\mathcal{O}(N^4)$ of SCF:

| method | idea | formal cost | role |
|---|---|---|---|
| **MP2** | 2nd-order perturbation theory in the fluctuation potential | $\mathcal{O}(N^5)$ | cheapest correlation; good for dispersion, weak on metals/π |
| **CISD / CCSD** | include single + double excitations (variational / exponential) | $\mathcal{O}(N^6)$ | CCSD is a reliable workhorse for closed-shell molecules |
| **CCSD(T)** | CCSD plus a perturbative triples correction | $\mathcal{O}(N^7)$ | the **"gold standard"** of single-reference chemistry |
| **Multi-reference (CASSCF, MRCI)** | several reference determinants | worse still | for static correlation: bond breaking, diradicals |

The pattern is a treadmill: each rung buys roughly an order of magnitude in the
error at the price of a higher power of the system size. CCSD(T) at the complete
basis-set limit is accurate enough to compete with experiment for small
molecules, and completely intractable for a protein. **Coupled cluster is not
variational** (its energy can dip below the exact energy), which is the price of
its favourable size-consistency. These are the trade-offs a practitioner weighs;
this book's job here is to place them on the map, not to teach how to run them.

## What Atomiverse Provides Today

For production-scale correlation, Atomiverse's answer is **density-functional
theory**. A hybrid functional such as B3LYP or PBE0 ([Chapter 6](ch06-hybrid-dft.md))
delivers much of the accuracy of an MP2/CCSD calculation for a fraction of the
cost, by approximating correlation inside the exchange-correlation functional
rather than constructing it explicitly from excited determinants. For the great
majority of ground-state structures, energies, and forces — and for every
gradient-driven workflow in this manual (AIMD, geometry optimization,
vibrations) — DFT is the method of choice.

On top of the SCF reference, Atomiverse also implements two **wavefunction**
correlation methods as post-SCF steps on the concrete method:

- **Closed-shell MP2** — `rhf.mp2(frame)` returns an `MP2Result` with the
  second-order correlation energy from the `(ia|jb)` MO integral block built by
  `ao2mo_ovov`. For water in STO-3G it recovers E_HF = −74.96302314 Ha →
  E_corr = −0.03554565 Ha → E_MP2 = −74.99856879 Ha (see
  [`examples/13_mp2.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/13_mp2.cpp); every digit is pinned by
  `val_mp2_vs_pyscf`).
- **Exact tiny full CI** — `rhf.fci(frame)` gives the *exact*
  in-basis energy for a handful of orbitals (the "Exact correlation in a finite
  basis" section below); for H₂/STO-3G, E_FCI = −1.1372702 Ha.

What remains out of scope is the expensive tail of the hierarchy: **no coupled
cluster, no multi-reference, no large-scale (Davidson/direct) CI**. If your
science *requires* CCSD(T)-quality correlation or a genuine multi-reference
treatment at production scale, Atomiverse is not the right tool today — but MP2
and full CI give you honest wavefunction reference points to calibrate against.

## Where Correlation Lives in the Architecture

It is worth seeing where a correlation method fits, because the answer
reinforces the architectural through-line of the whole manual — and because
`mp2()` and `fci()` occupy exactly this slot.

A post-Hartree-Fock method is not a new kind of loop. It is a **post-SCF
step**: it consumes the *converged* output of an SCF solve — the occupied and
virtual molecular orbitals, their orbital energies, and the two-electron
integrals transformed into the MO basis — and returns a correlation energy (and,
in principle, a correlated density and gradient).

In Atomiverse's vocabulary that is not a new `Driver` at all. The Gaussian SCF
engine already exposes exactly the inputs such a method needs:

- `cpu::scf::RHF rhf; rhf.mp2(frame);` (or `rhf.fci(frame)`) drives a fresh
  SCF with the dense two-electron (ERI) tensor materialized in the method's
  private workspace — the same integrals the CUDA `RHFFock` kernel reads.
- `rhf.run(frame)` puts the orbitals and their energies on
  `rhf.determinant()` and the total energy on `rhf.energy()`.

A correlation method would therefore be a component that **reads the SCF result
and writes a corrected energy into typed `frame.meta`** — architecturally an
*evaluator that runs after the SCF evaluator*, in the same slot pattern that
[Chapter 13](ch13-observables.md)'s observers occupy after the integrator. It
would reuse the existing integral engine rather than introduce a parallel one,
exactly as the hybrid-gradient routine in [Chapter 6](ch06-hybrid-dft.md) reuses
the ERI machinery. "One loop, many physics" extends cleanly to "one SCF result,
many post-processings" — and `mp2()` / `fci()` are exactly that: post-SCF
members of the concrete method that read the converged `SlaterDeterminant` and
return a correlation energy, in the same family as `forces()`, registered in
no Plan. Both throw `std::invalid_argument` when the SCF reference did not
converge — a correlation correction on top of a non-stationary mean field is
meaningless, and there is no fallback.

## Exact correlation in a finite basis: full CI for tiny systems

There is one place where Atomiverse computes correlation *exactly* — not
approximated by a functional, not truncated at second order, but the true
lowest energy the basis can express. That is **full configuration interaction
(full CI)**, available as a small post-SCF reference solver
(`RHF::fci`) for *tiny* systems only.

**The determinant expansion.** Hartree-Fock uses a single Slater determinant:
each electron in its own molecular orbital, the occupied set fixed. Full CI
lifts that restriction. Given `n_mo` molecular orbitals and `n_alpha`/`n_beta`
electrons per spin, it writes the wavefunction as a linear combination of
*every* determinant obtainable by distributing the electrons over the orbitals
in all possible ways:

$$ |\Psi_\text{FCI}\rangle = \sum_I c_I |D_I\rangle, \qquad
   N_\text{det} = \binom{n_\text{mo}}{n_\alpha}\binom{n_\text{mo}}{n_\beta}. $$

The reference determinant $|D_0\rangle$ is the RHF one; the rest are *excited*
determinants that promote electrons into the virtual (empty) orbitals. The
coefficients $c_I$ are the eigenvector components of the CI Hamiltonian
$H_{IJ} = \langle D_I | \hat H | D_J\rangle$, whose matrix elements follow from
the **Slater–Condon rules** (determinants differing by zero, one, or two
spin-orbitals contribute; three or more give exactly zero). Diagonalizing that
matrix and taking the lowest eigenvalue gives $E_\text{FCI}$; the correlation
energy is $E_\text{corr} = E_\text{FCI} - E_\text{RHF} \le 0$, always negative
because adding variational freedom can only lower the energy.

**Correlation is excited-determinant mixing.** The physical picture is direct:
the mean field misses the instantaneous avoidance of electrons, and full CI
recovers it by *mixing in* determinants where electrons occupy different
orbitals. When the reference already spans the whole space, there is nothing to
mix and correlation vanishes identically. **He/STO-3G** is exactly that
boundary — one basis function, one orbital, no virtuals, so
$N_\text{det} = \binom{1}{1}\binom{1}{1} = 1$: full CI *is* RHF and
$E_\text{corr} = 0$ to machine precision. **H2/STO-3G** is the smallest system
with real correlation: two orbitals, $N_\text{det} = \binom{2}{1}\binom{2}{1}
= 4$, with the reference $|1_\alpha 1_\beta\rangle$ coupling to the double
$|2_\alpha 2_\beta\rangle$. The pinned reference energy is
$E_\text{FCI} \approx -1.1372702\ \text{Hartree}$ (STO-3G, $R = 0.7414$ Å; the
same constant the tests and `examples/14_fci.cpp` assert).

A useful sanity check against [MP2](#the-hierarchy-mp2-ci-coupled-cluster):
FCI is the exact in-basis ground state, so it recovers *at least as much*
correlation as MP2 — $E_\text{corr}^\text{FCI} \le E_\text{corr}^\text{MP2} < 0$.
For this two-electron minimal-basis target MP2 *under-binds*: its correlation is
*less* negative than the exact value, so $E_\text{FCI} \le E_\text{MP2}$. (The
tests encode this correct inequality.)

**Why it is a reference, not a production method.** $N_\text{det}$ grows
factorially with system size, and the dense $N_\text{det} \times N_\text{det}$
eigenproblem grows as its cube. A handful of orbitals is already thousands of
determinants; a real molecule is astronomically out of reach. Atomiverse
therefore caps `fci()` at a hard TINY limit ($n_\text{mo} \le 14$,
$N_\text{det} \le 2000$) and throws past it. Its job is to be *exactly right*
on toy systems, as the golden yardstick that validates MP2 today and coupled
cluster tomorrow — the same role Szabo & Ostlund's hand-worked H2 table plays
in the textbook. For correlation at scale, use DFT (above) or MP2. This is
Knowles & Handy's determinant-driven FCI in its simplest dense form, with no
Davidson/direct-CI machinery, deliberately: correctness over scale.

## Status and Further Reading

**Status:** MP2 and tiny full CI are implemented and validated; coupled cluster,
multi-reference, and large-scale CI remain out of scope. Runnable examples:
[`examples/13_mp2.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/13_mp2.cpp) and
[`examples/14_fci.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/examples/14_fci.cpp). Validation benchmarks pin
every quoted number against reference values: `val_mp2_vs_pyscf` (MP2 across
{He,H₂,H₂O,NH₃,CH₄}×{STO-3G,6-31G}) and `val_fci_vs_pyscf` (exact FCI +
engine-MO self-consistency + the MP2-vs-FCI two-level cross-check).

For the production route to correlation, return to
[Chapter 6: Hybrid Functionals and Analytic Gradients](ch06-hybrid-dft.md). For
the periodic counterpart of the electronic-structure problem, continue to
[Chapter 8: Plane-Wave DFT for Periodic Systems](ch08-plane-wave.md).
