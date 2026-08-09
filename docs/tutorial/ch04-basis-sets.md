# Chapter 4. Basis Sets

Basis quality shows up immediately in ionization potentials. Koopmans'
theorem says \(\mathrm{IP}\approx -\varepsilon_\mathrm{HOMO}\). Atomiverse
RHF vs experiment (eV) for 6-31G — table below:

| System | Koopmans 6-31G | Experiment |
|--------|---------------:|-----------:|
| He | 24.875 | 24.587 |
| H₂ | 16.201 | 15.426 |
| H₂O | 13.643 | 12.620 |
| NH₃ | 11.294 | 10.800 |
| CH₄ | 14.804 | 13.600 |

ΔSCF (orbital relaxation, with charge-shifted SCF) for the same basis:

| System | Koopmans | ΔSCF | Experiment |
|--------|---------:|-----:|-----------:|
| He | 24.875 | 23.444 | 24.587 |
| H₂O | 13.643 | 10.978 | 12.620 |
| CH₄ | 14.796 | 13.419 | 13.600 |

STO-3G vs 6-31G for water: 10.65 eV → 13.64 eV toward experiment 12.62 eV.
Relaxation always lowers the IP relative to Koopmans; CH₄ and He move closer
to experiment, while H₂O still needs correlation beyond single-reference HF.

[Chapter 2](ch02-rhf.md) computed an RHF energy for H₂ in the minimal STO-3G
basis and quietly accepted whatever number came out. This chapter goes back
to that one constructor argument —

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
```

— and asks the question every practitioner eventually asks: **how much does
the answer depend on the basis set, and when have I used enough of one?**
The basis is the single largest *controllable* source of error in a
mean-field calculation, and understanding basis-set convergence is what
separates a number you can defend from a number you happened to get.

The six questions:

1. **The concept.** What a Gaussian basis is and why bigger is (usually)
   better.
2. **The Atomiverse representation.** `BasisSet`, the method's private
   integral workspace, and the shell machinery behind them.
3. **The C++ API.** Selecting a basis, reading the AO count, and running the
   same SCF across a series.
4. **The expected result.** A monotone-ish energy lowering from STO-3G to
   def2-SVP, with diminishing returns.
5. **The validation.** Each basis is validated against PySCF at the *same*
   basis, so "is my number right?" and "is my basis converged?" stay
   separate questions.
6. **The recurring pattern.** Convergence-with-respect-to-a-discretization
   is the same idea as the plane-wave cutoff and the DFT grid.

## The Concept: Finite Windows on an Infinite Function

A molecular orbital is a continuous function of space. We cannot store a
continuous function, so we expand it in a finite set of fixed **atomic-orbital
basis functions** $\{\chi_\mu\}$:

$$
\psi_i(\mathbf{r}) = \sum_\mu C_{\mu i}\, \chi_\mu(\mathbf{r}).
$$

The basis is the window through which the SCF sees the true orbital. A
minimal basis (one function per occupied atomic shell) is a narrow window: it
can represent the gross shape of a bond but not its polarization or its
radial flexibility. A larger basis — more functions per shell (split
valence), functions of higher angular momentum (polarization) — widens the
window, and the variational SCF energy can only go *down* (or stay equal) as
the window widens, because the minimizer has strictly more freedom.

Two practical facts follow:

- **Gaussians, not Slater functions.** Real atomic orbitals decay like
  $e^{-\zeta r}$ (Slater), but the four-index electron-repulsion integrals
  over Slater functions have no closed form. The universal compromise, since
  Boys (1950), is to build each basis function from a fixed contraction of
  Gaussians $e^{-\alpha r^2}$, for which every integral is analytic. "STO-3G"
  literally means *Slater-Type Orbital approximated by 3 Gaussians*.
- **The energy converges but never terminates.** There is no "exact" finite
  basis. You climb a ladder — STO-3G → 6-31G → 6-31G* → def2-SVP → … — until
  the property you care about stops moving by more than your target
  tolerance. That climb is *basis-set convergence*.

## The Representation: A `BasisSet` Enum Over a Shell Machine

Atomiverse names each basis with an enum and resolves it to concrete shells
inside the method's private integral workspace. From
[`basis_set.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/scf/basis_set.h):

```cpp
enum class BasisSet {
    STO3G,       // minimal (H–O). The production baseline.
    Pople631G,   // 6-31G  split-valence, no polarization
    Pople631Gs,  // 6-31G* d-polarization on heavy atoms
    Pople631pGs, // 6-31+G* heavy-atom diffuse sp + d-polarization
    Def2SVP,     // def2-SVP  Ahlrichs split-valence polarized
    Def2TZVP,    // def2-TZVP triple-zeta
    AugCcPVDZ,   // aug-cc-pVDZ Dunning DZ + diffuse augmentation
};
```

Under the hood a `BasisSet` + the frame's atoms is resolved into a
`ShellList` — flat AO descriptors with per-shell contraction lengths, which
is why the machine handles mixed contractions (6-31G's 6/3/1, def2-SVP's up
to 7) without special-casing. You do not touch `ShellList` directly for a
normal calculation; the method rebuilds it from the Frame on every
`run(frame)`. The AO count surfaces on the wavefunction — it is the row
dimension of the MO coefficient matrix:

```cpp
using Spin = cpu::scf::SlaterDeterminant::Spin;

rhf.run(frame);
const auto& det = rhf.determinant();
int n_ao    = static_cast<int>(det.coefficients(Spin::alpha).shape()[0]);
int n_alpha = det.n_electrons(Spin::alpha);   // (Σ Z − charge) split by spin;
                                              // = n_occ for a closed shell
```

`n_ao` is the number you watch: it is the dimension of every matrix in
the SCF, and cost scales steeply with it (roughly $N^3$ for the
diagonalization, worse for conventional integrals), which is the price of a
wider window.

### Element and basis coverage — the no-fallback rule

STO-3G is implemented for **H through O (Z = 1–8)**; the larger Pople and
def2/cc families extend the range through the **third period (Na–Ar, Z =
11–18)**, with the Pople sets additionally carrying K/Ca. Coverage is
honest: if you ask for a
basis/element combination that is not tabulated, the shell-list build inside
`run(frame)` **throws** `std::runtime_error` rather than
silently substituting a nearby basis. There are *no fallback values* by
design — a wrong answer that runs is worse than an exception that stops you.
Among the shipped sets, STO-3G through def2-SVP are production-ready;
def2-TZVP is partial (its f-shells outrun the current XC-grid basis
evaluation), so treat it as experimental for KS-DFT.

## The C++ API: One SCF, Swept Across a Series

The mechanical recipe from [Chapter 2](ch02-rhf.md) is unchanged; only the
`BasisSet` argument moves. A convergence study is therefore a loop over the
enum:

```cpp
using namespace ATV;
using Spin = cpu::scf::SlaterDeterminant::Spin;

cpu::Frame frame = /* fixed H2O geometry, positions in Å, type = Z */;

for (auto basis : {cpu::scf::BasisSet::STO3G,
                   cpu::scf::BasisSet::Pople631G,
                   cpu::scf::BasisSet::Pople631Gs,
                   cpu::scf::BasisSet::Def2SVP}) {
    cpu::scf::RHF rhf{basis, /*charge=*/0, /*tol=*/1e-9, /*max_iter=*/100};
    rhf.run(frame);

    const int n_ao = static_cast<int>(
        rhf.determinant().coefficients(Spin::alpha).shape()[0]);
    std::printf("%-12s  n_ao=%3d  E = %.8f Ha  (%d iters)\n",
                cpu::scf::basis_set_name(basis), n_ao,
                rhf.energy(), rhf.iterations());
}
```

`run(frame)` writes the eV total energy to `frame.meta.get(kEnergy)`;
`energy()` reports the same total (electronic + nuclear) energy in
**Hartree**, alongside `converged()` and `iterations()` on the method, with
the spin-resolved orbital eigenvalues on `determinant()`. `basis_set_name`
gives the human-readable label for logs.

Pass the basis to the constructor and run — there is no separate integral or
config step:

```cpp
// Restricted HF, MINAO guess
cpu::scf::RHF rhf{cpu::scf::BasisSet::Def2SVP};
rhf.run(frame);                               // rhf.energy() in Ha
```

The electron counts (`n_alpha`, `n_beta`) are resolved from the Frame
(Σ Z − charge) on every run, and the private integral workspace is rebuilt
whenever the geometry or the `basis` member changes — nothing is tied to a
stale pack. That rebuild-from-Frame contract is the same one the dynamics
and optimization chapters rely on when the method is bound at a `force`
phase.

## The Expected Result: Monotone Lowering, Diminishing Returns

For a closed-shell molecule at a fixed geometry, the RHF total energy should
*drop* as the basis grows, and the drops should shrink — the signature of
convergence. A representative water sweep looks qualitatively like:

```text
STO-3G       n_ao= 7   E ≈ -74.96 Ha     (crude; minimal window)
6-31G        n_ao=13   E ≈ -75.98 Ha     (split valence: big gain)
6-31G*       n_ao=19   E ≈ -76.01 Ha     (d-polarization on O)
def2-SVP     n_ao=24   E ≈ -76.02 Ha     (polarized double-ζ)
```

(Exact values depend on the geometry; the *pattern* — a large STO-3G→6-31G
step, then small polarization and def2-SVP corrections — is the lesson.) The
STO-3G energy is roughly a Hartree above the near-complete-basis limit, which
is why STO-3G is a teaching and smoke-test basis, not a production one. By
def2-SVP the *Hartree-Fock* energy is close to converged; the *remaining*
error is no longer the basis but the method (HF neglects correlation), a
distinction [Chapter 6](ch06-hybrid-dft.md) and the correlation-methods
status chapter take up.

The right way to read this table is by **numerical class**: the energy
*difference between two bases at the same geometry* is a
physical-approximation quantity (it measures basis incompleteness), whereas
"is the def2-SVP number itself correct?" is a separate,
numerical-tolerance question answered against a reference at that same basis
— which is exactly what the validation does.

## The Validation: Each Basis Checked Against PySCF at That Basis

Basis-set convergence and code correctness are two different things, and
Atomiverse keeps them separate. Correctness is pinned by comparing the
Atomiverse energy at a given basis against a genuine PySCF run *at the same
basis and geometry*, to tight numerical tolerance (RHF energy vs PySCF
**1e-7 Ha**). The reference numbers are generated by PySCF and checked into
the test as headers, so the parity test runs in CI without PySCF installed.

That separation is what lets the convergence table above mean something: each
row is independently certified to be the *right* energy for its basis, so the
*differences* between rows genuinely measure basis incompleteness and not a
bug that happens to cancel.

The production recommendation follows directly: set the method's `basis` to
`BasisSet::Def2SVP` for real closed-shell work (`BasisSet::Pople631G` for
quick tests), knowing STO-3G exists for pedagogy and fast regressions, and
def2-TZVP for pushing toward the basis-set limit where its f-shell support
allows.

## The Recurring Pattern: Convergence With Respect to a Discretization

Choosing a Gaussian basis is one instance of a pattern that appears in every
electronic-structure method in this book: **the physics is exact, the
discretization is not, and you converge the discretization until the answer
stops moving.**

- **Gaussian basis** (this chapter): converge the AO set STO-3G → def2-SVP.
- **Plane-wave DFT** ([Chapter 8](ch08-plane-wave.md)): converge the kinetic-energy
  cutoff `ecut` and the k-point mesh — a plane-wave basis is just a
  *different* finite window, converged with a single knob.
- **DFT XC grid** ([Chapter 5](ch05-kohn-sham-dft.md)): converge the radial/angular
  grid until the integrated electron count and `E_xc` stabilize.

In each case the discipline is identical: pick the knob, run a short series,
watch the property of interest, and report the setting at which it is
converged to your tolerance. A basis set is not a magic keyword — it is a
convergence parameter you are responsible for.

## Validation Reference

Basis-set correctness is pinned by the RHF/DFT-vs-PySCF parity suite,
`test_cpu_scf_basis_vs_pyscf` (Atomiverse energy vs a checked-in genuine
PySCF reference at the same basis, **1e-7 Ha**), with the reference
generators and provenance under
[`/* benchmarks removed; re-add later */ rhf`](https://github.com/MolCrafts/Atomiverse/blob/dev//* benchmarks removed; re-add later */ rhf). The basis-convergence
benchmark case (energy series with committed near-CBS references) is part of
the manual build-out plan; until it lands, the parity tests plus the sweep
recipe above are the way to certify a convergence study.

## Cartesian and spherical shells

A d shell can be written in two angular bases. **Cartesian** functions use all
monomials with \(l_x+l_y+l_z=2\) (six components). **Spherical** (real solid
harmonic) functions use the five \(Y_{2m}\). The six Cartesian functions
contain an \(s\)-type contaminant proportional to \(r^2\); likewise, ten
Cartesian f functions contain three \(p\)-type contaminants. Removing those
contaminants is a fixed linear transformation (Helgaker, Ch. 6; Schlegel &
Frisch, 1995).

The angular convention is part of the **basis-set definition**, not a free
user switch: Pople sets such as 6-31G* use Cartesian 6d; def2 and
correlation-consistent sets use spherical harmonics. The engine transforms
assembled matrices after the Cartesian integral layer.

Unit-normalised Cartesian d components do not share one radial factor:
\(d_{xx}\) and \(d_{xy}\) differ by \(\sqrt3\). The transformation coefficients
must absorb that ratio so that a unit-normalised d shell maps to a \(5\times5\)
unit spherical overlap.

```cpp
// def2-SVP is spherical by definition
cpu::scf::RHF rhf{cpu::scf::BasisSet::Def2SVP};
rhf.run(frame); // H2O: 24 spherical AOs (25 Cartesian)
```

Analytic gradients for spherical bases are not yet available; `forces()`
refuses rather than return an incomplete derivative.

Water RHF/def2-SVP energies agree with PySCF to \(\sim10^{-13}\,\mathrm{Ha}\)
in the spherical convention. The Cartesian 6d energy for the same primitive
set differs by about \(1.2\,\mathrm{mHa}\): that is a **different basis**, not
a numerical error.

## Polarisation, diffuse functions, and the third period

Pople names encode construction. `6-31G` is split-valence; `*` adds heavy-atom
d polarisation; `**` also polarises hydrogen; `+` / `++` add diffuse shells.
Dunning’s `cc-pVXZ` family builds polarisation into the correlation sequence;
`aug-` adds one diffuse shell per angular momentum present.

Polarisation and diffuse functions answer different questions. Polarisation
allows the density to **deform** (higher \(\ell\)). Diffuse functions allow it
to **extend** (smaller exponents). Neutral closed-shell ground states are
usually limited by polarisation; anions, Rydberg states, polarisabilities, and
hydrogen-bond tails often require diffuse functions.

Anions make the point concrete. With a compact basis, the extra electron has
nowhere to go: \(\varepsilon_{\mathrm{HOMO}}\) can even become **positive**,
which is a basis artefact rather than a physical unbound state. For OH⁻ at
\(r=0.964\,\mathrm{Å}\):

| Basis | \(\varepsilon_{\mathrm{HOMO}}\) / Ha |
|---|---:|
| STO-3G | \(+0.252\) (artefact) |
| 6-31G* | \(-0.038\) |
| 6-31+G* | \(-0.104\) |

From 6-31G* to 6-31+G* the total energy drops by about \(50\,\mathrm{mHa}\),
while the HOMO deepens by about \(66\,\mathrm{mHa}\). Energy convergence and
property convergence are not the same ladder.

Tables for 6-31G, 6-31G*, 6-31+G*, def2-SVP, def2-TZVP, and aug-cc-pVDZ cover
Na–Ar (and K/Ca in the Pople sets). Unsupported elements throw; there is no
silent fallback. Static tables are generated offline and compiled in.

| Quantity | Engine | PySCF 2.14.0 |
|---|---:|---:|
| \(\varepsilon_{\mathrm{HOMO}}\)(OH⁻, 6-31+G*) / Ha | −0.10357976 | −0.10357977 |
| \(E\)(OH⁻, 6-31+G*) / Ha | −75.37629336 | −75.37629336 |
| \(E\)(Cl⁻, 6-31+G*) / Ha | −459.53966104 | −459.53966104 |
| \(E\)(H₂O, aug-cc-pVDZ, sph.) / Ha | −76.04139352 | −76.04139352 |

## Effective core potentials

Deep core orbitals demand very steep primitives and contribute little to
bonding. An **effective core potential** (ECP) replaces the core by an
operator that acts on the valence electrons: it screens the nucleus and
enforces approximate Pauli orthogonality to the missing core. For heavy
elements the core is also relativistic; fitting a quasi-relativistic atomic
solution into an ECP is usually cheaper than an all-electron relativistic
treatment of the valence problem.

Semi-local ECPs (Kahn, Baybutt, Truhlar) use a local radial potential plus
angular projectors for lower \(\ell\). In the calculation, nuclear charges and
electron counts become **valence** quantities: for iodine with def2-ECP,
\(Z_{\mathrm{val}}=25\) and I₂ has 50 electrons, not 106.

def2-ECP parameters already include scalar relativistic effects. Combining
ECP and explicit x2c on the same atom double-counts those effects; the
engine rejects that combination element-wise.

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::Def2SVP};
rhf.ecp = "def2-ECP";
rhf.run(i2_frame); // I2, r = 2.666 Å → E ≈ −593.316 Ha
```

All-electron I₂ RHF energies are of order \(-10^4\,\mathrm{Ha}\); the ECP
calculation retains the chemically relevant valence space. Analytic ECP
gradients are not yet implemented.

| Quantity | Engine | PySCF |
|---|---:|---:|
| \(E\)(I₂, def2-SVP+ECP) / Ha | −593.31618188 | −593.31618188 |
| \(E\)(AuH, def2-SVP+ECP) / Ha | −135.34256701 | −135.34256701 |

## Basis-set superposition error

In a finite basis, monomer A at the dimer geometry can lower its energy by
using basis functions centred on B. That stabilisation is **basis-set
superposition error** (BSSE), not a physical interaction, and it makes raw
interaction energies too negative:

\[
\Delta E_{\mathrm{raw}} = E_{AB}^{AB} - E_A^A - E_B^B.
\]

The Boys–Bernardi counterpoise correction recomputes each monomer in the
**full dimer basis**, placing **ghost** centres (basis functions only; zero
nuclear charge and electron count) on the missing partner:

\[
\Delta E_{\mathrm{CP}} = E_{AB}^{AB} - E_A^{AB} - E_B^{AB}.
\]

The BSSE estimate used here is

\[
\mathrm{bsse}
  = (E_A^{AB}-E_A^A)+(E_B^{AB}-E_B^B)\le 0;
\]

its magnitude is the superposition error. Three limitations are standard:
small bases often **overcorrect**; site–site recipes for clusters are not
size-consistent with a full-cluster CP; and monomer **deformation** from
isolated to dimer geometry is a separate correction not included in a
fixed-geometry CP.

Ghost indices are method members (`ghost_atoms`), not Frame columns.
`Counterpoise` orchestrates five SCF runs on a configured method object:

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::Pople631Gs, /*charge=*/0, /*tol=*/1e-10};

cpu::scf::Counterpoise cp({0, 1, 2}, {3, 4, 5}); // water dimer
auto r = cp.run(dimer_frame, rhf);
// r.e_int_cp, r.bsse [Hartree]
```

Ghosts with Kohn–Sham methods are not supported. For the water dimer at
HF/6-31G* (Cartesian), \(\Delta E_{\mathrm{CP}}\approx -7.5\,\mathrm{mHa}\)
and \(\mathrm{bsse}\approx -1.5\,\mathrm{mHa}\), in agreement with an
external PySCF reference to \(10^{-6}\,\mathrm{Ha}\).
