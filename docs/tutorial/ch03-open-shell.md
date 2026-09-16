# Chapter 3. Unrestricted Hartree–Fock

Restricted Hartree-Fock is a good first model because it explains molecular
orbitals, density matrices, and self-consistency with the least amount of
machinery. It also fails in exactly the situations that matter most for open
shells and bond breaking. This chapter is about that failure and about the
smallest change that repairs it: allow the $\alpha$ and $\beta$ electrons
to occupy different spatial orbitals.

Dissociation limit: \(2\times E(\mathrm{H})=-0.9332\) Ha.

## Live Result: H₂ Dissociation

RHF cannot dissociate to two neutral H atoms; UHF can. The scan below is the
actual dissociation scan (columns: \(R\), \(E_\mathrm{RHF}\),
\(E_\mathrm{UHF}\), \(\langle S^2\rangle\)).

```molplot preset="molplot" theme="auto" width="100%" aspect="16:9"
mark: {type: line, point: true, tooltip: true, strokeWidth: 2}
data:
  values:
    - {R: 0.50, method: RHF, E: -1.042996}
    - {R: 0.75, method: RHF, E: -1.116151}
    - {R: 1.00, method: RHF, E: -1.066109}
    - {R: 1.25, method: RHF, E: -0.989114}
    - {R: 1.50, method: RHF, E: -0.910874}
    - {R: 1.75, method: RHF, E: -0.841349}
    - {R: 2.00, method: RHF, E: -0.783793}
    - {R: 2.50, method: RHF, E: -0.702944}
    - {R: 3.00, method: RHF, E: -0.656048}
    - {R: 3.50, method: RHF, E: -0.629820}
    - {R: 4.00, method: RHF, E: -0.614870}
    - {R: 0.50, method: UHF, E: -1.042996}
    - {R: 0.75, method: UHF, E: -1.116151}
    - {R: 1.00, method: UHF, E: -1.066109}
    - {R: 1.25, method: UHF, E: -0.994190}
    - {R: 1.50, method: UHF, E: -0.957707}
    - {R: 1.75, method: UHF, E: -0.943080}
    - {R: 2.00, method: UHF, E: -0.937213}
    - {R: 2.50, method: UHF, E: -0.933867}
    - {R: 3.00, method: UHF, E: -0.933285}
    - {R: 3.50, method: UHF, E: -0.933182}
    - {R: 4.00, method: UHF, E: -0.933166}
encoding:
  x: {field: R, type: quantitative, title: "R (Å)", scale: {zero: false}}
  y:
    field: E
    type: quantitative
    title: "E (Hartree)"
    scale: {zero: false, nice: true}
  color: {field: method, type: nominal, title: null}
  tooltip:
    - {field: method, type: nominal}
    - {field: R, type: quantitative, format: ".2f"}
    - {field: E, type: quantitative, format: ".6f"}
```

At \(R=4\) Å, UHF sits at **−0.933166 Ha** — within \(3\times 10^{-5}\) Ha of
the exact one-electron limit \(2\times E(\mathrm{H})=-0.9332\) Ha. RHF is still
**−0.61487 Ha**, more than 0.3 Ha too high. Spin contamination
\(\langle S^2\rangle\) rises from 0 near equilibrium toward 1 as the UHF
solution becomes two separate radicals.

## Why RHF Stops Being Enough

The central RHF assumption is that one spatial orbital is occupied by two
electrons of opposite spin. Near equilibrium, that is often a reasonable
picture. For $\mathrm{H}_2$ at $0.74$ Angstrom, it is exactly the right
place to begin.

The problem appears when the bond is stretched. Far apart, the physically
correct dissociation limit is two separate hydrogen atoms, each carrying one
electron. RHF refuses to describe that situation honestly because it insists
that both electrons remain in the same delocalized spatial orbital. The result
is not a small quantitative error. It is a qualitatively wrong picture of the
electrons.

RHF also has no way to describe a genuine odd-electron system such as the
hydrogen atom. There is simply no second electron available to occupy the same
spatial orbital with opposite spin. That is why the UHF example in this
repository begins with the hydrogen atom: it is the smallest case that forces
you to abandon the restricted ansatz.

## What UHF Changes And What It Keeps

UHF keeps the mean-field Hartree-Fock idea. The electrons still move in an
average field built from the current density, and the calculation is still an
SCF fixed-point problem. What changes is the wavefunction ansatz. Instead of one
orbital coefficient matrix $C$, we now have two:

$$
\varphi_i^\alpha(\mathbf{r}) = \sum_\mu C^\alpha_{\mu i}\chi_\mu(\mathbf{r}),
\qquad
\varphi_j^\beta(\mathbf{r}) = \sum_\mu C^\beta_{\mu j}\chi_\mu(\mathbf{r}).
$$

That one change forces the density and Fock matrices to split by spin:

$$
P^\alpha = C^\alpha_{\mathrm{occ}}\left(C^\alpha_{\mathrm{occ}}\right)^T,
\qquad
P^\beta = C^\beta_{\mathrm{occ}}\left(C^\beta_{\mathrm{occ}}\right)^T,
\qquad
P = P^\alpha + P^\beta.
$$

The Coulomb field depends on the total density because every electron repels
every other electron. Exchange depends only on same-spin density because the
exchange term comes from antisymmetry of the wavefunction. That gives the two
UHF Fock matrices:

$$
F^\alpha = H_{\mathrm{core}} + J(P) - K(P^\alpha),
\qquad
F^\beta = H_{\mathrm{core}} + J(P) - K(P^\beta).
$$

Once those matrices are built, the logic is completely familiar. Diagonalize
each spin channel, occupy the lowest orbitals in that channel, rebuild the two
densities, and iterate until they stop moving.

## Reading The Example

The key change in the code is as small as the key change in the theory:
instead of one occupation pattern, you now choose one for each spin channel.
In the example, that choice appears directly in calls such as

```cpp
cpu::scf::UHF uhf(n_alpha, n_beta);
uhf.basis = cpu::scf::BasisSet::STO3G;
uhf.tol = tol;
uhf.max_iter = max_iter;
uhf.run(frame);                       // eV energy on frame.meta[kEnergy]
double e_ha = uhf.energy();           // total energy [Hartree]
const auto& det = uhf.determinant();  // per-spin orbitals and occupations
```

The input you must choose carefully is the spin occupation. `n_alpha` and
`n_beta` are not bookkeeping details. They define the spin state. Their sum
must equal the total number of electrons, and their difference fixes
$M_S = (N_\alpha - N_\beta)/2$.

This is why the first example in the file is the hydrogen atom. It forces the
issue immediately: one electron means one occupied spin channel and one empty
spin channel. The second and third examples then show what happens when the
same unrestricted machinery is applied to $\mathrm{H}_2$ near equilibrium and
far from equilibrium.

An equilibrium closed-shell system such as $\mathrm{H}_2$ at $0.74$
Angstrom should come back with nearly identical $\alpha$ and $\beta$
channels and a spin contamination near zero. A hydrogen atom should come back
with `n_alpha = 1`, `n_beta = 0`, and a total energy near the STO-3G reference
of $-0.4666$ Hartree. A stretched $\mathrm{H}_2$ scan should show the
$\alpha$ and $\beta$ channels separating as the bond length grows.

## Charge and Multiplicity Conventions

**Concept.** An open-shell calculation is specified by two user-visible
integers: the molecular charge $q$ and the spin multiplicity $M$. Everything
the solver needs — how many $\alpha$ and how many $\beta$ electrons to occupy
— follows deterministically from those two numbers. Getting this translation
wrong is the single most common way to converge cleanly to the wrong state,
so Atomiverse makes it an explicit, validated step rather than a convention
you must remember.

**Math.** The electron count folds the charge in at integral-build time,

$$ N_e = \sum_A Z_A - q, $$

and the multiplicity $M = 2S + 1$ fixes the spin split through
$n_\alpha - n_\beta = M - 1$:

$$ n_\alpha = \frac{N_e + M - 1}{2}, \qquad
   n_\beta = \frac{N_e - M + 1}{2}. $$

These are integers only when $N_e$ and $M - 1$ have the same parity (even
$N_e$ pairs with odd $M$, odd $N_e$ with even $M$), and $n_\beta \ge 0$
requires $M \le N_e + 1$. Any violation means the requested state does not
exist.

**Algorithm.** The public surface of an open-shell method is the α/β pair
itself: `UHF` and `UKS` take `(n_alpha, n_beta)` as constructor arguments.
Multiplicity is a bookkeeping convention on your side of the boundary — apply
the two formulas above and pass the resulting counts:

```cpp
// H2O+ : N_e = Σ Z − q = 9, doublet (M = 2)
const int n_electrons = 9;
const int multiplicity = 2;                                 // M = 2S + 1
const int n_alpha = (n_electrons + multiplicity - 1) / 2;   // 5
const int n_beta  = (n_electrons - multiplicity + 1) / 2;   // 4

cpu::scf::UHF uhf(n_alpha, n_beta);   // (5, 4)
```

Guard the two validity conditions yourself before constructing: $N_e$ and
$M - 1$ must share parity (otherwise the integer divisions silently truncate),
and $M \le N_e + 1$ (otherwise $n_\beta$ goes negative). The engine performs
the same check internally — `cpu::scf::internal`'s occupation resolver throws
`std::invalid_argument` on either violation — but that type lives under
`cpu/scf/internal/` and is **not** part of the public API; do not include or
name it.

Worked table (all rows are validated against PySCF 2.13.1 and
regression-locked — see below):

| System | charge | $M$ | $N_e$ | $n_\alpha$ | $n_\beta$ |
|---|---|---|---|---|---|
| $\mathrm{H_2O}$ | 0 | 1 | 10 | 5 | 5 |
| $\mathrm{OH^-}$ | −1 | 1 | 10 | 5 | 5 |
| $\mathrm{H_2O^+}$ | +1 | 2 | 9 | 5 | 4 |
| O atom | 0 | 3 | 8 | 5 | 3 |

Rejected combinations (the internal resolver throws `std::invalid_argument`):
$\mathrm{H_2O}(0, 2)$ — even $N_e = 10$ with even $M$ breaks parity;
$\mathrm{H_2O^+}(+1, 1)$ — odd $N_e = 9$ with odd $M$ breaks parity;
$\mathrm{H}(0, 4)$ — $M = 4 > N_e + 1 = 2$ would need $n_\beta = -1$.

One convention contrast worth memorizing: PySCF's `mol.spin` is **not** the
multiplicity — it is $2S = n_\alpha - n_\beta = M - 1$. A PySCF doublet is
`spin=1`; here you write the same state as $M = 2$, i.e.
`cpu::scf::UHF{5, 4}`.

**Limitations.** Resolving the occupation fixes bookkeeping only. It cannot tell you
*which* multiplicity is the ground state (the O atom is a triplet because of
Hund's rule, not because $M=3$ is the only valid input), and it does not
protect against spin contamination — that diagnostic is the next section.

<!-- Every (charge, M, n_alpha, n_beta) tuple and every rejected combination
     in this section is regression-locked by
     tests/cpu/scf/test_spin_resolution.cpp (test_cpu_scf_spin_resolution):
     WorkedTableMatchesTutorialCh03, OhMinusChargeFoldsIntoElectronCount,
     RejectsInvalidCombinations, FeedsUhfConfigRawCounts,
     FeedsUksConfigRawCounts. Change one side only in lockstep with the
     other. -->

## Spin Contamination: What It Means Here

The price of unrestricted flexibility is that the UHF determinant is usually
not an exact eigenfunction of $\hat S^2$. That is what the spin
contamination number measures. In the usual notation, the contaminated amount is

$$
\Delta\langle \hat S^2 \rangle
= \langle \hat S^2 \rangle_{\mathrm{UHF}} - S(S+1).
$$

Near equilibrium in a closed-shell singlet, this number should be close to
zero. When $\mathrm{H}_2$ is stretched and the unrestricted solution breaks
spin symmetry, the contamination rises. That is not automatically a bug. In the
bond-breaking regime it is part of the mechanism by which UHF escapes the
artificially restricted RHF picture.

The repository example makes this tangible. Near equilibrium, RHF and UHF stay
close because the electrons still behave like a closed-shell pair. As the bond
is stretched, the unrestricted solution lowers the energy by letting the two
spins localize differently. The growth of spin contamination is the visible
trace of that symmetry breaking.

## ΔSCF: Ionization Energies Beyond Koopmans

The previous chapter's Koopmans section (see the Koopmans discussion in
[Chapter 2](ch02-rhf.md)) estimated the ionization
energy of a closed-shell molecule as $I_K = -\varepsilon_\text{HOMO}$: the
negative of the highest occupied orbital energy. That estimate is free — you
already have the orbital energies once the neutral SCF converges — but it rests
on a physical fiction. It assumes that when you remove an electron, the
remaining electrons do not move. They do. ΔSCF is the honest alternative: solve
the cation self-consistently and take an energy difference,

$$ I_{\Delta\text{SCF}} = E^{+}(N-1) - E(N). $$

Both energies are evaluated at the **same** geometry, basis, and convergence
threshold. That makes this a *vertical* ionization energy — the nuclei are
frozen at the neutral equilibrium while an electron is removed.

### Why ΔSCF Is Always Lower Than Koopmans

The relationship between the two estimates is not empirical; it is a two-line
variational argument. Koopmans' $-\varepsilon_\text{HOMO}$ is exactly the energy
of the $(N-1)$-electron determinant you get by deleting one electron from the
neutral orbitals and **freezing everything else**. The ΔSCF cation energy
$E^{+}$ is the result of variationally optimizing that same $(N-1)$-electron
wavefunction over the same Hilbert space. A variational optimum can only lower
the energy, so $E^{+} \le E_\text{frozen}$, and therefore

$$ I_{\Delta\text{SCF}} \le I_K = -\varepsilon_\text{HOMO}. $$

The gap between them, $I_K - I_{\Delta\text{SCF}} \ge 0$, is the **orbital
relaxation energy**: how much the remaining electrons lower their energy by
rearranging around the newly created hole. For a real molecule it is a fraction
of an eV to a couple of eV, and the ΔSCF value is the better one.

There is one clean teaching case where the two estimates are *exactly* equal.
For He and H₂ in a minimal (STO-3G) basis the cation has a single electron in a
spatial orbital that is fixed by symmetry — there is no second electron to
relax and no other orbital of the right symmetry to mix in. The relaxation
energy is identically zero, and the example prints $I_K = I_{\Delta\text{SCF}}$
to solver precision. Watch that row: it is the boundary condition that makes the
inequality meaningful everywhere else.

### Spin Bookkeeping for the Cation

Removing one electron from a closed-shell neutral turns an $N$-electron singlet
into an $(N-1)$-electron doublet ($M = 2$). Using the charge/multiplicity rules
from the section above, the electron count folds the $+1$ charge in at
integral-build time ($N_e = N - 1$), and the doublet split is

$$ n_\alpha = \frac{N}{2}, \qquad n_\beta = \frac{N}{2} - 1. $$

Concretely: the neutral has $N/2$ doubly-occupied levels; the cation keeps all
$N/2$ $\alpha$ orbitals and drops one $\beta$ electron. For H₂O ($N = 10$) that
is $(n_\alpha, n_\beta) = (5, 4)$; for He and H₂ ($N = 2$) it is $(1, 0)$.

### Warm-Starting the Cation

An open-shell SCF can converge cleanly to the *wrong* state — a hole in a lower
orbital rather than the HOMO, or an excited configuration. The defense is to
warm-start the cation from the neutral density instead of from a bare Hcore
guess. Atomiverse exposes this through `SCFInitialGuess`: seed both spin
channels with half of the converged neutral density,

```cpp
using Spin = cpu::scf::SlaterDeterminant::Spin;

cpu::scf::RHF neutral;
neutral.basis = basis;
neutral.charge = 0;
neutral.tol = tol;
neutral.max_iter = max_iter;
neutral.run(frame);

cpu::scf::SCFInitialGuess guess;                              // scf_guess.h
guess.P_alpha = neutral.determinant().density(Spin::alpha);   // half the total
guess.P_beta  = guess.P_alpha;                                // restricted density
guess.has_alpha = guess.has_beta = true;

cpu::scf::UHF cation(N/2, N/2 - 1);
cation.basis = basis;
cation.charge = 1;
cation.tol = tol;
cation.run(frame, &guess);
```

A bare Hcore cold start converges to the same energy on these small systems —
the example and its tests check that the two agree — but density warm-starting
is the habit to build: on larger or near-degenerate cations it is the
difference between the ground-state ion and a plausible-looking excited
solution.

### Monitoring Spin Contamination

Because the cation is unrestricted, check its $\langle \hat S^2 \rangle$ the
same way the previous section did. A clean doublet has

$$ \langle \hat S^2 \rangle = S(S+1) = \tfrac{1}{2}\cdot\tfrac{3}{2} = 0.75, $$

so the cation's `determinant().spin_contamination()` (which reports
$\langle \hat S^2 \rangle - S(S+1)$) should sit at ≈ 0. The example prints $\langle \hat S^2 \rangle =
\text{spin\_contamination} + 0.75$ for every cation; values noticeably above
0.75 warn that the unrestricted determinant has picked up higher-spin character
and the ΔSCF number should be read with caution.

### Vertical Versus Experiment

These are *vertical* ionization energies at a fixed neutral geometry. A measured
*adiabatic* ionization energy also lets the cation relax its geometry, which
lowers $E^{+}$ further and makes the adiabatic value smaller than the vertical
one. So even a well-converged ΔSCF number is only a semi-quantitative match to
an experimental ionization energy in a small basis — useful for trends and for
the pedagogy of relaxation, not for spectroscopic accuracy. Enlarging the basis
(STO-3G → 6-31G) and, eventually, adding correlation (a later chapter) moves the
numbers toward experiment; relaxing the cation geometry would convert vertical
to adiabatic. The worked table for all five systems in both bases is
a second UHF run on the cation with adjusted `charge` / `n_alpha` / `n_beta`.

## Running The Example

The build path is straightforward:

The example is a good companion to this chapter because it places RHF and UHF
side by side instead of treating UHF as a separate world. You can watch the
method collapse back to RHF when the restricted approximation is safe, then
watch it diverge only when that extra freedom becomes physically necessary.

## Where Readers Usually Go Wrong

The most common mistake is giving the wrong spin occupation. If the total
electron count is wrong, everything downstream is wrong. If the spin split is
wrong, the solver may converge cleanly to the wrong state. The second common
mistake is to read nonzero spin contamination as a sign that the calculation
failed. In the bond-breaking example it is the opposite: contamination is the
visible trace of unrestricted symmetry breaking doing the job RHF could not do.

The more subtle mistake is to compare only total energies and ignore the rest of
the result. UHF is a spin-resolved method. If the $\alpha$ and $\beta$
densities, orbital energies, and contamination all look like the restricted
solution, then the unrestricted machinery has not actually changed the physics,
even if the solver was configured for UHF.

## SCF stability analysis

Earlier sections say that RHF “fails” for stretched H₂. More precisely: SCF
convergence only guarantees a **stationary point** of the energy with respect
to orbital rotations (vanishing gradient). That point may be a minimum or a
saddle. Distinguishing them uses the second derivative—the orbital Hessian.
In the real closed-shell spatial-orbital form its occupied–virtual blocks are

\[
{}^1A_{ai,bj} = \delta_{ab}\delta_{ij}(\varepsilon_a-\varepsilon_i)
              + 2(ai|jb) - (ij|ab), \qquad
{}^1B_{ai,bj} = 2(ai|bj) - (aj|bi)
\]
\[
{}^3A_{ai,bj} = \delta_{ab}\delta_{ij}(\varepsilon_a-\varepsilon_i)
              - (ij|ab), \qquad
{}^3B_{ai,bj} = -(aj|bi).
\]

The lowest eigenvalues of \({}^1(A+B)\) and \({}^3(A+B)\) (Hartree) are the
curvatures along the softest orbital-rotation directions in each channel. A
non-negative value means a local minimum in that channel; a negative value
means a descent direction exists—the SCF solution is not the lowest single
determinant admitted by the theory.

Seeger and Pople (*J. Chem. Phys.* **66**, 3045 (1977)) classify those
directions as **internal** versus **external**:

- **Internal** (singlet channel): the rotated state remains a real closed-shell
  RHF determinant. A negative lowest eigenvalue of \({}^1(A+B)\) means another
  RHF solution is lower.
- **External** (triplet channel): the rotation breaks α/β spatial equality and
  leaves the RHF manifold toward UHF. A negative lowest eigenvalue of
  \({}^3(A+B)\) means a lower UHF solution exists—RHF is a minimum under its
  constraint and a saddle without it.

The two channels are different matrix blocks: internal stability does not
imply external stability. Stretched H₂ sits in exactly that trap.

### Example: stability of stretched H₂

`RHF::stability()` after `run()` returns the lowest eigenvalues, stability
flags, and—when external instability is found—orbitals rotated along the
unstable mode for use as a UHF guess:

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.run(frame); // H2, R = 1.5 Å

auto st = rhf.stability();
// st.internal_eigenvalue ≈ +0.71 Ha  → internally stable
// st.external_eigenvalue ≈ −0.21 Ha  → externally unstable
if (!st.external_stable) {
    // Build a broken-symmetry guess from st.rotated_C; see
    // regressions/scf/h2_rhf_stability for the full construction.
    cpu::scf::UHF uhf{1, 1};
    uhf.basis = cpu::scf::BasisSet::STO3G;
    uhf.run(frame, &guess);
    // uhf.energy() ≈ −0.958 Ha < rhf.energy() ≈ −0.911 Ha
}
```

At \(R=1.5\,\mathrm{Å}\) the RHF–UHF gap on the dissociation curve
(\(\sim 0.047\,\mathrm{Ha}\)) is the descent indicated by the external
eigenvalue. Scanning \(\lambda_{\min}\) with \(R\), the external eigenvalue
crosses zero near \(R\approx 1.15\,\mathrm{Å}\) (Coulson–Fischer point);
beyond that RHF is a saddle.

Internally and externally stable/unstable status is only two channels of one
orbital Hessian (matrix-free \(\sigma=(A\pm B)v\)). The same kernel underlies
second-order SCF, CPHF response, and TDHF excitation energies.

| \(R\) (Å) | Channel | Engine | PySCF 2.14.0 |
|---|---|---:|---:|
| 0.735 | internal \({}^1(A+B)\) | +1.135177 | +1.135177 |
| 0.735 | external \({}^3(A+B)\) | +0.411452 | +0.411452 |
| 1.5 | internal \({}^1(A+B)\) | +0.708897 | +0.708897 |
| 1.5 | external \({}^3(A+B)\) | −0.209247 | −0.209247 |
| 1.5 | \(E_{\mathrm{UHF}}-E_{\mathrm{RHF}}\) | −0.046833 | −0.046833 |

PySCF’s `mf.stability(external=True)` at \(R=1.5\,\mathrm{Å}\) reports the
same internal-stable / external-unstable diagnosis. Geometry and commands are
in `regressions/scf/h2_rhf_stability/REFERENCE.md`.

## Restricted Open-Shell HF

UHF gives α and β independent spatial orbitals and therefore spin
contamination: \(\langle S^2\rangle\) is no longer \(S(S+1)\). For stretched
H₂ that contamination is the price of static correlation. For a well-behaved
open shell near equilibrium (triplet O₂, methyl radical) one often wants open
occupation **and** a pure spin eigenstate. ROHF uses **one** set of spatial
orbitals partitioned into closed (doubly occupied), open (singly occupied α),
and virtual spaces, with β occupation a subset of α. That single MO set
implies \(\langle S^2\rangle=S(S+1)\) identically (Roothaan, *Rev. Mod.
Phys.* **32**, 179 (1960)).

The energy functional is the same Pople–Nesbet form as UHF, but a single MO
set means \(F_\alpha\) and \(F_\beta\) cannot be diagonalised separately: they
are coupled into one effective Fock matrix. Atomiverse follows PySCF’s
canonical ROHF assembly (Guest–Saunders coefficients).

### Coupling coefficients are a convention

The diagonal-block coupling coefficients (Roothaan, Guest–Saunders,
McWeeny–Diercksen, …) all converge to the **same total energy**—the energy
functional does not depend on them—but they produce **different orbital
energies**. \(E_{\mathrm{total}}\) is portable; \(\varepsilon\) is
convention-bound. This engine matches PySCF’s Guest–Saunders canonical
convention; comparing \(\varepsilon\) across coefficient families is not a
bug report.

### Examples: O₂ and methyl

```cpp
// O2 triplet, R = 1.2075 Å, (n_alpha, n_beta) = (9, 7), STO-3G
cpu::scf::ROHF rohf(9, 7); // requires n_alpha >= n_beta
rohf.basis = cpu::scf::BasisSet::STO3G;
rohf.tol = 1e-12;
rohf.run(frame);
// energy ≈ −147.632167 Ha; spin contamination = 0

// planar D3h methyl, R(C–H) = 1.079 Å, (5, 4)
cpu::scf::ROHF ch3(5, 4);
ch3.basis = cpu::scf::BasisSet::STO3G;
ch3.run(frame_ch3); // ≈ −39.072133 Ha, <S²> = 0.75
```

Unsupported paths raise rather than take a wrong branch: `forces()`,
`mp2()`/`fci()`, `soscf`, and `level_shift` are not available for ROHF.

### When to choose ROHF or UHF

- **Spin-pure reference** (downstream correlation methods, clean
  \(\langle S^2\rangle\)): ROHF.
- **Variationally best single determinant**, bond breaking, strong static
  correlation: UHF. On O₂ releasing the spin constraint lowers the energy by
  about \(1.8\,\mathrm{mHa}\) with \(\langle S^2\rangle-2\approx 3.4\times10^{-3}\).
- Near equilibrium with small contamination the energies are close; the choice
  is then driven by what the next method expects of the reference.

| System | Quantity | Engine | PySCF 2.14.0 |
|---|---|---:|---:|
| O₂ (9,7) | \(E\) ROHF / Ha | −147.632166990682 | −147.632166990682 |
| O₂ (9,7) | \(\langle S^2\rangle\) ROHF | 2 | 2 |
| O₂ (9,7) | \(E\) UHF / Ha | −147.633946785450 | −147.633946785450 |
| O₂ (9,7) | \(\langle S^2\rangle\) UHF | 2.003410857681 | 2.003410857681 |
| CH₃ (5,4) | \(E\) ROHF / Ha | −39.072132718153 | −39.072132718153 |
| CH₃ (5,4) | \(\langle S^2\rangle\) ROHF | 0.75 | 0.75 |

## Generalized HF and spin symmetry breaking

The methods of this chapter form a ladder of removed constraints:

| Method | Spatial orbitals | Spin structure | Constraint removed |
|---|---|---|---|
| RHF | one set, shared α/β | collinear, paired | — |
| ROHF | one set, split occupation | collinear, spin-pure | all doubly occupied |
| UHF | separate α/β | collinear, may contaminate | α/β same orbitals |
| GHF | two-component spinors | **non-collinear** | spin quantisation along one axis |

RHF, ROHF, and UHF still quantise spin along a common axis. Generalized HF
(GHF) drops that axis: each MO is a two-component spinor
\(\phi_i=(\phi_i^\alpha;\phi_i^\beta)\). The density is a \(2n\times 2n\)
block matrix with spin-off-diagonal blocks \(P^{\alpha\beta}\)—degrees of
freedom UHF cannot represent. Coulomb depends only on the charge density;
exchange couples all four spin blocks.

Fukutome (*Int. J. Quantum Chem.* **20**, 955 (1981)) classifies HF mean
fields by broken symmetries. The real-algebra GHF implemented here corresponds
to a torsional spin-density-wave sector: magnetisation lies in one spin plane
(\(\sigma_x,\sigma_z\) real; \(\sigma_y\) imaginary). Complex GHF and
spin–orbit coupling are out of scope.

### When the extra freedom matters

Without spin–orbit coupling every UHF solution is a GHF stationary point
(block-diagonal spinor density). For unfrustrated spin systems the GHF
minimum is collinear and \(E_{\mathrm{GHF}}=E_{\mathrm{UHF}}\) exactly. That
collapse is a correctness check: on H₂ the GHF energy must match UHF to
\(\sim 10^{-8}\,\mathrm{Ha}\).

The freedom is useful under **spin frustration**. Equilateral H₃ cannot
satisfy three pairwise antiparallel preferences in a collinear picture. GHF
can arrange three coplanar moments at \(120^\circ\) (a Néel fan). After
convergence the Mulliken spin vectors (real algebra: magnetisation in the
\(x\)–\(z\) spin plane) have equal lengths and mutual angles of \(120^\circ\).

### Example: triangular H₃

```cpp
// equilateral H3, side 1.0 Å, STO-3G
cpu::scf::GHF ghf(3); // total electron count for spinor aufbau
ghf.basis = cpu::scf::BasisSet::STO3G;
ghf.tol = 1e-12;
ghf.guess_spin_rotation = 2.0 * M_PI / 3.0; // geometry-adapted Néel angle
ghf.run(frame);
// energy ≈ −1.340440 Ha, below best collinear UHF by ~4.5 mHa
```

**The initial guess must break collinearity.** A collinear guess is invariant
under the SCF map (the Fock matrix stays block diagonal), so DIIS can converge
cleanly to a collinear saddle whose energy equals UHF. Setting
`guess_spin_rotation = 0` exhibits that trap (unit-tested). Construction of
the non-collinear guess follows the usual local-moment recipe: unrotated
charge density plus site moments with angles
\(\theta_\mu = \texttt{guess\_spin\_rotation}\cdot\mu\). Angles that are too
small fall back into the collinear basin; for triangular H₃ use \(2\pi/3\).

On the same geometry collinear UHF itself has two stationary points (a pure
doublet and a lower broken-symmetry solution). Cold-start UHF in this engine
lands on the former; PySCF’s MINAO often lands on the latter. The non-collinear
GHF minimum lies below both; full verification is in
`regressions/scf/h3_ghf_sto3g/REFERENCE.md`.

Analytic nuclear forces are supported (`ghf.forces(frame)`; pin in
`regressions/scf/ghf-grad-h3-sto3g/`). Unsupported (explicit errors):
`mp2()`/`fci()`, `soscf`, `level_shift`, EDIIS/ADIIS, non-Hcore guesses.
Spinor exchange requires a stored dense ERI (`compute_eri`, on by default).

| System | Quantity | Engine | PySCF 2.14.0 |
|---|---|---:|---:|
| equilateral H₃ | \(E\) GHF (non-collinear) / Ha | −1.340440343541 | −1.340440343541 |
| equilateral H₃ | \(E\) UHF (broken collinear) / Ha | −1.335980054018 | −1.335980054018 |
| equilateral H₃ | \(E\) UHF (symmetric doublet) / Ha | −1.328175755381 | −1.328175755381 |
| equilateral H₃ | \(\|P^{\alpha\beta}\|_F\) GHF | 0.978059 | 0.978059 |
| equilateral H₃ | max \|g\| GHF / Ha·Bohr⁻¹ | 0.109386 | 0.109386 |
| H₂ (\(R=0.74\)) | \(E_{\mathrm{GHF}}-E_{\mathrm{UHF}}\) / Ha | \(<10^{-15}\) | 0 (theorem) |

The next chapter keeps the spin-resolved structure of UHF but changes the mean
field itself. Instead of explicit Hartree–Fock exchange, it uses an
exchange-correlation functional on a real-space grid:
[Chapter 5](ch05-kohn-sham-dft.md).
