# Chapter 1. Systems and Units

Units are eV, Å, fs, amu, eV/Å, K. Precision is tiered
(`StorageReal` / `WorkReal` / always-double `AccReal`). Frame is
`blocks` + `meta` + `box` only.

Every later chapter reuses this model. Finite-difference gradient checks are the
universal validation: analytic forces from any evaluator must match energy
differences within a documented tolerance before they feed MD or Minimize.

```cpp
cpu::Frame f(n);
auto x = f.span(kAtom_x);          // (Å)
f.box = cpu::Box::cubic(L);     // (Å)
// after a force evaluation:
double E = f.meta.get(kEnergy);    // (eV) for classical/AM1; SCF writes eV too
```

Cross-check the textbook energy in the RHF results table of Chapter 2 —
an end-to-end proof that units, Frame, and SCF agree.

Before any method — Hartree-Fock, DFT, molecular dynamics — there is a
system: a set of atoms in a box, described in a fixed set of units, stored
at a chosen numerical precision, and only trusted once it has been
*validated* against an independent oracle. This chapter is about those four
foundations. It runs almost no physics; instead it establishes the
vocabulary and the discipline every later chapter takes for granted.

The six questions this book asks of every topic apply here too:

1. **The concept.** What is a "system" in computational chemistry, and why
   do units and precision deserve a chapter of their own?
2. **The Atomiverse representation.** How does a `cpu::Frame` hold a system,
   which units are canonical, and what the tiered precision model buys you.
3. **The C++ API.** `cpu::Frame`, `cpu::Box` / `Mat3`, `core/units.h`,
   `core/precision.h`, and the finite-difference gradient checker.
4. **The expected result.** What "correct" looks like — and its numerical
   class.
5. **The validation.** How a finite-difference gradient certifies an
   analytic one.
6. **The recurring pattern.** Why the same "one canonical value, one
   independent check" discipline reappears in every method chapter.

If the architecture (Driver / Component / Frame) is new to you, read
[Chapter 0](ch00-architecture.md) first. This
chapter assumes you know that a `cpu::Frame` is the shared state object and
picks up from there.

## The Concept: A System Is Coordinates Plus Conventions

A molecular or condensed-phase system is, at bottom, a list of nuclei with
positions and charges, optionally a periodic cell, and — for dynamics — a
velocity per atom. That much is obvious. What is less obvious to newcomers
is that the *conventions* wrapped around those numbers are as much a part of
the system as the numbers themselves:

- **Units.** A bond length of `0.74` is meaningless until you say Ångström.
  A force of `0.01` is meaningless until you say eV/Å (or Hartree/Bohr).
  Quantum chemistry and molecular dynamics grew up in different unit systems
  — Hartree/Bohr on one side, eV/Å/fs/amu on the other — and mixing them
  silently is the single most common source of "my energy is off by 27×"
  bugs.
- **Precision.** Every number is a finite-width float. Where you spend 64
  bits and where you can afford 32 is a deliberate engineering choice, not
  an accident of the compiler.
- **Validation.** A gradient that *looks* right is worthless; a gradient
  that agrees with a finite-difference probe of its own energy to
  1e-5 Ha/Bohr is trustworthy.

Atomiverse fixes each of these as an explicit, documented policy so that a
number read out of a `cpu::Frame` always means exactly one thing.

## The Representation: One Frame, One Unit System

### The frame holds the system

There is no separate "System" or "Molecule" class. The
[`cpu::Frame`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/frame/frame.h) *is* the system.
You size it once with the atom count, fill the per-atom arrays, set the box,
and hand it to a driver:

```cpp
using namespace ATV;

cpu::Frame frame(3);                       // three atoms: an H2O

auto type = frame.span(kAtom_type);
auto x = frame.span(kAtom_x);
auto y = frame.span(kAtom_y);
auto z = frame.span(kAtom_z);
auto mass = frame.span(kAtom_mass);

type[0] = 8; type[1] = 1; type[2] = 1;     // atomic number Z
x[0] = 0.0; y[0] = 0.0;     z[0] = 0.1178;
x[1] = 0.0; y[1] = 0.7555;  z[1] = -0.4712;
x[2] = 0.0; y[2] = -0.7555; z[2] = -0.4712;

mass[0] = 15.999; mass[1] = 1.008; mass[2] = 1.008; // amu

frame.box = cpu::Box::cubic(100.0);      // a big non-interacting cell (Å)
```

For a small molecule the initializer-list form used throughout the test
suite is more compact and just as valid:

```cpp
cpu::Frame frame(3);
const int type[] = {8, 1, 1};
const double x[] = {0.0, 0.0, 0.0};
const double y[] = {0.0, 0.7555, -0.7555};
const double z[] = {0.1178, -0.4712, -0.4712};
const double mass[] = {15.999, 1.008, 1.008};
for (int i = 0; i < frame.nlocal(); ++i) {
    frame.span(kAtom_type)[i] = type[i];
    frame.span(kAtom_x)[i] = x[i];
    frame.span(kAtom_y)[i] = y[i];
    frame.span(kAtom_z)[i] = z[i];
    frame.span(kAtom_mass)[i] = mass[i];
}
frame.box = cpu::Box::cubic(100.0);
```

Two conventions here are load-bearing and reused in every quantum chapter:
`frame.span(kAtom_type)[i]` stores the **atomic number Z** (not a force-field type
index), and positions are in **Ångström**. Results land back on the same
object: `frame.meta.get(kEnergy)` (potential energy, eV) and the
`kAtom_fx/fy/fz` columns (eV/Å) after a driver runs.

### The box is `cpu::Box`

The simulation cell is [`cpu::Box`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/cpu/frame/box.h),
constructed most often with `cpu::Box::cubic(L)` and assigned to
`frame.box`. It owns the 3×3 cell matrix plus a per-step-immutable cache and
exposes the queries force kernels need:

```cpp
cpu::Box box = cpu::Box::cubic(20.0);      // or Box::orthorhombic(lx, ly, lz)
double lx  = box.length(0);                // edge along x (Å)
double vol = box.volume();                 // [Å^3]
bool  cube = box.is_orthorhombic();        // true here
```

For an isolated molecule the box only has to be large enough that periodic
images do not interact; `Mat3::cubic(100.0)` is the idiom used across the
SCF tests. For condensed-phase MD the box is the physical cell and its
minimum-image helper (`min_image`) is on the hot path — see
[Chapter 10](ch10-classical-md.md).

### Units are eV / Å / fs / amu — with one explicit bridge to Hartree/Bohr

The MD-internal ("metal") unit system is the canonical one:
**energy in eV, length in Å, time in fs, mass in amu, temperature in K,
force in eV/Å.** Electronic-structure solvers work *internally* in
Hartree/Bohr/Rydberg but convert into `frame.meta.get(kEnergy)` in eV before
returning, so the frame you read is always in metal units.

When you do need to cross the bridge — say to compare a force against a
PySCF gradient reported in Hartree/Bohr — do it explicitly through
[`core/units.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/units.h), never with a
hand-typed magic number:

```cpp
using namespace ATV::units;

double r_bohr  = convert(0.74, Length::Angstrom, Length::Bohr);   // 1.3984
double e_ev    = convert(-1.1167, Energy::Hartree, Energy::eV);   // -30.39
double f_au    = convert(f_ev_A, Force::eVPerAngstrom, Force::HartreePerBohr);
```

The conversion factors (`BOHR = 0.52917721092 Å`,
`HARTREE2EV = 27.211386245988`) are pinned to PySCF's `nist.py` so integrals
match the ground-truth path bit-for-bit. Named constants like
`ATV::units::kB_ev_per_k` (Boltzmann, eV/K) and `ATV::units::kHartreeInEv` are
available for the common conversions.

## The Precision Model: Spend Bits Where They Matter

A naïve engine picks one floating type — usually `double` — and uses it
everywhere. Atomiverse instead follows LAMMPS and *tiers* precision by
**role**, in [`core/precision.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/precision.h).
The principle: the wide, safe type is the default; a narrower type is spent
only at a deliberately chosen role, never at an accumulator where it would
corrupt results.

### Floating-point tiers

| tier | role | default | may lower to `float`? |
|---|---|---|---|
| `ATV::StorageReal` | per-atom persistent state (x, v, f as stored) | `double` | yes |
| `ATV::WorkReal` | transient per-kernel compute scalars | `double` | yes |
| `ATV::AccReal` | accumulators, energy, virial, SCF matrices | `double` | **never** |

`StorageReal` and `WorkReal` can be lowered to single precision at build
time with `-DATV_REAL_STORAGE=float` and `-DATV_REAL_WORK=float` — the
LAMMPS "mixed mode" recipe, where pairwise math runs in single precision but
*accumulates* into double. `AccReal` is `double` by construction and has no
override: it is the one place precision is never economized, because an
energy sum or a Fock-matrix element in `float` would swamp the physics with
round-off. The CUDA backend mirrors these exactly as `real_t` / `work_t` /
`prec_t` (`cuda/frame/types.cuh`).

### Integer tiers

The same "size by role" idea applies to integers (a port of LAMMPS
`lmptype.h`), so 64-bit width is paid only where a count can actually
overflow 32 bits:

| tier | role | SMALLBIG (default) | BIGBIG |
|---|---|---|---|
| `ATV::LocalInt` | on-rank counts / indices | int32 | int32 |
| `ATV::ImageInt` | periodic image flags | int32 | int64 |
| `ATV::TagInt` | global atom / molecule IDs | int32 | int64 |
| `ATV::BigInt` | global totals (natoms, ntimestep) | int64 | int64 |

`-DATV_ID_SIZE=bigbig` widens `TagInt` and `ImageInt` to 64-bit for
billion-atom runs; `BigInt` is always 64-bit so a global count never wraps.
There is no catch-all floating-point or integer type: every declaration names
the tier matching its physical role.

You rarely construct these types by hand — the frame's arrays already carry
the right tier. What matters is understanding *why* an energy read from
`frame.meta.get(kEnergy)` is trustworthy even in a mixed-precision build: the
storage may be `float`, but everything that *sums* is `double`.

## The Validation Discipline: Finite Differences Certify a Gradient

Here is the concrete, runnable pay-off of this chapter — and the template
for how every later result is checked.

An analytic nuclear gradient is fast but easy to get subtly wrong (a missing
Pulay term, a sign error, a grid-response omission). The independent oracle
is the *definition* of a derivative: displace each coordinate by ±h, run the
energy at both points, and take the central difference. If the analytic
gradient and the finite-difference gradient agree, both are almost certainly
correct.

Atomiverse ships this oracle as a public function so you can validate any
geometry yourself, using the same
`RHF::forces` (private gradient engine) /
`scf_finite_difference_gradient` pair the test suite uses:

```cpp
using namespace ATV;

cpu::Frame frame = /* H2O, positions in Å, type = Z */;

cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G, /*charge=*/0,
                  /*tol=*/1e-12};   // tight for FD comparisons
rhf.run(frame);

// Analytic forces [n_atoms, 3] eV/Å — method owns private workspace.
auto analytic = rhf.forces(frame);
// max |F| or FD comparisons use analytic.forces / analytic.energy
```

Two details are worth internalizing because they recur:

- **The SCF must be gradient-stationary.** Finite differences probe the
  *true* energy surface, so if the SCF is only loosely converged the ±h
  energies carry convergence noise and the FD gradient is dominated by it.
  For a tight check, pass a tight tolerance to the constructor
  (`RHF{basis, 0, /*tol=*/1e-12}`) and set `rhf.conv_tol_grad = 1e-9`.
  (This is exactly what the vibration tests do — see
  [Chapter 14](ch14-vibrations.md).)
- **The step size `h` trades truncation against round-off.** The default
  `5e-5 Å` sits in the sweet spot for double-precision energies; far smaller
  and catastrophic cancellation dominates, far larger and the
  central-difference truncation error `O(h^2)` shows.

### The result and its class

The book classifies every result claim so you know *what kind of check
applies*. Following the validation policy:

| class | meaning | check |
|---|---|---|
| exact | bit-reproducible | golden file / `static_assert` (integer counts, phase order) |
| numerical tolerance | float round-off only | abs/rel compare (analytic ∇ vs FD **1e-5 Ha/Bohr**) |
| physical approximation | method/basis error | compare to a converged reference |
| stochastic variation | RNG-seeded dynamics | seed control + ensemble means |

The analytic-vs-FD gradient agreement is a **numerical-tolerance** claim: the
two computations are of the *same* quantity, so any disagreement beyond
round-off is a bug. Contrast that with an RHF energy vs experiment, which is
a *physical-approximation* claim — the gap is the method, not the code.
Never report a number without saying which class it belongs to.

## The Recurring Pattern: One Canonical Value, One Independent Check

The discipline you just saw — compute a quantity one way, then confirm it a
completely different way that shares no code — is the spine of the whole
manual:

- **Gradients** (this chapter, [Chapter 6](ch06-hybrid-dft.md)): analytic
  vs finite difference.
- **Frequencies** ([Chapter 14](ch14-vibrations.md)): the finite-difference
  Hessian *is* the independent check on the analytic gradient, then compared
  against PySCF's analytic Hessian.
- **XC functionals** ([Chapter 5](ch05-kohn-sham-dft.md)): the native implementation vs
  libxc to 1e-12.
- **Energy conservation** ([Chapter 10](ch10-classical-md.md)): an NVE
  trajectory that keeps total energy flat is the integrator's own
  self-check.
- **Backends**: CPU is the reference; CUDA is *compared*, not assumed
  identical, because it runs mixed precision.

CPU is always the reference backend, and every physical quantity has a CPU
golden value. That is why this foundations chapter lives here: units,
precision, and the finite-difference habit are what make a "golden value"
mean something.

## Validation Reference

The finite-difference gradient checker shown here is exercised in
[`tests/cpu/scf/test_gradient.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/cpu/scf/test_gradient.cpp)
(RHF/UHF/KS analytic gradients vs central differences, STO-3G) and, for
hybrids on a moving XC grid, in
[`tests/cpu/scf/test_hybrid_gradient.cpp`](https://github.com/MolCrafts/Atomiverse/blob/dev/tests/cpu/scf/test_hybrid_gradient.cpp)
(analytic vs finite-difference to **1e-5 Ha/Bohr**, plus translational
invariance `|Σ_A F_A| ≤ 1e-8 Ha/Bohr`). The tiered-precision types are
defined and documented in
[`include/atomiverse/core/precision.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/precision.h);
the unit registry in
[`include/atomiverse/core/units.h`](https://github.com/MolCrafts/Atomiverse/blob/dev/include/atomiverse/core/units.h).
