# Systems and Units

Before choosing a solver, we need to decide what the calculation represents.
Two argon atoms in empty space, a water molecule in solution, and a repeating
crystal are different physical systems even when their inputs all begin with
a list of coordinates. Making those choices explicit is the first step
toward a result that another person can understand and reproduce.

**Learning goals.** Describe a state, identify boundary conditions, convert
energies and forces consistently, and separate model error from numerical error.

**Before you begin.** Coordinates and elementary algebra are enough. Return
to [Mathematical Tools](mathematical-tools.md) when derivatives first appear.

## Energy, force, and state

A configuration contains nuclear positions
$\mathbf R=(\mathbf R_1,\ldots,\mathbf R_N)$. A classical dynamical state also
contains velocities and masses. A potential assigns an energy $U(\mathbf R)$
to each configuration; the force on atom $A$ is

$$\mathbf F_A=-\nabla_A U.$$

The gradient collects the slopes along three coordinate directions. The
minus sign means that an unconstrained force points downhill in energy.
Finding a force and integrating motion are separate tasks: an energy model
does not, by itself, specify how a trajectory is advanced.

A periodic cell repeats its contents by lattice translations. An isolated
boundary condition does not. A large box may reduce image interactions, but
does not change the mathematical boundary condition of a periodic solver.
Record both the coordinates and the boundary treatment.

## Dimensional reasoning

The companion examples use eV, Å, fs, and amu for frame quantities. A force
therefore has units eV/Å and pressure has units eV/Å³. Electronic method
objects may additionally expose Hartree energies and bohr lengths. An energy
conversion alone is insufficient for a force: both its numerator and its
length denominator must be converted.

For orientation, $1\,E_h\simeq27.211386$ eV and $1\,a_0\simeq0.529177$ Å,
so $1\,E_h/a_0\simeq51.4221$ eV/Å. Use the project's unit registry for
implementation, rather than repeatedly entering rounded constants. The
eV–Å–fs–amu convention also requires a conversion factor in Newton's equation;
its numerical values are not a coherent SI unit system.

More precision cannot repair a missing physical interaction. Distinguish
error in the model, error in solving that model, and uncertainty from finite
sampling. Each calls for a different convergence or validation experiment.

## Specifying the physical problem

A coordinate file is only part of a model. Before calculating, record:

- **Particles and interactions.** Identify the elements or force-field types,
  masses, interaction model, and parameter source. A classical type label is
  not automatically an atomic number.
- **Electronic state, when needed.** Specify total charge and spin convention,
  as well as the basis or pseudopotential. Geometry alone does not determine
  the electronic problem supplied to a solver.
- **Boundaries.** State which directions are periodic and how long-range
  interactions are treated. For an isolated calculation, identify the actual
  isolated-boundary implementation rather than relying on a large cell alone.
- **Initial motion.** For dynamics, give velocities or their initialization
  distribution, any removed center-of-mass motion, and the random seed.
- **Requested result.** An energy at fixed geometry, a stationary structure,
  and a time average require different calculations and stopping criteria.

Keep this description alongside the input, not only in a lab notebook that
another reader cannot access. It provides a practical starting point for
reproducing a result or explaining why two calculations differ.

## Three kinds of uncertainty

**Model error** concerns the physics omitted or approximated: for example,
a pair model may miss a material's environmental response. **Numerical error**
concerns the solution of that model: a time step, basis cutoff, convergence
tolerance, or rounding choice may be inadequate. **Sampling uncertainty**
concerns estimating an average from a finite, often correlated trajectory.

Choose a check that addresses the relevant question. Finite differences test
whether a force is consistent with an energy. Time-step refinement tests
integration. Independent equilibrated runs test reproducibility of a
statistical estimate. Comparison with experiment also tests the model and
the correspondence between simulated and experimental conditions.

Two implementations can agree because they share the same approximation,
input mistake, or data source. Conversely, an apparent disagreement can be
a difference in units or boundaries rather than an algorithmic failure.
The [external-reference appendix](appendix-pyscf-oracle.md) explains how to
separate these possibilities without treating a reference program as infallible.

## Practice plan

Begin with two argon atoms on one axis. Write their positions as
$(0,0,0)$ and $(r,0,0)$, with positive separation $r$, and specify masses,
pair parameters, and boundaries. The pair model is developed in
[Molecular Dynamics](ch10-classical-md.md); here the aim is to learn which
inputs and checks must accompany even this small calculation.

1. Record the units of each input and each requested output. Convert one
   force between energy-per-length conventions by converting both factors.
2. Translate both atoms by the same vector, keeping the boundary treatment
   consistent, and exchange their labels. An isolated, field-free model of
   identical atoms should give the same energy. This tests symmetry, not
   agreement with experiment.
3. Check one force component against central energy differences at several
   displacements. Use a separation away from a cutoff or singularity.
4. Save the model description, measured difference, tolerance, and software
   revision. A failed check identifies something to investigate; it is not
   a reason to hide the result or immediately make the calculation larger.

The frame and water fragments below introduce data storage and an electronic
force check. They are not substitutes for a complete two-argon setup, and
allocation alone does not initialize a physical state.

## Setup fragments

These original fragments and build notes are retained for reference.
They may omit initialization; use the complete model example when running code.

```cpp
cpu::Frame f(n);
auto x = f.span(kAtom_x);          // (Å)
f.box = cpu::Box::cubic(L);     // (Å)
// after a force evaluation:
double E = f.meta.get(kEnergy);    // (eV) for classical/AM1; SCF writes eV too
```

## Frame initialization

### The frame holds the system

In these CPU examples,
[`cpu::Frame`cpp
using namespace ATV;

cpu::Frame frame(3);                       // three atoms: an H2O

auto type = frame.span(kAtom_atomic_number);
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
    frame.span(kAtom_atomic_number)[i] = type[i];
    frame.span(kAtom_x)[i] = x[i];
    frame.span(kAtom_y)[i] = y[i];
    frame.span(kAtom_z)[i] = z[i];
    frame.span(kAtom_mass)[i] = mass[i];
}
frame.box = cpu::Box::cubic(100.0);
```

Two conventions here are load-bearing and reused in every quantum chapter:
`frame.span(kAtom_atomic_number)[i]` stores the **atomic number Z** — the
species column. The force-field type index is a separate column,
`kAtom_type`, read only by per-type force-field tables; a quantum method
never reads it. Positions are in **Ångström**. Results land back on the same
object: `frame.meta.get(kEnergy)` (potential energy, eV) and the
`kAtom_fx/fy/fz` columns (eV/Å) after a driver runs.

### The box is `cpu::Box`

The simulation cell is [`cpu::Boxcpu::Box::cubic(L)` and assigned to
`frame.box`. It owns the 3×3 cell matrix plus a per-step-immutable cache and
exposes the queries force kernels need:

```cpp
cpu::Box box = cpu::Box::cubic(20.0);      // or Box::orthorhombic(lx, ly, lz)
double lx  = box.length(0);                // edge along x (Å)
double vol = box.volume();                 // [Å^3]
bool  cube = box.is_orthorhombic();        // true here
```

A cell object and a periodic Hamiltonian are not the same thing. A Gaussian
molecular solver may store a cell without summing over periodic images;
a periodic electrostatic solver may still couple images in a large cell.
Check the evaluator's boundary convention and, where relevant, converge
the vacuum spacing. For condensed-phase MD, the cell and periodic-image
rules are part of the physical model.

### Converting units

The frame convention used in these examples is:
**energy in eV, length in Å, time in fs, mass in amu, temperature in K,
force in eV/Å.** Electronic-structure solvers work *internally* in
Hartree/Bohr/Rydberg but convert into `frame.meta.get(kEnergy)` in eV before
returning, so these examples read frame energies in eV. Do not equate this convention
with another program's similarly named unit style; compare each base unit,
particularly the time unit.

When you do need to cross the bridge — say to compare a force against a
PySCF gradient reported in Hartree/Bohr — do it explicitly through
[`core/units.h`cpp
using namespace ATV::units;

double r_bohr  = convert(0.74, Length::Angstrom, Length::Bohr);   // 1.3984
double e_ev    = convert(-1.1167, Energy::Hartree, Energy::eV);   // -30.39
double f_au    = convert(f_ev_A, Force::eVPerAngstrom, Force::HartreePerBohr);
```

The conversion factors (`BOHR = 0.52917721092 Å`,
`HARTREE2EV = 27.211386245988`) are the values quoted by the inherited reference material. Using the same
conversion constants removes one source of discrepancy; it does not imply
bitwise agreement of integrals or other calculations. Named constants like
`ATV::units::kB_ev_per_k` (Boltzmann, eV/K) and `ATV::units::kHartreeInEv` are
available for the common conversions.

The following numerical material is retained from the source snapshot.
It was not rerun for this editorial revision; match its input and provenance
before using it as a validation target.

## Numerical precision

Floating-point formats trade memory and speed against rounding error.
The reference implementation distinguishes precision by **role**, in [`core/precision.hfloat`? |
|---|---|---|---|
| `ATV::StorageReal` | per-atom persistent state (x, v, f as stored) | `double` | yes |
| `ATV::WorkReal` | transient per-kernel compute scalars | `double` | yes |
| `ATV::AccReal` | accumulators, energy, virial, SCF matrices | `double` | **never** |

`StorageReal` and `WorkReal` can be lowered to single precision at build
time with `-DATV_REAL_STORAGE=float` and `-DATV_REAL_WORK=float` — the
LAMMPS "mixed mode" recipe, where pairwise math runs in single precision but
*accumulates* into double. `AccReal` is `double` by construction and has no
override: it is the one place precision is never economized, to reduce accumulation error in energies and electronic matrices. This
policy is a safeguard, not a proof that a computed result is accurate.

The CUDA backend has its own single knob, `-DATV_CUDA_PREC_MODE=double|mixed|single`,
and assigns one type per physical role (`cuda/frame/types.cuh`). The
default `double` is FP64 throughout. `mixed` follows OpenMM: force math,
splines and FFTs in FP32; coordinates stored as an FP32 value plus an FP32
correction and reconstructed in FP64 for integration; velocities, masses,
thermostat/barostat state, energies and virials in FP64; forces accumulated
in 64-bit fixed point (OpenMM's resolution and range, about ±2.2·10⁶ eV/Å).
A force outside that range stops integration and is reported with the atom
tag and step when the run returns.

### Integer tiers

The same "size by role" idea applies to integers (a port of LAMMPS
`lmptype.h`), so 64-bit width is paid only where a count can actually
overflow 32 bits:

| tier | role | SMALLBIG | BIGBIG (default) |
|---|---|---|---|
| `ATV::LocalInt` | on-rank counts / indices | int32 | int32 |
| `ATV::ImageInt` | one periodic image flag (per component) | int32 | int32 |
| `ATV::TagInt` | global atom / molecule IDs | int32 | int64 |
| `ATV::BigInt` | global totals (natoms, ntimestep) | int64 | int64 |

The default `-DATV_ID_SIZE=bigbig` keeps `TagInt` 64-bit for billion-atom
runs; `smallbig` narrows it to 32-bit.
`ImageInt` stays int32 in both builds: Atomiverse stores the three image
counters unpacked, so each reaches ±2³¹ box crossings (LAMMPS packs three
into one `imageint`, ±2²⁰ each under BIGBIG). `BigInt` is always 64-bit to
increase the representable range of global counts; any finite-width integer
still has a maximum value.
There is no catch-all floating-point or integer type: every declaration names
the tier matching its physical role.

A wider accumulator cannot recover information already lost when a position
or an intermediate force was rounded. Test mixed precision against a
well-converged calculation of the same model, using the property that will
support the physical conclusion. Bitwise agreement is a different requirement
from agreement within a justified numerical tolerance.

## Finite-difference checks

The following water fragment illustrates the structure of an electronic
force check. It contains a geometry placeholder and is not a standalone
program; use a complete example matching the selected source revision.

An analytic nuclear gradient is fast but easy to get subtly wrong (a missing
Pulay term, a sign error, a grid-response omission). The independent oracle
is the *definition* of a derivative: displace each coordinate by ±h, run the
energy at both points, and take the central difference. Agreement over a stable displacement range supports consistency of the
energy and gradient at the tested geometries. It does not establish that
the Hamiltonian is physically adequate or that the two paths share no error.

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
  chosen model's numerically evaluated energy surface, so if the SCF is only loosely converged the ±h
  energies carry convergence noise and the FD gradient is dominated by it.
  For a tight check, pass a tight tolerance to the constructor
  (`RHF{basis, 0, /*tol=*/1e-12}`) and set `rhf.conv_tol_grad = 1e-9`.
  (This is exactly what the vibration tests do — see
  [Molecular Vibrations](ch14-vibrations.md).)
- **The step size `h` trades truncation against round-off.** The inherited fragment quotes
  `5e-5 Å` as a starting value, not a universally optimal displacement.
  Seek a stable interval for the chosen geometry, energy accuracy, and
  target tolerance, as explained in [Mathematical Tools](mathematical-tools.md).

### The result and its class

The book classifies every result claim so you know *what kind of check
applies*. Following the validation policy:

| class | meaning | check |
|---|---|---|
| exact | bit-reproducible | golden file / `static_assert` (integer counts, phase order) |
| numerical tolerance | controlled discretization, iteration, and rounding errors | abs/rel compare (analytic ∇ vs FD **1e-5 Ha/Bohr**) |
| physical approximation | method/basis error | compare to a converged reference |
| stochastic variation | RNG-seeded dynamics | seed control + ensemble means |

The analytic-versus-finite-difference comparison concerns two estimates of
the same model derivative. Differences can reflect displacement truncation,
incomplete electronic convergence, rounding, or an implementation error.
Vary the numerical controls before diagnosing a bug. An RHF comparison with
experiment introduces further differences in the physical model and in
experimental conditions; it should not be interpreted as a code-only test.

## Reference provenance

Analytic gradients are checked end-to-end against PySCF in
`h2o_rhf_sto3g_grad`
and, on a larger basis, in
`h2o_rhf_def2svp_grad`.
The gradient-result plumbing itself has a unit test in
[`tests/cpu/scf/test_gradient_result.cpp`](https://github.com/MolCrafts/Atomiverse/blob/master/tests/cpu/scf/test_gradient_result.cpp). The tiered-precision types are
defined and documented in
`precision.h`;
the unit registry in
`units.h`.

## Questions for reflection

Why does converting Hartree to eV not finish a force conversion? Which error would remain if a calculation used exact arithmetic but an unsuitable potential?
