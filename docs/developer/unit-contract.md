# Units across the project

A run chooses one LAMMPS style with `set_units` (Python: `units=`). Bare
component parameters use that style. Frame stores numeric data and serializable
source-unit metadata; it has no unit algorithms or unit setters.

## One definition of every physical scale

`core/physical_units.h` defines all eight styles against SI. Mass and energy
with a mole normalization resolve to per-particle values. `core/unit_constants.h`
is the source of physical constants. `core/unit_system.h` prepares the equation
conversions from these definitions: kinetic energy, velocity impulses, density,
pressure, electric field, voltage and Coulomb coupling. There is no separate
hand-maintained table of equation factors.

Electric potential and electric field have independent declarations. In
particular, LAMMPS `electron` uses V/cm for fields and V for potential. Pressure
uses each style's declared pressure unit; Frame virial and kinetic stress remain
extensive quantities with energy dimensions.

LJ can run as a reduced model without any physical reference. Coupling it to SI,
a physical force field, a quantum solver or a physical clock requires a
`ReducedReference` containing an identity and positive finite mass [kg], length
[m] and energy [J] per particle. Time, charge and temperature derive from those
scales. No material reference is invented.

## Preparation and numerical kernels

`Dimension`, `Unit` and `Quantity` in `core/quantity.h` compose dimensions and
check conversions at preparation boundaries. Factories accepting a style are
named `from_style`. String parsing belongs at input boundaries.

```cpp
constexpr auto real = units::PhysicalUnits::from_style(units::Style::real);
constexpr auto metal = units::PhysicalUnits::from_style(units::Style::metal);
double ev = units::convert<real.energy(), metal.energy()>(value);
```

For runtime-selected styles, `Conversion` stores exactly one `double` and is
trivially copyable. Its call operator is a scalar multiplication. A fix prepares
its coefficients in `build()` and uses raw numeric arrays during execution:

```cpp
void FixEfield::build(const BuildContext& ctx) {
    field_units_ = ctx.runtime.units.electric_field;
}
void FixEfield::launch(const LaunchContext& ctx) {
    const double ex = field_units_(Ex);
    // The atom loop uses ex directly: fx[i] += charge[i] * ex.
}
```

Public field edits remain effective on the next launch. CUDA kernel argument
blocks carry prepared numeric coefficients. Numerical loops do not parse names,
query a registry, allocate quantities or check dimensions.

`python3 tools/check_unit_codegen.py --compiler /path/to/g++` compares optimized
assembly with handwritten arithmetic for dynamic scaling, identity conversion
and compile-time dimension conversion. Identity conversion disappears; the other
two cases have the same instructions as their handwritten equivalents.
Actual transformation of arrays still requires reading, scaling and writing
those arrays. CUDA's fixed-point force representation retains its finite
resolution; unit conversion does not remove representation error.

## Data and solver boundaries

Free operations in `cpu/units/frame.h` handle Frame exchange:

- `annotate(frame, profile)` declares what existing values mean.
- `convert(frame, profile, optional_reference)` returns transformed data.
- Passing an lvalue preserves it; passing an rvalue reuses its buffers.
- `for_frame(run, frame)` binds serialized LJ reference metadata to a run.

External molpy input requires `meta["units_style"]`. Imports adapt it to the
selected run style and exports restore the destination's source style. Matching
styles skip conversion. Internal execution buffers can be annotated from their
owning run's already-known convention. Readers of files without unit metadata
require an explicit source profile. Checkpoints and recorders persist source
style and reduced-reference metadata.

```python
from atomiverse import units
from atomiverse.md import MD, FixEfield
from atomiverse.md.integrator import VelocityVerlet

units.annotate(frame, "real")       # existing data: Angstrom, fs, amu, e
sim = MD(units="real")
sim.set_integrator(VelocityVerlet(0.5))
sim.set_evaluator(FixEfield(0.02, 0.0, 0.0))  # V/Angstrom
sim.run(frame, 100)

run_units = units.from_style("real")
kT = run_units.thermal_energy(300.0)  # energy in the selected style
```

`atomiverse.units.convert(frame, target, reference=...)` returns a converted copy
for explicit exchange. `forcefield.compile` converts parameters dimension by
dimension from the force field's declared style into the selected run style,
including masses, Coulomb coefficients and bonded coefficients. Source Frame
units are independent of force-field parameter units. C++ coefficient imports
have the same explicit boundary:

```cpp
auto coefficients = cpu::load_lammps_coeffs(path,
    units::PhysicalUnits::from_style(units::Style::real),
    units::PhysicalUnits::from_style(units::Style::metal));
```

AM1-BCC accepts the molecular graph's source convention through its `units=`
argument; its geometry is prepared for AM1 and its returned charges use that
source convention. An LJ graph additionally supplies a reduced reference.

Canonical floating columns and scalar metadata have unit declarations in
`core/field_units.h`, including energy channels, extensive stress, multipoles,
electrode charge, voltage and capacitance. Identity, topology and image columns
remain integer data. Unknown floating fields fail at conversion boundaries;
new physical outputs must add an explicit schema declaration.

Electronic kernels keep their documented native convention (`electronic`:
Angstrom, eV, fs, amu, e), distinct from LAMMPS `electron` (Bohr, Hartree, fs).
Solver boundaries prepare native geometry and publish energies, forces, charges
and extensive stress back in the outer run's units. Publication preserves parent
array allocations and cached column handles. Molecular mean field, post-HF,
semiempirical, PW/GPW and QM/MM paths use these boundaries. Native wavefunction
and spectroscopic result objects retain their explicitly documented units;
Frame results follow the run. Vibrational finalization converts its workspace
into the units declared by the spectroscopic result before mass weighting.

CUDA nested electronic solvers use a persistent private device buffer only when
physical scales differ. One fused import and export surrounds the solve; SCF
iterations do not perform conversions. Compatible physical scales borrow the
parent Frame directly. Input and output stay on the device. The integrator owns
the run clock; consumers requiring femtoseconds declare that requirement and
receive a prepared timestep after any LJ reference is bound.
Nested methods inherit the public run convention even while their numerical
context temporarily uses native units. Asynchronous output caches its prepared
unit coefficients at build time, so a later native solve cannot change the
meaning of already captured data.

## Numerical compatibility

The physical definitions replace historical rounded equation constants. Small
changes from old golden outputs are expected and must be explained where those
goldens are compared. Electric fields follow the declared V/um and V/nm units in
micro/nano styles; their force conversions are respectively 1000 and
160.2176634, replacing the inconsistent historical factors of one. QEq uses the
run's Coulomb prefactor; electrode solvers convert public voltage before forming
their energy/charge equations. There is no legacy unannotated external-input
fallback.
