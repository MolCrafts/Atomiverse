# Python Package

Install, configure and run Atomiverse from Python: `pip install atomiverse`
for the pure-Python shell, then a native build from a source checkout. The
C++ toolchain, CMake flags and first C++ run are a separate page —
[Getting started](getting-started.md) — and the Driver / Component / Frame
vocabulary these bindings mirror is in [Core concepts](concepts.md).

## Installation

Atomiverse uses a **two-phase** Python install: a pure-Python package from
PyPI, then a native extension you build for your machine. Both phases go
through pip, so `pip uninstall atomiverse` removes everything and a new
release replaces everything.

### Phase 1 — pure Python package

```bash
pip install atomiverse
```

This installs the Python API shell and CLI only — **no C++ compile**, no
prebuilt `_atomiverse` binary. Requires Python ≥ 3.12, NumPy, and
`molcrafts-molpy` (the only user-facing Frame type). **Do not `import molpy`**
from user code — molpy's Rust core is an engine-private schema.

```bash
python -m atomiverse            # version + install guidance
python -m atomiverse status     # extension: not installed
```

Importing compute symbols (`atomiverse.MD`, …) without a native extension
raises `ImportError` pointing at phase 2.

### Phase 2 — native build from a checkout

The extension is compiled from the **source tag matching the installed
version**, together with a local molpy checkout, using the toolchain described in
[Getting started](getting-started.md) — a C++20 compiler, CMake, BLAS/LAPACK,
a Rust toolchain for molpy's native core, and FFTW3 for `pw` / `kspace`.

```bash
V=$(python -m atomiverse --version)
git clone --recursive --branch v$V https://github.com/MolCrafts/Atomiverse
export ATV_MOLPY_PATH=/absolute/path/to/merged/molpy
pip install 'atomiverse[native]'        # build-time: scikit-build-core, nanobind, maturin

# 1) config   resolve modules → CMake defines, record the build
python -m atomiverse config md pw --mpi --src Atomiverse

# 2) compile  pip wheel → scikit-build-core → CMake, and molpy from the local checkout
python -m atomiverse compile -j 8

# 3) install  pip install of both wheels over the pure package
python -m atomiverse install

python -m atomiverse status
python -m atomiverse run Atomiverse/examples/python/md_lj_argon.py
```

`compile` builds in `<src>/build-python` (override with `config
--build-dir`); a repeat `compile` reuses that CMake tree and only rebuilds
what changed. It builds two wheels into `<build-dir>/dist/`: the extension,
and molpy from `<ATV_MOLPY_PATH>/molpy` (cargo output kept in
`<build-dir>/molpy-target`, so that rebuild is incremental too). `install`
hands both to `pip install --force-reinstall --no-deps` — the extension and
its `_build_info.json` (the defines it was compiled with, which `status`
prints) are ordinary files of the installed distribution.

**Why molpy is rebuilt:** its Python extension and Atomiverse's `molcxx`
bridge exchange a shared native Frame handle. They must use the same local
source tree and compatible ABI. `install` replaces the installed molpy wheel
with the one built from that checkout. No molpy release is pinned yet.

**Modules** are positional on `config` only: `md`, `minimize`, `mc`, `evaluate`,
`vib`, `path`, `hf`, `pw`, `gpw`, `am1`, `cuda`, `kspace`, `potentials`,
`neighbor`. The wheel contains **exactly** the modules listed (plus their
closure) — every other product module is set OFF. Enabling `md` / `mc` /
`minimize` auto-enables `potentials` + `neighbor`; `path` pulls in
`minimize`; `kspace` pulls in `potentials`.

**Global options** (on `config`, never under a module name):

| Flag | CMake |
|------|--------|
| `--mpi` / `--no-mpi` | `ATV_USE_MPI` |
| `--real-storage {float,double}` | `ATV_REAL_STORAGE` |
| `--real-work {float,double}` | `ATV_REAL_WORK` |
| `--id-size {bigbig,smallbig}` | `ATV_ID_SIZE` |
| `-D KEY=VALUE` | unbound defines only (e.g. `-D BLA_VENDOR=OpenBLAS`) |

TOML recipe (XOR with positional modules; globals in `[global]` only):

```bash
python -m atomiverse config --config atomiverse-build.toml --src Atomiverse
```

```toml
[modules]
md = true
pw = true

[global]
mpi = true
real_storage = "double"
```

**`--hf`** is the flag spelling of the `hf` module: it configures Gaussian
SCF (`ATV_BUILD_HF=ON`) exactly like every other product module.

**Version lock:** the checkout's `__version__` must equal the installed
package version — `config` refuses otherwise, so a native build never
silently swaps the release you installed for a different tree.

**Dry-run** prints the resolved defines and the exact pip commands:

```bash
python -m atomiverse config md --mpi --src Atomiverse --dry-run
python -m atomiverse compile --dry-run
python -m atomiverse install --dry-run
```

**Developing Atomiverse itself:** `pip install -e 'python/[dev,native]'`
installs the source tree editable; the same `config` / `compile` /
`install` stages then build the extension against it.

## What this package exposes (and what it doesn't)

Atomiverse binds **drivers and components only** — there is no Atomiverse
Frame type. Simulation state crosses the boundary as a **`molpy.Frame`**:
each `run()` borrows the frame, advances a transient execution frame in C++,
and writes positions / velocities / forces / energy back into the same
`molpy.Frame`.

Current surface:

- **MD** — the `MD` driver and its components under the canonical C++
  (LAMMPS-style) names: `VelocityVerlet`, `NoseHooverNPT`,
  `ParrinelloRahmanNPT`, `InitVelocity`, `NoseHoover`, `Langevin`, `CSVR`,
  `BerendsenBarostat`, `NeighborList`, `Thermo`, `XYZ`.
- **Real-space force field** — the plural `set_evaluator` menu. Pair styles
  `PairLJCut`, `PairMorse`, `PairBuck`, `PairBorn`, `PairYukawa`, `PairWCA`
  and `PairDpd`; bonded styles `BondHarmonic`, `BondFene`, `AngleHarmonic`,
  `DihedralMultiHarmonic`, `DihedralCharmm` and `ImproperHarmonic`. A
  real-space style has no build flag of its own, so all of them are in every
  native build. The table under *MD driver* below gives the signatures and units.
- **Electrostatics** — `PairCoulCut` (truncated Coulomb) and the Ewald
  split `CoulLong` + `KSpacePME`. The reciprocal solver needs FFTW3, so
  `KSpacePME` is present only when the `kspace` module is built.
  `PairCoulCut` takes 1-2/1-3/1-4 weights and an explicit Coulomb constant,
  and `PairLJCut.table(...)` is the per-type LJ form with weights and an
  `energy_shift` switch — the two kernels a molpy `lj/cut` + `coul/cut`
  force field translates to (see *Force fields from molpy*).
- **Geometry optimization** — CPU `minimizer.Minimize` with three fixed-cell
  optimizers (`LBFGS`, `CG`, `FIRE`), four variable-cell relaxers
  (`IsotropicCellRelax`, `AnisotropicCellRelax`, `TriclinicCellRelax`,
  `JointIsotropicRelax`) and independent loop policies — `ForceConverged`
  for a fixed cell, `PressureConverged` / `StressConverged` for a variable
  one. Any force source `MD` accepts is accepted here.
- **Vibrational analysis** — `Hessian` (central finite differences over any
  registered force source) and `HessianResult` (wavenumbers, normal modes,
  RRHO thermochemistry). Present when the `vib` module is built
  (`python -m atomiverse config md vib …`).
- **Monte Carlo** — the three ensemble presets `MC` (NVT), `NptMC` and
  `GCMC` with their movers `TranslateMove`, `VolumeMove`,
  `InsertDeleteMove` and `MoveSelector`. Present when the `mc` module is
  built (`python -m atomiverse config md mc …`).
- **Plane-wave DFT** — `PW` / `PWResult`, i.e. energy,
  forces, cell stress and plane-wave Born-Oppenheimer MD (present when the
  `pw` module is built).
- **Gaussian SCF** — the concrete mean-field methods `RHF`, `UHF`, `RKS`,
  `UKS` for molecular Hartree-Fock and Kohn-Sham DFT (present when the `hf`
  module is built).

GPU minimization fails loud and remains deferred.

## Quickstart: Argon MD

```python
import numpy as np
import molpy
import atomiverse as av

# 1. Build an FCC lattice of 500 Ar atoms as a molpy.Frame
n_cells = 5
a0 = 5.26        # lattice constant [Å]
basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
positions = [
    ((ix + bx) * a0, (iy + by) * a0, (iz + bz) * a0)
    for ix in range(n_cells) for iy in range(n_cells) for iz in range(n_cells)
    for bx, by, bz in basis
]
n = len(positions)
xs, ys, zs = (np.array(c, dtype=np.float64) for c in zip(*positions))

frame = molpy.Frame()
atoms = molpy.Block()
atoms.insert("x", xs)
atoms.insert("y", ys)
atoms.insert("z", zs)
atoms.insert("mass", np.full(n, 39.948))   # Ar [amu]
atoms.insert("atomic_number", np.full(n, 18, dtype=np.uint32))
frame["atoms"] = atoms
frame.box = molpy.Box.cube(n_cells * a0)

# 2. Configure the MD driver (categorical setters — same as C++)
epsilon, sigma, cutoff = 0.0104, 3.4, 8.5   # LJ-Ar [eV, Å, Å]

sim = av.MD()
sim.set_integrator(av.VelocityVerlet(0.001))   # timestep [fs]
sim.set_observer(av.InitVelocity(temperature=10.0, seed=42))
sim.set_neighbor(av.NeighborList(cutoff))
sim.set_evaluator(av.PairLJCut(epsilon, sigma, cutoff))
sim.set_thermo(100)

# 3. Run (build() is called automatically)
sim.run(frame, 1000)

# 4. Read results back from the SAME molpy.Frame
x = np.asarray(frame["atoms"]["x"])
pe = frame.meta["energy"]                   # float [eV]
ke = frame.meta["kinetic"]                  # float [eV]
print(f"PE = {pe:.4f} eV,  KE = {ke:.4f} eV,  x[0] = {x[0]:.3f} Å")
```

## Namespaces: canonical deep paths + top-level convenience

The deep paths are canonical and mirror the C++ backend directory layout;
the top level re-exports the SAME classes under LAMMPS-style compound
names (`av.PairLJCut is av.potentials.pair.LJCut`). The quickstart above
uses the convenience names; the same driver in canonical style:

```python
sim = av.md.MD()
sim.set_integrator(av.md.integrator.VelocityVerlet(0.001))
sim.set_observer(av.md.InitVelocity(temperature=10.0, seed=42))
sim.set_neighbor(av.neighbor.NeighborList(cutoff))
sim.set_evaluator(av.potentials.pair.LJCut(epsilon, sigma, cutoff))
sim.set_thermo(100)
```

The full mapping table lives in `help(atomiverse)`.

Backend selection is a `device=` constructor parameter (PyTorch-style),
never a namespace: `av.md.MD(dt=0.001, device="cpu")` is the default;
`device="cuda"` raises `ValueError` until the CUDA bindings land.

## Charged systems: Coulomb and PME

Charges are **Frame data**: add a per-atom `charge` column [e] to the
`atoms` block, the same way you added `x`, `mass` and `atomic_number`. On an
existing frame the block is a live view, so this writes through:

```python
frame["atoms"].insert("charge", np.array(q, dtype=np.float64))   # [e]
```

A run that registers a Coulomb evaluator on a frame without that column
raises instead of reading every charge as zero. That asymmetry is
deliberate: a forgotten charge column and an intentionally neutral system
look identical once the frame reaches C++, and the forgotten one produces
finite, plausible, silently wrong numbers. If you really do mean zero,
insert an explicit zero column and the check passes.

Two electrostatic models, and they do not mix:

```python
# 1. Truncated Coulomb — self-contained, no reciprocal partner.
sim.set_evaluator(av.PairCoulCut(cutoff))

# 2. Ewald/PME split — BOTH halves, at the SAME alpha.
alpha = 3.5 / cutoff        # [1/Å]; see below
sim.set_evaluator(av.CoulLong(alpha, cutoff))
sim.set_evaluator(av.KSpacePME(alpha, grid_spacing=0.5, order=6))
```

Both models read the neighbor list, so both need
`sim.set_neighbor(av.NeighborList(cutoff))` with a neighbor cutoff at least
as large as the Coulomb cutoff — nothing checks this for you.

**Why a split at all.** In a periodic cell the plain `q_i q_j / r` sum
converges only conditionally: `1/r` decays too slowly to truncate. An
*Ewald split* rewrites it as a screened short-range part,
`erfc(alpha·r) q_i q_j / r`, which *does* decay fast enough to cut off, plus
a smooth remainder that converges in a few Fourier components. `alpha`
[1/Å] sets where that boundary falls: larger `alpha` screens the real-space
term harder and moves work into the reciprocal sum. The LAMMPS rule of
thumb `alpha = 3.5 / cutoff` puts the screened pair term at
`erfc(3.5) ≈ 7e-7` of the bare one by the time it reaches the cutoff.

The split is a matched pair: the driver requires exactly one real-space and
one reciprocal-space participant, so installing `CoulLong` or `KSpacePME`
alone raises at `build()`. On the reciprocal side, `KSpacePME` is *smooth
Particle-Mesh Ewald* — charges are spread onto a mesh, the mesh is
Fourier-transformed, multiplied by a kernel fixed by the cell, `alpha` and
`order`, and transformed back. `grid_spacing` [Å] sets how fine that
mesh is (smaller = more accurate and costlier) and `order` is the B-spline
interpolation order — the width in mesh points of the stencil each charge
is spread over (≥ 3, default 6).

Deep paths: `av.potentials.pair.CoulCut`, `av.longrange.CoulLong`,
`av.longrange.KSpacePME`.

**How this is checked.** `python/tests/test_md.py` builds a rock-salt NaCl
cell (64 ions, `charge` = ±1 e, nearest-neighbour spacing d = 2.8201 Å,
cutoff 5.0 Å) and compares the PME energy against the analytic *Madelung*
lattice energy — the exact electrostatic energy of an infinite ionic
lattice, `E = −½·N·M·k_e/d`, where `N` is the ion count, `M = 1.7475646` is
the rock-salt Madelung constant (a purely geometric lattice sum),
`k_e = 14.399645 eV·Å/e²` is the Coulomb constant in these units, and `d` is
the nearest-neighbour spacing. The Python gate is 5e-4 relative, which is
the PME discretization budget at 0.5 Å / order 6; the equivalent C++ case
`npt_pme_nacl` measures 3.8e-8 on the same settings.

## Force fields from molpy

molpy composes a force field from LAMMPS-standard styles
(`ff.def_style(category, name, params)` plus that style's types) and types a
molecule against it — GAFF / GAFF2, OPLS-AA, MMFF or a hand-built one.
`atomiverse.forcefield.compile(forcefield, frame)` translates it **style by
style, one molpy style to one Atomiverse evaluator**, with no knowledge of
which force field it is reading:

| molpy style | Atomiverse evaluator |
|---|---|
| `pair lj/cut` | `PairLJCut.table` — per-type rows, cross pairs by the style's `mixing` (arithmetic / geometric / sixthpower), LJ weights from `ff.special_bonds`, the style's `cutoff` and `shift` |
| `pair coul/cut` | `PairCoulCut` — Coulomb weights, the style's `cutoff`, its `coulomb` constant over `dielectric` |
| `bond harmonic`, `angle harmonic` | `BondHarmonic`, `AngleHarmonic` |
| `dihedral periodic` / `charmm` | `DihedralCharmm` (one row per nonzero periodic term) |
| `improper cvff` / `periodic` / `harmonic` | `ImproperCvff` (periodic converted exactly by molpy) / `ImproperHarmonic` |

Parameters convert by the dimensions molpy's style registry declares
(`E`, `L`, `Q`, `A`) from the force field's `units` (`real` or `metal`) to
eV / Å / e / rad. Any other style — including `coul/long` / `coul/long/pme`,
whose Coulomb weights and constant `CoulLong` cannot take — is refused by
name, never approximated. The result is a `Compiled(frame, evaluators,
cutoff)`: the frame the driver runs on (the force-field atom type as integer
`atoms.type_id`), the evaluators in style order, and the largest pair cutoff
for the `NeighborList`.

There is no vacuum mode: a gas-phase calculation of one molecule is
composition. Set the pair styles' cutoff beyond the molecule and make the box
large enough that no periodic image comes within it — every pair is then
counted once with bare Coulomb, the Amber `ntb=0, cut=999` model:

```python
import numpy as np
import molpy as mp
import atomiverse as av
from atomiverse import forcefield as avff

mol = mp.io.smiles.read_smiles("C[N+](C)(C)CCBr").to_atomistic()
mol, _ = mp.conformer.Conformer(seed=1).generate(mol)
typed = mp.ff.typifier.AtdTypifier(parameter_set="gaff2").typify(mol)
gaff = mp.ff.typifier.GaffTypifier(parameter_set="gaff2")
typed = gaff.typify(typed)
ff = gaff.forcefield()                       # pair lj/cut + pair coul/cut, ...

frame = typed.to_frame()
charges = av.AM1BCCCharges(method=av.AM1(charge=1)).assign(mol)
frame["atoms"].insert("charge", charges)

xyz = np.stack([np.asarray(frame["atoms"][k]) for k in "xyz"], axis=1)
diameter = max(np.linalg.norm(xyz - p, axis=1).max() for p in xyz)
rc, skin = diameter + 5.0, 2.0
for name in ("lj/cut", "coul/cut"):
    ff.get_style("pair", name).set(cutoff=rc)
box = diameter + rc + 2 * skin
for i, k in enumerate("xyz"):
    frame["atoms"][k] = xyz[:, i] - xyz[:, i].mean() + box / 2
frame.box = mp.Box.cube(box)

system = avff.compile(ff, frame)
opt = av.Minimize()
opt.set_optimizer(av.LBFGS())
opt.add_criterion(av.ForceConverged(ftol=1e-4))
opt.set_neighbor(av.NeighborList(system.cutoff, skin))
for term in system.evaluators:
    opt.set_evaluator(term)
opt.run(system.frame, 2000)
```

For a periodic system keep the force field's own cutoffs (GAFF: 10 Å) — or,
for Ewald electrostatics, replace its `coul/cut` style and compose `CoulLong` +
`KSpacePME` / `KSpacePPPM` yourself; that is a different model, not a
translation. `compile` refuses a frame without a box, a type a style leaves
unparameterized, or a style parameter the kernel cannot honour (LJ exponents
other than 12-6, a Coulomb `delta`, a CHARMM dihedral weight `w`). The frame's
atom kind is the force-field type, so the XYZ recorder, which spells a kind as
an element, does not apply to it; write structures with molpy's writers from
the typed frame.

## AM1 semiempirical single point, forces and BOMD

This section uses **CPU AM1**. C++ also has CPU RM1 and a restricted CUDA
AM1/RM1 energy/charge/analytic-force implementation with short MD and
steepest-descent composition checks. Neither RM1 nor CUDA NDDO has matching
Python bindings yet; these C++ checks do not establish Python CUDA support. `device="cuda"` does not turn `av.AM1` into the native
CUDA method. CPU NDDO is molecular and closed shell; non-singlet requests
are rejected even though the constructor accepts a `multiplicity` argument.

**AM1** — *Austin Model 1* — is a **semiempirical** electronic-structure
method: it solves an approximate quantum model of the valence electrons in
which most of the expensive electron-electron integrals are dropped (the NDDO
approximation, *neglect of diatomic differential overlap*) and the survivors
are replaced by numbers fitted once per chemical element. It costs far less
than the Gaussian-basis Hartree-Fock of the next section and far more than a
classical force field, and it is only as good as its fit.

`AM1` computes energy [eV], Mulliken charges [e] — per-atom partial
charges obtained by splitting each electron's population among the orbitals it
occupies — and nuclear forces [eV/Å]. The `atomic_number` column carries the
atomic number Z, which is how the method knows which element parameters to
use:

```python
import numpy as np
import atomiverse as av

# `frame` is a molpy.Frame with x/y/z + atomic_number columns.
am1 = av.AM1(charge=0)
av.single_point(frame, am1)                    # overwrite energy + charges; preserve forces
q = np.asarray(frame["atoms"]["charge"])       # Mulliken charges [e]
res = am1.last_result()                        # Hartree / kcal-mol detail
print(res.heat_of_formation_kcal_mol, res.iterations)
```

`forces(frame)` runs the same solve and *returns* the forces (negative energy
gradient) as an
`(n_atoms, 3)` array [eV/Å], writing the same numbers into the frame's
fx/fy/fz columns:

```python
f = am1.forces(frame)         # (n_atoms, 3) [eV/Å]
print(f.sum(axis=0))          # near 0 from equal-and-opposite pair contributions
```

CPU AM1 and RM1 compute **analytic forces** by propagating three Cartesian
derivatives through each pair's one-electron, two-electron and core-repulsion
expressions, then contracting with the converged density. No displaced
geometries are needed for forces; the model parameters are unchanged.

The SCF solve must satisfy both `density_tol` and `energy_tol`. At electronic
stationarity the density-response term vanishes, and the NDDO orthonormal
basis has no overlap Pulay term. Incomplete SCF convergence still limits force
accuracy. Equal-and-opposite pair contributions make the net force vanish to
round-off; this alone does not validate individual force components. The
existing AM1/RM1 C++ regression independently checks all force components
against executed OpenMOPAC 23.2.5 H₂O, CO and NH₃ fixtures for both methods.
See the [semiempirical roadmap](../roadmap/semiempirical.md) for the exact
element, backend and evidence limits; passing the AM1 Python suite does not
establish RM1/CUDA Python support.

Before the SCF, CPU AM1/RM1/MNDO/PM3 (and `AM1BCCCharges`) refuse a geometry
in which two atoms are closer than 0.6 × the sum of their covalent radii
(C–C 0.912 Å, C–H 0.642 Å, H–H 0.372 Å; the shortest real bonds sit near 0.77
of the sum). The `ValueError` gives the number of offending pairs and the
closest ones by 0-based atom index, element, distance and limit. An SCF that
exhausts `max_iter` raises `RuntimeError` with the iteration count, the last
density and energy changes, and both tolerances.

### Born-Oppenheimer MD

The same evaluator in the MD evaluator slot makes the run **BOMD** —
Born-Oppenheimer molecular dynamics: the electrons are re-converged at every
geometry the nuclei reach, one AM1 SCF solve per step, and the nuclei are
integrated on the forces from it. No `NeighborList` is involved — AM1 is
molecular, not a pair style. The frame needs one column the single point above
did not: a per-atom `mass` [amu], which the integrator divides by.

```python
# one f64 mass per atom [amu] — a water frame shown
frame["atoms"].insert("mass", np.array([15.999, 1.008, 1.008]))

sim = av.MD()
sim.set_integrator(av.VelocityVerlet(0.25))    # [fs] — light atoms, short dt
sim.set_observer(av.InitVelocity(300.0, 42))   # [K], seed
sim.set_evaluator(av.AM1(charge=0))
sim.run(frame, 100)

print(frame.meta["energy"], frame.meta["kinetic"])   # [eV]
```

The Python driver constructs its own method from the supplied parameters.
It retains that method's SCF workspace and accepted density during the run;
initial-force preparation and trial evaluations do not advance the accepted
electronic guess. The standalone `am1.last_result()` object is not updated by
`sim.run()`; per-step energy, charges and forces land on the Frame. This
Python ownership rule does not apply to a C++ `.add_ref(method)` mount.

## Gaussian SCF single point and forces

**SCF** stands for *self-consistent field*, and it names the loop these
methods run. Each is a **mean-field** model of the electrons: instead of
tracking every electron's repulsion of every other one, each electron is
taken to move in the *averaged* field of all the rest. That averaged field
is built from the orbitals it determines, so the equations cannot be solved
in one shot — you guess, build the field, diagonalize, rebuild, and repeat
until the energy stops moving. The orbitals themselves are expanded in a
fixed set of Gaussian functions, the **basis set**; a larger basis is more
accurate and more expensive.

Four methods, named by two choices. **R**estricted means the spin-up (α) and
spin-down (β) electrons share one set of spatial orbitals — a closed shell,
every occupied orbital doubly filled — while **U**nrestricted gives the two
spin channels their own orbitals, which is what a system with unpaired
electrons (a radical, an ion, triplet O₂) needs. **HF** is Hartree-Fock:
electron exchange exactly, electron correlation not at all. **KS** is
Kohn-Sham density-functional theory, which replaces both by an approximate
*exchange-correlation functional* — a recipe turning the electron density
into an energy — that you name with the `functional` argument.

The Python surface is the **concrete method** — `RHF`, `UHF`, `RKS`, `UKS`.
There is no `SCF` facade and no config bag: basis, charge, multiplicity,
tolerance and iteration cap are constructor arguments.

```python
import numpy as np
import atomiverse as av

# `frame` is a molpy.Frame carrying x/y/z + atomic_number — the worked
# example at the end of this section builds one from scratch.
rhf = av.RHF(basis="STO-3G", charge=0, tol=1e-9, max_iter=100)
rhf.run(frame)                        # frame.meta["energy"] is eV
print(rhf.energy(), rhf.converged(), rhf.iterations())   # Hartree / flag / count

forces = rhf.forces(frame)            # (N, 3) numpy array [eV/Å]
fz = np.asarray(frame["atoms"]["fz"]) # same values written back to the frame
```

The four constructors. `basis`, `charge`, the α/β counts and `xc` are fixed
at construction and read-only afterwards; `tol` / `max_iter` stay writable.

| Method | Signature |
| --- | --- |
| `RHF` | `RHF(basis, charge=0, tol=1e-9, max_iter=100)` |
| `UHF` | `UHF(basis, n_alpha, n_beta, charge=0, tol=1e-8, max_iter=100)` |
| `RKS` | `RKS(basis, xc, charge=0, tol=1e-9, max_iter=100)` |
| `UKS` | `UKS(basis, n_alpha, n_beta, xc="LSDA", charge=0, tol=1e-8, max_iter=100)` |

`basis` is one of `STO-3G`, `6-31G`, `6-31G*`, `6-31+G*`, `def2-SVP`,
`def2-TZVP`, `aug-cc-pVDZ`; `xc` is one of `LSDA`, `PBE`, `PBEsol`,
`RevPBE`, `RPBE`, `BLYP`, `OptB88VdW`, `B3LYP`, `PBE0`, `B2PLYP`. An
unsupported name raises `ValueError` listing the supported ones — there is
no fallback basis or functional.

**Units.** `charge` [e], `tol` [Hartree]. `energy()` is Hartree — the atomic
unit of energy the solver works in, 1 Ha = 27.211386 eV — while
`frame.meta["energy"]` is the same number in eV (the public Frame currency);
`forces()` is eV/Å and also writes `fx`/`fy`/`fz` back to the frame, the same
write-back contract as MD. `run()` computes **no** forces: after it, a
frame's `fx`/`fy`/`fz` are whatever they were on the way in (zeros for a
fresh frame), so call `forces()` when you want them.

**When is it converged?** `tol` is the threshold on |ΔE|, the change in total
energy between two successive iterations. `converged()` is `True` only when
that *and* a second measure are met: the Frobenius norm of the DIIS error
matrix below √`tol`. DIIS — *direct inversion of the iterative subspace* — is
the standard accelerator for this loop, and its error matrix is the natural
residual: it measures how far the current electron density is from being
self-consistent with the field it generates. Hitting `max_iter` first ends
the loop with `converged()` `False` rather than raising, so check the flag
before trusting the energy.

**Open shells.** For `UHF` / `UKS` the α/β split is resolved at `run` /
`forces` time from N_e = ΣZ − charge and the multiplicity M = 2S + 1 (S is
the total spin: M = 1 singlet, 2 doublet, 3 triplet), as
n_α = (N_e + M − 1)/2 and n_β = (N_e − M + 1)/2. Those are whole numbers
only for an even N_e with an odd M or an odd N_e with an even M, so a parity
mismatch (e.g. a doublet with an even electron count) raises `ValueError`
there rather than at construction.

Deep path: `av.scf.RHF`, … . These methods drive their own SCF loop, so they
are used directly; `single_point(frame, rhf)` accepts them and simply calls
`run`.

**Worked example — water, RHF/STO-3G.**

```python
import math
import numpy as np
import molpy
import atomiverse as av

r_oh, half = 0.95, 0.5 * math.radians(104.5)        # Å, half of ∠HOH
hx, hz = r_oh * math.sin(half), r_oh * math.cos(half)

frame = molpy.Frame()
atoms = molpy.Block()
atoms.insert("x", np.array([0.0, hx, -hx]))
atoms.insert("y", np.zeros(3))
atoms.insert("z", np.array([0.0, hz, hz]))
atoms.insert("mass", np.array([15.999, 1.008, 1.008]))
atoms.insert("atomic_number", np.array([8, 1, 1], dtype=np.uint32))
frame["atoms"] = atoms
frame.box = molpy.Box.cube(20.0)                    # gas phase

rhf = av.RHF("STO-3G")
rhf.run(frame)
assert rhf.converged()
print(rhf.energy())        # total energy [Hartree]
```

The printed energy reproduces **−74.961754 Ha**, the value
section `h2o_rhf_sto3g` of case `scf/rhf` pins at exactly this geometry (LibreTexts
*Live Textbook of Physical Chemistry* §28.2, cross-checked against PySCF
2.13), inside that section's stated **1 × 10⁻³ Ha** window. `python/tests/test_scf.py`
grades this same geometry against the same number and window.

## CPU geometry minimization

Minimize uses the same typed component composition as C++. The optimizer and
termination policy are separate objects; state stays in one `molpy.Frame` and
the return value carries only loop control metadata:

```python
opt = av.Minimize()
opt.add_criterion(av.ForceConverged(ftol=0.05, rms_ftol=0.03))
opt.set_optimizer(av.LBFGS(memory=5, alpha0=0.01, max_ls=10))
opt.set_neighbor(av.NeighborList(cutoff, 0.3))
opt.set_evaluator(av.PairLJCut(epsilon, sigma, cutoff))

result = opt.run(frame, max_iter=100)
print(result.iterations, result.converged, result.stop_reason)
```

Configuration is one-shot, exactly as for `MD`: the plan is compiled once, so
every setter raises `RuntimeError` after the first completed `run()`. Calling
`run()` again is still legal and re-optimizes from the frame's current state,
which is how you interleave analysis with chunks of optimization.

### Choosing an optimizer

The optimizer slot is singular — the three fixed-cell algorithms are
alternatives, not a stack:

| Optimizer | Shape | Use it when |
| --- | --- | --- |
| `LBFGS(memory, alpha0, max_ls)` | quasi-Newton with a line search | the default; fastest near a minimum |
| `CG(alpha0)` | Polak-Ribière⁺ conjugate gradient | low memory, or L-BFGS curvature keeps getting discarded |
| `FIRE(dt0, dt_max, …)` | damped dynamics, no line search | far from the minimum, or the force source is noisy |

When a CPU `LBFGS` or `CG` line search finds no lower energy within its trial
budget, the optimizer drops its curvature history (CG restarts) and searches
once more along steepest descent from the same geometry. If that fails too,
the run stops at the last accepted geometry with `converged == False` and
`stop_reason` `"lbfgs_line_search_failed"` / `"cg_line_search_failed"`; it no
longer raises. Only an evaluator error during a trial is an exception.

`memory` and `max_ls` are counts; `alpha0` is a step size in Å²/eV for `LBFGS`
and `CG` — the trial displacement per unit force — but a dimensionless
velocity-mixing fraction for `FIRE`, whose own step scale is `dt0` / `dt_max`
in fs.

Tolerances are **not** an optimizer argument. An optimizer publishes force /
displacement / energy metrics and a *criterion* judges them, which is what
makes the three interchangeable under one `ForceConverged`:

```python
opt.set_optimizer(av.CG())                          # or FIRE(dt0=1.0, dt_max=20.0)
opt.add_criterion(av.ForceConverged(ftol=0.01, rms_ftol=0.005))
```

`FIRE`'s defaults are ASE's and assume O(1) eV/Å forces. A soft system — an Ar
Lennard-Jones dimer has a residual of ~10⁻² eV/Å — needs a longer pseudo-time
scale, hence the `dt0=1.0, dt_max=20.0` retune above
(`fire_lj_dimer`).

`ForceConverged` has five tolerances and **all five** must hold at once: the
largest per-atom force `ftol` and the RMS force `rms_ftol` (both eV/Å), the
largest and RMS displacement of the last step `max_dtol` / `rms_dtol` (Å), and
the energy change `etol` (eV).

Any force source `MD.set_evaluator` accepts works here too — the two facades
read one dispatch table — so a bonded force field, an Ewald split
(`CoulLong` + `KSpacePME` at the same `alpha`), `AM1` or
`PW` all drive an optimization.

One iteration is free, whichever optimizer you choose: the minimizer phase
order runs `search_direction` before `force`, so iteration 0 has no force
evaluation of its own and steps from whatever the frame arrived with — zero on
a freshly built frame, i.e. no move at all. Budget the extra iteration rather
than reading a 1-iteration run as a no-op.

### Variable-cell relaxation

The four relaxers move the periodic **box** rather than the atoms within it,
reading the virial the evaluators wrote to the frame:

| Relaxer | Degrees of freedom |
| --- | --- |
| `IsotropicCellRelax(P_target, alpha0, max_scale, bulk_mod)` | one scale factor — volume only |
| `AnisotropicCellRelax(…, relax_x, relax_y, relax_z)` | three independent box edges, maskable per axis |
| `TriclinicCellRelax(alpha0, max_strain, bulk_mod)` | full symmetric strain, so the cell may shear |
| `JointIsotropicRelax(P_target, atom_step, …)` | volume **and** a steepest-descent atom step |

Match the criterion to the algorithm: a cell relaxer can sit at zero force
while far from the target pressure, so `ForceConverged` would declare victory
on an unrelaxed cell — and it fails quietly, because the relaxer publishes its
pressure residual into the very metric slot `ForceConverged` reads as a force.
Use `PressureConverged` (the isotropic mean) or, per axis, `StressConverged` —
and if you masked an axis on `AnisotropicCellRelax`, mask the same axis on
`StressConverged`, or that direction's reaction stress blocks convergence
forever. Both criteria read the same `stress` entry and divide by the same
volume; they differ only in judging the mean versus each diagonal component.

Two more gates the API cannot check for you.

**The evaluator must write the virial.** Every classical style does by
default — the pair styles (`PairLJCut`, `PairCoulCut`, `CoulLong` +
`KSpacePME`, `PairMorse`, `PairBuck`, `PairBorn`, `PairYukawa`, `PairWCA`,
`PairDpd`) and the bonded ones alike; `PairDpd` contributes its *conservative*
virial only, which is the term a pressure is built from. `AM1` writes
none — it is a molecular method with no cell — so it cannot drive a cell step.
A mounted `PW` computes its analytic virial whenever the plan consumes stress
(a cell relaxer or a `StressConverged` / `PressureConverged` criterion). With a
zero virial the cell never moves and `PressureConverged(P_target=0.0, …)`
reports success on the untouched cell. (`PW.stress(frame)` standalone returns
the plane-wave cell stress as a Voigt 6-vector in eV/Å³.)

**And the first iteration is free.** The step uses the stress the previous
iteration's force pass left behind, and iteration 0 has none of its own — it
reads whatever the frame arrived with, zero on a freshly built one, so at
`P_target = 0` its scale factor is exactly 1 and the cell starts moving on
iteration 1.

```python
opt = av.Minimize()
opt.set_optimizer(
    av.IsotropicCellRelax(P_target=0.0, alpha0=0.08, max_scale=0.08, bulk_mod=0.5)
)
opt.add_criterion(av.PressureConverged(P_target=0.0, ptol=5e-4))
opt.set_neighbor(av.NeighborList(cutoff, 0.3))
opt.set_evaluator(av.PairLJCut(epsilon, sigma, cutoff))

opt.run(frame, max_iter=1000)
length = float(np.asarray(frame.box.lengths)[0])       # relaxed cell edge [A]
stress = np.asarray(frame.meta["stress"])              # extensive Voigt virial [eV]
pressure = (stress[0] + stress[1] + stress[2]) / (3.0 * frame.box.volume())
```

Read the units carefully: `frame.meta["stress"]` is the **extensive** Voigt
virial in eV for every classical evaluator, which is why the pressure above
divides by the cell volume — `PressureConverged` uses the same formula.
Pressure targets are in eV/Å³, LAMMPS sign (positive is compressive);
1 bar ≈ 6.2415×10⁻⁷ eV/Å³.

Set expectations from what the C++ regression actually measures. Two Ar atoms
3.0 Å apart in a 6.0 Å cube — compressed well inside the Lennard-Jones
minimum — expand to L ≈ 6.32 Å with |P| ≈ 1.1×10⁻³ eV/Å³ after 200 iterations,
and reach |P| < 5×10⁻⁴ eV/Å³ after 559
(`cell_lj_pressure`). These are steepest-descent
skeletons with a soft modulus, not equation-of-state solvers: budget iterations
generously and check `result.converged` rather than assuming the budget was
enough.

Under `atomiverse-launcher` (a multi-rank MPI world hosting the interpreter)
the whole configuration is re-sent to the worker ranks as a task payload on
every `run()`, and that payload's vocabulary is narrower than what the facade
accepts: `PairLJCut`, the `NeighborList` and the seven optimizers. A run
configured with any other force source raises `RuntimeError` naming the
components it cannot carry, rather than optimizing with a term silently
missing. Single-process runs are unaffected.

Only the CPU backend is active for this surface. `device="cuda"` fails before
build or compute; GPU materialization is explicitly deferred.

## Vibrational analysis: frequencies and thermochemistry

Near a stationary point the energy surface is well approximated by a
quadratic form, and the matrix of its second derivatives with respect to the
nuclear Cartesian coordinates — the **Hessian** — is what turns a single
geometry into a spectrum. `Hessian` builds it by central finite differences:
it displaces every Cartesian coordinate by ±h (h = 1e-3 Å), differences the
forces the registered evaluator returns, symmetrizes, mass-weights, projects
out the translations and rotations and diagonalizes the rest. That is **6N
force evaluations** for N atoms, so the cost is the force source's.

Minimize first. The harmonic approximation only means anything at a
stationary point; on a non-stationary geometry the driver does not fail, it
silently returns the curvature of a slope, usually with spurious imaginary
modes.

The mode count is the **projected** one: `frequencies` has exactly 3N−6
entries (3N−5 for a linear molecule, and none at all for a lone atom), never
3N with five near-zero values to filter out. Imaginary modes — directions in
which the energy curves *down*, so the geometry is a saddle point — come back
as **negative** wavenumbers, the universal `imaginary_freq=False` chemistry
convention, and `n_imaginary` counts them.

Two Argon atoms at the Lennard-Jones minimum are the smallest system with a
closed-form answer, so it is worth running once to see the shape:

```python
import numpy as np
import molpy
import atomiverse as av

eps, sigma, cutoff = 0.0104, 3.4, 8.5        # [eV], [Å], [Å]
r_min = 2.0 ** (1.0 / 6.0) * sigma           # 3.81637 Å

frame = molpy.Frame()
atoms = molpy.Block()
atoms.insert("x", np.zeros(2))
atoms.insert("y", np.zeros(2))
atoms.insert("z", np.array([0.0, r_min]))
atoms.insert("atomic_number", np.full(2, 18, dtype=np.uint32))   # Ar
frame["atoms"] = atoms
frame.box = molpy.Box.cube(40.0)             # > 2 × cutoff: isolated dimer

driver = av.Hessian()                        # 298.15 K, 1 atm, singlet, σ = 1
driver.set_evaluator(av.PairLJCut(eps, sigma, cutoff))
driver.set_neighbor(av.NeighborList(cutoff, 1.0))

result = driver.run(frame)
print(result.frequencies)     # ≈ [26.4564] — one mode: 3N-5 = 1 for a diatomic
print(result.is_linear, result.n_imaginary)     # True 0
print(result.hessian.shape, result.normal_modes.shape)   # (6, 6) (6, 1)
print(result.zpe, result.g_thermal)             # [Hartree]
print(result.s_tot)                             # [Hartree/K]
```

`run()` also writes the frame back: the positions return to the equilibrium
geometry they arrived with, and the force columns hold the forces evaluated
there.

The one frequency is the analytic well curvature: with
`k = V''(r_min) = 72 ε / (2^(1/3) σ²)` and the reduced mass μ = m_Ar/2, the
stretch is `ν̃ = √(k/μ) / (2πc) = 26.456348 cm⁻¹`. The only error budget
between that closed form and the printed number is the central-difference
truncation at h = 1e-3 Å, which is 2.1 × 10⁻⁶ relative.
`python/tests/test_vib.py` derives the 26.456348 cm⁻¹ from the engine's own
CODATA-2014 constants by two independent unit paths (they agree to ~1e-16)
and grades the driver against it at 1e-3 relative — loose enough to also
cover a single-precision (`ATV_REAL_STORAGE=float`) wheel.

**Masses come from Z, not from the `mass` column.** The analysis reads each
atom's `atomic_number` [Z] and looks the IUPAC-2013 standard atomic weight up
per element, so editing a per-atom `mass` cannot move a frequency (a mass
mismatch shifts every wavenumber, ν ~ m^−1/2). The table covers **Z = 1..18**
(H..Ar) and has no fallback: anything else raises `ValueError` naming the
atomic number.

The force slot takes any bound source and has **no default** — a run with
nothing registered raises `RuntimeError`, because the Hessian of no potential
is not a physical zero. `PairLJCut` and the Coulomb styles need a
`NeighborList`; the O(N²) pair styles and the electronic-structure evaluators
do not:

```python
# `h2` is a molpy.Frame of two H atoms (atomic_number = 1) in a large cell.
driver = av.Hessian()
driver.set_evaluator(av.AM1())    # 6N SCF solves, no neighbor list
result = driver.run(h2)
```

Configure the driver fully before the first `run()`: the execution plan is
compiled exactly once, so every setter raises `RuntimeError` afterwards, and
a different configuration means a new `Hessian` object.

**File output is opt-in.** `frequency_tsv` and `mode_xyz_prefix` default to
the empty string, which writes nothing at all — pass a path to opt in:

```python
driver = av.Hessian(
    temperature=298.15,      # [K]
    pressure=101325.0,       # [Pa] — enters the translational entropy only
    multiplicity=1,          # 2S+1; contributes R ln(M) to the entropy
    symmetry_number=2,       # rotational σ (2 for H2O, 3 for NH3, 12 for C6H6)
    frequency_tsv="freq.tsv",
    mode_xyz_prefix="anim",  # writes anim_mode0.xyz, anim_mode1.xyz, …
)
```

The TSV is one row per mode (`index`, `frequency_cm-1`, `type`) with a `#`
trailer carrying the mode counts, the ZPE and the Gibbs terms; one XYZ
animation is written per **real** mode (imaginary modes are not animated),
named `<prefix>_mode<k>.xyz` after that mode's index in `frequencies`. Each
animation is one sinusoidal period in 20 frames with a 0.5 Å peak
displacement — C++ defaults, not exposed on the Python constructor.
Nothing derives `symmetry_number` for you — leaving it at 1 for a symmetric
molecule overestimates the rotational entropy by R ln σ, and so
under-estimates `g_tot`.

Units on the result: frequencies [cm⁻¹]; Hessian [eV/Å²]; `zpe`, `u_tot`,
`h_tot`, `g_tot`, `g_thermal` [Hartree]; `s_tot`, `s_trans`, `s_rot`,
`s_vib`, `s_elec` [Hartree/K]. The thermochemistry is in atomic units
because it is a faithful port of PySCF's `thermo()`; 1 Hartree = 27.211386 eV.

The electronic energy `E0` is carried as **zero** throughout that port, so
`u_tot`, `h_tot` and `g_tot` are the *thermal* parts of U, H and G — add
your own electronic energy for an absolute one — and `g_tot` and
`g_thermal` therefore hold the same number. `g_thermal` is the one to
compare against another code, because it does not require the two codes to
agree on an electronic energy first.

This is *molecular* vibrational analysis — free-molecule translations and
rotations are projected out. Periodic phonons are a different calculation and
are not on this surface, and neither are analytic (non-finite-difference)
Hessians.

## Monte Carlo: NVT, NPT and μVT

The MD driver above moves atoms by integrating Newton's equations. Monte
Carlo instead produces configurations by a guided random walk and never
computes a force. The walk is a *Markov chain* — each configuration is
proposed from the previous one and from nothing else — and one step of it
is a **trial**: a *mover* proposes a small random change, the registered
energy sources score the changed configuration, and the proposal is either
**accepted** (kept) or **rejected** (undone). Accepting with probability
`min(1, exp(-ΔE / k_BT))` — the *Metropolis* rule, `ΔE` [eV] being the trial
energy minus the last accepted energy — is what makes the visited
configurations Boltzmann-distributed, so a plain average along the chain is
an ensemble average. Downhill trials are always accepted; uphill ones are
accepted often enough that the chain can climb out of a local minimum.

An *ensemble* is the set of macroscopic quantities held fixed, and it is the
acceptance rule that decides which one you are sampling: `MC` is canonical
NVT (fixed particle number, volume, temperature), `NptMC` is
isothermal-isobaric NPT (fixed N, pressure, temperature — the volume is
sampled) and `GCMC` is grand-canonical μVT (fixed chemical potential μ,
volume, temperature — the particle number is sampled).

The handle takes the same categorical setters as MD, minus everything that
only makes sense with dynamics: `set_evaluator` (plural — energy sources are
summed), `set_neighbor`, and the MC-only `set_mover` (singular). There is no
integrator, no thermostat, no `set_thermo`/`set_record`, and no explicit
`build()`. Units are the package's own — temperature [K], pressure [eV/Å³]
(LAMMPS sign, positive = compressive), chemical potential [eV], mass [amu],
displacement [Å], volume [Å³], energy [eV].

```python
import atomiverse as av

# `frame` is the Argon molpy.Frame from the quickstart above:
# 500 atoms in a cubic cell of L = 5 × 5.26 Å = 26.3 Å.
cutoff, skin = 8.5, 1.0                            # [Å]
sim = av.MC(300.0, seed=42)                        # temperature [K]
sim.set_evaluator(av.PairLJCut(0.0104, 3.4, cutoff))
sim.set_neighbor(av.NeighborList(cutoff, skin))
sim.set_mover(av.TranslateMove(0.3))               # half-width [Å]

sim.run(frame, 20_000)                             # equilibrate
sim.run(frame, 100_000)                            # sample — SAME chain

print(sim.acceptance_ratio())    # aim for ~0.3-0.5; tune the half-width
print(sim.current_energy())      # [eV] == float(frame.meta["energy"])
```

`n_trials` counts **single trial moves**, not sweeps: each count is one
proposal and one accept/reject decision, so 500 atoms need ~500× more
trials here than they would "one attempted move per particle" sweeps.

Two sizing rules are yours to keep, because nothing checks them. The
neighbor cutoff must be at least the pair potential's cutoff, or pairs
inside the potential's range are never enumerated. And `cutoff + skin` must
stay **below half the shortest box edge**, since pair distances are taken
under the minimum-image convention — each pair is seen only through its
nearest periodic copy. The snippet above sits at 8.5 + 1.0 = 9.5 Å against
L/2 = 13.15 Å. Under `NptMC` the cell moves, so the margin has to hold at
the smallest volume the chain visits, not just at the starting cell.

After every `run()` the frame holds the current positions, the box and the
last accepted energy in `frame.meta["energy"]`. It does **not** hold forces:
Monte Carlo scores energies only and never assembles a gradient, so
`fx`/`fy`/`fz` come back zeroed. Velocities are left untouched — nothing
here moves in time.

Consecutive `run()` calls **continue one Markov chain**: the tallies
accumulate and the configuration carries over, which is what makes the
`run(equilibrate); run(sample)` pattern above mean what it says. The frame
is materialized once, on the first call — edits made to the `molpy.Frame`
between calls are ignored, and a fresh chain means a fresh sampler object.

The execution plan is compiled on that same first call, so a `set_*` call
afterwards could not affect the running chain; rather than silently doing
nothing it raises `RuntimeError`. Configure the sampler completely before
the first `run()`, and start a new sampler when you want a new setup.

Statistics are sampler accessors (there are no acceptance keys on the
frame). `attempts()`, `accepts()` and `acceptance_ratio()` aggregate over
every move type; pass a *verb* — the kind of move a trial turned out to be,
one of `"displace"`, `"box"`, `"insert"`, `"delete"` — for one of them. An
unknown name raises `ValueError`. The aggregate also counts trials that
ended with no verb at all (a deletion drawn on an empty cell, a volume draw
that would take V ≤ 0; both always rejected), so `attempts()` is exactly the
number of trials run while the four named verbs can sum to less.

`set_neighbor` takes the ordinary `NeighborList(cutoff, skin)` but builds it
in the MC-correct form (`every=1, delay=0, check=yes`): an accepted trial
move can cross the skin at any trial, so unlike MD the rebuild check is
never deferred.

### Mixed ensembles: one move per trial

A trial is one proposal, so a mixed ensemble goes through a `MoveSelector`
— it draws exactly one child per trial with the given relative weights.
Registering two movers side by side would fire both every trial, which is
not a Monte Carlo trial at all.

```python
# NPT-MC: displacements + isotropic volume moves, 50/50.
npt = av.NptMC(180.0, pressure=5.0e-5, seed=1337)   # P [eV/Å³]
npt.set_evaluator(av.PairLJCut(0.0104, 3.4, 6.8))
npt.set_neighbor(av.NeighborList(6.8, 1.0))
npt.set_mover(
    av.MoveSelector()
    .add(av.TranslateMove(1.0), 0.5)
    .add(av.VolumeMove(400.0), 0.5)                 # ΔV half-width [Å³]
)
npt.run(frame, 200_000)
print(npt.current_volume(), frame.box.volume())     # [Å³], the same cell
print(npt.acceptance_ratio("box"))                  # volume moves alone
```

```python
# μVT (GCMC): displacements + insertion/deletion of one species.
gcmc = av.GCMC(180.0, mu=-0.177, mass=39.948, seed=97)  # μ [eV], m [amu]
gcmc.set_evaluator(av.PairLJCut(0.0104, 3.4, 8.5))
gcmc.set_neighbor(av.NeighborList(8.5, 1.0))
gcmc.set_mover(
    av.MoveSelector()
    .add(av.TranslateMove(1.36), 0.5)
    .add(av.InsertDeleteMove(18, 39.948), 0.5)      # z = 18 (Ar), m [amu]
)

gcmc.run(frame, 50_000)                             # equilibrate
population = []
for _ in range(1000):
    gcmc.run(frame, 10)                             # decorrelate
    population.append(gcmc.n_atoms())
print(sum(population) / len(population))            # ⟨N⟩
```

**What μ means here.** `mu` is the *total* chemical potential — the
ideal-gas part included, not the excess part alone. It reaches the
acceptance rule only through the *activity* z = e^{βμ}/Λ³ [1/Å³], where
β = 1/k_BT and Λ = h/√(2πmk_BT) [Å] is the thermal de Broglie wavelength,
the quantum length scale of a free particle of mass m at temperature T. For
an ideal gas z *is* the number density, so μ = k_BT·ln(zΛ³) turns a target
density into the number you pass. The −0.177 eV above is exactly that:
zσ³ = 0.05 for argon (m = 39.948 amu, σ = 3.4 Å) at T = 180 K. This is the
convention of section `gcmc_lj` of case `mc/metropolis` (Frenkel & Smit
eq. 5.6.9), which is what its ⟨N⟩(z) pin measures.

**Match the mass.** The `InsertDeleteMove` mass **must** be the mass `GCMC`
built Λ from — two different masses mean the sampled chemical potential is
not the one you asked for, and nothing compares them.

Because N fluctuates, `run()` **rebuilds** the frame's `atoms` block
whenever the population changed: every column comes back at the new N,
including `mass` and `atomic_number`, which the fixed-N path treats as
immutable inputs and never writes.

Monte Carlo is single-rank by design — no MC task exists in the MPI
launcher protocol — so `run()` raises `RuntimeError` under
`atomiverse-launcher` instead of silently sampling one independent chain
per rank.

The quantitative ensemble physics is pinned in the C++ regressions, not
here: `npt_lj` grades ⟨V⟩ against the exact finite-N result
⟨V⟩ = (N+1)k_BT/P + N·B₂ (Frenkel & Smit, *Understanding Molecular
Simulation* 2nd ed. §5.4) — it lands 0.05 % off, inside a tolerance of
3 blocked standard errors plus 0.2 % for the virial truncation, and it also
grades the ensemble *width* σ_V = √(N+1)·k_BT/P — and
`gcmc_lj` grades ⟨N⟩(z) at two activities against the Mayer
expansion ρ = z − 2B₂z² (Hill, *An Introduction to Statistical
Thermodynamics* §15-3). B₂ is the second virial coefficient, the leading
correction to ideal-gas behaviour, evaluated by quadrature for exactly the
cut-and-shifted potential the engine samples. Read their pin notes
before choosing a state point.

## Key concepts

### molpy.Frame in, molpy.Frame out

The frame must carry an `atoms` block with `x, y, z, atomic_number` columns
(`mass` required for MD; `atomic_number` [uint] carries Z — the string
`type` column is a force-field label Atomiverse does not read). After
`run()` the frame holds updated `x/y/z`, `vx/vy/vz`, and `fx/fy/fz`. Global
results are exact-dtype typed metadata in `frame.meta`, read back as plain
Python values: `energy`, `kinetic`, and `conserved` are `float` (`f64` on
the wire); `stress` and `stress_kinetic` are 6-element sequences (`f64x6`).
There is no synthetic `scalars` or `stress` block. Missing required fields
fail loud with a `RuntimeError` naming the field.

### MD driver

Components register through **categorical setters**, mirroring the C++
`cpu::md::MD` / `Driver` API (same shape as `examples/cpp/08_md_lj.cpp`):

- `sim.set_integrator(VelocityVerlet(dt))`
- `sim.set_thermostat(NoseHoover(T, tau_T, dt))`
- `sim.set_barostat(BerendsenBarostat(P, tau_P, dt))` — rides alongside the
  integrator (see *Constant pressure: NPT* below)
- `sim.set_observer(InitVelocity(T, seed))`
- `sim.set_neighbor(NeighborList(cutoff, skin=-1))`
- `sim.set_evaluator(PairLJCut / BondHarmonic / …)` — plural energy sources
- `sim.set_thermo(N)` — LAMMPS-style print cadence
- `sim.set_record(XYZ(path, freq))` — trajectory output (LAMMPS `dump`):
  one XYZ frame every `freq` steps, element symbols taken from the frame's
  per-atom `atomic_number` column

Escape hatch for custom phase/order: `sim.on(phase).add(comp, order=0)`. It
covers exactly the six components listed under `_PHASE_DISPATCH` in
`atomiverse/md/__init__.py` — `PairLJCut`, `NeighborList`, `BondHarmonic`,
`AngleHarmonic`, `DihedralMultiHarmonic` and `Thermo` — and raises `TypeError`
naming that list for anything else, including every style of the widened
catalogue below. Those go through `set_evaluator`, which is where phase and
order are the driver's business anyway.

The `set_evaluator` slot is plural, so a force field is several calls. Only
`PairLJCut` and the electrostatic styles consume a `NeighborList` — every
other pair style in the table below walks all pairs O(N²) and needs none, so
size the system accordingly. A bonded style reads the matching topology
block of the frame (`bonds`, `angles`, `dihedrals`, `impropers`) and indexes
its parameters by that block's `type_id`.

Every pair style is truncated at a cutoff `r_c` [Å] — pairs farther apart are
skipped — and `r_c` is an explicit argument everywhere except `PairWCA`, which
derives it from `sigma`. "**Shifted at the cutoff**" in the table below means the
style subtracts the constant `U(r_c)` from every pair energy it does count, so
the energy reaches zero *continuously* at `r_c` rather than jumping there. A
constant has zero derivative, so this leaves the force untouched — the force
still steps at `r_c`, which over a long run is a slow source of energy drift.
`PairWCA` and `PairDpd` are the two that avoid even that, by construction
rather than by subtraction: WCA cuts exactly at the Lennard-Jones minimum where
the force is already zero, and DPD's weight `1 − r/r_c` vanishes there on its
own.

| Setter | Component | Description |
|--------|-----------|-------------|
| `set_integrator` | `VelocityVerlet(dt)` | Symplectic integrator |
| `set_integrator` | `NoseHooverNPT(T, tau_T, P, tau_P, dt, drag=0, mtchain=3, mpchain=3, couple_axis=(True,True,True), T_stop=-1)` | MTK NPT (`fix npt`) — replaces the plain integrator |
| `set_integrator` | `ParrinelloRahmanNPT(T, P, tau_T, tau_P, dt)` | Full-cell MTK NPT — cell tilts evolve; replaces the plain integrator (note the argument order) |
| `set_observer` | `InitVelocity(T, seed)` | Maxwell-Boltzmann init |
| `set_thermostat` | `NoseHoover(T, tau_T, dt)` | NVT thermostat |
| `set_thermostat` | `Langevin(T_start, T_stop, damp, dt, seed)` | Langevin thermostat |
| `set_thermostat` | `CSVR(T, tau, dt, seed)` | Bussi velocity rescale |
| `set_barostat` | `BerendsenBarostat(P, tau_P, dt, bulk_modulus=1.0)` | Berendsen isotropic barostat (`fix press/berendsen iso`) — installed *alongside* `VelocityVerlet` + a thermostat |
| `set_neighbor` | `NeighborList(cutoff, skin=-1)` | Cell-list pair builder |
| `set_evaluator` | `PairLJCut(ε, σ, cutoff, s12=1, s13=1, s14=1, energy_shift=True)` or `PairLJCut.table(type_i, type_j, ε, σ, cutoff, …)` | Lennard-Jones 12-6, single-type or per-type table (CPU) |
| `set_evaluator` | `PairCoulCut(cutoff, s12=1, s13=1, s14=1, coulomb=14.399645)` | Truncated Coulomb — complete on its own; needs a per-atom `charge` column [e] |
| `set_evaluator` | `CoulLong(alpha, cutoff)` | Ewald/PME **real-space** half; `alpha` [1/Å] must match its reciprocal partner |
| `set_evaluator` | `KSpacePME(alpha, grid_spacing=1.0, order=6)` | Ewald/PME **reciprocal-space** half; only on `ATV_BUILD_KSPACE=ON` wheels |
| `set_evaluator` | `PairMorse(D, alpha, r0, cutoff)` | Morse well `D(1−e^{−α(r−r₀)})²`, shifted at the cutoff. `D` [eV], `alpha` [1/Å], `r0` [Å], `cutoff` [Å] |
| `set_evaluator` | `PairBuck(A, rho, C, cutoff)` | Buckingham `A e^{−r/ρ} − C/r⁶`, shifted at the cutoff. `A` [eV], `rho` [Å], `C` [eV·Å⁶], `cutoff` [Å] |
| `set_evaluator` | `PairBorn(type_i, type_j, A, rho, sigma, C, D, cutoff)` | Born-Mayer-Huggins / Fumi-Tosi table, one entry per row across the parallel sequences; type keys are **atomic numbers**, unlisted pairs do not interact. `A` [eV], `rho` [Å], `sigma` [Å], `C` [eV·Å⁶], `D` [eV·Å⁸], `cutoff` [Å] |
| `set_evaluator` | `PairYukawa(A, kappa, cutoff)` | Screened Coulomb `A e^{−κr}/r` — reads no `charge` column; `A` [eV·Å] is the whole strength. `kappa` [1/Å], `cutoff` [Å] |
| `set_evaluator` | `PairWCA(epsilon, sigma)` | Purely repulsive LJ, cut and shifted at `2^(1/6)·σ` — the cutoff is **derived**, not an argument. `epsilon` [eV], `sigma` [Å] |
| `set_evaluator` | `PairDpd(a, gamma, temperature, cutoff, dt, seed=1)` | Groot-Warren DPD — a *thermostat* as well as a potential, so install it instead of one; stress carries the conservative virial only and the random force is redrawn every step. `a` [eV/Å] (a force), `gamma` [eV·fs/Å²], `temperature` [K], `cutoff` [Å], `dt` [fs] and must equal the integrator's |
| `set_evaluator` | `BondHarmonic(K, r0)` | Bond stretching `K(r − r₀)²`. `K` [eV/Å²], `r0` [Å], per bond type |
| `set_evaluator` | `BondFene(K, R0)` | Kremer-Grest FENE bond `−½KR₀²ln(1 − (r/R₀)²)`; `r ≥ R0` raises `RuntimeError` rather than reporting an infinite energy. `K` [eV/Å²], `R0` [Å], per bond type |
| `set_evaluator` | `AngleHarmonic(K, θ0)` | Angle bending `K(θ − θ₀)²`. `K` [eV/rad²], `θ0` [rad], per angle type |
| `set_evaluator` | `DihedralMultiHarmonic(Cn)` | Dihedral `Σₙ Cₙ cosⁿφ`. `Cn` [eV], one coefficient list per dihedral type |
| `set_evaluator` | `DihedralCharmm(K, n, delta)` | CHARMM periodic torsion `K(1 + cos(nφ − δ))`. `K` [eV], `n` an integer multiplicity ≥ 1 (minima per full turn), `delta` [rad], per dihedral type |
| `set_evaluator` | `ImproperHarmonic(K, chi0)` | Harmonic out-of-plane torsion `K(χ − χ₀)²`; reads the frame's `impropers` block. `K` [eV/rad²], `chi0` [rad], per improper type |
| `set_thermo` | `N` (int) | Thermo print every N steps |
| `Thermo(freq)` | end_of_step | Thermodynamic output |
| `set_record` | `XYZ(filename, freq=1)` | XYZ trajectory at end_of_step — one frame whenever `step % freq == 0`. Within a `run()` call `step` is zero-based (`0 … n_steps-1`), so 10 steps at `freq=5` write 2 frames (steps 0 and 5), never 3. A recorder truncates the file on its first write and appends afterwards, so give a second run its own `XYZ(...)` |

#### Constant pressure: NPT

An *ensemble* is the set of macroscopic quantities held fixed while the atoms
move. Bare `VelocityVerlet` holds particle number, cell volume and total
energy fixed (NVE). Adding a *thermostat* — a component that couples the atoms
to a notional heat bath so their mean kinetic energy tracks a target
temperature `T` [K] — gives NVT. NPT goes one step further: the simulation
cell is allowed to change size so that the internal pressure tracks a target
`P`, and the component that rescales the cell is the *barostat*. Pressure is
in eV/Å³ with the LAMMPS sign (positive ⇔ compressive), so **1 bar ≈
6.2415e-7 eV/Å³**; pass `0.0` for a zero-pressure run. Every route below
derives the pressure from the stress tensor, and the driver auto-installs the
virial (stress) observer that supplies it — you never register that yourself.

There are two routes, and they are mutually exclusive.

**Berendsen — a barostat beside the integrator.** Keep `VelocityVerlet` and a
thermostat, then add `set_barostat(BerendsenBarostat(P, tau_P, dt))`. Once per
step it rescales the cell edges and every atom position by one isotropic
factor that drives the instantaneous pressure toward `P` with relaxation time
`tau_P` [fs] (LAMMPS `Pdamp`). The optional `bulk_modulus` (β [eV/Å³], LAMMPS
`modulus`) divides that factor, so a larger β means a smaller per-step rescale
and slower relaxation. Its default is **1.0 eV/Å³**, deliberately *not*
LAMMPS's `10.0`; pass `10.0` explicitly for LAMMPS-identical relaxation.
Berendsen relaxes pressure smoothly and is a good equilibration tool, but it
does not sample a rigorous NPT distribution.

**MTK — an NPT integrator in place of the plain one.** `NoseHooverNPT` and
`ParrinelloRahmanNPT` are Martyna-Tobias-Klein integrators: the position and
velocity update, a Nose-Hoover thermostat and a barostat are fused into a
single time step. They therefore *replace* `VelocityVerlet` and go in through
`set_integrator` — do not also install a plain integrator, a separate
thermostat, or a barostat. `NoseHooverNPT` keeps the cell axis-aligned (only
the edge lengths move); `ParrinelloRahmanNPT` propagates the full cell matrix,
so the off-diagonal tilt factors move too — the choice when the *shape* of the
cell, not just its size, has to relax. Mind the argument order: `NoseHooverNPT`
pairs each setpoint with its own coupling time, `ParrinelloRahmanNPT` takes
both setpoints first. Keyword names are `T_target` / `P_target` (the table
above abbreviates them).

```python
# Rigorous NPT — one component covers integration, thermostat and barostat
sim.set_integrator(av.NoseHooverNPT(
    300.0,        # T_target [K]
    100.0,        # tau_T    [fs]
    6.2415e-7,    # P_target [eV/Å³] ≈ 1 bar
    1000.0,       # tau_P    [fs]
    0.5,          # dt       [fs]
))

# Full-cell NPT — same physics, tilt factors free.
# Argument order here is T, P, tau_T, tau_P, dt.
sim.set_integrator(av.ParrinelloRahmanNPT(300.0, 6.2415e-7, 100.0, 1000.0, 0.5))

# Berendsen equilibration — barostat rides alongside integrator + thermostat
sim.set_integrator(av.VelocityVerlet(0.5))
sim.set_thermostat(av.NoseHoover(300.0, 100.0, 0.5))
sim.set_barostat(av.BerendsenBarostat(6.2415e-7, 1000.0, 0.5))
```

`NoseHooverNPT` additionally exposes the LAMMPS `fix npt` knob tail, each
argument optional and already at its C++ default:

- `drag` — empirical extra damping on the Nose-Hoover chains; `0` disables it.
- `mtchain` / `mpchain` — length of the thermostat chain attached to the atoms
  and to the barostat. A *chain* is a cascade of fictitious thermostat
  variables, each one thermostatting the previous; 3 is the standard choice.
- `couple_axis` — per-axis barostat enable `(x, y, z)`. Setting
  `couple_axis[i] = False` freezes cell axis `i`: that edge never rescales,
  e.g. `(False, False, True)` lets only the z edge relax (uniaxial coupling).
- `T_stop` — end temperature [K] of a LAMMPS-style linear temperature ramp.
  With `T_stop >= 0` the setpoint moves linearly from `T` to `T_stop` across
  the span of the run; the span is wired automatically for each `run()` call,
  so every `run(frame, n_steps)` performs one complete ramp over its own
  steps. The default `T_stop = -1` (any negative value) folds back to `T`,
  i.e. no ramp.

### Units

| Quantity | Unit |
|----------|------|
| Distance | Å (Angstrom) |
| Energy | eV |
| Time | fs |
| Mass | amu |
| Temperature | K |
| Force | eV/Å |

## Plane-wave DFT (`pw` module)

```python
pw = av.PW(potential=av.GthPseudopotential(14, "GTH-PBE-q4"),   # analytic GTH for Si; no external file
           ecut_ry=30.0, k_mesh=4)
pw.build(frame)
result = pw.run(frame)
print(result.energy)      # [Ry]
print(result.converged, result.iterations)
```

A plane-wave basis *is* the set of plane waves commensurate with the
simulation cell, so `frame.box` is not optional: every entry point
(`build`, `run`, `forces`, `stress`) raises `RuntimeError` naming the box
rather than iterating to a NaN energy.

`result` also carries `energy_ha` [Hartree], `magnetization` [electrons],
`e_fermi` [Ry], `n_elec_occ`, `e_entropy` [Ry] and `n_kpts`. The last four
are the metallic-smearing channels — with fixed occupations `e_fermi`,
`n_elec_occ` and `e_entropy` stay 0, because that path runs no Fermi-level
search and fractional occupations are what carry an electronic entropy.

### Forces and cell stress

```python
forces = pw.forces(frame)   # (n_atoms, 3) [eV/Å]; also writes fx/fy/fz
print(frame.meta["energy"])  # [eV] — the SCF that produced those forces
stress = pw.stress(frame)   # Voigt xx, yy, zz, xy, xz, yz [eV/Å³]
```

`forces` are analytic — local Hellmann-Feynman + Ewald ion-ion +
Kleinman-Bylander nonlocal, evaluated at the converged density rather than
by differencing energies — and are written back to the frame's `fx`/`fy`/`fz`
columns along with that SCF's total energy in eV. `stress` is the
Nielsen-Martin cell derivative in the mechanical (CP2K) sign,
σ = −(1/V)·∂E/∂ε, so a cell compressed below its equilibrium lattice
constant returns a positive diagonal. Its returned vector is intensive
[eV/Å³], while `frame.meta["stress"]` stores the extensive virial
`stress * frame.box.volume()` [eV], excluding the kinetic contribution.
Each standalone derivative call runs its own reference SCF and updates
`last_result` with that call's convergence status and energy. Recorded
closed-form domains consume certified unstrained final bands; other domains
retain frozen-density NSCF strain differences. The current domain limits and
validation records are listed in the [DFT roadmap](../roadmap/dft.md).

Note the unit split: `PWResult` keeps the solver's Rydberg — 1 Ry = ½
Hartree, spelled 1 Ry = 13.605693 eV in the engine
(`cpu::pw::qe_units::rydberg_to_ev`, the Quantum ESPRESSO-rounded value) —
while Frame energies use eV, forces use eV/Å, and the Frame's extensive
virial uses eV. Divide that virial by the cell volume [Å³] to obtain stress
[eV/Å³]. Molecular SCF and GPW result energies use Hartree; these method
result conventions do not change the shared Frame units.

### Spin, functionals and metals

```python
pw = av.PW(potential=av.GthPseudopotential(14, "GTH-PBE-q4"), ecut_ry=30.0, k_mesh=4,
           xc="LSDA")       # name, not an enum; unknown names raise ValueError
pw.spin_polarized = True    # collinear; needed for an odd electron count
pw.multiplicity = 2         # fixed occupations: N↑ − N↓ = M − 1 exactly
pw.smearing = "gaussian"    # or "none" (default), "fermi_dirac",
pw.degauss = 0.01           #    "methfessel_paxton"; width [Ry]
pw.extra_bands = 8          # empty states above E_F for the μ search
```

`xc` takes the same functional names as `RKS(xc=...)`: `"LSDA"`,
`"PBE"` (default), `"PBEsol"`, `"RevPBE"`, `"RPBE"`, `"BLYP"`,
`"OptB88VdW"`, `"B3LYP"`, `"PBE0"`, `"B2PLYP"`. There is no fallback — an
unknown name raises `ValueError` listing them.

The default restricted path doubly occupies every band, so it needs an even
valence-electron count and refuses an odd one with `ValueError` (a single H
atom in a box is the standard case: set `spin_polarized = True` and
`multiplicity = 2`). Metallic smearing accepts an odd count instead, by
letting the occupations be fractional. The two are not yet combinable on the
derivative paths: `spin_polarized` together with a `smearing` gives a valid
`run`, but `forces` and `stress` raise `ValueError` — that analytic path is
not implemented.

A pinned example: Si₂ in the fcc primitive cell at *a* = 5.43 Å with one atom
displaced +0.08 Å along *x*, `GTH-PBE-q4`, `ecut_ry = 8`, Γ-only
(`k_mesh = 1`), `tol = 1e-6`, `max_iter = 40`, `xc = "LSDA"` gives
max|F| = 2.517979 eV/Å — the CPU number
section `cuda-pw-forces` of case `pw/rks` pins, where
it is the oracle for the CUDA twin.

### Plane-wave Born-Oppenheimer MD

```python
from atomiverse.md import MD
from atomiverse.md.integrator import VelocityVerlet
from atomiverse.pw import PW, GthPseudopotential

pw = PW(potential=GthPseudopotential(14, "GTH-PBE-q4"), ecut_ry=8.0, k_mesh=1)
pw.tol = 1e-4
pw.max_iter = 12            # one step is one whole SCF — budget it

sim = MD()
sim.set_integrator(VelocityVerlet(0.5))    # dt [fs]
sim.set_evaluator(pw)
sim.run(frame, 3)
print(frame.meta["energy"])  # [eV]
```

`PW` rides the same plural `set_evaluator` slot as the pair styles
and `AM1`; registering it makes the loop Born-Oppenheimer dynamics —
the electrons re-converged at every geometry the nuclei reach, with analytic
forces from the converged density. It takes no `NeighborList` (a plane-wave
Hamiltonian is not a pair style), needs the frame's `atomic_number` [Z] and
`mass` [amu] columns and a box, and the driver borrows the same method instance, keeping its Python owner
alive. Per-step results land on both the frame and the method's `last_result()`. Cost is the thing to
plan around: a step is *one* SCF solve (the analytic forces reuse the
converged density), seconds of work at production settings. A
Python-side run therefore wants a small cell, a low `ecut_ry` and a modest
`max_iter`.

### Immutable norm-conserving potential

| Member | What it takes |
|--------|---------------|
| `potential` | Immutable `GthPseudopotential(atomic_number, name)` or `UpfPseudopotential(path)`. GTH selects a published table for an explicit element; UPF loads its declared element and integral valence once. |

A provider is required. Every Frame atom must match its atomic number;
mixed-species frames are rejected for both built-ins. USPP, PAW, NLCC and
spin-orbit channels are rejected at admission. The CPU PW path uses one
NC operator implementation for built-in and independently authored C++
providers. A physical provider never owns an electronic Driver.

Without the `pw` module the PW types are not exported; rebuild with
`python -m atomiverse config pw …` to use this API.

## Timing

```python
sim.enable_timing()
sim.build()
sim.run(frame, 1000)
print(sim.summary())   # per-phase wall-clock breakdown
```

## Examples

`examples/python/` holds four scripts. They use the MolCrafts stack only —
molpy and numpy — never ASE or matplotlib.

The first three run on a `config md minimize am1` build, need no external program, finish in
seconds, and **exit nonzero** when their own sanity check fails, so they
can be smoke-run:

| Script | What it demonstrates |
|--------|----------------------|
| `md_lj_argon.py` | NVT molecular dynamics: 64 Lennard-Jones argon atoms, `VelocityVerlet` + `Langevin`, 200 steps; reads potential and kinetic energy back from `frame.meta`. Checks the energies are finite, the kinetic energy positive and the atoms moved. |
| `minimize_lj_dimer.py` | L-BFGS geometry optimization of an Ar dimer onto the analytic minimum `r* = 2^(1/6)·sigma`, with `ForceConverged` as the separate stop policy. Checks \|r − r\*\| < 0.01 Å and that the criterion — not the iteration budget — ended the run. |
| `am1_ch4.py` | AM1 semiempirical single point on methane: total energy [eV], heat of formation [kcal/mol] and Mulliken charges [e] written back to the frame's `charge` column. Checks the SCF converged, the charges sum to zero and carbon is negative. |

```bash
cd /path/to/Atomiverse
# Development tree; an installed wheel needs no PYTHONPATH at all.
export PYTHONPATH="$PWD/build-py:$PWD/python/src"

python examples/python/md_lj_argon.py
python examples/python/minimize_lj_dimer.py
python examples/python/am1_ch4.py
```

`gaff_parameterize.py` types ethanol with molpy's GAFF2 typifiers, assigns
AM1-BCC charges with `atomiverse.AM1BCCCharges` (needs the `semiempirical`
module) and writes `ethanol.mol2` / `ethanol.frcmod` to the working directory.

## Next steps

- **Benchmarks** — `python -m benchmarks` (see `benchmarks/README.md`).
- **Tests** — `python/tests/test_surface.py` documents the exact exported
  surface and its fail-loud contract.
