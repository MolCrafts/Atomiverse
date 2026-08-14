# Getting Started

## Installation

Atomiverse uses a **two-phase** Python install: a pure-Python package first,
then an optional on-demand native build.

### Phase 1 — pure Python package

```bash
pip install atomiverse
# or, from a source checkout (development):
#   pip install -e python/
```

This installs the Python API shell and CLI only — **no C++ compile**, no
prebuilt `_atomiverse` binary. Requires Python ≥ 3.9, NumPy, and
`molcrafts-molrs==0.13.1` (frame schema v2; other molrs releases are rejected).
This must match the vendored submodule pin in `.gitmodules` / `molrs/Cargo.toml`.

After phase 1 you can run:

```bash
python -m atomiverse          # shows version + install guidance
python -m atomiverse status  # no native build yet
```

Importing compute symbols (`atomiverse.MD`, …) without a native extension
raises `ImportError` pointing at phase 2.

### Phase 2 — native build (CMake stages)

The CLI mirrors the cmake pipeline and **streams raw cmake output**:

```bash
# 1) configure  (cmake -S -B -D…)
python -m atomiverse config md pw --mpi --src /path/to/Atomiverse

# 2) compile    (cmake --build)
python -m atomiverse compile -j 8

# 3) install    (cmake --install → site-packages)
python -m atomiverse install

python -m atomiverse status
```

**Modules** are positional on `config` only: `md`, `minimize`, `mc`, `evaluate`,
`vib`, `path`, `hf`, `pw`, `gpw`, `am1`, `cuda`, `kspace`, `potentials`,
`neighbor`.  
Enabling `md` / `mc` / `minimize` auto-enables `potentials` + `neighbor`.

**Global options** (on `config`, never under a module name):

| Flag | CMake |
|------|--------|
| `--mpi` / `--no-mpi` | `ATV_USE_MPI` |
| `--real-storage {float,double}` | `ATV_REAL_STORAGE` |
| `--real-work {float,double}` | `ATV_REAL_WORK` |
| `--id-size {bigbig,smallbig}` | `ATV_ID_SIZE` |
| `-D KEY=VALUE` | unbound defines only |

TOML recipe (XOR with positional modules; globals in `[global]` only):

```bash
python -m atomiverse config --config atomiverse-build.toml --src /path/to/Atomiverse
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

**Version lock:** checkout `python/pyproject.toml` must match the installed
package version. **`--src` is required** for the staged cmake pipeline.

**Dry-run:**

```bash
python -m atomiverse config md --mpi --src /path/to/Atomiverse --dry-run
python -m atomiverse compile --dry-run
python -m atomiverse install --dry-run
```

## What this package exposes (and what it doesn't)

Atomiverse binds **drivers and components only** — there is no Atomiverse
Frame type. Simulation state crosses the boundary as a **`molrs.Frame`**:
each `run()` borrows the frame, advances a transient execution frame in C++,
and writes positions / velocities / forces / energy back into the same
`molrs.Frame`.

Current surface:

- **MD** — the `MD` driver and its components under the canonical C++
  (LAMMPS-style) names: `VelocityVerlet`, `NoseHooverNPT`,
  `ParrinelloRahmanNPT`, `InitVelocity`, `NoseHoover`, `Langevin`, `CSVR`,
  `BerendsenBarostat`, `PairLJCut`, `NeighborList`, `BondHarmonic`,
  `AngleHarmonic`, `DihedralMultiHarmonic`, `Thermo`, `XYZ`.
- **Electrostatics** — `PairCoulCut` (truncated Coulomb) and the Ewald
  split `CoulLong` + `KSpacePME`. The reciprocal solver needs FFTW3, so
  `KSpacePME` is present only on wheels built with `ATV_BUILD_KSPACE=ON`
  (the default).
- **Geometry optimization** — CPU `minimizer.Minimize` with the `LBFGS`
  optimizer component and independent `ForceConverged` loop policy.
- **Plane-wave DFT** — `PW` / `PWResult` (PW-enabled wheels only; PW types
  are absent from wheels built without the backend).
- **Gaussian SCF** — the concrete mean-field methods `RHF`, `UHF`, `RKS`,
  `UKS` for molecular Hartree-Fock and Kohn-Sham DFT. Needs
  `ATV_BUILD_HF=ON`, which the shipped wheel sets by default; on a wheel
  built without it the four names are absent.

GPU minimization fails loud and remains deferred.

## Quickstart: Argon MD

```python
import numpy as np
import molrs
import atomiverse as av

# 1. Build an FCC lattice of 500 Ar atoms as a molrs.Frame
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

frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("x", xs)
atoms.insert("y", ys)
atoms.insert("z", zs)
atoms.insert("mass", np.full(n, 39.948))   # Ar [amu]
atoms.insert("atomic_number", np.full(n, 18, dtype=np.uint32))
frame["atoms"] = atoms
frame.box = molrs.Box.cube(n_cells * a0)

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

# 4. Read results back from the SAME molrs.Frame
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
`regressions/md/npt_pme_nacl` measures 3.8e-8 on the same settings.

## AM1 semiempirical single point, forces and BOMD

**AM1** — *Austin Model 1* — is a **semiempirical** electronic-structure
method: it solves an approximate quantum model of the valence electrons in
which most of the expensive electron-electron integrals are dropped (the NDDO
approximation, *neglect of diatomic differential overlap*) and the survivors
are replaced by numbers fitted once per chemical element. It costs far less
than the Gaussian-basis Hartree-Fock of the next section and far more than a
classical force field, and it is only as good as its fit.

`AM1Evaluator` computes energy [eV], Mulliken charges [e] — per-atom partial
charges obtained by splitting each electron's population among the orbitals it
occupies — and nuclear forces [eV/Å]. The `atomic_number` column carries the
atomic number Z, which is how the evaluator knows which element parameters to
use:

```python
import numpy as np
import atomiverse as av

# `frame` is a molrs.Frame with x/y/z + atomic_number columns.
ev = av.AM1Evaluator(charge=0)
av.single_point(frame, ev)                     # energy + charges onto frame
q = np.asarray(frame["atoms"]["charge"])       # Mulliken charges [e]
res = ev.last_result()                         # Hartree / kcal-mol detail
print(res.heat_of_formation_kcal_mol, res.iterations)
```

`forces(frame)` runs the same solve and *returns* the gradient as an
`(n_atoms, 3)` array [eV/Å], writing the same numbers into the frame's
fx/fy/fz columns:

```python
f = ev.forces(frame)          # (n_atoms, 3) [eV/Å]
print(f.sum(axis=0))          # exactly 0 — see below, this is not a check
```

The gradient is a **central finite difference of the energy at fixed
density**, with a step of `h = 1e-4` Å per Cartesian component. The SCF loop —
*self-consistent field*, the iteration that re-solves the electrons in the
averaged field they themselves generate, until the two agree — is run once;
then its converged electron density `P` is held while the energy is
re-evaluated at the displaced geometries and differenced.

Holding `P` fixed is not where the approximation lives. At convergence the
energy is stationary against any change of the density that keeps the electron
count, so the density-response part of `dE/dR` drops out; and NDDO treats its
orbital basis as orthonormal, so there is no basis-derivative ("Pulay") term to
recover either. What is approximate is the difference quotient itself, which
reproduces the exact NDDO gradient to `O(h²)` plus round-off. Closed-form NDDO
derivatives are a separate engine queue.

One consequence worth knowing before you use it as a sanity check: only
`3·(n_atoms − 1)` differences are actually taken, and the force on the last
atom is filled in as minus the sum of all the others — exact for an isolated
molecule, since a rigid translation cannot change `E`. So `f.sum(axis=0)` is
zero *by construction* and tells you nothing about the quality of the gradient.

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
sim.set_evaluator(av.AM1Evaluator(charge=0))
sim.run(frame, 100)

print(frame.meta["energy"], frame.meta["kinetic"])   # [eV]
```

The driver installs its own gradient-enabled copy of the evaluator's five SCF
parameters, so `ev.last_result()` is *not* updated by `run()`; the per-step
results (energy, charges, forces) land on the frame.

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

# `frame` is a molrs.Frame carrying x/y/z + atomic_number — the worked
# example at the end of this section builds one from scratch.
rhf = av.RHF(basis="STO-3G", charge=0, tol=1e-9, max_iter=100)
rhf.run(frame)                        # frame.meta["energy"] is eV
print(rhf.energy(), rhf.converged(), rhf.iterations())   # Hartree / flag / count

forces = rhf.forces(frame)            # (N, 3) numpy array [eV/Å]
fz = np.asarray(frame["atoms"]["fz"]) # same values written back to the frame
```

The four constructors and their defaults:

| Method | Signature |
| --- | --- |
| `RHF` | `RHF(basis="STO-3G", charge=0, tol=1e-9, max_iter=100)` |
| `UHF` | `UHF(basis="STO-3G", charge=0, multiplicity=1, tol=1e-8, max_iter=100)` |
| `RKS` | `RKS(functional="PBE", basis="STO-3G", charge=0, tol=1e-9, max_iter=100)` |
| `UKS` | `UKS(functional="LSDA", basis="STO-3G", charge=0, multiplicity=1, tol=1e-8, max_iter=100)` |

`basis` is one of `STO-3G`, `6-31G`, `6-31G*`, `6-31+G*`, `def2-SVP`,
`def2-TZVP`, `aug-cc-pVDZ`; `functional` is one of `LSDA`, `PBE`, `PBEsol`,
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
import molrs
import atomiverse as av

r_oh, half = 0.95, 0.5 * math.radians(104.5)        # Å, half of ∠HOH
hx, hz = r_oh * math.sin(half), r_oh * math.cos(half)

frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("x", np.array([0.0, hx, -hx]))
atoms.insert("y", np.zeros(3))
atoms.insert("z", np.array([0.0, hz, hz]))
atoms.insert("mass", np.array([15.999, 1.008, 1.008]))
atoms.insert("atomic_number", np.array([8, 1, 1], dtype=np.uint32))
frame["atoms"] = atoms
frame.box = molrs.Box.cube(20.0)                    # gas phase

rhf = av.RHF()                                      # STO-3G defaults
rhf.run(frame)
assert rhf.converged()
print(rhf.energy())        # total energy [Hartree]
```

The printed energy reproduces **−74.961754 Ha**, the value
`regressions/scf/h2o_rhf_sto3g` pins at exactly this geometry (LibreTexts
*Live Textbook of Physical Chemistry* §28.2, cross-checked against PySCF
2.13), inside that case's stated **1 × 10⁻³ Ha** window —
see `regressions/scf/h2o_rhf_sto3g/REFERENCE.md`. `python/tests/test_scf.py`
grades this same geometry against the same number and window.

## CPU geometry minimization

Minimize uses the same typed component composition as C++. The optimizer and
termination policy are separate objects; state stays in one `molrs.Frame` and
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

Only the CPU backend is active for this surface. `device="cuda"` fails before
build or compute; GPU materialization is explicitly deferred.

## Key concepts

### molrs.Frame in, molrs.Frame out

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
`cpu::md::MD` / `Driver` API (same shape as `examples/07_md_lj.cpp`):

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

Escape hatch for custom phase/order: `sim.on(phase).add(comp, order=0)`.

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
| `set_evaluator` | `PairLJCut(ε, σ, cutoff)` | Lennard-Jones 12-6 |
| `set_evaluator` | `PairCoulCut(cutoff)` | Truncated Coulomb — complete on its own; needs a per-atom `charge` column [e] |
| `set_evaluator` | `CoulLong(alpha, cutoff)` | Ewald/PME **real-space** half; `alpha` [1/Å] must match its reciprocal partner |
| `set_evaluator` | `KSpacePME(alpha, grid_spacing=1.0, order=6)` | Ewald/PME **reciprocal-space** half; only on `ATV_BUILD_KSPACE=ON` wheels |
| `set_evaluator` | `BondHarmonic(K, r0)` | Bond stretching |
| `set_evaluator` | `AngleHarmonic(K, θ0)` | Angle bending |
| `set_evaluator` | `DihedralMultiHarmonic(Cn)` | Dihedral potential |
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

## Plane-wave DFT (PW-enabled wheels)

```python
pw = av.PW()
pw.gth_name = "GTH-PBE-q4"   # analytic GTH for Si; no external file
pw.ecut_ry = 30.0
pw.k_mesh = 4
pw.build(frame)
result = pw.run(frame)
print(result.energy)      # [Ry]
print(result.converged, result.iterations)
```

### Pseudopotential: exactly one of two, and there is no default

| Member | What it takes |
|--------|---------------|
| `gth_name` | In-tree analytic Goedecker-Teter-Hutter parameterization, e.g. `"GTH-PBE-q4"` (Si), `"GTH-PBE-q1"` (H), `"GTH-PBE-q6"` (O). No file needed. In-tree coverage: H, O, Na, Cl, Ar, Si, Ni, Zn. One potential serves the whole cell, so the frame must be single-species. |
| `upf_path` | Path to a **norm-conserving** UPF file, e.g. the in-tree `tests/data/pw/Si_lda_nc.upf` (ONCVPSP). |

Setting both, or neither, raises. Ultrasoft (`pseudo_type="USPP"`) and PAW
files are rejected when read: the plane-wave path applies no augmentation
charge, so integrating one would be wrong by an unbounded amount. There is
no toy/bare-Coulomb fallback either — a bare valence-charge potential is not
representable in a plane-wave basis at any finite cutoff.

On a default wheel (`ATV_BUILD_PW=OFF`) the PW types are not exported;
rebuild with `ATV_BUILD_PW=ON` to use this API.

## Timing

```python
sim.enable_timing()
sim.build()
sim.run(frame, 1000)
print(sim.summary())   # per-phase wall-clock breakdown
```

## Next steps

- **Examples** — `python/examples/pw_qe_compare.py` (plane-wave DFT vs
  Quantum ESPRESSO for Si diamond). Install extras with
  `pip install -e './python[examples]'` (`molcrafts-molplot` + `molcrafts-molpy`;
  not matplotlib / ASE).
- **Benchmarks** — `python/benchmarks/bench_md.py` for system-size scaling.
- **Tests** — `python/tests/test_surface.py` documents the exact exported
  surface and its fail-loud contract.
