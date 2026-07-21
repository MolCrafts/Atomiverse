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
`molcrafts-molrs==0.7.0` (frame schema v2; other molrs releases are rejected).

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
`vib`, `path`, `pw`, `gpw`, `am1`, `cuda`, `kspace`, `potentials`, `neighbor`.  
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

**`--hf` is rejected** until SCF Python bindings land.

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
  (LAMMPS-style) names: `VelocityVerlet`, `InitVelocity`, `NoseHoover`,
  `PairLJCut`, `NeighborList`, `BondHarmonic`, `AngleHarmonic`,
  `DihedralMultiHarmonic`, `Thermo`.
- **Geometry optimization** — CPU `minimizer.Minimize` with the `LBFGS`
  optimizer component and independent `ForceConverged` loop policy.
- **Plane-wave DFT** — `PWConfig` / `PWSolver` / `PWResult` (PW-enabled
  wheels only; PW types are absent from wheels built without the backend).

GPU minimization fails loud and remains deferred. Gaussian-SCF Python
bindings remain a later CPU breadth slice.

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
atoms.insert("type", np.zeros(n, dtype=np.int32))
frame["atoms"] = atoms
frame.simbox = molrs.Box.cube(n_cells * a0)

# 2. Configure the MD driver
epsilon, sigma, cutoff = 0.0104, 3.4, 8.5   # LJ-Ar [eV, Å, Å]

sim = av.MD(dt=0.001)                        # timestep [fs]
sim.add(av.InitVelocity(temperature=10.0, seed=42))
sim.on("force").add(av.NeighborList(cutoff), order=-10) \
               .add(av.PairLJCut(epsilon, sigma, cutoff))
sim.on("end_of_step").add(av.Thermo(100))

# 3. Run (build() is called automatically)
sim.run(frame, 1000)

# 4. Read results back from the SAME molrs.Frame
x = np.asarray(frame["atoms"]["x"])
energy = frame.meta["energy"]               # molrs.MetaValue(dtype="f64")
kinetic = frame.meta["kinetic"]
pe = float(energy.value)
ke = float(kinetic.value)
print(f"PE = {pe:.4f} eV,  KE = {ke:.4f} eV,  x[0] = {x[0]:.3f} Å")
```

## Namespaces: canonical deep paths + top-level convenience

The deep paths are canonical and mirror the C++ backend directory layout;
the top level re-exports the SAME classes under LAMMPS-style compound
names (`av.PairLJCut is av.potentials.pair.LJCut`). The quickstart above
uses the convenience names; the same driver in canonical style:

```python
sim = av.md.MD(dt=0.001)
sim.add(av.md.InitVelocity(temperature=10.0, seed=42))
sim.on("force").add(av.neighbor.NeighborList(cutoff), order=-10) \
               .add(av.potentials.pair.LJCut(epsilon, sigma, cutoff))
sim.on("end_of_step").add(av.recorder.Thermo(100))
```

The full mapping table lives in `help(atomiverse)`.

Backend selection is a `device=` constructor parameter (PyTorch-style),
never a namespace: `av.md.MD(dt=0.001, device="cpu")` is the default;
`device="cuda"` raises `ValueError` until the CUDA bindings land.

## AM1 semiempirical single point

`AM1Evaluator` computes energy [eV] and Mulliken charges [e] in one
`single_point` pass (`type` column = atomic number Z; no AM1 forces yet):

```python
ev = av.AM1Evaluator(av.AM1Config(charge=0))
av.single_point(frame, ev)                     # energy + charges onto frame
q = np.asarray(frame["atoms"]["charge"])       # Mulliken charges [e]
res = ev.last_result()                         # Hartree / kcal-mol detail
print(res.heat_of_formation_kcal_mol, res.iterations)
```

## CPU geometry minimization

Minimize uses the same typed component composition as C++. The optimizer and
termination policy are separate objects; state stays in one `molrs.Frame` and
the return value carries only loop control metadata:

```python
opt = av.Minimize(
    convergence=av.ForceConverged(ftol=0.05, rms_ftol=0.03)
)
opt.add(av.LBFGS(memory=5, alpha0=0.01, max_ls=10))
opt.on("force").add(av.NeighborList(cutoff, 0.3), order=-10) \
               .add(av.PairLJCut(epsilon, sigma, cutoff))

result = opt.run(frame, max_iter=100)
print(result.iterations, result.converged, result.stop_reason)
```

Only the CPU backend is active for this surface. `device="cuda"` fails before
build or compute; GPU materialization is explicitly deferred.

## Key concepts

### molrs.Frame in, molrs.Frame out

The frame must carry an `atoms` block with `x, y, z, type` columns (`mass`
required for MD). After `run()` the frame holds updated `x/y/z`, `vx/vy/vz`,
and `fx/fy/fz`. Global results are exact-dtype `molrs.MetaValue` entries in
`frame.meta`: `energy`, `kinetic`, and `conserved` are `f64`; `stress` and
`stress_kinetic` are `f64x6`. There is no synthetic `scalars` or `stress`
block. Read the payload through `.value` and inspect `.dtype` when the exact
wire type matters. Missing required fields fail loud with a `RuntimeError`
naming the field.

### MD driver

Components register through two paths, mirroring the C++ API:

- `sim.add(comp)` — components with default phase bindings
  (`VelocityVerlet`, `InitVelocity`, `NoseHoover`).
- `sim.on(phase).add(comp, order=0)` — explicit phase binding, where
  *phase* is one of `"initial_integrate"`, `"pre_force"`, `"force"`,
  `"post_force"`, `"final_integrate"`, `"end_of_step"`.

| Component | Phase | Description |
|-----------|-------|-------------|
| `VelocityVerlet(dt)` | initial/final integrate | Symplectic integrator |
| `InitVelocity(T, seed)` | build-time | Maxwell-Boltzmann init |
| `NoseHoover(T_target, tau_T, dt)` | initial/final integrate | NVT thermostat |
| `NeighborList(cutoff, skin=-1)` | force (order −10) | Cell-list pair builder |
| `PairLJCut(ε, σ, cutoff)` | force | Lennard-Jones 12-6 |
| `BondHarmonic(K, r0)` | force | Bond stretching |
| `AngleHarmonic(K, θ0)` | force | Angle bending |
| `DihedralMultiHarmonic(Cn)` | force | Dihedral potential |
| `Thermo(freq)` | end_of_step | Thermodynamic output |

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
cfg = av.PWConfig()
cfg.upf_path = "Si.upf"
cfg.ecut_ry = 30.0
cfg.k_mesh = 4
solver = av.PWSolver(cfg)
solver.build(frame)
result = solver.run(frame)
print(result.energy)      # [Ry]
print(result.converged, result.iterations)
```

An empty `upf_path` runs a toy local potential (useful as a smoke test;
results are not physical). On a default wheel (`ATV_BUILD_PW=OFF`) the PW
types are not exported; rebuild with `ATV_BUILD_PW=ON` to use this API.

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
