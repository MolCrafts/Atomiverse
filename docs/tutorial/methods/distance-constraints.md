# SHAKE and RATTLE distance constraints

`cpu::md::FixShake` / `cuda::md::FixShake` enforce bond and angle 1–3
distances with iterative force-level SHAKE. `FixRattle` also projects
relative velocities onto the constraint tangent space after the final kick.
Each constraint removes one degree of freedom from temperature and thermostat
calculations. Both components publish the constraint virial, including all six
stress components, for pressure and NPT barostats.

```python
from atomiverse.md import FixShake, MD
from atomiverse.md.integrator import NoseHooverNPT

sim = MD(device="cuda", units="real")
sim.set_integrator(NoseHooverNPT(1000, 100, 1, 1000, 1.0))
constraint = FixShake(1.0, tolerance=1e-6, max_iter=1000)
constraint.constrain_bond(1, 1.0969)  # the bond type value the Frame stores
sim.set_observer(constraint)
# Install the force field and neighbor list, then build and run on your frame.
```

Use `FixRattle` with the same arguments for position and velocity constraints.
C++ uses the same calls, e.g. `cpu::md::FixRattle{dt, tolerance, max_iter}`.
The driver supplies its integrator timestep through `TimestepConsumer`.
To hold an angle, call `constrain_angle(type, distance_13)`; supply the 1–3
length rather than an angle in degrees. The type is the value the Frame's
bond/angle type column holds — the same index the bonded potential tables
use. Atomiverse's LAMMPS data reader stores zero-based types (LAMMPS bond type
2 becomes 1); a molpy Frame passes its `type_id` through unchanged, so a
1-based molpy `type_id` 2 is constrained with `constrain_bond(2, ...)`.

SHAKE predicts the next unconstrained positions from the current coordinates,
half-step velocities and physical forces. A cluster solve supplies the reaction
forces that make the next drift satisfy the prescribed distances. The forces
are added before the final Verlet half-kick; their virial uses the current bond
vectors and `1/(dt²*ftm2v)`. Initialization uses a full-step velocity and the
corresponding `dt²/2` predictor. A separate coordinate projection handles box
dilation. This is a force-level SHAKE algorithm, without RATTLE's final
velocity projection.

RATTLE accumulates position-projection multipliers before force evaluation
clears the force/stress buffers, then tallies them in `post_force` with
`2/(dt²*ftm2v)` for its half-step impulse. At initialization a coupled solve of
`r·a_relative + v_relative² = 0` supplies the reaction virial without changing
physical forces. CPU MPI counts each redundant cross-rank pair once through
endpoint ownership. CUDA performs one solve per disjoint cluster and supports
graph replay. Constraint failures stop the run rather than returning an
unconstrained trajectory. Lengths, times and pressures follow the driver
unit style. Tolerances below the storage/integration precision floor are
rejected.

These components implement MD distance constraints. CUDA runs on one GPU;
multi-GPU constraint communication and minimization constraints are not
implemented. Statistical comparisons still require matching timestep,
constraints, degrees of freedom, thermostat and barostat parameters.
