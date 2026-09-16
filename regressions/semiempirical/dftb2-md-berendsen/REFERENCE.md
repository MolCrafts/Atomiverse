# dftb2-md-berendsen — SCC-DFTB2 forces under a Berendsen thermostat

```text
oracle: literature
oracle_version: Berendsen 1984 (formula); DFTB+ not a trajectory twin
family_oracle: dftb+
```

## Why there is no DFTB+ pin here

Berendsen dynamics is not time-reversible and the two codes seed velocities
from different generators, so step-by-step agreement between ATV and DFTB+
does not exist to be pinned. The DFTB+-pinned quantities on this path are
energy, forces and stress (`dftb2-ewald-gamma`, `dftb2-analytic-stress`); what
this case gates is the **wiring and the control law**, which are internal
claims and are tested internally.

The thermostat's scaling factor itself is pinned against the published formula
in `tests/cpu/dynamics/test_berendsen.cpp` — that is a single-method unit test
and belongs there, not here.

## Method

Berendsen weak coupling (Berendsen, Postma, van Gunsteren, DiNola, Haak,
*J. Chem. Phys.* **81**, 3684 (1984), DOI
[10.1063/1.448118](https://doi.org/10.1063/1.448118)) — `cpu::md::Berendsen`,
the LAMMPS `fix temp/berendsen` / DFTB+ `Thermostat = Berendsen` control law —
on analytic SCC-DFTB2 forces, `boundary = lattice`, mio-1-1, with the ASPC
charge guess (`DFTB2::predictor(3)`) live so the MD path is exercised end to
end.

Until this case there was **no DFTB-in-MD regression at all**; the only
semiempirical-in-MD case was AM1.

## System

4 waters, cubic 5.6 Å, Γ. dt = 0.2 fs (the DFTB+ / Hooven protocol step),
τ = 20 fs, setpoint 300 K, Maxwell–Boltzmann start at **900 K**, 120 steps
(24 fs = 1.2 τ).

## Gates

The case runs the same trajectory **twice**, thermostat off and on. Asserting
"the temperature ended near 300 K" on its own would prove nothing: a 900 K
Maxwell–Boltzmann start equipartitions down on its own within tens of fs, so
the free trajectory has to be the control.

| Gate | Requirement | Observed |
| ---- | ----------- | -------- |
| Trajectory finite | E, T finite in both runs | ok |
| Thermostat beats the free run | \|T−300\| smaller with the bath | **381 K vs 597 K** |
| Reaches the setpoint window | \|T−300\| < 120 K | **81 K** |

The window is 120 K because 1.2 τ of ideal exponential decay still leaves ~30%
of the initial 600 K excess; it is set from the control law, not from what
happened to pass.

## Re-run

```bash
cmake --build <build> --target run_reg_dftb2-md-berendsen
```
