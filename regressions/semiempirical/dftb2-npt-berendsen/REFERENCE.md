# dftb2-npt-berendsen — the DFTB2 cell virial reaching the barostat

```text
oracle: literature
oracle_version: Berendsen 1984 (formula); internal differential gate
family_oracle: dftb+
```

## What this case exists to catch

No electronic evaluator declares `provides_system_stress()`. `md::Virial` is
auto-injected because `cpu::Barostat::requires_system_stress()` is true, and
the **electronic** contribution reaches the barostat only because `DFTB2`
accumulates onto the same `kStress` key at a compatible phase
(`core/frame_schema.h:281-297`). That is a live coupling with nothing holding
it: a build where the configurational virial stopped arriving would still run,
still be stable, and still look reasonable — the barostat would simply be
steering on the kinetic term, which is positive definite, so the cell would
only ever expand.

So the gate is **differential and signed**, not a tolerance on a number.

## Method

`cpu::md::Barostat` (Berendsen isotropic pressure coupling, Berendsen et al.,
*J. Chem. Phys.* **81**, 3684 (1984); LAMMPS `fix press/berendsen`) fed by the
analytic SCC-DFTB2 cell virial, plus `cpu::md::Berendsen` on the temperature.
`P_target = 1 bar` written as `1.0 * units::bar_to_ev_per_a3` — the named
conversion added with this case, rather than the hand-copied factor that used
to live in doc comments.

## System

4 waters in a **stretched** cubic 7.4 Å cell (≈0.15 g/cm³, V₀ = 405.22 Å³).
Stretched on purpose: there the kinetic virial is positive (expand) while the
configurational virial is in tension (contract), so the two runs must move the
cell in **opposite directions**.

dt = 0.2 fs, τ_T = 20 fs, τ_P = 20 fs, T = 300 K, 60 steps.
`bulk_modulus = 0.1 eV/Å³` — deliberately softer than the class default of 1.0,
because at the default this cell moves 0.01% in 60 steps and a sign test on a
signal that small is one rounding from meaningless. It is a test knob, not a
physical claim.

## Gates

| Gate | Requirement | Observed |
| ---- | ----------- | -------- |
| Cell finite and positive | — | ok |
| Split is real | (V_off − V_on)/V₀ > 0.002 | **0.0065** |
| Kinetic-only run expands | V_off > V₀ | **+1.43 Å³** |
| **With the DFTB2 virial the cell contracts** | V_on < V₀ | **−1.18 Å³** |

## Re-run

```bash
cmake --build <build> --target run_reg_dftb2-npt-berendsen
```
