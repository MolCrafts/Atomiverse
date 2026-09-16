# Reference — REMD exchange acceptance on a harmonic dimer

## Physics

Four temperatures, two atoms joined by a harmonic bond, nested MD +
Metropolis exchange

```
P_acc = min(1, exp[(β_i − β_j)(U_i − U_j)])
```

(Hukushima & Nemoto, *J. Phys. Soc. Jpn.* **65**, 1604 (1996); Sugita &
Okamoto, *Chem. Phys. Lett.* **314**, 141 (1999)). On accept, configurations
swap and velocities rescale by √(T_i/T_j) so each slot keeps its thermostat
target.

That rule is the Metropolis test for swapping between two systems that are
each sampling the **canonical** ensemble at their own β. So the mean
acceptance is not a free parameter — it is an integral over the two
distributions, and for an analytic potential it can be evaluated exactly.

## What is checked

### The exchange acceptance, against its exact canonical value

The replica potential is one harmonic bond,

```
U(r) = ½ k (r − r₀)²,     k = 5 eV/Å²,  r₀ = 4 Å
```

so the canonical distribution of the bond length at temperature T is known in
closed form — including the spherical Jacobian of the relative coordinate,
which is why ⟨U⟩ is *not* exactly ½k_BT:

```
P_T(r) ∝ r² exp(−U(r)/k_BT)
```

The mean acceptance for one attempt between slots i and j is then

```
⟨P_acc⟩ = ∫∫ P_i(r) P_j(r') min(1, exp[(β_i − β_j)(U(r) − U(r'))]) dr dr'
```

which the case evaluates by quadrature — no fitted number, no snapshot of our
own output, the same "compute the exact reference in the case" pattern the
`npt_lj` B₂ quadrature uses.

Ladder: geometric with ratio 3 — **200 / 600 / 1800 / 5400 K**. Wide on
purpose: it puts the exact acceptance near ⅔ instead of against the ceiling,
so a criterion that accepts everything is 21 standard errors away rather than
two.

| Edge | exact ⟨P_acc⟩ |
|---|---|
| (0,1) 200↔600 K | 0.6664 |
| (1,2) 600↔1800 K | 0.6660 |
| (2,3) 1800↔5400 K | 0.6645 |
| **schedule mean** | **0.6656** |

The even/odd schedule attempts (0,1) and (2,3) on even rounds and (1,2) on odd
ones, so over `n_ex` rounds each edge is tried `n_ex/2` times and the expected
overall ratio is the mean of the three.

Measured over 600 rounds (900 attempts, 800 fs between rounds ≈ six spring
periods, so the attempts are effectively independent Bernoulli trials):

```
ratio = 0.6722    vs 0.6656 exact    (+0.42 binomial SE, tol 4 SE = 0.063)
```

### Structural checks (retained)

- The even/odd schedule produces exactly `(n_ex/2)·2 + (n_ex/2)·1 = 900`
  attempts for a 4-replica ladder over 600 rounds.
- Collapse returns a 2-atom Frame (lowest-T slot) whose bond length is a
  thermal sample of that slot (|r − r₀| ≤ 9σ(T₀)).
- Seed 7 is deterministic across two runs.

## Why the previous version of this case reported 1.000

`Remd` used to give each replica a `VelocityVerlet` and a one-shot
`InitVelocity` draw and **no thermostat** — every replica was a
constant-energy trajectory. Replica exchange is not defined over those: β_i is
then a label on a slot rather than a property of a distribution, and the
acceptance ratio measures nothing.

The old case made that concrete. Its replicas were an LJ dimer with
ε/k_B = 121 K started 4 Å apart at 100–250 K; with no thermostat the pair
slowly separated in every slot until each replica's potential energy was
whatever its own trajectory had reached, `(β_i − β_j)(U_i − U_j)` lost its
sign structure, and the exchange accepted **60 of 60** attempts.

`Remd` now attaches a CSVR thermostat (Bussi, Donadio, Parrinello, *J. Chem.
Phys.* **126**, 014101 (2007)) at `temperatures[i]` to replica i, with the
coupling time on the public member `thermostat_tau` [fs]. The unit-level pin
is `tests/cpu/path/test_remd.cpp`
(`Remd.ReplicasAreThermostattedAtTheirSlotTemperature`), which starts a block
compressed onto the LJ repulsive wall: a thermostatted slot comes back at
270 K against a 300 K setpoint, a constant-energy one at 862 K.

The exchange swap is also keyed on **tags**, not rows: each replica runs its
own MD program with its own neighbour list, `atom_modify sort` reorders owned
rows at its own cadence in each, and a row-wise swap would hand replica i one
atom's coordinates under another atom's tag.

## Deferred

Per-slot temperature-marginal histograms would need `Remd` to publish its
replicas; the acceptance integral above pins the same physics through the one
statistic the driver already exposes.
