# Reference — SETTLE rigid TIP3P geometry

- S. Miyamoto and P. A. Kollman, "SETTLE: An analytical version of the SHAKE
  and RATTLE algorithm for rigid water models", *J. Comput. Chem.* **13**, 952
  (1992) — the algorithm, including the Appendix's closed form.
- W. L. Jorgensen et al., *J. Chem. Phys.* **79**, 926 (1983) — TIP3P:
  r(OH) = 0.9572 Å, θ(HOH) = 104.52° ⇒ r(HH) = 2 r sin(θ/2) = 1.5139 Å;
  the Lennard-Jones site is on **oxygen only** (ε = 0.1521 kcal/mol,
  σ = 3.1507 Å) and the intramolecular non-bonded terms are excluded.

## What this case pins

Two properties, and the second is the one that has teeth.

### 1. The constraint invariant

Two TIP3P waters 3.5 Å apart (so the O–O Lennard-Jones is a real force **and**
a real torque on each molecule), integrated for 1 ps at `dt = 2 fs`. Every
constrained distance — the two O–H bonds and the H–H separation that encodes
the H–O–H angle — must hold to a **relative error ≤ 1e-10** for the whole
trajectory. Measured: **1.3 × 10⁻¹⁵**, i.e. the algorithm is exact and the
residual is rounding.

A *flexible* water could not be integrated at 2 fs at all: the O–H stretch has
a period of ~9 fs. Removing that vibration is the entire practical point of
rigid water.

### 2. SETTLE must do nothing to a molecule that is already rigid

This is what separates SETTLE from "snap the triangle back into shape".

SETTLE is the **closed-form solution of the SHAKE equations** for a three-site
rigid body: the answer depends on the *reference* (constraint-satisfying)
triangle at time `t` as well as on the unconstrained triangle at `t + dt`.
Miyamoto & Kollman rotate the reference triangle, about the unconstrained
configuration's centre of mass, into the orientation SHAKE would have
converged to.

Reconstructing the ideal triangle from the *unconstrained* geometry alone
reproduces the bond lengths and **nothing else**. With no memory of where the
molecule was, the reconstruction is free to mirror it — and the check in §1
stays perfectly green while the molecule teleports every step. That is not a
hypothetical: it is exactly what the pre-2026-07 implementation did (a 1e-4 Å
perturbation moved a hydrogen **1.837 Å**, across the molecule, and the bond
lengths came out exact to twelve digits).

So the second gate integrates one isolated rigid water with **no forces on it**
and compares against the exact free-rigid-body answer, which is a pure
translation of the centre of mass at constant velocity (Newton's first law —
no fitted number, no tolerance to argue about):

```
r_i(t) = r_i(0) + v_com t
```

Gate: worst |r − r_exact| ≤ 1e-9 Å after 1 ps. Measured: **2.2 × 10⁻¹³ Å**.

## Cross-references

Unit-level invariants — the rigid-body identity, centre-of-mass conservation,
row-vs-tag identity across a spatial reorder, the per-launch Frame resolution
budget, the equal-hydrogen-mass precondition, and a molecule straddling a
periodic face — are pinned in `tests/cpu/dynamics/test_fix_settle.cpp`.

The general iterative alternative is `cpu::md::FixRattle`
(`regressions/md/rigid_water`): SETTLE is analytic and three-site-only,
RATTLE is iterative and holds any set of distances.
