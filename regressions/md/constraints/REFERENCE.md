# md/constraints

One case; every section below is run by `atv_regressions --case md/constraints` and keeps its own oracle record.


## Reference — RATTLE rigid water: rigidity and equipartition

### What this case pins

Two things, and the second is the one that matters.

#### 1. The constraint invariant

A constraint is not judged by a number it reproduces; it is judged by an **invariant it
never breaks**. Every constrained distance — the two O–H bonds and the H–H distance
that encodes the H–O–H angle — must stay at its target for the whole trajectory.

Measured: worst relative deviation **5 × 10⁻¹¹** over 5 ps at `dt = 2 fs`. (A *flexible*
water could not be integrated at 2 fs at all: the O–H stretch has a period of ~9 fs.
Removing that vibration is the entire practical point of rigid water.)

Geometry: TIP3P (Jorgensen et al., *J. Chem. Phys.* **79**, 926 (1983)) —
r(OH) = 0.9572 Å, θ(HOH) = 104.52°, hence d(HH) = 2 r sin(θ/2) = 1.5139 Å.

#### 2. Equipartition — which is really a test of the degree-of-freedom count

The **equipartition theorem**: at equilibrium each quadratic degree of freedom carries
`½ k_B T`, so

$$
T = \frac{2\,\text{KE}}{N_\text{dof}\,k_B}.
$$

(Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed., §2.2; Frenkel & Smit,
§4.3.)

A **rigid** water molecule has **6** degrees of freedom — 3 translational, 3 rotational
— not 9. The three constraints remove the three vibrational modes. For N molecules with
centre-of-mass motion removed:

```
Ndof = 6 N_mol - 3        NOT   3 N_atoms - 3 = 9 N_mol - 3
```

So the assertion is: thermostat the box at 300 K, and the temperature computed with the
**correct** Ndof must come back at 300 K.

Measured, 343 rigid waters at liquid density, CSVR thermostat at 300 K, averaged over
6 ps:

```
Ndof (rigid, 6/mol) = 2055     [3N-3 would be 3084]
<T>                 = 300.08 K   vs  300 K setpoint   (0.03 %, tol 3 %)
<T> with naive Ndof = 199.96 K   <- 33 % cold
```

**That last line is the bug this case exists to prevent.** Before this work, five
separate sites in the engine hardcoded `Ndof = 3N − 3`. With rigid water that ratio is
`(6N−3)/(9N−3) ≈ 2/3`, so the engine would have reported the water as ~33 % colder than
it was — and worse, since a thermostat's target kinetic energy is `Ndof k_B T / 2`, it
would have *driven the system to the wrong temperature* and reported success. Nothing
crashes. The trajectory is simply wrong.

The 3 % tolerance is the statistical spread of a thermostatted mean over 6 ps of a
343-molecule box, not a fudge factor: a broken DOF count misses by 33 %, an order of
magnitude outside it.

### What this case is NOT

**It is still not a TIP3P water simulation — but it is no longer a caricature of one.**

The box now runs at **liquid density** (O–O spacing 3.10 Å, 343 molecules), with the
Lennard-Jones site on **oxygen only**, exactly as TIP3P specifies. That became possible
when `PairLJCut` gained a per-type coefficient table: before it had one, a single ε/σ
applied to every atom, the hydrogens were forced to wear the oxygen's σ = 3.15 Å, and a
box at this spacing tore itself apart — no constraint solver can hold a molecule that is
being blown apart. Earlier versions of this case had to run dilute for that reason. They
no longer do.

What is still missing is **charges**. TIP3P is mostly electrostatics; the engine's PME
exists but is not wired into this case. Without charges there are no hydrogen bonds, so
this is a fluid of rigid LJ rotors at water's density, not liquid water.

That does not weaken what the case pins. Rigidity is a property of the constraint solver
and equipartition is a property of the DOF bookkeeping; both hold at any density and
under any force field. A real water benchmark — structure, diffusion, the O–O radial
distribution — belongs in `benchmarks/`, against an external engine, once charges are
wired.

### Rows are not an identity — tags are

The geometry check resolves each molecule's three atoms through the Frame's
tag index, not by row arithmetic (`3m, 3m+1, 3m+2`). It has to: `atom_modify
sort` spatially reorders the owned atoms at the first reneighbour after
`nfreq` steps (LAMMPS default 1000), which is inside this run. A row-indexed
check measures the distance between atoms of three *different* molecules from
step 1000 onward and reports a "geometry drift" of ~1e+01 while the constraint
solver is in fact holding every real bond to 5e-11. The engine's own solver
keys on tags for exactly this reason (`FixRattle`); an out-of-library consumer
must do the same.

### Cross-references

The constraint algebra itself (RATTLE: position stage, velocity stage, iteration to
tolerance) is unit-tested in `tests/cpu/md/test_fix_rattle.cpp`, including the second
invariant this case does not check — that the relative velocity along every constrained
bond vanishes, `d|r|²/dt = 2 r·(v_i − v_j) = 0`. Skip the velocity stage and the bond
*lengths* stay right while the molecule quietly absorbs energy along its own bonds.

- H. C. Andersen, "RATTLE: A 'velocity' version of the SHAKE algorithm for molecular
  dynamics calculations", *J. Comput. Phys.* **52**, 24 (1983).
- J.-P. Ryckaert, G. Ciccotti, H. J. C. Berendsen, "Numerical integration of the
  Cartesian equations of motion of a system with constraints", *J. Comput. Phys.* **23**,
  327 (1977) — the original SHAKE.


## Reference — SETTLE rigid TIP3P geometry

- S. Miyamoto and P. A. Kollman, "SETTLE: An analytical version of the SHAKE
  and RATTLE algorithm for rigid water models", *J. Comput. Chem.* **13**, 952
  (1992) — the algorithm, including the Appendix's closed form.
- W. L. Jorgensen et al., *J. Chem. Phys.* **79**, 926 (1983) — TIP3P:
  r(OH) = 0.9572 Å, θ(HOH) = 104.52° ⇒ r(HH) = 2 r sin(θ/2) = 1.5139 Å;
  the Lennard-Jones site is on **oxygen only** (ε = 0.1521 kcal/mol,
  σ = 3.1507 Å) and the intramolecular non-bonded terms are excluded.

### What this case pins

Two properties, and the second is the one that has teeth.

#### 1. The constraint invariant

Two TIP3P waters 3.5 Å apart (so the O–O Lennard-Jones is a real force **and**
a real torque on each molecule), integrated for 1 ps at `dt = 2 fs`. Every
constrained distance — the two O–H bonds and the H–H separation that encodes
the H–O–H angle — must hold to a **relative error ≤ 1e-10** for the whole
trajectory. Measured: **1.3 × 10⁻¹⁵**, i.e. the algorithm is exact and the
residual is rounding.

A *flexible* water could not be integrated at 2 fs at all: the O–H stretch has
a period of ~9 fs. Removing that vibration is the entire practical point of
rigid water.

#### 2. SETTLE must do nothing to a molecule that is already rigid

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

### Cross-references

Unit-level invariants — the rigid-body identity, centre-of-mass conservation,
row-vs-tag identity across a spatial reorder, the per-launch Frame resolution
budget, the equal-hydrogen-mass precondition, and a molecule straddling a
periodic face — are pinned in `tests/cpu/dynamics/test_fix_settle.cpp`.

The general iterative alternative is `cpu::md::FixRattle`
(`regressions/md/constraints`): SETTLE is analytic and three-site-only,
RATTLE is iterative and holds any set of distances.


## virtual-site-multi-topology — general virtual sites (TIP4P + TIP5P)

oracle: lammps  
oracle_version: 30 Mar 2026 - Development (develop / patch_30Mar2026-1074-g7f680de296)  
binary: `/home/jicli594/.local/x86_64/bin/lmp`

### What is pinned

**TIP4P** two-water single-point energy and **parent** forces using Atomiverse
`cpu::md::VirtualSites` (Weighted3 M-sites as Frame atoms) + `PairLJCut` +
`PairCoulCut`, matched to LAMMPS `pair_style lj/cut/tip4p/cut` (same physics as
`four-site-tip4p-foothold`, different ATV path).

**TIP5P** is dual-gated by geometry only (`r_OL`, `θ_LOL`) via OutOfPlane3
construction (no LAMMPS `tip5p` pair on host binary → no second energy pin).

| Item | Value |
|------|--------|
| r_OH | 0.9572 Å |
| θ_HOH | 104.52° |
| qdist (OM) | 0.15 Å |
| q_M (site) | −1.04 e (O charge 0 on Frame) |
| q_H | +0.52 e |
| ε_OO | 0.1550 kcal/mol → eV via `kcal_per_mol_to_ev` |
| σ_OO | 3.15365 Å |
| cut_lj = cut_coul | 10.0 Å |
| LJ shift | yes |
| special_bonds | lj/coul 0 0 0 |
| geometry | O–O = 2.8 Å along x; water2 rotated π about z |
| box | cubic 20 Å, periodic |

### Offline pin path

```text
tools/metalwalls_oracle/inputs/lammps/virtual-site-multi-topology/
  in.tip4p_dimer_metal
  data.tip4p_dimer_metal
  log.metal
```

(Same twin as `four-site-tip4p-foothold`; regenerate with `lmp -in in.tip4p_dimer_metal`.)

### Pin values (units metal = eV, eV/Å)

| Quantity | Value |
|----------|--------|
| \(E\) | `0.52924228894955327451` eV |

Forces on **parents only** (O1 H1 H2 O2 H3 H4), fy = 0 — after Feenstra
redistribute (M force must be zero):

| atom | fx [eV/Å] | fz [eV/Å] |
|------|-----------|-----------|
| 1 O | −0.094382238074752045076 | 0.33187666441899288383 |
| 2 H | −1.0593571566097745862 | −0.31522207906906996033 |
| 3 H | −0.057823257196248978307 | −0.016654585349923187176 |
| 4 O | 0.094382238074751600987 | 0.33187666441899305037 |
| 5 H | 1.0593571566097748082 | −0.31522207906906962727 |
| 6 H | 0.057823257196249200351 | −0.01665458534992322534 |

### Tolerances (dual-gate)

- Energy: \(\|E - E_{\mathrm{pin}}\| \le 10^{-8}\) eV  
- Geometry TIP4P: OM = qdist within \(10^{-10}\) Å  
- Forces: per-component on parents \(\le 10^{-6}\) eV/Å; site F = 0  
- TIP5P: \(r_{OL}=0.70\) Å and \(\theta_{LOL}=109.47°\) within \(10^{-8}\)  

### Perf

Fair wall: ATV `reg_virtual-site-multi-topology` vs LAMMPS
`in.tip4p_dimer_metal` / `run 0` (serial `OMP_NUM_THREADS=1`).

### Perf result

| Item | Value |
|------|--------|
| host | x86_64 login, Release MD-only build |
| T_atv (mean / launch) | 0.01055 s |
| T_oracle (mean / launch) | 0.1260 s |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) | **0.0837** |
| within_10x | **true** |
| perf_exempt | false |
| method | 50 process launches after 3 warmups; `OMP_NUM_THREADS=1` |


## four-site-tip4p-foothold — TIP4P four-site water dimer

oracle: lammps  
oracle_version: 30 Mar 2026 - Development (develop / patch_30Mar2026-1074-g7f680de296)  
binary: `/home/jicli594/.local/x86_64/bin/lmp`

### What is pinned

Classic **Jorgensen TIP4P** (1983/85) two-water **single-point** energy and forces
with LAMMPS `pair_style lj/cut/tip4p/cut` and Atomiverse
`cpu::PairLJCutTip4pCut` (M virtual site + Feenstra force partition).

| Item | Value |
|------|--------|
| r_OH | 0.9572 Å |
| θ_HOH | 104.52° |
| qdist (OM) | 0.15 Å |
| q_O (on M) | −1.04 e |
| q_H | +0.52 e |
| ε_OO | 0.1550 kcal/mol → eV via `kcal_per_mol_to_ev` |
| σ_OO | 3.15365 Å |
| cut_lj = cut_coul | 10.0 Å |
| LJ shift | yes (`pair_modify shift yes` / ATV V(rc)=0) |
| special_bonds | lj/coul 0 0 0 |
| geometry | O–O = 2.8 Å along x; water2 rotated π about z |
| box | cubic 20 Å, periodic |

### Offline pin path

```text
tools/metalwalls_oracle/inputs/lammps/four-site-tip4p-foothold/
  in.tip4p_dimer_metal
  data.tip4p_dimer_metal
  log.metal
```

Regenerate:

```bash
cd tools/metalwalls_oracle/inputs/lammps/four-site-tip4p-foothold
lmp -in in.tip4p_dimer_metal -log log.metal
```

### Pin values (units metal = eV, eV/Å)

| Quantity | Value |
|----------|--------|
| \(E\) | `0.52924228894955327451` eV |
| \(E_{\mathrm{vdwl}}\) | `0.057185016777344513383` eV |
| \(E_{\mathrm{coul}}\) | `0.4720572721722087195` eV |

Forces (atom order O1 H1 H2 O2 H3 H4), fy = 0:

| atom | fx [eV/Å] | fz [eV/Å] |
|------|-----------|-----------|
| 1 O | −0.094382238074752045076 | 0.33187666441899288383 |
| 2 H | −1.0593571566097745862 | −0.31522207906906996033 |
| 3 H | −0.057823257196248978307 | −0.016654585349923187176 |
| 4 O | 0.094382238074751600987 | 0.33187666441899305037 |
| 5 H | 1.0593571566097748082 | −0.31522207906906962727 |
| 6 H | 0.057823257196249200351 | −0.01665458534992322534 |

### Tolerances (dual-gate)

- Energy: \(\|E - E_{\mathrm{pin}}\| \le 10^{-8}\) eV  
- Geometry: reconstructed OM = qdist within \(10^{-10}\) Å  
- Forces: per-component \(\le 10^{-6}\) eV/Å  

### Perf

Fair wall: same dimer single-point, ATV Evaluate (`reg_four-site-tip4p-foothold`)
vs LAMMPS `in.tip4p_dimer_metal` / `run 0` (both **serial** `OMP_NUM_THREADS=1`,
50 process launches after 3 warmups; includes process start + build).

### Perf result

| Item | Value |
|------|--------|
| host | x86_64 login, Release MD-only build |
| T_atv (mean / launch) | 0.0523 s |
| T_oracle (mean / launch) | 0.1548 s |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}}\) | **0.338** |
| within_10x | **true** |
| perf_exempt | false |


## Reference — ForceField bag → Evaluate single-point

### Roles

- `cpu::ForceField` — owns potential objects (`set_potential` / `set_evaluator` / …).
- `cpu::Evaluate` — one-pass **Driver** preset (single-point energy/force).
- `ff.apply(eval)` moves the bag into the driver before `build`/`run`.

### Checks

Two waters with LJ + coul/cut + harmonic bond/angle and `special_bonds 0 0 0`.
Energy finite, |E| < 50 eV.
