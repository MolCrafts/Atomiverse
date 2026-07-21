# Reference — EAM copper: cohesive energy and lattice constant

## The potential

A. P. Sutton and J. Chen, **"Long-range Finnis-Sinclair potentials"**,
*Philos. Mag. Lett.* **61**, 139 (1990). DOI 10.1080/09500839008206493.

An analytic embedded-atom (Finnis-Sinclair-class) potential:

```
E    = eps * [ 1/2 * sum_i sum_{j!=i} (a/r_ij)^n  -  c * sum_i sqrt(rho_i) ]
rho_i = sum_{j!=i} (a/r_ij)^m
```

i.e. EAM with `phi(r) = eps (a/r)^n`, `f(r) = (a/r)^m`, `F(rho) = -eps c sqrt(rho)`.
The **square root** is the whole point: `F` is nonlinear in `rho`, so the energy does
not decompose into independent pairs and the force between `i` and `j` depends on how
densely embedded *both* of them are. This is the first genuinely many-body potential in
the engine.

Copper parameters, as published (`cpu/potentials/eam_data/eam_tables.h`):

| n | m | eps [eV] | c | a [Å] |
|---|---|---|---|---|
| 9 | 6 | 1.2382e-2 | 39.432 | 3.61 |

## What is asserted

Sutton and Chen **fit** these parameters to two experimental properties of fcc copper,
so those two numbers are what the potential must reproduce:

| Quantity | Value | Source |
|---|---|---|
| Equilibrium lattice constant `a0` | **3.61 Å** | the `a` parameter is *defined* as the fcc lattice constant |
| Cohesive energy `E_coh` | **3.54 eV/atom** | experimental Cu cohesive energy, the SC fit target (Kittel, *Introduction to Solid State Physics*, Table 3 — 3.49 eV; the SC/EAM literature quotes 3.54 eV) |

The case relaxes fcc Cu by scanning the lattice constant, parabolically refines the
minimum, and checks both.

**This is not a golden-value test.** The expected numbers come from the paper's fit
targets, not from our output. Because they are *fit* targets, this case doubles as an
independent check on our hand-transcription of the parameter table: if `n`, `m`, `eps`,
`c`, or `a` were mis-copied, or the functional form mis-implemented, the minimum would
not land on 3.61 Å.

## Measured, and the tolerance

At `cutoff = 10 Å` (864 atoms, 6×6×6 fcc):

```
equilibrium a0 = 3.6121 A   vs  3.61 A     (0.06% high)
cohesive E     = 3.4828 eV  vs  3.54 eV    (1.62% low)
```

Tolerance: **2 %** on both. The lattice constant lands within 0.06 %, which is the
strong statement — it says the functional form and every parameter are right.

The cohesive energy is systematically ~1.6 % **low**, and that is expected rather than
a defect: the Sutton-Chen form is a **power law with no intrinsic range**, so a finite
cutoff truncates a tail that is still contributing. Truncating the density sum
underestimates `rho`, and since `F(rho) = -eps c sqrt(rho)` is monotonically decreasing,
too little density means too little binding. The deficit therefore has a predicted sign,
and it shrinks monotonically as the cutoff grows — measured here:

| cutoff [Å] | E_coh [eV] | deficit |
|---|---|---|
| 5 | 3.3725 | −4.7 % |
| 8 | 3.4651 | −2.1 % |
| 10 | 3.4828 | −1.6 % |

Converging upward toward 3.54, exactly as the truncation argument requires. The cutoff is
capped at 10 Å here only because the minimum-image convention needs `L > 2 r_c`, and a
bigger box costs runtime this case does not have. The 2 % tolerance is set to admit the
known truncation deficit while still failing any real error — a sign flip, a missing
embedding term, or a mis-transcribed parameter all move the answer by far more.

## What else this case pins

The forces are **not** checked here — they are checked where they belong, in
`tests/cpu/potentials/test_eam.cpp`, against a finite-difference oracle of the very
energy this case evaluates (agreement to 1e-6 eV/Å). That gate is mandatory for every
force in the engine: the GPW path once shipped a force that was the gradient of a
*different* functional than its energy, and it survived 17 commits because the tests
only ever checked energies.
