# md/manybody-potentials

One case; every section below is run by `atv_regressions --case md/manybody-potentials` and keeps its own oracle record.


## Reference — EAM copper: cohesive energy and lattice constant

### The potential

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

### What is asserted

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

### Measured, and the tolerance

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

### What else this case pins

The forces are **not** checked here — they are checked where they belong, in
`tests/cpu/potentials/test_eam.cpp`, against a finite-difference oracle of the very
energy this case evaluates (agreement to 1e-6 eV/Å). That gate is mandatory for every
force in the engine: the GPW path once shipped a force that was the gradient of a
*different* functional than its energy, and it survived 17 commits because the tests
only ever checked energies.


## Reference — Stillinger-Weber silicon

### The potential

F. H. Stillinger and T. A. Weber, **"Computer simulation of local order in condensed
phases of silicon"**, *Phys. Rev. B* **31**, 5262 (1985).
DOI 10.1103/PhysRevB.31.5262.

```
E = sum_{i<j} phi2(r_ij)  +  sum_i sum_{j<k} phi3(r_ij, r_ik, theta_jik)

phi2(r) = A*eps*[B*(sigma/r)^p - (sigma/r)^q] * exp(sigma/(r - a*sigma))
phi3    = lambda*eps*(cos(theta) - cos(theta0))^2
          * exp(gamma*sigma/(r_ij - a*sigma)) * exp(gamma*sigma/(r_ik - a*sigma))
```

Silicon parameters, as published (`cpu/potentials/sw_data/sw_tables.h`):

| eps [eV] | sigma [Å] | A | B | p | q | a | lambda | gamma | cos(theta0) |
|---|---|---|---|---|---|---|---|---|---|
| 2.1683 | 2.0951 | 7.049556277 | 0.6022245584 | 4 | 0 | 1.80 | 21.0 | 1.20 | −1/3 |

Both terms vanish *identically* at `r = a*sigma = 3.7712 Å`, so the cutoff is exact and
there is no truncation tail — unlike the power-law EAM next door.

`theta0 = arccos(−1/3) = 109.47°` is the **tetrahedral angle**. The three-body term is a
penalty on deviating from it.

### What is asserted

#### 1 & 2. The two fitted numbers

Stillinger and Weber fit the potential so that diamond-cubic silicon is the ground state
with the experimental lattice constant and cohesive energy:

| Quantity | Value |
|---|---|
| Equilibrium lattice constant `a0` | **5.431 Å** |
| Cohesive energy `E_coh` | **4.3363 eV/atom** |

Measured (216 atoms, lattice scan with parabolic refinement of the minimum):

```
equilibrium a0 = 5.4312 A    vs  5.431 A     (0.00%)
cohesive E     = 4.3364 eV   vs  4.3363 eV   (0.00%)
```

Tolerance **1 %**, and the result lands four to five significant figures deep. It can:
SW is strictly cutoff-terminated, so unlike the Sutton-Chen EAM there is no truncated
tail to excuse an error with. Because both numbers are *fit targets*, agreement to this
depth is an independent check that every one of the ten parameters was transcribed
correctly and that the functional form — including the angular term — is right.

#### 3. Diamond must beat close packing

The physics test, and the reason this potential exists at all.

Silicon crystallizes in the **open** tetrahedral diamond lattice (coordination 4), not in
a close-packed one (coordination 12). **No pair potential can reproduce that.** A pair
potential's energy is a sum over separations, so it is always lowered by adding more
neighbours at the optimal distance — it invariably favours close packing. The only way to
prefer an open structure is to make the energy depend on the **angles** between bonds,
which is exactly what `phi3` does.

So the case also computes fcc silicon **at the same number density** as diamond (so the
comparison is about structure, not volume) and requires it to be *higher* in energy:

```
E(diamond)            = -4.3364 eV/atom
E(fcc, same density)  = -3.4520 eV/atom     (higher — correct)
```

If the three-body term were dropped, mis-signed, or its angular derivative broken, fcc
would win this comparison and the case would fail. That makes this check, not the two
fitted numbers, the one that actually tests *three-bodyness*.

### What is checked elsewhere

Forces — including the angular derivative, where a three-body implementation most easily
gets a sign wrong — are gated against a finite-difference oracle of this same energy in
`tests/cpu/potentials/test_sw.cpp` (agreement to 1e-5 eV/Å), together with the
sum-of-forces-is-zero check that the triplet decomposition must satisfy.
