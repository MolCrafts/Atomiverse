# Reference — Stillinger-Weber silicon

## The potential

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

## What is asserted

### 1 & 2. The two fitted numbers

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

### 3. Diamond must beat close packing

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

## What is checked elsewhere

Forces — including the angular derivative, where a three-body implementation most easily
gets a sign wrong — are gated against a finite-difference oracle of this same energy in
`tests/cpu/potentials/test_sw.cpp` (agreement to 1e-5 eV/Å), together with the
sum-of-forces-is-zero check that the triplet decomposition must satisfy.
