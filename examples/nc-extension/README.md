# An independent CPU norm-conserving potential

This separate C++20 project derives `ATV::electronic::NormConservingPotential`.
`PublishedSi` directly supplies the published Si GTH-PBE-q4 parameters of
Krack, *Theoretical Chemistry Accounts* 114 (2005) 145. It does not wrap the
built-in GTH provider and does not use an enum identity, private header,
UPF reader, Fourier kernel, electronic workspace or a second SCF loop.

Build/install Atomiverse with PW and EVALUATE enabled, then copy **this
whole directory outside the checkout** and configure against a relocated
installation:

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/relocated/install
cmake --build build
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./build/nc-extension operators
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./build/nc-extension si2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./build/nc-extension lifecycle
# When the installation also enables the corresponding optional module:
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ./build/nc-extension md
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ./build/nc-extension minimize
```

Only `find_package(Atomiverse CONFIG REQUIRED COMPONENTS PW EVALUATE)` and
`Atomiverse::atomiverse` are used. There is no source-tree include path.
Each acceptance command enforces a complete 60 s bound at its declared
thread width: si2/lifecycle at 1, optional md/minimize at 4. These are
independent workflow checks, not new registered method cases. Native compilation, numerical and relocated
consumer validation are **pending** at this edit; no passing run is claimed.

`PublishedSi` owns its increasing uniform radius mesh and exact positive
`dr/di` Jacobian, and every beta span is an exact prefix including its
integration endpoint. `beta=r*p_l` has units Bohr^-1/2, the row-major D is
in Ha, with a genuine off-diagonal s-channel coupling. The local callback
writes V[Ha] and dV/dr[Ha/Bohr], including their smooth-origin limits and
Coulomb tail. Private storage is constructed once; const calls allocate
nothing and retain no output spans. The provider is shared by const pointer,
while each PW owns its electronic histories. PW owns all Fourier transforms,
phases, integrations and force contractions. Unsupported USPP/PAW, NLCC and
spin-orbit data are outside this scalar NC contract, not alternate flags in
this example. Their admission failures belong to the existing reader tests.

`operators` checks the actual supplied radial tables against elementary
Gaussian integrals: unit projector norms, s-overlap sqrt(3/5), and the
nonlocal action on `(p1+.4*p2)Y00+.2*p1Y10` using the full D. Local origin,
tail and analytic radial derivatives are checked separately. The short
Simpson sum is a radial inner-product check, not a copied PW operator.
`si2` checks actual external E/all six nonzero force components for standalone
and nested Evaluate using the same method instance, energy-only force
preservation, const-Frame preservation, and repeated nested evaluation without
double accumulation. `lifecycle` checks shared-provider isolation, changed
geometry/cell, species mismatch before publication, accepted-result
preservation and successful same-object retry. No stress qualification is
claimed by this example. The optional `md` command uses the exact same PW/provider identity. It
poisons old force columns, checks the first Verlet displacement against the
pinned initial force, continues the trajectory, changes host geometry/cell,
and compares final E/full forces with a fresh method using the same provider.
The optional `minimize` command uses existing L-BFGS/Armijo for one iteration
with the preset ForceConverged criterion. Its overlong first trial must be
rejected (accepted displacement strictly below that trial), and E/full forces
at the accepted geometry are compared with a fresh method. These fresh-method
comparisons are supplements, not independent oracles. One optimization step
is not a claim of a stationary minimum. Each optional command checks the
external absolute pin before motion. The `#ifdef ATV_HAS_MD` and
`ATV_HAS_MINIMIZER` branches preserve the minimal PW+Evaluate closure; a
requested mode whose module is disabled rejects explicitly.

## Independent same-Hamiltonian pin

**Actually executed QE7.5** (reported commit
`770a0b2d12928a67048e2f3da8d10d057e52179e-dirty`) reads the same published
coefficients through its native `.gth` reader. Its own analytic local and
nonlocal Fourier operators are used; there is no Atomiverse-generated UPF or
reimplemented external solver. Original input, parameter block, full stdout,
commands, executable/source hashes and extraction script are archived in
`tools/diagnostics/nc-extension/`. `pins.h` contains only numbers printed by
that execution. Native output is never used as an oracle.

The periodic cubic 6 Å cell contains Si(2,2,2) and Si(3.8,3.2,3.0) Å. Both
engines use PBE, the same Si GTH-PBE-q4, Γ, ecutwfc 12 Ry/ecutrho 48 Ry,
restricted Fermi–Dirac smearing 0.02 Ry and six bands. The pin is Mermin F,
including -TS. Native controls are tol 1e-9, max_iter 200, mixing 0.2 and two
extra bands. QE converged in 35 iterations at conv_thr 1e-11 Ry; its printed
SCF accuracy is 1.2e-12 Ry. QE explicitly uses 40³ dense/smooth FFT grids.
Native uses a 26³ base density grid and 40³ private PBE quadrature; both
represent the same 48 Ry density sphere. This records the discretization
rather than claiming an identical software implementation.

QE printed F=-15.25142886 Ry. Atom1 force is
(+0.03584083,+0.04163713,+0.03671977) Ry/Bohr; atom2 is its negative.
The quoted bounds were fixed before native acceptance: **1e-5 Ha** in E
and **1e-4 Ha/Bohr per Cartesian force component**. QE print precision is
5e-9 Ha in E and 2.5e-9 Ha/Bohr in F; its SCF error is smaller still. The
bounds allow residual finite-grid/radial quadrature differences and are
stricter than a generic millihartree agreement. They are not a claim of
cutoff-converged material properties. The local/normalization/action checks
use analytic bounds 3e-12/3e-11; the local derivative central difference uses
h=1e-5 Bohr and 2e-7 Ha/Bohr to bound its O(h²) truncation.

QE's process wall 0.65 s is raw provenance at environment requests
OMP_NUM_THREADS=1/OPENBLAS_NUM_THREADS=1. It is not a native comparison or a
fair speed ratio. The earlier fixed-occupation diagnostic failed to converge
in 200 iterations and is retained separately; none of its outputs are pins.
