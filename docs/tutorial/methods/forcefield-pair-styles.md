# Classical short-range pair styles and walls

Short-range pair potentials and continuum walls for molten salts, ionic
crystals and electrode walls, each a Component on `cpu::Evaluate` or
`cpu::md::MD` (never a Driver category), parameterized by constructor
arguments and public members (no `*Config` bag, no factory). Energies land on
`kEnergy`, forces on `kAtom_f*`.

- `cpu::PairBorn` — Born–Mayer–Huggins (LAMMPS `pair_style born`),
  `Potential`; single-type `PairBorn{A, rho, sigma, C, D, rc}` or a multi-type
  `std::vector<BornTypeCoeff>` table (Fumi–Tosi ion–ion short range; no
  Lorentz–Berthelot mixing — unlisted type pairs do not interact).
- `cpu::PairBuck(A, rho, C, cutoff)` — Buckingham (LAMMPS `pair_style buck`),
  `Potential`.
- `cpu::md::FixWallLJ1043` — Steele 10-4-3 continuum wall (LAMMPS
  `fix wall/lj1043`), `Bias`, default `md::Phase::post_force` order +100.
- Coulomb is always a separate component (`cpu::PairCoulCut`, or `CoulLong` +
  a reciprocal partner).

All sections are CPU single points (CUDA rows are ○ on the force-field
roadmap). The case also registers the sections `coul_soft_dimer` and
`lj_soft_dimer` (see `case.cmake`), which have no section on this page.

Case: `ctest -R '^forcefield/pair-styles$'` — sources and oracle record in [`regressions/forcefield/pair-styles/`](../../../regressions/forcefield/pair-styles/). One section of the case per `##` heading below: `atv_regressions --case forcefield/pair-styles --section <name>`.

## Common setup

- **Build:** every section requires the `POTENTIALS` module; CPU login shell
  with `buildenv-gcc/2026.03-mpich`, as in the top-level build instructions.
- **Oracle: LAMMPS** for every section — the styles exist in LAMMPS, so LAMMPS
  is the binding oracle (MetalWalls is reserved for MetalWalls-only physics; the
  styles belong to the electrochemical force-field slice E8 of
  `.claude/notes/metalwalls-capability-map.md`). Version: **30 Mar 2026 -
  Development** (`develop` / `patch_30Mar2026-1074-g7f680de296`), local binary
  `/home/jicli594/.local/x86_64/bin/lmp`, `units metal` (eV, Å — the public
  Atomiverse surface), `run 0`, thermo PE + forces dumped with `%.16e`. Offline
  pins only — ctest never launches LAMMPS. Each pin is also checked against an
  independent analytic evaluation of the same algebra.
- **Energy shift.** `PairBorn` and `PairBuck` always shift so \(U(r_c)=0\)
  (LAMMPS `pair_modify shift yes`); forces use the unshifted derivative (the
  shift is constant). `PairCoulCut` is unshifted.
- **Neighbor list.** `as_neighbor_consumer()` returns null on `PairBorn` and
  `PairBuck`, so no run is required to install a `NeighborList`. Single-type
  `PairBorn` walks `ctx.md.pair_list` when a `NeighborList` published one and
  falls back to an O(N²) all-pairs walk otherwise; the multi-type `PairBorn`
  table and `PairBuck` walk all pairs, O(N²).
- **Performance: `perf_exempt`** for every section — each pin is a 1–4-atom
  single point whose wall is dominated by process startup (LAMMPS `run 0`
  ~ms, ATV `Evaluate` ~µs–ms), not the pair kernel (`ratio_atv_over_oracle`
  n/a, `within_10x` exempt). A fair bulk twin (molten-salt box or multi-atom
  slab) against LAMMPS on the same lattice is not measured.
- **Out of scope:** special-bond scaling tables, long-range Coulomb in these
  pins, CUDA.

## born-repulsion-foothold — Born–Mayer–Huggins ion dimer (LAMMPS `pair_style born`) {#born-repulsion-foothold}

### Summary

Born–Mayer–Huggins (BMH) short-range pair energy and force: exponential core
repulsion plus attractive \(r^{-6}\) and \(r^{-8}\) tails, shifted so
\(U(r_c)=0\). The section is a two-atom single-type dimer against LAMMPS
`pair_style born` (`pair_modify shift yes`), asserting energy and \(F_x\) to
\(10^{-10}\) eV (eV/Å). Coulomb is not included — compose with `PairCoulCut` /
`coul/long` for ionic force fields (see [`fumi-tosi-nacl`](#fumi-tosi-nacl)).

### Method

LAMMPS `pair_style born` form (coeff order `A rho sigma C D`; \(\rho\) is the
softness length in Å):

\[
E(r) = A\,e^{(\sigma-r)/\rho} - \frac{C}{r^{6}} + \frac{D}{r^{8}},
\quad r < r_c,
\qquad
E_{\mathrm{shift}}(r) = E(r) - E(r_c)
\]

\[
\frac{\mathrm{d}E}{\mathrm{d}r}
  = -\frac{A}{\rho}\,e^{(\sigma-r)/\rho}
  + \frac{6C}{r^{7}}
  - \frac{8D}{r^{9}}
\]

| Item | Setting |
| ---- | ------- |
| Atoms | 2 (single type, `kAtom_type = 0`) |
| Separation \(r\) | 2.5 Å along \(x\) |
| Box | cubic 40 Å, fully periodic |
| \(A\) | 1000.0 eV |
| \(\rho\) | 3.0 Å |
| \(\sigma\) | 1.0 Å |
| \(C\) | 50.0 eV·Å⁶ |
| \(D\) | 20.0 eV·Å⁸ |
| \(r_c\) | 8.0 Å |
| Driver | `cpu::Evaluate` + `set_evaluator(PairBorn{…})` |

LAMMPS twin: `pair_style born 8.0`, `pair_coeff 1 1 1000.0 3.0 1.0 50.0 20.0`,
`pair_modify shift yes`.

### Usage

```cpp
const double A = 1000.0;   // eV
const double rho = 3.0;    // Å
const double sigma = 1.0;  // Å
const double C = 50.0;     // eV·Å⁶
const double D = 20.0;     // eV·Å⁸
const double rc = 8.0;     // Å
const double r = 2.5;      // Å

cpu::Frame f(2);
f.span(kAtom_x)[0] = 0;
f.span(kAtom_y)[0] = 0;
f.span(kAtom_z)[0] = 0;
f.span(kAtom_x)[1] = r;
f.span(kAtom_y)[1] = 0;
f.span(kAtom_z)[1] = 0;
for (int i = 0; i < 2; ++i) {
    f.span(kAtom_mass)[i] = 1.0;
    f.span(kAtom_type)[i] = 0;
    f.span(kAtom_tag)[i] = i + 1;
}
f.box = cpu::Box::cubic(40.0);

cpu::Evaluate eval;
eval.set_evaluator(cpu::PairBorn{A, rho, sigma, C, D, rc});
eval.build();
eval.run(f);

const double E = f.meta.get(kEnergy);  // [eV]
```

Multi-type Fumi–Tosi table:

```cpp
std::vector<cpu::BornTypeCoeff> coeffs{
    {/*type_i=*/0, /*type_j=*/0, A00, rho00, sigma00, C00, D00},
    {0, 1, A01, rho01, sigma01, C01, D01},
    {1, 1, A11, rho11, sigma11, C11, D11},
};
cpu::PairBorn born(std::move(coeffs), /*cutoff [Å]=*/rc);
```

On classical MD the same component binds with `sim.set_evaluator(...)`. Files:
`pair_born.h` (`PairBorn`, `BornTypeCoeff`), `src/cpu/potentials/pair_born.cpp`;
unit tests in `tests/cpu/potentials/test_pair_born.cpp` (analytic shifted \(U\)
and \(\mathrm{d}U/\mathrm{d}r\) to \(10^{-12}\)).

### Validation (oracle: LAMMPS)

| Quantity | Atomiverse / pin | LAMMPS pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E\) [eV] | **509.3671885909988** | **509.3671885909988** | \(\lvert\Delta E\rvert\le 10^{-10}\) | shifted pair energy |
| \(F_{1x}\) [eV/Å] | **−2.0172730961087780×10²** | **−2.0172730961087780×10²** | \(\lvert\Delta F\rvert\le 10^{-10}\) | atom at \(x=0\) |
| \(F_{2x}\) [eV/Å] | **+2.0172730961087780×10²** | **+2.0172730961087780×10²** | \(\lvert\Delta F\rvert\le 10^{-10}\) | atom at \(x=2.5\) Å |
| \(F_{1y},F_{1z},F_{2y},F_{2z}\) | **0** | **0** | \(10^{-10}\) | transverse components |

### References

- LAMMPS `pair_style born` (Born–Mayer–Huggins), `pair_modify shift`
- Born–Mayer; Huggins–Mayer / Fumi–Tosi molten-salt short-range forms

## buckingham-pair-lammps — Buckingham ion dimer (LAMMPS `pair_style buck`) {#buckingham-pair-lammps}

### Summary

`cpu::PairBuck` evaluates the Buckingham (Born–Mayer repulsion + \(C/r^{6}\)
dispersion) pair potential, matching LAMMPS `pair_style buck` with
`pair_modify shift yes`, on a two-atom single-type dimer. Energy and forces are
asserted against the offline LAMMPS pins to \(10^{-10}\) eV (eV/Å) — a
dimer single point is algebraically exact against the shifted formula, so the
tolerance catches a wrong shift, units or force sign without neighbor-rebuild
noise.

### Method

\[
E(r) = A\,e^{-r/\rho} - \frac{C}{r^{6}},\quad r < r_c,
\qquad
E_{\mathrm{shift}}(r) = E(r) - E(r_c),\qquad
E(r_c)=A\,e^{-r_c/\rho}-\frac{C}{r_c^{6}}
\]

\[
\frac{\mathrm{d}E}{\mathrm{d}r}
  = -\frac{A}{\rho}\,e^{-r/\rho} + \frac{6C}{r^{7}}
\]

Newton third-law pair forces along \(\mathbf{r}_{ij}\); min-image under PBC.

| Item | Setting |
| ---- | ------- |
| Atoms | 2, single type; atom 0 at origin, atom 1 at \((2.5, 0, 0)\) Å |
| Box | cubic 40 Å, PBC |
| \(A\) | 1000.0 eV |
| \(\rho\) | 0.3 Å |
| \(C\) | 50.0 eV·Å⁶ |
| \(r_c\) | 8.0 Å |
| LAMMPS twin | `lmp -in in.buckingham_dimer` |

**Out of scope:** multi-type coefficients / mixing (the multi-type salt path is
`PairBorn`, [`fumi-tosi-nacl`](#fumi-tosi-nacl)).

### Usage

Same frame as [`born-repulsion-foothold`](#born-repulsion-foothold):

```cpp
cpu::Evaluate eval;
eval.set_evaluator(cpu::PairBuck{/*A=*/1000.0, /*rho=*/0.3, /*C=*/50.0,
                                 /*rc=*/8.0});
eval.build();
eval.run(f);

const double E = f.meta.get(kEnergy);           // eV
const double fx0 = f.span(kAtom_fx)[0];         // eV/Å
const double fx1 = f.span(kAtom_fx)[1];         // eV/Å
```

Files: `pair_buck.h`, `src/cpu/potentials/pair_buck.cpp`; unit tests in
`tests/cpu/potentials/test_pair_buck.cpp` (analytic \(E,F\); zero at \(r_c\);
invalid parameters). Roadmap: `docs/roadmap/force-fields.md` (Buckingham CPU ✓).

### Validation (oracle: LAMMPS)

| Quantity | Atomiverse / pin | LAMMPS offline pin | Tol | Notes |
| -------- | ---------------: | -----------------: | --- | ----- |
| \(E\) [eV] | **0.03576020865970150342** | **0.03576020865970150342** | \(1\times 10^{-10}\) eV | |
| \(F_{1x}\) [eV/Å] | **−0.30971158806504689** | **−0.30971158806504689** | \(1\times 10^{-10}\) eV/Å | atom 0 (at origin) |
| \(F_{2x}\) [eV/Å] | **+0.30971158806504689** | **+0.30971158806504689** | \(1\times 10^{-10}\) eV/Å | atom 1; Newton III |
| \(F_{iy}, F_{iz}\) | **0** | **0** | \(1\times 10^{-10}\) eV/Å | transverse components |

## fumi-tosi-nacl — Fumi–Tosi multi-type Born + Coulomb, NaCl four-ion square {#fumi-tosi-nacl}

### Summary

Fumi–Tosi short-range ion–ion interactions for molten salts / alkali halides
are the BMH pair with per type-pair coefficients plus a separate Coulomb term:
multi-type `cpu::PairBorn` + `cpu::PairCoulCut` on `cpu::Evaluate`, against
LAMMPS `pair_style hybrid/overlay born coul/cut` on a four-ion planar NaCl
square.

### Method

BMH (shifted, as in [`born-repulsion-foothold`](#born-repulsion-foothold),
LAMMPS `pair_modify pair born shift yes`):

\[
E_{\mathrm{born}}(r)=A\,e^{(\sigma-r)/\rho}-\frac{C}{r^{6}}+\frac{D}{r^{8}}-E(r_c),
\quad r < r_c
\]

Coulomb (unshifted cut, LAMMPS metal `coul/cut`):

\[
E_{\mathrm{coul}}(r)=\mathrm{qqr2e}\,\frac{q_i q_j}{r},\quad
\mathrm{qqr2e}=14.399645\ \mathrm{eV\cdotÅ}
\]

(same as `units::coulomb`). LAMMPS twin:

```text
pair_style hybrid/overlay born 12.0 coul/cut 12.0
pair_modify pair born shift yes
```

| Field | Value |
| ----- | ----- |
| Atoms | 4 (2 Na + 2 Cl planar square) |
| Edge \(d\) | 2.82 Å |
| Positions [Å] | Na(10,10,20), Cl(12.82,10,20), Na(12.82,12.82,20), Cl(10,12.82,20) |
| Box | cubic 40 Å, PBC |
| Types | Na = frame type 0 / LAMMPS type 1; Cl = 1 / 2 |
| Charges | \(q_{\mathrm{Na}}=+1\), \(q_{\mathrm{Cl}}=-1\) |
| \(r_c\) | 12.0 Å (Born and Coulomb) |

| Pair | \(A\) [eV] | \(\rho\) [Å] | \(\sigma\) [Å] | \(C\) [eV·Å⁶] | \(D\) [eV·Å⁸] |
| ---- | ---------: | -----------: | -------------: | ------------: | ------------: |
| Na–Na | 25.4435 | 0.317 | 2.340 | 101.1719 | 48.1771 |
| Cl–Cl | 25.4435 | 0.317 | 3.170 | 278.122 | 694.480 |
| Na–Cl | 20.3548 | 0.317 | 2.755 | 48.1771 | 48.1771 |

The parameter set is a regression table, not a production NaCl force-field
catalog; callers supply their own Fumi–Tosi tables as `BornTypeCoeff`.

### Usage

```cpp
const double d = 2.82;
const double rc = 12.0;

cpu::Frame f(4);
const double xs[4] = {10.0, 10.0 + d, 10.0 + d, 10.0};
const double ys[4] = {10.0, 10.0, 10.0 + d, 10.0 + d};
const int types[4] = {0, 1, 0, 1};
const double qs[4] = {1.0, -1.0, 1.0, -1.0};
const double masses[4] = {22.9898, 35.453, 22.9898, 35.453};
for (int i = 0; i < 4; ++i) {
    f.span(kAtom_x)[i] = xs[i];
    f.span(kAtom_y)[i] = ys[i];
    f.span(kAtom_z)[i] = 20.0;
    f.span(kAtom_type)[i] = types[i];
    f.span(kAtom_charge)[i] = qs[i];
    f.span(kAtom_mass)[i] = masses[i];
    f.span(kAtom_tag)[i] = i + 1;
}
f.box = cpu::Box::cubic(40.0);

cpu::PairBorn born(
    {{0, 0, 25.4435, 0.317, 2.340, 101.1719, 48.1771},
     {1, 1, 25.4435, 0.317, 3.170, 278.122, 694.480},
     {0, 1, 20.3548, 0.317, 2.755, 48.1771, 48.1771}},
    rc);

cpu::Evaluate eval;
eval.set_evaluator(std::move(born));
eval.set_evaluator(cpu::PairCoulCut{rc});
eval.set_neighbor(cpu::md::NeighborList<>{rc, /*skin=*/0.3});
eval.build();
eval.run(f);
```

The multi-type `PairBorn` walks all pairs; `PairCoulCut` uses the list.

### Validation (oracle: LAMMPS)

Twin: `lmp -in in.fumi_tosi_nacl4` (run 0; metal units).

| Quantity | Atomiverse / pin | LAMMPS | Tol | Notes |
| -------- | ---------------: | -----: | --- | ----- |
| \(E\) [eV] | `54.7707332302457` | same | \(1\times 10^{-9}\) eV | total PE = Born + Coulomb |
| \(E_{\mathrm{vdwl}}\) (Born) [eV] | — | `67.97442356628899` | diagnostic | LAMMPS thermo `evdwl` |
| \(E_{\mathrm{coul}}\) [eV] | — | `-13.20369033604329` | diagnostic | LAMMPS thermo `ecoul` |
| \(F_{0x}=F_{0y}\) (Na, atom 0) [eV/Å] | `-5.1254154966280602e+01` | same | \(1\times 10^{-8}\) eV/Å | planar symmetry |
| \(F_{1x}=-F_{1y}\) (Cl, atom 1) [eV/Å] | `+5.5205811735300870e+01` | same | \(1\times 10^{-8}\) eV/Å | |
| \(F_{iz}\) | `0` | `0` | \(1\times 10^{-8}\) eV/Å | planar square |
| Atom 2 force | \(-F_0\) | symmetry | \(1\times 10^{-8}\) eV/Å | opposite-corner Na |

The section asserts \(E\), \(F_0\), \(F_1\), zero \(F_z\) and the opposite-corner
Na force; the \(E_{\mathrm{vdwl}}\) / \(E_{\mathrm{coul}}\) split is recorded in
`REFERENCE.md` only.

### Limitations

- **Four ions** prove multi-type table + hybrid Born/Coulomb parity, not bulk
  molten-salt throughput or thermodynamics.
- **Cutoff Coulomb only.** No PME / `coul/long` in this pin.

### References

- LAMMPS `pair_style born`, `pair_style coul/cut`, `pair_style hybrid/overlay`
- F. G. Fumi, M. P. Tosi, *J. Phys. Chem. Solids* **25**, 31 (1964); M. P. Tosi, F. G. Fumi, *J. Phys. Chem. Solids* **25**, 45 (1964)
- Roadmap: `docs/roadmap/force-fields.md` (Born–Mayer–Huggins / Fumi–Tosi)

## steele-wall-argon — Steele 10-4-3 continuum wall, single Ar (LAMMPS `fix wall/lj1043`) {#steele-wall-argon}

### Summary

Steele 10-4-3 is the integrated Lennard-Jones continuum wall for a planar solid
surface (Steele 1973). `cpu::md::FixWallLJ1043` mirrors LAMMPS
`fix wall/lj1043` — not `wall/lj93`, which is the **9-3** integrated form; do
not compare against it. Energy is shifted so \(U(r_c)=0\); force is along the
wall normal only. The section is a one-atom Ar single point above a \(z\)-lo
wall at \(z=0\), pinned to the LAMMPS PE and \(F_z\).

### Method

For perpendicular distance \(r\) from the wall plane to a fluid-side atom, with
well depth \(\varepsilon\) and length \(\sigma\):

\[
U(r)=2\pi\varepsilon\left[
  \frac{2}{5}\left(\frac{\sigma}{r}\right)^{10}
  -\left(\frac{\sigma}{r}\right)^{4}
  -\frac{\sqrt{2}\,\sigma^{3}}{3\bigl(r+(0.61/\sqrt{2})\sigma\bigr)^{3}}
\right]-U(r_c),\qquad 0 < r < r_c.
\]

For \(r\ge r_c\), \(U=0\) and \(F=0\). Force follows LAMMPS `wall_particle`:
only the normal component is nonzero. A lo wall (`side = -1`) places fluid at
coordinates greater than `wall_pos`; \(F_z < 0\) at \(r=4\) Å is attraction
toward the wall on the equilibrium side of the well.

| Item | Setting |
| ---- | ------- |
| Atoms | 1 Ar (mass 39.948 amu) |
| Position | \((10, 10, 4)\) Å |
| Box | \(20\times 20\times 30\) Å orthorhombic |
| Wall | axis \(z\) (`axis=2`), lo (`side=-1`), `wall_pos=0` Å |
| \(\varepsilon\) | 0.010338 eV |
| \(\sigma\) | 3.40 Å |
| \(r_c\) | 12.0 Å |
| Pair | none (wall only; LAMMPS twin uses `pair_style zero`) |

LAMMPS twin: `fix wall/lj1043 zlo 0.0 0.010338 3.40 12.0`,
`fix_modify energy yes`, `pair_style zero`.

**Out of scope:** variable / moving wall position, several walls in one object,
lattice units, `fld yes` / field-only modes.

### Usage

```cpp
cpu::Frame f(1);
f.span(kAtom_x)[0] = 10.0;
f.span(kAtom_y)[0] = 10.0;
f.span(kAtom_z)[0] = 4.0;   // r = 4 Å above zlo wall at 0
f.span(kAtom_mass)[0] = 39.948;
f.span(kAtom_type)[0] = 0;
f.span(kAtom_tag)[0] = 1;
f.box = cpu::Box::orthorhombic(20.0, 20.0, 30.0);

cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate)
    .add(cpu::md::FixWallLJ1043{/*axis=*/2, /*side=*/-1, /*wall_pos=*/0.0,
                                /*epsilon [eV]=*/0.010338,
                                /*sigma [Å]=*/3.40,
                                /*cutoff [Å]=*/12.0});
eval.build();
eval.run(f);

const double E = f.meta.get(kEnergy);   // [eV]
const double fz = f.span(kAtom_fz)[0];  // [eV/Å]
```

On `cpu::md::MD` the component binds at `post_force` (+100), after
pair/bonded/kspace forces. Files: `fix_wall_lj1043.h`,
`src/cpu/dynamics/fix_wall_lj1043.cpp` (precompute + `launch` with the LAMMPS
coefficients).

### Validation (oracle: LAMMPS)

| Quantity | Atomiverse / pin | LAMMPS pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E\) [eV] | **−0.035247965699900359** | same | \(\lvert\Delta E\rvert\le 10^{-10}\) | |
| \(F_z\) [eV/Å] | **−0.025162258813859549** | same | \(\lvert\Delta F_z\rvert\le 10^{-10}\) | normal force |
| \(F_x, F_y\) [eV/Å] | **0** | 0 | \(\lvert F_{x,y}\rvert\le 10^{-12}\) | planar wall |
| Analytic vs LAMMPS dump | — | forces.dump %.16e | ≥ 1e-15 | independent closed-form check |

Cross-check heights (same \(\varepsilon,\sigma,r_c\); not asserted):

| \(r\) [Å] | \(E\) [eV] | \(F_z\) [eV/Å] |
| --------: | ---------: | -------------: |
| 3.0 | −2.892345800523042×10⁻² | +1.508268784423809×10⁻¹ |
| 4.0 | −3.524796569990036×10⁻² | −2.516225881385955×10⁻² |
| 5.0 | −1.687854854072667×10⁻² | −1.207703871038502×10⁻² |
| ≥ \(r_c\) | 0 | 0 |

### Limitations

- **Behind-wall atoms.** \(r\le 0\) raises `std::runtime_error` naming the axis,
  the wall position and the offending coordinate. The 10-4-3 form is defined
  only on \(0<r<r_c\) — it diverges as \(r\to0^+\) and turns attractive past its
  pole at \(r=-(0.61/\sqrt2)\sigma\) — so a behind-wall atom has no wall energy
  to report. Reduce the timestep or move the wall.

### References

- W. A. Steele, *Surf. Sci.* **36**, 317 (1973); LAMMPS `fix wall` / `wall/lj1043` documentation
- Roadmap: [`docs/roadmap/force-fields.md`](../../roadmap/force-fields.md) (Steele 10-4-3 / `FixWallLJ1043`)
