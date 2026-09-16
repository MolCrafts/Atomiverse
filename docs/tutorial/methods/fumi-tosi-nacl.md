# Fumi–Tosi multi-type Born + Coulomb — NaCl four-ion square

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `fumi-tosi-nacl` |
| Regression | `regressions/forcefield/fumi-tosi-nacl/` |
| Public API | `cpu::PairBorn` (multi-type table) + `cpu::PairCoulCut` on `cpu::Evaluate` |
| Roadmap | map salt/wall FF |
| Oracle | **LAMMPS** `pair_style hybrid/overlay born coul/cut` (`pair_modify pair born shift yes`) |

## Summary

**Fumi–Tosi** short-range ion–ion interactions for molten salts / alkali
halides are the Born–Mayer–Huggins (BMH) pair
\(A e^{(\sigma-r)/\rho}-C/r^{6}+D/r^{8}\) with **per type-pair** coefficients
(no Lorentz–Berthelot mixing) plus a separate Coulomb term. Atomiverse
implements BMH as `cpu::PairBorn` (LAMMPS `pair_style born`, energy shift at
cutoff always on) and Coulomb as `cpu::PairCoulCut` (LAMMPS `coul/cut`,
unshifted). The product regression
`reg_fumi-tosi-nacl` is a **four-ion planar NaCl square** single-point
energy/force pin against offline LAMMPS metal units
(`hybrid/overlay born + coul/cut`). Oracle is **LAMMPS**, not MetalWalls —
the feature is classical pair style that also exists in LAMMPS. Fair wall is
**`perf_exempt`**: four-atom startup noise dominates both engines; bulk
NeighborList-backed molten-salt twin deferred.

## Method

### Born–Mayer–Huggins (shifted)

Matches LAMMPS `pair_style born` with `pair_modify pair born shift yes`:

\[
E_{\mathrm{born}}(r)=A\,e^{(\sigma-r)/\rho}-\frac{C}{r^{6}}+\frac{D}{r^{8}}-E(r_c),
\quad r < r_c
\]

with
\(E(r_c)=A e^{(\sigma-r_c)/\rho}-C/r_c^{6}+D/r_c^{8}\) so \(E_{\mathrm{born}}(r_c)=0\).
`pair_coeff` order is \(A\), \(\rho\), \(\sigma\), \(C\), \(D\) (metal: eV, Å).

Force uses the unshifted derivative (shift is constant):

\[
\frac{\mathrm{d}U}{\mathrm{d}r}=-\frac{A}{\rho}\,e^{(\sigma-r)/\rho}+\frac{6C}{r^{7}}-\frac{8D}{r^{9}}.
\]

### Coulomb (unshifted cut)

\[
E_{\mathrm{coul}}(r)=\mathrm{qqr2e}\,\frac{q_i q_j}{r},\quad
\mathrm{qqr2e}=14.399645\ \mathrm{eV\cdotÅ}
\]

(same as Atomiverse `units::coulomb` / LAMMPS metal `coul/cut`).

### Twin composition

```text
pair_style hybrid/overlay born 12.0 coul/cut 12.0
pair_modify pair born shift yes
```

Atomiverse: multi-type `cpu::PairBorn{coeffs, rc}` + `cpu::PairCoulCut{rc}`.
Unlisted type pairs do **not** interact under BMH (no silent mixing).

### Geometry / parameters (this pin)

| Field | Value |
| ----- | ----- |
| Atoms | 4 (2 Na + 2 Cl planar square) |
| Edge \(d\) | 2.82 Å |
| Positions [Å] | Na(10,10,20), Cl(12.82,10,20), Na(12.82,12.82,20), Cl(10,12.82,20) |
| Box | cubic 40 Å, PBC |
| Types | Na = frame type 0 / LAMMPS type 1; Cl = 1 / 2 |
| Charges | \(q_{\mathrm{Na}}=+1\), \(q_{\mathrm{Cl}}=-1\) |
| \(r_c\) | 12.0 Å (Born and Coulomb) |
| Born shift | yes (always on in `PairBorn`) |

### Multi-type Born table (metal)

| Pair | \(A\) [eV] | \(\rho\) [Å] | \(\sigma\) [Å] | \(C\) [eV·Å⁶] | \(D\) [eV·Å⁸] |
| ---- | ---------: | -----------: | -------------: | ------------: | ------------: |
| Na–Na | 25.4435 | 0.317 | 2.340 | 101.1719 | 48.1771 |
| Cl–Cl | 25.4435 | 0.317 | 3.170 | 278.122 | 694.480 |
| Na–Cl | 20.3548 | 0.317 | 2.755 | 48.1771 | 48.1771 |

**Key exclusions (non-goals of this slug)**

- Production NeighborList-backed bulk molten-salt MD twin (current `PairBorn`
  is O(N²) all-pairs; `as_neighbor_consumer()` returns null — same foothold
  shape as `PairBuck` / single-type Born).
- Long-range Coulomb (`coul/long` + PME/`KSpace*`) — pin uses `coul/cut` only.
- MetalWalls as oracle (feature exists in LAMMPS → LAMMPS priority).
- Live LAMMPS inside ctest (offline pin only).
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame`, potential ctor args; no Config bag). Minimal
single-point matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_born.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"

using namespace ATV;

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

const double E = f.meta.get(kEnergy);  // [eV]
// forces: f.span(kAtom_fx), kAtom_fy, kAtom_fz
```

Headers: `include/atomiverse/cpu/potentials/pair_born.h`,
`pair_coul_cut.h`. Regression source:
`regressions/forcefield/fumi-tosi-nacl/reg_fumi-tosi-nacl.cpp`.

```bash
# MD-only configure (see ) then:
cmake --build build --target reg_fumi-tosi-nacl -j$(nproc)
./build/regressions/reg_fumi-tosi-nacl
```

Offline LAMMPS twin (never in ctest):

```bash
# tools/metalwalls_oracle/inputs/lammps/in.fumi_tosi_nacl4
lmp -in in.fumi_tosi_nacl4
```

## Validation (vs LAMMPS)

Source: `regressions/forcefield/fumi-tosi-nacl/REFERENCE.md` (offline pin;
**ctest never launches LAMMPS**).

| Field | Value |
| ----- | ----- |
| Oracle | **LAMMPS** |
| `oracle_version` | 30 Mar 2026 - Development (`develop` / `patch_30Mar2026-1074-g7f680de296`) |
| Twin input | `tools/metalwalls_oracle/inputs/lammps/in.fumi_tosi_nacl4` |
| Command | `lmp -in in.fumi_tosi_nacl4` (run 0; metal units) |
| Pin date | 2026-07-24 |

| Quantity | Atomiverse / pin | LAMMPS | Tol | Notes |
| -------- | ---------------: | -----: | --- | ----- |
| \(E\) [eV] | `54.7707332302457` | same (offline) | \(1\times 10^{-9}\) eV | total PE = Born + Coulomb |
| \(E_{\mathrm{vdwl}}\) (Born) [eV] | — | `67.97442356628899` | diagnostic | LAMMPS thermo `evdwl` |
| \(E_{\mathrm{coul}}\) [eV] | — | `-13.20369033604329` | diagnostic | LAMMPS thermo `ecoul` |
| \(F_{0x}=F_{0y}\) (Na, atom 0) [eV/Å] | `-5.1254154966280602e+01` | same | \(1\times 10^{-8}\) eV/Å | planar symmetry |
| \(F_{1x}=-F_{1y}\) (Cl, atom 1) [eV/Å] | `+5.5205811735300870e+01` | same | \(1\times 10^{-8}\) eV/Å | |
| \(F_{iz}\) | `0` | `0` | \(1\times 10^{-8}\) eV/Å | planar square |
| Atom 2 force | \(-F_0\) | symmetry | \(1\times 10^{-8}\) eV/Å | opposite corner Na |

Regression gates hard-code \(E\), \(F_0\), \(F_1\), zero \(F_z\), and opposite
corner Na force; component split \(E_{\mathrm{vdwl}}\) / \(E_{\mathrm{coul}}\)
is recorded in REFERENCE only.

## Performance

Source: `regressions/forcefield/fumi-tosi-nacl/REFERENCE.md`,
the case `REFERENCE.md`.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | -----------------------------------------: | ---- | ------ |
| 4-ion single-point Born+Coul | n/a | n/a | **perf_exempt** | exempt | wall dominated by process startup, not pair kernel |

```text
perf_exempt: true
perf_exempt_reason: 4-ion single-point pin; wall dominated by process startup, not pair kernel. Fair bulk NL twin deferred with production NeighborList-backed PairBorn.
ratio_atv_over_oracle: n/a
within_10x: exempt
```

Optional future microbench (not a ctest gate): when a production NL multi-type
`PairBorn` lands, re-measure vs LAMMPS `born` (+ `coul/cut` or `coul/long`) on
a molten-salt box.

## Limitations

- **Foothold size.** Four ions prove multi-type table + hybrid Born/Coulomb
  parity, not bulk molten-salt throughput or thermodynamics.
- **O(N²) pair walk.** `PairBorn` currently withdraws the neighbor-list gate
  (`as_neighbor_consumer() == nullptr`); Coulomb may still use the list. Bulk
  NL-backed BMH is roadmap follow-on.
- **Cutoff Coulomb only.** No PME / `coul/long` in this pin; electrolyte /
  electrode long-range stack is separate Electrochem map work.
- **No MetalWalls comparison.** Oracle priority binds LAMMPS for shared pair
  styles; MW would only apply for MW-only electrode physics.
- **Parameter set is textbook/regression, not a production NaCl force field
  catalog** — callers supply their own Fumi–Tosi tables as `BornTypeCoeff`.

## References

- Offline pin: [`regressions/forcefield/fumi-tosi-nacl/REFERENCE.md`](../../../regressions/forcefield/fumi-tosi-nacl/REFERENCE.md)
- Regression: `regressions/forcefield/fumi-tosi-nacl/reg_fumi-tosi-nacl.cpp`
- LAMMPS twin: `tools/metalwalls_oracle/inputs/lammps/in.fumi_tosi_nacl4`
- Pin notes: `tools/metalwalls_oracle/pin_notes.md` → `fumi-tosi-nacl`
- Roadmap: `docs/roadmap/force-fields.md` (Born–Mayer–Huggins / Fumi–Tosi)
- Public API: `include/atomiverse/cpu/potentials/pair_born.h`,
  `pair_coul_cut.h`
- Single-type Born foothold: `regressions/forcefield/born-repulsion-foothold/`
- LAMMPS doc: `pair_style born`, `pair_style coul/cut`, `pair_style hybrid/overlay`
- Fumi–Tosi / BMH context: Fumi & Tosi, *J. Phys. Chem. Solids* **25**, 31 (1964);
  Tosi & Fumi, *J. Phys. Chem. Solids* **25**, 45 (1964)
