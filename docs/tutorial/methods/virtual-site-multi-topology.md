# General multi-topology virtual sites — TIP4P dimer + TIP5P geometry (E9)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `virtual-site-multi-topology` |
| Regression | `regressions/md/virtual-site-multi-topology/` |
| Public API | `cpu::md::VirtualSites` (+ nested `Place` / `Redistribute` Components); ordinary `PairLJCut` + `PairCoulCut` |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → **E9 done** (wave 3, 2026-07-24); map slice E9 molecular water |
| Oracle | **LAMMPS** `pair_style lj/cut/tip4p/cut` (metal units; offline pin). TIP5P dual-gated by geometry only |

## Summary

Massless **virtual sites** (TIP4P M-site, TIP5P lone pairs) are ordinary Frame
atoms whose positions are reconstructed each step from parent atoms, with
site forces redistributed onto parents (Feenstra / GROMACS-style) so total
force is conserved. Atomiverse exposes this as `cpu::md::VirtualSites` — a
table of constructions plus `place` / `redistribute` ops, bindable as nested
`Place` / `Redistribute` Components on `cpu::Evaluate` or `cpu::md::MD`. Pair
styles stay ordinary (`PairLJCut`, `PairCoulCut`, …): this is **not** a
specialized tip4p pair. The product regression
`reg_virtual-site-multi-topology` pins a **two-water TIP4P** single-point
energy and parent forces against the same offline LAMMPS metal twin used by
`four-site-tip4p-foothold` (different ATV path: general VirtualSites vs the
earlier tip4p foothold), and dual-gates **TIP5P** equilibrium geometry
(\(r_{\mathrm{OL}}\), \(\theta_{\mathrm{LOL}}\)) via `OutOfPlane3`. **Oracle
is LAMMPS** (feature overlap → LAMMPS priority; host binary has no
`tip5p` pair for a second energy pin). Fair wall
\(r = T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}} = 0.0837\) (within 10×).

## Method

### Weighted3 (TIP4P M-site)

Site position as a linear combination of oxygen and two hydrogens (min-image
parents relative to O, LAMMPS tip4p convention):

\[
\mathbf{r}_s
  =(1-\alpha)\,\mathbf{r}_0
  +\frac{\alpha}{2}\,\mathbf{r}_1
  +\frac{\alpha}{2}\,\mathbf{r}_2
  =\mathbf{r}_0+\frac{\alpha}{2}(\mathbf{d}_1+\mathbf{d}_2)
\]

with \(\mathbf{d}_i=\mathrm{min\_image}(\mathbf{r}_i-\mathbf{r}_0)\) and

\[
\alpha=\frac{q_{\mathrm{dist}}}{\cos(\theta_{\mathrm{HOH}}/2)\,r_{\mathrm{OH}}}
\]

(`VirtualSites::tip4p_alpha`). Force partition (constant Jacobian):

\[
\mathbf{F}_0\leftarrow\mathbf{F}_0+(1-\alpha)\mathbf{F}_s,\quad
\mathbf{F}_{1,2}\leftarrow\mathbf{F}_{1,2}+\tfrac{\alpha}{2}\mathbf{F}_s,\quad
\mathbf{F}_s\leftarrow\mathbf{0}.
\]

Classic TIP4P: charge on M and H, zero on O; LJ only on O–O; intramolecular
pairs excluded via `special_bonds` lj/coul 0 0 0 and O–H / O–M topology.

### OutOfPlane3 (TIP5P lone pair)

\[
\mathbf{r}_s=\mathbf{r}_0+a\,\mathbf{d}_1+b\,\mathbf{d}_2
  +c\,(\mathbf{d}_1\times\mathbf{d}_2)
\]

Equilibrium coefficients \((a,b,c)\) for Mahoney–Jorgensen TIP5P
(\(r_{\mathrm{OL}}=0.70\) Å, \(\theta_{\mathrm{LOL}}=109.47^\circ\), same
\(r_{\mathrm{OH}}\), \(\theta_{\mathrm{HOH}}\)) come from
`VirtualSites::tip5p_out_of_plane_coeffs` (`sign = ±1` selects L1 vs L2).
Force redistribute includes the cross-term Jacobian (GROMACS vsite3OUT).
This regression **geometry-gates** TIP5P only; no offline TIP5P energy pin
(host LAMMPS has no `tip5p` pair style).

### Twin physics (TIP4P energy/force)

Same single-point as `four-site-tip4p-foothold` / LAMMPS
`lj/cut/tip4p/cut`: LJ + Coulomb with tip4p charge geometry, energy shift on
LJ, special 0 0 0. ATV path: place M → neighbor + `PairLJCut` + `PairCoulCut`
→ redistribute M forces onto O/H.

| Item | Setting |
| ---- | ------- |
| Geometry | O–O = 2.8 Å along \(x\); water2 rotated \(\pi\) about \(z\) |
| \(r_{\mathrm{OH}}\) | 0.9572 Å |
| \(\theta_{\mathrm{HOH}}\) | 104.52° |
| \(q_{\mathrm{dist}}\) (OM) | 0.15 Å |
| \(q_M\), \(q_H\), \(q_O\) | −1.04 e, +0.52 e, 0 |
| \(\varepsilon_{\mathrm{OO}}\) | 0.1550 kcal/mol → eV via `kcal_per_mol_to_ev` |
| \(\sigma_{\mathrm{OO}}\) | 3.15365 Å |
| cut LJ = cut Coulomb | 10.0 Å |
| LJ shift | yes |
| special_bonds | lj/coul 0 0 0 |
| Box | cubic 20 Å, PBC |
| Frame layout | 8 atoms (O H H M × 2); types 0=O, 1=H, 2=M; M mass 0 |

**Key exclusions (non-goals of this slug)**

- Specialized `pair tip4p` / `lj/cut/tip4p/cut` component (sites + ordinary pairs instead).
- Live LAMMPS or MetalWalls inside ctest (offline pin only).
- Full TIP5P energy/force twin (no host `tip5p` pair → geometry dual-gate only).
- Rigid constraints (SETTLE/SHAKE) coupled to sites; production multi-residue
  GROMACS vsite catalog beyond Weighted3 / OutOfPlane3.
- Public `*Config` bags or `make_*` / `build_*` factories.
- MetalWalls as oracle (feature exists in LAMMPS → LAMMPS priority).

## Implementation

Public surface only (`cpu::Frame` + `VirtualSites` methods / nested Components;
no Config bag). Minimal Evaluate path matching the regression:

```cpp
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/virtual_sites.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_real.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_coul_cut.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

const double r_oh = 0.9572;
const double theta = 104.52 * units::kDegreeInRad;
const double qdist = 0.15;
const double eps = cpu::kcal_per_mol_to_ev(0.1550);
const double sigma = 3.15365;
const double cut = 10.0;
const double qM = -1.04, qH = 0.52;
const double alpha = cpu::md::VirtualSites::tip4p_alpha(qdist, r_oh, theta);

// Frame: 8 atoms O H H M × 2 at classic TIP4P dimer geometry (see regression);
// tags 1..8; types 0=O, 1=H, 2=M; bonds O–H, O–M + angle H–O–H; special list.
cpu::Frame f(8);
// ... fill x/y/z, type, mass, charge, tag; Box::cubic(20); rebuild_tag_index;
// ... add_bond / add_angle / build_special_list as in reg_*.cpp

cpu::md::VirtualSites vs;
vs.add_weighted3(/*M=*/4, /*O=*/1, /*H1=*/2, /*H2=*/3, alpha);
vs.add_weighted3(/*M=*/8, /*O=*/5, /*H1=*/6, /*H2=*/7, alpha);
vs.place(f);  // dual-gate: |OM − qdist| ≤ 1e-10 Å

cpu::md::VirtualSites::Place place{vs};
cpu::md::VirtualSites::Redistribute redist{vs};

cpu::PairLJCut lj{{{0, 0, eps, sigma}}, cut};
lj.set_special_bonds(0.0, 0.0, 0.0);
cpu::PairCoulCut coul{cut};
coul.set_special_bonds(0.0, 0.0, 0.0);

cpu::Evaluate eval;
eval.on(evaluate::Phase::prepare).add_ref(place, +100);
eval.set_neighbor(cpu::md::NeighborList<>{cut, 0.3, 1, 0, true});
eval.set_evaluator(std::move(lj));
eval.set_evaluator(std::move(coul));
eval.on(evaluate::Phase::finalize).add_ref(redist, -100);
eval.build();
eval.run(f);

const double E = f.meta.get(kEnergy);  // [eV]
// parent forces after Feenstra; site F = 0
```

TIP5P geometry dual-gate (no energy):

```cpp
double a1, b1, c1, a2, b2, c2;
cpu::md::VirtualSites::tip5p_out_of_plane_coeffs(
    /*r_ol=*/0.70, /*theta_lol=*/109.47 * units::kDegreeInRad,
    r_oh, theta, +1, a1, b1, c1);
cpu::md::VirtualSites::tip5p_out_of_plane_coeffs(
    0.70, 109.47 * units::kDegreeInRad, r_oh, theta, -1, a2, b2, c2);

cpu::md::VirtualSites tip5p;
tip5p.add_out_of_plane3(/*L1=*/4, /*O=*/1, /*H1=*/2, /*H2=*/3, a1, b1, c1);
tip5p.add_out_of_plane3(/*L2=*/5, 1, 2, 3, a2, b2, c2);
tip5p.place(w);  // gate r_OL and θ_LOL
```

MD binding (defaults on nested Components): `Place` at
`md::Phase::post_integrate` (+100) and `pre_force` (+50); `Redistribute` at
`post_force` (+50). Architecture rules:

- **Driver stays `cpu::Evaluate` / `cpu::md::MD`.** Virtual sites are
  place/redistribute Components + a table object — not a new Driver category.
- Sites are **Frame atoms**; pair evaluators never special-case tip4p geometry.
- No public Config bag; registration is `add_weighted3` / `add_out_of_plane3`.

Build and run (MD-only configure; CPU login pattern from `AGENTS.md` /
metalwalls loop):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_virtual-site-multi-topology -j"$(nproc)"
  ./build/regressions/reg_virtual-site-multi-topology
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/virtual_sites.h` | Public `VirtualSites`, `Place`, `Redistribute` |
| `src/cpu/dynamics/virtual_sites.cpp` | place / Feenstra redistribute / TIP4P α / TIP5P coeffs |
| `regressions/md/virtual-site-multi-topology/reg_virtual-site-multi-topology.cpp` | Product gate |
| `tests/cpu/dynamics/test_virtual_sites.cpp` | Unit coverage |

Offline LAMMPS twin (never in ctest):

```bash
# tools/metalwalls_oracle/inputs/lammps/virtual-site-multi-topology/
lmp -in in.tip4p_dimer_metal
```

## Validation (vs LAMMPS)

Source:
[`regressions/md/virtual-site-multi-topology/REFERENCE.md`](../../../regressions/md/virtual-site-multi-topology/REFERENCE.md)
(offline pin; **ctest never launches LAMMPS**). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).

| Field | Value |
| ----- | ----- |
| Oracle | **LAMMPS** |
| `oracle_version` | 30 Mar 2026 - Development (`develop` / `patch_30Mar2026-1074-g7f680de296`) |
| Twin | `tools/metalwalls_oracle/inputs/lammps/virtual-site-multi-topology/` (`in.tip4p_dimer_metal`, `data.tip4p_dimer_metal`, `log.metal`) |
| Units | metal (eV, eV/Å) |
| Command | `lmp -in in.tip4p_dimer_metal` (`run 0`) |

**Why LAMMPS (not MetalWalls):** four-site TIP4P pair physics also exists in
LAMMPS (`lj/cut/tip4p/cut`). Oracle priority binds LAMMPS for shared classical
MD; MetalWalls is reserved for MW-only electrode physics.

### Accuracy (hard-coded pins)

| Quantity | Atomiverse / pin | LAMMPS (offline) | Tol | Notes |
| -------- | ---------------: | ---------------: | --- | ----- |
| \(E\) [eV] | `0.52924228894955327451` | same | \(10^{-8}\) eV | total PE after place + LJ + Coulomb |
| OM [Å] | `0.15` (= qdist) | geometry | \(10^{-10}\) Å | after `place` |
| \(F_x,F_z\) parents O1 H1 H2 O2 H3 H4 [eV/Å] | see table below | same | \(10^{-6}\) eV/Å | after redistribute; \(F_y=0\) |
| Site force on M | \(\mathbf{0}\) | n/a (internal) | \(10^{-12}\) eV/Å | Feenstra residual |
| TIP5P \(r_{\mathrm{OL}}\) [Å] | `0.70` | geometry | \(10^{-8}\) Å | both lone pairs |
| TIP5P \(\theta_{\mathrm{LOL}}\) [rad] | \(109.47^\circ\) | geometry | \(10^{-8}\) rad | no energy pin |

Parent force pin (LAMMPS metal; atoms 1–6 = O1 H1 H2 O2 H3 H4):

| atom | \(f_x\) [eV/Å] | \(f_z\) [eV/Å] |
| ---- | -------------: | -------------: |
| 1 O | −0.094382238074752045076 | 0.33187666441899288383 |
| 2 H | −1.0593571566097745862 | −0.31522207906906996033 |
| 3 H | −0.057823257196248978307 | −0.016654585349923187176 |
| 4 O | 0.094382238074751600987 | 0.33187666441899305037 |
| 5 H | 1.0593571566097748082 | −0.31522207906906962727 |
| 6 H | 0.057823257196249200351 | −0.01665458534992322534 |

## Performance

Source: `regressions/md/virtual-site-multi-topology/REFERENCE.md`,
`docs/developer/metalwalls-method-handoff.md` (handoff rounds ratio to **0.084**).

Fair wall: ATV `reg_virtual-site-multi-topology` vs LAMMPS
`in.tip4p_dimer_metal` / `run 0`, serial `OMP_NUM_THREADS=1`, 50 process
launches after 3 warmups on x86_64 login (Release MD-only build).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | -----------------------------------------: | ---- | ------ |
| TIP4P dimer single-point (`run 0`) | 0.1260 | 0.01055 | **0.0837** | ✓ | process-launch mean; small system |

```text
perf_exempt: false
ratio_atv_over_oracle: 0.0837
within_10x: true
method: 50 launches after 3 warmups; OMP_NUM_THREADS=1
```

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}} \le 10\) unless exempted —
satisfied (\(r \approx 0.084\)).

## Limitations

- **Foothold system.** Two rigid waters prove multi-topology place + Feenstra
  + ordinary pairs vs LAMMPS tip4p energy/forces, not bulk aqueous MD
  throughput or electrode–water production twins.
- **TIP5P energy unpinned.** Host LAMMPS lacks `tip5p` pair; only
  \(r_{\mathrm{OL}}\) / \(\theta_{\mathrm{LOL}}\) dual-gate. Full five-site
  energy twin needs a different oracle binary or literature pin.
- **Construction catalog.** Only Weighted3 and OutOfPlane3 in this wave;
  further GROMACS-style vsite kinds (2-body, 4-body, COG, …) are follow-on.
- **No specialized tip4p pair.** Callers must register sites, topology for
  special exclusions, and ordinary LJ/Coulomb; mis-topology silently changes
  intramolecular exclusions.
- **Not MetalWalls-oracle.** Shared classical MD → LAMMPS priority (E9
  molecular-water map slice still documents electrode context; this slug is
  the multi-topology engine, not electrode coupling).

## References

- Offline pin: [`regressions/md/virtual-site-multi-topology/REFERENCE.md`](../../../regressions/md/virtual-site-multi-topology/REFERENCE.md)
- Regression: `regressions/md/virtual-site-multi-topology/reg_virtual-site-multi-topology.cpp`
- LAMMPS twin: `tools/metalwalls_oracle/inputs/lammps/virtual-site-multi-topology/`
- Related foothold: `regressions/md/four-site-tip4p-foothold/` (same physics pin, earlier ATV path)
- Queue / handoff: `docs/developer/metalwalls-method-queue.md` (E9),
  `docs/developer/metalwalls-method-handoff.md` (wave 3, ratio **0.084**)
- Roadmap: `docs/roadmap/molecular-dynamics.md` (virtual sites / TIP4P–TIP5P)
- Public API: `include/atomiverse/cpu/dynamics/virtual_sites.h`
- Capability map: `.claude/notes/metalwalls-capability-map.md` → **E9**
- LAMMPS: `pair_style lj/cut/tip4p/cut` (metal)
- TIP4P: Jorgensen et al., *J. Chem. Phys.* **79**, 926 (1983); *J. Chem. Phys.* **103**, 10252 (1985 context)
- TIP5P: Mahoney & Jorgensen, *J. Chem. Phys.* **112**, 8910 (2000)
- Virtual-site force partition: Feenstra et al., *J. Comput. Chem.* **20**, 786 (1999); GROMACS vsite machinery
