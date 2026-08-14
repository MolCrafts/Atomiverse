# Four-site TIP4P (M virtual site) — water dimer foothold

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `four-site-tip4p-foothold` |
| Regression | `regressions/md/four-site-tip4p-foothold/` |
| Public API | `cpu::PairLJCutTip4pCut` + `cpu::Evaluate` + `cpu::md::NeighborList` |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → **done** (wave 1, 2026-07-23); MetalWalls map **E9** |
| Oracle | **LAMMPS** `pair_style lj/cut/tip4p/cut` (not MetalWalls) |

## Summary

Classic **Jorgensen TIP4P** four-site water: LJ on oxygen, Coulomb on a
**massless M site** along the HOH bisector, with **Feenstra** redistribution of
M-site forces onto O/H so molecular force and torque are preserved. Atomiverse
exposes this as the pair potential component `cpu::PairLJCutTip4pCut` (LAMMPS
name `lj/cut/tip4p/cut`), bound into the ordinary force pipeline of
`cpu::Evaluate` (or `cpu::md::MD`) — **not** a new Driver category. This slug is
the **CPU virtual-site foothold**: a two-water **single-point** energy and force
check against an **offline LAMMPS** pin (units metal = eV). Product gate
`reg_four-site-tip4p-foothold` asserts energy, M-site geometry, and six-atom
forces. Fair wall ratio \(T_{\mathrm{ATV}}/T_{\mathrm{LAMMPS}} = \mathbf{0.338}\)
(within 10×).

## Method

**TIP4P electrostatics.** Each water has three mass sites (O, H1, H2) and one
massless charge site M. With equilibrium geometry \(r_{\mathrm{OH}}\),
\(\theta_{\mathrm{HOH}}\), and O→M distance \(q_{\mathrm{dist}}\):

\[
\mathbf{r}_M = \mathbf{r}_O + \alpha\,\bigl((\mathbf{r}_{H1}-\mathbf{r}_O)
  + (\mathbf{r}_{H2}-\mathbf{r}_O)\bigr),\qquad
\alpha = \frac{q_{\mathrm{dist}}}{\cos(\theta/2)\,r_{\mathrm{OH}}}.
\]

Oxygen nominal charge \(q_O\) is evaluated at \(\mathbf{r}_M\); hydrogens keep
\(q_H\). LJ acts only on O–O (H–H and O–H LJ zero in the classic model).

**Force partition (Feenstra).** Coulomb forces computed at M are projected back
to O, H1, H2 so that total force and torque on the rigid water are unchanged
(Feenstra, *J. Comput. Chem.* **20**, 786 (1999) — same construction LAMMPS
uses in `pair_lj_cut_tip4p_cut`).

**Pair cutoffs.** This foothold uses **cutoff** LJ and **cutoff** Coulomb
(`lj/cut/tip4p/cut`), not long-range PME/PPPM TIP4P variants. LJ is shifted so
\(V(r_c)=0\) (LAMMPS `pair_modify shift yes`). Special-bond scales are
`lj/coul 0 0 0` (exclude 1–2, 1–3, 1–4 entirely).

**Pinned system parameters** (classic TIP4P 1983/85):

| Item | Value |
| ---- | ----- |
| \(r_{\mathrm{OH}}\) | 0.9572 Å |
| \(\theta_{\mathrm{HOH}}\) | 104.52° |
| \(q_{\mathrm{dist}}\) (OM) | 0.15 Å |
| \(q_O\) (on M) | −1.04 e |
| \(q_H\) | +0.52 e |
| \(\varepsilon_{\mathrm{OO}}\) | 0.1550 kcal/mol → eV via `kcal_per_mol_to_ev` |
| \(\sigma_{\mathrm{OO}}\) | 3.15365 Å |
| \(r_{c,\mathrm{LJ}} = r_{c,\mathrm{Coul}}\) | 10.0 Å |
| Geometry | O–O = 2.8 Å along \(x\); second water rotated \(\pi\) about \(z\) |
| Box | cubic 20 Å, periodic |
| Topology | O–H bonds + H–O–H angles; tags LAMMPS tip4p order (H of O tag \(t\) are \(t{+}1\), \(t{+}2\)) |

**What this pin includes**

- Single-point energy + forces on a 6-atom dimer.
- M-site reconstruction check \(\lvert\mathrm{OM}-q_{\mathrm{dist}}\rvert\le 10^{-10}\) Å.
- Offline LAMMPS pin only (no live third-party at ctest).

**Key exclusions (non-goals of this slug)**

- General multi-topology virtual sites beyond rigid TIP4P water (queue follow-on).
- `lj/cut/tip4p/long` / PME TIP4P, TIP4P/2005, TIP4P/Ice, TIP5P.
- Rigid SHAKE/RATTLE constraints, full MD trajectory, or continuum solvent.
- MetalWalls as oracle (LAMMPS owns this pair style).
- Public `*Config` bags / `make_*` factories.

## Implementation

Public surface: constructor arguments on `PairLJCutTip4pCut` (no Config bag);
`Evaluate` + neighbor list for a single-point force phase. Frame carries O/H
types, charges, tags, bonds/angles, and special list.

```cpp
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/io/lammps_real.h"   // kcal_per_mol_to_ev
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/potentials/pair_lj_cut_tip4p_cut.h"

using namespace ATV;

// Classic TIP4P (Jorgensen 1983/85); types 0=O, 1=H
const double r_oh  = 0.9572;
const double theta = 104.52 * units::kDegreeInRad;
const double qdist = 0.15;
const double eps   = cpu::kcal_per_mol_to_ev(0.1550);
const double sigma = 3.15365;
const double cut   = 10.0;

// Build 6-atom dimer Frame (O–O = 2.8 Å), set type/mass/charge/tag,
// add O–H bonds + H–O–H angles, rebuild special list — see regression.

cpu::PairLJCutTip4pCut tip4p(/*type_O=*/0, /*type_H=*/1, qdist, r_oh, theta,
                             eps, sigma, cut, cut);
tip4p.set_special_bonds(0.0, 0.0, 0.0);

cpu::Evaluate eval;
eval.set_neighbor(
    cpu::md::NeighborList<>{tip4p.neighbor_cutoff(), 0.3, 1, 0, true});
eval.set_evaluator(std::move(tip4p));
eval.build();
eval.run(frame);

const double E = frame.meta.get(kEnergy);   // [eV]
// forces: frame.span(kAtom_fx/y/z)
```

```bash
# MD-only configure (this loop — do not enable HF/PW/GPW/CUDA)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_four-site-tip4p-foothold -j$(nproc)
'
./build/regressions/reg_four-site-tip4p-foothold
```

Relevant headers:

- `include/atomiverse/cpu/potentials/pair_lj_cut_tip4p_cut.h`
- `include/atomiverse/cpu/evaluate/evaluate.h`
- `include/atomiverse/cpu/neighbor/neighbor_list.h`
- Regression: `regressions/md/four-site-tip4p-foothold/reg_four-site-tip4p-foothold.cpp`

## Validation (vs LAMMPS)

**Oracle: LAMMPS**, not MetalWalls. Offline pin only; never live `lmp` in ctest.

Source: `regressions/md/four-site-tip4p-foothold/REFERENCE.md`  
Oracle version: **30 Mar 2026 - Development** (`develop` /
`patch_30Mar2026-1074-g7f680de296`)  
Inputs: `tools/metalwalls_oracle/inputs/lammps/four-site-tip4p-foothold/`
(`in.tip4p_dimer_metal`, `data.tip4p_dimer_metal`, `log.metal`)

| Quantity | Atomiverse / pin | LAMMPS pin | Tol | Notes |
| -------- | ---------------: | ---------: | --- | ----- |
| \(E\) [eV] | `0.52924228894955327451` | same (hard-coded) | \(10^{-8}\) eV | total PE, metal units |
| \(E_{\mathrm{vdwl}}\) [eV] | — | `0.057185016777344513383` | (report only) | LJ portion of pin log |
| \(E_{\mathrm{coul}}\) [eV] | — | `0.4720572721722087195` | (report only) | Coulomb portion |
| OM geometry [Å] | reconstructed \(= q_{\mathrm{dist}}\) | \(q_{\mathrm{dist}}=0.15\) | \(10^{-10}\) Å | dual-gate |
| \(f_x, f_z\) (6 atoms) [eV/Å] | match pin table | see REFERENCE | \(10^{-6}\) eV/Å per component | \(f_y = 0\) within \(10^{-10}\) |

Pin forces (atom order O1 H1 H2 O2 H3 H4), \(f_y = 0\):

| atom | \(f_x\) [eV/Å] | \(f_z\) [eV/Å] |
| ---- | -------------: | -------------: |
| 1 O | −0.094382238074752045076 | 0.33187666441899288383 |
| 2 H | −1.0593571566097745862 | −0.31522207906906996033 |
| 3 H | −0.057823257196248978307 | −0.016654585349923187176 |
| 4 O | 0.094382238074751600987 | 0.33187666441899305037 |
| 5 H | 1.0593571566097748082 | −0.31522207906906962727 |
| 6 H | 0.057823257196249200351 | −0.01665458534992322534 |

Regenerate pin (operator host with LAMMPS):

```bash
cd tools/metalwalls_oracle/inputs/lammps/four-site-tip4p-foothold
lmp -in in.tip4p_dimer_metal -log log.metal
```

## Performance

Source: `regressions/md/four-site-tip4p-foothold/REFERENCE.md` (Perf result).

Fair wall: same dimer single-point — ATV `reg_four-site-tip4p-foothold` vs
LAMMPS `in.tip4p_dimer_metal` / `run 0`. Both **serial**
(`OMP_NUM_THREADS=1`); 50 process launches after 3 warmups; wall includes
process start + build. Host: x86_64 login, Release MD-only build.
`perf_exempt = false`.

| Twin | \(T_{\mathrm{LAMMPS}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | -------------------------: | ---------------------: | -----------------------------------------: | ---- | ------ |
| TIP4P dimer SP | 0.1548 | 0.0523 | **0.338** | ✓ | process-launch fair wall, not in-process kernel microbench |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}} \le 10\) unless exempted — **pass**.

## Limitations

- **Foothold only:** rigid classic TIP4P dimer energy/force; not a production
  water library (no parameter database, no automatic molecule templates).
- **Tag convention:** H atoms of oxygen with tag \(t\) must be \(t{+}1\) and
  \(t{+}2\) (LAMMPS tip4p convention).
- **No long-range TIP4P** in this slug; charged-slab / PME water needs separate
  map work and a different pair style.
- **General virtual sites** (arbitrary 2/3/4-site constructions, out-of-plane M)
  remain open on the MetalWalls/classical MD queue.
- Fair wall ratio is dominated by process startup on a tiny system; do not
  over-interpret as bulk MD throughput.
- Oracle is **LAMMPS**; MetalWalls is not used for this pair style.

## References

- Offline pin: `regressions/md/four-site-tip4p-foothold/REFERENCE.md`
- Pin notes: `tools/metalwalls_oracle/pin_notes.md` (`four-site-tip4p-foothold`)
- Oracle inputs: `tools/metalwalls_oracle/inputs/lammps/four-site-tip4p-foothold/`
- Queue / handoff: `docs/developer/metalwalls-method-queue.md`,
  `docs/developer/metalwalls-method-handoff.md`
- Design: `docs/developer/metalwalls-method-loop.md`
- Map: `.claude/notes/metalwalls-capability-map.md` (E9)
- W. L. Jorgensen *et al.*, *J. Chem. Phys.* **79**, 926 (1983); TIP4P
  geometry/parameters as used by LAMMPS `lj/cut/tip4p/cut`
- B. R. Brooks *et al.* / Feenstra force partition:
  Feenstra, *J. Comput. Chem.* **20**, 786 (1999)
- LAMMPS doc: `pair_style lj/cut/tip4p/cut`
