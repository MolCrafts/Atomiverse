# Constant-potential CG electrode solve — multi-site plate twin

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `const-pot-cg-large-electrode` |
| Regression | `regressions/md/const-pot-cg-large-electrode/` |
| Public API | `cpu::md::ConstPotCG` (Component on `cpu::md::MD`, default `pre_force`) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → **E2 done** (wave 2, 2026-07-24); map slice E2 |
| Oracle | **MetalWalls** algorithm class (Jacobi-PCG electrode charge solve); dual numeric twin = in-tree `cpu::md::ConstPotBO` |

## Summary

**Constant-potential** electrode electrostatics for **large multi-site walls**:
same Siepmann–Sprik Gaussian elastance \(A\) and electrolyte RHS \(b\) as
`ConstPotBO`, but the linear system \(A Q = \Psi - b\) (plus electroneutrality
projector) is solved with **Jacobi-preconditioned conjugate gradient** instead
of a dense inverse. Public surface is `cpu::md::ConstPotCG` — a **Component**
bound at `md::Phase::pre_force` on `cpu::md::MD` (not a Driver category; not
inside `KSpace*`). It writes solved charges to `kAtom_charge` before
pair / k-space force. This regression is the **E2 CG foothold**: a \(4\times 4\)
planar capacitor per wall (\(N_{\mathrm{elec}}=32\)) with dual-twin correctness
vs dense BO on the **identical** \(A\), \(\eta\), \(\Psi\), and neutrality
projector. Oracle tag is **metalwalls** (MW-style iterative electrode path for
large \(N\)); no live MetalWalls binary is required at ctest. Fair wall vs an
external MW binary is **`perf_exempt`**.

## Method

**Physics.** Same BO electrode model as E1 (`const-pot-bo-parallel-plate`):
Gaussian electrode charges of width \(\eta\) [1/Å]. Elastance matrix elements
(metal units, \(k_e =\) `units::coulomb` \(= 14.399645\) eV·Å/e²):

\[
A_{ii} = k_e\,\eta\sqrt{\frac{2}{\pi}}
  + \sum_{(n_x,n_y)\neq(0,0)} k_e\,\frac{\operatorname{erf}\!\bigl((\eta/\sqrt{2})\,r_{n}\bigr)}{r_{n}},
\quad
A_{ij} = \sum_{n_x,n_y} k_e\,\frac{\operatorname{erf}\!\bigl(\eta_{ij}\,r_{ij,n}\bigr)}{r_{ij,n}}
\]

with \(\eta_{ij} = \eta_i\eta_j / \sqrt{\eta_i^2+\eta_j^2}\) (v1 shares one
\(\eta\) for all electrode sites). Electrolyte contribution:

\[
b_i = \sum_{j\notin\mathrm{elec}} \sum_{n_x,n_y}
  k_e\,q_j\,\frac{\operatorname{erf}(\eta\,r_{ij,n})}{r_{ij,n}}.
\]

Unconstrained solve \(A Q = \Psi - b\). With **electroneutrality**
(`electroneutrality = true`), charges use the same neutrality projector as
dense BO / LAMMPS ELECTRODE `symm on` / MetalWalls BO so \(\sum_i Q_i = 0\).

**Solver difference (this slug).** Dense BO forms \(A^{-1}\) (Cholesky / SPD
inverse). `ConstPotCG` keeps the dense row-major \(A\) and applies
**Jacobi-PCG** (`pcg_solve_`) with relative residual tolerance `cg_tol` and
cap `max_cg_iter`. Same physics matrix → CG and BO site charges must agree to
machine-relevant tolerances when \(A\) and \(b\) match.

**v1 lattice treatment (this pin).** Regression uses
`xy_image_range = 0` (primary cell / min-image only) so multi-site wall
assembly stays cheap; this is **not** the E1 madelung image-range-250 pin
vs full `ewald/electrode`. Primary-cell sanity still pins
\(Q_{\mathrm{top}}\) on the 2-electrode madelung geometry to the dense-BO
primary value **0.27175356** e.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Geometry (large) | Bottom/top walls: \(4\times 4\) grids at \(z=\{8,12\}\) Å; electrolyte \(\pm 0.5\,e\) at box-center \(xy\), \(z=\{9,11\}\) Å |
| \(N_{\mathrm{elec}}\) | **32** (16 + 16); gate \(\ge 18\) |
| Box | \(L_x=L_y=4\) Å, \(L_z=20\) Å, orthorhombic, fully periodic ATV |
| Applied \(\Psi\) | \(\Psi_{\mathrm{bot}}=0\) V, \(\Psi_{\mathrm{top}}=1\) V |
| Gaussian width | \(\eta = 2.0\) Å⁻¹ |
| Neutrality | on (`symm`) |
| Image range | `xy_image_range = 0` |
| CG controls | `cg_tol = 1e-14` (standalone twin); `max_cg_iter = 2000` |
| Binding | standalone `build`/`launch`, and MD `add_ref` → default `pre_force` @ −50 |

**Key exclusions (non-goals of this slug)**

- Live MetalWalls or LAMMPS ELECTRODE binary in ctest.
- Full reciprocal electrode Ewald / MW 2D Ewald matrix (related E2:
  `slab-2d-ewald-foothold`).
- ASPC charge warm-start (`electrode-charge-predictor` sibling).
- Thomas–Fermi metallicity, finite-field single electrode, piston \(P_z\)
  (other E-series slugs).
- Charge solve inside PME/`KSpace*` — forbidden by architecture.
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + `ConstPotCG` ctor arg / members; no Config
bag). Minimal multi-site solve matching the regression twin:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/const_pot_cg.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

// frame: 32 electrode sites (types 1,2) + 2 electrolyte ions (type 3);
// box orthorhombic(4, 4, 20); see reg_const-pot-cg-large-electrode.cpp
// Component defaults to pre_force order −50.

cpu::md::ConstPotCG cg(/*eta [1/Å]=*/2.0);
cg.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
cg.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
cg.electroneutrality = true;
cg.xy_image_range = 0;
cg.cg_tol = 1e-14;
cg.max_cg_iter = 2000;
cg.rebuild_matrix_every_step = true;  // optional if geometry moves

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/12.0, /*skin=*/0.5});
sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/12.0});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.1});
sim.add_ref(cg);   // or add(std::move(cg)) to own
sim.build();
sim.run(frame, /*nsteps=*/1);

const double q_bot = cg.total_charge(0);  // [e]
const double q_top = cg.total_charge(1);  // [e]
const int iters = cg.cg_iterations();
// frame.span(kAtom_charge)[0..N_elec-1] updated; electrolyte unchanged
```

Dual-twin check (binding accuracy gate — same \(A\)): run the dense BO path
on a second Frame with the same MD composition:

```cpp
cpu::md::ConstPotBO bo(/*eta=*/2.0);
bo.add_electrode(1, 0.0).add_electrode(2, 1.0);
bo.electroneutrality = true;
bo.xy_image_range = 0;

cpu::md::MD sim_bo;
// ... same neighbor / coulomb / integrator as above ...
sim_bo.add_ref(bo);
sim_bo.build();
sim_bo.run(frame_bo, /*nsteps=*/1);
// assert max_i |Q_i^CG - Q_i^BO| <= 1e-8 e; group totals match
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** Electrodes are Components; never a new
  Driver category.
- Parameters are **constructor arguments and public members** — no
  `ConstPotCGConfig`.
- Frame is the sole I/O currency; outputs land on `kAtom_charge`.
- Dense BO remains the small-\(N\) / dual-twin path; CG is the large-\(N\)
  product API for this slug.

Build and run the regression (MD-only configure; CPU login pattern from
`AGENTS.md` / metalwalls loop):

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
  cmake --build build --target reg_const-pot-cg-large-electrode -j"$(nproc)"
  ./build/regressions/reg_const-pot-cg-large-electrode
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/const_pot_cg.h` | Public `ConstPotCG` |
| `src/cpu/dynamics/const_pot_cg.cpp` | Elastance \(A\), Jacobi-PCG, neutrality, write-back |
| `include/atomiverse/cpu/dynamics/const_pot_bo.h` | Dense dual twin `ConstPotBO` |
| `include/atomiverse/cpu/dynamics/md.h` | Host preset `cpu::md::MD` |
| `regressions/md/const-pot-cg-large-electrode/reg_const-pot-cg-large-electrode.cpp` | Product gate |
| `tests/cpu/dynamics/test_const_pot_cg.cpp` | Unit surface (not e2e) |

## Validation (vs MetalWalls algorithm class / dense BO dual twin)

Source:
[`regressions/md/const-pot-cg-large-electrode/REFERENCE.md`](../../../regressions/md/const-pot-cg-large-electrode/REFERENCE.md)
(offline pin; **never** live MetalWalls or LAMMPS in ctest). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).

**Oracle:** **MetalWalls** (algorithm class: iterative Jacobi-PCG electrode
charge solve for large \(N_{\mathrm{elec}}\); map E2). Binding **numeric**
oracle for site charges is the **in-tree dual twin** `cpu::md::ConstPotBO` on
the identical Gaussian elastance \(A\) (2026-07-24). No live MW binary is
required for the product gate.

**Why MetalWalls (not LAMMPS):** iterative CG on multi-site walls is the
MetalWalls-style large-electrode path (map E2). The dual correctness twin is
dense BO, not an external LAMMPS `fix electrode/conp` log (E1 madelung remains
the LAMMPS ELECTRODE foothold on a 4-atom plate).

### Accuracy

| Quantity | Atomiverse gate / pin | Twin / pin | Tol | Notes |
| -------- | --------------------: | ---------: | --- | ----- |
| \(N_{\mathrm{elec}}\) | \(\ge 18\) (product: **32**) | geometry | exact | \(4\times 4\) per wall |
| \(\max_i \lvert Q_i^{\mathrm{CG}} - Q_i^{\mathrm{BO}}\rvert\) [e] | \(\le 10^{-8}\) | dense BO dual | **1e-8** | site-wise dual twin |
| \(\lvert Q_{\mathrm{group}}^{\mathrm{CG}} - Q_{\mathrm{group}}^{\mathrm{BO}}\rvert\) [e] | \(\le 10^{-8}\) | dense BO dual | **1e-8** | bot and top groups |
| \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-10}\) | ~0 (`symm`) | **1e-10** | electroneutrality residual |
| Primary \(Q_{\mathrm{top}}\) [e] | within **1e-6** of pin | **0.27175356** | abs \(\le\) **1e-6** | 2-elec madelung, `xy_image_range=0`, dense BO / CG |
| Electrolyte \(q\) | unchanged \(\{+0.5,-0.5\}\) | fixed | exact write-back | only electrode rows rewritten |
| MD-bound \(Q_{\mathrm{top}}\) vs BO | \(\lvert\Delta\rvert \le 10^{-6}\) e | dense BO | **1e-6** | one VV step, `PairCoulCut` |

Geometry, \(\eta\), \(\Psi\), image range, and tolerances: **Method** table
and REFERENCE (match `reg_const-pot-cg-large-electrode`).

Related unit surface (not the e2e pin):
`tests/cpu/dynamics/test_const_pot_cg.cpp`.

### Offline dual twin (never ctest live MW)

```bash
# MD-only build, then product gate only (no LAMMPS/MetalWalls launch):
./build/regressions/reg_const-pot-cg-large-electrode
# asserts CG vs BO site charges; prints wall_ms CG / wall_ms BO (internal)
```

When MetalWalls is available offline: run an equivalent multi-site
constant-potential CG electrode demo, record group \(Q\) into REFERENCE, and
keep ctest free of the binary.

## Performance

Source: REFERENCE.md dual-gate summary; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md);
queue
[`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| MW multi-site CG / ATV `ConstPotCG` | — | — | **perf_exempt** | n/a | No MetalWalls runtime binary on worker PATH for fair CG twin |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | no MetalWalls runtime binary on worker PATH for fair CG twin; dual correctness twin is in-tree `ConstPotBO` (not an external oracle wall) |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Internal informational walls (`wall_ms CG` / `wall_ms BO` on the 32-site
plate) are printed by the regression binary and are **not** the 10× oracle
gate. Do **not** invent wall-clock ratios. When a MetalWalls binary is
available offline, re-run fair bench, record version + walls, and replace
`perf_exempt` in REFERENCE.

## Limitations

- **Primary-cell elastance for the large plate.** `xy_image_range = 0` is a
  multi-site correctness foothold, not full 2D-image / electrode-Ewald parity
  with E1 madelung or MW production cells.
- **Dense \(A\) storage.** Jacobi-PCG still builds a dense
  \(N_{\mathrm{elec}}\times N_{\mathrm{elec}}\) matrix; matrix-free / sparse /
  Ewald-electrode matvecs for very large walls are follow-on work.
- **No external MW fair wall** until MetalWalls is on the pin host —
  `perf_exempt` stands.
- **No ASPC warm-start on `ConstPotCG` product path in this slug** — predictor
  is the sibling `electrode-charge-predictor` / BO `use_cg` path.
- **Frozen-wall default.** `rebuild_matrix_every_step = false` unless the
  caller sets it.
- **Scope of E2 wave only** for this page: BO foothold, charge observer,
  freeze, short NVT, Gaussian form factor, slab 2D Ewald, etc. are sibling
  slugs with their own pages when written.

## References

- Offline pin: [`regressions/md/const-pot-cg-large-electrode/REFERENCE.md`](../../../regressions/md/const-pot-cg-large-electrode/REFERENCE.md)
- Product gate: [`regressions/md/const-pot-cg-large-electrode/reg_const-pot-cg-large-electrode.cpp`](../../../regressions/md/const-pot-cg-large-electrode/reg_const-pot-cg-large-electrode.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/const_pot_cg.h`](../../../include/atomiverse/cpu/dynamics/const_pot_cg.h)
- Dense dual twin: [`include/atomiverse/cpu/dynamics/const_pot_bo.h`](../../../include/atomiverse/cpu/dynamics/const_pot_bo.h)
- Sibling E1 report: [`const-pot-bo-parallel-plate.md`](const-pot-bo-parallel-plate.md)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Capability map: [`.claude/notes/metalwalls-capability-map.md`](../../../.claude/notes/metalwalls-capability-map.md)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md)
- Literature / method basis: Siepmann–Sprik Gaussian electrode charges; MetalWalls constant-potential CG electrode solve (map E2); dual twin aligned with dense BO / LAMMPS ELECTRODE `symm` projector (Ahrens-Iwers et al., *J. Chem. Phys.* **157**, 084801 (2022) for the ELECTRODE formulation context)
