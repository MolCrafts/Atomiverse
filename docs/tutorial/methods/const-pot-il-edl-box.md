# Constant-potential ionic-liquid EDL production box

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `const-pot-il-edl-box` |
| Regression | `regressions/md/const-pot-il-edl-box/` |
| Public API | `cpu::md::ConstPotBO` (CG + ASPC members `use_cg`, `cg_tol`, `predictor_order`), `cpu::md::FixFreeze`, `cpu::ElectrodeChargeObserver`, `cpu::PotentialProfile`, `cpu::SpeciesDensityProfile` — all Components on `cpu::md::MD` |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → map §9 planar graphene + const-pot + IL (wave 5, 2026-07-27) |
| Oracle | **MetalWalls git c8ad175d** — empty capacitor re-pinned 2026-07-29; with-IL absolute Q not expressible under image-sum elastance |

## Summary

**Production electric-double-layer (EDL) box** (MetalWalls capability map §9,
wave 5): this slug is not a new solver — it is the *composition* gate that the
wave-1..4 constant-potential pieces assemble into MetalWalls' flagship use
case. Two planar honeycomb (graphene-like) carbon walls held at
\(\Psi = \mp 1\) V by `cpu::md::ConstPotBO` (CG solve + ASPC predictor,
MetalWalls graphite Gaussian width) sandwich a non-polarizable single-site
coarse-grained ionic liquid (12 cations + 12 anions, BMIM⁺/PF₆⁻-like), with
LJ + `CoulLong` + `KSpacePME` forces and a short Nosé–Hoover NVT run. The
regression pins the empty-capacitor step-0 BO solve against **real MetalWalls
c8ad175d** (\(|Q|\approx 0.2076\,e\), `xy_image_range=80`), asserts EDL
screening enhancement \(|Q|_{\mathrm{IL}} > |Q|_{\mathrm{empty}}\) (absolute
with-IL \(Q\) under image-sum is not MW-expressible — MW Ewald gives
\(|Q|\approx 4.82\,e\)), then checks run-time invariants over 600 fs:
electrode neutrality, capacitive polarity, frozen walls, \(\Psi(z)\) sign,
and ion-density-profile normalization. Fair wall vs MetalWalls is
**`perf_exempt`** (product box).

## Method

**Physics.** Siepmann–Sprik / MetalWalls constant-potential electrodes:
Gaussian charges of width \(\eta\) on every electrode site, solved each step
so each site sits at its applied potential, with electrode-only
electroneutrality \(\sum_i Q_i = 0\) (Scalfi, Dufils, Reeves, Rotenberg,
Salanne, *J. Chem. Phys.* **153**, 174704 (2020), DOI
[10.1063/5.0028232](https://doi.org/10.1063/5.0028232); Marin-Laflèche et
al., MetalWalls, *JOSS* **5**, 2373 (2020), DOI
[10.21105/joss.02373](https://doi.org/10.21105/joss.02373)). Between the
walls, a charged electrolyte polarizes the electrodes far beyond the bare
vacuum-capacitor response — the EDL screening enhancement is the headline
observable of this gate.

Production path details (all public members, no Config bag):

- **CG solve** (`use_cg`, `cg_tol = 1e-10`, `max_cg_iters = 5000`) instead of
  the dense \(O(N^3)\) factorization — the large-electrode route.
- **ASPC predictor** (`predictor_order = 2`) warm-starts each step's CG from
  the extrapolated charge history — the MetalWalls production path.
- η = **1.805132 Å⁻¹**: the MetalWalls graphite Gaussian width
  (0.955234 bohr⁻¹, MW example inputs).
- `xy_image_range = 80` (truncated xy-image elastance; empty capacitor
  approaches MW 2D-Ewald; with-IL absolute \(Q\) still model-gapped).

### System (production EDL box)

| Quantity | Value |
| -------- | ----- |
| Box | 17.04 × 17.216585 × 40 Å (orthorhombic, 3D PBC) |
| Walls | honeycomb C, \(a_{\mathrm{CC}}\) = 1.42 Å, 4-atom rect cell, 4×7 → **112 sites/wall** |
| Wall planes | z = 12 Å (type 2, Ψ = −1 V), z = 28 Å (type 3, Ψ = +1 V), frozen (`FixFreeze`) |
| IL gap | 16 Å interior < 24 Å periodic return path (EDL response dominates) |
| Ions | 12 + 12 single-site CG (±1 e, non-polarizable), deterministic 3×3(−centre)×3 grid, checkerboard + one swap per outer layer (nascent-EDL polarization) |
| Cation LJ | σ = 4.38 Å, ε = 0.026532 eV, m = 139.22 amu (BMIM⁺-like) |
| Anion LJ | σ = 5.06 Å, ε = 0.048815 eV, m = 144.96 amu (PF₆⁻-like) |
| Wall C LJ | σ = 3.37 Å, ε = 0.0023837 eV; **no wall-type diagonal** → zero intra-wall LJ (LAMMPS `neigh_modify exclude` analogue) |
| Electrostatics | `CoulLong{α=0.3 Å⁻¹, r_c=8 Å}` + `KSpacePME{α=0.3, spacing 0.5 Å, order 5}` |
| Neighbor / LJ cutoff | 10 Å + 0.5 skin / 10 Å |
| Const-pot | η = 1.805132 Å⁻¹, CG tol 1e-10, ASPC order 2, electrode-only neutrality |
| NVT | VelocityVerlet dt = 1 fs, Nosé–Hoover 400 K τ = 100 fs, 600 steps, `InitVelocity{400 K, seed 11}` |

Ion sizes/masses follow the coarse-grained imidazolium family of Roy &
Maroncelli, *J. Phys. Chem. B* **114**, 12629 (2010) and Merlet et al.,
*J. Phys. Chem. C* **116**, 7687 (2012) — each ion reduced to its dominant
interaction site. The regression pins solver algebra and physics invariants,
**not** FF-sensitive observables.

### Offline oracle pin

MetalWalls binary **c8ad175d** at
`…/work/opt/metalwalls/mw`. Twin inputs:
`tools/metalwalls_oracle/inputs/metalwalls/const-pot-il-edl-box/mw_twin/`.

| case | MW \(Q_{\mathrm{left}}\) [e] |
|------|------------------------------|
| empty | **−2.076457628129008×10⁻¹** |
| with-IL | −4.818386149307364 (ATV image-sum does not match; gate order only) |

The wave-5 `pin_il_edl.py` min-image self-golden is **retired**.

### Key exclusions (non-goals of this slug)

- Live MetalWalls or LAMMPS inside ctest (offline pin only).
- Full electrode Ewald matrix (ATV still uses truncated xy-image elastance).
- Polarizable IL (Thole/PIM is the sibling `pim-npt-polarizable-il`).
- All-atom IL force field / FF-sensitive observables (single-site CG only).
- Nanoporous-carbon geometry (pending sibling `nanoporous-carbon-const-pot`).
- New Driver category, public `*Config` bags, `make_*` / `build_*` factories.

## Implementation

Public composition only — every piece is a Component on `cpu::md::MD`, Frame
in / Frame out. Condensed from
`regressions/md/const-pot-il-edl-box/reg_const-pot-il-edl-box.cpp`:

```cpp
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/fix_freeze.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/longrange/pme.h"
#include "atomiverse/cpu/observe/electrode_charge.h"
#include "atomiverse/cpu/observe/potential_profile.h"
#include "atomiverse/cpu/observe/species_density_profile.h"
#include "atomiverse/cpu/potentials/coul_long.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

// Production NVT: constant-potential walls + CG IL + observers (Frame only).
// Types 2/3 are the walls; step-0 BO charge is the pinned quantity after run.
cpu::md::ConstPotBO cpot(/*eta [1/Å]=*/1.805132);
cpot.add_electrode(/*type=*/2, /*Psi [V]=*/-1.0);
cpot.add_electrode(/*type=*/3, /*Psi [V]=*/+1.0);
cpot.use_cg = true;                   // large-electrode CG route
cpot.cg_tol = 1e-10;
cpot.max_cg_iters = 5000;
cpot.predictor_order = 2;             // ASPC warm start (MW production path)

cpu::ElectrodeChargeObserver qobs({2, 3});
cpu::PotentialProfile psi(/*n_bins=*/80, /*z_min=*/0.0, /*z_max=*/40.0);
cpu::SpeciesDensityProfile rho({0, 1}, /*n_bins=*/32, /*z_min=*/12.0,
                               /*z_max=*/28.0);

cpu::md::MD sim;
sim.add(cpu::md::InitVelocity{/*T=*/400.0, /*seed=*/11u});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/1.0});
sim.set_thermostat(cpu::md::NoseHoover{400.0, /*tau=*/100.0, 1.0});
sim.set_evaluator(cpu::PairLJCut{{{0, 0, 0.026532, 4.38},   // cation
                                  {1, 1, 0.048815, 5.06},   // anion
                                  {0, 2, eps_cw, sig_cw},   // explicit LB
                                  {0, 3, eps_cw, sig_cw},   // ion–wall rows;
                                  {1, 2, eps_aw, sig_aw},   // no wall diagonal
                                  {1, 3, eps_aw, sig_aw}},
                                 /*cutoff=*/10.0});
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/10.0, /*skin=*/0.5});
sim.on(md::Phase::force).add(cpu::CoulLong{/*alpha=*/0.3, /*cutoff=*/8.0});
sim.add(cpu::KSpacePME{/*alpha=*/0.3, /*spacing=*/0.5, /*order=*/5});
sim.add(std::move(cpot));             // solved charges land on kAtom_charge
sim.add(cpu::md::FixFreeze{2});
sim.add(cpu::md::FixFreeze{3});
sim.add_ref(qobs);
sim.add_ref(psi);
sim.add_ref(rho);
sim.build();
sim.run(frame, 600);                  // 0.6 ps NVT at 400 K

double QL = qobs.total_charge(0), QR = qobs.total_charge(1);  // [e]
double dpsi = psi.psi()[56] - psi.psi()[24];  // Ψ(z_R) − Ψ(z_L) [V]
double n0 = rho.mean_density(0, /*bin=*/0);   // cation run-mean ρ_s(z) [1/Å³]
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** Const-pot, freeze, and all observers are
  Components — no new Driver category, no charge solve inside `KSpace*`.
- Parameters are **constructor arguments and public members** — no
  `ConstPotBOConfig`.
- Frame is the sole I/O currency; solved electrode charges land on
  `kAtom_charge` every step.
- Observer vs Recorder split holds: `ElectrodeChargeObserver` /
  `PotentialProfile` / `SpeciesDensityProfile` compute derived quantities
  only; nothing here writes files.

Build and run the regression (MD-only configure; CPU login pattern; needs
`ATV_BUILD_KSPACE`):

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
  cmake --build build --target reg_const-pot-il-edl-box -j"$(nproc)"
  ./build/regressions/reg_const-pot-il-edl-box
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/const_pot_bo.h` | Public `ConstPotBO` (CG + ASPC members) |
| `regressions/md/const-pot-il-edl-box/reg_const-pot-il-edl-box.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/metalwalls/const-pot-il-edl-box/pin_il_edl.py` | Offline numpy pin |

## Validation (vs MetalWalls-form BO algebra)

Source:
[`regressions/md/const-pot-il-edl-box/REFERENCE.md`](../../../regressions/md/const-pot-il-edl-box/REFERENCE.md)
(offline pin; **never** live MetalWalls in ctest). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).

### Accuracy

Hard pins — deterministic step-0 BO solves on the exact initial geometry
(ATV CG at tol 1e-12 vs the numpy direct solve):

| Quantity | ATV gate | Oracle pin (numpy BO algebra) | Tol | Notes |
| -------- | -------: | ----------------------------: | --- | ----- |
| empty capacitor \(Q_{\mathrm{left}}\) [e] | within tol of pin | **−4.965149459545e−01** | abs ≤ **1e−6** e | IL charges zeroed |
| empty capacitor \(Q_{\mathrm{right}}\) [e] | within tol of pin | **+4.965149459545e−01** | abs ≤ **1e−6** e | IL charges zeroed |
| with-IL \(Q_{\mathrm{left}}\) [e] | within tol of pin | **−1.552770528510e+00** | abs ≤ **1e−6** e | nascent-EDL config |
| with-IL \(Q_{\mathrm{right}}\) [e] | within tol of pin | **+1.552770528510e+00** | abs ≤ **1e−6** e | nascent-EDL config |
| EDL screening \(\lvert Q\rvert_{\mathrm{IL}}/\lvert Q\rvert_{\mathrm{empty}}\) | ≈ **3.13** | > 1 required | exact ordering | electrolyte enhances induced charge |
| polarity | \(Q_{\mathrm{left}} < 0 < Q_{\mathrm{right}}\) | same | exact | matches ΔΨ = ∓1 V |
| electrode neutrality [e] | residual | ~0 | ≤ **1e−8** | both solves |

Run-time invariants over 600 fs NVT (trajectory-dependent, therefore
asserted as invariants, not hard-pinned):

| Invariant | Gate |
| --------- | ---- |
| finite \(E\), \(K > 0\), finite charges | every 100-step chunk |
| \(\lvert Q_L + Q_R\rvert\) [e] | ≤ **1e−6** each chunk |
| polarity after 600 fs | \(Q_L < 0 < Q_R\) |
| walls immobile | max \(\lvert\Delta z\rvert\) ≤ **1e−10** Å |
| \(\Psi(z)\) (80 bins over [0, 40 Å)) | \(\Psi(z_R) - \Psi(z_L) > 0\) (applied ΔΨ = +2 V) |
| ion confinement (32 bins over [12, 28 Å)) | all 12+12 ions between the walls at the final step |
| run-mean profile normalization | \(\sum_z \rho_s A\,dz = 12 \pm\) **1e−6** per species |
| EDL layering (4 Å first layer) | printed diagnostics (0.6 ps — not asserted) |

### Offline pin repro (never ctest)

```bash
python3 tools/metalwalls_oracle/inputs/metalwalls/const-pot-il-edl-box/pin_il_edl.py
```

## Performance

Source: REFERENCE.md; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md).

| Twin | \(T_{\mathrm{MetalWalls}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------------: | -----------------------: | ----: | ---- | ------ |
| IL EDL box, 600 fs NVT | — | — | **perf_exempt** | n/a | MW-only physics (Gaussian constant-potential electrodes + IL EDL box); no MetalWalls twin binary, no LAMMPS ELECTRODE package in the available `lmp` |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | MW-only physics; no MetalWalls twin binary and no LAMMPS ELECTRODE in the available `lmp` for a fair wall-clock twin |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a MetalWalls binary is available,
re-run the offline fair bench, record version + walls, and replace
`perf_exempt` in REFERENCE.

## Limitations

- **Min-image electrode algebra** (`xy_image_range = 0`): the documented E1
  foothold level, not the full MetalWalls electrode Ewald matrix.
- **Single-site CG IL, non-polarizable, unscaled ±1 e charges** — pins solver
  algebra and physics invariants, not FF-sensitive observables (density
  profiles, differential capacitance vs experiment).
- **0.6 ps trajectory**: EDL layering numbers are printed diagnostics, not
  converged structure; production sampling is downstream usage.
- **Planar walls only** — nanoporous carbon is the pending sibling
  (`nanoporous-carbon-const-pot`).
- **No fair speed ratio** until MetalWalls is on the pin host.
- Composition gate only: the individual solver features (CG, ASPC, TF
  metallicity, per-electrode η, MPI halo) are pinned by their own sibling
  slugs.

## References

- Offline pin: [`regressions/md/const-pot-il-edl-box/REFERENCE.md`](../../../regressions/md/const-pot-il-edl-box/REFERENCE.md)
- Product gate: [`regressions/md/const-pot-il-edl-box/reg_const-pot-il-edl-box.cpp`](../../../regressions/md/const-pot-il-edl-box/reg_const-pot-il-edl-box.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/const_pot_bo.h`](../../../include/atomiverse/cpu/dynamics/const_pot_bo.h)
- Pin script: [`tools/metalwalls_oracle/inputs/metalwalls/const-pot-il-edl-box/pin_il_edl.py`](../../../tools/metalwalls_oracle/inputs/metalwalls/const-pot-il-edl-box/pin_il_edl.py)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md)
- Capability map: `.claude/notes/metalwalls-capability-map.md` (§9 planar graphene + const-pot + IL)
- Literature: Scalfi, Dufils, Reeves, Rotenberg, Salanne, *J. Chem. Phys.* **153**, 174704 (2020), DOI [10.1063/5.0028232](https://doi.org/10.1063/5.0028232); Marin-Laflèche et al., MetalWalls, *JOSS* **5**, 2373 (2020), DOI [10.21105/joss.02373](https://doi.org/10.21105/joss.02373); Roy & Maroncelli, *J. Phys. Chem. B* **114**, 12629 (2010); Merlet et al., *J. Phys. Chem. C* **116**, 7687 (2012)
- Related siblings: [`const-pot-bo-parallel-plate.md`](const-pot-bo-parallel-plate.md), [`const-pot-cg-large-electrode.md`](const-pot-cg-large-electrode.md), [`thomas-fermi-electrode.md`](thomas-fermi-electrode.md)
