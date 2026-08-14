# Per-electrode Gaussian width η — MetalWalls multi-electrode (V, η)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `per-electrode-eta` |
| Regression | `regressions/md/per-electrode-eta/` |
| Public API | `cpu::md::ConstPotBO::add_electrode(type, Psi, eta)` (3-arg overload; per-group Gaussian width) on the existing `ConstPotBO` Component (`cpu::md::MD`, default `pre_force`) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → wave 5 (2026-07-27); map slice "Multi-electrode, per-electrode \(V,\eta\)" (**P0**) |
| Oracle | **literature** closed form — Reed–Lanning–Madden 2007, Scalfi et al. 2020, MetalWalls JOSS 2020 (no MW binary on the agent host) |

## Summary

**Per-electrode Gaussian charge widths** (MetalWalls capability map
"Multi-electrode, per-electrode \(V,\eta\)"): each electrode group carries its
own Gaussian width \(\eta_e\) [1/Å] instead of one shared solver value —
MetalWalls' per-`species` `charge gaussian` input. Public surface is still
`cpu::md::ConstPotBO` — a **Component** on `cpu::md::MD` at
`md::Phase::pre_force` — extended with a 3-argument
`add_electrode(type, Psi, eta)` overload; the 2-argument overload keeps
inheriting the shared ctor `eta`. Dissimilar widths mix in the elastance
off-diagonals via \(\eta_{ij}=\eta_i\eta_j/\sqrt{\eta_i^2+\eta_j^2}\). No new
Driver category, no Config bag, no charge solve inside `KSpace*`. The
regression pins a **two-site vacuum capacitor with dissimilar η** against the
exact literature closed form (rel ≤ 1e-9), and brackets a multi-atom
mixed-η two-wall system strictly between its two shared-η twins. Fair wall
vs MetalWalls is **`perf_exempt`** (no MW binary on the agent host; the
LAMMPS ELECTRODE fix has a single `eta` per fix, so no offline twin either).

## Method

**Physics.** Reed, Lanning, Madden, *J. Chem. Phys.* **126**, 084704 (2007),
DOI [10.1063/1.2464084](https://doi.org/10.1063/1.2464084); Scalfi, Dufils,
Reeves, Rotenberg, Salanne, *J. Chem. Phys.* **153**, 174704 (2020), DOI
[10.1063/5.0028232](https://doi.org/10.1063/5.0028232); Marin-Laflèche et al.
(MetalWalls), *J. Open Source Softw.* **5**(53), 2373 (2020), DOI
[10.21105/joss.02373](https://doi.org/10.21105/joss.02373). The
constant-potential elastance matrix over Gaussian electrode charges with
per-site widths \(\eta_i\) has the published closed-form elements (metal
units \(e\), Å, eV; \(k_e=\) `units::coulomb` \(= 14.399645\) eV·Å/e²):

\[
A_{ii} = k_e\sqrt{2/\pi}\;\eta_i,
\qquad
A_{ij} = k_e\,\frac{\mathrm{erf}(\eta_{ij} r_{ij})}{r_{ij}},
\qquad
\eta_{ij} = \frac{\eta_i\eta_j}{\sqrt{\eta_i^2+\eta_j^2}},
\]

and the electrode–electrolyte RHS \(b_i = k_e\,q_j\,\mathrm{erf}(\eta_i r)/r\).
Smaller \(\eta\) (wider Gaussian) lowers the self-elastance \(A_{ii}\), so at
fixed \(\Delta\Psi\) the induced charge — and the capacitance — grows
monotonically as either wall's width shrinks.

The underlying BO solve is the E1 foothold (`const-pot-bo-parallel-plate`):
Siepmann–Sprik Gaussian charges, optional 2D \(xy\) lattice images for planar
PBC, electroneutrality projector so \(\sum_i Q_i = 0\).

**Why literature (not a live oracle):** the loop row is oracle=metalwalls,
but no MetalWalls binary is installed on the host, and the LAMMPS ELECTRODE
fix exposes a single `eta` per fix — per the `tools/metalwalls_oracle`
contract the pin is the exact literature closed form with full citation. The
would-be MW twin input is preserved at
`tools/metalwalls_oracle/inputs/metalwalls/per-electrode-eta/runtime.inpt`
(never run in ctest).

### Case 1 — two-site vacuum capacitor, dissimilar η (exact pin)

| Quantity | Value |
| -------- | ----- |
| \(\eta_1\) (bottom) | **1.979** Å⁻¹ (LAMMPS ELECTRODE default; Ahrens-Iwers et al. *JCP* **157**, 084801 (2022)) |
| \(\eta_2\) (top) | **0.955** Å⁻¹ (Merlet/Salanne graphite width used in MW examples) |
| Separation \(d\) | 4 Å (sites at \(z = 18, 22\) Å; cubic box 40 Å) |
| \(\Psi_{\mathrm{bot}},\Psi_{\mathrm{top}}\) | 0 V, 1 V (\(\Delta\Psi=1\) V) |
| Electroneutrality | on (\(Q_2=-Q_1\)) |

Exact constrained BO solve:

\[
Q = \frac{\Delta\Psi}{A_{11}+A_{22}-2A_{12}} = \mathbf{0.037722100829787}\ e,
\]

with \(A_{11}=22.737234510126967\), \(A_{22}=10.972237977347778\),
\(A_{12}=3.599907138217925\), \(\eta_{12}=0.860091344088193\) Å⁻¹. Shared-η
twins of the same geometry bracket it:
\(Q(\eta{=}1.979\ \text{shared}) = 0.026126955854989\ e <
Q_{\mathrm{mixed}} < Q(\eta{=}0.955\ \text{shared}) \approx 0.0678\ e\).

### Case 2 — multi-atom two-wall, mixed η, 2D xy Gaussian images

2×2-atom square walls (\(a = 2\) Å, \(L_x=L_y=4\) Å) at \(z = 8\) and 13 Å
(separation 5 Å) in a 4×4×24 Å box; \(\eta_{\mathrm{bot}}=1.979\),
\(\eta_{\mathrm{top}}=0.955\) Å⁻¹, \(\Delta\Psi=1\) V,
`xy_image_range = 100`. Asserted self-consistency: electroneutrality, the
sign of \(Q_{\mathrm{top}}\) follows \(\Delta\Psi\), the mixed-η solution
lands **strictly between** its two shared-η twins, and the 3-arg overload at
equal η reproduces the 2-arg shared path bitwise-tight (≤ 1e-14 e/site).

### Key exclusions (non-goals of this slug)

- Live MetalWalls or LAMMPS inside ctest (offline literature pin only).
- Full electrode Ewald / MW reciprocal matrix (v1 2D-image \(A\)).
- Electrolyte between the walls (vacuum capacitor pins).
- Per-**atom** (row-wise) η — widths are per electrode *group*.
- A new Driver / solver class — only the `add_electrode` overload on
  `ConstPotBO`.
- Public `*Config` bags or `make_*` / `build_*` factories.
- Fair wall-clock twin vs a MetalWalls binary (none on agent host).

## Implementation

Public surface only (`cpu::Frame` + `ConstPotBO` ctor / members; no Config
bag). Minimal standalone Case-1 pin matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

// Two-site vacuum capacitor: sites at z = 18 and 22 Å (d = 4 Å).
cpu::Frame frame(2);
const double z[2] = {18.0, 22.0};
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_x)[i] = frame.span(kAtom_y)[i] = 0.0;
    frame.span(kAtom_z)[i] = z[i];
    frame.span(kAtom_charge)[i] = 0.0;
    frame.span(kAtom_type)[i] = i + 1;
    frame.span(kAtom_mass)[i] = 196.97;
    frame.span(kAtom_tag)[i] = i + 1;
}
frame.box = cpu::Box::cubic(40.0);

// Defaults to pre_force order −50.
// 3-arg overload: per-electrode Gaussian width overrides the shared eta.
cpu::md::ConstPotBO conp(/*shared eta [1/Å]=*/2.0);
conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0, /*eta [1/Å]=*/1.979);
conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0, /*eta [1/Å]=*/0.955);
conp.electroneutrality = true;
conp.xy_image_range = 100;

cpu::md::MD sim;
// ... neighbor / coulomb / integrator as needed ...
sim.add_ref(conp);   // or add(std::move(conp)) to own
sim.build();
sim.run(frame, /*nsteps=*/1);

const double q_top = conp.total_charge(1);        // [e] → 0.037722100829787
const double resid = conp.neutrality_residual();  // ≤ 1e-12
// frame.span(kAtom_charge) updated on the electrode rows
```

The 2-arg `add_electrode(type, Psi)` keeps the shared ctor `eta` (a group
`eta` of 0 means "inherit"); at equal widths the two paths are identical to
≤ 1e-14 e/site.

Architecture rules:

- **Driver stays `cpu::md::MD`.** Per-electrode η is an overload + member on
  the existing `ConstPotBO` Component; never a new Driver category.
- Parameters are **constructor arguments and public members** — no
  `ElectrodeConfig` / per-group bag type on the public path.
- Frame is the sole I/O currency; solved charges land on `kAtom_charge`.
- Mixed widths only touch the elastance elements (\(A_{ii}\), \(A_{ij}\),
  \(b_i\)); electroneutrality, TF hardness (`l_tf`), CG and ASPC compose
  unchanged on the same component.

Build and run the regression (MD-only configure; CPU login pattern):

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
  cmake --build build --target reg_per-electrode-eta -j"$(nproc)"
  ./build/regressions/reg_per-electrode-eta
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/const_pot_bo.h` | Public `ConstPotBO` (`add_electrode(type, Psi, eta)` 3-arg overload; per-group `eta`) |
| `src/cpu/dynamics/const_pot_bo.cpp` | Per-row \(\eta_i\), mixed \(\eta_{ij}\) elastance + RHS |
| `regressions/md/per-electrode-eta/reg_per-electrode-eta.cpp` | Product gate |
| `tools/metalwalls_oracle/pin_notes_per-electrode-eta.md` | Pin derivation |
| `tools/metalwalls_oracle/inputs/metalwalls/per-electrode-eta/runtime.inpt` | Would-be MW twin input (preserved, never run) |

## Validation (vs literature closed form)

Source:
[`regressions/md/per-electrode-eta/REFERENCE.md`](../../../regressions/md/per-electrode-eta/REFERENCE.md)
(offline pin; **never** a live oracle in ctest). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).

**Oracle:** literature — Reed–Lanning–Madden *JCP* **126**, 084704 (2007);
Scalfi et al. *JCP* **153**, 174704 (2020); MetalWalls JOSS **5**(53), 2373
(2020). The pin is the exact constrained BO solve
\(Q = \Delta\Psi/(A_{11}+A_{22}-2A_{12})\) with the published Gaussian
elastance elements at dissimilar widths.

### Accuracy

| Quantity | Atomiverse gate | Oracle pin (literature) | Tol | Notes |
| -------- | --------------: | ----------------------: | --- | ----- |
| Case 1 \(Q_{\mathrm{top}}\) [e] | within rel 1e-9 of pin | **+0.037722100829787** | rel ≤ **1e-9** | \(\eta_1=1.979\), \(\eta_2=0.955\) Å⁻¹, \(d=4\) Å, \(\Delta\Psi=1\) V |
| Case 1 \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-12}\) | 0 (electroneutral) | **1e-12** | projector on |
| Case 2 \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-10}\) | 0 (electroneutral) | **1e-10** | 2×2 walls, `xy_image_range = 100` |
| Case 2 \(\mathrm{sign}(Q_{\mathrm{top}})\) | \(+1\) at \(\Delta\Psi=+1\) V | \(+\) | exact | capacitive polarity |
| Case 2 bracket \(Q^{\eta=1.979} < Q^{\mathrm{mixed}} < Q^{\eta=0.955}\) | required | strict ordering | exact | smaller η ⇒ smaller \(A_{ii}\) ⇒ more charge |
| 3-arg `add_electrode` at equal η vs 2-arg shared path [e/site] | identical | 0 | **1e-14** | overload equivalence |

Elastance intermediates fixed by the pin: \(A_{11}=22.737234510126967\),
\(A_{22}=10.972237977347778\), \(A_{12}=3.599907138217925\),
\(\eta_{12}=0.860091344088193\) Å⁻¹ (all in eV, Å, e with
\(k_e=14.399645\) eV·Å/e²). Geometry and tolerances match
`reg_per-electrode-eta` exactly.

### Offline pin repro (never ctest)

```bash
# Derivation: tools/metalwalls_oracle/pin_notes_per-electrode-eta.md
# Optional, if a MetalWalls binary is installed:
#   run tools/metalwalls_oracle/inputs/metalwalls/per-electrode-eta/runtime.inpt
```

## Performance

Source: REFERENCE.md; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| mixed-η BO solve | — | — | **perf_exempt** | n/a | no runnable oracle implements this physics on the host |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | no MetalWalls binary on the agent host; LAMMPS ELECTRODE has a single `eta` per fix, so no offline per-electrode-η twin was runnable — no fair wall-clock ratio exists |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a MetalWalls binary is available,
run the preserved `runtime.inpt` twin, record version + walls, and replace
`perf_exempt` in REFERENCE.

## Limitations

- **Per-group, not per-atom.** η is set per electrode group
  (`add_electrode`); MetalWalls' per-`species` granularity is matched, but no
  row-wise η override exists.
- **v1 elastance only.** Real-space Gaussian + 2D \(xy\) images; no full
  electrode Ewald / MW reciprocal matrix.
- **Vacuum capacitor pins only** — no electrolyte RHS exercised at mixed η
  in this regression (the mixed-η \(b_i\) uses the same published erf form).
- **Dense \(O(N_{\mathrm{elec}}^3)\) path** for the BO solve in the pins (CG
  is optional on the same component; not exercised here).
- **No fair speed ratio** until a MetalWalls binary is on the pin host.
- **Scope of this slug only:** TF metallicity (`thomas-fermi-electrode`,
  `voronoi-volume-tf`), constant-charge walls, CG/ASPC are sibling slugs.

## References

- Offline pin: [`regressions/md/per-electrode-eta/REFERENCE.md`](../../../regressions/md/per-electrode-eta/REFERENCE.md)
- Product gate: [`regressions/md/per-electrode-eta/reg_per-electrode-eta.cpp`](../../../regressions/md/per-electrode-eta/reg_per-electrode-eta.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/const_pot_bo.h`](../../../include/atomiverse/cpu/dynamics/const_pot_bo.h)
- Pin derivation: [`tools/metalwalls_oracle/pin_notes_per-electrode-eta.md`](../../../tools/metalwalls_oracle/pin_notes_per-electrode-eta.md)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md)
- Capability map: `.claude/notes/metalwalls-capability-map.md` (multi-electrode \(V,\eta\), P0)
- Literature: Reed, Lanning, Madden, *J. Chem. Phys.* **126**, 084704 (2007), DOI [10.1063/1.2464084](https://doi.org/10.1063/1.2464084); Scalfi, Dufils, Reeves, Rotenberg, Salanne, *J. Chem. Phys.* **153**, 174704 (2020), DOI [10.1063/5.0028232](https://doi.org/10.1063/5.0028232); Marin-Laflèche et al., *J. Open Source Softw.* **5**(53), 2373 (2020), DOI [10.21105/joss.02373](https://doi.org/10.21105/joss.02373); Ahrens-Iwers et al., *J. Chem. Phys.* **157**, 084801 (2022) (η = 1.979 Å⁻¹ default)
- Related footholds: [`const-pot-bo-parallel-plate.md`](const-pot-bo-parallel-plate.md), [`thomas-fermi-electrode.md`](thomas-fermi-electrode.md)
