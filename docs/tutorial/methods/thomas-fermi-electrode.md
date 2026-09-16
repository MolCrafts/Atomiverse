# Thomas–Fermi finite metallicity electrode — gold capacitor toy

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `thomas-fermi-electrode` |
| Regression | `regressions/md/thomas-fermi-electrode/` |
| Public API | `cpu::md::ConstPotBO` members `l_tf`, `site_density`, `capacitance()` (Component on `cpu::md::MD`, default `pre_force`) |
| Roadmap | map slice E7 |
| Oracle | **MetalWalls** binary, git `c8ad175d` (offline run; not LAMMPS) |

## Summary

**Thomas–Fermi (TF) electrode metallicity** (MetalWalls capability map **E7**):
finite screening length \(l_{\mathrm{TF}}\) hardens the Gaussian electrode elastance
diagonal so the metal is imperfect. Perfect metal is recovered at
\(l_{\mathrm{TF}}=0\). Public surface is still `cpu::md::ConstPotBO` — a
**Component** on `cpu::md::MD` at `md::Phase::pre_force` — with public members
`l_tf` [Å] and `site_density` [1/Å³]; no new Driver category and no charge solve
inside `KSpace*`. This regression is the **gold capacitor toy**: empty
parallel-plate Au lattice (no electrolyte), dual-gate perfect metal vs
\(l_{\mathrm{TF}}=2\) Å. The pin is **MetalWalls c8ad175d output** from the twin
input under
`tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/mw_twin/`
(run offline, never in ctest). Product gate `reg_thomas-fermi-electrode` asserts
electroneutrality, charge polarity, \(C(l_{\mathrm{TF}}=2)<C(0)\), and relative
agreement ≤ **1%** on \(Q_{\mathrm{top}}\) and integral capacitance \(C\) at both
\(l_{\mathrm{TF}}\). Fair wall vs MetalWalls is \(r\approx 4.4\).

> **Re-pinned 2026-07-29.** Waves 1–5 pinned this slug from a Python script that
> re-implemented `ConstPotBO`'s own truncated \(xy\)-image sum at
> `xy_image_range = 40` — a self-golden, not an oracle. Against the real
> MetalWalls binary that pin is **+6.13 % high**: the truncated lattice sum
> converges to the exact 2D-periodic elastance only as \(O(1/R)\). The model is
> correct; the numerics were not converged. See REFERENCE.md.

## Method

**Physics.** Scalfi, Dufils, Reeves, Rotenberg, Salanne, *J. Chem. Phys.*
**153**, 174704 (2020), DOI
[10.1063/5.0028232](https://doi.org/10.1063/5.0028232) (arXiv:1910.13341).
MetalWalls encodes finite metallicity as a **diagonal Thomas–Fermi hardness**
on the electrode elastance matrix \(A\) (metal units \(e\), Å, eV;
\(k_e=\) `units::coulomb` \(= 14.399645\) eV·Å/e²):

\[
A_{ii}\;\mathrel{+}= 4\pi\,k_e\,l_{\mathrm{TF}}^{2}\,d,
\qquad d = \text{site number density [1/Å³]}.
\]

Perfect metal is \(l_{\mathrm{TF}}=0\) (no hardness shift). Larger
\(l_{\mathrm{TF}}\) increases \(A_{ii}\), reduces induced electrode charge at
fixed \(\Delta\Psi\), and therefore **lowers** the integral capacitance
\(C=|Q|/|\Delta\Psi|\).

The underlying BO solve is the same as E1 (`const-pot-bo-parallel-plate`):
Siepmann–Sprik Gaussian electrode charges of width \(\eta\), 2D \(xy\) lattice
images for planar PBC, optional electroneutrality projector

\[
Q = \Bigl(A^{-1} - \frac{A^{-1}ee^{\mathsf T}A^{-1}}{e^{\mathsf T}A^{-1}e}\Bigr)(\Psi - b)
\]

so \(\sum_i Q_i = 0\). This pin has **no electrolyte** (\(b=0\)): empty vacuum
capacitor.

**Why MetalWalls (not LAMMPS):** LAMMPS ELECTRODE implements perfect-conductor
constant-potential electrodes only. Finite TF metallicity is **MW-only** in
this loop’s oracle priority → bind to MetalWalls / Scalfi, not LAMMPS.

### System (gold capacitor toy)

Empty parallel-plate capacitor (no electrolyte):

| Quantity | Value |
| -------- | ----- |
| Au lattice \(a_0\) | 4.07 Å |
| Site density \(d = 4/a_0^3\) | **0.059330329963** Å⁻³ |
| Plane spacing \(a = a_0/2\) | 2.035 Å |
| Planes per electrode | 3 (1 site per plane at cell centre) |
| \(L_x = L_y\) | \(a_0 = 4.07\) Å |
| First-plane gap \(L\) | 20.0 Å |
| \(\Psi_{\mathrm{bot}},\Psi_{\mathrm{top}}\) | 0 V, 1 V (\(\Delta\Psi=1\) V) |
| Gaussian \(\eta = 1/0.56\) | **1.785714285714** Å⁻¹ (MetalWalls \(w=0.56\) Å) |
| 2D \(xy\) image half-range | 1280 (convergence knob — see below) |
| Electroneutrality | on |
| Box \(L_z\) | 80 Å (electrode path uses 2D \(xy\) images only) |

Atom \(z\) [Å]: bot `{5.0, 7.035, 9.07}`, top `{29.07, 31.105, 33.14}`
(types 1 / 2). Six Au sites total.

### MetalWalls pin

Twin input:
`tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/mw_twin/`
(`num_pbc 2`, `electrode_charges matrix_inversion`, `charge_neutrality true`),
run as `mpirun -n 1 mw`; numbers are `total_charges.out` verbatim (MetalWalls'
atomic unit of charge *is* e, so no conversion — the unit work is on the inputs:
Å→bohr, V→hartree/e, Å⁻¹→bohr⁻¹).

| \(l_{\mathrm{TF}}\) [Å] | \(Q_{\mathrm{top}}\) [e] | \(C=\|Q\|/\Delta\Psi\) [e/V] |
| ----------------------: | -----------------------: | ---------------------------: |
| **0.0** (perfect metal) | **+4.354124119406114e−03** | **4.354124119406114e−03** |
| **2.0** (finite metallicity) | **+3.707555065364343e−03** | **3.707555065364343e−03** |

Continuum cross-check (paper Eq. 8 with \(L_{\mathrm{vac}}=L-a\)):
\(C_{\mathrm{cont}}(0)\approx 5.10\times 10^{-3}\),
\(C_{\mathrm{cont}}(2)\approx 4.17\times 10^{-3}\) [e/V] on this area — same
trend (discrete 1-site/plane lattice under-samples lateral density vs
continuum; the gate is the **MetalWalls** pin above).

### `xy_image_range` is a convergence knob, not a system property

`ConstPotBO` builds \(A\) as a direct square-image lattice sum truncated at
half-range \(R\). The exact 2D-periodic elastance (MetalWalls `num_pbc 2`, Ewald)
is the \(R\to\infty\) limit, and each pair misses a lattice tail
\(\simeq \pi z_{ij}^{2}/(R L A_{\mathrm{cell}})\) — an \(O(1/R)\) error that grows
as \(z^2\). This toy's 20 Å gap over a 4.07 Å lateral cell is the worst case:

| \(R\) | \(Q_{\mathrm{top}}(l_{\mathrm{TF}}=0)\) [e] | vs MetalWalls |
| ----: | ------------------------------------------: | ------------: |
| 40 | 4.620945546104e−03 | **+6.128 %** |
| 160 | 4.417905982930e−03 | +1.465 % |
| 320 | 4.385778337944e−03 | +0.727 % |
| 640 | 4.369898601631e−03 | +0.362 % |
| **1280** | **4.362004736725e−03** | **+0.181 %** |
| 2560 | 4.358069301166e−03 | +0.091 % |
| exact 2D Ewald | 4.354141528379e−03 | +0.0004 % |

Reproduce with
`tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/tf_lattice_convergence.py`.

### Key exclusions (non-goals of this slug)

- Live MetalWalls or LAMMPS inside ctest (offline pin only).
- Full electrode Ewald / MetalWalls reciprocal matrix (v1 2D-image \(A\)).
- Electrolyte / IL between walls (empty capacitor).
- TF as a separate Driver or solver class — only members on `ConstPotBO`.
- Charge solve inside PME / `KSpace*`.
- Public `*Config` bags or `make_*` / `build_*` factories.
- Full 2D Ewald elastance inside `ConstPotBO` (the truncated image sum stays
  the foothold; see the convergence table above).

## Implementation

Public surface only (`cpu::Frame` + `ConstPotBO` ctor / members; no Config
bag). Minimal standalone dual-gate matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

constexpr double kEta = 1.0 / 0.56;              // [1/Å]
constexpr double kA0 = 4.07;                     // [Å]
constexpr double kD = 4.0 / (kA0 * kA0 * kA0);   // site density [1/Å³]
constexpr double kPlane = kA0 / 2.0;
constexpr double kLGap = 20.0;

cpu::Frame gold_capacitor() {
    cpu::Frame f(6);
    const double z_surf_bot = 5.0 + 2.0 * kPlane;
    const double z_surf_top = z_surf_bot + kLGap;
    const double z[6] = {
        z_surf_bot - 2.0 * kPlane, z_surf_bot - kPlane, z_surf_bot,
        z_surf_top, z_surf_top + kPlane, z_surf_top + 2.0 * kPlane,
    };
    const int typ[6] = {1, 1, 1, 2, 2, 2};
    for (int i = 0; i < 6; ++i) {
        f.span(kAtom_x)[i] = f.span(kAtom_y)[i] = 0.0;
        f.span(kAtom_z)[i] = z[i];
        f.span(kAtom_charge)[i] = 0.0;
        f.span(kAtom_type)[i] = typ[i];
        f.span(kAtom_mass)[i] = 196.966553;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.box = cpu::Box::orthorhombic(kA0, kA0, 80.0);
    return f;
}

// Perfect metal (l_tf = 0) and finite metallicity (l_tf = 2 Å).
// Defaults to pre_force order −50; Frame is the only I/O token.
for (double l_tf : {0.0, 2.0}) {
    auto frame = gold_capacitor();
    cpu::md::ConstPotBO conp(kEta);
    conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
    conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
    conp.electroneutrality = true;
    conp.xy_image_range = 1280;  // O(1/R) convergence knob - see above
    conp.l_tf = l_tf;            // [Å]; 0 = perfect metal
    conp.site_density = kD;      // [1/Å³]; required when l_tf > 0

    cpu::md::MD sim;
    // ... neighbor / coulomb / integrator as needed ...
    sim.add_ref(conp);   // or add(std::move(conp)) to own
    sim.build();
    sim.run(frame, /*nsteps=*/1);

    const double q_top = conp.total_charge(1);  // [e]
    const double c = conp.capacitance();        // [e/V]
    // frame.span(kAtom_charge)[0..5] updated for electrode rows
}
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** Electrodes / TF are Components; never a new
  Driver category.
- Parameters are **constructor arguments and public members** — no
  `ConstPotBOConfig` / `ThomasFermiConfig`.
- Frame is the sole I/O currency; solved charges land on `kAtom_charge`.
- `l_tf > 0` requires `site_density > 0` (throws otherwise).

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
  cmake --build build --target reg_thomas-fermi-electrode -j"$(nproc)"
  ./build/regressions/reg_thomas-fermi-electrode
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/dynamics/const_pot_bo.h` | Public `ConstPotBO` (`l_tf`, `site_density`, `capacitance()`) |
| `src/cpu/dynamics/const_pot_bo.cpp` | Dense \(A\) + TF diagonal hardness \(A_{ii}\mathrel{+}=4\pi k_e l_{\mathrm{TF}}^2 d\) |
| `regressions/md/thomas-fermi-electrode/reg_thomas-fermi-electrode.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/mw_twin/` | MetalWalls twin input (the pin) |
| `tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/tf_lattice_convergence.py` | \(O(1/R)\) truncation diagnosis + exact 2D Ewald |

## Validation (vs the MetalWalls binary)

Source:
[`regressions/md/thomas-fermi-electrode/REFERENCE.md`](../../../regressions/md/thomas-fermi-electrode/REFERENCE.md)
(offline run; **never** live MetalWalls in ctest). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md).

**Oracle:** MetalWalls git `c8ad175d`, `src/thomas_fermi.f90`
(`gradQ_V(i,i) += 4*pi*ltf**2 / voronoi_volume`) — the atomic-unit form of
Scalfi et al. *JCP* **153**, 174704 (2020), identical to Atomiverse's
\(A_{ii}\mathrel{+}= 4\pi k_e l_{\mathrm{TF}}^{2} d\) in metal units. The
same binary reproduces its own shipped `tests/thomas_fermi/LiCl-Al-inv`
reference to 1e−12 relative, which is the check that the oracle run is sane.

**Why MetalWalls (not LAMMPS):** finite metallicity is **not** a LAMMPS
ELECTRODE feature (perfect conductor only). Binding oracle priority for this
loop: LAMMPS when the feature overlaps; **MetalWalls for MW-only physics**
(this slug).

### Accuracy

| Quantity | Atomiverse gate / pin | MetalWalls c8ad175d | Tol | Notes |
| -------- | --------------------: | --------------------------: | --- | ----- |
| \(Q_{\mathrm{top}}(l_{\mathrm{TF}}=0)\) [e] | within **1%** of pin | **+4.354124119406114e−03** | rel ≤ **0.01** | perfect metal (ATV +0.181 %) |
| \(Q_{\mathrm{top}}(l_{\mathrm{TF}}=2)\) [e] | within **1%** of pin | **+3.707555065364343e−03** | rel ≤ **0.01** | finite metallicity (ATV +0.183 %) |
| \(C(l_{\mathrm{TF}}=0)\) [e/V] | dual-gate vs pin | **4.354124119406114e−03** | rel ≤ **0.01** | \(\Delta\Psi=1\) V |
| \(C(l_{\mathrm{TF}}=2)\) [e/V] | dual-gate vs pin | **3.707555065364343e−03** | rel ≤ **0.01** | \(\Delta\Psi=1\) V |
| \(C(l_{\mathrm{TF}}=2) < C(l_{\mathrm{TF}}=0)\) | required | exact ordering | exact | metallicity lowers \(C\) |
| \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-10}\) | ~0 (neutrality) | **1e-10** | both \(l_{\mathrm{TF}}\) |
| \(\mathrm{sign}(Q_{\mathrm{top}})\) | \(+1\) at \(\Delta\Psi=+1\) V | \(+\) | exact | capacitive polarity |
| Frame write-back \(\sum q_{\mathrm{top}}\) | equals `total_charge(1)` | — | **1e-12** | rows 3..5 |

Geometry, \(\eta\), \(d\), \(\Psi\), image range, and tolerances: **Method**
table and REFERENCE (match `reg_thomas-fermi-electrode`).

### Offline pin repro (never ctest)

```bash
# oracle: copy mw_twin/, substitute @LTF@ (0.0 or 3.779452249251540 bohr), run
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
cd /tmp/tf && mpirun -n 1 /path/to/metalwalls/mw      # -> total_charges.out

# truncation diagnosis (needs numpy + scipy)
python3 tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/tf_lattice_convergence.py
```

## Performance

Source: REFERENCE.md dual-gate summary; handoff
the case `REFERENCE.md`.

| Twin | \(T_{\mathrm{MetalWalls}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------------: | -----------------------: | ----: | ---- | ------ |
| gold-cap TF BO solve (per elastance build + solve, 1 rank, `OMP_NUM_THREADS=1`) | 0.20 | 0.885 | **4.4** | yes | MetalWalls' 0.20 s is MPI-startup dominated, so \(r\) flatters Atomiverse |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **false** |
| `ratio_atv_over_oracle` | **4.4** |
| `within_10x` | **yes** |

The gap is algorithmic: MetalWalls sums the 2D-periodic elastance by Ewald,
Atomiverse evaluates \(21\times2561^2\approx1.4\times10^{8}\) `erf` images per
solve just to reach 0.18 %. Giving `ConstPotBO` a 2D Ewald elastance closes both
the accuracy tail and this ratio at once.

## Limitations

- **Discrete 1-site/plane lattice.** Under-samples lateral Au density vs
  continuum capacitor formulas; continuum numbers in REFERENCE are trend
  cross-checks only — the product gate is the discrete pin.
- **v1 elastance only.** Real-space Gaussian + 2D \(xy\) images; no full
  electrode Ewald / MW reciprocal matrix.
- **Empty vacuum only** in this foothold — no electrolyte \(b\), no MD
  trajectory product for TF walls (compose with short NVT siblings if needed).
- **Dense \(O(N_{\mathrm{elec}}^3)\) path** for the BO solve (CG is optional
  on the same component for larger electrodes; not exercised by this pin).
- **\(O(1/R)\) elastance truncation.** `xy_image_range` must be raised until the
  answer stops moving; 1280 here for 0.18 %. A 2D Ewald elastance would remove
  the knob entirely.
- **Scope of E7 only** for this page: perfect-metal BO (E1), CG large
  electrode, ASPC predictor, piston \(P_z\), etc. are sibling slugs.

## References

- Pin + conversions: [`regressions/md/thomas-fermi-electrode/REFERENCE.md`](../../../regressions/md/thomas-fermi-electrode/REFERENCE.md)
- Product gate: [`regressions/md/thomas-fermi-electrode/reg_thomas-fermi-electrode.cpp`](../../../regressions/md/thomas-fermi-electrode/reg_thomas-fermi-electrode.cpp)
- Public header: [`include/atomiverse/cpu/dynamics/const_pot_bo.h`](../../../include/atomiverse/cpu/dynamics/const_pot_bo.h)
- MetalWalls twin input: [`tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/mw_twin/`](../../../tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/mw_twin/)
- Truncation diagnosis: [`tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/tf_lattice_convergence.py`](../../../tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/tf_lattice_convergence.py)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md)
- Capability map: `.claude/notes/metalwalls-capability-map.md` (E7 metallicity)
- Literature: Scalfi, Dufils, Reeves, Rotenberg, Salanne, *J. Chem. Phys.* **153**, 174704 (2020), DOI [10.1063/5.0028232](https://doi.org/10.1063/5.0028232) (arXiv:1910.13341); MetalWalls TF hardness model
- Related E1 foothold: [`const-pot-bo-parallel-plate.md`](const-pot-bo-parallel-plate.md)
