# GPW CELL_OPT — MOLOPT re-twin (ice-like water chain)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-variable-cell-molopt` |
| Regression | `regressions/gpw/gpw-variable-cell-molopt/` |
| Public API | `cpu::minimizer::Minimize` + `AnisotropicCellRelax` + `StressConverged` + `cpu::gpw::GPWEvaluator` (`compute_stress`, `stress_axes`) |
| Roadmap / queue | [dft.md](../../roadmap/dft.md) variable-cell / CELL_OPT; [cp2k-method-queue.md](../../developer/cp2k-method-queue.md) row `gpw-variable-cell-molopt` |
| Wave | CP2K capability harness wave 2 — **done** (2026-07-25) |
| Dual gate | Correctness vs offline CP2K 2026.1 CELL_OPT; speed ratio **~7.5** (≤10×) |

## Summary

**Constrained variable-cell relaxation (CELL_OPT)** for closed-shell
**Γ-point GPW** on a proton-ordered **periodic water chain** (ice Ih
c-axis strand motif: 1 H₂O / cell, H-bonded to its own \(x\)-images).
Ionic fractional coordinates are **fixed**; only the chain axis \(a\)
relaxes under \(P_{\mathrm{target}}=0\) while \(b=c=8\) Å vacuum edges
stay bit-identical.

This page is the **pin-quality MOLOPT re-twin** of
[`gpw-variable-cell-ice`](../../../regressions/gpw/gpw-variable-cell-ice/)
(GTH-PADE compact pivot). Product path:

- electronic structure: `cpu::gpw::GPWEvaluator` with **PBE** /
  **DZVP-MOLOPT-SR-GTH** / **GTH-PBE**, product **analytic fixed-\(P\)**
  cell stress (`compute_stress = true`, `stress_use_fd = false`),
  `stress_axes = {x}`;
- outer loop: `cpu::minimizer::Minimize` +
  `cpu::minimizer::AnisotropicCellRelax` (`relax_x` only,
  (x-only).

Hard-coded offline **CP2K 2026.1** CELL_OPT pins (never live from ctest):
\(a^\star = 3.200237\) Å, \(E^\star = -17.184075549824836\) Ha.
Product gate: `reg_gpw-variable-cell-molopt`. Fair wall
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 7.5\).

## Method

### Electronic structure (GPW / Quickstep)

Gaussian-and-plane-waves Kohn–Sham (VandeVondele et al., *Comput. Phys.
Commun.* **167**, 103 (2005)): density matrix in contracted Gaussians;
Hartree / XC / local short-range GTH PP on a uniform grid; Poisson under
`PERIODIC XYZ` (`nonperiodic = false`). Public energy on Frame is eV
(`kEnergy`); pin channel is Hartree via `last_energy_hartree()`.

**Basis choice (binding for e2e).** Full **DZVP-MOLOPT-GTH** periodic
image-sum collocation still exceeds the ≤ 60 s regression budget even
for a single SCF on this chain (measured 2026-07-25). The product twin
therefore uses the short-range MOLOPT table **DZVP-MOLOPT-SR-GTH**
(same family as `gpw-pbe-d3bj-ice` / `gpw-ts-ice`) — still a MOLOPT
Gaussian basis + PBE + GTH-PBE product path. Full-MOLOPT CP2K CELL_OPT
pins are retained as a secondary oracle input for the GPW perf campaign
(not dual-gated in the regression).

### Cell stress (analytic fixed-\(P\))

Product path is **Hellmann–Feynman fixed-\(P\)** orthorhombic diagonal
stress with Pulay \(-\mathrm{Tr}(W\,S)\) (see
[gpw-analytic-stress](gpw-analytic-stress.md)), not multi-SCF strain-FD.
Sign matches CP2K `STRESS_TENSOR`:

\[
\sigma_{aa}
=
-\Omega^{-1}
\frac{\partial E}{\partial\varepsilon_{aa}},
\qquad a\in\{x,y,z\},
\]

with Frame `kStress` intensive **[eV/Å³]** Voigt and pin channel
`last_stress_ha_bohr3()` **[Ha/Bohr³]**. Axes mask
`stress_axes = {true,false,false}` forces \(\sigma_{yy}=\sigma_{zz}=0\)
in the published tensor so constrained edges never drive residual /
scale (CP2K `CONSTRAINT YZ` twin).

### Variable-cell optimizer

`AnisotropicCellRelax` independently scales orthorhombic edges toward
\(P_{\mathrm{target}}\) (here 0). For GPW intensive stress:

\[
s_a \approx 1 - \alpha\,\frac{\sigma_{aa}-P_{\mathrm{target}}}{B},
\qquad |s_a-1|\le \texttt{max\_scale},
\]

with soft bulk modulus \(B = \texttt{bulk\_mod}\) [eV/Å³] from offline
\(E(a)\) curvature (full-MOLOPT ENERGY scan →
\(d\sigma_{xx}/d\varepsilon \approx 0.20\,\mathrm{eV/Å}^3\)).
Masked axes (`relax_y = relax_z = false`) are never scaled and never
enter the residual. Cartesian ion positions scale affinely with the
cell so **fractional** coordinates stay fixed (CP2K `SCALED T` +
`FIXED_ATOMS`).

Termination is a **Criterion**, not an optimizer self-judge:
`StressConverged{0, 5×10⁻⁴ eV/Å³, x-only}` →
`Verdict::Converged` when \(|\sigma_{xx}-0|\le\texttt{stol}\).

### System (binding twin)

| Knob | Value |
| ---- | ----- |
| Motif | 1 H₂O / cell, H-bonded chain along \(a\) |
| Cell | \(a_0 = 3.24\) Å start; \(b = c = 8\) Å fixed; `PERIODIC XYZ` |
| XC | **PBE** |
| Basis | **DZVP-MOLOPT-SR-GTH** (O, H) |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 300 Ry |
| SCF | `tol` / EPS_SCF \(= 1\times 10^{-7}\), mix 0.3, max_iter 200 |
| Stress | analytic fixed-\(P\), \(h=2\times 10^{-3}\), axes `{x}` |
| Relax | \(\alpha=1.0\), `bulk_mod = 0.20` eV/Å³, `max_scale = 0.05` |
| Converge | `StressConverged{0, 5e-4 eV/Å³, x-only}`; budget 6 outer steps |
| Ions | forces off (`compute_forces = false`); fractions fixed |

**Fractional coordinates (fixed; CP2K `SCALED T`):**

| Atom | \(f_x\) | \(f_y\) | \(f_z\) |
| ---- | ------: | ------: | ------: |
| O | 0.000000 | 0 | 0.000000 |
| H1 | 0.319067 | 0 | 0.000000 |
| H2 | −0.079900 | 0 | 0.115830 |

Cartesian start: \(x_i = f_{x,i}\,a_0\), \(z_{\mathrm{H2}} = f_{z}\,c\).

### Relation to `gpw-variable-cell-ice`

| | ice (GTH-PADE pivot) | this slug (MOLOPT re-twin) |
| - | -------------------- | -------------------------- |
| XC / basis / PP | PADE / SZV+DZV-GTH-PADE | PBE / DZVP-MOLOPT-SR-GTH / GTH-PBE |
| \(a^\star\) (CP2K) | 3.211148 Å | **3.200237 Å** |
| Stress path | multi-SCF strain-FD | **analytic fixed-\(P\)** |
| Wall ATV / CP2K | ~5.1 / 3.9 s | **42.5 / 5.67 s** |
| Fair \(r\) | ~1.3 | **~7.5** |

### Key exclusions

- Full **DZVP-MOLOPT-GTH** e2e dual-gate (collocation > 60 s / SCF).
- Joint ion + cell optimization (ions fixed; no `GEO_OPT` + `CELL_OPT`).
- Triclinic / shear; isotropic `IsotropicCellRelax` product pin (use
  PW `pw-variable-cell-si` for isotropic CELL_OPT).
- UKS / k-point stress during relax; D3/D4/TS/MBD on the cell path.
- Multi-SCF strain-FD as product (still available as `stress_use_fd`).
- Live CP2K from ctest; public `*Config` / `make_*` factories; CUDA GPW.

## Implementation

Public surface only (`cpu::Frame`, `Kind` / lookup constructors,
`GPWEvaluator` ctor args + members, Minimize preset, Optimizer +
Criterion). No public `*Config`, no `make_*` / `build_*` factories.
Match the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/driver/criteria.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/minimize/anisotropic_cell_relax.h"
#include "atomiverse/cpu/minimize/minimize.h"

using namespace ATV;

// Offline CP2K 2026.1 CELL_OPT pins (hard-coded; never live oracle).
constexpr double kA_cp2k = 3.200237;            // Å
constexpr double kE_cp2k = -17.184075549824836; // Ha
constexpr double kA0 = 3.24, kB = 8.0, kC = 8.0;
constexpr double kF1x = 0.319067, kF2x = -0.079900, kF2z = 0.115830;

cpu::Frame frame(3);
const int z[3] = {8, 1, 1};
const double x[3]  = {0.0, kF1x * kA0, kF2x * kA0};
const double y[3]  = {0.0, 0.0, 0.0};
const double zc[3] = {0.0, 0.0, kF2z * kC};
for (int i = 0; i < 3; ++i) {
    frame.span(kAtom_type)[i] = z[i];
    frame.span(kAtom_x)[i] = x[i];
    frame.span(kAtom_y)[i] = y[i];
    frame.span(kAtom_z)[i] = zc[i];
    frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
        frame.span(kAtom_fz)[i] = 0;
}
frame.box = cpu::Box::orthorhombic(kA0, kB, kC);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
const cpu::gpw::Kind H{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-SR-GTH"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

cpu::gpw::GPWEvaluator eval(
    /*kinds*/ {O, H}, /*cutoff_ry*/ 300.0, /*max_iter*/ 200, /*tol*/ 1e-7,
    /*mix*/ 0.3, /*xc*/ cpu::gpw::XcKind::PBE, /*nonperiodic*/ false,
    /*compute_forces*/ false);
eval.compute_stress = true;
eval.stress_use_fd = false;               // product analytic fixed-P
eval.stress_strain_h = 2e-3;
eval.stress_axes = {true, false, false};  // chain axis only

cpu::minimizer::Minimize opt;
cpu::minimizer::AnisotropicCellRelax cell(/*P_target*/ 0.0, /*alpha*/ 1.0);
cell.bulk_mod = 0.20;          // from offline E(a) curvature [eV/Å³]
cell.max_scale = 0.05;
cell.relax_y = false;
cell.relax_z = false;
opt.set_optimizer_ref(cell);
opt.add_criterion(cpu::StressConverged{0.0, /*stol=*/5e-4,
                                       /*x*/ true, /*y*/ false,
                                       /*z*/ false});
opt.on(minimizer::Phase::force).add_ref(eval);
opt.build();
opt.run(frame, 6);

// a = frame.box.length(0); E_ha = eval.last_energy_hartree();
// residual σ_xx = frame.meta.get(kStress)[0]; // eV/Å³
// opt.converged() / opt.stop_reason() — loop owns termination
```

Architecture rules exercised:

- **Frame** is sole I/O (`blocks` + `meta` + `box`); no `*System`.
- Parameters are **ctor args / members** — no public `*Config`.
- **Driver owns the loop**; stress residual is judged by a Criterion
  (`StressConverged`), not by the optimizer.
- Force/stress binding is explicit
  `on(minimizer::Phase::force).add_ref(eval)`.
- Ownership: `set_optimizer_ref` / `add_ref` borrow; no factory wrappers.

Build and run the regression (requires GPW + HF + PW + regressions):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_GPW=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-variable-cell-molopt
  ./build/regressions/reg_gpw-variable-cell-molopt
  # or: cmake --build build --target run_reg_gpw-variable-cell-molopt
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `compute_stress`, `stress_use_fd`, `stress_axes`, `last_stress_ha_bohr3()` |
| `include/atomiverse/cpu/minimize/anisotropic_cell_relax.h` | Diagonal cell optimizer + axis mask |
| `include/atomiverse/cpu/driver/criteria.h` | `StressConverged` |
| `include/atomiverse/cpu/minimize/minimize.h` | Minimize preset |
| `src/cpu/gpw/internal/gpw_stress.cpp` | Analytic fixed-\(P\) + FD oracle |
| `regressions/gpw/gpw-variable-cell-molopt/reg_gpw-variable-cell-molopt.cpp` | Product gate |
| `regressions/gpw/gpw-variable-cell-molopt/REFERENCE.md` | Offline CP2K pins + fair wall |
| `tools/cp2k_oracle/inputs/gpw-variable-cell-molopt/` | `chain_cellopt.inp`, full-MOLOPT secondary, E(a) scan |

## Validation (vs CP2K)

Source:
[`regressions/gpw/gpw-variable-cell-molopt/REFERENCE.md`](../../../regressions/gpw/gpw-variable-cell-molopt/REFERENCE.md).  
Oracle: offline **CP2K 2026.1** (`cp2k.psmp`, EasyBuild
`2026.1-gmpflf-2026.03`), Arrhenius epyc login, 1 rank × 4 OMP. Inputs under
`tools/cp2k_oracle/inputs/gpw-variable-cell-molopt/`. Hard-coded pins in the
regression binary — **never** live CP2K from ctest.

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine | CP2K 2026.1 Quickstep GPW |
| Twin | `chain_cellopt.inp` (SR-MOLOPT, `CELL_OPT`, `CONSTRAINT YZ`, `FIXED_ATOMS`, analytical stress) |
| Host / threads | Arrhenius epyc; `OMP_NUM_THREADS=4`, `OMP_PROC_BIND=close` |
| Atomiverse path | Minimize + AnisotropicCellRelax + StressConverged + GPWEvaluator |

### Headline pin — SR-MOLOPT CELL_OPT (e2e twin)

| Quantity | a₀ = 3.328 (scan start) | a₀ = 3.24 (fair twin) |
| -------- | ----------------------: | --------------------: |
| \(a^\star\) [Å] | **3.200237** | 3.200292 |
| \(E^\star\) [Ha] | **−17.184075549824836** | −17.184075557775 |
| Wall (CP2K timing table) | 9.55 s | **5.67 s** |

Regression hard-codes \(a^\star = 3.200237\),
\(E^\star = -17.184075549824836\) Ha (a₀ = 3.328 oracle run; \(a^\star\)
agrees with the a₀ = 3.24 fair twin to \(< 10^{-4}\) Å).

### Secondary pin — full DZVP-MOLOPT-GTH CELL_OPT (not e2e)

| Quantity | Value |
| -------- | ----: |
| \(a^\star\) | 3.200252 Å |
| \(E^\star\) | −17.185176472246 Ha |
| Wall (timing table) | 23.5 s |

ATV cannot dual-gate this basis inside ≤ 60 s today (periodic collocation).

### \(E(a)\) curvature (full MOLOPT ENERGY scan; `bulk_mod` source)

| \(a\) [Å] | \(E\) [Ha] |
| --------: | ---------: |
| 3.16 | −17.185048204416361 |
| 3.20 | −17.185176430631692 |
| 3.24 | −17.185069177601907 |

\(d^2E/da^2 \approx 0.147\,\mathrm{Ha/Å}^2\) →
\(d\sigma_{xx}/d\varepsilon \approx 0.20\,\mathrm{eV/Å}^3\) →
`bulk_mod = 0.20`.

### Hard gates (measured 2026-07-25, Release, 4 OMP)

| Gate | Tolerance | Measured |
| ---- | --------- | -------- |
| relax converged | — | `stress_converged` ✓ |
| \(\lvert a_{\mathrm{ATV}}-a^\star_{\mathrm{CP2K}}\rvert\) | ≤ 0.02 Å | **0.0012 Å** (3.201465 vs 3.200237) |
| \(b\), \(c\) untouched | bit-identical | 8.0 / 8.0 ✓ |
| residual \(\lvert\sigma_{xx}\rvert\) | ≤ \(6\times 10^{-4}\) eV/Å³ | **5.1×10⁻⁵** eV/Å³ |
| \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}\rvert\) | ≤ \(1\times 10^{-3}\) Ha | **1.4×10⁻⁴** Ha |
| wall | ≤ 60 s | **42.5 s** |

| Quantity | Atomiverse / pin | CP2K (offline) | Tol | Notes |
| -------- | ---------------: | -------------: | --- | ----- |
| \(a^\star\) [Å] | 3.201465 (measured) | **3.200237** | 0.02 Å | chain axis only |
| \(E_{\mathrm{tot}}\) [Ha] | (within gate) | **−17.184075549824836** | \(1\times 10^{-3}\) Ha | pin channel Ha |
| \(\sigma_{xx}\) residual [eV/Å³] | **5.1×10⁻⁵** | ~0 at CELL_OPT min | \(6\times 10^{-4}\) | criterion stol \(5\times 10^{-4}\) |
| \(b\), \(c\) [Å] | 8.0, 8.0 | 8.0, 8.0 | bit-id | CONSTRAINT YZ |

Offline oracle sketch (host with CP2K module; **not** ctest):

```bash
module load buildenv-gcc/2026.03-mpich
export CP2KROOT=/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.1-gmpflf-2026.03
export CP2K_DATA_DIR=$CP2KROOT/share/cp2k/data
export PATH=$CP2KROOT/bin:$PATH
export OMP_NUM_THREADS=4
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
cp $REPO/tools/cp2k_oracle/inputs/gpw-variable-cell-molopt/chain_cellopt.inp .
mpirun -np 1 cp2k.psmp -i chain_cellopt.inp -o cellopt.out
# CELL | Vector a  → a* ; ENERGY| Total FORCE_EVAL → E*
```

## Performance

Source: `REFERENCE.md` fair wall on Arrhenius epyc login (same host class
as pin), 1 rank × 4 OMP, `OMP_PROC_BIND=close`. Queue / handoff record
ratio **~7.5**.

- **ATV:** `./build/regressions/reg_gpw-variable-cell-molopt` — outer
  CELL_OPT with analytic stress per SCF (measured **42.5 s**).
- **CP2K fair denominator:** same a₀ = **3.24 Å** CELL_OPT twin wall
  from timing table (**5.67 s**), not the longer a₀ = 3.328 scan start.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| Water-chain CELL_OPT, SR-MOLOPT, a₀=3.24 | **5.67** | **42.5** | **7.5** | ✓ | Primary fair gate |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\).
`perf_exempt = false`. `within_10x = true`.

Cost drivers vs CP2K: ATV periodic MOLOPT collocation + multiple outer
stress-SCF cycles; analytic fixed-\(P\) already removes the multi-SCF
strain-FD tax that forced the ice pivot to compact GTH-PADE.

## Limitations

- **SR-MOLOPT product twin** — full DZVP-MOLOPT-GTH CELL_OPT is offline
  pinned but not dual-gated in ≤ 60 s; re-enable when collocation cost
  drops (GPW perf campaign).
- **Ions fixed** — proves cell machinery + GPW stress axes mask; not a
  free GEO_OPT+CELL_OPT production recipe.
- **Orthorhombic diagonal only** — no triclinic shear; masked axes are
  hard zeros on the published stress for constrained runs.
- **Closed-shell Γ RKS** — no UKS / k-point cell relax on this slug
  (k-stress product is a separate capability:
  `gpw-kpoint-stress`).
- **No dispersion** on the cell path here (D3/D4/TS/MBD are other rows).
- Public Frame energy remains **eV**; Ha pins are internal / citation
  channels only.
- Unit tests under `tests/cpu/minimize/` exercise axis masks; **e2e
  CP2K parity is exclusively the regression binary**.

## References

- Offline pin + fair wall:
  `regressions/gpw/gpw-variable-cell-molopt/REFERENCE.md`
- Regression:
  `regressions/gpw/gpw-variable-cell-molopt/reg_gpw-variable-cell-molopt.cpp`
- Twin inputs: `tools/cp2k_oracle/inputs/gpw-variable-cell-molopt/`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Queue / handoff / loop: `docs/developer/cp2k-method-queue.md`,
  `docs/developer/cp2k-method-handoff.md`,
  `docs/developer/cp2k-method-loop.md`
- Prior foothold (GTH-PADE pivot):
  `regressions/gpw/gpw-variable-cell-ice/`
- Analytic stress product path: [gpw-analytic-stress](gpw-analytic-stress.md),
  strain-FD history: [gpw-stress-water](gpw-stress-water.md)
- Narrative GPW: `docs/tutorial/ch09-gpw.md`; optimization:
  `docs/tutorial/ch12-geometry-optimization.md`
- Roadmap: `docs/roadmap/dft.md` (variable-cell / CELL_OPT prose)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
- Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985) (stress context)
- CP2K Quickstep `CELL_OPT` / `STRESS_TENSOR` / `CONSTRAINT` manual
