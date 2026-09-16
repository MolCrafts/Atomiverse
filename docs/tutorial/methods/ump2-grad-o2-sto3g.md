# UMP2 analytic gradient — triplet O₂ / STO-3G

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ump2-grad-o2-sto3g` |
| Regression | `regressions/scf/ump2-grad-o2-sto3g/` |
| Public API | `cpu::scf::UHF::mp2_forces(Frame&, SCFInitialGuess*)` → `GradientResult` |
| Energy sibling | `regressions/scf/ump2-o2-sto3g` |
| Roadmap | post-HF UMP2 forces (PySCF campaign) |

## Summary

Analytic nuclear forces on the **total UMP2 energy** \(E_{\mathrm{UHF}} +
E_{\mathrm{corr}}\) for high-spin triplet O₂ (\(n_\alpha=9\), \(n_\beta=7\))
with Cartesian STO-3G. Public path is the concrete unrestricted method
`cpu::scf::UHF` on a `cpu::Frame`; parameters are constructor args / members —
no `*Config` bag. Forces return as `GradientResult` ([eV/Å]); the regression
converts to Ha/Bohr for the pin. Multi-minima O₂ UHF is warm-started from ROHF
densities (same path as the energy sibling). End-to-end gate:
`regressions/scf/ump2-grad-o2-sto3g` (`reg_ump2-grad-o2-sto3g`). Offline oracle:
PySCF 2.14.0 `mp.UMP2(...).nuc_grad_method().kernel()` pinned in `REFERENCE.md`
(never live in ctest). Fair wall
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 1.37\).

## Method

**Reference.** Unrestricted Hartree–Fock (UHF) for open-shell triplet O₂. UMP2
is second-order Møller–Plesset perturbation theory in spin orbitals on that
UHF reference (unprojected; residual \(\langle S^2\rangle\) contamination is
accepted as in raw PySCF `mp.UMP2`). No frozen core.

\[
E_{\mathrm{UMP2}} = E_{\mathrm{UHF}} + E_{\mathrm{corr}}^{\mathrm{UMP2}},
\qquad
E_{\mathrm{corr}}^{\mathrm{UMP2}} = E_{\mathrm{SS}} + E_{\mathrm{OS}}.
\]

**Gradient composition.** Analytic UMP2 nuclear gradient (Pople / Helgaker
response form; PySCF `pyscf.grad.ump2`):

- Spin-resolved **Lagrangian 1-RDM** (unrelaxed MP2 density plus orbital-response
  Z-vector on the unrestricted \(A+B\) coupling blocks).
- **Non-separable 2-RDM** from the \(T_2\) amplitudes.
- Contraction of those densities with **skeleton integral derivatives**
  (\(\partial h/\partial R\), \(\partial S/\partial R\), \(\partial(ij|kl)/\partial R\)).

Both Atomiverse and PySCF deliver **analytic** UMP2 gradients on this pin;
agreement to \(\sim 10^{-6}\)–\(10^{-8}\) Ha/Bohr is expected on the minimal
system. Missing response or 2-RDM terms typically fail by \(\ge 10^{-3}\)
Ha/Bohr.

**Approximations / scope of this case.**

- Canonical UMP2 only: **no** frozen core, **no** RI/DF, **no** SCS scaling
  on this path (`mp2_forces` rejects ROHF, RI, `frozen > 0`, non-default SCS
  scales).
- Basis: STO-3G **Cartesian** (`cart=True`).
- Analytic nuclear gradient only; no UMP2 Hessian / vib analysis on this slug.
- Cold UHF Hcore/MINAO often lands on a higher local solution for O₂; the
  regression **warm-starts** from ROHF \(\alpha/\beta\) densities so UHF
  matches the PySCF minimum used for the pin (same multi-minima note as
  `ump2-o2-sto3g`).

## Implementation

Headers: `include/atomiverse/cpu/scf/methods.h` (concrete `UHF` / `ROHF`),
`include/atomiverse/cpu/scf/mean_field_method.h` (`mp2_forces` API),
`include/atomiverse/cpu/scf/gradient_result.h`,
`include/atomiverse/cpu/frame/frame.h`. Private analytic kernel:
`include/atomiverse/cpu/scf/internal/mp2_gradient.h`
(`ump2_analytic_forces`). Units: public forces [eV/Å]; pins [Ha/Bohr] via
`units::convert`.

```cpp
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"

cpu::Frame frame(2);
// O₂: Z=8, r = 1.2075 Å along z; cubic box 20 Å
frame.span(kAtom_type)[0] = frame.span(kAtom_type)[1] = 8;
frame.span(kAtom_z)[0] = 0.0;
frame.span(kAtom_z)[1] = 1.2075;
frame.box = cpu::Box::cubic(20.0);

// Warm-start: ROHF densities → UHF (multi-minima for O₂)
cpu::scf::ROHF rohf(/*n_alpha=*/9, /*n_beta=*/7);
rohf.basis = cpu::scf::BasisSet::STO3G;
rohf.tol = 1e-12;
rohf.max_iter = 200;
rohf.run(frame);

using Spin = cpu::scf::SlaterDeterminant::Spin;
cpu::scf::SCFInitialGuess guess;
guess.P_alpha = rohf.determinant().density(Spin::alpha);
guess.P_beta  = rohf.determinant().density(Spin::beta);
guess.has_alpha = true;
guess.has_beta  = true;

cpu::scf::UHF uhf(/*n_alpha=*/9, /*n_beta=*/7);
uhf.basis = cpu::scf::BasisSet::STO3G;
uhf.tol = 1e-12;
uhf.max_iter = 200;
const cpu::scf::GradientResult res = uhf.mp2_forces(frame, &guess);
// res.forces(A, d) in eV/Å; res.energy total UMP2 [eV]
// Ha/Bohr: -units::convert(f, Force::eVPerAngstrom, Force::HartreePerBohr)
// (sign: GradientResult stores forces = −∇E)
```

Build / run the regression (CPU login toolchain; see `AGENTS.md`):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_ump2-grad-o2-sto3g -j$(nproc)
  ./build/regressions/reg_ump2-grad-o2-sto3g
'
```

Consumer source:
`regressions/scf/ump2-grad-o2-sto3g/reg_ump2-grad-o2-sto3g.cpp`.

## Validation (vs PySCF)

Source: [`regressions/scf/ump2-grad-o2-sto3g/REFERENCE.md`](../../../regressions/scf/ump2-grad-o2-sto3g/REFERENCE.md)
(offline pin; PySCF never in ctest).

| Field | Value |
| ----- | ----- |
| Geometry (Å) | O(0, 0, 0); O(0, 0, 1.2075) |
| Basis | STO-3G, Cartesian |
| Method | UHF (spin \(2S=2\) triplet) → canonical UMP2 (no frozen core) |
| Convergence | `conv_tol = 1e-12` |
| PySCF | **2.14.0** |
| Oracle | `mp.UMP2(mf).run(); m2.nuc_grad_method().kernel()` |

Energy pins (same geometry / basis as energy sibling `ump2-o2-sto3g`):

| Quantity | Pin [Ha] | Notes |
| -------- | -------: | ----- |
| \(E_{\mathrm{UHF}}\) | −147.633946785450 | REFERENCE |
| \(E_{\mathrm{corr}}\) (UMP2) | −0.096393144130 | REFERENCE |
| \(E_{\mathrm{UMP2}}\) | −147.730339929580 | \(E_{\mathrm{UHF}}+E_{\mathrm{corr}}\) |

Analytic gradient \(\partial E/\partial R\) [Ha/Bohr] (collinear diatomic on
\(z\); \(x,y\) ~0):

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| O (0) | 0 | 0 | **+1.665132663664e−01** |
| O (1) | 0 | 0 | **−1.665132663664e−01** |

| Gate | Tol | Notes |
| ---- | --: | ----- |
| Element-wise \(\lvert g_{\mathrm{ATV}} - g_{\mathrm{PySCF}}\rvert\) | **\(1\times 10^{-5}\)** Ha/Bohr | reg gate; expected match \(\sim 10^{-6}\)–\(10^{-8}\) |
| Translational invariance \(\lvert\sum_A g_A\rvert_{\max}\) | **\(1\times 10^{-8}\)** Ha/Bohr | \(\sum_A g_A \approx 0\) to \(\sim 10^{-15}\) on pin |

Why \(1\times 10^{-5}\) Ha/Bohr: covers packing / AO-slice assembly noise
without masking a missing Z-vector or 2-RDM term (those errors are typically
\(\ge 10^{-3}\) Ha/Bohr).

Oracle re-pin:

```bash
# tools/setup_pyscf_env.sh  # once
python tools/pyscf_oracle/pin_scf.py ump2 --geom o2 --basis sto-3g \
    --spin 2 --conv-tol 1e-12 --grad
```

## Performance

Source: `REFERENCE.md` fair-bench block. Handoff ratio **1.37**
(the case `REFERENCE.md`).

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `ump2_grad` (UHF + UMP2 + analytic grad) | **0.296** | **0.404** | **1.37** | ✓ | matched method |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\).

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python tools/pyscf_oracle/fair_bench.py --method ump2_grad --geom o2 \
    --basis sto-3g --spin 2 --repeat 3 -- \
    ./build/regressions/reg_ump2-grad-o2-sto3g
```

**Twin caveats.** Reg binary includes ROHF warm-start, full element-wise +
translational checks, and unit conversion; PySCF side is UHF + UMP2 +
`nuc_grad_method` only. Walls are short (\(\sim 0.3\)–\(0.4\) s) — ratio noise
from OS jitter can be material at this scale. Energy sibling `ump2-o2-sto3g`
has a separate fair wall (energy-only); do not conflate that ratio with this
`ump2_grad` pair.

## Limitations

- Single textbook case: O₂ triplet, STO-3G cart, canonical UMP2 only.
- `mp2_forces` rejects ROHF / ROMP2 gradients, RI/DF, frozen core, and SCS
  scale deviations (header contract; no silent fallback).
- No UMP2 Hessian / vibrational analysis on this slug.
- Spherical (non-Cartesian) AO packs not supported on this gradient path.
- CUDA UMP2 / UMP2 forces: not a product path.
- Multi-minima UHF for O₂ requires an explicit density guess (ROHF warm-start
  in the regression); cold UHF may miss the PySCF minimum.
- Perf rollup table may lag REFERENCE until a bulk refresh of
  ``.

## References

- Offline pin: `regressions/scf/ump2-grad-o2-sto3g/REFERENCE.md`
- Energy sibling pin: `regressions/scf/ump2-o2-sto3g/REFERENCE.md`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`;
  `tools/pyscf_oracle/pin_scf.py`, `tools/setup_pyscf_env.sh`
- Energy handbook: [`ump2-o2-sto3g.md`](ump2-o2-sto3g.md)
- Pople, Krishnan, Schlegel, Binkley, *Int. J. Quantum Chem. Symp.* **13**, 225
  (1979)
- Helgaker, Jørgensen, Olsen, *Molecular Electronic-Structure Theory* (response /
  gradient chapters)
- PySCF `pyscf.grad.ump2`; `mp.UMP2`
