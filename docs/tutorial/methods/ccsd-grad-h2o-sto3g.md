# CCSD nuclear gradient — H₂O / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ccsd-grad-h2o-sto3g` |
| Regression | `regressions/scf/ccsd-grad-h2o-sto3g/` |
| Public API | `cpu::scf::RHF::ccsd_forces(Frame&)` → `GradientResult` |
| Roadmap | [Post-HF §3 Coupled cluster](../../roadmap/post-hf.md) |

## Summary

Closed-shell **CCSD nuclear forces** on H₂O / STO-3G after a converged RHF
reference. Product surface is `RHF::ccsd_forces(frame)`: drives a fresh RHF,
then the same RHF→CCSD energy path as `RHF::ccsd` / `ccsd_energy`, and returns
`GradientResult{forces, energy}` with forces in **eV/Å** and total CCSD energy
in **eV** on the public surface (internal pin math is Ha/Bohr). This foothold
implements \(\partial E_{\mathrm{CCSD}}/\partial R\) as a **central finite
difference** of the analytic CCSD total energy (\(h = 10^{-4}\) Bohr), not the
density-based \(\Lambda\) / Lagrangian analytic path. Offline oracle pin is
PySCF 2.14.0 **analytic** `cc.CCSD` + `nuc_grad_method().kernel()` (never live
in ctest). End-to-end gate: `regressions/scf/ccsd-grad-h2o-sto3g` → binary
`reg_ccsd-grad-h2o-sto3g`. Fair-bench ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 1.665\) (within 10×).

## Method

**Energy surface.** Same closed-shell CCSD as the energy product path:

\[
\lvert\Psi_{\mathrm{CCSD}}\rangle = e^{T_1+T_2}\lvert\Phi_0\rangle,\qquad
E_{\mathrm{CCSD}} = \langle\Phi_0\rvert H e^{T}\lvert\Phi_0\rangle_{\mathrm{c}},
\]

with spin-orbital Gauss–Stanton intermediates (Stanton et al. 1991; Gauss &
Stanton 1995). Occupied spin-orbitals: α then β over spatial occupied MOs;
virtuals likewise.

**Gradient (this foothold).** Central finite difference of the total CCSD
energy on nuclear Cartesian displacements in Bohr:

\[
\frac{\partial E}{\partial R_{A,d}}
\approx
\frac{E(R_{A,d}+h) - E(R_{A,d}-h)}{2h},\qquad h = 10^{-4}\,\mathrm{Bohr}.
\]

Each displaced geometry re-runs the same RHF→`ccsd_energy` path. Public
`ccsd_forces` negates the energy derivatives into **forces** and converts to
eV/Å for Frame/`GradientResult`. Regression checks convert back to Ha/Bohr for
comparison with the PySCF analytic pin.

**Oracle (PySCF).** Analytic nuclear gradient via the \(\Lambda\) equations and
Lagrangian (Gauss & Stanton 1996; PySCF `pyscf.grad.ccsd`). Element-wise FD vs
analytic agreement on this system is \(\le 4\times 10^{-8}\) Ha/Bohr (well under
the product gate).

**Key approximations / scope of this foothold:**

- Single-reference, closed-shell RHF only (no UCCSD / ROCCSD forces).
- **No** density-based analytic gradient yet (\(\Lambda\), Lagrangian 2-RDM,
  Z-vector) — product follow-up for performance / large \(N\).
- No frozen core, no RI/DF-CCSD gradients (API throws).
- No CCSD(T) forces, no EOM-CCSD gradients, no Hessian.
- Canonical MO integrals; no DF/CD/F12/local CC.
- H₂O/STO-3G Cartesian: molecule in the yz-plane → \(x\) components ~0.

## Implementation

Public headers only for consumers (`Frame` + concrete `RHF`; no `*Config`):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/gradient_result.h"
#include "atomiverse/cpu/scf/methods.h"

cpu::Frame frame(3);
// O(0,0,0.1173); H(0,±0.7572,−0.4692) Å; Z = 8,1,1; Box::cubic(20)
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.max_iter = 200;

const cpu::scf::GradientResult res = rhf.ccsd_forces(frame);
// res.forces(A, d) [eV/Å]; res.energy [eV] (total CCSD)
// Compare vs pin: -convert(force, eV/Å → Ha/Bohr)
```

`GradientResult` fields (`include/atomiverse/cpu/scf/gradient_result.h`):
`forces` \([n_{\mathrm{atoms}}, 3]\) eV/Å, `energy` eV.

**Build / run regression** (CPU login; modules as in AGENTS.md):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_ccsd-grad-h2o-sto3g -j$(nproc)
  ./build/regressions/reg_ccsd-grad-h2o-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `RHF::ccsd_forces(Frame&)` |
| `include/atomiverse/cpu/scf/gradient_result.h` | Public `GradientResult` |
| `src/cpu/scf/internal/ccsd_gradient.cpp` | FD of CCSD energy |
| `src/cpu/scf/mean_field_method.cpp` | Public entry (RHF-only gates) |
| `regressions/scf/ccsd-grad-h2o-sto3g/reg_ccsd-grad-h2o-sto3g.cpp` | E2E consumer |

## Validation (vs PySCF)

Source:
[`regressions/scf/ccsd-grad-h2o-sto3g/REFERENCE.md`](../../../regressions/scf/ccsd-grad-h2o-sto3g/REFERENCE.md)
(offline pin; PySCF never in ctest). Protocol:
`docs/tutorial/appendix-pyscf-oracle.md`.

| Setting | Value |
| ------- | ----- |
| Molecule | H₂O, charge 0, singlet |
| Geometry (Å) | O(0, 0, 0.1173); H(0, ±0.7572, −0.4692) |
| Basis | STO-3G, **Cartesian** (`cart=True`) |
| SCF | RHF, `conv_tol=1e-12` |
| CCSD | `pyscf.cc.CCSD(mf)`, `conv_tol=1e-10`; no frozen core |
| Gradient | `mycc.nuc_grad_method().kernel()` (analytic) |
| PySCF | 2.14.0 |
| Command | `pin_scf.py ccsd --geom water --basis sto-3g --grad --conv-tol 1e-12` |

### Energies (diagnostic / same surface)

| Quantity | Pin | Notes |
| -------- | --: | ----- |
| \(E_{\mathrm{HF}}\) | −74.963023138463 Ha | RHF reference |
| \(E_{\mathrm{corr}}\) (CCSD) | −0.049438562975 Ha | |
| \(E_{\mathrm{CCSD}}\) | −75.012461701437 Ha | \(E_{\mathrm{HF}}+E_{\mathrm{corr}}\) |

### Analytic gradient pin \({\partial E}/{\partial R}\) [Ha/Bohr]

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| O | 0 | 0 | −1.106969560091×10⁻¹ |
| H | 0 | −4.698911449636×10⁻² | +5.534847800455×10⁻² |
| H | 0 | +4.698911449636×10⁻² | +5.534847800455×10⁻² |

\(\sum_A \partial E/\partial R_A \approx (0,0,0)\) to ~10⁻¹⁵ Ha/Bohr.

| Check | Tol | Notes |
| ----- | --- | ----- |
| Element-wise vs PySCF analytic pin | **1e-5 Ha/Bohr** | Product gate; FD agrees \(\le 4\times 10^{-8}\) on this system |
| Translational invariance \(\lvert\sum_A g_A\rvert_{\max}\) | **1e-8 Ha/Bohr** | Product gate |

Oracle repro (human / offline only):

```bash
bash tools/setup_pyscf_env.sh
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py ccsd \
    --geom water --basis sto-3g --grad --conv-tol 1e-12
```

## Performance

Source: fair wall in
[`regressions/scf/ccsd-grad-h2o-sto3g/REFERENCE.md`](../../../regressions/scf/ccsd-grad-h2o-sto3g/REFERENCE.md)
(median of 3; campaign handoff ratio **1.665**). Gate:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\).

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `ccsd_grad` (H₂O/STO-3G) | 0.241 | 0.401 | **1.665** | ✓ | ATV = FD of \(E_{\mathrm{CCSD}}\) (many energy solves); PySCF = analytic \(\Lambda\) path; not a large-\(N\) predictor |

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method ccsd_grad --geom water --basis sto-3g --repeat 3 -- \
    ./build/regressions/reg_ccsd-grad-h2o-sto3g
```

No ⚡ optimization pass required on this case (\(r < 10\)).

## Limitations

- **FD foothold**, not \(\Lambda\)-based analytic CCSD gradient; cost scales as
  \(O(6N_{\mathrm{atoms}})\) full CCSD energy evaluations plus SCF.
- Closed-shell RHF reference only; open-shell / RI / frozen-core throw.
- No CCSD(T) forces, no EOM-CC / ADC response gradients, no Hessian.
- **CUDA / Kokkos:** ○ (CPU only).
- Energy sibling: `ccsd-h2-sto3g` / water CCSD(T) `ccsd-t-h2o-sto3g`; analytic
  density-based path is the product follow-up (Gauss & Stanton 1996; PySCF
  `pyscf.grad.ccsd`).
- **Perf twin** compares FD (ATV) to analytic (PySCF); \(r \sim 1.7\) is
  acceptable on this tiny basis but analytic \(\Lambda\) is the expected large-\(N\)
  win for both engines.

## References

- Offline pin: `regressions/scf/ccsd-grad-h2o-sto3g/REFERENCE.md`
- Regression consumer: `regressions/scf/ccsd-grad-h2o-sto3g/reg_ccsd-grad-h2o-sto3g.cpp`
- Energy sibling page: [ccsd-h2-sto3g](ccsd-h2-sto3g.md)
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Roadmap: `docs/roadmap/post-hf.md` (§3 Coupled cluster)
- Public API: `include/atomiverse/cpu/scf/mean_field_method.h` (`ccsd_forces`),
  `include/atomiverse/cpu/scf/gradient_result.h`
- J. Gauss, J. F. Stanton, *J. Chem. Phys.* **104**, 2574 (1996) (analytic
  CCSD gradient)
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure Theory*
  (response / gradient chapters)
- PySCF `pyscf.grad.ccsd`
