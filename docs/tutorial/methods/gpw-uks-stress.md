# Γ-UKS GPW analytic cell stress — O₂ / PBE / DZVP-MOLOPT-GTH (MT)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| slug | `gpw-uks-stress` |
| Regression | `regressions/gpw/gpw-uks-stress/` |
| Public API | `cpu::gpw::GPWEvaluator` (`n_alpha`/`n_beta`, `XcKind::PBE`, `compute_stress`, `stress_use_fd=false`); free `gpw_stress` / `gpw_stress_fd` |
| Related narrative | [Ch. 9 GPW](../ch09-gpw.md) |

## Summary

**Unrestricted Kohn–Sham (UKS) Gaussian-and-plane-waves (GPW)** **cell
stress** at the **Γ point** for open-shell systems: spin-resolved **PBE
GGA**, dual-spin density matrices, and the product **Hellmann–Feynman
fixed-\(P\)** stress path (one UKS SCF + central FD of the non-self-consistent
Lagrangian under orthorhombic strain). This page pins **triplet O₂** energy
and the analytic-vs-multi-SCF-FD stress dual-gate against offline **CP2K
2026.1 Quickstep** (energy from the sibling `gpw-uks-forces` ENERGY_FORCE
pin; stress twin input `STRESS_TENSOR NUMERICAL`).

Public path: construct `cpu::gpw::GPWEvaluator` with one `Kind` for O
(lookup constructors `GaussianBasis` / `GthPotential`), set members
(`n_alpha=7`, `n_beta=5`, `cutoff_ry`, `xc=PBE`, `nonperiodic`,
`compute_stress=true`, `stress_use_fd=false`), bind into `cpu::Evaluate`.
Frame energy is **eV** on `kEnergy`; pin channel is
`last_energy_hartree()` [Ha] and `last_stress_ha_bohr3()` [Ha/Bohr³]; public
stress on Frame is `kStress` [eV/Å³] Voigt. Product gate:
`reg_gpw-uks-stress`. Fair wall is **`perf_exempt`** (no CP2K NUMERICAL
stress wall on the pin host; dual-gate multi-SCF FD dominates wall).

Related: closed-shell analytic stress [gpw-analytic-stress](gpw-analytic-stress.md);
strain-FD water pin [gpw-stress-water](gpw-stress-water.md); energy/forces
sibling `regressions/gpw/gpw-uks-forces/`; PADE UKS energy foothold
[gpw-uks-o2-gamma](gpw-uks-o2-gamma.md). **Out of loop scope:** multi-k UKS
stress (residual; analytic path throws), molecular post-HF, classical MD.

## Method

**GPW (Quickstep family).** Wavefunction in a contracted Gaussian AO basis;
density collocation on a uniform real-space grid; Hartree via FFT Poisson;
XC on the grid; potential integrated back into the AO basis. Core electrons
removed by analytic **GTH** pseudopotentials. Grid resolution set by the
plane-wave cutoff in Rydberg. Citation: Lippert, Hutter, Parrinello,
*Mol. Phys.* **92**, 477 (1997); VandeVondele et al., *Comput. Phys. Commun.*
**167**, 103 (2005).

**UKS / collinear spin.** With \(N_\alpha \neq N_\beta\), build
\(\rho_\alpha\), \(\rho_\beta\); Hartree from
\(\rho=\rho_\alpha+\rho_\beta\); spin-resolved PBE GGA XC; diagonalize
separate \(H_\alpha\) and \(H_\beta\). Occupations are explicit integer
members `n_alpha` / `n_beta` (not a multiplicity bag). Valence bookkeeping
for this pin: O with GTH-PBE-q6 → 6 valence e⁻ per O → 12 total; multiplicity
3 ⇒ \(N_\alpha=7\), \(N_\beta=5\).

**Product cell stress (Hellmann–Feynman fixed-\(P\)).** After one converged
UKS SCF at the unstrained cell:

1. Keep \(P_\alpha,P_\beta\) and energy-weighted density matrices
   \(W_\sigma=\sum_i^{n_\sigma}\varepsilon_i^\sigma c_i^\sigma(c_i^\sigma)^\top\).
2. Apply dimensionless diagonal strain \(h\) at **fixed fractional
   coordinates** (scale \(L_a\) and Cartesian \(r_a\) by \(1\pm h\)).
3. Rebuild \(T\), \(V_{\mathrm{pp}}\), \(S\), grid, Poisson, spin XC; form the
   **non-self-consistent** Lagrangian
   \(E_L[P_\alpha,P_\beta;W_\alpha,W_\beta;\varepsilon]\) (do **not** re-run SCF).
4. Central difference:

\[
\sigma_{aa}
=
-\Omega^{-1}
\frac{\partial E_L}{\partial\varepsilon_{aa}}
\approx
-\Omega^{-1}
\frac{E_L(+h_a)-E_L(-h_a)}{2h},
\qquad a\in\{x,y,z\},
\]

with \(E\) in Hartree and \(\Omega\) in Bohr³ (CP2K `STRESS_TENSOR` sign).
Voigt order:
\((\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})\).
Default \(h = 2\times 10^{-3}\) (`stress_strain_h`). The \(-\mathrm{Tr}(W\,S)\)
piece is the Pulay contribution. Shear components are **exactly 0** in v1
(orthorhombic cell only).

**Oracle twin.** Multi-SCF strain-FD (`gpw_stress_fd`) recomputes a full UKS
SCF at each strained geometry (6 SCFs for three diagonal axes). Decisive
product accuracy gate is max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\|
on the diagonals — same-method twin of the energy functional, independent of
CP2K multigrid residual.

**Isolated electrostatics.** `nonperiodic = true` selects **Martyna–Tuckerman**
Poisson (`PERIODIC NONE` / `PSOLVER MT`). The molecule must sit **centered**
in the cell. Citation: Martyna & Tuckerman, *J. Chem. Phys.* **110**, 2810
(1999).

**Key exclusions (closed-spec / residual):**

- **Multi-k UKS stress** — analytic path still throws; residual on the CP2K
  queue (forces at multi-k UKS are productized separately).
- No triclinic / off-diagonal shear strain in v1.
- Absolute CP2K `STRESS_TENSOR NUMERICAL` Voigt digits **not** hard-coded
  (CP2K module unavailable on pin host 2026-07-25); energy soft twin reuses
  `gpw-uks-forces`.
- No ROKS, hybrids, DFT+U, or dispersion stress on this page.
- No public `*Config` bags / `make_*` factories; never live CP2K from ctest.
- CUDA GPW out of scope for this loop page.

## Implementation

Public surface only (`cpu::Frame`, concrete evaluator members; **no**
`*Config` bag, **no** `make_*` / `build_*` factories). Match the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;

// O₂, R = 1.2075 Å, centered in 12 Å cubic cell (MT requires centering)
cpu::Frame frame(2);
constexpr double L = 12.0;
constexpr double R = 1.2075;
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 8;
    frame.span(kAtom_mass)[i] = 15.999;
    frame.span(kAtom_x)[i] = 0.5 * L;
    frame.span(kAtom_y)[i] = 0.5 * L;
    frame.span(kAtom_fx)[i] = frame.span(kAtom_fy)[i] =
        frame.span(kAtom_fz)[i] = 0;
}
frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
frame.box = cpu::Box::cubic(L);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

cpu::gpw::GPWEvaluator eval({O});
eval.cutoff_ry = 400.0;
eval.max_iter = 200;
eval.tol = 1e-7;
eval.mix = 0.4;
eval.xc = cpu::gpw::XcKind::PBE;
eval.nonperiodic = true;
eval.compute_forces = false;
eval.compute_stress = true;
eval.stress_use_fd = false;   // product analytic fixed-P (default)
eval.stress_strain_h = 2e-3;
eval.n_alpha = 7;             // multiplicity 3, 12 valence e⁻
eval.n_beta = 5;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

const double e_ha = eval.last_energy_hartree();       // [Ha]
const double e_ev = frame.meta.get(kEnergy);          // [eV]
const auto& s = eval.last_stress_ha_bohr3();          // Voigt [Ha/Bohr³]
// frame.meta.get(kStress) is eV/Å³ Voigt
const double s2 = eval.last_s2();                     // Löwdin ⟨S²⟩
```

Multi-SCF FD oracle (public free function; dual-gate only, not the product
default):

```cpp
// After method.build(atoms) with the same UKS members…
const auto sfd = cpu::gpw::gpw_stress_fd(
    atoms, method, /*h=*/2e-3, /*axes*/ {true, true, true},
    /*C_guess*/ nullptr);
```

Build (GPW needs HF + PW + FFTW3) and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-uks-stress
  ./build/regressions/reg_gpw-uks-stress
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Evaluator | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Free stress API | `include/atomiverse/cpu/gpw/gpw_solver.h` (`gpw_stress` / `gpw_stress_fd`) |
| Stress implementation | `src/cpu/gpw/internal/gpw_stress.cpp` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_stress.cpp` |
| Regression | `regressions/gpw/gpw-uks-stress/reg_gpw-uks-stress.cpp` |
| Offline pins | `regressions/gpw/gpw-uks-stress/REFERENCE.md` |
| CP2K twin input | `tools/cp2k_oracle/inputs/gpw-uks-stress/o2_uks_stress_num.inp` |
| Energy pin provenance | `regressions/gpw/gpw-uks-forces/REFERENCE.md` |

## Validation (CP2K)

Source: `regressions/gpw/gpw-uks-stress/REFERENCE.md` (offline pin; **never**
live CP2K in ctest). Engine: **CP2K 2026.1** Quickstep. Energy digits from
sibling ENERGY_FORCE twin `gpw-uks-forces` (2026-07-25). Absolute NUMERICAL
stress Voigt digits **not invented** (CP2K module absent on pin host); the
decisive gate is internal analytic vs multi-SCF FD.

**Geometry / protocol (must match twin).**

| Item | Value |
| ---- | ----- |
| Molecule | O₂, bond along *z*, **centered** |
| \(R(\mathrm{O–O})\) | **1.2075 Å** |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT` / `nonperiodic`) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 |
| XC | **PBE** (GGA, spin-resolved) |
| Cutoff | 400 Ry |
| Spin | UKS, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| Stress | ATV analytic fixed-\(P\); CP2K twin `STRESS_TENSOR NUMERICAL`; \(h=2\times 10^{-3}\) |
| SCF twin notes | CP2K: diagonalization + Broyden α=0.4; ATV: mix 0.4 |

**Hard-coded gates (regression binary):**

| Quantity | Pin / check | Tol | Notes |
| -------- | ----------- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | CP2K **−31.928797634435675** | **1×10⁻³** Ha | Soft; same ENERGY pin as `gpw-uks-forces` |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| diag | internal dual-gate | **1×10⁻⁴** Ha/Bohr³ | Decisive product gate |
| Off-diagonal product \(\sigma_{xy,xz,yz}\) | \(\approx 0\) | **1×10⁻⁸** Ha/Bohr³ | Orthorhombic v1 |
| max\|\(\sigma_{\mathrm{diag}}\)\| floor | nonzero | \(> 1\times 10^{-8}\) Ha/Bohr³ | Degeneracy guard |
| Frame `kStress` vs Ha/Bohr³ | unit conversion | \(10^{-9}\) (eV/Å³) | `kHartreeInEv` / \(a_0^3\) |
| SCF | converged | — | `max_iter=200`, `tol=1e-7` |

**Atomiverse first-green numbers (2026-07-25, REFERENCE):**

| Quantity | Value |
| -------- | ----: |
| \(E_{\mathrm{ATV}}\) [Ha] | **−31.9288547285** (60 SCF iters) |
| \|E − E_cp2k\| | **5.7×10⁻⁵** Ha |
| \(\sigma_{xx}\) analytic | \(1.34582413\times 10^{-8}\) Ha/Bohr³ |
| \(\sigma_{yy}\) analytic | \(1.34582366\times 10^{-8}\) Ha/Bohr³ |
| \(\sigma_{zz}\) analytic | \(6.74708540\times 10^{-6}\) Ha/Bohr³ |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| | **5.0×10⁻¹¹** Ha/Bohr³ |
| \(\langle S^2\rangle\) | **2.0056** |

Dual-gate tol (**1×10⁻⁴**) is looser than closed-shell water (**1×10⁻⁵**)
because O₂ UKS residual and single-grid vs multigrid residual are larger;
measured residual is \(\sim 5\times 10^{-11}\). Energy soft check **1×10⁻³**
Ha matches the forces twin (first-green grid class).

Offline oracle sketch (host with CP2K module; not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
export OMP_NUM_THREADS=4
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/gpw-uks-stress/o2_uks_stress_num.inp
mpirun -n 1 cp2k.psmp -i o2_uks_stress_num.inp -o o2_uks_stress_num.out
# ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]  ~ -31.9288
# STRESS| Numerical stress tensor [bar]  …
# Convert: 1 Ha/Bohr³ = 2.94210107994716×10⁸ bar (CP2K internal table)
```

## Performance

Source: `REFERENCE.md` + the case `REFERENCE.md`. **Not** a
ctest gate.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| O₂ UKS-PBE / analytic product (OMP=8, Release) | n/a | **140.6** (product) / **236.5** (FD twin) | **n/a (`perf_exempt`)** | n/a | No CP2K NUMERICAL stress wall on pin host |
| Context only: sibling ENERGY_FORCE wall (`gpw-uks-forces`, OMP=4) | **~77.3** | **~140.6** (product-only, different OMP) | **~1.8** (informal) | ✓ informal | Not the acceptance ratio; NUMERICAL stress is the true stress twin |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | CP2K binary/module unavailable for fair NUMERICAL stress wall; dual-gate wall dominated by multi-SCF FD (6 SCFs), not product analytic |
| Dual-gate total wall | **~377 s** (product + FD) |
| Target when re-pinned | product-only Evaluate vs ENERGY_FORCE-class fair twin, \(r\le 10\) |

## Limitations

- **Absolute CP2K NUMERICAL stress digits deferred** until an offline CP2K
  re-run records Voigt pins; do not invent bar-scale values.
- **Multi-k UKS stress residual** — analytic path throws at multi-k with UKS
  occupations; closed-shell k-point stress and multi-k UKS forces are separate
  queue rows.
- **Orthorhombic only** — shear Voigt components hard-zero in v1; no triclinic
  cell.
- **Single density grid vs CP2K multigrid** (`NGRIDS 4` / `REL_CUTOFF 60`)
  dominates absolute energy residual (\(\sim 6\times 10^{-5}\) Ha here); dual-gate
  stress parity stays tight because both sides use the ATV functional.
- **Cost** — product analytic is one SCF + fixed-\(P\) samples; dual-gate adds
  six full UKS SCFs. Fair wall exemption until NUMERICAL twin is timed.
- **Geometry hygiene:** off-center molecules under MT break energy/stress
  parity; always center for isolated UKS pins.
- **ROKS / spin projection** not provided; residual \(\langle S^2\rangle\)
  contamination is accepted (Löwdin single-determinant).
- Unit tests under `tests/cpu/gpw/` only; e2e CP2K energy parity + dual-gate
  live exclusively in the regression binary.

## References

- Offline pin + walls: [`regressions/gpw/gpw-uks-stress/REFERENCE.md`](../../../regressions/gpw/gpw-uks-stress/REFERENCE.md)
- Regression consumer: [`reg_gpw-uks-stress.cpp`](../../../regressions/gpw/gpw-uks-stress/reg_gpw-uks-stress.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-uks-stress/`](../../../tools/cp2k_oracle/inputs/gpw-uks-stress/)
- Energy pin sibling: [`regressions/gpw/gpw-uks-forces/REFERENCE.md`](../../../regressions/gpw/gpw-uks-forces/REFERENCE.md)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Closed-shell analytic stress: [`gpw-analytic-stress.md`](gpw-analytic-stress.md)
- Strain-FD water pin: [`gpw-stress-water.md`](gpw-stress-water.md)
- PADE UKS energy foothold: [`gpw-uks-o2-gamma.md`](gpw-uks-o2-gamma.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- O. H. Nielsen, R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) — stress theorem
- CP2K Quickstep stress (`STRESS_TENSOR NUMERICAL` / `ANALYTICAL`): CP2K manual
