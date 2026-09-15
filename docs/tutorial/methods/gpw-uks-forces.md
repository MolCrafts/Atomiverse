# Γ-UKS GPW analytic forces — O₂ / PBE / DZVP-MOLOPT-GTH (MT)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-uks-forces` |
| Regression | `regressions/gpw/gpw-uks-forces/` |
| Public API | `cpu::gpw::GPWEvaluator` (`n_alpha`/`n_beta`, `XcKind::PBE`, `nonperiodic`, `compute_forces`); `cpu::Evaluate` |
| Roadmap | `docs/roadmap/dft.md` — Γ-UKS GPW ionic forces (PBE GGA) |

## Summary

**Unrestricted Kohn–Sham (UKS) Gaussian-and-plane-waves (GPW)** analytic nuclear
forces at the **Γ point** for open-shell systems. Product path: spin-resolved
**PBE GGA**, GTH-PBE-q6 / DZVP-MOLOPT-GTH, Martyna–Tuckerman isolated Poisson
(`nonperiodic = true`), Hellmann–Feynman + Pulay assembly with
\(W_\alpha+W_\beta\) and spin-resolved grid \(V_{\mathrm{eff}}^{\alpha/\beta}\).
This page pins **triplet O₂** energy and Cartesian forces against offline
**CP2K 2026.1 Quickstep** (`RUN_TYPE ENERGY_FORCE`, `LSD`, `MULTIPLICITY 3`).

Public path: construct `cpu::gpw::GPWEvaluator` with one `Kind` per element
(lookup constructors `GaussianBasis` / `GthPotential`), set members
(`n_alpha`, `n_beta`, `cutoff_ry`, `xc`, `nonperiodic`, `compute_forces`, …),
bind into `cpu::Evaluate`. Energy on `frame.meta` is **eV**; forces on
`frame.span(kAtom_f*)` are **eV/Å** (pin comparison converts to Ha / Ha·Bohr⁻¹).
Product gate: `reg_gpw-uks-forces`. Fair wall ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 2.52\) (≤10×).

Related: energy-only Γ-UKS PADE foothold
[gpw-uks-o2-gamma](gpw-uks-o2-gamma.md); closed-shell RKS forces
[gpw-forces-water](gpw-forces-water.md); multi-k UKS energy
`regressions/gpw/gpw-kpoints-uks` (forces remain Γ-only here). Narrative:
[Ch. 9 GPW](../ch09-gpw.md).

## Method

**GPW (Quickstep family).** Wavefunction in a contracted Gaussian AO basis;
density collocation on a uniform real-space grid; Hartree via FFT Poisson;
XC on the grid; potential integrated back into the AO basis. Core electrons
removed by analytic **GTH** pseudopotentials. Grid resolution set by the
plane-wave cutoff in Rydberg. Citation: Lippert, Hutter, Parrinello,
*Mol. Phys.* **92**, 477 (1997); VandeVondele et al., *Comput. Phys. Commun.*
**167**, 103 (2005).

**UKS / collinear spin.** With \(N_\alpha \neq N_\beta\), build spin densities
\(\rho_\alpha\), \(\rho_\beta\); Hartree from \(\rho=\rho_\alpha+\rho_\beta\);
spin-resolved PBE on the grid; diagonalize separate \(H_\alpha\) and
\(H_\beta\). Occupations are explicit integer members `n_alpha` / `n_beta`
(not a multiplicity bag). When both are zero, the closed-shell `n_occ` path
is unchanged.

**Ionic forces (analytic product path, `gpw_forces_analytic` UKS branch):**

\[
\mathbf{F}_A = -\frac{\partial E}{\partial \mathbf{R}_A}
=
\mathbf{F}_A^{\mathrm{HF}}
+ \mathbf{F}_A^{\mathrm{Pulay}}
+ \mathbf{F}_A^{\mathrm{grid/core}},
\]

where Hellmann–Feynman collects explicit \(\mathbf{R}_A\)-dependence of the
Hamiltonian (local PP, nonlocal projectors, nuclear–electron terms), Pulay
contracts the **spin-summed** energy-weighted density
\(W = W_\alpha + W_\beta\) with \(\partial S/\partial R_A\), and grid/core
terms use spin-resolved effective potentials
\(V_{\mathrm{eff}}^{\alpha/\beta}\) consistent with the SCF functional
(including collocated short-range local PP when `local_pp_on_grid` is true —
the product default).

**PBE GGA (spin-resolved).** Perdew–Burke–Ernzerhof exchange–correlation on
\(\rho_\alpha,\rho_\beta\) and their gradients (`XcKind::PBE`). Citation:
Perdew, Burke, Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996).

**Isolated electrostatics.** `nonperiodic = true` selects the
**Martyna–Tuckerman** Poisson twin of CP2K `PERIODIC NONE` / `PSOLVER MT`.
The molecule must sit **centered** in the cell: nonperiodic MT grids cover
\([0,L)\); atoms near an edge truncate Gaussian tails and destroy energy and
force parity. Citation: Martyna & Tuckerman, *J. Chem. Phys.* **110**, 2810
(1999).

**Valence bookkeeping (this pin).** O with GTH-PBE-q6 → 6 valence e⁻ per O →
12 valence e⁻ total. Multiplicity 3 ⇒ \(N_\alpha=7\), \(N_\beta=5\)
(not all-electron \(9/7\)).

**Twin settings (binding):**

| Knob | Value |
| ---- | ----- |
| System | Isolated O₂, **centered** |
| \(R(\mathrm{O–O})\) | **1.2075 Å** along *z* |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT` / `nonperiodic`) |
| XC | **PBE** (GGA, spin-resolved) |
| Basis | DZVP-MOLOPT-GTH |
| PP | GTH-PBE-q6 |
| Grid | CUTOFF **400** Ry (CP2K `REL_CUTOFF 60`, `NGRIDS 4`) |
| SCF | `EPS_SCF` / `tol` \(1\times 10^{-7}\); mix α=0.4 |
| Spin | UKS / LSD, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| `local_pp_on_grid` | **true** (product default; energy and gradient both read it) |
| Observable | Cart forces [Ha/Bohr]; \(E_{\mathrm{tot}}\) [Ha] soft check |

**Key exclusions (this page / acceptance):**

- **No k-point UKS forces** — Γ only; multi-k RKS forces are
  `gpw-kpoint-forces`; multi-k UKS forces remain follow-on.
- **No UKS stress** — cell stress paths are separate (`gpw-stress-water`,
  `gpw-kpoint-stress`); UKS stress may throw / unpinned.
- **No ROKS** GPW forces; no spin projection of \(\mathbf{F}\).
- **Not** the PADE energy-only foothold (`gpw-uks-o2-gamma`); this pin is
  **PBE + ENERGY_FORCE**.
- **Not** all-electron molecular UKS gradients (Gaussian SCF / PySCF loop).
- Live CP2K from ctest / regression binary — **offline pin only**.
- Finite-difference forces as the product gate (unit test only:
  `tests/cpu/gpw/test_cpu_gpw_forces.cpp` UKS H₂ analytic vs FD, 5e-5 Ha/Bohr
  on the ATV energy functional).

**Historical consistency note.** `local_pp_on_grid` must be the same for
energy and analytic gradient. Energy-only grid PP with a force path that
omits collocated local SR PP produced a ZnO analytic-vs-FD residual of
order 0.53 Ha/Bohr; the product path keeps both sides on grid PP
(`cpu/gpw/internal/local_pp.h`).

## Implementation

Public surface only (`cpu::Frame`, `cpu::gpw::Kind` / lookup constructors,
`GPWEvaluator` members; **no** public `*Config`, **no** `make_*` / `build_*`
factories):

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

cpu::gpw::Kind O{
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
eval.compute_forces = true;
eval.n_alpha = 7;  // multiplicity 3, 12 valence e⁻
eval.n_beta = 5;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

// Energy: frame.meta.get(kEnergy) [eV]; also eval.last_energy_hartree() [Ha]
// Forces: kAtom_fx/y/z [eV/Å] → Ha/Bohr via units::kEvPerAngstromInHartreePerBohr
// Spin:   eval.last_s2()  Löwdin ⟨S²⟩
```

Build (GPW needs FFTW3 + HF + PW) and run the regression:

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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-uks-forces
  ./build/regressions/reg_gpw-uks-forces
  # or: cmake --build build --target run_reg_gpw-uks-forces
'
```

Relevant sources:

| Piece | Path |
| ----- | ---- |
| Evaluator | `include/atomiverse/cpu/gpw/gpw_evaluator.h` |
| Solver (UKS SCF) | `include/atomiverse/cpu/gpw/gpw_solver.h`, `src/cpu/gpw/gpw_solver.cpp` |
| Analytic forces (UKS branch) | `gpw_forces_analytic` under `src/cpu/gpw/` / `internal/` |
| Unit tests (not e2e) | `tests/cpu/gpw/test_cpu_gpw_forces.cpp` (`UksH2AnalyticVsFiniteDifference`) |
| Regression | `regressions/gpw/gpw-uks-forces/reg_gpw-uks-forces.cpp` |
| Offline pin | `regressions/gpw/gpw-uks-forces/REFERENCE.md` |
| CP2K twin input | `tools/cp2k_oracle/inputs/gpw-uks-forces/o2_uks_forces.inp` |

## Validation (CP2K)

Source: `regressions/gpw/gpw-uks-forces/REFERENCE.md` (offline pin; **never**
live CP2K in ctest). Engine: **CP2K 2026.1** (`cp2k.psmp`), EasyBuild module
stack on Arrhenius. Re-pin date: **2026-07-25**. `RUN_TYPE ENERGY_FORCE`.

| Quantity | CP2K pin | Atomiverse gate | Tol | Notes |
| -------- | -------: | --------------- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.928797634435675** | \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}\rvert\) | **1e-3 Ha** | Soft check; ENERGY\| FORCE_EVAL (QS) line |
| \(F_z\) O1 [Ha/Bohr] | **−3.50081576×10⁻²** | max component \(\lvert\Delta F\rvert\) | **5e-3** | First-green; ATV single density grid vs CP2K multigrid |
| \(F_z\) O2 [Ha/Bohr] | **+3.50081358×10⁻²** | same | **5e-3** | Equal-and-opposite bond force |
| \(F_x,F_y\) | ~1e-12 (numerical) | same | **5e-3** | Bond along *z* |
| \(\|\sum_A\mathbf{F}_A\|\) [Ha/Bohr] | CP2K print residual \(\sim 2\times 10^{-8}\) | \(\le 5\times 10^{-4}\) | **5e-4** | Translational residual gate |
| SCF | converged | `last_result().converged` | — | `max_iter=200`, `tol=1e-7` |

Hard-coded pins in `reg_gpw-uks-forces.cpp` (Ha/Bohr):

```text
O1:  2.02069184e-12,  2.05677955e-12, -3.50081576e-02
O2:  2.43439049e-12,  2.40869221e-12,  3.50081358e-02
E_tot_Ha: -31.928797634435675
```

Key CP2K log lines (from REFERENCE):

```text
  Total energy:                                               -31.92879767674141
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.928797634435675
 FORCES| Atomic forces [hartree/bohr]
 FORCES|   Atom     x               y               z               |f|
 FORCES|      1  2.02069184E-12  2.05677955E-12 -3.50081576E-02   3.50081576E-02
 FORCES|      2  2.43439049E-12  2.40869221E-12  3.50081358E-02   3.50081358E-02
 FORCES| Total atomic force                                       2.17692380E-08
```

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
| XC | PBE (GGA LSD) |
| Cutoff | 400 Ry |
| Spin | UKS / LSD, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| SCF twin notes | CP2K: diagonalization + Broyden α=0.4; ATV: DIIS-style mix 0.4 |

**Measured ATV vs pin (REFERENCE):**

| Metric | Value |
| ------ | ----: |
| max\|F_atv − F_cp2k\| (observed twin) | **≈ 2.7×10⁻³** Ha/Bohr |
| \|ΔE\| (observed) | **≈ 6×10⁻⁵** Ha |
| \|F_z\| scale | **~0.035** Ha/Bohr |
| Force gate | **5×10⁻³** Ha/Bohr max component |

**Tolerance rationale.** Same-method twin (Quickstep GPW UKS / product MT +
PBE + MOLOPT/GTH). Component force tol \(5\times 10^{-3}\) Ha/Bohr is
first-green: residual is dominated by Atomiverse **single density grid** vs
CP2K **NGRIDS 4 / REL_CUTOFF 60** multigrid, not a missing force term. Unit
gate `test_cpu_gpw_forces` (UKS H₂ analytic vs FD) is **5e-5** Ha/Bohr on the
ATV energy functional. Energy soft check at \(1\times 10^{-3}\) Ha covers
grid residual with room; observed |\(\Delta E\)| is ~6e-5 Ha. Translational
residual gate matches a safe numerical bound (CP2K’s own printed residual is
tighter, \(\sim 2\times 10^{-8}\)).

Offline re-run (operator only; not ctest):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
export OMP_NUM_THREADS=4
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
# copy tools/cp2k_oracle/inputs/gpw-uks-forces/o2_uks_forces.inp
mpirun -n 1 cp2k.psmp -i o2_uks_forces.inp -o o2_uks_forces.out
```

## Performance

Source: same `REFERENCE.md` fair-wall note (2026-07-25) and
the case `REFERENCE.md`. Not a ctest gate; optional offline
twin under `tools/cp2k_oracle/`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ---------------------------------------: | ---- | ------ |
| O₂ UKS-PBE / DZVP-MOLOPT-GTH / MT, Γ, ENERGY_FORCE | **77.3** | **195** | **2.52** | ✓ | CP2K: 1 MPI × 4 OMP, multigrid NGRIDS 4; ATV: Release, single density grid, OMP=4 |

Gate from closed acceptance: \(r \le 10\) or `perf_exempt`. Here
\(r \approx 2.52\) — product pass without exemption. Handoff / queue record
the same ratio **~2.52**.

## Limitations

- **Grid twin, not bit-identical SCF:** Atomiverse uses a **single** density
  grid; CP2K pin uses multigrid (`NGRIDS 4` / `REL_CUTOFF 60`). Force
  tolerance is deliberately **5e-3 Ha/Bohr** for first-green; may tighten
  when multigrid parity is product work.
- **Γ-UKS forces only** — multi-k UKS forces and UKS stress are out of scope
  for this regression (see `gpw-kpoint-forces`, `gpw-kpoints-uks`,
  `gpw-kpoint-stress` for related RKS/UKS product surfaces).
- **Isolated MT O₂ only** — no periodic supercell, no BOMD trajectory pin,
  no geometry-opt loop gate on this page.
- **PBE GGA** on this pin; PADE energy-only is `gpw-uks-o2-gamma`. Do not
  document ROKS / spin-projected forces as available.
- **Geometry hygiene:** off-center molecules under MT will fail energy and
  force parity; always center for isolated UKS/RKS pins.
- **Wall ~2.5× CP2K** on this twin is within gate; further speed work is not
  required for acceptance.
- **Unit analytic-vs-FD** lives under `tests/cpu/gpw/` only; e2e CP2K
  parity is exclusively the regression binary.
- Frame forces are eV/Å; literature / pin comparison must convert to
  Ha/Bohr (`units::kEvPerAngstromInHartreePerBohr`). Public energy remains
  eV on `frame.meta`; Ha is internal / pin units.

## References

- Offline pin: [`regressions/gpw/gpw-uks-forces/REFERENCE.md`](../../../regressions/gpw/gpw-uks-forces/REFERENCE.md)
- Regression consumer: [`reg_gpw-uks-forces.cpp`](../../../regressions/gpw/gpw-uks-forces/reg_gpw-uks-forces.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-uks-forces/`](../../../tools/cp2k_oracle/inputs/gpw-uks-forces/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Sibling tutorials: [gpw-uks-o2-gamma](gpw-uks-o2-gamma.md) (energy),
  [gpw-forces-water](gpw-forces-water.md) (closed-shell forces)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep forces
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
