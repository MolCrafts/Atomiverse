# GPW analytic cell stress at k-points — diamond Si / PBE / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-kpoint-stress` |
| Regression | `regressions/gpw/gpw-kpoint-stress/` |
| Public API | `cpu::gpw::gpw_stress` (analytic fixed-\(P_k\)); oracle `gpw_stress_fd`; `GpwDft` members `k_mesh`, `cell`, `cutoff_ry`, … (no `*Config`) |
| Roadmap | [dft.md](../../roadmap/dft.md) § Periodic DFT — GPW analytic cell stress at k-points |

## Summary

Closed-shell **Gaussian-and-plane-waves (GPW)** Kohn–Sham DFT with
**Monkhorst–Pack k-point sampling** and **Hellmann–Feynman fixed-\(P_k\)
analytic cell stress** on diamond silicon (body-centred-tetragonal 4-atom
cell, PBE, DZVP-MOLOPT-GTH + GTH-PBE-q4). Product path: one SCF at the
unstrained cell, then orthorhombic diagonal strain samples of the
**non-self-consistent** k-summed Lagrangian (fixed density matrices \(P_k\) and
energy-weighted \(W_k\)); multi-SCF strain-FD (`gpw_stress_fd`) is the dual-gate
oracle only. Product gate: `reg_gpw-kpoint-stress`. Offline CP2K twin input
ships at `tools/cp2k_oracle/inputs/gpw-kpoint-stress.inp`; absolute CP2K stress
digits are **not** frozen on the pin host (no CP2K module at pin time) —
decisive gate is analytic vs multi-SCF FD. Performance: **`perf_exempt`**.

## Method

**GPW KS (Quickstep class)** (VandeVondele et al., *Comput. Phys. Commun.*
**167**, 103 (2005)): AO density collocated on a uniform multigrid; FFT Poisson
+ grid XC; integrate back to the AO KS matrix; mix / diagonalize until residual
\(\le\) `tol`. Multi-k RKS uses phased lattice sums, Kleinman–Bylander nonlocal,
and per-k DIIS; `k_mesh` (or explicit `k_points`) is a member of `GpwDft`.

**Analytic cell stress (fixed-\(P_k\) + Pulay).** After one converged SCF:

1. Store occupied MOs \(C_{\mathrm{occ},k}\) and band energies \(\varepsilon_{ik}\)
   → rebuild \(P_k\), \(W_k = \sum_i 2\varepsilon_{ik}\,c_{ik}c_{ik}^\dagger\)
   per k.
2. Under orthorhombic dimensionless strain \(\varepsilon_{aa}=\pm h\) at **fixed
   fractional coordinates**, rebuild \(S_k\), \(T_k\), \(V_{\mathrm{nl}}^k\),
   grid, Poisson, and XC; evaluate the non-self-consistent Lagrangian
   \[
   E_L[\{P_k\},\{W_k\};\varepsilon]
     = \sum_k w_k\Bigl(
         \mathrm{Tr}(P_k T_k)
       + \mathrm{Tr}(P_k V_{\mathrm{pp}}^k)
       - \mathrm{Tr}(W_k S_k)
       \Bigr)
       + E_{\mathrm{es}}[\rho_P]
       + E_{\mathrm{xc}}[\rho_P]
       + E_{\mathrm{disp}}
   \]
   (do **not** re-converge SCF).
3. CP2K `STRESS_TENSOR` sign:
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
   with \(E\) in Hartree and \(\Omega\) in Bohr³. Voigt order
   \((\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})\).
   Off-diagonal shear is **exactly 0** (orthorhombic product scope).

**Oracle (not product).** `gpw_stress_fd` fully re-converges the SCF at each
strained cell (±h per diagonal). Used only for dual-gate tests / REFERENCE.

**Twin settings (binding):**

| Knob | Value |
| ---- | ----- |
| System | Diamond Si, bct 4-atom cell (same family as `gpw-kpoints-si`) |
| Lattice | \(a = 5.43\) Å; cell \((a/\sqrt{2},\,a/\sqrt{2},\,a)\) |
| Atoms (fractional) | \((0,0,0)\), \((\tfrac12,\tfrac12,\tfrac12)\), \((\tfrac12,0,\tfrac14)\), \((0,\tfrac12,\tfrac34)\) |
| Valence | 16 e⁻ → `n_occ = 8` |
| XC | PBE |
| Basis | DZVP-MOLOPT-GTH (Si) |
| PP | GTH-PBE-q4 |
| Density cutoff | 100 Ry (`local_pp_on_grid = true`) |
| k mesh | MONKHORST-PACK \(1\times1\times2\); time-reversal → 1 irreducible k |
| SCF | DIIS, `tol` \(1\times 10^{-8}\) Ha, `EPS_DEFAULT` \(1\times 10^{-7}\), `max_iter` 60 |
| Stress step | \(h = 2\times 10^{-3}\) (dimensionless) |
| Observable | Voigt \(\sigma\) [Ha/Bohr³]; \(E_{\mathrm{tot}}\) [Ha] |

**Key exclusions (closed-spec `out_of_scope`):**

- Absolute hard-coded CP2K `STRESS_TENSOR` bar/Ha values until offline CP2K
  is available on a pin host (twin input ready; do **not** invent digits).
- Triclinic / shear off-diagonal stress; UKS k-stress; DFT+U / hybrid /
  meta-GGA / GAPW stress.
- Variable-cell product drivers (`gpw-variable-cell-ice` /
  `gpw-variable-cell-molopt` are separate).
- Γ-only analytic stress product pin → `gpw-analytic-stress`; MT water
  NUMERICAL strain-FD pin → `gpw-stress-water`.
- Public `*Config` bags / `make_*` factories; live CP2K from ctest; CUDA GPW.

## Implementation

Public surface only (`GpwAtom`, `GpwDft` members, free `gpw_stress` /
`gpw_stress_fd`; no `*Config`, no `make_*` / `build_*`). Match the regression:

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

constexpr double kA2B = 1.0 / 0.52917720859; // Å → Bohr
const double a = 5.43 * kA2B;                // ~10.26120 Bohr
const double at = a / std::sqrt(2.0);

const GaussianBasis bSi(molrs::Element::Si, "DZVP-MOLOPT-GTH");
const GthPotential pSi(molrs::Element::Si, "GTH-PBE-q4");
const double frac[4][3] = {
    {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
std::vector<GpwAtom> atoms;
for (int i = 0; i < 4; ++i) {
    const double pos[3] = {frac[i][0] * at, frac[i][1] * at,
                           frac[i][2] * a};
    atoms.emplace_back(pos, bSi, pSi, 14);
}

GpwDft m;
m.cell[0] = m.cell[1] = at;
m.cell[2] = a;
m.cutoff_ry = 100.0;
m.n_occ = 8;
m.max_iter = 60;
m.tol = 1e-8;
m.xc = XcKind::PBE;
m.eps_default = 1e-7;
m.local_pp_on_grid = true;
m.k_mesh[0] = 1;
m.k_mesh[1] = 1;
m.k_mesh[2] = 2; // MP → κ_z = ±1/4, TR → one k

m.build(atoms);
const auto& er = m.run(atoms); // energy [Ha] on er.e_total

const double h = 2e-3;
const auto san = gpw_stress(atoms, m, h);    // analytic fixed-P_k [Ha/Bohr³]
const auto sfd = gpw_stress_fd(atoms, m, h); // multi-SCF FD oracle
// Voigt: san[0..2] = xx,yy,zz; san[3..5] off-diag (0 for orthorhombic)
```

`GPWEvaluator` product stress (`compute_stress`, default analytic) is the
Frame-facing path for Γ-point Evaluate runs; this regression exercises the
free multi-k solver API directly (same `gpw_stress` implementation that
supports Gamma and closed-shell multi-k RKS).

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
  cmake --build build -j$(nproc) --target atomiverse reg_gpw-kpoint-stress
  ./build/regressions/reg_gpw-kpoint-stress
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft`, `gpw_stress`, `gpw_stress_fd` |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | Product `compute_stress` / `stress_use_fd` (Evaluate) |
| `src/cpu/gpw/internal/gpw_stress.cpp` | Fixed-\(P_k\)+Pulay + multi-SCF FD |
| `tests/cpu/gpw/test_cpu_gpw_stress.cpp` | Unit dual-gate (Γ; not this e2e) |
| `regressions/gpw/gpw-kpoint-stress/reg_gpw-kpoint-stress.cpp` | Product dual-gate |
| `regressions/gpw/gpw-kpoint-stress/REFERENCE.md` | Offline pins + perf |
| `tools/cp2k_oracle/inputs/gpw-kpoint-stress.inp` | CP2K twin (`STRESS_TENSOR NUMERICAL`) |

## Validation (vs CP2K / dual-gate)

Source:
[`regressions/gpw/gpw-kpoint-stress/REFERENCE.md`](../../../regressions/gpw/gpw-kpoint-stress/REFERENCE.md).
Oracle strategy: offline **CP2K** twin when available (`cp2k.psmp` +
`tools/cp2k_oracle/inputs/gpw-kpoint-stress.inp`); **never** live CP2K from
ctest. Pin host (2026-07-25) had **no CP2K binary/module** — absolute
`STRESS_TENSOR` digits are not invented. Decisive product gate is **internal
dual-gate** analytic vs multi-SCF FD.

**Setup:**

| Item | Value |
| ---- | ----- |
| Engine class | CP2K Quickstep GPW (twin input ready) |
| Twin input | `tools/cp2k_oracle/inputs/gpw-kpoint-stress.inp` |
| CP2K stress mode (when run) | `STRESS_TENSOR NUMERICAL` |
| Atomiverse path | `GpwDft` + `gpw_stress` / `gpw_stress_fd`, mesh \(1\times1\times2\), \(h=2\times 10^{-3}\) |
| Deps | `gpw-kpoints-si` ✓, `gpw-analytic-stress` ✓ |

**Hard-coded product pins (first green, 2026-07-25; REFERENCE + regression):**

| Quantity | Pin / measured | Tol (gate) | Notes |
| -------- | -------------: | ---------- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−15.3483039265** | \(1\times 10^{-3}\) Ha | Soft absolute band (100 Ry mesh) |
| Soft fold band | vs `gpw-kpoints-si` A2 \(E=-15.3483059090\) Ha @ 95 Ry | \(5\times 10^{-3}\) Ha | Mesh/cutoff soft check class |
| \(\sigma_{xx}=\sigma_{yy}\) analytic [Ha/Bohr³] | **8.83135454×10⁻⁴** | \(1\times 10^{-4}\) soft abs | Frozen first-green |
| \(\sigma_{zz}\) analytic [Ha/Bohr³] | **2.60970168×10⁻⁴** | same | Tetragonal cell anisotropy |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| diag | **7.2×10⁻⁸** | \(1\times 10^{-4}\) | Decisive dual-gate |
| Off-diagonal product | 0 | \(1\times 10^{-8}\) | Orthorhombic v1 |
| max\|\(\sigma_{\mathrm{diag}}\)\| floor | — | \(>1\times 10^{-6}\) | Non-trivial magnitude |
| SCF iters (unperturbed) | **7** | — | REFERENCE |

**Tolerance rationale.** Dual-gate \(1\times 10^{-4}\) Ha/Bohr³ absorbs mesh /
\(O(h^2)\) residual while requiring analytic and multi-SCF FD to agree on
order and sign (measured residual \(\sim 10^{-7}\)). Absolute energy band
\(1\times 10^{-3}\) Ha freezes first-green integrity; replace energy/stress
absolute pins with cited CP2K `STRESS_TENSOR` numbers when offline CP2K runs
the twin.

Offline oracle sketch (host with CP2K module; not ctest):

```bash
module load ... CP2K/2026.1   # example
export CP2K_DATA_DIR=.../share/cp2k/data
cp2k.psmp -i tools/cp2k_oracle/inputs/gpw-kpoint-stress.inp -o kstress.out
# ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]  ...
# STRESS| Numerical stress tensor [bar] ...
# Convert: 1 Ha/Bohr³ = 2.94210107994716e8 bar (CP2K table)
```

## Performance

Source: `REFERENCE.md` / handoff. **`perf_exempt = true`** — no CP2K
binary/module on the pin host for a fair wall twin.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| Si bct k-stress dual-gate | n/a | **~171** (OMP=8, Release) | n/a | n/a | `perf_exempt` |

Gate: when CP2K is available, record single-rank walls for the twin and
target \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\). Dual-gate cost is
analytic pass + multi-SCF FD (6 strained SCF + reference) — expected heavy;
fair ratio should use an ENERGY_FORCE-class denominator if that is the queue
convention for stress twins (`gpw-stress-water`).

## Limitations

- **No absolute CP2K stress pin yet** — twin input is ready; digits await an
  offline CP2K host. Do not invent bar/Ha values.
- **Orthorhombic diagonal only** — no triclinic shear; off-diagonal Voigt
  components are hard zeros in product `gpw_stress`.
- **Closed-shell multi-k RKS** — UKS k-stress, hybrids, DFT+U, dispersion
  stress contributions not covered by this slug.
- **Incomplete basis / sparse mesh** — 100 Ry, \(1\times1\times2\) is a
  textbook dual-gate twin, not production-converged bulk Si stress.
- **Cost** — dual-gate regression ~minutes (OMP=8); product analytic path is
  one SCF + fixed-\(P_k\) samples; multi-SCF FD is oracle-only.
- Public Frame energy remains eV when written via Evaluate; this regression
  compares [Ha] / [Ha/Bohr³] on the free solver API.
- Unit tests under `tests/cpu/gpw/` only; e2e parity is exclusively
  `reg_gpw-kpoint-stress`.

## References

- Offline pin + fair wall: `regressions/gpw/gpw-kpoint-stress/REFERENCE.md`
- Twin input: `tools/cp2k_oracle/inputs/gpw-kpoint-stress.inp`
- Oracle notes: `tools/cp2k_oracle/README.md`
- Related: [gpw-analytic-stress](gpw-analytic-stress.md) (Γ fixed-\(P\)),
  [gpw-stress-water](gpw-stress-water.md) (MT NUMERICAL pin),
  k-point energy twin family `gpw-kpoints-si`, k-forces
  `regressions/gpw/gpw-kpoint-forces/`
- Roadmap: `docs/roadmap/dft.md` (GPW analytic cell stress at Γ and k-points ✓)
- Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985)
- J. VandeVondele, M. Krack, F. Mohamed, M. Parrinello, T. Chassaing,
  J. Hutter, *Comput. Phys. Commun.* **167**, 103 (2005)
- CP2K Quickstep stress (`STRESS_TENSOR NUMERICAL` / `ANALYTICAL`): CP2K
  manual, FORCE_EVAL / QS stress tensor section
