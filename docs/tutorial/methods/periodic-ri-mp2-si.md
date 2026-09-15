# Periodic Γ-point RI-GPW MP2 — sc-Si₂ / DZVP-MOLOPT-GTH (product path, PBE reference)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-ri-mp2-si` |
| Regression | `regressions/gpw/periodic-ri-mp2-si/` |
| Public API | `cpu::gpw::GpwDft::mp2(atoms)` → `cpu::gpw::Mp2Result` (`cpu/gpw/mp2.h`); members `mp2_ri` (default **true**), `mp2_n_virt`, `mp2_os_scale`, `mp2_ss_scale` |
| Roadmap | ratio **~0.09**; closes capability after foothold `periodic-mp2-si`; unblocks `periodic-rpa-si` |

## Summary

**Product path for second-order Møller–Plesset correlation under periodic
boundary conditions** on top of a converged Γ-point closed-shell GPW PBE SCF.
Occupied–virtual MO repulsion integrals \((ia|jb)\) are assembled the
**RI-GPW** way (Del Ben, Hutter, VandeVondele, *JCTC* **8**, 4177 (2012);
CP2K `&WF_CORRELATION` `METHOD MP2_GPW` + `&WFC_GPW`): FFT each pair density
\(\rho_{ia}=\psi_i\psi_a\), store the plane-wave Coulomb intermediate, and
contract in real or G-space. On this deck the RI-GPW kernel is numerically
identical to the Poisson foothold (`mp2_ri = false`, slug `periodic-mp2-si`)
to \(10^{-10}\) Ha — same Coulomb operator, product intermediate layout.
Absolute correlation energy is pinned to offline **CP2K 2026.1**
(\(E_{\mathrm{corr}} = -0.276087740659\) Ha, tol \(10^{-3}\) Ha). Public
surface: concrete `GpwDft` members + `mp2()` — no `*Config`, no factories.
Fair wall vs CP2K is **within_10x** (\(r \approx 0.09\)).

## Method

### SCF reference

Γ-point periodic MOs come from a closed-shell **GPW RKS PBE** solve
(Quickstep dual representation: Gaussian density matrix + plane-wave Hartree/XC
on a uniform grid). Lattice-summed AOs, GTH pseudopotential, MOLOPT basis —
same deck as the canonical foothold.

Canonical eigenvalues \(\varepsilon_p\) enter the MP2 denominators
**unshifted** (the SCF `level_shift` is a convergence aid only).

### RI-GPW \((ia|jb)\) assembly

Occupied–virtual pair densities are **neutral**
(\(\int \rho_{ia} = S_{ia} = 0\)), so the periodic Coulomb operator with the
G=0 component dropped is divergence-free at Γ:

\[
\rho_{ia}(\mathbf r) = \psi_i(\mathbf r)\,\psi_a(\mathbf r),
\qquad
\tilde\rho_{ia}(\mathbf G)
  = \mathrm{FFT}[\rho_{ia}],
\qquad
B_{ia}(\mathbf G)
  = \sqrt{\frac{4\pi}{|\mathbf G|^2}}\,
    \tilde\rho_{ia}(\mathbf G)
\quad (\mathbf G \neq 0).
\]

The four-index MO ERIs contract as a plane-wave resolution of the identity
(Del Ben et al.; CP2K `MP2_GPW`):

\[
(ia|jb)
  = \frac{w}{N_g}
    \sum_{\mathbf G}
    B_{ia}^*(\mathbf G)\, B_{jb}(\mathbf G)
\]

(equivalently: IFFT \(V_{ia}\) from \(4\pi\tilde\rho_{ia}/|\mathbf G|^2\) and
real-space quadrature \(\sum_g w\, V_{ia}(g)\,\rho_{jb}(g)\)). Memory layout is
three-index \(\mathcal O(n_{\mathrm{ov}} N_g)\); the foothold Poisson path
solves one Helmholtz/Poisson equation per pair and stores the same
\(V_{ia}(\mathbf r)\) without the explicit \(B\) product form.

### Closed-shell doubles sum

Canonical OS/SS decomposition (identical convention to molecular
`cpu::scf::MP2Result` and foothold `cpu::gpw::Mp2Result`):

\[
E^{(2)}
  = -\sum_{ijab}
    \frac{(ia|jb)\,[\,2(ia|jb) - (ib|ja)\,]}
         {\varepsilon_a + \varepsilon_b - \varepsilon_i - \varepsilon_j}
  = E_{\mathrm{OS}} + E_{\mathrm{SS}}.
\]

Optional Grimme scales (`mp2_os_scale`, `mp2_ss_scale`, default 1.0) multiply
the unscaled OS/SS pieces into the reported `e_corr`. Correlation is evaluated
on **KS (PBE) orbitals** — the standard periodic foothold / double-hybrid
ingredient. A true MP2@HF reference awaits periodic exact exchange
(blocked row `periodic-hybrid-gpw`).

### What this pin includes

| Item | Setting |
| ---- | ------- |
| Spin | Closed-shell RKS (Γ only) |
| SCF XC | **PBE** (`XcKind::PBE`) |
| System | artificial sc-Si₂ (diamond-fragment motif) |
| Lattice | cubic \(a = 5.43\) Å, fully periodic, Γ only |
| Positions [Å] | Si \((0,0,0)\); Si \((a/4,a/4,a/4)=(1.3575,1.3575,1.3575)\) |
| Basis | DZVP-MOLOPT-GTH (Si) |
| Pseudopotential | GTH-PBE-q4 (zion 4) → 8 valence e⁻ → \(n_{\mathrm{occ}}=4\) |
| Density / WFC cutoff | **30 Ry** (single grid; e2e ≤ 60 s) |
| SCF | tol \(10^{-6}\) Ha, mix 0.3, DIIS, **level_shift 0.3 Ha**, max_iter 80 |
| MP2 | all kept virtuals (`mp2_n_virt = 0`); `mp2_ri = true`; OS/SS scales 1.0 |

**Note (small gap):** Γ-only sc-Si₂ is an artificial small-gap crystal
(unshifted KS gap \(\approx 0.015\) Ha). SCF without the level shift oscillates
for hundreds of iterations. The gap is far above the `GpwDft::mp2` \(10^{-6}\) Ha
degeneracy guard. Physical diamond-Si₈ at production cutoffs exceeds the
regression wall budget — residual.

### Key exclusions (non-goals of this slug)

- Multi-k / Brillouin-zone MP2 (residual; SCF may be multi-k, `mp2()` rejects it).
- Gaussian cFIT / `RI_AUX` auxiliary tables (plane-wave RI identity only).
- UMP2 / open-shell; OT SCF (no canonical virtuals).
- True MP2@HF (needs periodic hybrid / exact exchange).
- Periodic RPA / GW (consumer: `periodic-rpa-si`, after this product).
- Live CP2K (or any third-party) inside ctest.
- Public `*Config` bags / `make_*` / `build_*` factories.
- CUDA GPW (CPU-only in this loop).

## Implementation

Public surface only (`GpwDft` members + lookup constructors; matches
`reg_periodic-ri-mp2-si.cpp`):

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/mp2.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

constexpr double kA2B = 1.8897259885789;
constexpr double a_bohr = 5.43 * kA2B;

const GaussianBasis bas(molrs::Element::Si, "DZVP-MOLOPT-GTH");
const GthPotential pot(molrs::Element::Si, "GTH-PBE-q4");
const double pos0[3] = {0.0, 0.0, 0.0};
const double pos1[3] = {a_bohr / 4.0, a_bohr / 4.0, a_bohr / 4.0};

std::vector<GpwAtom> atoms;
atoms.emplace_back(pos0, bas, pot, 14);
atoms.emplace_back(pos1, bas, pot, 14);

GpwDft m;
m.cell[0] = m.cell[1] = m.cell[2] = a_bohr;  // Bohr
m.cutoff_ry = 30.0;
m.n_occ = 4;
m.max_iter = 80;
m.tol = 1e-6;
m.mix = 0.3;
m.level_shift = 0.3;                          // convergence aid
m.xc = XcKind::PBE;
m.mp2_ri = true;                              // product default
m.build(atoms);
const GpwResult& scf = m.run(atoms);          // SCF [Ha] on result

m.mp2_n_virt = 0;                             // 0 = all kept virtuals
const Mp2Result& r = m.mp2(atoms);
// r.ri == true; r.e_corr (< 0) [Ha]; r.e_corr_os / r.e_corr_ss;
// r.e_total = r.e_scf + r.e_corr
```

Guards (`std::logic_error` / `std::invalid_argument`): call before
`build()` / converged `run()`; UKS; `use_ot`; multi-k SCF result; vanishing
HOMO–LUMO gap (\(< 10^{-6}\) Ha); no virtuals left after cap. Isolated
(`nonperiodic`) cells always fall back to the Poisson path regardless of
`mp2_ri`. Kernels are private (`cpu/gpw/internal/mp2_kernels.h`).

Build and run the regression (CPU login; modules as in `AGENTS.md`):

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
  cmake --build build -j$(nproc) --target atomiverse reg_periodic-ri-mp2-si
  ./build/regressions/reg_periodic-ri-mp2-si
  # or: cmake --build build --target run_reg_periodic-ri-mp2-si
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/mp2.h` | `Mp2Result` public value type (`ri`, OS/SS, totals) |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::{mp2_ri,mp2_n_virt,mp2_os_scale,mp2_ss_scale,mp2}` |
| `include/atomiverse/cpu/gpw/internal/mp2_kernels.h` | Private MO-on-grid / RI-GPW / doubles-sum kernels |
| `src/cpu/gpw/mp2.cpp` | `GpwDft::mp2` dispatch (RI vs Poisson) |
| `src/cpu/gpw/internal/mp2_kernels.cpp` | Kernel implementations |
| `include/atomiverse/cpu/gpw/gaussian_basis.h` | `GaussianBasis(Element, name)` lookup |
| `include/atomiverse/cpu/gpw/gth_potential.h` | `GthPotential(Element, name)` lookup |
| `regressions/gpw/periodic-ri-mp2-si/reg_periodic-ri-mp2-si.cpp` | Product gate (RI≡Poisson + CP2K pin) |
| `tools/cp2k_oracle/inputs/periodic-ri-mp2-si.inp` | Offline CP2K twin (never ctest) |
| `tests/cpu/gpw/test_cpu_gpw_mp2.cpp` | Unit: doubles sum, MO-on-grid, Poisson Hartree identity |

## Validation (vs CP2K)

Source:
[`regressions/gpw/periodic-ri-mp2-si/REFERENCE.md`](../../../regressions/gpw/periodic-ri-mp2-si/REFERENCE.md).
Oracle: **offline CP2K 2026.1** (`cp2k.psmp`), input
`tools/cp2k_oracle/inputs/periodic-ri-mp2-si.inp` — never launched by ctest.
Pin date 2026-07-25, host arrhenius1 (EPYC), OMP=4, 1 MPI rank.

**Setup:**

| Item | Value |
| ---- | ----- |
| System | artificial sc-Si₂, charge 0, singlet (closed-shell RKS) |
| Lattice | cubic \(a = 5.43\) Å, periodic XYZ, Γ only |
| Positions [Å] | Si (0, 0, 0); Si (1.3575, 1.3575, 1.3575) |
| Basis | DZVP-MOLOPT-GTH |
| Pseudopotential | GTH-PBE-q4 (zion 4) |
| XC (SCF) | PBE |
| Cutoff | 30 Ry (SCF `MGRID` + `WFC_GPW`) |
| Valence | 8 e⁻ → \(n_{\mathrm{occ}} = 4\) |
| SCF | `EPS_SCF 1e-6`, `LEVEL_SHIFT 0.3`, direct \(P\)-mixing \(\alpha=0.3\), `ADDED_MOS 22` |
| Correlation | CP2K `METHOD MP2_GPW` / ATV `mp2_ri = true` |
| Provenance | Del Ben et al. 2012; tables from CP2K GTH / MOLOPT data family |

**Accuracy gate (product):**

| Quantity | Atomiverse / pin | CP2K 2026.1 | Tol | Notes |
| -------- | ---------------: | ----------: | --- | ----- |
| \(E_{\mathrm{corr}}\) (RI-GPW) | within band of pin | **−0.276087740659 Ha** | **\(1\times 10^{-3}\) Ha** | Hard offline pin in `reg_periodic-ri-mp2-si` |
| \(E_{\mathrm{corr}}\) OS (CP2K SO) | — | −0.237578621868 Ha | — | CP2K spin-opposite channel |
| \(E_{\mathrm{corr}}\) SS | — | −0.038509118792 Ha | — | CP2K same-spin channel |
| \(E_{\mathrm{SCF}}\) (PBE) | ~−7.596 Ha (ATV deck) | −7.595338517019 Ha | soft | Grid/SCF twin; not the e2e hard assert |
| \(E_{\mathrm{total}}\) | \(E_{\mathrm{SCF}}+E_{\mathrm{corr}}\) | −7.871426257678 Ha | — | CP2K total |
| \(\lvert E_{\mathrm{corr}}(\mathrm{RI})-E_{\mathrm{corr}}(\mathrm{Poisson})\rvert\) | \(0\) (same kernel) | — | **\(1\times 10^{-10}\) Ha** | Hard identity vs `mp2_ri=false` |
| OS + SS ≡ \(E_{\mathrm{corr}}\); \(E_{\mathrm{tot}}\equiv E_{\mathrm{SCF}}+E_{\mathrm{corr}}\) | exact | — | \(10^{-12}\) | Algebraic |
| `Mp2Result::ri` | `true` on product path | — | — | Flag gate |
| SCF | converged; \(E_{\mathrm{corr}}<0\) finite | — | hard | Product gate |

Unit-level gates (not e2e, under `tests/cpu/gpw/test_cpu_gpw_mp2.cpp`):
closed-shell doubles sum ≡ explicit spin-orbital MP2 sum (\(10^{-12}\));
MO-on-grid ≡ Γ-image-summed AO contraction (\(10^{-9}\));
\((pp|pp)\) ≡ \(2\times\) Poisson Hartree energy of the same pair density
(\(10^{-10}\)); guard throws.

Offline re-pin (human only; never ctest):

```bash
# when a CP2K binary is available (cite release in REFERENCE.md)
cp2k.psmp -i tools/cp2k_oracle/inputs/periodic-ri-mp2-si.inp \
          -o /tmp/periodic-ri-mp2-si.out
# record MP2 Energy / TOTAL ENERGY [Ha]; tighten tol if grids match closer
```

Sibling foothold (canonical Poisson path, first-green ATV band until re-pin):
[`periodic-mp2-si`](periodic-mp2-si.md) /
`tools/cp2k_oracle/inputs/periodic-mp2-si.inp`.

## Performance

Source: [`REFERENCE.md`](../../../regressions/gpw/periodic-ri-mp2-si/REFERENCE.md)
(2026-07-25); queue/handoff ratio **~0.09** (`within_10x`).

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **false** |
| Host / threads | arrhenius1 EPYC, OMP=4, single process (ATV); CP2K 1 rank × OMP=4 |
| \(T_{\mathrm{CP2K}}\) (full ENERGY, SCF + MP2_GPW) | **576.7 s** |
| \(T_{\mathrm{ATV}}\) (SCF + RI MP2 + Poisson cross-check) | **~16–30 s** (typical) |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **~0.03–0.05** raw; queue records **~0.09** |
| `within_10x` | **yes** (\(r \ll 10\)) |

Caveat: ATV regression wall includes an intentional RI vs Poisson cross-check
(second full MP2). CP2K wall is the full QS ENERGY job (SCF + `MP2_GPW`).
Deck is tiny (30 Ry, 2 atoms); ratio is **not** a production-size claim.
Never launch CP2K from ctest.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| sc-Si₂ RI-GPW MP2 (PBE, 30 Ry, Γ) | 576.7 | ~16–30 | **~0.09** (queue) | ✓ | ATV wall may include can. cross-check; not bulk Si |

Campaign queue: → done, harness wave 1
2026-07-25. Handoff: the case `REFERENCE.md`.

## Limitations

- **Γ only.** Multi-k / sampled BZ MP2 is residual; `mp2()` throws on multi-k
  SCF results.
- **Plane-wave RI identity**, not Gaussian cFIT / `RI_AUX` fitting tables.
  Accuracy is grid-cutoff limited (same as SCF density grid / `WFC_GPW`).
- **MP2@PBE**, not MP2@HF. Periodic hybrid/exact exchange is blocked
  (`periodic-hybrid-gpw`).
- **Artificial sc-Si₂** at 30 Ry for the ≤ 60 s e2e budget — not production
  diamond Si₈ / higher cutoffs.
- **Closed-shell only.** UKS / UMP2 / OT path out of scope.
- **CPU only.** CUDA GPW remains blocked on the CP2K loop.
- Correlation energies in the product gate are reported in **Hartree**
  (internal SCF/MP2 convention, cited). Public Frame single-point energies
  elsewhere remain **eV** on `frame.meta`.
- Basis/PP tables are static in-tree headers; unsupported element/name
  **throws** (no silent fallback).

## References

- Offline pin: `regressions/gpw/periodic-ri-mp2-si/REFERENCE.md`
- Regression consumer: `regressions/gpw/periodic-ri-mp2-si/reg_periodic-ri-mp2-si.cpp`
- Oracle twin: `tools/cp2k_oracle/inputs/periodic-ri-mp2-si.inp`
- Oracle contract: `tools/cp2k_oracle/README.md`
- Foothold sibling: [`periodic-mp2-si`](periodic-mp2-si.md) (Poisson
  `DIRECT_CANONICAL`-class; `mp2_ri = false`)
- Narrative GPW chapter: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Correlation chapter (molecular): [`docs/tutorial/ch07-correlation.md`](../ch07-correlation.md)
- Public API: `include/atomiverse/cpu/gpw/mp2.h`,
  `include/atomiverse/cpu/gpw/gpw_solver.h`
- M. Del Ben, J. Hutter, J. VandeVondele, *J. Chem. Theory Comput.* **8**, 4177
  (2012) — RI-GPW MP2 / CP2K `MP2_GPW`.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997);
  *Theor. Chem. Acc.* **103**, 124 (1999) — GPW / Quickstep.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996)
  — PBE.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — GTH.
- M. Krack, *Theor. Chem. Acc.* **114**, 145 (2005) — GTH tables.
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **127**, 114105 (2007) — MOLOPT.
- C. Møller, M. S. Plesset, *Phys. Rev.* **46**, 618 (1934) — MP2.
