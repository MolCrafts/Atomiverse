# GPW collinear multi-k UKS — H₂ identity suite + O₂ CP2K twin

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-kpoints-uks` |
| Regression | `regressions/gpw/gpw-kpoints-uks/` |
| Public API | `cpu::gpw::GpwDft` (`n_alpha`/`n_beta`, `k_mesh` / `k_points`+`k_point_weights`, `k_time_reversal`, `XcKind`) |
| Track | CP2K capability () |
| Roadmap | [dft.md](../../roadmap/dft.md) — multi-k collinear UKS GPW; queue `gpw-kpoints-uks` (done 2026-07-25) |

## Summary

**Collinear spin-polarized Gaussian-and-plane-waves (GPW) Kohn–Sham with
Monkhorst–Pack Brillouin-zone sampling** — the product ladder step after
restricted multi-k GPW (`gpw-kpoints-si`) and Γ-only UKS (`gpw-uks-o2-gamma`,
`gpw-uks-pbe`). Concrete method `cpu::gpw::GpwDft` with dual-spin occupations
(`n_alpha`/`n_beta`), complex Bloch densities per \(\mathbf{k}\), spin-resolved
XC, and optional time-reversal folding of \(\pm\mathbf{k}\).

The **ctest regression** is a fast **H₂ high-spin identity suite** (compact
DZV-GTH-PADE, wall ~4 s class): Γ-limit, closed-shell UKS ≡ RKS multi-k,
time-reversal 8↔4, k-sampling activation, and a frozen absolute \(E_{\mathrm{tot}}\)
pin. The **offline CP2K product twin** is **triplet O₂** UKS-PBE / DZVP-MOLOPT-GTH
on a periodic \(2\times2\times2\) mesh (not re-run in ctest; wall exceeds the
60 s class on dual Γ+MP MOLOPT). Public energy on `Frame` remains **eV** when
routed through evaluators; this regression uses `GpwDft` internals in **Ha**.
Gate: `reg_gpw-kpoints-uks`. **No live CP2K in ctest** — offline pins only.
**perf_exempt** (fair twin would need ATV O₂ MOLOPT multi-k wall; H₂ has no
CP2K twin).

Related narrative: [Ch. 9 GPW](../ch09-gpw.md). PW multi-k collinear UKS
(different basis/PP): [`pw-uks-kmesh-fe`](pw-uks-kmesh-fe.md).

## Method

**GPW (Quickstep family).** Contracted Gaussian AO wavefunction; density on a
uniform real-space grid; Hartree via FFT Poisson; XC on the grid; potential
back-projected to the AO basis. Core electrons via analytic **GTH**
pseudopotentials. Grid set by the plane-wave cutoff in Rydberg. Citation:
Lippert, Hutter, Parrinello, *Mol. Phys.* **92**, 477 (1997).

**Multi-k (Monkhorst–Pack).** With \(N_d > 1\) on any axis (or an explicit
`k_points` list), the solver takes the **complex Bloch** path:

- Per-\(\mathbf{k}\) dual spin densities \(P_k^\alpha\), \(P_k^\beta\) and
  Hamiltonians \(H_k^\sigma\).
- k-weights from the MP mesh (CP2K convention
  \(\kappa_d = (2r - N_d - 1)/(2 N_d)\); even \(N\) is Γ-avoiding / shifted).
- Optional **time reversal** (`k_time_reversal = true`): fold \(\pm\mathbf{k}\)
  pairs (exact for collinear RKS/UKS without SOC); retained partner gets
  doubled weight. Even \(2\times2\times2\) → 8 → **4** irreducible k under TR.

Citation: Monkhorst & Pack, *Phys. Rev. B* **13**, 5188 (1976).

**UKS / collinear LSD.** With \(N_\alpha + N_\beta > 0\), occupations are
explicit integer members `n_alpha` / `n_beta` (not a multiplicity bag).
Hartree from total density \(\rho = \rho_\alpha + \rho_\beta\); XC
spin-resolved (PADE LSD on the identity suite; product twin uses PBE).
Magnetization for fixed fillings:

\[
M = N_\alpha - N_\beta
\]

(identity of the occupation assignment; multi-k does not change fixed
integer fillings).

**Regression system (identity suite).** High-spin H₂ toy — same class of
internal exactness identities as `gpw-kpoints-si`, extended to UKS:

| Item | Value |
| ---- | ----- |
| Cell | \(8\times8\times8\) Bohr, periodic |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| Spin | \(N_\alpha=2\), \(N_\beta=0\) (high-spin toy; \(M=2\)) |
| XC | PADE LSD |
| Cutoff (main pin) | 120 Ry |
| k mesh | MONKHORST-PACK \(2\times2\times2\), TR → 4 k |

**Offline product twin (documentation only).** O₂ \(R=1.2075\) Å, cubic
**8 Å** periodic, DZVP-MOLOPT-GTH + GTH-PBE-q6, **UKS PBE**, multiplicity 3
(\(N_\alpha=7\), \(N_\beta=5\)), CUTOFF 200 Ry, NGRIDS 1, MP \(2\times2\times2\).

**Key exclusions (this slug).**

- No **analytic multi-k UKS forces / stress** (see `gpw-kpoint-forces`,
  `gpw-kpoint-stress`, `gpw-uks-forces`).
- No **OT** / **DFT+U** / **multigrid** / **nonperiodic MT** on the k path
  (throw at `build()`).
- No **noncollinear** spin / SOC; no metallic smearing on this dual-spin path.
- No **ATV O₂ MOLOPT multi-k** absolute energy in ctest (wall budget); O₂ is
  the offline CP2K product pin only.
- Not plane-wave NC-UPF UKS (`pw-uks-kmesh-fe` / `pw-spin-polarized-o2`).

## Implementation

Public surface only: construct `cpu::gpw::GpwDft`, set members (**no**
`*Config` bag, **no** `make_*` / `build_*` factories), `build(atoms)` →
`run(atoms)`. Lookup constructors for basis/PP only:
`GaussianBasis(Element, name)`, `GthPotential(Element, name)`. Header:
`include/atomiverse/cpu/gpw/gpw_solver.h`. Internals stay under
`cpu/gpw/internal/`.

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV;
using cpu::gpw::GpwDft;
using cpu::gpw::XcKind;

// High-spin H₂ toy (Bohr); matches reg_gpw-kpoints-uks main pin settings
const double L = 8.0;  // Bohr
const cpu::gpw::GaussianBasis bH(molrs::Element::H, "DZV-GTH-PADE");
const cpu::gpw::GthPotential pH(molrs::Element::H, "GTH-PADE-q1");
const double r0[3] = {0.5 * L - 0.7, 0.5 * L, 0.5 * L};
const double r1[3] = {0.5 * L + 0.7, 0.5 * L, 0.5 * L};
std::vector<cpu::gpw::GpwAtom> atoms;
atoms.emplace_back(r0, bH, pH, 1);
atoms.emplace_back(r1, bH, pH, 1);

GpwDft mp;
mp.cell[0] = mp.cell[1] = mp.cell[2] = L;
mp.cutoff_ry = 120.0;
mp.n_alpha = 2;                 // high-spin toy; M = 2
mp.n_beta = 0;
mp.max_iter = 80;
mp.tol = 1e-7;
mp.xc = XcKind::PADE;
mp.k_mesh[0] = mp.k_mesh[1] = mp.k_mesh[2] = 2;  // multi-k UKS
mp.k_time_reversal = true;      // 8 → 4 irreducible k
mp.build(atoms);
const cpu::gpw::GpwResult& r = mp.run(atoms);
// r.e_total [Ha], r.converged, r.n_kpts (== 4 under TR), r.eps_k, r.k_frac
```

Explicit Γ on the k path (for the Γ-limit identity) uses
`k_points = {{0,0,0}}` and `k_point_weights = {1.0}` instead of `k_mesh`.
Closed-shell multi-k RKS still uses `n_occ`; matching UKS uses
`n_alpha = n_beta = n_occ`.

Build (GPW needs HF+PW+FFTW3) and run the regression:

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
  cmake --build build --target reg_gpw-kpoints-uks -j$(nproc)
  ./build/regressions/reg_gpw-kpoints-uks
'
```

**File pointers**

| Role | Path |
| ---- | ---- |
| Public API | `include/atomiverse/cpu/gpw/gpw_solver.h` (`GpwDft`) |
| Solver impl | `src/cpu/gpw/gpw_solver.cpp` |
| Regression | `regressions/gpw/gpw-kpoints-uks/reg_gpw-kpoints-uks.cpp` |
| Offline pin | `regressions/gpw/gpw-kpoints-uks/REFERENCE.md` |
| CP2K twin input | `tools/cp2k_oracle/inputs/gpw-kpoints-uks/o2_uks_kpoints.inp` |
| Related Γ UKS | `regressions/gpw/gpw-uks-o2-gamma`, `gpw-uks-pbe` |
| Related RKS k | `regressions/gpw/gpw-kpoints-si` |

## Validation (CP2K)

Source: [`regressions/gpw/gpw-kpoints-uks/REFERENCE.md`](../../../regressions/gpw/gpw-kpoints-uks/REFERENCE.md)
(offline only; **never live CP2K in ctest**). Closed acceptance 2026-07-25;
spec history under git (`.claude/specs/gpw-kpoints-uks/` deleted at done).

### Identity gates (decisive; independent of oracle)

| Gate | Result (2026-07-25) | Tol | Notes |
| ---- | ------------------: | --- | ----- |
| A1 Γ-limit (\(\mathbf{k}=\Gamma\) complex vs real Γ-UKS) | \(\lvert\Delta\rvert \approx 4\times10^{-11}\) Ha | \(1\times10^{-6}\) Ha | same occupations / cell / cutoff |
| A2 closed-shell multi-k UKS ≡ multi-k RKS | \(\lvert\Delta\rvert \approx 0\) | \(5\times10^{-6}\) Ha | \(N_\alpha=N_\beta=1\) vs `n_occ=1` |
| A3 time-reversal 8 k vs 4 k | \(\lvert\Delta\rvert \approx 0\) | \(5\times10^{-6}\) Ha | `n_kpts` 8 vs 4; energies match |
| A4 k-sampling activation \(\lvert E_{\mathrm{MP}}-E_\Gamma\rvert\) | \(2.79\times10^{-2}\) Ha | \(\ge 1\times10^{-4}\) Ha | mesh must change energy |

### Absolute pin (first-green ATV, H₂ suite)

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------- | --- | ----- |
| \(E_{\mathrm{MP}}\) [Ha] | **−0.719059673672** | frozen first-green ATV (REFERENCE) | \(1\times10^{-3}\) Ha | MP \(2\times2\times2\) TR, 120 Ry, PADE |
| \(E_\Gamma\) [Ha] | −0.691163126674 | same suite (diagnostic) | — | activation vs MP |
| \(M = N_\alpha - N_\beta\) | **2** | fixed occupations | exact | high-spin H₂ toy |
| SCF | converged | — | — | `max_iter=80`, `tol=1e-7` |
| Payload | `n_kpts==4`, `eps_k` / `k_frac` / weights | — | shapes | TR-folded mesh |

### Offline CP2K product twin (O₂, documentation)

Not a ctest energy gate (H₂ identities prove the k-UKS path; O₂ absolute
product energy is for future ATV re-twins).

| Setting | Value |
| ------- | ----- |
| Engine | CP2K **2026.1** (`cp2k.psmp`, libcp2k 2026.1) |
| Input | `tools/cp2k_oracle/inputs/gpw-kpoints-uks/o2_uks_kpoints.inp` |
| System | O₂, \(R(\mathrm{O{-}O})=1.2075\) Å, cubic **8 Å**, `PERIODIC XYZ` |
| Coordinates (Å) | O (4, 4, 3.39625), O (4, 4, 4.60375) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q6 |
| XC | PBE, `LSD`, multiplicity 3 → \(N_\alpha=7\), \(N_\beta=5\) |
| Cutoff | 200 Ry, NGRIDS 1, REL_CUTOFF 40 |
| **k**-mesh | MONKHORST-PACK \(2\times2\times2\) |
| SCF twin | diagonalization + Broyden α=0.3, `EPS_SCF=1e-6` |
| Date | 2026-07-25 |

| Quantity | CP2K pin | Notes |
| -------- | -------: | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.92842018097085** | SCF converged in **22** steps |
| Wall (1 rank × OMP=4) | **≈ 175 s** | `CP2K` timer MAXIMUM |

Key CP2K log lines (from REFERENCE):

```text
  *** SCF run converged in    22 steps ***
  Total energy:                                               -31.92842018097085
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.928420180970853
```

Offline re-run (operator only; not ctest):

```bash
# See tools/cp2k_oracle/README.md — load CP2K module / local install;
# never invoke from ctest.
cd /tmp && # link BASIS_MOLOPT, GTH_POTENTIALS into cwd
# copy tools/cp2k_oracle/inputs/gpw-kpoints-uks/o2_uks_kpoints.inp
mpirun -np 1 cp2k.psmp -i o2_uks_kpoints.inp -o o2_uks_kpoints.out
```

## Performance

Source: `REFERENCE.md` § Perf; queue/handoff mark this slug **perf_exempt**.

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| Reason | Fair twin wall would need ATV O₂ MOLOPT multi-k product run (not the H₂ identity suite); H₂ has no CP2K twin |
| \(T_{\mathrm{ATV}}\) (regression A1–A5, OMP=8, Release) | **~4.4 s** |
| \(T_{\mathrm{CP2K}}\) (O₂ product twin) | ~175 s |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a** (exempt) |

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -----------------------: | ----------------------: | ----: | ---- | ------ |
| H₂ multi-k UKS identity suite | n/a | ~4.4 | **n/a** | exempt | No CP2K H₂ twin; not product geometry |
| O₂ UKS-PBE MOLOPT MP \(2\times2\times2\) | ~175 | n/a (not in ctest) | **n/a** | exempt | Product documentation pin only |

Dual gate: (1) identities A1–A4 + absolute \(E_{\mathrm{MP}}\) A5 with SCF
converge; (2) perf exempt with reason above. Queue row: ratio **perf_exempt**
(2026-07-25). Wall budget: e2e ≤ 60 s single-rank CPU (H₂ suite).

When an ATV O₂ product wall is recorded offline against the same input,
replace with measured \(r \le 10\).

## Limitations

- **H₂ identities, not O₂ absolute in ctest.** Product O₂ MOLOPT multi-k is
  documented via CP2K offline only; ATV absolute O₂ re-pin is follow-on when
  wall budget allows.
- **Single density grid** on the k path; CP2K product twin uses NGRIDS 1 for
  the O₂ pin (multigrid parity is separate product work).
- **Fixed occupations only** — no metallic smearing / fractional fillings on
  this dual-spin multi-k path.
- **No forces / stress** in acceptance for this slug (separate k-point and
  UKS force/stress regressions).
- **k-path constraints:** periodic, `local_pp_on_grid`, collinear RKS/UKS;
  OT / DFT+U / multigrid / nonperiodic throw at `build()`.
- **Out of scope forever for this page:** noncollinear spin, SOC, hybrid GPW
  k-points, CUDA GPW, plane-wave NC-UPF multi-k UKS (see `pw-uks-kmesh-fe`).

## References

- Offline pin: [`regressions/gpw/gpw-kpoints-uks/REFERENCE.md`](../../../regressions/gpw/gpw-kpoints-uks/REFERENCE.md)
- Regression consumer: [`reg_gpw-kpoints-uks.cpp`](../../../regressions/gpw/gpw-kpoints-uks/reg_gpw-kpoints-uks.cpp)
- CP2K twin: [`tools/cp2k_oracle/inputs/gpw-kpoints-uks/`](../../../tools/cp2k_oracle/inputs/gpw-kpoints-uks/)
- Oracle contract: [`tools/cp2k_oracle/README.md`](../../../tools/cp2k_oracle/README.md)
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md)
- Related Γ-UKS GPW: [`gpw-uks-o2-gamma`](gpw-uks-o2-gamma.md)
- Related PW multi-k UKS: [`pw-uks-kmesh-fe`](pw-uks-kmesh-fe.md)
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — MP mesh
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
