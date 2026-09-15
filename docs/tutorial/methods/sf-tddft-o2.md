# Collinear spin-flip full TDDFT (SF-RPA) — O₂ UHF / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `sf-tddft-o2` |
| Regression | `regressions/scf/sf-tddft-o2/` |
| Public API | `cpu::scf::UHF::sf_tddft(n_states)` after `UHF::run` |
| Roadmap | [Excited states](../../roadmap/excited-states.md) — Spin-flip TDDFT ✓ |

## Summary

Collinear spin-flip full linear response (SF-TDHF / SF-RPA) on a high-spin
**UHF** reference: the same Ms-decreasing singles manifold as SF-TDA
(\(i_\alpha\to a_\beta\)), with both \(A\) and \(B\) exchange-only blocks and a
dense non-Hermitian RPA solve. Use when the Tamm–Dancoff approximation is
insufficient and the full \(B\) coupling is required for open-shell multiplet /
spin-flip poles. API is the concrete method `cpu::scf::UHF` with members set
in-place, then `sf_tddft(n)` → `ExcitedStatesResult` (energies [eV],
`tda=false`, `alda=false`). Textbook regression: `regressions/scf/sf-tddft-o2`
(`reg_sf-tddft-o2`), O₂ triplet / STO-3G vs offline PySCF 2.14.0 pin.

## Method

**Reference.** Unrestricted HF, high-spin O₂ (\(2S=2\)), \((n_\alpha,n_\beta)=(9,7)\).
Ground-state SCF supplies spin-channel MOs and orbital energies
\(\varepsilon^\alpha,\varepsilon^\beta\).

**Response space.** Pure spin-flip singles only (majority occupied → minority
virtual). For \(n_\alpha\ge n_\beta\):

\[
\{i_\alpha \to a_\beta\},\qquad
n_{\mathrm{ov}} = n_\alpha^{\mathrm{occ}}\,n_\beta^{\mathrm{virt}}
\]

(STO-3G O₂: \(9\times 3 = 27\)). Oscillator strengths are zero in the collinear
picture (spin-forbidden from the high-spin reference).

**Kernel (Shao, Head-Gordon & Krylov 2003; collinear HF).** Coulomb vanishes for
pure spin-flip; only exchange remains:

\[
A_{ia,jb}
=
\delta_{ij}\delta_{ab}
\bigl(\varepsilon_a^\beta - \varepsilon_i^\alpha\bigr)
- (ij|ab)
\]

\[
B_{ia,jb}
=
-(ib|aj)
\]

with \(i,j\) majority-occupied and \(a,b\) minority-virtual.

**Eigenproblem.** Non-Hermitian RPA:

\[
\begin{bmatrix}A&B\\-B&-A\end{bmatrix}
\begin{bmatrix}X\\Y\end{bmatrix}
=
\omega
\begin{bmatrix}X\\Y\end{bmatrix}
\]

solved densely (`dgeev` on the \(2n_{\mathrm{ov}}\) matrix). On this O₂/STO-3G
UHF reference \((A-B)\) is **not** positive definite
(\(\lambda_{\min}(A-B)\approx -0.46\) Ha), so the Hermitian Casida reduction used
by closed-shell `RKS::tddft` is undefined; the product path is the non-Hermitian
form. Observable: lowest **positive real** SF-RPA roots \(\omega\) [eV].

**Approximations / exclusions.**

- Collinear SF-RPA / SF-TDHF kernel (HF exchange only); no SF-TDDFT XC kernel,
  no noncollinear spin-flip, no UKS SF path.
- Dense non-Hermitian solve only; hard cap \(n_{\mathrm{ov}}\le 256\) (doubled
  space ≤ 512); no iterative Davidson path.
- No SF analytic gradients / NAC on this surface.
- Full RPA (not TDA): compared with SF-TDA on the same reference, the \(B\)
  block shifts the spectrum; near-zero multiplet roots of SF-TDA are no longer
  the lowest positive RPA poles under indefinite \((A-B)\).

## Implementation

Public surface: `cpu::Frame` + concrete `cpu::scf::UHF` / `ROHF` (warm-start).
No `*Config` bags; parameters are constructor args and public members.

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

// O₂, r(O–O)=1.2075 Å, Z=8; box cubic 20 Å (see reg)
cpu::Frame frame = /* … */;

// Multi-minima: ROHF → UHF density guess (same pattern as spin-flip-tda-o2)
cpu::scf::ROHF rohf(9, 7);
rohf.basis = cpu::scf::BasisSet::STO3G;
rohf.tol = 1e-12;
rohf.max_iter = 200;
rohf.run(frame);

using Spin = cpu::scf::SlaterDeterminant::Spin;
cpu::scf::SCFInitialGuess guess;
guess.P_alpha = rohf.determinant().density(Spin::alpha);
guess.P_beta  = rohf.determinant().density(Spin::beta);
guess.has_alpha = guess.has_beta = true;

cpu::scf::UHF uhf(9, 7);
uhf.basis = cpu::scf::BasisSet::STO3G;
uhf.tol = 1e-12;
uhf.max_iter = 200;
uhf.run(frame, &guess);

cpu::scf::ExcitedStatesResult es = uhf.sf_tddft(/*n_states=*/5);
// es.energy_ev[i]  [eV];  es.tda == false;  es.alda == false
```

**Build / run regression**

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_sf-tddft-o2 -j$(nproc)
  ./build/regressions/reg_sf-tddft-o2
'
```

**File pointers**

| Role | Path |
| ---- | ---- |
| Public API | `include/atomiverse/cpu/scf/uhf.h` (`UHF::sf_tddft`) |
| Result type | `include/atomiverse/cpu/scf/tdhf.h` (`ExcitedStatesResult`) |
| Implementation | `src/cpu/scf/tdhf.cpp` (`UHF::sf_tddft`) |
| Regression | `regressions/scf/sf-tddft-o2/reg_sf-tddft-o2.cpp` |
| Offline pin | `regressions/scf/sf-tddft-o2/REFERENCE.md` |
| Oracle | `tools/pyscf_oracle/pin_scf.py sf-tddft` (never in ctest) |

## Validation (vs PySCF)

Source: [`regressions/scf/sf-tddft-o2/REFERENCE.md`](../../../regressions/scf/sf-tddft-o2/REFERENCE.md)
(offline pin only; **no live PySCF in ctest**).

| Setting | Value |
| ------- | ----- |
| System | O₂, \(r(\mathrm{O{-}O})=1.2075\) Å, charge 0, \(2S=2\) |
| Geometry | O (0,0,0); O (0,0,1.2075) [Å] |
| Basis | STO-3G |
| Angular | Cartesian (`cart=True`; no \(d\) on STO-3G) |
| Occupation | \((n_\alpha,n_\beta)=(9,7)\); \(n_{\mathrm{ov}}=27\) |
| GS SCF | UHF, `conv_tol=1e-12` (ROHF warm-start → UHF) |
| Response | dense non-Hermitian `dgeev` of \(2n_{\mathrm{ov}}\) RPA matrix; `nstates=5` |
| Oracle | PySCF **2.14.0** UHF MOs + `ao2mo` SF \(A,B\) |
| Ha→eV | 27.211386245988 (CODATA; `units::kHartreeInEv`) |
| Job gate | \(\lvert\omega_i - \mathrm{pin}_i\rvert < 10^{-4}\) eV |

PySCF mainline has no product SF-TDDFT; the twin rebuilds \(A,B\) from UHF MOs
(`pin_scf.py sf-tddft` / REFERENCE verbatim script) and reports the lowest
positive real RPA poles.

| Quantity | Atomiverse / pin | PySCF pin | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E_{\mathrm{UHF}}\) [Ha] | −147.633946785450 | −147.633946785450 | \(10^{-4}\) Ha (sanity) | \(S^2\approx 2.003411\) |
| \(a_{\min}(A-B)\) [Ha] | −0.461612348195 | −0.461612348195 | (diagnostic) | Casida Hermitian form undefined |
| \(\omega_0\) [eV] | 4.334691644974 | 4.334691644974 | \(10^{-4}\) eV | lowest positive SF-RPA root |
| \(\omega_1\) [eV] | 7.155381562193 | 7.155381562193 | \(10^{-4}\) eV | degenerate pair with \(\omega_2\) |
| \(\omega_2\) [eV] | 7.155381562193 | 7.155381562193 | \(10^{-4}\) eV | |
| \(\omega_3\) [eV] | 8.181367981325 | 8.181367981325 | \(10^{-4}\) eV | degenerate pair with \(\omega_4\) |
| \(\omega_4\) [eV] | 8.181367981325 | 8.181367981325 | \(10^{-4}\) eV | |

Pins copied verbatim from REFERENCE (2026-07-23). Regression hard-codes the same
five roots and UHF energy. SF-TDA roots on the same reference (0.045 / 1.568 /
1.568 / 3.038 / 7.849 eV) are **not** interchangeable with these SF-RPA poles.

## Performance

Source: `regressions/scf/sf-tddft-o2/REFERENCE.md` § Performance (also handoff
ratio **9.51**). Protocol: median of 5 walls; `OMP_NUM_THREADS=1`; fair twin =
ROHF warm-start → UHF + dense SF \(A,B\) non-Hermitian RPA, same geom/basis/method
class.

```bash
export OMP_NUM_THREADS=1 OMP_PROC_BIND=close
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method sf_tddft --geom o2 --basis sto-3g --spin 2 --nstates 5 --repeat 5 \
  -- ./build/regressions/reg_sf-tddft-o2
```

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r=T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| O₂ UHF/STO-3G SF-RPA (`sf_tddft`) | **0.0412** | **0.392** | **9.51** | ✓ | Twin is custom SF \(A,B\)+`eigvals`, not product PySCF SF-TDDFT; micro-system walls are noisy; ATV does full ROHF+UHF+SF path in the regression binary |

Gate: \(r\le 10\) unless exempted — **within_10x = true**.

## Limitations

- **Scope:** collinear SF-RPA (SF-TDHF) on UHF only. No SF-TDDFT XC kernel, no
  noncollinear SF, no UKS SF path.
- **Dense only:** \(n_{\mathrm{ov}}>256\) throws; no production Davidson / disk
  path for large SF spaces.
- **Indefinite \((A-B)\):** Hermitian Casida reduction is not used; complex /
  non-positive poles can appear in general open-shell SF-RPA — product surface
  reports lowest positive real roots only.
- **Properties:** energies only on this API; no SF oscillator strengths
  (spin-forbidden), gradients, or NAC.
- **Spectrum shift vs SF-TDA:** full \(B\) coupling reorders poles relative to
  `UHF::sf_tda` on the same reference; do not mix pins across the two methods.
- **Speed debt:** \(r\approx 9.51\) passes the 10× gate but is near the ceiling
  on this micro-case; SCF/ERI cost dominates the twin comparison.
- **Roadmap (open):** SF XC kernels, iterative solvers / larger bases, CUDA —
  see `docs/roadmap/excited-states.md`.

## References

- Offline pin: `regressions/scf/sf-tddft-o2/REFERENCE.md`
- Perf numbers: same REFERENCE § Performance; rollup context the case `REFERENCE.md` (ratio 9.51)
- Sibling TDA report: `docs/tutorial/methods/spin-flip-tda-o2.md`
- Oracle: `tools/setup_pyscf_env.sh`, `tools/pyscf_oracle/pin_scf.py sf-tddft`,
  `tools/pyscf_oracle/fair_bench.py --method sf_tddft`
- Y. Shao, M. Head-Gordon, A. I. Krylov, *J. Chem. Phys.* **118**, 4807 (2003)
  — SF-TDDFT/CIS
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDHF/RPA response
  review
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA
  (SF-TDA sibling)
- Roadmap: `docs/roadmap/excited-states.md`
