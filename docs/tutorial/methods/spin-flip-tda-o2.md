# Collinear spin-flip TDA (SF-CIS) — O₂ UHF / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `spin-flip-tda-o2` |
| Regression | `regressions/scf/spin-flip-tda-o2/` |
| Public API | `cpu::scf::UHF::sf_tda(n_states)` after `UHF::run` |
| Roadmap | [Excited states](../../roadmap/excited-states.md) — Spin-flip TDA ✓ |

## Summary

Collinear spin-flip Tamm–Dancoff (SF-CIS / SF-TDA) on a high-spin **UHF**
reference: Ms-decreasing singles \(i_\alpha\to a_\beta\) (or \(\beta\to\alpha\)
when \(n_\beta>n_\alpha\)) form the response space; the collinear HF kernel is
exchange-only. Use for open-shell multiplet reconstruction and spin-flip roots
when a product SF-TDDFT path is not required. API is the concrete method
`cpu::scf::UHF` with members set in-place, then `sf_tda(n)` →
`ExcitedStatesResult` (energies [eV], `tda=true`, `alda=false`). Textbook
regression: `regressions/scf/spin-flip-tda-o2` (`reg_spin-flip-tda-o2`), O₂
triplet / STO-3G vs offline PySCF 2.14.0 pin.

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

**Hamiltonian (Shao, Head-Gordon & Krylov 2003; collinear HF).** Coulomb vanishes
for pure spin-flip; only exchange remains:

\[
A_{ia,jb}
=
\delta_{ij}\delta_{ab}
\bigl(\varepsilon_a^\beta - \varepsilon_i^\alpha\bigr)
- (ij|ab)
\]

with \(i,j\) majority-occupied and \(a,b\) minority-virtual. Dense Hermitian
solve \(A X = \omega X\) (`eigh` of full \(A\)).

**Approximations / exclusions.**

- Tamm–Dancoff only (no \(B\) block / full SF-TDDFT RPA).
- Collinear SF-CIS kernel (HF exchange); no SF-TDDFT XC kernel, no noncollinear
  spin-flip.
- Dense \(A\), hard cap \(n_{\mathrm{ov}}\le 512\) (no iterative Davidson path).
- No SF analytic gradients / NAC on this surface.
- Root 0 may sit near zero (Ms multiplet reconstruction under mild UHF spin
  contamination); physical roots start higher.

## Implementation

Public surface: `cpu::Frame` + concrete `cpu::scf::UHF` / `ROHF` (warm-start).
No `*Config` bags; parameters are constructor args and public members.

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

// O₂, r(O–O)=1.2075 Å, Z=8; box cubic 20 Å (see reg)
cpu::Frame frame = /* … */;

// Multi-minima: ROHF → UHF density guess (same pattern as ump2-o2-sto3g)
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

cpu::scf::ExcitedStatesResult es = uhf.sf_tda(/*n_states=*/5);
// es.energy_ev[i]  [eV];  es.tda == true;  es.alda == false
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
  cmake --build build --target reg_spin-flip-tda-o2 -j$(nproc)
  ./build/regressions/reg_spin-flip-tda-o2
'
```

**File pointers**

| Role | Path |
| ---- | ---- |
| Public API | `include/atomiverse/cpu/scf/uhf.h` (`UHF::sf_tda`) |
| Result type | `include/atomiverse/cpu/scf/tdhf.h` (`ExcitedStatesResult`) |
| Implementation | `src/cpu/scf/tdhf.cpp` (`UHF::sf_tda`) |
| Regression | `regressions/scf/spin-flip-tda-o2/reg_spin-flip-tda-o2.cpp` |
| Offline pin | `regressions/scf/spin-flip-tda-o2/REFERENCE.md` |
| Oracle | `tools/pyscf_oracle/pin_scf.py sf-tda` (never in ctest) |

## Validation (vs PySCF)

Source: [`regressions/scf/spin-flip-tda-o2/REFERENCE.md`](../../../regressions/scf/spin-flip-tda-o2/REFERENCE.md)
(offline pin only; **no live PySCF in ctest**).

| Setting | Value |
| ------- | ----- |
| System | O₂, \(r(\mathrm{O{-}O})=1.2075\) Å, charge 0, \(2S=2\) |
| Geometry | O (0,0,0); O (0,0,1.2075) [Å] |
| Basis | STO-3G |
| Angular | Cartesian (`cart=True`; no \(d\) on STO-3G) |
| Occupation | \((n_\alpha,n_\beta)=(9,7)\); \(n_{\mathrm{ov}}=27\) |
| GS SCF | UHF, `conv_tol=1e-12` (ROHF warm-start → UHF) |
| Response | dense `eigh` of full \(A\); `nstates=5` |
| Oracle | PySCF **2.14.0** UHF MOs + `ao2mo` SF-CIS matrix |
| Ha→eV | 27.211386245988 (CODATA; `units::kHartreeInEv`) |
| Job gate | \(\lvert\omega_i - \mathrm{pin}_i\rvert < 10^{-4}\) eV |

PySCF mainline has no product SF-TDA; the twin rebuilds \(A\) from UHF MOs
(`pin_scf.py sf-tda` / REFERENCE verbatim script).

| Quantity | Atomiverse / pin | PySCF pin | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E_{\mathrm{UHF}}\) [Ha] | −147.633946785450 | −147.633946785450 | \(10^{-4}\) Ha (sanity) | \(S^2\approx 2.003411\) |
| \(\omega_0\) [eV] | 0.045146996469 | 0.045146996469 | \(10^{-4}\) eV | near-zero multiplet / contamination |
| \(\omega_1\) [eV] | 1.567707515056 | 1.567707515056 | \(10^{-4}\) eV | degenerate pair with \(\omega_2\) |
| \(\omega_2\) [eV] | 1.567707515056 | 1.567707515056 | \(10^{-4}\) eV | |
| \(\omega_3\) [eV] | 3.038448959259 | 3.038448959259 | \(10^{-4}\) eV | |
| \(\omega_4\) [eV] | 7.848628442263 | 7.848628442263 | \(10^{-4}\) eV | |

Pins copied verbatim from REFERENCE (2026-07-23). Regression hard-codes the same
five roots and UHF energy.

## Performance

Source: `regressions/scf/spin-flip-tda-o2/REFERENCE.md` § Performance (also
handoff ratio **9.56**). Protocol: median of 5 walls; `OMP_NUM_THREADS=1`; fair
twin = ROHF warm-start → UHF + dense SF-CIS \(A\), same geom/basis/method class.

```bash
export OMP_NUM_THREADS=1 OMP_PROC_BIND=close
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method sf_tda --geom o2 --basis sto-3g --spin 2 --nstates 5 --repeat 5 \
  -- ./build/regressions/reg_spin-flip-tda-o2
```

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r=T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| O₂ UHF/STO-3G SF-CIS (`sf_tda`) | **0.0326** | **0.311** | **9.56** | ✓ | Twin is custom SF-CIS \(A\), not product PySCF SF-TDA; micro-system walls are noisy; ATV does full ROHF+UHF+SF path in the regression binary |

Gate: \(r\le 10\) unless exempted — **within_10x = true**.

## Limitations

- **Scope:** collinear SF-CIS on UHF only. No spin-flip TDDFT (XC), no full
  SF-RPA (\(A/B\)), no noncollinear SF, no UKS SF path.
- **Dense only:** \(n_{\mathrm{ov}}>512\) throws; no production Davidson / disk
  path for large SF spaces.
- **Properties:** energies only on this API; no SF oscillator strengths (spin-
  forbidden), gradients, or NAC.
- **Reference quality:** root 0 tracks Ms reconstruction / UHF contamination —
  not a spectroscopic “first excited state” without analysis.
- **Speed debt:** \(r\approx 9.56\) passes the 10× gate but is near the ceiling on
  this micro-case; SCF/ERI cost dominates the twin comparison.
- **Roadmap (open):** SF-TDDFT, full SF-LR, larger-basis / iterative solvers,
  CUDA — see `docs/roadmap/excited-states.md` (Spin-flip TDDFT ○).

## References

- Offline pin: `regressions/scf/spin-flip-tda-o2/REFERENCE.md`
- Perf numbers: same REFERENCE § Performance; rollup context the case `REFERENCE.md` (ratio 9.56)
- Oracle: `tools/setup_pyscf_env.sh`, `tools/pyscf_oracle/pin_scf.py sf-tda`,
  `tools/pyscf_oracle/fair_bench.py --method sf_tda`
- Y. Shao, M. Head-Gordon, A. I. Krylov, *J. Chem. Phys.* **118**, 4807 (2003)
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — response review
- Roadmap: `docs/roadmap/excited-states.md`
