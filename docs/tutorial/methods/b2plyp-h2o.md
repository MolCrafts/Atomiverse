# B2PLYP double hybrid — H₂O / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `b2plyp-h2o` |
| Regression | `regressions/scf/b2plyp-h2o/` |
| Public API | `cpu::scf::RKS{XCType::B2PLYP}` + `mp2(frame)` with \(a_c\) scales |
| Roadmap | `docs/roadmap/dft.md` §7 Double-Hybrid Functionals |

## Summary

Restricted B2PLYP (Grimme 2006) total energy on gas-phase water / STO-3G: hybrid KS (\(a_x=0.53\) HF + semi-local B88/LYP) plus scaled canonical RMP2 on the KS orbitals (\(a_c=0.27\)). Public path is concrete `cpu::scf::RKS` with `XCType::B2PLYP`; set `mp2_os_scale = mp2_ss_scale = hybrid_pt2_fraction(B2PLYP)` and call `mp2(frame)` — KS SCF alone is not the double-hybrid total. Offline accuracy pin is PySCF 2.14.0 composed as `0.53*HF + 0.47*B88 + 0.73*LYP` + \(a_c\cdot E_{\mathrm{MP2}}\) (hosted libxc often lacks a named B2PLYP composite). Product gate: `regressions/scf/b2plyp-h2o` (`reg_b2plyp-h2o`); fair wall ratio \(r=T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\approx 1.75\) (≤10×).

## Method

**Double-hybrid XC (Grimme B2PLYP).**

\[
E_{\mathrm{xc}}
  = (1-a_x)\,E_x^{\mathrm{B88}}
  + a_x\,E_x^{\mathrm{HF}}
  + (1-a_c)\,E_c^{\mathrm{LYP}}
  + a_c\,E_c^{\mathrm{MP2}},
\qquad a_x = 0.53,\quad a_c = 0.27.
\]

Semi-local absolute scales in the grid evaluator: \(0.47\) B88 + \(0.73\) LYP. Exact exchange \(a_x\) is injected by the Fock builder (`hybrid_exchange_fraction`); PT2 is **not** inside `evaluate_xc`.

**Spin / reference.** Closed-shell restricted KS (RKS): one set of spatial MOs, \(n_\alpha=n_\beta\). Correlation is closed-shell canonical RMP2 on the converged KS orbitals and orbital energies (same composition class as PySCF `mp.MP2(mf)`). Spin-component scales are set equal, \(c_{\mathrm{OS}}=c_{\mathrm{SS}}=a_c\), so

\[
E_{\mathrm{corr}} = a_c\,(E_{\mathrm{OS}}+E_{\mathrm{SS}}),\qquad
E_{\mathrm{tot}} = E_{\mathrm{KS}} + E_{\mathrm{corr}}.
\]

**Key approximations / exclusions.**

- Canonical \(O(N^5)\) RMP2 (no RI, no frozen core on this foothold: `ri=false`, `frozen=0`).
- MatchPySCF quadrature for the hybrid-KS semi-local part.
- No B2PLYP analytic forces, Hessian, TDDFT, or open-shell (UKS/ROKS) double hybrid in this regression.
- No other double hybrids (B2GP-PLYP, DSD-*, ωB97X-2, SCS-DH) on this path — roadmap rows remain ○.

## Implementation

Public surface only (`cpu::Frame` + concrete method members; no `*Config`, no factories):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

// H₂O geometry [Å] as in regressions/scf/b2plyp-h2o; 20 Å cubic box
ATV::cpu::Frame frame(3);
// ... type / x,y,z / box ...

using ATV::cpu::scf::XCType;
ATV::cpu::scf::RKS rks(XCType::B2PLYP);
rks.basis = ATV::cpu::scf::BasisSet::STO3G;
rks.tol = 1e-10;
rks.max_iter = 100;

const double a_c = ATV::cpu::scf::hybrid_pt2_fraction(XCType::B2PLYP); // 0.27
rks.mp2_os_scale = a_c;
rks.mp2_ss_scale = a_c;

const ATV::cpu::scf::MP2Result dh = rks.mp2(frame);
// rks.energy()      [Ha] hybrid KS only
// dh.e_corr_os/ss   [Ha] unscaled channels
// dh.e_corr         [Ha] a_c * (OS+SS)
// dh.e_tot          [Ha] E_KS + a_c * E_MP2
```

Functional metadata (no bag type):

| Helper | B2PLYP value |
| ------ | -----------: |
| `hybrid_exchange_fraction` | 0.53 |
| `hybrid_pt2_fraction` | 0.27 |
| `is_gga` | `true` |

Build and run the regression:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_b2plyp-h2o -j$(nproc)
  ./build/regressions/reg_b2plyp-h2o
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/rks.h` | `RKS` ctor / `XCType` latch |
| `include/atomiverse/cpu/scf/xc_functional.h` | `XCType::B2PLYP`, hybrid/PT2 fractions, `evaluate_xc` |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `mp2`, `mp2_os_scale` / `mp2_ss_scale` |
| `src/cpu/scf/xc_functional.cpp` | B2PLYP semi-local grid + fraction tables |
| `regressions/scf/b2plyp-h2o/reg_b2plyp-h2o.cpp` | Product gate |
| `regressions/scf/b2plyp-h2o/REFERENCE.md` | Offline pins + fair_bench numbers |

## Validation (vs PySCF)

Source: [`regressions/scf/b2plyp-h2o/REFERENCE.md`](../../../regressions/scf/b2plyp-h2o/REFERENCE.md).
Oracle: offline PySCF **2.14.0** composed B2PLYP (not a golden of Atomiverse output; never live-import PySCF in ctest).

**Setup:**

| Item | Value |
| ---- | ----- |
| Molecule | H₂O, charge 0, closed shell |
| Geometry [Å] | O (0, 0, 0.1173); H (0, ±0.7572, −0.4692) |
| Box | 20 Å cubic (gas-phase cell only) |
| Basis | STO-3G |
| Angular | **Cartesian** (`cart=True`; s/p only) |
| Oracle SCF | `conv_tol = 1e-12` |
| Atomiverse SCF | `tol = 1e-10`, MatchPySCF grid |
| PySCF XC string | `0.53*HF + 0.47*GGA_X_B88 + 0.73*GGA_C_LYP` |
| Product tol | **1×10⁻⁶ Ha** on e_ks, unscaled MP2, PT2, e_tot |

**Pinned energies [Hartree]:**

| Quantity | Pin (PySCF 2.14.0) | Tol | Notes |
| -------- | -----------------: | --- | ----- |
| \(E_{\mathrm{KS}}\) (hybrid only) | −75.198832360592 | 1e-6 Ha | `rks.energy()` after `mp2` SCF |
| \(E_{\mathrm{MP2}}\) (unscaled) | −0.044726379786 | 1e-6 Ha | `e_corr_os + e_corr_ss` |
| \(a_c\cdot E_{\mathrm{MP2}}\) | −0.012076122542 | 1e-6 Ha | `dh.e_corr` |
| **\(E_{\mathrm{B2PLYP}}\) total** | **−75.210908483134** | **1e-6 Ha** | `dh.e_tot` |

Regression also asserts \(E_{\mathrm{tot}}=E_{\mathrm{KS}}+E_{\mathrm{corr}}\), \(E_{\mathrm{corr}}=a_c\cdot(E_{\mathrm{OS}}+E_{\mathrm{SS}})\), `!dh.ri`, `dh.frozen==0`, and SCF convergence.

Pin command (offline only):

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py b2plyp \
  --geom water --basis sto-3g --conv-tol 1e-12
```

## Performance

Source: fair wall in `regressions/scf/b2plyp-h2o/REFERENCE.md` (worktree 2026-07-23); worker/handoff ratio **1.75**. No standing row in at handbook write time.

Protocol: `tools/pyscf_oracle/fair_bench.py`, median of ≥3 runs, `OMP_NUM_THREADS=1`.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r=T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | --------------------------------------: | ---- | ------ |
| `b2plyp` H₂O/STO-3G | 0.291 | 0.509 | **1.75** | ✓ | composed hybrid KS + scaled MP2 both sides |

```bash
export OMP_NUM_THREADS=1 OMP_PROC_BIND=close
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method b2plyp --geom water --basis sto-3g --repeat 3 --max-ratio 10 \
  -- ./build/regressions/reg_b2plyp-h2o
```

**Caveats.** (1) Twin is the full double-hybrid class (hybrid SCF + MP2), not KS-only. (2) Tiny STO-3G foothold — wall ratio does not forecast large-basis \(O(N^5)\) MP2 scaling. (3) Hosted libxc may lack named B2PLYP; fair_bench/pin compose the same hybrid string as Atomiverse. Gate \(r\le 10\) passed; no optimize pass required.

## Limitations

- Energy-only foothold: no B2PLYP nuclear forces, Hessian, or response (TDA/TDDFT) product gate.
- Restricted closed-shell only; open-shell double hybrids not shipped here.
- No RI-B2PLYP, frozen-core B2PLYP, or spin-component-scaled double hybrids (SCS-DH).
- Sibling functionals (B2GP-PLYP, DSD-*, PBE0-DH, ωB97X-2) remain roadmap ○ (`docs/roadmap/dft.md` §7).
- CUDA / Kokkos B2PLYP ○.
- Speed: ~1.75× PySCF on this microcase; larger systems will be dominated by the RMP2 step unless RI/CD is added later.
- Oracle path depends on composed XC when libxc omits `B2PLYP` — document when re-pinning.

## References

- Grimme, S. *J. Chem. Phys.* **124**, 034108 (2006), doi:10.1063/1.2148954 (B2PLYP parameters \(a_x=0.53\), \(a_c=0.27\)).
- Becke, A. D. *Phys. Rev. A* **38**, 3098 (1988) (B88); Lee–Yang–Parr *Phys. Rev. B* **37**, 785 (1988) (LYP).
- Offline pin + fair_bench numbers: `regressions/scf/b2plyp-h2o/REFERENCE.md`
- Regression consumer: `regressions/scf/b2plyp-h2o/reg_b2plyp-h2o.cpp`
- Oracle helpers: `tools/pyscf_oracle/pin_scf.py`, `tools/pyscf_oracle/fair_bench.py`; env `tools/setup_pyscf_env.sh`
- Roadmap: `docs/roadmap/dft.md` §7
- Campaign queue/handoff: the case `REFERENCE.md`
- Perf rollup (if populated later): - Public API headers: `include/atomiverse/cpu/scf/rks.h`, `xc_functional.h`, `mean_field_method.h`
