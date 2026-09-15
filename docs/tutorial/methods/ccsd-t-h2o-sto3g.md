# CCSD(T) — H₂O / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ccsd-t-h2o-sto3g` |
| Regression | `regressions/scf/ccsd-t-h2o-sto3g/` |
| Public API | `cpu::scf::RHF::ccsd_t(Frame&)` → `CCSDTResult` |
| Roadmap | [Post-HF §3 Coupled cluster](../../roadmap/post-hf.md) |

## Summary

Closed-shell **CCSD(T)** correlation energy on H₂O / STO-3G after a converged
RHF reference. Product surface is `RHF::ccsd_t(frame)`: drives a fresh RHF,
spin-orbital Gauss–Stanton CCSD amplitudes, then the non-iterative (T)
triples correction, and returns
`CCSDTResult{e_corr, e_tot, e_corr_ccsd, e_t, iterations, converged}` in
Hartree (`e_corr = E_CCSD + E_(T)`). Offline oracle pin is PySCF 2.14.0
`cc.CCSD` + `ccsd_t()` (never live in ctest). End-to-end gate:
`regressions/scf/ccsd-t-h2o-sto3g` → binary `reg_ccsd-t-h2o-sto3g`. Fair-bench
ratio \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 0.333\) (within 10×;
startup-dominated).

## Method

**Ansatz.** Cluster operator \(T = T_1 + T_2\) on a closed-shell RHF Slater
determinant, plus a non-iterative triples correction \(E_{(T)}\)
(Raghavachari 1989; Watts–Gauss–Bartlett 1993):

\[
\lvert\Psi_{\mathrm{CCSD}}\rangle = e^{T}\lvert\Phi_0\rangle,\qquad
E_{\mathrm{corr}}^{\mathrm{CCSD}} = E_{\mathrm{CCSD}} - E_{\mathrm{RHF}},
\]
\[
E_{\mathrm{corr}}^{\mathrm{CCSD(T)}} = E_{\mathrm{corr}}^{\mathrm{CCSD}} + E_{(T)},
\qquad
E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}^{\mathrm{CCSD(T)}}.
\]

**Spin treatment.** Same spin-orbital Gauss–Stanton intermediates as
`RHF::ccsd` (Stanton et al. 1991; Gauss & Stanton 1995 Table III). Occupied
spin-orbitals: α then β over spatial occupied MOs; virtuals likewise. (T)
uses the spin-orbital formula of JCP **98**, 8718 (1993) (PySCF `gccsd_t`
path, energy-matched to restricted `ccsd_t()`).

**Key approximations / scope of this foothold:**

- Single-reference, closed-shell RHF only (no UCCSD(T) / ROCCSD(T)).
- Energy only — no \(\Lambda\) equations, no 1-RDM/2-RDM, no analytic gradient.
- Non-iterative (T) only; no iterative CCSDT / CC3 / CCSDT-n.
- Canonical MO integrals; no DF/CD/F12/local CC.
- Jacobi micro-iteration on \(t_1,t_2\) until max\(|\Delta t|\) below internal
  tol; product gate also requires `converged == true`.
- H₂O/STO-3G: 7 AO; 5 occupied + 2 virtual spatial MO (spin-orbital
  \(n_o=10\), \(n_v=4\)).

## Implementation

Public headers only for consumers (`Frame` + concrete `RHF`; no `*Config`):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

cpu::Frame frame(3);
// O(0,0,0.1173); H(0,±0.7572,−0.4692) Å; Z = 8,1,1; Box::cubic(20)
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.max_iter = 100;

const cpu::scf::CCSDTResult ct = rhf.ccsd_t(frame);
// ct.e_corr = E_CCSD + E_(T); ct.e_corr_ccsd; ct.e_t; ct.e_tot [Ha]
// ct.converged; ct.iterations; rhf.energy() = E_RHF [Ha]
```

`CCSDTResult` fields (`include/atomiverse/cpu/scf/ccsd.h`): `e_corr`, `e_tot`,
`e_corr_ccsd`, `e_t` [Hartree], `iterations`, `converged`. Pack-level
`ccsd_t_energy` is private engine code.

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
  cmake --build build --target reg_ccsd-t-h2o-sto3g -j$(nproc)
  ./build/regressions/reg_ccsd-t-h2o-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/ccsd.h` | Public `CCSDResult` / `CCSDTResult` |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `RHF::ccsd` / `RHF::ccsd_t` |
| `include/atomiverse/cpu/scf/internal/ccsd_pack.h` | Private pack entries |
| `src/cpu/scf/ccsd.cpp` | Gauss–Stanton residual + (T) |
| `regressions/scf/ccsd-t-h2o-sto3g/reg_ccsd-t-h2o-sto3g.cpp` | E2E consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/ccsd-t-h2o-sto3g/REFERENCE.md`](../../../regressions/scf/ccsd-t-h2o-sto3g/REFERENCE.md)
(offline pin; PySCF never in ctest). Protocol:
`docs/tutorial/appendix-pyscf-oracle.md`.

| Setting | Value |
| ------- | ----- |
| Molecule | H₂O, charge 0, singlet |
| Geometry (Å) | O(0,0,0.1173); H(0,±0.7572,−0.4692) — `pin_scf` `water` preset |
| Basis | STO-3G |
| Angular | `cart=True` (s/p only; engine-neutral vs spherical for STO-3G) |
| SCF | RHF, `conv_tol=1e-12` |
| CCSD | `pyscf.cc.CCSD(mf)`, `conv_tol=1e-12` |
| (T) | `mycc.ccsd_t()` |
| AO / MO | 7 AO; 5 occ, 2 virt spatial |
| PySCF | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −74.963023138462930 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CCSD) | −0.049438563031963 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{(T)}\) | −0.000067409682944 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CCSD(T)) | **−0.049505972714907 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(E_{\mathrm{tot}}\) (CCSD(T)) | **−75.012529111177841 Ha** | **\(10^{-6}\) Ha** | Product gate |
| Amplitudes | — | `converged == true` | Product gate |
| Consistency | \(E_{\mathrm{corr}} = E_{\mathrm{CCSD}} + E_{(T)}\) | \(10^{-12}\) Ha | Regression check |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Regression check |

Oracle repro (human / offline only):

```bash
bash tools/setup_pyscf_env.sh
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py ccsd-t \
    --geom water --basis sto-3g --conv-tol 1e-12
```

## Performance

Source: fair wall in
[`regressions/scf/ccsd-t-h2o-sto3g/REFERENCE.md`](../../../regressions/scf/ccsd-t-h2o-sto3g/REFERENCE.md)
(2026-07-23; login node; `OMP_NUM_THREADS=1`; median of ≥3). Campaign rollup
ratio **0.333** (the case `REFERENCE.md`). No standing comparison table in-tree at draft time.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `ccsd_t` (H₂O/STO-3G) | 0.0933 | 0.0311 | **0.333** | ✓ | Startup + small-\(N\) dominated (7 AO; spin-orbital \(n_o=10\), \(n_v=4\)); ATV faster than PySCF (\(r \ll 1\)); not a production-scale CCSD(T) stress test |

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method ccsd_t --geom water --basis sto-3g --repeat 3 -- \
    ./build/regressions/reg_ccsd-t-h2o-sto3g
```

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). No ⚡ optimization pass
required on this case.

## Limitations

- **Energy only** on closed-shell RHF; no UCCSD(T) / ROCCSD(T); no \(\Lambda\) /
  density; no analytic gradient or Hessian.
- **No iterative triples** (CCSDT / CC3 / CCSDT-n); non-iterative (T) only.
- **No DF/CD/F12/local** CC; full MO \(O(N^7)\) (T) path only.
- **CUDA / Kokkos:** ○ (CPU only).
- **Roadmap next axes** (queue / post-hf / excited-states): `eom-ee-ccsd`
  (EOM-EE on the CCSD energy path — **shipped**); higher CC / local /
  open-shell / CCSD(T) gradient still ○ — see `docs/roadmap/post-hf.md`,
  `docs/roadmap/excited-states.md`.
- **Perf twin** is the same method family on a minimal system; wall times are
  dominated by process startup and tiny MO spaces, so \(r\) is not a reliable
  large-\(N\) predictor.

## References

- Offline pin: `regressions/scf/ccsd-t-h2o-sto3g/REFERENCE.md`
- Regression consumer: `regressions/scf/ccsd-t-h2o-sto3g/reg_ccsd-t-h2o-sto3g.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Roadmap: `docs/roadmap/post-hf.md` (§3 Coupled cluster)
- Public API: `include/atomiverse/cpu/scf/ccsd.h`
- Sibling CCSD foothold: `docs/tutorial/methods/ccsd-h2-sto3g.md`
- K. Raghavachari, G. W. Trucks, J. A. Pople, M. Head-Gordon, *Chem. Phys.
  Lett.* **157**, 479 (1989). DOI
  [10.1016/S0009-2614(89)87395-6](https://doi.org/10.1016/S0009-2614(89)87395-6)
- J. D. Watts, J. Gauss, R. J. Bartlett, *J. Chem. Phys.* **98**, 8718 (1993).
  DOI [10.1063/1.464480](https://doi.org/10.1063/1.464480)
- J. F. Stanton, J. Gauss, J. D. Watts, R. J. Bartlett, *J. Chem. Phys.*
  **94**, 4334 (1991). DOI [10.1063/1.460620](https://doi.org/10.1063/1.460620)
- J. Gauss, J. F. Stanton, *J. Chem. Phys.* **103**, 3561 (1995) (Table III
  intermediates)
