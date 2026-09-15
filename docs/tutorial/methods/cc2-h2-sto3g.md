# CC2 — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `cc2-h2-sto3g` |
| Regression | `regressions/scf/cc2-h2-sto3g/` |
| Public API | `cpu::scf::RHF::cc2(Frame&)` → `CC2Result` |
| Roadmap | [Post-HF §3 Coupled cluster](../../roadmap/post-hf.md) |

## Summary

Closed-shell **CC2** (Christiansen–Koch–Jørgensen second-order approximate
coupled-cluster singles and doubles) correlation energy on H₂ / STO-3G after a
converged RHF reference. Product surface is `RHF::cc2(frame)`: the call drives
a fresh RHF (if needed), then spin-orbital CC2 amplitude iteration, and returns
`CC2Result{e_corr, e_tot, iterations, converged}` in Hartree. Full singles
residual; doubles residual truncated to at most linear in \(T_2\) with the
\(T_1\)-dressed Fock (no quadratic \(T_2\) ring/ladder). On this minimal H₂
basis \(T_1 \approx 0\) by symmetry, so CC2 coincides with MP2 to
\(\sim 10^{-15}\) Ha and is intentionally distinct from full CCSD/FCI. Offline
oracle pin is PySCF 2.14.0 `cc.rccsd.RCCSD` with `cc2=True` (stock PySCF has no
top-level `cc.CC2`). End-to-end gate: `regressions/scf/cc2-h2-sto3g` → binary
`reg_cc2-h2-sto3g`. Fair-bench ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 0.392\) (within 10×;
startup-dominated).

## Method

**Ansatz.** Approximate CC on a closed-shell RHF Slater determinant
\(\lvert\Phi_0\rangle\). Cluster operator \(T = T_1 + T_2\); energy as the
connected projection of the similarity-transformed Hamiltonian (same form as
CCSD for the energy expression once amplitudes are known):

\[
E_{\mathrm{corr}} = E_{\mathrm{CC2}} - E_{\mathrm{RHF}},\qquad
E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}.
\]

**CC2 truncation** (Christiansen, Koch & Jørgensen, CPL **243**, 409 (1995)):

- **Singles residual:** full CCSD singles equations (all terms retained).
- **Doubles residual:** at most **linear** in \(T_2\), with the Fock operator
  \(T_1\)-dressed; **no** quadratic \(T_2\) ring / ladder contributions that
  appear in full CCSD.
- Iteration: Jacobi micro-steps on \(t_1,t_2\) until max\(|\Delta t|\) falls
  below the internal tolerance; product gate also requires `converged == true`.

**Spin treatment.** Spin-orbital residual path shared with the in-tree CCSD
kernel (`ccsd_solve(..., cc2=true)`). Occupied spin-orbitals: α then β over
spatial occupied MOs; virtuals likewise. Matches PySCF `rccsd.RCCSD` with
`cc2=True` on the same RHF orbitals.

**Key approximations / scope of this foothold:**

- Single-reference, closed-shell RHF only (open-shell throws — no silent
  fallback).
- Energy only — no \(\Lambda\) equations, no 1-RDM/2-RDM, no analytic gradient.
- No EOM-CC2 roots on this path (queue follow-on `eom-cc2-h2`).
- Canonical MO integrals; no DF/CD/F12/local CC.
- H₂/STO-3G: 2 AO, 1 occupied + 1 virtual spatial MO; \(T_1 \approx 0\); CC2
  \(E_{\mathrm{corr}}\) equals RMP2 on the pin geometry/basis; full CCSD /
  FCI \(E_{\mathrm{corr}} = -0.020524527\) Ha is intentionally larger in
  magnitude.

## Implementation

Public headers only for consumers (`Frame` + concrete `RHF`; no `*Config`):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

cpu::Frame frame(2);
// H(0,0,0), H(0,0,0.74) Å; Z=1; Box::cubic(20)
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.max_iter = 100;

const cpu::scf::CC2Result cc2 = rhf.cc2(frame);
// cc2.e_corr, cc2.e_tot [Ha]; cc2.converged; cc2.iterations
// rhf.energy() = E_RHF [Ha] after the internal RHF
```

`CC2Result` fields (`include/atomiverse/cpu/scf/cc2.h`): `e_corr`, `e_tot`
[Hartree], `iterations`, `converged`. `cc2_energy` is private.
is private (`internal/ccsd_pack.h`); the same `ccsd_solve` path with
`cc2=true` lives in `src/cpu/scf/ccsd.cpp`.

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
  cmake --build build --target reg_cc2-h2-sto3g -j$(nproc)
  ./build/regressions/reg_cc2-h2-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/cc2.h` | Public `CC2Result` |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `RHF::cc2(Frame&)` |
| `include/atomiverse/cpu/scf/internal/ccsd_pack.h` | Private `cc2_energy` |
| `src/cpu/scf/ccsd.cpp` | Shared residual / energy (`cc2` flag) |
| `regressions/scf/cc2-h2-sto3g/reg_cc2-h2-sto3g.cpp` | E2E consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/cc2-h2-sto3g/REFERENCE.md`](../../../regressions/scf/cc2-h2-sto3g/REFERENCE.md)
(offline pin; PySCF never in ctest). Protocol:
`docs/tutorial/appendix-pyscf-oracle.md`.

| Setting | Value |
| ------- | ----- |
| Molecule | H₂, charge 0, singlet |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — `pin_scf` `h2` preset |
| Basis | STO-3G |
| Angular | `cart=True` (s only; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CC2 | `pyscf.cc.rccsd.RCCSD(mf); my.cc2=True`, `conv_tol=1e-12` |
| AO / MO | 2 AO; 1 occ, 1 virt spatial |
| PySCF | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CC2) | **−0.013138073589533 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(E_{\mathrm{tot}}\) (CC2) | **−1.129897380985958 Ha** | **\(10^{-6}\) Ha** | Product gate |
| Amplitudes | — | `converged == true` | Product gate |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Regression check |

Cross-check (same geom/basis): PySCF `mp.MP2` gives the same \(E_{\mathrm{corr}}\)
(T₁ vanishes; CC2 reduces to MP2). Full CCSD \(E_{\mathrm{corr}} = -0.020524527\) Ha
(FCI) is intentionally different.

Oracle repro (human / offline only):

```bash
bash tools/setup_pyscf_env.sh
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py cc2 \
    --geom h2 --basis sto-3g --conv-tol 1e-12
```

## Performance

Source: fair wall in
[`regressions/scf/cc2-h2-sto3g/REFERENCE.md`](../../../regressions/scf/cc2-h2-sto3g/REFERENCE.md)
(2026-07-23; login node; `OMP_NUM_THREADS=1`; median of ≥3). Campaign rollup
ratio **0.392** (the case `REFERENCE.md`). No standing
`` table in-tree at draft time.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `cc2` (H₂/STO-3G) | 0.0350 | 0.0137 | **0.392** | ✓ | Startup-dominated foothold; ATV faster than PySCF (\(r \ll 1\)); not a production-scale CC2 stress test |

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method cc2 --geom h2 --basis sto-3g --repeat 3 -- \
    ./build/regressions/reg_cc2-h2-sto3g
```

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). No ⚡ optimization pass
required on this case.

## Limitations

- **Energy only** on closed-shell RHF; no UCC2, no λ / density, no analytic
  gradient or Hessian.
- **No EOM-CC2** on this product path (follow-on queue row `eom-cc2-h2` once
  roots are implemented).
- **No DF/CD/F12/local** CC; full MO path only.
- **CUDA / Kokkos:** ○ (CPU only).
- On H₂/STO-3G, CC2 is **not** a numerical stress of the approximate-CC
  doubles truncation relative to MP2 (\(T_1 \approx 0\)); use multi-electron /
  multi-MO systems to exercise the CC2 residual difference.
- **Roadmap next axes:** EOM-CC2; open-shell / higher CC / local / DF still ○ —
  see `docs/roadmap/post-hf.md`, `docs/roadmap/excited-states.md`.
- **Perf twin** is the same method family on a minimal system; wall times are
  dominated by process startup and tiny MO spaces, so \(r\) is not a reliable
  large-\(N\) predictor.

## References

- Offline pin: `regressions/scf/cc2-h2-sto3g/REFERENCE.md`
- Regression consumer: `regressions/scf/cc2-h2-sto3g/reg_cc2-h2-sto3g.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Roadmap: `docs/roadmap/post-hf.md` (§3 Coupled cluster)
- Public API: `include/atomiverse/cpu/scf/cc2.h`
- Related handbook: [ccsd-h2-sto3g](ccsd-h2-sto3g.md),
  [mp3-h2-sto3g](mp3-h2-sto3g.md)
- O. Christiansen, H. Koch, P. Jørgensen, *The second-order approximate
  coupled cluster singles and doubles model CC2*, *Chem. Phys. Lett.*
  **243**, 409 (1995). DOI
  [10.1016/0009-2614(95)00841-Q](https://doi.org/10.1016/0009-2614(95)00841-Q)
