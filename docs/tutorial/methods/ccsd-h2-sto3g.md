# CCSD — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `ccsd-h2-sto3g` |
| Regression | `regressions/scf/ccsd-h2-sto3g/` |
| Public API | `cpu::scf::RHF::ccsd(Frame&)` → `CCSDResult` |
| Roadmap | [Post-HF §3 Coupled cluster](../../roadmap/post-hf.md) |

## Summary

Closed-shell **CCSD** correlation energy on H₂ / STO-3G after a converged RHF
reference. Product surface is `RHF::ccsd(frame)`: the call drives a fresh RHF
(if needed), then spin-orbital Gauss–Stanton amplitude iteration, and returns
`CCSDResult{e_corr, e_tot, iterations, converged}` in Hartree. On this two-electron
system CCSD coincides with FCI in the basis. Offline oracle pin is PySCF 2.14.0
`cc.CCSD` (never live in ctest). End-to-end gate:
`regressions/scf/ccsd-h2-sto3g` → binary `reg_ccsd-h2-sto3g`. Fair-bench ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 0.317\) (within 10×; startup-dominated).

## Method

**Ansatz.** Cluster operator \(T = T_1 + T_2\) on a closed-shell RHF Slater
determinant \(\lvert\Phi_0\rangle\):

\[
\lvert\Psi_{\mathrm{CCSD}}\rangle = e^{T}\lvert\Phi_0\rangle,\qquad
E_{\mathrm{CCSD}} = \langle\Phi_0\rvert H e^{T}\lvert\Phi_0\rangle_{\mathrm{c}},
\quad
E_{\mathrm{corr}} = E_{\mathrm{CCSD}} - E_{\mathrm{RHF}}.
\]

**Spin treatment.** Spin-orbital Gauss–Stanton intermediates (Stanton et al.
1991; Gauss & Stanton 1995 Table III). Occupied spin-orbitals: α then β over
spatial occupied MOs; virtuals likewise. Matches PySCF `cc.CCSD` on the same
RHF orbitals.

**Key approximations / scope of this foothold:**

- Single-reference, closed-shell RHF only (no UCCSD / ROCCSD).
- Energy only — no \(\Lambda\) equations, no 1-RDM/2-RDM, no analytic gradient.
- No higher \(T_n\) here; perturbative (T) is a separate product path
  (`RHF::ccsd_t` / `ccsd-t-h2o-sto3g`).
- Canonical MO integrals; no DF/CD/F12/local CC.
- Jacobi micro-iteration on \(t_1,t_2\) until max\(|\Delta t|\) below internal tol;
  product gate also requires `converged == true`.
- H₂/STO-3G: 2 AO, 1 occupied + 1 virtual spatial MO; \(T_1 \approx 0\) by
  symmetry; CCSD = FCI to \(\sim 10^{-11}\) Ha in the basis.

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

const cpu::scf::CCSDResult ccsd = rhf.ccsd(frame);
// ccsd.e_corr, ccsd.e_tot [Ha]; ccsd.converged; ccsd.iterations
// rhf.energy() = E_RHF [Ha] after the internal RHF
```

`CCSDResult` fields (`include/atomiverse/cpu/scf/ccsd.h`): `e_corr`, `e_tot`
[Hartree], `iterations`, `converged`. `ccsd_energy` is private.
is private (`internal/ccsd_pack.h`).

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
  cmake --build build --target reg_ccsd-h2-sto3g -j$(nproc)
  ./build/regressions/reg_ccsd-h2-sto3g
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/ccsd.h` | Public `CCSDResult` |
| `include/atomiverse/cpu/scf/mean_field_method.h` | `RHF::ccsd(Frame&)` |
| `include/atomiverse/cpu/scf/internal/ccsd_pack.h` | Private pack entry |
| `src/cpu/scf/ccsd.cpp` | Gauss–Stanton residual / energy |
| `regressions/scf/ccsd-h2-sto3g/reg_ccsd-h2-sto3g.cpp` | E2E consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/ccsd-h2-sto3g/REFERENCE.md`](../../../regressions/scf/ccsd-h2-sto3g/REFERENCE.md)
(offline pin; PySCF never in ctest). Protocol:
`docs/tutorial/appendix-pyscf-oracle.md`.

| Setting | Value |
| ------- | ----- |
| Molecule | H₂, charge 0, singlet |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — `pin_scf` `h2` preset |
| Basis | STO-3G |
| Angular | `cart=True` (s only; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CCSD | `pyscf.cc.CCSD(mf)`, `conv_tol=1e-12` |
| AO / MO | 2 AO; 1 occ, 1 virt spatial |
| PySCF | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396426 Ha | \(10^{-6}\) Ha | Diagnostic |
| \(E_{\mathrm{corr}}\) (CCSD) | **−0.020524527092106 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(E_{\mathrm{tot}}\) (CCSD) | **−1.137283834488532 Ha** | **\(10^{-6}\) Ha** | Product gate |
| Amplitudes | — | `converged == true` | Product gate |
| Consistency | \(E_{\mathrm{tot}} = E_{\mathrm{RHF}} + E_{\mathrm{corr}}\) | \(10^{-12}\) Ha | Regression check |

Oracle repro (human / offline only):

```bash
bash tools/setup_pyscf_env.sh
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py ccsd \
    --geom h2 --basis sto-3g --conv-tol 1e-12
```

## Performance

Source: fair wall in
[`regressions/scf/ccsd-h2-sto3g/REFERENCE.md`](../../../regressions/scf/ccsd-h2-sto3g/REFERENCE.md)
(2026-07-23; login node; `OMP_NUM_THREADS=1`; median of ≥3). Campaign rollup
ratio **0.317** (the case `REFERENCE.md`). No standing
`` table in-tree at draft time.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `ccsd` (H₂/STO-3G) | 0.0482 | 0.0153 | **0.317** | ✓ | Startup-dominated foothold; ATV faster than PySCF (\(r \ll 1\)); not a production-scale CCSD stress test |

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method ccsd --geom h2 --basis sto-3g --repeat 3 -- \
    ./build/regressions/reg_ccsd-h2-sto3g
```

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). No ⚡ optimization pass
required on this case.

## Limitations

- **Energy only** on closed-shell RHF; no UCCSD, no (T), no λ / density, no
  analytic gradient or Hessian.
- **No DF/CD/F12/local** CC; full MO \(O(N^6)\) path only.
- **CUDA / Kokkos:** ○ (CPU only).
- **Roadmap next axes** (queue / post-hf): `ccsd-t-h2o-sto3g` (perturbative
  triples — **shipped**); `eom-ee-ccsd` (EOM-EE on this energy path); higher CC / local /
  open-shell still ○ — see `docs/roadmap/post-hf.md`,
  `docs/roadmap/excited-states.md`.
- **Perf twin** is the same method family on a minimal system; wall times are
  dominated by process startup and tiny MO spaces, so \(r\) is not a reliable
  large-\(N\) predictor.

## References

- Offline pin: `regressions/scf/ccsd-h2-sto3g/REFERENCE.md`
- Regression consumer: `regressions/scf/ccsd-h2-sto3g/reg_ccsd-h2-sto3g.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Roadmap: `docs/roadmap/post-hf.md` (§3 Coupled cluster)
- Public API: `include/atomiverse/cpu/scf/ccsd.h`
- J. F. Stanton, J. Gauss, J. D. Watts, R. J. Bartlett, *J. Chem. Phys.*
  **94**, 4334 (1991). DOI [10.1063/1.460620](https://doi.org/10.1063/1.460620)
- J. Gauss, J. F. Stanton, *J. Chem. Phys.* **103**, 3561 (1995) (Table III
  intermediates)
