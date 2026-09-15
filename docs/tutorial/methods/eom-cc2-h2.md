# EOM-CC2 — H₂ / STO-3G (singlet S1)

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `eom-cc2-h2` |
| Regression | `regressions/scf/eom-cc2-h2/` |
| Public API | `cpu::scf::RHF::eom_cc2(Frame&, n_states)` → `EOMCC2Result` |
| Roadmap | [Excited states §3 EOM-CC](../../roadmap/excited-states.md) |

## Summary

Closed-shell **EOM-CC2** singlet electronic excitation energies on a converged
RHF → CC2 reference. Product surface is `RHF::eom_cc2(frame, n_states)`: the
call drives RHF (with ERI), spin-orbital Christiansen–Koch–Jørgensen CC2
amplitudes, then a dense non-symmetric eigen-solve of the EOM-EE Jacobian
**evaluated at those CC2 \(T_1/T_2\)** (same right-hand \(\sigma\) as
EOM-EE-CCSD / PySCF `eom_gccsd`), returning singlet roots \(\omega\) [eV] plus
CC2 energetics [Ha]. Foothold case is H₂ / STO-3G. Offline pin is PySCF 2.14.0
`cc.rccsd.RCCSD` with `cc2=True` then `eomee_ccsd_singlet` (stock PySCF has no
top-level `EOMCC2`; never live in ctest). End-to-end gate:
`regressions/scf/eom-cc2-h2` → binary `reg_eom-cc2-h2`. Fair-bench ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 0.300\) (within 10×;
startup-dominated).

## Method

**Ground-state CC2.** Cluster operator \(T = T_1 + T_2\) on a closed-shell RHF
determinant \(\lvert\Phi_0\rangle\). CC2 keeps the full singles residual and
truncates the doubles residual to at most linear in \(T_2\) with a
\(T_1\)-dressed Fock (Christiansen, Koch & Jørgensen, CPL **243**, 409 (1995);
see also handbook [cc2-h2-sto3g](cc2-h2-sto3g.md)).

\[
\lvert\Psi_{\mathrm{CC2}}\rangle = e^{T}\lvert\Phi_0\rangle,\qquad
\bar{H} = e^{-T} H e^{T}
\quad\text{(with CC2 amplitudes)}.
\]

**EOM-EE at CC2 amplitudes.** Excited states as right eigenstates of the
similarity-transformed Hamiltonian in the singles–doubles space — the same
Jacobian structure as EOM-EE-CCSD, but \(T\) is the CC2 solution:

\[
\bar{H} R_k = \omega_k R_k,\qquad
R_k = R_1 + R_2
\quad\text{(spin-orbital singles + doubles)}.
\]

Right-hand \(\sigma\) equations match Wang, Tu & Wang, *JCTC* **10**, 5567
(2014) Eqs. (9)–(10) as in PySCF `eom_gccsd.eeccsd_matvec`. Packed vector
basis: \(r_1\) plus \(\mathrm{tril}(i>j)\times\mathrm{tril}(a>b)\) \(r_2\).
Roots from dense non-symmetric `dgeev` on the Jacobian; **singlet** selection
by \(\alpha\leftrightarrow\beta\) symmetry of the \(R_1\) block on a
closed-shell reference. \(\omega\) reported in eV via
\(E_{\mathrm{h}}\rightarrow\mathrm{eV}\) conversion in `core/units.h`.

**Spin treatment.** Spin-orbital EE Jacobian on a closed-shell RHF→CC2
reference only. Product path returns singlet roots (PySCF twin:
`RCCSD(cc2=True).eomee_ccsd_singlet`). No open-shell / UCC2 reference.

**Key approximations / scope of this foothold:**

- Single-reference closed-shell EE only — no IP/EA/SF/DIP/DEA channels on this
  path.
- Energies only — no left vectors, transition moments, \(\Lambda\), densities,
  or analytic EOM gradients.
- Dense full Jacobian diagonalization (no Davidson / iterative \(\sigma\) on
  large systems in this foothold).
- Canonical MO integrals; no DF/CD/F12/local CC / CVS / STEOM.
- H₂/STO-3G: 2 AO; 1 occ + 1 virt spatial MO. Full EOM-EE-CCSD on the same
  geometry gives S1 = 26.3659666 eV (REFERENCE cross-check); CC2 amplitudes
  shift the root by ~0.20 eV (S1 ≈ 26.165 eV).

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

const cpu::scf::EOMCC2Result eom = rhf.eom_cc2(frame, /*n_states=*/1);
// eom.energy_ev[0] [eV]; eom.e_cc2_ha, eom.e_corr_ha [Ha];
// eom.cc2_iterations; eom.converged
```

`EOMCC2Result` (`include/atomiverse/cpu/scf/eom_cc2.h`): `energy_ev`
(ascending singlets [eV]), `e_corr_ha`, `e_cc2_ha` [Hartree],
`cc2_iterations`, `converged`. Pack-level
`eom_cc2_energies` is private engine code.
(`internal/eom_cc2_pack.h`). `RHF::eom_cc2` forces ERI, runs SCF + CC2, then
refuses a non-converged reference (no silent fallback). Implementation reuses
the EOM-EE Jacobian path in `src/cpu/scf/eom_ee_ccsd.cpp` with CC2 amplitudes
from `ccsd_solve(..., cc2=true)`.

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
  cmake --build build --target reg_eom-cc2-h2 -j$(nproc)
  ./build/regressions/reg_eom-cc2-h2
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/eom_cc2.h` | Public `EOMCC2Result` |
| `include/atomiverse/cpu/scf/rhf.h` | `RHF::eom_cc2(Frame&, n_states)` |
| `include/atomiverse/cpu/scf/internal/eom_cc2_pack.h` | Private pack entry |
| `src/cpu/scf/eom_ee_ccsd.cpp` | CC2 amplitudes + EE Jacobian / singlet select |
| `regressions/scf/eom-cc2-h2/reg_eom-cc2-h2.cpp` | E2E consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/eom-cc2-h2/REFERENCE.md`](../../../regressions/scf/eom-cc2-h2/REFERENCE.md)
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
| EOM | `mycc.eomee_ccsd_singlet(nroots=1)` |
| AO / MO | 2 AO; 1 occ, 1 virt spatial |
| PySCF | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{CC2}}\) | **−1.129897380985958 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(E_{\mathrm{corr}}\) (CC2) | −0.013138073589533 Ha | \(10^{-6}\) Ha | REFERENCE |
| \(\omega_{\mathrm{S1}}\) (singlet EOM-CC2) | **26.164970973311735 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_{\mathrm{S1}}\) (Ha cross-check) | 0.961544948014894 Ha | — | REFERENCE only |
| Convergence | — | `converged == true` | Product gate |

Cross-check (same geom/basis): full EOM-EE-CCSD S1 = 26.3659666 eV (distinct;
CC2 amplitudes shift the root by ~0.20 eV).

Oracle repro (human / offline only):

```bash
bash tools/setup_pyscf_env.sh
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py eom-cc2 \
    --geom h2 --basis sto-3g --nstates 1 --conv-tol 1e-12
```

## Performance

Source: fair wall in
[`regressions/scf/eom-cc2-h2/REFERENCE.md`](../../../regressions/scf/eom-cc2-h2/REFERENCE.md)
(2026-07-23; login node; `OMP_NUM_THREADS=1`; median of ≥3). Campaign rollup
ratio **0.300** (the case `REFERENCE.md`). No standing comparison table in-tree at draft time.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `eom-cc2` (H₂/STO-3G) | 0.0779 | 0.0234 | **0.300** | ✓ | Startup-dominated foothold; ATV faster than PySCF (\(r \ll 1\)); not a production-scale EOM stress test |

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method eom-cc2 --geom h2 --basis sto-3g --nstates 1 --repeat 3 -- \
    ./build/regressions/reg_eom-cc2-h2
```

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). No ⚡ optimization pass
required on this case.

## Limitations

- **Singlet EE only** on closed-shell RHF+CC2; IP/EA/SF/DIP/DEA, CVS-EOM,
  STEOM, EOM-MP2 — roadmap rows (`docs/roadmap/excited-states.md` §3).
- **Energies only** — no left eigenvectors, oscillator strengths, densities, or
  analytic EOM-CC2 gradient/Hessian.
- **Dense Jacobian** path; no large-system iterative Davidson product path yet.
- **No DF/CD/F12/local** EOM; full MO ground-state CC2 + dense EE space.
- **CUDA / Kokkos:** ○ (CPU only).
- **Distinct from EOM-EE-CCSD:** same EE Jacobian, different \(T\); on this
  H₂ foothold the S1 shift vs full CCSD is ~0.20 eV (not FCI-equivalent).
- **Perf twin** is the same method family on a minimal system; wall times are
  dominated by process startup and tiny MO spaces, so \(r\) is not a reliable
  large-\(N\) predictor.

## References

- Offline pin: `regressions/scf/eom-cc2-h2/REFERENCE.md`
- Regression consumer: `regressions/scf/eom-cc2-h2/reg_eom-cc2-h2.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Roadmap: `docs/roadmap/excited-states.md` (§3 Equation-of-Motion Coupled Cluster);
  ground-state CC2: `docs/roadmap/post-hf.md` (§3), handbook
  [cc2-h2-sto3g](cc2-h2-sto3g.md)
- Related handbook: [eom-ee-ccsd](eom-ee-ccsd.md)
- Public API: `include/atomiverse/cpu/scf/eom_cc2.h`,
  `include/atomiverse/cpu/scf/rhf.h`
- O. Christiansen, H. Koch, P. Jørgensen, *The second-order approximate
  coupled cluster singles and doubles model CC2*, *Chem. Phys. Lett.*
  **243**, 409 (1995). DOI
  [10.1016/0009-2614(95)00841-Q](https://doi.org/10.1016/0009-2614(95)00841-Q)
- J. F. Stanton & R. J. Bartlett, *J. Chem. Phys.* **98**, 7029 (1993).
- Z. Wang, Z. Tu & F. Wang, *J. Chem. Theory Comput.* **10**, 5567 (2014)
  (EE \(\sigma\) equations as in PySCF `eom_gccsd`)
