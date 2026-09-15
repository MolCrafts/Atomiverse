# EOM-EE-CCSD — H₂ / STO-3G (singlet S1)

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `eom-ee-ccsd` |
| Regression | `regressions/scf/eom-ee-ccsd/` |
| Public API | `cpu::scf::RHF::eom_ee_ccsd(Frame&, n_states)` → `EOMEECCSDResult` |
| Roadmap | [Excited states §3 EOM-CC](../../roadmap/excited-states.md) |

## Summary

Closed-shell **EOM-EE-CCSD** singlet electronic excitation energies on a
converged RHF → CCSD reference. Product surface is
`RHF::eom_ee_ccsd(frame, n_states)`: the call drives RHF (with ERI), spin-orbital
Gauss–Stanton CCSD, then a dense non-symmetric eigen-solve of the EOM-EE
Jacobian, returning singlet roots \(\omega\) [eV] plus CCSD energetics [Ha].
Foothold case is H₂ / STO-3G (2e: EOM-EE-CCSD coincides with FCI in the basis).
Offline pin is PySCF 2.14.0 `cc.CCSD(...).eomee_ccsd_singlet` (never live in
ctest). End-to-end gate: `regressions/scf/eom-ee-ccsd` → binary
`reg_eom-ee-ccsd`. Fair-bench ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 0.193\) (within 10×; startup-dominated).

## Method

**Ground-state CCSD.** Cluster operator \(T = T_1 + T_2\) on a closed-shell RHF
determinant \(\lvert\Phi_0\rangle\):

\[
\lvert\Psi_{\mathrm{CCSD}}\rangle = e^{T}\lvert\Phi_0\rangle,\qquad
\bar{H} = e^{-T} H e^{T}.
\]

**EOM-EE.** Excited states as right eigenstates of the similarity-transformed
Hamiltonian in the singles–doubles space:

\[
\bar{H} R_k = \omega_k R_k,\qquad
R_k = R_1 + R_2
\quad\text{(spin-orbital singles + doubles)}.
\]

Intermediates follow Gauss–Stanton / PySCF `gintermediates` (EOM section);
right-hand \(\sigma\) equations match Wang, Tu & Wang, *JCTC* **10**, 5567
(2014) Eqs. (9)–(10) as in PySCF `eom_gccsd.eeccsd_matvec`. Packed vector basis:
\(r_1\) plus \(\mathrm{tril}(i>j)\times\mathrm{tril}(a>b)\) \(r_2\). Roots from
dense non-symmetric `dgeev` on the Jacobian; **singlet** selection by \(\alpha
\leftrightarrow \beta\) symmetry of the \(R_1\) block on a closed-shell
reference. \(\omega\) reported in eV via \(E_{\mathrm{h}}\rightarrow\mathrm{eV}\)
conversion in `core/units.h`.

**Spin treatment.** Spin-orbital EE Jacobian on a closed-shell RHF→CCSD
reference only. Product path returns singlet roots (PySCF twin:
`eomee_ccsd_singlet`). No open-shell / UCCSD reference.

**Key approximations / scope of this foothold:**

- Single-reference closed-shell EE only — no IP/EA/SF/DIP/DEA channels.
- Energies only — no left vectors, transition moments, \(\Lambda\), densities,
  or analytic EOM gradients.
- Dense full Jacobian diagonalization (no Davidson / iterative \(\sigma\) on
  large systems in this foothold).
- Canonical MO integrals; no DF/CD/F12/local CC / CVS / STEOM.
- H₂/STO-3G: 2 AO; 1 occ + 1 virt spatial MO; EOM-EE-CCSD = FCI excitation
  ladder in the basis (singlet S1 ≈ 26.366 eV).

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

const cpu::scf::EOMEECCSDResult eom = rhf.eom_ee_ccsd(frame, /*n_states=*/1);
// eom.energy_ev[0] [eV]; eom.e_ccsd_ha, eom.e_corr_ha [Ha];
// eom.ccsd_iterations; eom.converged
```

`EOMEECCSDResult` (`include/atomiverse/cpu/scf/eom_ee_ccsd.h`): `energy_ev`
(ascending singlets [eV]), `e_corr_ha`, `e_ccsd_ha` [Hartree],
`ccsd_iterations`, `converged`. Pack-level
`eom_ee_ccsd_energies` is private engine code.
(`internal/eom_ee_ccsd_pack.h`). `RHF::eom_ee_ccsd` forces `compute_eri`, runs
SCF, then refuses a non-converged reference (no silent fallback).

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
  cmake --build build --target reg_eom-ee-ccsd -j$(nproc)
  ./build/regressions/reg_eom-ee-ccsd
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/eom_ee_ccsd.h` | Public `EOMEECCSDResult` |
| `include/atomiverse/cpu/scf/rhf.h` | `RHF::eom_ee_ccsd(Frame&, n_states)` |
| `include/atomiverse/cpu/scf/internal/eom_ee_ccsd_pack.h` | Private pack entry |
| `src/cpu/scf/eom_ee_ccsd.cpp` | Intermediates, Jacobian, singlet select |
| `regressions/scf/eom-ee-ccsd/reg_eom-ee-ccsd.cpp` | E2E consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/eom-ee-ccsd/REFERENCE.md`](../../../regressions/scf/eom-ee-ccsd/REFERENCE.md)
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
| EOM | `mycc.eomee_ccsd_singlet(nroots=1)` |
| AO / MO | 2 AO; 1 occ, 1 virt spatial |
| PySCF | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{CCSD}}\) | **−1.137283834497130 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(\omega_{\mathrm{S1}}\) (singlet EOM-EE) | **26.365966612792256 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_{\mathrm{S1}}\) (Ha cross-check) | 0.968931401526066 Ha | — | REFERENCE only |
| Convergence | — | `converged == true` | Product gate |

FCI excitation ladder on this basis (eV, REFERENCE): 16.504 (triplet),
**26.366 (singlet)**, 44.094 — EOM-EE-CCSD matches FCI for 2e.

Oracle repro (human / offline only):

```bash
bash tools/setup_pyscf_env.sh
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py eom-ee-ccsd \
    --geom h2 --basis sto-3g --nstates 1 --conv-tol 1e-12
```

## Performance

Source: fair wall in
[`regressions/scf/eom-ee-ccsd/REFERENCE.md`](../../../regressions/scf/eom-ee-ccsd/REFERENCE.md)
(2026-07-23; login node; `OMP_NUM_THREADS=1`; median of ≥3). Campaign rollup
ratio **0.193** (the case `REFERENCE.md`). No standing comparison table in-tree at draft time.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `eom-ee-ccsd` (H₂/STO-3G) | 0.0725 | 0.0140 | **0.193** | ✓ | Startup-dominated foothold; ATV faster than PySCF (\(r \ll 1\)); not a production-scale EOM stress test |

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method eom-ee-ccsd --geom h2 --basis sto-3g --nstates 1 --repeat 3 -- \
    ./build/regressions/reg_eom-ee-ccsd
```

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). No ⚡ optimization pass
required on this case.

## Limitations

- **Singlet EE only** on closed-shell RHF+CCSD; IP/EA/SF/DIP/DEA, CVS-EOM,
  STEOM, EOM-CC2/MP2 — roadmap ○ (`docs/roadmap/excited-states.md` §3).
- **Energies only** — no left eigenvectors, oscillator strengths, densities, or
  analytic EOM-CCSD gradient/Hessian.
- **Dense Jacobian** path; no large-system iterative Davidson product path yet.
- **No DF/CD/F12/local** EOM; full MO \(O(N^6)\) ground state + dense EE space.
- **CUDA / Kokkos:** ○ (CPU only).
- **Perf twin** is the same method family on a minimal system; wall times are
  dominated by process startup and tiny MO spaces, so \(r\) is not a reliable
  large-\(N\) predictor.

## References

- Offline pin: `regressions/scf/eom-ee-ccsd/REFERENCE.md`
- Regression consumer: `regressions/scf/eom-ee-ccsd/reg_eom-ee-ccsd.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Roadmap: `docs/roadmap/excited-states.md` (§3 Equation-of-Motion Coupled Cluster);
  ground-state CC: `docs/roadmap/post-hf.md` (§3)
- Public API: `include/atomiverse/cpu/scf/eom_ee_ccsd.h`,
  `include/atomiverse/cpu/scf/rhf.h`
- J. F. Stanton & R. J. Bartlett, *J. Chem. Phys.* **98**, 7029 (1993).
  DOI [10.1063/1.464746](https://doi.org/10.1063/1.464746)
- Z. Wang, Z. Tu & F. Wang, *J. Chem. Theory Comput.* **10**, 5567 (2014)
  (EE \(\sigma\) equations)
- J. Gauss & J. F. Stanton, *J. Chem. Phys.* **103**, 3561 (1995) (Table III
  intermediates; EOM section)
