# EOM-SF-CCSD — O₂ / STO-3G (spin-flip)

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `eom-sf-ccsd-o2` |
| Regression | `regressions/scf/eom-sf-ccsd-o2/` |
| Public API | `cpu::scf::UHF::eom_sf_ccsd(Frame&, n_states, guess*)` → `EOMSFCCSDResult` |
| Roadmap | [Excited states §3 EOM-CC / SF](../../roadmap/excited-states.md) |

## Summary

Open-shell **EOM-SF-CCSD** spin-flip excitation energies on a high-spin UHF →
UCCSD reference. Product surface is `UHF::eom_sf_ccsd(frame, n_states, guess)`:
the call drives unrestricted SCF (optional density guess), unrestricted
spin-orbital CCSD, then equation-of-motion spin-flip on the
similarity-transformed Hamiltonian, returning majority-channel SF roots
\(\omega\) [eV] plus UCCSD energetics [Ha]. Textbook foothold is triplet O₂ /
STO-3G (\(n_\alpha=9\), \(n_\beta=7\)); ROHF density warm-start avoids UHF
multi-minima. Offline pin is PySCF 2.14.0 `cc.UCCSD(...).eomsf_ccsd` (never live
in ctest). End-to-end gate: `regressions/scf/eom-sf-ccsd-o2` → binary
`reg_eom-sf-ccsd-o2`. Fair-bench ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 3.12\) (within 10×).

## Method

**Reference.** High-spin unrestricted HF, then unrestricted CCSD:

\[
\lvert\Psi_{\mathrm{UCCSD}}\rangle = e^{T}\lvert\Phi_{\mathrm{UHF}}\rangle,\qquad
T = T_1 + T_2,\qquad
\bar{H} = e^{-T} H e^{T}.
\]

**EOM spin-flip.** Excited (or multiplet-partner) states as right eigenstates of
\(\bar{H}\) in the singles–doubles space with \(\Delta M_S = -1\) (majority
\(\alpha\to\beta\) when \(n_\alpha\ge n_\beta\)):

\[
\bar{H} R_k = \omega_k R_k,\qquad
R_k = R_1 + R_2
\quad\text{(spin-orbital singles + doubles; SF-selected)}.
\]

ATV builds a dense non-symmetric spin-orbital Jacobian and selects the
**majority SF subspace** (α-occupied → β-virtual channel weight), then reports
the lowest real roots. PySCF twin: `UCCSD.eomsf_ccsd` / `EOMEESpinFlip`
(Davidson). \(\omega\) reported in eV via \(E_{\mathrm{h}}\rightarrow\mathrm{eV}\)
in `core/units.h`.

**Davidson vs dense ladder (O₂ pin).** PySCF `eomsf_ccsd(nroots=5)` alone can
miss intermediate SF roots (jumps from ~1.59 eV to ~8.81 eV). Requesting
`nroots≥10–15` and taking the lowest five recovers the full low-energy ladder.
ATV’s dense eigen-solve of the SF-selected Jacobian matches that full lowest
ladder; accuracy pins use PySCF `nroots=15` → lowest 5.

**Spin treatment.** Open-shell UHF + UCCSD only. Targets multiplet
reconstruction and SF excitations from high-spin references (e.g. O₂ triplet →
\(M_S=0\) singlets/triplets). No closed-shell EE path here (that is
`RHF::eom_ee_ccsd`).

**Key approximations / scope of this foothold:**

- SF channel only on UHF→UCCSD — no EE/IP/EA/DIP/DEA product paths on this
  surface; no GHF reference.
- Energies only — no left vectors, transition moments, \(\Lambda\), densities,
  or analytic EOM gradients.
- Dense full Jacobian diagonalization (no large-system iterative \(\sigma\)
  product path in this foothold).
- Canonical MO integrals; no DF/CD/F12/local CC / CVS / STEOM.
- O₂/STO-3G: 10 spatial AO/MO; \(n_\alpha=9\), \(n_\beta=7\); near-zero
  \(\omega_0\) is multiplet reconstruction under mild spin contamination.

## Implementation

Public headers only for consumers (`Frame` + concrete `UHF` / `ROHF`; no
`*Config`):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

// O₂, r(O–O)=1.2075 Å, Z=8; Box::cubic(20) — see reg
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

const cpu::scf::EOMSFCCSDResult eom =
    uhf.eom_sf_ccsd(frame, /*n_states=*/5, &guess);
// eom.energy_ev[k] [eV]; eom.e_ccsd_ha, eom.e_corr_ha [Ha];
// eom.ccsd_iterations; eom.converged
```

`EOMSFCCSDResult` (`include/atomiverse/cpu/scf/eom_sf_ccsd.h`): `energy_ev`
(ascending SF roots [eV]), `e_corr_ha`, `e_ccsd_ha` [Hartree],
`ccsd_iterations`, `converged`. `eom_sf_ccsd_energies` is private.
is private (`internal/eom_sf_ccsd_pack.h`). `UHF::eom_sf_ccsd` forces ERI/SCF as
needed and refuses a non-converged reference (no silent fallback).

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
  cmake --build build --target reg_eom-sf-ccsd-o2 -j$(nproc)
  ./build/regressions/reg_eom-sf-ccsd-o2
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/eom_sf_ccsd.h` | Public `EOMSFCCSDResult` |
| `include/atomiverse/cpu/scf/uhf.h` | `UHF::eom_sf_ccsd(Frame&, n_states, guess*)` |
| `include/atomiverse/cpu/scf/internal/eom_sf_ccsd_pack.h` | Private pack entry |
| `src/cpu/scf/eom_ee_ccsd.cpp` | Shared EOM infrastructure; SF selection + pack |
| `regressions/scf/eom-sf-ccsd-o2/reg_eom-sf-ccsd-o2.cpp` | E2E consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/eom-sf-ccsd-o2/REFERENCE.md`](../../../regressions/scf/eom-sf-ccsd-o2/REFERENCE.md)
(offline pin; PySCF never in ctest). Protocol:
`docs/tutorial/appendix-pyscf-oracle.md`.

| Setting | Value |
| ------- | ----- |
| Molecule | O₂, charge 0, triplet (`spin=2` = 2S) |
| Geometry (Å) | O(0,0,0); O(0,0,1.2075) — `pin_scf` `o2` preset |
| Basis | STO-3G |
| Angular | `cart=True` |
| SCF | UHF, `conv_tol=1e-12`; ATV uses ROHF warm-start |
| CCSD | `pyscf.cc.UCCSD(mf)`, `conv_tol=1e-10` |
| EOM | `mycc.eomsf_ccsd(nroots=15)` then **lowest 5** real roots |
| Electrons | \(n_\alpha=9\), \(n_\beta=7\) |
| AO / MO | 10 spatial AO/MO |
| PySCF | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{UHF}}\) | −147.633946785450 Ha | \(10^{-6}\) Ha | REFERENCE (SCF context) |
| \(E_{\mathrm{UCCSD}}\) | **−147.741936323434 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(E_{\mathrm{corr}}\) | −0.107989537984 Ha | \(10^{-6}\) Ha | REFERENCE |
| \(\omega_0\) (SF, near-zero multiplet) | **0.004154770937 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_1\) | **1.024028019354 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_2\) | **1.024028354230 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_3\) | **1.586396740693 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\omega_4\) | **6.184403293079 eV** | **\(10^{-4}\) eV** | Product gate |
| Convergence | — | `converged == true` | Product gate |

Oracle repro (human / offline only):

```bash
bash tools/setup_pyscf_env.sh
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py eom-sf-ccsd \
    --geom o2 --basis sto-3g --spin 2 --nstates 15 --conv-tol 1e-12
```

## Performance

Source: fair wall in
[`regressions/scf/eom-sf-ccsd-o2/REFERENCE.md`](../../../regressions/scf/eom-sf-ccsd-o2/REFERENCE.md)
(2026-07-23; login node; `OMP_NUM_THREADS=1`; median of ≥3). Campaign rollup
ratio **3.12** (the case `REFERENCE.md`).

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `eom-sf-ccsd` (O₂/STO-3G) | 0.317 | 0.988 | **3.12** | ✓ | ATV: UHF (ROHF warm-start) + UCCSD + majority-SF dense Jacobian; PySCF twin for wall uses `eomsf_ccsd(nroots=5)` Davidson (accuracy pins use `nroots=15` lowest 5) |

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method eom-sf-ccsd --geom o2 --basis sto-3g --spin 2 --nstates 5 --repeat 3 -- \
    ./build/regressions/reg_eom-sf-ccsd-o2
```

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). No ⚡ optimization pass
required on this case (\(r=3.12\)).

## Limitations

- **SF on UHF+UCCSD only** — EE (`RHF::eom_ee_ccsd`), EA, IP, DIP/DEA, CVS-EOM,
  STEOM, EOM-CC2/MP2 remain separate roadmap rows
  (`docs/roadmap/excited-states.md` §3).
- **Energies only** — no left eigenvectors, oscillator strengths, densities, or
  analytic EOM-SF-CCSD gradient/Hessian.
- **Dense Jacobian** path; no large-system iterative Davidson product path yet.
- **No DF/CD/F12/local** EOM; full MO \(O(N^6)\) ground state + dense SF space.
- **Multi-minima UHF** on O₂ requires ROHF warm-start for reproducible roots;
  bare UHF may land on a different SCF minimum.
- **PySCF Davidson vs dense ladder:** fair-bench twin uses `nroots=5` for wall;
  accuracy pins need `nroots=15` (or ATV dense SF selection) to recover the same
  intermediate roots.
- **CUDA / Kokkos:** ○ (CPU only).
- **Perf twin** is the same method family on a minimal open-shell system; \(r\)
  is not a large-\(N\) predictor for full SF-EOM-CCSD.

## References

- Offline pin: `regressions/scf/eom-sf-ccsd-o2/REFERENCE.md`
- Regression consumer: `regressions/scf/eom-sf-ccsd-o2/reg_eom-sf-ccsd-o2.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Roadmap: `docs/roadmap/excited-states.md` (§3 Equation-of-Motion Coupled Cluster);
  ground-state CC: `docs/roadmap/post-hf.md` (§3)
- Public API: `include/atomiverse/cpu/scf/eom_sf_ccsd.h`,
  `include/atomiverse/cpu/scf/uhf.h`
- Related: SF-TDA foothold [`spin-flip-tda-o2.md`](spin-flip-tda-o2.md);
  closed-shell EOM-EE [`eom-ee-ccsd.md`](eom-ee-ccsd.md)
- A. I. Krylov, *Chem. Phys. Lett.* **338**, 375 (2001).
  DOI [10.1016/S0009-2614(01)00287-1](https://doi.org/10.1016/S0009-2614(01)00287-1)
- S. V. Levchenko & A. I. Krylov, *J. Chem. Phys.* **120**, 175 (2004).
  DOI [10.1063/1.1630018](https://doi.org/10.1063/1.1630018)
- PySCF: `pyscf.cc.eom_uccsd.EOMEESpinFlip` / `UCCSD.eomsf_ccsd`
