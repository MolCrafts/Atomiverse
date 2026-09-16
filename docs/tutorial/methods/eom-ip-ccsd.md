# EOM-IP-CCSD — H₂O / STO-3G (lowest IP)

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `eom-ip-ccsd` |
| Regression | `regressions/scf/eom-ip-ccsd/` |
| Public API | `cpu::scf::RHF::eom_ip_ccsd(Frame&, n_states)` → `EOMIPCCSDResult` |
| Roadmap | [Excited states §3 EOM-CC](../../roadmap/excited-states.md) |

## Summary

Closed-shell **EOM-IP-CCSD** ionization potentials on a converged RHF → CCSD
reference. Product surface is `RHF::eom_ip_ccsd(frame, n_states)`: the call
drives RHF (with ERI), spin-orbital Gauss–Stanton CCSD, then a dense
non-symmetric eigen-solve of the EOM-IP Jacobian, returning unique ionization
roots \(\omega = E_{\mathrm{CCSD}}(N) - E(N-1)\) [eV] (spin degeneracy
de-duplicated) plus CCSD energetics [Ha]. Foothold case is H₂O / STO-3G
(valence IP₀ ≈ 8.42 eV; not an experimental vertical IP). Offline pin is
PySCF 2.14.0 `cc.CCSD(...).ipccsd` (never live in ctest). End-to-end gate:
`regressions/scf/eom-ip-ccsd` → binary `reg_eom-ip-ccsd`. Fair-bench ratio
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} = 0.754\) (within 10×;
startup-dominated).

## Method

**Ground-state CCSD.** Cluster operator \(T = T_1 + T_2\) on a closed-shell RHF
determinant \(\lvert\Phi_0\rangle\):

\[
\lvert\Psi_{\mathrm{CCSD}}\rangle = e^{T}\lvert\Phi_0\rangle,\qquad
\bar{H} = e^{-T} H e^{T}.
\]

**EOM-IP.** Ionized states as right eigenstates of the similarity-transformed
Hamiltonian in the 1h / 2h–1p space (Stanton & Bartlett; Nooijen & Snijders;
Tu–Wang–Li):

\[
\bar{H} R_k = \omega_k R_k,\qquad
\omega_k = E_{\mathrm{CCSD}}(N) - E_k(N-1),
\]

with right operator

\[
R_k = R_1 + R_2
\quad\text{(spin-orbital 1h + 2h–1p)}.
\]

Right-hand \(\sigma\) equations follow Tu–Wang–Li Eqs. (8)–(9) as coded in
PySCF `eom_gccsd.ipccsd_matvec`. Roots from dense non-symmetric `dgeev` on the
spin-orbital EOM-IP Jacobian; **spin-degenerate** ionization partners are
de-duplicated before return. \(\omega\) reported in eV via
\(E_{\mathrm{h}}\rightarrow\mathrm{eV}\) conversion in `core/units.h`
(CODATA \(1\,\mathrm{Ha}=27.211386245988\,\mathrm{eV}\),
`units::kHartreeInEv`).

**Spin treatment.** Spin-orbital IP Jacobian on a closed-shell RHF→CCSD
reference only. Product path matches PySCF spatial RCCSD `ipccsd` / GCCSD
`EOMIP` on the same RHF reference after spin de-duplication. No open-shell /
UCCSD reference.

**Key approximations / scope of this foothold:**

- Single-reference closed-shell **ionization** only — no EE/EA/SF/DIP/DEA
  channels on this API.
- Energies only — no left vectors, Dyson orbitals, ionization densities, or
  analytic EOM-IP gradients.
- Dense full Jacobian diagonalization (no Davidson / iterative \(\sigma\) on
  large systems in this foothold).
- Canonical MO integrals; no DF/CD/F12/local CC / CVS / STEOM.
- H₂O/STO-3G: 7 AO; 5 occ + 2 virt spatial MO; lowest IP is valence (~8.42 eV
  on this minimal basis). Higher roots IP₁ ≈ 10.91 eV, IP₂ ≈ 16.62 eV are
  REFERENCE diagnostics only (product gate is IP₀).

## Implementation

Public headers only for consumers (`Frame` + concrete `RHF`; no `*Config`):

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

cpu::Frame frame(3);
// O(0,0,0.1173); H(0,±0.7572,−0.4692) Å; Z={8,1,1}; Box::cubic(20)
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.max_iter = 100;

const cpu::scf::EOMIPCCSDResult eom = rhf.eom_ip_ccsd(frame, /*n_states=*/1);
// eom.energy_ev[0] [eV]; eom.e_ccsd_ha, eom.e_corr_ha [Ha];
// eom.ccsd_iterations; eom.converged
```

`EOMIPCCSDResult` (`include/atomiverse/cpu/scf/eom_ip_ccsd.h`): `energy_ev`
(ascending unique ionization roots [eV]), `e_corr_ha`, `e_ccsd_ha` [Hartree],
`ccsd_iterations`, `converged`. Pack-level
`eom_ip_ccsd_energies` is private engine code.
(`internal/eom_ip_ccsd_pack.h`). `RHF::eom_ip_ccsd` forces `compute_eri`, runs
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
  cmake --build build --target reg_eom-ip-ccsd -j$(nproc)
  ./build/regressions/reg_eom-ip-ccsd
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/eom_ip_ccsd.h` | Public `EOMIPCCSDResult` |
| `include/atomiverse/cpu/scf/rhf.h` | `RHF::eom_ip_ccsd(Frame&, n_states)` |
| `include/atomiverse/cpu/scf/internal/eom_ip_ccsd_pack.h` | Private pack entry |
| `src/cpu/scf/eom_ip_ccsd.cpp` | Intermediates, Jacobian, root de-dup |
| `regressions/scf/eom-ip-ccsd/reg_eom-ip-ccsd.cpp` | E2E consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/eom-ip-ccsd/REFERENCE.md`](../../../regressions/scf/eom-ip-ccsd/REFERENCE.md)
(offline pin; PySCF never in ctest). Protocol:
`docs/tutorial/appendix-pyscf-oracle.md`.

| Setting | Value |
| ------- | ----- |
| Molecule | H₂O, charge 0, singlet |
| Geometry (Å) | O(0,0,0.1173); H(0,±0.7572,−0.4692) — `pin_scf` `water` preset |
| Basis | STO-3G |
| Angular | `cart=True` (no \(d\) shells; engine-neutral) |
| SCF | RHF, `conv_tol=1e-12` |
| CCSD | `pyscf.cc.CCSD(mf)`, `conv_tol=1e-12` |
| EOM | `mycc.ipccsd(nroots≥1)` (spatial RCCSD IP; matches GCCSD EOMIP after spin de-dupe) |
| AO / MO | 7 AO; 5 occ, 2 virt spatial |
| PySCF | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{CCSD}}\) | **−75.01246170149486 Ha** | **\(10^{-6}\) Ha** | Product gate |
| \(\mathrm{IP}_0\) (lowest EOM-IP) | **8.415727034061687 eV** | **\(10^{-4}\) eV** | Product gate |
| \(\mathrm{IP}_0\) (Ha cross-check) | 0.309272264117102 Ha | — | REFERENCE only |
| \(\mathrm{IP}_1\) (diagnostic) | ≈ 10.911018291684588 eV | — | Not gated |
| \(\mathrm{IP}_2\) (diagnostic) | ≈ 16.62369673430045 eV | — | Not gated |
| Convergence | — | `converged == true` | Product gate |

The pin is the PySCF EOM-IP-CCSD root, not an experimental vertical IP. On this
minimal basis IP₀ is ~8.42 eV (valence).

Oracle repro (human / offline only):

```bash
bash tools/setup_pyscf_env.sh
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py eom-ip-ccsd \
    --geom water --basis sto-3g --nstates 3 --conv-tol 1e-12
```

## Performance

Source: fair wall in
[`regressions/scf/eom-ip-ccsd/REFERENCE.md`](../../../regressions/scf/eom-ip-ccsd/REFERENCE.md)
(2026-07-23; login node; `OMP_NUM_THREADS=1`; median of ≥3). Campaign rollup
ratio **0.754** (the case `REFERENCE.md`). No standing comparison table in-tree at draft time.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `eom-ip-ccsd` (H₂O/STO-3G) | 0.1470 | 0.1108 | **0.754** | ✓ | Startup-dominated foothold; ATV slightly faster than PySCF (\(r < 1\)); not a production-scale EOM stress test |

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method eom-ip-ccsd --geom water --basis sto-3g --nstates 1 --repeat 3 -- \
    ./build/regressions/reg_eom-ip-ccsd
```

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\). No ⚡ optimization pass
required on this case.

## Limitations

- **Ionization only** on closed-shell RHF+CCSD; EE/EA/SF/DIP/DEA ship as
  separate product paths where available (see
  `docs/roadmap/excited-states.md` §3). CVS-EOM, STEOM, EOM-MP2, and
  analytic EOM-CCSD gradients remain ○.
- **Energies only** — no left eigenvectors, Dyson orbitals, densities, or
  analytic EOM-IP gradient/Hessian.
- **Dense Jacobian** path; no large-system iterative Davidson product path yet.
- **No DF/CD/F12/local** EOM; full MO \(O(N^6)\) ground state + dense IP space.
- **CUDA / Kokkos:** ○ (CPU only).
- **Perf twin** is the same method family on a minimal system; wall times are
  dominated by process startup and tiny MO spaces, so \(r\) is not a reliable
  large-\(N\) predictor.

## References

- Offline pin: `regressions/scf/eom-ip-ccsd/REFERENCE.md`
- Regression consumer: `regressions/scf/eom-ip-ccsd/reg_eom-ip-ccsd.cpp`
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Roadmap: `docs/roadmap/excited-states.md` (§3 Equation-of-Motion Coupled Cluster);
  ground-state CC: `docs/roadmap/post-hf.md` (§3)
- Public API: `include/atomiverse/cpu/scf/eom_ip_ccsd.h`,
  `include/atomiverse/cpu/scf/rhf.h`
- J. F. Stanton & R. J. Bartlett, *The equation of motion coupled-cluster
  method…*, *J. Chem. Phys.* **98**, 7029 (1993).
  DOI [10.1063/1.464746](https://doi.org/10.1063/1.464746)
- M. Nooijen & J. G. Snijders, *Coupled cluster approach to the single-particle
  Green's function*, *Int. J. Quantum Chem.* **S26**, 55 (1992).
- Z. Tu, F. Wang & X. Li, *Equation-of-motion coupled-cluster method for
  ionized states…*, *J. Chem. Phys.* **136**, 174102 (2012).
  DOI [10.1063/1.4704893](https://doi.org/10.1063/1.4704893)
- PySCF `eom_gccsd.ipccsd_matvec` (Tu–Wang–Li Eqs. (8)–(9) σ equations)
