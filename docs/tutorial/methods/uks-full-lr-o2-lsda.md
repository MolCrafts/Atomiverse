# UKS full-LR TDDFT (Casida A/B) — O₂ / LSDA / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `uks-full-lr-o2-lsda` |
| Regression | `regressions/scf/uks-full-lr-o2-lsda/` |
| Public API | `cpu::scf::UKS` + `UKS::tddft` (not `UKS::tda`) |
| Roadmap | [Excited states](../../roadmap/excited-states.md) — UKS full-LR TDDFT ✓ |

## Summary

Open-shell **full linear-response TDDFT** (Casida **A,B** / RPA) on a **UKS/LSDA**
triplet O₂ reference (\(2S=2\)). Unlike UKS TDA (`UKS::tda`), de-excitation
channels are retained; `ExcitedStatesResult::tda == false` and `alda == true`
on the pure-LSDA path. Primary observables: five lowest spin-conserving
excitation roots **ω [eV]**. Public surface is concrete `cpu::scf::UKS` with
Frame I/O only — no `*Config` bags, no factories, no live PySCF in ctest.
Offline oracle and gates: `regressions/scf/uks-full-lr-o2-lsda/REFERENCE.md`;
binary `reg_uks-full-lr-o2-lsda`.

## Method

**Reference.** Unrestricted Kohn–Sham with Slater exchange + VWN5 correlation
(Atomiverse `XCType::LSDA` ≡ PySCF `xc="LDA,VWN"`). Triplet O₂:
\((n_\alpha, n_\beta) = (9, 7)\), charge 0, spin \(2S=2\).

**Response.** Spin-conserving Casida full RPA (not Tamm–Dancoff):

\[
A_{ia\sigma,jb\sigma'}
=
\delta_{\sigma\sigma'}\delta_{ij}\delta_{ab}
(\varepsilon_a^\sigma-\varepsilon_i^\sigma)
+(ia\sigma|jb\sigma')
+(ia\sigma|f^{\mathrm{xc}}_{\sigma\sigma'}|jb\sigma')
\]

\[
B_{ia\sigma,jb\sigma'}
=
(ia\sigma|bj\sigma')
+(ia\sigma|f^{\mathrm{xc}}_{\sigma\sigma'}|jb\sigma')
\]

with pure DFT \(\alpha_{\mathrm{hf}}=0\) and adiabatic spin-resolved
\(f^{\mathrm{xc}}\) from LSDA second derivatives on the KS grid (PySCF
`tdscf.TDDFT` on `dft.UKS`). For real MOs the Coulomb/fxc blocks of **A** and
**B** coincide, so \(A-B=\mathrm{diag}(\varepsilon_a-\varepsilon_i)\); the
Hermitian Casida reduction via \((A-B)^{\pm 1/2}\) applies (same path as
`RKS::tddft`).

**Observable.** Lowest Ms-preserving excitation roots **ω [eV]** of the
triplet reference.

**Contrast TDA (same geom/basis/XC; not a gate).** Full RPA leaves root 0
essentially unchanged and **lowers** roots 3–4 by ~0.017 eV:

| Theory | root 0 [eV] | root 3 [eV] |
| ------ | ----------: | ----------: |
| **UKS full RPA TDDFT** | **7.620917** | **8.386321** |
| UKS TDA | 7.620917 | 8.403285 |

The regression must not hard-gate against the TDA root-3 pin (8.403 eV).

**Not included.** Hybrid UKS full-LR, spin-flip TDDFT (separate product:
[sf-tddft-o2](sf-tddft-o2.md)), restricted excitation windows, CVS, real-time
TDDFT, analytic UKS full-LR gradients / NACs.

## Implementation

Public headers: `atomiverse/cpu/frame/frame.h`, `atomiverse/cpu/scf/methods.h`
(or `uks.h` + `tdhf.h`). Parameters are constructor args / members on `UKS` —
no public `*Config`. Multi-minima caution: regression uses ROKS warm-start →
UKS density guess (same pattern as `uks-tda-o2-lsda`).

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

// O₂, r(O–O) = 1.2075 Å; Z = 8; cubic box 20 Å
cpu::Frame frame(2);
// … place O at (0,0,0) and (0,0,1.2075); type/mass as in reg …

// ROKS warm-start (optional; stabilizes UKS multi-minima)
cpu::scf::ROKS roks(9, 7, cpu::scf::XCType::LSDA);
roks.basis = cpu::scf::BasisSet::STO3G;
roks.tol = 1e-12;
roks.max_iter = 200;
roks.grid_level = cpu::scf::GridLevel::MatchPySCF;
roks.run(frame);

using Spin = cpu::scf::SlaterDeterminant::Spin;
cpu::scf::SCFInitialGuess guess;
guess.P_alpha = roks.determinant().density(Spin::alpha);
guess.P_beta  = roks.determinant().density(Spin::beta);
guess.has_alpha = true;
guess.has_beta  = true;

cpu::scf::UKS uks(9, 7, cpu::scf::XCType::LSDA);
uks.basis = cpu::scf::BasisSet::STO3G;
uks.tol = 1e-12;
uks.max_iter = 200;
uks.grid_level = cpu::scf::GridLevel::MatchPySCF;
uks.run(frame, &guess);

// Full RPA — not UKS::tda
const cpu::scf::ExcitedStatesResult es = uks.tddft(5);
// es.energy_ev[i] [eV]; es.tda == false; es.alda == true (LSDA)
```

Build / run regression (HF enabled; CUDA off on login node):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_uks-full-lr-o2-lsda -j$(nproc)
  ./build/regressions/reg_uks-full-lr-o2-lsda
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/uks.h` | `UKS::tddft` / `UKS::tda` API |
| `include/atomiverse/cpu/scf/tdhf.h` | `ExcitedStatesResult`, channels |
| `src/cpu/scf/tdhf.cpp` | dense Hermitian RPA from A,B |
| `regressions/scf/uks-full-lr-o2-lsda/reg_uks-full-lr-o2-lsda.cpp` | e2e consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/uks-full-lr-o2-lsda/REFERENCE.md`](../../../regressions/scf/uks-full-lr-o2-lsda/REFERENCE.md)
(offline pin; PySCF never imported by CMake/ctest).

| Setup | Value |
| ----- | ----- |
| System | O₂, \(r(\mathrm{O–O})=1.2075\) Å |
| Basis | STO-3G |
| Angular | Cartesian (`cart=True`; no \(d\) → cart/sph no-op) |
| XC / method | UKS/LSDA (`LDA,VWN`) + `tdscf.TDDFT` (full RPA, not TDA) |
| Occupation | \((n_\alpha, n_\beta)=(9, 7)\), spin \(2S=2\) |
| SCF `conv_tol` | \(1\times 10^{-12}\) |
| TDDFT | default PySCF `TDDFT`; `nstates=5` |
| PySCF | **2.14.0** |
| Ha→eV | 27.211386245988 (CODATA ≡ `units::kHartreeInEv`) |

Geometry (Å):

| Atom | x | y | z |
| ---- | -: | -: | -: |
| O | 0.0 | 0.0 | 0.0 |
| O | 0.0 | 0.0 | 1.2075 |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{UKS}}\) [Ha] | **−147.193805922283** | \(1\times 10^{-4}\) Ha | ground-state sanity; S²≈2.000202 |
| ω₀ [eV] | **7.620917323839** | **\(1\times 10^{-4}\) eV** | job gate |
| ω₁ [eV] | **7.642419562717** | **\(1\times 10^{-4}\) eV** | job gate |
| ω₂ [eV] | **7.642420073101** | **\(1\times 10^{-4}\) eV** | job gate (near-degenerate with ω₁) |
| ω₃ [eV] | **8.386321248270** | **\(1\times 10^{-4}\) eV** | job gate |
| ω₄ [eV] | **8.386321248273** | **\(1\times 10^{-4}\) eV** | job gate (degenerate with ω₃) |
| `tda` / `alda` | false / true | — | reg rejects TDA flag; requires ALDA on LSDA |
| Guard | ≠ TDA root 3 **8.403285** eV | — | must not match `uks-tda-o2-lsda` root 3 |

Oracle command (offline only):

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py uks-tddft \
  --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN --nstates 5 --conv-tol 1e-12
```

## Performance

Source: `REFERENCE.md` fair-bench table (2026-07-23); ratio **2.937**.
Twin class: UKS/LSDA + full RPA TDDFT (`uks_tddft`). Medians of 3 runs;
\(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\); gate \(r\le 10\); `perf_exempt=false`.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `uks_tddft` LSDA O₂/STO-3G | 0.229 | 0.672 | **2.937** | ✓ | `OMP_NUM_THREADS=1` both sides |

Reproduce:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method uks_tddft --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN \
    --nstates 5 --repeat 3 -- \
    ./build/regressions/reg_uks-full-lr-o2-lsda
'
```

**Twin caveats:** reg binary includes ROKS warm-start, pin checks, and TDA
guard; PySCF side is twin kernel only. Textbook STO-3G O₂ does not stress
dense RPA scaling at production molecular sizes.

## Limitations

- **Scope:** pure-LSDA `UKS::tddft` only; hybrid UKS full-LR and open-shell
  hybrid response are not this product path.
- **TDA vs RPA:** do not hard-gate roots against `uks-tda-o2-lsda` (esp. root 3
  ≈ 8.403 eV); full RPA shifts higher roots via **B**.
- **Multi-minima:** UKS/O₂ can land in alternate SCF solutions; ROKS density
  warm-start is the documented stabilization (same as TDA pin).
- **Dense RPA ceiling:** production path follows the closed-shell dense
  Hermitian reduction; large \(n_{\mathrm{ov}}\) spaces are not the textbook
  gate of this regression.
- **Gradients/NACs:** analytic UKS full-LR TDDFT gradient / NAC is not the
  product path here.
- **Oracle isolation:** PySCF is offline pin only (`tools/setup_pyscf_env.sh`,
  `tools/pyscf_oracle/pin_scf.py`); never a ctest dependency.
- **Related pins:** closed-shell pure full LR ([full-lr-tddft-h2-pbe](full-lr-tddft-h2-pbe.md));
  hybrid closed-shell RPA ([h2-b3lyp-rpa-tddft](h2-b3lyp-rpa-tddft.md));
  spin-flip full TDDFT ([sf-tddft-o2](sf-tddft-o2.md)).

## References

- Offline pin: [`regressions/scf/uks-full-lr-o2-lsda/REFERENCE.md`](../../../regressions/scf/uks-full-lr-o2-lsda/REFERENCE.md)
- Roadmap: [`docs/roadmap/excited-states.md`](../../roadmap/excited-states.md)
- Related methods: [full-lr-tddft-h2-pbe](full-lr-tddft-h2-pbe.md) (RKS pure full LR),
  [h2-b3lyp-rpa-tddft](h2-b3lyp-rpa-tddft.md) (RKS hybrid RPA),
  [sf-tddft-o2](sf-tddft-o2.md) (spin-flip full RPA)
- M. E. Casida, in *Recent Advances in Density Functional Methods*, Part I
  (World Scientific, 1995) — Casida equations (A,B)
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA vs full
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF
