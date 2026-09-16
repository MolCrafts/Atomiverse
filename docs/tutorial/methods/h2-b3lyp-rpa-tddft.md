# Hybrid full-LR TDDFT (B3LYP RPA) — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `h2-b3lyp-rpa-tddft` |
| Regression | `regressions/scf/h2-b3lyp-rpa-tddft/` |
| Public API | `cpu::scf::RKS` + `RKS::tddft` |
| Roadmap | [Excited states](../../roadmap/excited-states.md) — full linear-response TDDFT (hybrid) |

## Summary

Closed-shell **full linear-response TDDFT** (Casida **A,B** / RPA) on an **RKS/B3LYP** reference for H₂/STO-3G. Unlike hybrid TDA (`RKS::tda`; [h2-b3lyp-hybrid-tda](h2-b3lyp-hybrid-tda.md)), de-excitation (**B**) is retained; `ExcitedStatesResult::tda == false` and `alda == false` on the hybrid path. Primary observable: first singlet **S₁ [eV]** (one singles channel on this system: \(n_{\mathrm{occ}}=1\), \(n_{\mathrm{virt}}=1\)). Public surface is concrete `cpu::scf::RKS` with Frame I/O only — no `*Config` bags, no factories, no live PySCF in ctest. Offline oracle and gates: `regressions/scf/h2-b3lyp-rpa-tddft/REFERENCE.md`; binary `reg_h2-b3lyp-rpa-tddft`.

## Method

Ground state is restricted Kohn–Sham with B3LYP (global hybrid). Vertical response solves the Casida RPA problem (not Tamm–Dancoff):

\[
\begin{pmatrix}
\mathbf{A} & \mathbf{B} \\
\mathbf{B}^* & \mathbf{A}^*
\end{pmatrix}
\begin{pmatrix}
\mathbf{X} \\
\mathbf{Y}
\end{pmatrix}
=
\omega
\begin{pmatrix}
\mathbf{X} \\
\mathbf{Y}
\end{pmatrix}.
\]

Implementation reduces the Hermitian form
\((\mathbf{A}-\mathbf{B})^{1/2}(\mathbf{A}+\mathbf{B})(\mathbf{A}-\mathbf{B})^{1/2}\mathbf{Z}=\omega^{2}\mathbf{Z}\)
(dense path, \(n_{\mathrm{ov}}\le 512\)).

- **Spin:** closed-shell singlet channel (production path); triplet non-LSDA omits the finite-difference \(f^{xc}\) kernel.
- **Hybrid kernel:** exact-exchange fraction \(\alpha_{\mathrm{hf}}\) scales HF-exchange blocks on **both** A and B; adiabatic semi-local \(f^{xc}\) from finite difference of \(V_{xc}\) on the KS grid (`alda = false`).
- **Contrast TDA:** hybrid TDA is \(\mathbf{A}\mathbf{X}=\omega\mathbf{X}\) only; same geom/basis/XC gives S₁ ≈ 25.928 eV (TDA) vs ≈ 25.308 eV (full RPA) — full RPA lies below TDA when de-excitation is kept.
- **Contrast pure RPA:** pure GGA full LR uses semi-local \(f^{xc}\) without \(\alpha_{\mathrm{hf}}\) ([full-lr-tddft-h2-pbe](full-lr-tddft-h2-pbe.md)).
- **Not included:** UKS full-LR, spin-flip TDDFT, restricted excitation windows, CVS, real-time TDDFT, analytic hybrid full-LR TDDFT gradients (CIS/TDA forces on RHF are a separate product).

## Implementation

Public headers: `atomiverse/cpu/frame/frame.h`, `atomiverse/cpu/scf/methods.h` (or `rks.h` + `tdhf.h`). Parameters are constructor args / members on `RKS` — no public `*Config`.

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

// Geometry [Å]: H–H = 0.74 (regression places bond on +x; pin uses +z)
cpu::Frame f(2);
f.span(kAtom_x)[0] = 0.0;  f.span(kAtom_y)[0] = 0.0;  f.span(kAtom_z)[0] = 0.0;
f.span(kAtom_x)[1] = 0.74; f.span(kAtom_y)[1] = 0.0;  f.span(kAtom_z)[1] = 0.0;
f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
f.box = cpu::Box::cubic(20.0);

cpu::scf::RKS rks(cpu::scf::XCType::B3LYP);
rks.basis = cpu::scf::BasisSet::STO3G;
rks.tol = 1e-10;
rks.max_iter = 100;
rks.run(f);

const cpu::scf::ExcitedStatesResult es =
    rks.tddft(1, cpu::scf::ExcitationChannel::singlet);
// es.energy_ev[0] [eV]; es.tda == false; es.alda == false on hybrid
// optional: es.oscillator_strength[0] (length gauge)
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
  cmake --build build --target reg_h2-b3lyp-rpa-tddft -j$(nproc)
  ./build/regressions/reg_h2-b3lyp-rpa-tddft
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/scf/rks.h` | `RKS::tddft` API |
| `include/atomiverse/cpu/scf/tdhf.h` | `ExcitedStatesResult`, channels |
| `src/cpu/scf/tdhf.cpp` | dense Hermitian RPA from A,B |
| `include/atomiverse/cpu/scf/internal/orbital_hessian.h` | hybrid A/B + \(f^{xc}\) |
| `regressions/scf/h2-b3lyp-rpa-tddft/reg_h2-b3lyp-rpa-tddft.cpp` | e2e consumer |

## Validation (vs PySCF)

Source: [`regressions/scf/h2-b3lyp-rpa-tddft/REFERENCE.md`](../../../regressions/scf/h2-b3lyp-rpa-tddft/REFERENCE.md) (offline pin; PySCF never imported by CMake/ctest).

| Setup | Value |
| ----- | ----- |
| System | H₂, \(r(\mathrm{H–H})=0.74\) Å |
| Basis | STO-3G |
| Angular | Cartesian (`cart=True`; no d shells → cart/sph no-op) |
| XC / method | RKS/B3LYP + `tdscf.TDDFT` (full RPA, not TDA) |
| SCF `conv_tol` | \(1\times 10^{-12}\) (oracle); ATV reg uses `tol = 1e-10` |
| TDDFT | default PySCF `TDDFT`; `nstates=3` requested (1 physical root) |
| PySCF | **2.14.0** |
| Ha→eV | 27.211386245988 (CODATA ≡ `units::kHartreeInEv`) |

Geometry (Å):

| Atom | x | y | z |
| ---- | -: | -: | -: |
| H | 0.0 | 0.0 | 0.0 |
| H | 0.0 | 0.0 | 0.74 |

(Regression places the bond on +x; same \(r=0.74\) Å.)

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RKS}}\) [Ha] | **−1.165418410673** | \(1\times 10^{-4}\) Ha | ground-state sanity |
| S₁ [eV] (primary) | **25.30779734533265** | **\(1\times 10^{-4}\) eV** | job gate |
| S₁ [Ha] | **0.9300443981998157** | \(\approx 3.7\times 10^{-6}\) Ha | consistency via Ha→eV |
| \(f\) (length) | **0.8608891208128704** | 0.15 | optional band |
| `tda` / `alda` | false / false | — | reg rejects TDA / ALDA flags |
| Guard | ≠ 25.928364822972 eV | — | must not match hybrid-TDA pin |

Oracle command (offline only):

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rks-tddft \
  --geom h2 --basis sto-3g --xc b3lyp --nstates 3 --conv-tol 1e-12
```

Context (same geom/basis; not gates): CIS/RHF-TDA ≈ 25.81 eV; B3LYP hybrid TDA ≈ 25.93 eV; **B3LYP full RPA ≈ 25.31 eV**.

## Performance

Source: `REFERENCE.md` fair-bench table (2026-07-23); worker ratio **0.430**. Twin class: RKS/B3LYP + full RPA TDDFT (`rks_tddft`). Medians of ≥3 runs; \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\); gate \(r\le 10\).

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| `rks_tddft` B3LYP H₂/STO-3G | 0.172 | 0.074 | **0.430** | ✓ | `OMP_NUM_THREADS=1` both sides |

Reproduce:

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method rks_tddft --geom h2 --basis sto-3g --xc b3lyp --nstates 3 \
    --repeat 3 --max-ratio 10.0 -- \
    ./build/regressions/reg_h2-b3lyp-rpa-tddft
'
```

**Twin caveats:** reg binary includes pin checks/guards; PySCF side is twin kernel only. The rollup currently lists related full-LR under `full-lr-tddft-h2-pbe` with a **TDA** twin (`rks_tda`, \(r\approx 0.27\)) — not interchangeable with this hybrid RPA twin. Micro-system H₂/STO-3G (\(n_{\mathrm{ov}}=1\)) does not stress dense RPA scaling.

## Limitations

- **Scope:** closed-shell `RKS::tddft` only; UKS full-LR, spin-flip full TDDFT, and open-shell hybrids not in this product path.
- **Dense RPA ceiling:** \(n_{\mathrm{ov}}\le 512\); larger spaces throw (no automatic Davidson fallback on this API surface for production hybrid path as documented).
- **Triplet / non-LSDA:** FD \(f^{xc}\) calibrated for singlet; triplet non-LSDA path omits that kernel.
- **Gradients/NACs:** analytic CIS/TDA forces and NACs live on `RHF`; hybrid full-LR TDDFT analytic gradient is not the product path here.
- **Oracle isolation:** PySCF is offline pin only (`tools/setup_pyscf_env.sh`, `tools/pyscf_oracle/pin_scf.py`); never a ctest dependency.
- **Related pins:** do not hard-gate this S₁ against hybrid TDA (25.928 eV) or pure PBE full LR ([full-lr-tddft-h2-pbe](full-lr-tddft-h2-pbe.md)).

## References

- Offline pin: [`regressions/scf/h2-b3lyp-rpa-tddft/REFERENCE.md`](../../../regressions/scf/h2-b3lyp-rpa-tddft/REFERENCE.md)
- Roadmap: [`docs/roadmap/excited-states.md`](../../roadmap/excited-states.md)
- Related methods: [h2-b3lyp-hybrid-tda](h2-b3lyp-hybrid-tda.md) (TDA), [full-lr-tddft-h2-pbe](full-lr-tddft-h2-pbe.md) (pure GGA full LR)
- M. E. Casida, in *Recent Advances in Density Functional Methods*, Part I (World Scientific, 1995) — Casida (A,B)
- S. Hirata, M. Head-Gordon, *Chem. Phys. Lett.* **314**, 291 (1999) — TDA vs full
- A. Dreuw, M. Head-Gordon, *Chem. Rev.* **105**, 4009 (2005) — TDDFT response
- A. D. Becke, *J. Chem. Phys.* **98**, 5648 (1993) — B3LYP hybrid form (PySCF `xc='B3LYP'`)
- Q. Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) — PySCF
