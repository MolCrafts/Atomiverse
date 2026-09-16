# Multi-k RI-GPW direct RPA — diamond Si₄ (bct) / MP 2×1×1

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-rpa-kpoints` |
| Regression | `regressions/gpw/periodic-rpa-kpoints/` |
| Public API | `cpu::gpw::GpwDft::direct_rpa(atoms)` → `cpu::gpw::RpaResult` (`cpu/gpw/rpa.h`); members `k_mesh`, `k_time_reversal`, `mp2_ri`, `mp2_n_virt` |
| Roadmap | [dft.md](../../roadmap/dft.md) §10 Periodic RPA ✓ (multi-k k-diagonal) |
| Oracle | CP2K offline twin `tools/cp2k_oracle/inputs/periodic-rpa-kpoints.inp` (never live in ctest) |

## Summary

**k-diagonal direct RPA (dRPA)** correlation on a multi-k **GPW PBE** SCF for
diamond silicon in a body-centred tetragonal 4-atom cell. For each sampled
\(\mathbf{k}\), the Coulomb-only Casida spectrum is built from **same-\(k\)**
(vertical, \(q=0\)) occupied–virtual pair densities via the **RI-GPW**
Poisson product path (complex Bloch products; real Poisson on Re/Im of the
pair density), reduced by the **Furche plasmon formula**, and Brillouin-zone
averaged. This closes the multi-k residual after the Γ-only product
`periodic-rpa-si` (where `direct_rpa` previously threw for \(N_k>1\)). Public
path is members on concrete `GpwDft` → `build` → `run` → `direct_rpa` — no
`*Config` bag, no factory. E2E: `reg_periodic-rpa-kpoints`. Fair wall vs CP2K
is **`perf_exempt`** (no CP2K binary on the pin host; CP2K RI-RPA ≠ exact
k-diagonal plasmon on PW-RI \((ia|jb)\)).

## Method

**Level.** Closed-shell **direct RPA** correlation energy on Kohn–Sham (PBE)
orbitals under periodic boundary conditions — the periodic GPW analogue of
molecular `RHF::direct_rpa` / Γ `GpwDft::direct_rpa`. Not TDHF excitation RPA,
not frequency-integrated stock CP2K RI-RPA with EXX total-energy assembly, not
RPAX/SOSEX/ACFDT product paths.

**Spin.** Closed-shell spatial MOs only (RKS). UKS / open-shell dRPA is out of
scope for this product.

**k treatment (product).** **k-diagonal** only: each \(\mathbf{k}\) builds its
own Casida block from same-\(k\) ov pairs. Full inter-\(k\) / finite-\(q\)
dielectric coupling is a follow-on residual.

\[
E_c
  = \sum_k w_k\,\tfrac12\Bigl(\sum_s \Omega_{k,s} - \mathrm{Tr}\,A_k\Bigr).
\]

**Kernel.** Per \(k\), singlet spin-adapted Coulomb-only Casida
(\(\alpha_{\mathrm{hf}}=0\)):

\[
A_{ia,jb}(k)
  = (\varepsilon_{a k}-\varepsilon_{i k})\,\delta_{ij}\delta_{ab}
    + 2\,(ia|jb)_k,
\qquad
B_{ia,jb}(k) = 2\,(ia|jb)_k.
\]

**\((ia|jb)\) route.** RI-GPW (Del Ben, Hutter, VandeVondele, *JCTC* **8**,
4177 (2012)): collocate complex Bloch pair densities on the density grid,
solve the periodic FFT Poisson equation (linear real solve on Re/Im),
contract in G-space. Product path requires `mp2_ri = true` for multi-k;
Γ recovery with real MOs also accepts the Poisson foothold used by
`periodic-rpa-si` / unit tests.

**Correlation.** Furche plasmon formula on the positive eigenvalues \(\Omega\)
of the Hermitian reduction
\((A-B)^{1/2}(A+B)(A-B)^{1/2}\). Exact (no imaginary-frequency quadrature) at
the level of the discrete Casida matrix.

**Total energy.** **dRPA@KS**: \(E_{\mathrm{total}} = E_{\mathrm{SCF}} + E_c\).
No EXX replacement — periodic exact exchange remains blocked
([hartree-fock.md](../../roadmap/hartree-fock.md) periodic HF rows).

**Excluded.** Full \(q\)-resolved RPA dielectric; multi-k G0W0 (separate residual
after Γ `periodic-gw-si`); RPAX / SOSEX; CP2K-style minimax + Gaussian
`RI_AUX` frequency integration as the accuracy twin; CUDA.

## Implementation

Public surface only (`GpwDft` members; `GpwAtom` + lookup constructors for
basis/PP). Headers: `include/atomiverse/cpu/gpw/gpw_solver.h`,
`rpa.h`, `gaussian_basis.h`, `gth_potential.h`. Kernels private under
`cpu/gpw/internal/`.

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/rpa.h"
#include "atomiverse/molrs.hpp"

// Diamond Si bct 4-atom cell: a = 5.43 Å → Bohr; a_t = a/√2, c = a
// frac: (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) — same as gpw-kpoints-si
std::vector<cpu::gpw::GpwAtom> atoms = /* build with
    GaussianBasis(Si, "DZVP-MOLOPT-GTH"),
    GthPotential(Si, "GTH-PBE-q4") */;

cpu::gpw::GpwDft m;
m.cell[0] = m.cell[1] = a_bohr / std::sqrt(2.0);
m.cell[2] = a_bohr;
m.cutoff_ry = 80.0;
m.n_occ = 8;                 // 16 valence e⁻ / 2
m.max_iter = 60;
m.tol = 1e-7;
m.mix = 0.3;
m.level_shift = 0.0;
m.eps_default = 1e-7;
m.xc = cpu::gpw::XcKind::PBE;
m.diis_depth = 6;
m.mp2_ri = true;             // multi-k product path
m.mp2_n_virt = 2;            // e2e virt cap (≤ 60 s)
m.k_mesh[0] = 2;
m.k_mesh[1] = 1;
m.k_mesh[2] = 1;
m.k_time_reversal = false;   // FULL_GRID / no symmetry → 2 k, w = ½
m.build(atoms);
const cpu::gpw::GpwResult& scf = m.run(atoms);

const cpu::gpw::RpaResult& r = m.direct_rpa(atoms);
// r.e_corr, r.e_scf, r.e_total [Ha]; r.tr_A; r.omega_ha;
// r.n_occ, r.n_virt, r.n_kpts, r.ri
```

`RpaResult` fields: `e_corr`, `e_scf`, `e_total`, `tr_A`, `n_occ`, `n_virt`,
`ri`, `n_kpts`, `omega_ha` (multi-k: all \(k\)-block roots concatenated then
sorted; length \(N_k \cdot n_{\mathrm{occ}} \cdot n_{\mathrm{virt}}\) for the
k-diagonal product). Requires prior converged closed-shell `run()` with
stored \(C_k\); multi-k without RI is not the product path.

```bash
# configure once per tree (login shell + Lmod — see AGENTS.md)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_periodic-rpa-kpoints
  ./build/regressions/reg_periodic-rpa-kpoints
'
```

Source consumer:
`regressions/gpw/periodic-rpa-kpoints/reg_periodic-rpa-kpoints.cpp`.
Unit kernel Γ recovery (\(N_k=1\) multi-k ≡ real `direct_rpa_rhf_energy`):
`tests/cpu/gpw/test_cpu_gpw_rpa.cpp`.

## Validation (vs CP2K)

Source:
[`regressions/gpw/periodic-rpa-kpoints/REFERENCE.md`](../../../regressions/gpw/periodic-rpa-kpoints/REFERENCE.md)
(offline pin only; CP2K never launched from CMake/ctest).

| Setting | Value |
| ------- | ----- |
| Crystal | diamond Si, **bct 4-atom** (same as `gpw-kpoints-si` / `periodic-mp2-kpoints`) |
| Cell | \(a_t = a/\sqrt{2}\), \(c = a\); \(a =\) **5.43 Å** |
| Positions (frac) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC | **PBE**, cutoff **80 Ry**, `eps_default` 1e-7 |
| Valence | 16 e⁻ → \(n_{\mathrm{occ}} = 8\) |
| k | **MP 2×1×1**, `k_time_reversal=false` → \(N_k=2\), \(w_k=\tfrac12\) |
| SCF | DIIS 6, tol 1e-7, mix 0.3, **no level_shift** |
| RPA | RI-GPW k-diagonal dRPA, **`mp2_n_virt = 2`** (e2e ≤ 60 s) |
| Oracle input | `tools/cp2k_oracle/inputs/periodic-rpa-kpoints.inp` |
| Pin date | 2026-07-25 |

### Offline oracle status

**No CP2K binary / module on the worker host at pin time** (same status as
`periodic-mp2-kpoints` / `periodic-rpa-si`). Absolute CP2K correlation digits
are **not invented**. CP2K's default path is frequency-integrated **RI-RPA**
with Gaussian `RI_AUX`, minimax quadrature, and **EXX** total-energy assembly
— not the exact k-diagonal Furche plasmon on PW-RI \((ia|jb)\). Printed CP2K
\(E_c\) is a near-twin diagnostic only; the hard pin is the Atomiverse plasmon
band below.

### Binding gates

1. **Hard multi-k:** SCF converged; `C_k.size() == n_kpts`; `e_corr` finite and
   **strictly negative**; all \(\Omega > 0\); \(e_{\mathrm{total}} \equiv
   e_{\mathrm{scf}} + e_{\mathrm{corr}}\); `n_occ`/`n_virt`; `ri` and
   `n_kpts` flags; \(\lvert\omega\rvert = N_k\cdot n_{\mathrm{occ}}\cdot
   n_{\mathrm{virt}}\).
2. **Kernel Γ recovery (unit):** multi-k kernel at \(N_k=1\) ≡ real
   `direct_rpa_rhf_energy` (`test_cpu_gpw_rpa`).
3. **First-green absolute band** (setup integrity) — freeze ATV digits; replace
   with fair CP2K when a plasmon-equivalent twin exists.

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{SCF}}\) (2k, PBE, 80 Ry) | **−15.5751782413** Ha | setup | 7 iters; stable |
| **\(E_c\) (2k, \(n_{\mathrm{virt}}=2\))** | **≈ −0.0163** Ha | **5e-3 Ha** | primary e2e band (OMP orbital scatter) |
| \(E_c\) (OMP=1, deterministic) | −0.0157519268 Ha | info | REFERENCE sample |
| \(E_c\) (OMP=4, sample range) | −0.0154 … −0.0173 Ha | info | multi-k SCF scatter |
| \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\) | 8 / 2 | exact | regression gate |
| \(N_k\) / `ri` | 2 / true | exact | multi-k RI product |
| \(n_\Omega\) | \(2\cdot 8\cdot 2 = 32\) | exact | k-diagonal spectrum size |
| Wall (SCF + RPA) | **~3 s** | ≤ 60 s | EPYC login, e2e budget |

Regression absolute tol on \(E_c\): **5×10⁻³ Ha** about −0.0163 (setup band
covering OMP orbital scatter on the multi-k SCF; **not** a CP2K digit twin).

## Performance

Source: REFERENCE + the case `REFERENCE.md` (ratio
**perf_exempt**). Gate when a fair twin exists:
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| multi-k RI-GPW dRPA Si₄-bct | — | ~3 (SCF+RPA) | **perf_exempt** | — | no CP2K binary on pin host; CP2K RI-RPA ≠ k-diagonal plasmon; EXX path differs |

Never launch CP2K from ctest. When a fair algorithm twin and binary are
available, re-pin wall and accuracy in REFERENCE and handoff.

## Limitations

- **k-diagonal product only.** No full inter-\(k\) / finite-\(q\) RPA dielectric
  matrix; BZ average is \(\sum_k w_k E_c(k)\) on vertical transitions.
- **Closed-shell RKS / dRPA@KS.** No UKS dRPA; no EXX-replaced RPA total energy
  (periodic HF blocked).
- **RI-GPW multi-k path.** Product requires `mp2_ri = true`; virt space capped
  at `mp2_n_virt = 2` for the e2e wall budget (production decks raise this).
- **Not a stock-CP2K RI-RPA digit twin.** Frequency-integrated Gaussian
  `RI_AUX` + minimax + EXX assembly is a different algorithm; absolute CP2K
  \(E_c\) is diagnostic until a fair plasmon path is defined offline.
- **OMP scatter on multi-k SCF orbitals.** Absolute \(E_c\) band is deliberately
  loose (5e-3 Ha); unit tier pins the kernel at \(N_k=1\).
- **CPU only.** CUDA multi-k / RPA is ○.
- **Follow-ons.** Fair CP2K re-pin; denser MP meshes / full virt space; full
  \(q\)-resolved RPA; multi-k G0W0 residual; physical diamond-Si₈ production
  cutoffs.

## References

- Offline pin: `regressions/gpw/periodic-rpa-kpoints/REFERENCE.md`
- Oracle input: `tools/cp2k_oracle/inputs/periodic-rpa-kpoints.inp`
- Oracle rules: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` §10 Periodic RPA ✓ (Γ + multi-k k-diagonal)
- Γ foothold tutorial / reg: [periodic-mp2-si.md](periodic-mp2-si.md),
  `regressions/gpw/periodic-rpa-si`
- Multi-k MP2 peer: `regressions/gpw/periodic-mp2-kpoints`
- Molecular dRPA (Furche plasmon, exact MO-ERI): [direct-rpa-h2-sto3g.md](direct-rpa-h2-sto3g.md)
- Public headers: `include/atomiverse/cpu/gpw/rpa.h`, `gpw_solver.h`
- Literature:
  - Furche, *J. Chem. Phys.* **114**, 5982 (2001), DOI 10.1063/1.1353585
  - Ren et al., *New J. Phys.* **14**, 053020 (2012)
  - Del Ben, Hutter, VandeVondele, *J. Chem. Theory Comput.* **8**, 4177 (2012)
    (GPW \((ia|jb)\) / RI-GPW layout)
