# Periodic finite-\(q\) inter-\(k\) spectral G0W0 — diamond Si (bct 4-atom, MP 2×1×1)

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-gw-finite-q` |
| Regression | `regressions/gpw/periodic-gw-finite-q/` |
| Public API | `cpu::gpw::GpwDft::g0w0(atoms)` → `cpu::gpw::G0W0Result` (`cpu/gpw/g0w0.h`) |
| Roadmap | residual close of Periodic GW ✓ — product multi-k path after Γ (`periodic-gw-si`) and k-diagonal (`periodic-gw-kpoints`) |
| Ratio | **`perf_exempt`** (CP2K RI-G0W0 ≠ finite-\(q\) spectral + MT) |

## Summary

Closed-shell **finite-\(q\) inter-\(k\) spectral G0W0** quasiparticle (QP)
energies on a multi-k **GPW PBE** reference. For each crystal-momentum
transfer \(q\) on the same closed Monkhorst–Pack mesh as \(k\), the
Coulomb-only Casida spectrum is built from momentum-conserving transitions
\(i_k\to a_{k+q}\) via RI-GPW Poisson (complex Bloch; same dielectric as
multi-k dRPA). \(X+Y\) amplitudes form spectral transition potentials
\(W_s^q\); the correlation self-energy at target \(k\) contracts intermediate
states at \(k-q\) with weight \(1/N_k\). Exact-exchange diagonals use
truncated (Martyna–Tuckerman) Coulomb so the product is G=0-safe without a
periodic-HF path. Local \(v_{\mathrm{xc}}\) is PBE on the BZ-weighted density.

This **supersedes** the multi-k **k-diagonal** foothold of
`periodic-gw-kpoints` as the product multi-k G0W0 path. Public path:
`GpwDft` members → `build` / `run` → `g0w0(atoms)` — no public `*Config`, no
factory. Primary product observable: **minimum direct QP gap [eV]** over the
sampled mesh. Oracle policy: offline CP2K twin input only
(`tools/cp2k_oracle/`); never live CP2K in ctest.

## Method

**Level.** One-shot \(G_0W_0\): Green’s function and screened interaction from
the converged multi-k GPW RKS (PBE) solution; no self-consistency in \(G\) or
\(W\).

**Spin.** Closed-shell spatial MOs only (RKS). Open-shell / UKS G0W0 is out of
scope (see UMP2 residual product elsewhere; UGW held).

**k / q sampling.** Product multi-k path stores complex Bloch orbitals
\(C_k\) from SCF. Screening and self-energy use the **full finite-\(q\)**
mesh: for every transfer \(q = \mathrm{wrap}(k_a - k_i)\) on the same closed
MP set as \(k\), build the dRPA dielectric from transitions
\(i_k \to a_{k+q}\). When \(N_k = 1\) and orbitals are real-valued complex,
the multi-k kernel recovers the real Γ spectral path (unit-tested; no full
SCF under `tests/`).

**Screening.** Direct RPA (no exchange in the RPA kernel) — same inter-k
dielectric path as `GpwDft::direct_rpa` / `periodic-rpa-finite-q`. Integrals
are **RI-GPW** (product requires `mp2_ri = true`). Virtual count in the
screening space is capped by `mp2_n_virt` (e2e pin: **2**).

**Self-energy.** Spectral (frequency-exact) sum over RPA poles for the
diagonal correlation self-energy \(\mathrm{Re}\,\Sigma_c(\omega)\). Per
target \(k\), intermediate states at \(k-q\) enter with BZ weight \(1/N_k\).

**Exact exchange and local XC.** Diagonals \(\Sigma_{x,p}\) use truncated
(MT) Coulomb; \(v_{\mathrm{xc},p}\) is the local PBE potential on the
BZ-weighted density projected onto MOs at each \(k\).

**Quasiparticle equation** (Newton, orbit-by-orbit, per \(k\)):

\[
\omega = \varepsilon_p + \mathrm{Re}\,\Sigma_{c,p}(\omega)
         + \Sigma_{x,p} - v_{\mathrm{xc},p}.
\]

**Not** the linearized \(Z\)-factor form. References: Hybertsen & Louie,
*Phys. Rev. B* **34**, 5390 (1986); Furche, *J. Chem. Phys.* **114**, 5982
(2001); molecular twin `RHF::g0w0` / spectral form.

**Observables (public `G0W0Result`):**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varepsilon_p^{\mathrm{QP}}\) | QP orbital energies (reported \(k\)) | Ha |
| `gap_ev` | min direct \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}} - \varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) over mesh | eV |
| `mf_gap_ev` | min direct KS gap over mesh | eV |
| `ip_ev` / `ea_ev` | \(-\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) / \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) at report \(k\) | eV |
| `n_kpts` / `ri` / `converged` | mesh size; RI path; all Newton solves OK | — |

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Crystal | diamond Si, **bct 4-atom** cell |
| Cell | \(a_t = a/\sqrt{2}\), \(c = a\); \(a = 5.43\) Å |
| Positions (frac) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| XC | PBE; density cutoff **80 Ry**; `eps_default` \(10^{-7}\) |
| Valence | 16 e⁻ → \(n_{\mathrm{occ}} = 8\) |
| k / q | MP **2×1×1**, `k_time_reversal = false` → 2 \(k\) = 2 \(q\) |
| SCF | DIIS depth 6, tol \(10^{-7}\), mix 0.3, **no** level shift |
| G0W0 | finite-\(q\) inter-\(k\) spectral, **`mp2_ri`**, **`mp2_n_virt = 2`** |

**Excluded (non-goals of this slug)**

- Frequency-integrated / analytic-continuation GW (CP2K default RI-G0W0 path)
- k-diagonal-only multi-k G0W0 (historical foothold `periodic-gw-kpoints`)
- Periodic exact exchange without MT truncation; full HF hybrid GPW
- Open-shell UGW / URPA; BSE; evGW / qsGW / scGW
- Full virtual space / production cutoffs / large MP meshes (e2e ≤ 60 s)
- Live CP2K inside ctest; CUDA ports

## Implementation

Public surface only: concrete `cpu::gpw::GpwDft` members + `G0W0Result`.
Lookup constructors for basis/PP tables only. No public `*Config` bags, no
`make_*` / `build_*` factories.

```cpp
#include "atomiverse/cpu/gpw/g0w0.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/molrs.hpp"

using namespace ATV::cpu::gpw;

// bct diamond Si₄: a = 5.43 Å → Bohr; four GpwAtom with
// GaussianBasis(Si, "DZVP-MOLOPT-GTH"), GthPotential(Si, "GTH-PBE-q4")
// and fractional images of a_t = a/√2, c = a (see reg_*.cpp).

GpwDft m;
m.cell[0] = m.cell[1] = a_bohr / std::sqrt(2.0);
m.cell[2] = a_bohr;
m.cutoff_ry = 80.0;
m.n_occ = 8;
m.max_iter = 60;
m.tol = 1e-7;
m.mix = 0.3;
m.level_shift = 0.0;
m.eps_default = 1e-7;
m.xc = XcKind::PBE;
m.mp2_ri = true;          // required for multi-k G0W0
m.mp2_n_virt = 2;         // screening virtual cap (e2e budget)
m.k_mesh[0] = 2;
m.k_mesh[1] = 1;
m.k_mesh[2] = 1;
m.k_time_reversal = false;
m.diis_depth = 6;
m.build(atoms);
const GpwResult& scf = m.run(atoms);   // multi-k PBE SCF → C_k, eps_k

const G0W0Result& gw = m.g0w0(atoms);  // finite-q inter-k product
// gw.gap_ev [eV], gw.mf_gap_ev, gw.ip_ev, gw.ea_ev
// gw.qp_energy_ha / mf_energy_ha (report k of min direct QP gap)
// gw.n_kpts == 2, gw.ri, gw.converged
```

**Guards (throw, no silent fallback):** multi-k needs periodic cell and
`mp2_ri = true`; vanishing HOMO–LUMO gap on the mesh (< 1e-6 Ha); missing
`C_k` / unconverged SCF; nonperiodic multi-k. Γ (\(N_k=1\)) real path remains
available for smaller decks.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON \
        -DATV_BUILD_GPW=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_periodic-gw-finite-q
  ./build/regressions/reg_periodic-gw-finite-q
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/g0w0.h` | Public `G0W0Result` |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::g0w0` |
| `src/cpu/gpw/g0w0.cpp` | Γ + multi-k finite-\(q\) driver |
| `include/atomiverse/cpu/gpw/internal/mp2_kernels.h` | `spectral_g0w0_rhf_kpoints` (private) |
| `regressions/gpw/periodic-gw-finite-q/` | E2E product gate + `REFERENCE.md` |
| `tools/cp2k_oracle/inputs/periodic-gw-finite-q.inp` | Offline CP2K twin (never ctest) |
| `tests/cpu/gpw/` (`test_cpu_gpw_g0w0`) | Kernel Γ recovery (\(N_k=1\)) unit |

## Validation (vs CP2K)

Source:
[`regressions/gpw/periodic-gw-finite-q/REFERENCE.md`](../../../regressions/gpw/periodic-gw-finite-q/REFERENCE.md).
Oracle: offline **CP2K** input twin
`tools/cp2k_oracle/inputs/periodic-gw-finite-q.inp` (multi-k GPW RKS PBE,
MONKHORST-PACK 2 1 1, FULL_GRID / no symmetry ≈
`k_time_reversal=false`, ADDED_MOS 2, `&RI_G0W0` with CORR_MOS_OCC 8 /
CORR_MOS_VIRT 2). **Never** launch CP2K from ctest.

**Algorithm note (binding):** CP2K’s default path is **frequency-integrated
RI-GW** with Gaussian `RI_AUX` — not finite-\(q\) spectral G0W0 with MT
exchange. Absolute CP2K QP gaps are therefore **not** a hard digit twin.
Product gates are multi-k finite-\(q\) integrity + first-green absolute band
on ATV `gap_ev` (replace with fair CP2K when a same-algorithm twin exists).

**Setup (hard-coded from REFERENCE):**

| Item | Value |
| ---- | ----- |
| Crystal | diamond Si, bct 4-atom |
| Lattice | \(a = 5.43\) Å; \(a_t = a/\sqrt{2}\), \(c = a\) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 |
| XC / cutoff | PBE / 80 Ry |
| k-mesh | MP 2×1×1, equal weights, no time-reversal fold |
| Screening | RI-GPW finite-\(q\), \(n_{\mathrm{virt}} = 2\) |
| Provenance | first green 2026-07-26; harness wave 5 `dev` |

**Accuracy / product gates:**

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | ------ | --- | ----- |
| SCF | converged; `n_kpts = 2`; `C_k.size() = 2` | setup twin | hard | Product gate |
| \(E_{\mathrm{SCF}}\) | **−15.5751782413** Ha (7 iters) | same deck as multi-k RPA | setup | First-green integrity |
| KS min-direct gap | **1.561168** eV | diagnostic | finite > 0 | Product gate |
| **QP min-direct gap** | **7.529834** eV | **not** CP2K digit twin | **0.5 eV** | Product absolute band |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) / \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | +0.171486 / +0.448203 Ha | report \(k\) | HOMO < LUMO | Product gate |
| Newton | `converged == true` | — | hard | All orbits / all \(k\) |
| RI / virt | `ri == true`; `n_occ = 8`; `n_virt = 2` | ADDED_MOS 2 twin | hard | Product gate |
| Kernel Γ recovery (unit) | multi-k \(N_k=1\) ≡ real spectral QP | same orbitals | unit | `test_cpu_gpw_g0w0` |

Finite-\(q\) \(W\) **softens** the gap relative to the historical k-diagonal
foothold (~8.01 eV on the same class of deck). MT exchange + tiny virt space
can place absolute QP levels above zero — the **gap** is the hard product
observable, not the absolute HOMO/LUMO positions.

Offline re-pin (human only; never ctest):

```bash
# when a CP2K binary is available — diagnostic only for this algorithm pair
cp2k.psmp -i tools/cp2k_oracle/inputs/periodic-gw-finite-q.inp \
          -o /tmp/periodic-gw-finite-q.out
# do not invent absolute CP2K QP digits into the hard gate without a
# same-algorithm (finite-q spectral) twin
```

## Performance

Source: [`REFERENCE.md`](../../../regressions/gpw/periodic-gw-finite-q/REFERENCE.md),
the case `REFERENCE.md`
(wave 5, 2026-07-26).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| finite-q spectral G0W0 (this) | n/a | ~10 (SCF+G0W0, OMP=4) | **perf_exempt** | n/a | No fair twin: CP2K RI-G0W0 ≠ finite-\(q\) spectral + MT exchange; no CP2K binary on pin host |

| Item | Value |
| ---- | ----- |
| `perf_exempt` | **true** |
| Reason | Algorithm mismatch with CP2K default RI-G0W0; no fair wall twin. Wall budget of the regression is ≤ 60 s (e2e). |
| \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) | **n/a (exempt)** |
| `within_10x` | n/a (exempt) |

## Limitations

- **Screening virtuals capped** at `mp2_n_virt = 2` for the e2e budget —
  production gaps need a larger virtual space and higher cutoffs.
- **MT exchange** is a G=0-safe product choice, not full periodic HF;
  absolute QP levels can sit above zero on this tiny deck.
- **Not bit-parity** vs CP2K RI-G0W0; do not treat CP2K frequency-integrated
  QP numbers as the hard regression oracle for this path.
- Closed-shell RKS only; no UGW, no BSE, no self-consistent GW flavors.
- CPU GPW only (this loop); CUDA ports out of scope.
- Follow-ons: denser MP meshes, larger virt space, optional same-algorithm
  fair twin if CP2K exposes a finite-\(q\) spectral path; production
  diamond-Si₈ decks at higher cutoffs.

## References

- Offline pin: `regressions/gpw/periodic-gw-finite-q/REFERENCE.md`
- Product gate: `regressions/gpw/periodic-gw-finite-q/reg_periodic-gw-finite-q.cpp`
- CP2K twin input: `tools/cp2k_oracle/inputs/periodic-gw-finite-q.inp`
- Oracle rules: `tools/cp2k_oracle/README.md` (offline only; never ctest)
- Prior chain: Γ `periodic-gw-si` → k-diagonal `periodic-gw-kpoints` → **this**
  finite-\(q\) product; dielectric twin `periodic-rpa-finite-q`
- Molecular spectral G0W0 (PySCF loop): [g0w0-h2-sto3g](g0w0-h2-sto3g.md)
- Literature: Hybertsen & Louie, PRB **34**, 5390 (1986); Furche, JCP **114**,
  5982 (2001)
- Public headers: `include/atomiverse/cpu/gpw/g0w0.h`, `gpw_solver.h`
- Impl: `src/cpu/gpw/g0w0.cpp`; private kernel `spectral_g0w0_rhf_kpoints`
