# Multi-k unrestricted direct RPA (URPA@UKS) — diamond Si₄-bct / DZVP-MOLOPT-GTH

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `periodic-urpa-kpoints` |
| Regression | `regressions/gpw/periodic-urpa-kpoints/` |
| Public API | `cpu::gpw::GpwDft::direct_rpa(atoms)` → `cpu::gpw::RpaResult` (`cpu/gpw/rpa.h`); multi-k UKS via members `n_alpha` / `n_beta`, `k_mesh`, `mp2_ri`, `mp2_n_virt` |
| Roadmap | `docs/roadmap/dft.md` §10–11 (periodic RPA / URPA) |
| Oracle | offline CP2K twin `tools/cp2k_oracle/inputs/periodic-urpa-kpoints.inp` (never live in ctest) |

## Summary

**Spin-conserving direct RPA (dRPA) correlation on a multi-k GPW UKS
reference** — the unrestricted product path **URPA@UKS** after closed-shell
multi-k dRPA (`periodic-rpa-kpoints` / `periodic-rpa-finite-q`) and Γ URPA
(`periodic-urpa-si`). Screening is Coulomb-only Casida over the composite
α→α ⊕ β→β particle–hole space, with **finite-\(q\) inter-\(k\)** momentum
conservation on a closed Monkhorst–Pack mesh. Correlation energy is the
**Furche plasmon formula** Brillouin-zone-averaged over transfer \(q\).

Public path: construct `cpu::gpw::GpwDft`, set members (occupations, k-mesh,
RI + virtual cap), `build` → `run` (UKS SCF) → `direct_rpa(atoms)`. Result
fields are Hartree internally; Frame-facing energy for SCF remains eV on
`meta` when used through evaluators. Product gate:
`reg_periodic-urpa-kpoints`. Fair wall vs CP2K is **`perf_exempt`** (algorithm
class mismatch + no CP2K binary required on the pin host).

Related: Γ URPA `regressions/gpw/periodic-urpa-si`; multi-k RKS dRPA
`regressions/gpw/periodic-rpa-kpoints`; multi-k UGW
`regressions/gpw/periodic-ugw-kpoints`; molecular dRPA
[direct-rpa-h2-sto3g](direct-rpa-h2-sto3g.md) (PySCF loop, not this page).

## Method

**Level.** Direct random-phase approximation as a **standalone correlation
energy** on top of a converged multi-k UKS KS reference (dRPA@KS). Not TDHF
excitation RPA, not TDDFT Casida, not RPAX/SOSEX, not frequency-integrated
ACFDT product RPA as in stock CP2K `&RI_RPA`.

**Reference.** Multi-k **GPW UKS PBE** with explicit integer occupations
\(n_\alpha\), \(n_\beta\). Bloch MOs \(C_k^\alpha\), \(C_k^\beta\) and
eigenvalues \(\varepsilon_k^{\alpha/\beta}\) from DIIS SCF. This pin uses
closed-shell occupations \(n_\alpha = n_\beta = 8\) (multiplicity 1) so
URPA dual-gates closed-shell multi-k dRPA when orbitals match.

**Kernel (spin-conserving URPA).** Composite OV space = α block then β.
Coulomb-only (\(\alpha_{\mathrm{hf}}=0\)) Casida matrices; no closed-shell
factor of 2 — spins are explicit. For each transfer \(q\) on the closed MP
mesh, transitions are \(i_k^\sigma \to a_{k+q}^\sigma\) with Coulomb at
momentum transfer \(q\) via **RI-GPW** plane-wave products
(`mp2_ri = true` required for multi-k product).

**Correlation (Furche plasmon, multi-k):**

\[
E_c=\sum_q w_q\,\tfrac12\Bigl(\sum_s\Omega_{q,s}-\mathrm{Tr}\,A_q\Bigr),
\]

with \(\Omega_{q,s}\) the positive eigenvalues of the Hermitian reduction of
\((A,B)_q\). Total energy reported is **dRPA@KS**
\(E_{\mathrm{SCF}}+E_c\) (no EXX replacement — periodic exact exchange
remains blocked).

**Spectrum length (this pin).** \(N_k=2\), \(n_{\mathrm{ov}}^\alpha = n_{\mathrm{ov}}^\beta = 8\cdot 2 = 16\),
composite \(n_{\mathrm{ov}}=32\); finite-\(q\) product concatenates roots over
all ordered \((k,q)\) pairs → \(n_\Omega = N_k^2 \cdot 32 = 128\).

**Dual-gate identity.** When \(C^\alpha \approx C^\beta\) and
\(n_\alpha=n_\beta\), multi-k URPA \(E_c\) **bit-matches** multi-k RKS dRPA
on the same geometry/basis/cutoff (first-green
`periodic-rpa-kpoints` \(E_c\)).

**Key exclusions.**

- No frequency-axis / minimax RI-RPA twin as the hard accuracy gate (CP2K
  default path ≠ plasmon formula).
- No EXX total-energy assembly; no RPAX / SOSEX / GW self-energy (see UGW).
- Virtual space capped for e2e wall (`mp2_n_virt = 2` per spin).
- No MPI scaling claim; single-rank pin.
- Not molecular all-electron URPA (Gaussian SCF / PySCF loop).

## Implementation

Public surface only (`GpwDft` members + `GpwAtom` lookup kinds; **no** public
`*Config`, no `make_*` / `build_*` factories). Headers:
`include/atomiverse/cpu/gpw/gpw_solver.h`, `rpa.h`, `gaussian_basis.h`,
`gth_potential.h`. Kernels private under `cpu/gpw/internal/`
(`direct_rpa_uhf_energy_kpoints`).

```cpp
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_solver.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/gpw/rpa.h"
#include "atomiverse/molrs.hpp"

// diamond Si body-centred tetragonal 4-atom cell, a = 5.43 Å
// (same deck as periodic-rpa-kpoints / periodic-ugw-kpoints)
std::vector<cpu::gpw::GpwAtom> atoms = /* Si ×4: DZVP-MOLOPT-GTH + GTH-PBE-q4 */;

cpu::gpw::GpwDft m;
m.cell[0] = m.cell[1] = a_bohr / std::sqrt(2.0);
m.cell[2] = a_bohr;
m.cutoff_ry = 80.0;
m.n_alpha = 8;
m.n_beta  = 8;
m.xc = cpu::gpw::XcKind::PBE;
m.tol = 1e-7;
m.mix = 0.3;
m.level_shift = 0.0;
m.diis_depth = 6;
m.k_mesh[0] = 2;
m.k_mesh[1] = 1;
m.k_mesh[2] = 1;
m.k_time_reversal = false;   // full grid, equal weights w_k = 1/2
m.mp2_ri = true;             // multi-k product requires RI-GPW
m.mp2_n_virt = 2;            // per-spin virtual cap (e2e ≤ 60 s)
m.build(atoms);
const cpu::gpw::GpwResult& scf = m.run(atoms);   // multi-k UKS
const cpu::gpw::RpaResult& r = m.direct_rpa(atoms);  // multi-k URPA, no throw
// r.e_corr, r.e_scf, r.e_total [Ha]; r.tr_A; r.omega_ha; r.n_kpts; r.ri
```

`RpaResult` (URPA multi-k): `e_corr`, `e_scf`, `e_total`, `tr_A` [Ha];
`n_occ` / `n_virt` are **spin-orbital** totals (\(n_\alpha+n_\beta\),
\(n_{\mathrm{virt}}^\alpha+n_{\mathrm{virt}}^\beta\)); `n_kpts`, `ri`;
`omega_ha` positive plasmon roots (ascending, length \(N_k^2\cdot
(n_{\mathrm{ov}}^\alpha+n_{\mathrm{ov}}^\beta)\)).

Guards: `std::logic_error` without converged `run()`; multi-k product
requires `mp2_ri` and non-vanishing HOMO–LUMO structure consistent with
other GPW post-SCF paths. OT references (no canonical virtuals) are out of
scope for this entry point.

```bash
# configure (login shell + Lmod — see AGENTS.md)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build -j$(nproc) --target atomiverse reg_periodic-urpa-kpoints
  ./build/regressions/reg_periodic-urpa-kpoints
  # or: cmake --build build --target run_reg_periodic-urpa-kpoints
'
```

Unit tier (no full SCF): `tests/cpu/gpw/test_cpu_gpw_rpa.cpp` —
`UhfKpointsNk1DualGatesRhf`, `UhfFiniteQTwoKSpectrumShape`.

## Validation (vs CP2K / dual-gate)

Source: [`regressions/gpw/periodic-urpa-kpoints/REFERENCE.md`](../../../regressions/gpw/periodic-urpa-kpoints/REFERENCE.md)
(offline pin only; CP2K never imported from CMake/ctest).

| Setting | Value |
| ------- | ----- |
| Crystal | diamond Si, **bct 4-atom** (a_t = a/√2, c = a; a = **5.43 Å**) |
| Positions (frac) | (0,0,0), (½,½,½), (½,0,¼), (0,½,¾) |
| Basis / PP | **DZVP-MOLOPT-GTH** / **GTH-PBE-q4** |
| XC / spin | **PBE UKS**, \(n_\alpha=n_\beta=8\) (16 valence e⁻) |
| Cutoff | **80 Ry** |
| k-mesh | **MP 2×1×1**, `k_time_reversal=false` → 2 k, \(w=1/2\) |
| SCF | DIIS 6, tol 1e-7, mix 0.3, **no level_shift** |
| RPA | finite-\(q\) inter-\(k\) RI-GPW URPA, **n_virt = 2 per spin** |
| Oracle input | `tools/cp2k_oracle/inputs/periodic-urpa-kpoints.inp` |
| Pin date | 2026-07-26 |

**Oracle status.** No CP2K binary required on the pin host. CP2K's default
`&RI_RPA` path is **frequency-integrated** RI-RPA with Gaussian `RI_AUX` and
EXX total-energy assembly — **not** the exact finite-\(q\) plasmon formula on
PW-RI \((ia|jb)\). Absolute CP2K \(E_c\) is therefore a near-twin diagnostic
only, not the hard accuracy gate.

### Binding gates

1. **Hard multi-k URPA:** SCF converged; `C_k.size()==C_k_beta.size()==n_kpts`;
   `direct_rpa` no throw; `n_kpts==2`; `ri`; \(E_c\) finite and **strictly
   negative**; all \(\Omega>0\); \(e_{\mathrm{total}} \equiv e_{\mathrm{scf}}+e_{\mathrm{corr}}\);
   `n_occ=16`, `n_virt=4`; \(\|omega\|=128\).
2. **Kernel unit:** multi-k UHF at \(N_k=1\) with \(C^\alpha=C^\beta\) recovers
   real closed-shell / Γ URPA plasmon (`test_cpu_gpw_rpa`).
3. **Dual-gate band** vs RKS multi-k first-green \(E_c\)
   (`periodic-rpa-kpoints`): \(|E_c - (-0.0736273682)|\le 5\times 10^{-3}\) Ha.

### Hard-coded pins (first green, 2026-07-26, OMP=8)

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{SCF}}\) (2k UKS-PBE, 80 Ry) | **−15.5751782413** Ha | setup | 7 iters; ≡ RKS multi-k twin; \(\langle S^2\rangle=8\) |
| **\(E_c\) (2k URPA, n_virt=2/spin)** | **−0.0736273682** Ha | **5e-3 Ha** | **primary dual-gate** (bit-match RKS multi-k) |
| Tr \(A\) (composite BZ avg) | 17.508325 Ha | diagnostic | |
| \(n_{\mathrm{occ}}\) / \(n_{\mathrm{virt}}\) / \(n_\Omega\) | 16 / 4 / 128 | exact | spin-orbital counts |
| Wall (SCF + URPA) | ~2.7 s | ≤ 60 s | e2e budget |
| Dual-gate ref (RKS multi-k dRPA) | −0.0736273682 Ha | same band | `periodic-rpa-kpoints` |

## Performance

Source: REFERENCE + the case `REFERENCE.md`.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| multi-k URPA Si₄-bct | — | ~2.7 | **perf_exempt** | — | CP2K RI-RPA ≠ finite-\(q\) plasmon; EXX block changes total-energy path; no CP2K binary required on pin host |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\) when a fair algorithm twin
and binary exist; this slug is explicitly **`perf_exempt`**. Never launch
CP2K from ctest.

## Limitations

- **Algorithm class vs stock CP2K.** Hard gate is dual-gate / setup integrity,
  not absolute minimax RI-RPA \(E_c\). Re-pin absolute CP2K only after a
  documented fair twin (same plasmon formula or agreed near-twin residual).
- **Closed-shell occupations on this deck.** The product exercises multi-k
  URPA machinery with \(n_\alpha=n_\beta\); open-shell multi-k URPA
  (\(n_\alpha\neq n_\beta\)) is the natural extension, not this regression’s
  dual-gate.
- **Virtual truncation.** `mp2_n_virt=2` is an e2e wall choice; full virtual
  space and denser k-meshes are follow-on cost/accuracy work.
- **RI-GPW only for multi-k.** Non-RI multi-k URPA is not a product path.
- **No EXX / hybrid RPA total energy.** dRPA@KS only until periodic HF/hybrid
  GPW unblocks.
- **CPU only.** CUDA GPW / URPA is ○ / blocked in the CP2K loop map.
- **Scope of pin.** Tiny bct Si₄ at 80 Ry is a capability foothold, not a
  production diamond-Si phonon/RPA suite.

## References

- Offline pin: `regressions/gpw/periodic-urpa-kpoints/REFERENCE.md`
- Regression binary: `regressions/gpw/periodic-urpa-kpoints/reg_periodic-urpa-kpoints.cpp`
- CP2K input twin: `tools/cp2k_oracle/inputs/periodic-urpa-kpoints.inp`
- Oracle rules: `tools/cp2k_oracle/README.md`
- Roadmap: `docs/roadmap/dft.md` (periodic RPA / URPA)
- Public headers: `include/atomiverse/cpu/gpw/rpa.h`, `gpw_solver.h`
- Unit kernels: `tests/cpu/gpw/test_cpu_gpw_rpa.cpp`
- Sibling slugs: `periodic-urpa-si`, `periodic-rpa-kpoints`,
  `periodic-rpa-finite-q`, `periodic-ugw-kpoints`
- Furche, *J. Chem. Phys.* **114**, 5982 (2001), DOI 10.1063/1.1353585
- Ren et al., *New J. Phys.* **14**, 053020 (2012)
- Del Ben, Hutter, VandeVondele, *J. Chem. Theory Comput.* **8**, 4177 (2012)
  (GPW \((ia|jb)\) / WF correlation route)
