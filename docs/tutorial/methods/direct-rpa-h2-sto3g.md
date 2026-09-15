# Direct RPA — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `direct-rpa-h2-sto3g` |
| Regression | `regressions/scf/direct-rpa-h2-sto3g/` |
| Public API | `cpu::scf::RHF::direct_rpa()` → `DirectRPAResult` |
| Roadmap | [excited-states.md](../../roadmap/excited-states.md) § Direct RPA |

## Summary

Closed-shell **direct RPA (dRPA)** correlation energy on a converged **RHF**
reference: Coulomb-only Casida \(A,B\) (\(\alpha_{\mathrm{hf}}=0\)) and the
**Furche plasmon formula** on the full dRPA spectrum. Primary observable is
\(E_{\mathrm{c}}^{\mathrm{dRPA}}\) [Ha]; total energy and poles are reported
for gates. Public path is `RHF::run(frame)` then `RHF::direct_rpa()` — no
`*Config` bag, no factory. Regression binary: `reg_direct-rpa-h2-sto3g` under
`regressions/scf/direct-rpa-h2-sto3g/`. Twin is the exact MO-ERI plasmon rebuild
on PySCF 2.14.0 RHF orbitals (not stock DF + Legendre `gw.rpa.RPA`). Same dRPA
poles feed the G0W0 screening twin (`g0w0-h2-sto3g`).

## Method

**Level.** Direct random-phase approximation as a **standalone correlation
energy** (roadmap Direct RPA). Not TDHF excitation RPA (`RHF::tdhf`), not
TDDFT Casida, not RPAX/SOSEX.

**Spin.** Closed-shell spatial MOs only (`RHF`). Open-shell / UHF is out of
scope.

**Kernel.** Singlet spin-adapted Casida matrices with **Coulomb only**
(\(\alpha_{\mathrm{hf}}=0\)):

\[
A_{ia,jb} = (\varepsilon_a - \varepsilon_i)\,\delta_{ij}\delta_{ab}
  + 2\,(ia|jb),\qquad
B_{ia,jb} = 2\,(ia|jb).
\]

**Correlation.** Furche plasmon formula (exact; no frequency quadrature, no
density fitting):

\[
E_{\mathrm{c}}^{\mathrm{dRPA}}
  = \tfrac12\Bigl(\sum_s \Omega_s - \mathrm{Tr}\,A\Bigr),
\]

with \(\Omega_s\) the positive eigenvalues of the Hermitian reduction
\((A-B)^{1/2}(A+B)(A-B)^{1/2}\). Equivalent to the imaginary-frequency
integral \(\frac{1}{2\pi}\int_0^\infty\mathrm{d}\omega\,
\mathrm{Tr}[\ln(1-\chi_0 v)+\chi_0 v]\) in the non-DF limit.

**Spectrum.** Full dRPA spectrum:
\(n_{\mathrm{states}} = n_{\mathrm{occ}}\cdot n_{\mathrm{vir}} = 1\) on
H₂/STO-3G.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(E_{\mathrm{c}}^{\mathrm{dRPA}}\) | plasmon correlation energy | Ha |
| \(E_{\mathrm{tot}}\) | \(E_{\mathrm{RHF}} + E_{\mathrm{c}}\) | Ha |
| \(\Omega_s\) | dRPA excitation energies (ascending) | Ha |
| \(\mathrm{Tr}\,A\) | diagnostic for plasmon identity | Ha |

**Excluded.** Density-fitted RPA; imaginary-axis Legendre quadrature (stock
PySCF `gw.rpa.RPA`); RPAX / SOSEX / ACFDT product paths; open-shell dRPA;
CUDA.

## Implementation

Frame in → RHF members → `run` → post-SCF `direct_rpa()`. Headers:
`include/atomiverse/cpu/scf/rhf.h`, `rpa.h`; impl `src/cpu/scf/rpa.cpp`.

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

cpu::Frame f(2);
// H2, r = 0.74 Å on +z (REFERENCE.md); type = Z
f.span(kAtom_z)[1] = 0.74;
f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
f.box = cpu::Box::cubic(20.0);

cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.tol = 1e-12;
rhf.max_iter = 100;
rhf.run(f);

cpu::scf::DirectRPAResult rpa = rhf.direct_rpa();
// rpa.e_corr, rpa.e_tot [Hartree]; rpa.omega_ha; rpa.tr_A
```

`DirectRPAResult` fields: `e_corr`, `e_tot`, `omega_ha`, `tr_A`. Requires prior
successful `run()`; throws if called without SCF result, on non-restricted
reference, or with \(n_{\mathrm{occ}}=0\) or \(n_{\mathrm{vir}}=0\).

```bash
# configure (login shell + Lmod — see AGENTS.md)
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON \
        -DATV_BUILD_REGRESSIONS=ON -DATV_BUILD_CUDA=OFF \
        -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_direct-rpa-h2-sto3g -j$(nproc)
  ./build/regressions/reg_direct-rpa-h2-sto3g
'
```

Source consumer: `regressions/scf/direct-rpa-h2-sto3g/reg_direct-rpa-h2-sto3g.cpp`.

## Validation (vs PySCF)

Source: [`regressions/scf/direct-rpa-h2-sto3g/REFERENCE.md`](../../../regressions/scf/direct-rpa-h2-sto3g/REFERENCE.md)
(offline pin only; PySCF never imported from CMake/ctest).

| Setting | Value |
| ------- | ----- |
| Geometry | H₂, \(r(\mathrm{H–H})=0.74\) Å; H at \((0,0,0)\), \((0,0,0.74)\) |
| Basis | STO-3G |
| Angular | Cartesian (`cart=True`; STO-3G has no d shells) |
| SCF | `conv_tol = 1e-12` |
| Response | full dRPA spectrum, \(n_{\mathrm{states}}=1\) |
| Correlation | Furche plasmon (exact MO ERIs; no DF, no frequency grid) |
| Oracle | PySCF **2.14.0** (plasmon rebuild on RHF orbitals + exact MO ERIs) |
| Pin date | 2026-07-23 |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396 Ha | 1e-6 Ha | regression gate |
| dRPA \(\Omega_1\) | **1.570852128278** Ha (42.745063997913 eV) | 1e-6 Ha | same pole as G0W0 screening |
| \(\mathrm{Tr}\,A\) | **1.612118275776** Ha | 1e-6 Ha | diagnostic |
| **\(E_{\mathrm{c}}^{\mathrm{dRPA}}\)** | **−0.020633073749** Ha | **1e-6 Ha** | **primary gate** |
| \(E_{\mathrm{tot}}^{\mathrm{dRPA}}\) | **−1.137392381145** Ha | 1e-6 Ha | \(E_{\mathrm{RHF}}+E_{\mathrm{c}}\) |
| DF twin (`gw.rpa.RPA`, `nw=100`) | −0.020624247051 Ha | — | info; ~9e-6 Ha DF residual |
| RMP2 (context) | −0.013138073590 Ha | — | sanity: \(E_c^{\mathrm{dRPA}} < E_c^{\mathrm{MP2}}\) |

Additional gates: plasmon identity
\(|E_c - \tfrac12(\Omega_1-\mathrm{Tr}A)| < 10^{-12}\) Ha; \(e_{\mathrm{tot}}\)
from pieces within \(10^{-12}\) Ha.

Oracle repro:

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py direct-rpa \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

## Performance

Source: same REFERENCE (fair_bench 2026-07-23; median of 5; `OMP_NUM_THREADS=1`).
Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\) unless exempted.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| exact MO-ERI plasmon dRPA | 0.0217 | 0.0165 | **0.760** | ✓ | ATV = RHF + dRPA plasmon; PySCF twin = RHF + exact MO-ERI plasmon (not DF `gw.rpa.RPA`) |

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method direct_rpa --geom h2 --basis sto-3g --repeat 5 \
  -- ./build/regressions/reg_direct-rpa-h2-sto3g
```

**Twin caveats.** (1) Method class match is non-DF plasmon only — do not A/B
against stock DF + Legendre `gw.rpa.RPA` for the accuracy gate. (2) Wall times
include RHF SCF on both sides of this fair_bench recipe; pure post-SCF dRPA
cost is not split out. (3) Microsecond-scale reg binary on a 2-AO system —
ratio is informative for the foothold, not a large-basis throughput claim.

## Limitations

- **Closed-shell RHF only.** No UHF/UKS dRPA, no ROHF.
- **Exact-ERI plasmon product path only.** No DF-RPA or imaginary-axis
  quadrature in the public API; DF numbers appear only as offline twin info.
- **No exchange in the RPA kernel.** RPAX, SOSEX, and ACFDT-RPA are roadmap-open
  ([excited-states.md](../../roadmap/excited-states.md)).
- **Correlation energy, not excitations as product.** TDHF/TDDFT roots are
  separate entry points; this returns \(E_c\) and poles as diagnostics for the
  plasmon formula.
- **CPU only.** CUDA Direct RPA is ○.
- **Scope of pin.** H₂/STO-3G is a minimal foothold (1 occupied, 1 virtual);
  larger systems / basis sets and dense RPA spectra are follow-on work.
- **Speed.** Current ratio \(\ll 10\times\) on the pin (0.760); no standing
  multi-system perf suite for dRPA beyond this twin.

## References

- Offline pin: `regressions/scf/direct-rpa-h2-sto3g/REFERENCE.md`
- Perf (in REFERENCE): fair_bench numbers above; handoff
  the case `REFERENCE.md` records ratio **0.760**
- Roadmap: `docs/roadmap/excited-states.md` (Direct RPA ✓ CPU)
- Public headers: `include/atomiverse/cpu/scf/rpa.h`, `rhf.h`
- Impl: `src/cpu/scf/rpa.cpp`
- G0W0 twin (same dRPA poles): `regressions/scf/g0w0-h2-sto3g`,
  [g0w0-h2-sto3g.md](g0w0-h2-sto3g.md)
- Furche, *J. Chem. Phys.* **114**, 5982 (2001), DOI 10.1063/1.1353585
- Ren et al., *New J. Phys.* **14**, 053020 (2012)
- Zhu & Chan, *J. Chem. Theory Comput.* **17**, 727 (2021); PySCF `gw.rpa`
  (DF + imaginary-axis quadrature companion)
