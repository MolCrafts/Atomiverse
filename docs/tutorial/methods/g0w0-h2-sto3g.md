# G0W0 — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `g0w0-h2-sto3g` |
| Regression | `regressions/scf/g0w0-h2-sto3g/` |
| Public API | `cpu::scf::RHF::g0w0()` → `G0W0Result` |
| Roadmap | [excited-states.md](../../roadmap/excited-states.md) §5 G0W0 |

## Summary

Closed-shell **spectral G0W0** quasiparticle (QP) energies on a converged **RHF**
reference: direct-RPA (dRPA) screening of \(W\), frequency-exact diagonal
correlation self-energy, non-linear Newton QP equation per MO. Primary
observable on this pin is the **HOMO–LUMO QP gap [eV]** (also IP / EA). Public
path is `RHF::run(frame)` then `RHF::g0w0()` — no `*Config` bag, no factory.
Regression binary: `reg_g0w0-h2-sto3g` under `regressions/scf/g0w0-h2-sto3g/`.
Twin is PySCF 2.14.0 `gw.gw_exact.GWExact` + `tdscf.dRPA` on HF density
(`dft.RKS` with `xc='hf'`), not the analytic-continuation `gw.GW` path.

## Method

**Level.** One-shot \(G_0W_0\): Green’s function and screened interaction built
from the RHF mean-field solution; no self-consistency in \(G\) or \(W\).

**Spin.** Closed-shell spatial MOs only (`RHF`). Open-shell / UHF / UKS G0W0 is
out of scope.

**Screening.** Direct RPA (no exchange in the RPA kernel; \(\alpha_{\mathrm{hf}}=0\)
in the dRPA channel). Full dRPA spectrum for the pin: \(n_{\mathrm{states}} =
n_{\mathrm{occ}}\cdot n_{\mathrm{vir}} = 1\) on H₂/STO-3G.

**Self-energy.** Spectral (frequency-exact) sum over RPA poles for the diagonal
correlation self-energy \(\Sigma_c(\omega)\). Broadening \(\eta = 10^{-8}\) Ha
in the spectral representation (REFERENCE).

**Quasiparticle equation.** For each MO \(p\),

\[
\varepsilon_p^{\mathrm{QP}} = \varepsilon_p^{\mathrm{HF}} +
  \Sigma_{c,pp}(\varepsilon_p^{\mathrm{QP}}),
\]

solved by non-linear Newton (tolerance \(10^{-8}\) Ha). **Not** the linearized
\(Z\)-factor form.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varepsilon_p^{\mathrm{QP}}\) | QP orbital energies | Ha |
| gap | \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}} - \varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) | eV |
| IP | \(-\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) | eV |
| EA | \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | eV |

**Excluded.** Analytic-continuation GW (`GWAC` / default `gw.GW`); evGW / qsGW /
scGW; BSE; RPA total energies; open-shell G0W0; CUDA.

## Implementation

Frame in → RHF members → `run` → post-SCF `g0w0()`. Headers:
`include/atomiverse/cpu/scf/rhf.h`, `g0w0.h`; impl `src/cpu/scf/g0w0.cpp`.

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

cpu::scf::G0W0Result gw = rhf.g0w0();
// gw.gap_ev, gw.ip_ev, gw.ea_ev, gw.qp_energy_ha, gw.converged
```

`G0W0Result` fields: `qp_energy_ha`, `mf_energy_ha`, `gap_ev`, `ip_ev`, `ea_ev`,
`n_occ`, `converged`. Requires prior successful `run()`; throws if called
without SCF result, on non-restricted reference, or with \(n_{\mathrm{occ}}=0\)
or \(n_{\mathrm{vir}}=0\).

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
  cmake --build build --target reg_g0w0-h2-sto3g -j$(nproc)
  ./build/regressions/reg_g0w0-h2-sto3g
'
```

Source consumer: `regressions/scf/g0w0-h2-sto3g/reg_g0w0-h2-sto3g.cpp`.

## Validation (vs PySCF)

Source: [`regressions/scf/g0w0-h2-sto3g/REFERENCE.md`](../../../regressions/scf/g0w0-h2-sto3g/REFERENCE.md)
(offline pin only; PySCF never imported from CMake/ctest).

| Setting | Value |
| ------- | ----- |
| Geometry | H₂, \(r(\mathrm{H–H})=0.74\) Å; H at \((0,0,0)\), \((0,0,0.74)\) |
| Basis | STO-3G |
| Angular | Cartesian (`cart=True`; STO-3G has no d shells) |
| SCF | `conv_tol = 1e-12` |
| Screening | full dRPA spectrum, \(n_{\mathrm{states}}=1\) |
| QP solve | Newton, tol \(10^{-8}\) Ha; \(\eta=10^{-8}\) Ha |
| Oracle | PySCF **2.14.0** (`gw.gw_exact.GWExact` + `tdscf.dRPA` on `RKS(xc='hf')`) |
| Pin date | 2026-07-23 |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396 Ha | 1e-6 Ha | regression gate |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{HF}}\) | −0.578553859832 Ha | — | mean-field (info) |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{HF}}\) | +0.671143491914 Ha | — | mean-field (info) |
| HF gap | 34.005997328943 eV | — | info |
| dRPA \(\Omega_1\) | 1.570852128278 Ha (42.745063997913 eV) | — | info |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{G0W0}}\) | **−0.596957673490** Ha | 1e-5 Ha | product |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{G0W0}}\) | **+0.689547305573** Ha | 1e-5 Ha | product |
| **QP gap** | **35.007583892680 eV** | **1e-4 eV** | **primary gate** |
| IP \(-\varepsilon_H^{\mathrm{QP}}\) | **16.244045825851 eV** | 1e-4 eV | product |
| EA \(\varepsilon_L^{\mathrm{QP}}\) | **18.763538066829 eV** | 1e-4 eV | product |
| AC twin gap (`gw.GW`) | 35.007177297967 eV | — | ~0.4 meV off spectral; not the gate |

Sanity: G0W0 gap > HF gap on this system (enforced in the regression binary).

Oracle repro:

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py g0w0 \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

## Performance

Source: same REFERENCE (fair_bench 2026-07-23; median of 5; `OMP_NUM_THREADS=1`).
Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\) unless exempted.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| spectral G0W0 | 0.1148 | 0.0125 | **0.109** | ✓ | ATV binary = RHF + spectral G0W0; PySCF = RKS(hf)+dRPA+GWExact |

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method g0w0 --geom h2 --basis sto-3g --repeat 5 \
  -- ./build/regressions/reg_g0w0-h2-sto3g
```

**Twin caveats.** (1) Method class match is spectral / exact-frequency only —
do not A/B against AC `gw.GW`. (2) Wall times include RHF SCF on both sides of
this fair_bench recipe; pure post-SCF QP cost is not split out. (3) Microsecond
reg binary on a 2-AO system — ratio is informative for the foothold, not a
large-basis throughput claim.

## Limitations

- **Closed-shell RHF only.** No UHF/UKS G0W0, no ROHF.
- **Spectral product path only.** No analytic-continuation GW in the public API;
  AC numbers appear only as offline twin info.
- **No self-consistency.** evGW, qsGW, and scGW are roadmap-open
  ([excited-states.md](../../roadmap/excited-states.md) §5).
- **No BSE / RPA total energy** from this entry point; dRPA exists only as the
  screening step inside G0W0.
- **CPU only.** CUDA G0W0 is ○.
- **Scope of pin.** H₂/STO-3G is a minimal foothold (1 occupied, 1 virtual);
  larger systems / basis sets and numerical robustness under denser RPA spectra
  are follow-on work.
- **Speed.** Current ratio \(\ll 10\times\) on the pin; no standing multi-system
  perf suite for G0W0 beyond this twin.

## References

- Offline pin: `regressions/scf/g0w0-h2-sto3g/REFERENCE.md`
- Perf (in REFERENCE; no separate `` row file in tree at
  write time): fair_bench numbers above; handoff
  the case `REFERENCE.md` records ratio **0.109**
- Roadmap: `docs/roadmap/excited-states.md` (G0W0 ✓ CPU)
- Public headers: `include/atomiverse/cpu/scf/g0w0.h`, `rhf.h`
- Impl: `src/cpu/scf/g0w0.cpp`
- Hybertsen & Louie, *Phys. Rev. B* **34**, 5390 (1986), DOI
  10.1103/PhysRevB.34.5390
- van Setten et al., *J. Chem. Theory Comput.* **11**, 5665 (2015)
- Zhu & Chan, *J. Chem. Theory Comput.* **17**, 727 (2021) (AC path companion)
- PySCF spectral: Berkelbach et al., `pyscf.gw.gw_exact`
