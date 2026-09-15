# BSE@G0W0 — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `bse-h2-sto3g` |
| Regression | `regressions/scf/bse-h2-sto3g/` |
| Public API | `cpu::scf::RHF::bse(n_states, tda)` → `BSEResult` |
| Roadmap | [excited-states.md](../../roadmap/excited-states.md) § BSE@G0W0 |

## Summary

Closed-shell **Bethe–Salpeter equation (BSE)** on **spectral G0W0**
quasiparticle poles of a converged **RHF** reference. Static **direct-RPA**
screened Coulomb \(W\) is built from exact 4-centre MO ERIs (no density
fitting). Primary observable: lowest singlet excitation **S1 [eV]**. Public
path is `RHF::run(frame)` then `RHF::bse(n_states, tda)` — no `*Config` bag, no
factory. Regression binary: `reg_bse-h2-sto3g` under
`regressions/scf/bse-h2-sto3g/`. Twin is PySCF 2.14.0 exact-ERI + spectral
G0W0 BSE (not the DF-only stock `pyscf.gw.bse.BSE(gw.GW(...))` path). This is
the excited-state foothold after `reg_g0w0-h2-sto3g`.

## Method

**Level.** One-shot BSE@G0W0: QP energies from spectral \(G_0W_0\) replace
mean-field orbital energies in the particle–hole / hole–particle Casida
matrices; the ph attraction is the **static** dRPA screened interaction \(W\).

**Spin.** Closed-shell spatial MOs only (`RHF`). Singlet channel in v1.
Open-shell / triplet / spin-flip BSE is out of scope.

**Screening.** Static direct-RPA \(W\) with **QP** orbital energies (PySCF BSE
convention):

\[
\begin{aligned}
W &= v + v\,\Pi\,v,\\
\Pi &= 4\,(D^{-1}-4v_{\mathrm{ph}})^{-1},\\
D^{-1}_{ia} &= \varepsilon_i^{\mathrm{QP}}-\varepsilon_a^{\mathrm{QP}}.
\end{aligned}
\]

Exact 4-centre MO ERIs; no DF, no RI.

**Working equations** (singlet; PySCF `gw.bse` / Hillenbrand–Li–Zhu 2025):

\[
\begin{aligned}
A_{ia,jb} &= (\varepsilon_a^{\mathrm{QP}}-\varepsilon_i^{\mathrm{QP}})
            \delta_{ij}\delta_{ab} + 2(ia|jb) - W_{ij,ab}, \\
B_{ia,jb} &= 2(ia|jb) - W_{ib,aj}.
\end{aligned}
\]

Full BSE solves the Casida RPA problem on \((A,B)\); BSE-TDA diagonalizes \(A\)
only (\(B=0\)).

**Underlying G0W0.** Spectral / frequency-exact poles (`RHF::g0w0` twin of
`gw.gw_exact.GWExact` + dRPA) — same as `reg_g0w0-h2-sto3g`.

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\omega_{0n}\) | BSE excitation energies (ascending) | eV |
| S1 | lowest singlet root | eV |
| gap (embedded) | G0W0 QP gap via `BSEResult::gw` | eV |

**Excluded.** DF-BSE production path; dynamic \(W(\omega)\); open-shell /
triplet / SF-BSE; TDDFT/TDHF (different kernels); CUDA.

## Implementation

Frame in → RHF members → `run` → post-SCF `bse()`. Headers:
`include/atomiverse/cpu/scf/rhf.h`, `bse.h`, `g0w0.h`; consumer includes
`methods.h`.

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

cpu::scf::BSEResult bse = rhf.bse(/*n_states=*/1, /*tda=*/false);
// bse.energy_ev[0]  // S1 [eV]
// bse.gw.gap_ev     // underlying G0W0 QP gap [eV]
// bse.tda, bse.channel, bse.gw.converged
```

`BSEResult` fields: `energy_ev` (ascending, eV), `tda`, `channel` (singlet),
`gw` (`G0W0Result`). Full BSE is the default (`tda=false`); TDA is the optional
companion gate in the regression.

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
  cmake --build build --target reg_bse-h2-sto3g -j$(nproc)
  ./build/regressions/reg_bse-h2-sto3g
'
```

Source consumer: `regressions/scf/bse-h2-sto3g/reg_bse-h2-sto3g.cpp`.

## Validation (vs PySCF)

Source: [`regressions/scf/bse-h2-sto3g/REFERENCE.md`](../../../regressions/scf/bse-h2-sto3g/REFERENCE.md)
(offline pin only; PySCF never imported from CMake/ctest).

| Setting | Value |
| ------- | ----- |
| Geometry | H₂, \(r(\mathrm{H–H})=0.74\) Å; H at \((0,0,0)\), \((0,0,0.74)\) |
| Basis | STO-3G |
| Angular | Cartesian (`cart=True`; STO-3G has no d shells) |
| SCF | `conv_tol = 1e-12` |
| G0W0 | spectral / frequency-exact (`GWExact` + dRPA) |
| BSE | dense full diagonalization; \(n_{\mathrm{ov}}=1\) on H₂/STO-3G |
| Screening | static dRPA \(W\), QP orbital energies |
| Oracle | PySCF **2.14.0** (exact-ERI BSE@spectral-G0W0 twin) |
| Pin date | 2026-07-23 |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396 Ha | 1e-6 Ha | regression gate |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{G0W0}}\) | −0.596957673490 Ha | — | info (same as G0W0 pin) |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{G0W0}}\) | +0.689547305573 Ha | — | info |
| G0W0 QP gap | 35.007583892680 eV | **1e-4 eV** | sanity / secondary gate |
| **BSE S1** (full, exact ERI) | **25.956271542931 eV** | **1e-4 eV** | **primary gate** |
| BSE-TDA S1 | 26.809052342076 eV | 1e-4 eV | companion |
| DF stock twin `gw.bse.BSE` S1 | ~25.954 eV | — | info only; ~2 meV DF error |

Sanity (enforced in the regression binary): full BSE S1 < TDA S1; BSE S1 < QP
gap.

Oracle repro:

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py bse \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

## Performance

Source: same REFERENCE (fair_bench 2026-07-23; median of 5; `OMP_NUM_THREADS=1`).
Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\) unless exempted.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| exact-ERI BSE@spectral-G0W0 | 0.2233 | 0.0134 | **0.060** | ✓ | ATV binary = RHF + spectral G0W0 + BSE; PySCF = RKS(hf)+dRPA+GWExact+exact-ERI static BSE |

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
  --method bse --geom h2 --basis sto-3g --repeat 5 \
  -- ./build/regressions/reg_bse-h2-sto3g
```

**Twin caveats.** (1) Method class match is exact-ERI + spectral G0W0 only —
do not A/B against DF stock `gw.bse.BSE`. (2) Wall times include RHF SCF and
G0W0 on both sides of this fair_bench recipe; pure BSE diagonalization cost is
not split out. (3) Microsecond-scale reg binary on a 2-AO system (\(n_{\mathrm{ov}}=1\))
— ratio is informative for the foothold, not a large-basis throughput claim.

## Limitations

- **Closed-shell RHF / singlet only.** No UHF, ROHF, triplet, or spin-flip BSE.
- **Static \(W\) only.** Frequency-dependent / dynamical BSE is out of scope.
- **Exact ERI product path.** No DF/RI BSE in the public API; DF numbers appear
  only as offline twin info (~2 meV on this pin).
- **Not TDDFT/TDHF.** Different response kernels; use CIS/TDHF/TDDFT entry points
  for those methods.
- **CPU only.** CUDA BSE is ○.
- **Scope of pin.** H₂/STO-3G is a minimal foothold (\(n_{\mathrm{occ}}=n_{\mathrm{vir}}=1\));
  larger systems, denser ph spaces, and numerical robustness of static \(W\)
  construction are follow-on work.
- **Speed.** Current ratio \(\ll 10\times\) on the pin (handoff records
  **0.060**); no standing multi-system perf suite for BSE beyond this twin.

## References

- Offline pin: `regressions/scf/bse-h2-sto3g/REFERENCE.md`
- Perf (in REFERENCE): fair_bench numbers above; handoff
  the case `REFERENCE.md` records ratio **0.060**
- Roadmap: `docs/roadmap/excited-states.md` (BSE@G0W0 ✓ CPU)
- Public headers: `include/atomiverse/cpu/scf/bse.h`, `g0w0.h`, `rhf.h`
- Prerequisite pin: `regressions/scf/g0w0-h2-sto3g/REFERENCE.md` /
  [g0w0-h2-sto3g.md](g0w0-h2-sto3g.md)
- Hillenbrand, Li & Zhu, *J. Chem. Phys.* **162**, 174117 (2025) (PySCF BSE)
- *J. Comput. Chem.* **38**, 383 (2017) (molecular BSE / static \(W\))
- Hybertsen & Louie, *Phys. Rev. B* **34**, 5390 (1986) (GW quasiparticles)
- PySCF BSE module: `pyscf.gw.bse` (DF production path; exact-ERI twin in REFERENCE)
