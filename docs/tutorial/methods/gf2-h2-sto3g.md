# One-shot GF2 (G0F2) — H₂ / STO-3G

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gf2-h2-sto3g` |
| Regression | `regressions/scf/gf2-h2-sto3g/` |
| Public API | `cpu::scf::RHF::gf2()` → `GF2Result` |
| Roadmap | [excited-states.md](../../roadmap/excited-states.md) § many-body / GF2 |

## Summary

Closed-shell **one-shot second-order Green’s function** (G0F2 / GF2) quasiparticle
energies on a converged **RHF** reference: the 2h1p + 2p1h second-order self-energy
is built in the auxiliary (Lehmann) representation from the HF Green’s function,
then the Dyson equation is solved **once** with the HF Fock matrix
\(F = \mathrm{diag}(\varepsilon_{\mathrm{HF}})\). Primary observable is the
**HOMO–LUMO QP gap [eV]** (also IP / EA); Galitskii–Migdal correlation energy is
reported and equals canonical MP2 on a pure HF reference. Public path:
`RHF::run(frame)` then `RHF::gf2()` — no `*Config` bag, no factory. Regression
binary: `reg_gf2-h2-sto3g` under `regressions/scf/gf2-h2-sto3g/`. Twin is PySCF
2.14.0 `agf2.RAGF2` **one-shot** path (SE from `init_gf()` →
`se.get_greens_function(F_HF)`), not full self-consistent AGF2.

## Method

**Level.** One-shot GF2 (G0F2): self-energy from the HF Green’s function; single
Dyson solve; no outer density/Fock self-consistency.

**Spin.** Closed-shell spatial MOs only (`RHF`). Open-shell / UHF / ROHF GF2 is
out of scope for this product path.

**Self-energy.** Second-order \(\Sigma^{(2)}\) in the auxiliary representation
(2h1p + 2p1h couplings). Construction matches PySCF AGF2 with
`os_factor = ss_factor = 1` and `weight_tol = 1e-11` to drop uncoupled poles
(REFERENCE).

**Dyson equation.** Single Hermitian diagonalization of the extended Fock matrix

\[
\begin{pmatrix}
F & V \\
V^\dagger & \varepsilon_{\mathrm{aux}}
\end{pmatrix},
\]

where \(F = \mathrm{diag}(\varepsilon_{\mathrm{HF}})\) and \(V\) couples physical
MOs to auxiliary poles. Chemical potential by binary search for \(N = 2\)
electrons (RHF occupancy 2).

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(\varepsilon_p^{\mathrm{QP}}\) | QP orbital energies (physical poles) | Ha |
| gap | \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}} - \varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) | eV |
| IP | \(-\varepsilon_{\mathrm{HOMO}}^{\mathrm{QP}}\) | eV |
| EA | \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{QP}}\) | eV |
| \(E_{\mathrm{corr}}\) | Galitskii–Migdal from one-shot SE (= MP2 on HF ref.) | Ha |

**Excluded.** G0W0 (screened \(W\); bare \(v\) here); ADC(2); self-consistent
AGF2 (outer density/Fock loop — separate roadmap row); open-shell GF2; CUDA.

## Implementation

Frame in → RHF members → `run` → post-SCF `gf2()`. Headers:
`include/atomiverse/cpu/scf/rhf.h`, `gf2.h`; impl `src/cpu/scf/gf2.cpp`.

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

cpu::scf::GF2Result gf = rhf.gf2();
// gf.gap_ev, gf.ip_ev, gf.ea_ev, gf.qp_energy_ha, gf.e_corr, gf.converged
```

`GF2Result` fields: `qp_energy_ha`, `mf_energy_ha`, `pole_energy_ha`, `gap_ev`,
`ip_ev`, `ea_ev`, `e_corr`, `e_tot`, `n_occ`, `converged`. Requires prior
successful `run()` on a closed-shell RHF reference.

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
  cmake --build build --target reg_gf2-h2-sto3g -j$(nproc)
  ./build/regressions/reg_gf2-h2-sto3g
'
```

Source consumer: `regressions/scf/gf2-h2-sto3g/reg_gf2-h2-sto3g.cpp`.

## Validation (vs PySCF)

Source: [`regressions/scf/gf2-h2-sto3g/REFERENCE.md`](../../../regressions/scf/gf2-h2-sto3g/REFERENCE.md)
(offline pin only; PySCF never imported from CMake/ctest).

| Setting | Value |
| ------- | ----- |
| Geometry | H₂, \(r(\mathrm{H–H})=0.74\) Å; H at \((0,0,0)\), \((0,0,0.74)\) |
| Basis | STO-3G |
| Angular | Cartesian (`cart=True`; STO-3G has no d shells) |
| SCF | RHF, `conv_tol = 1e-12` |
| Self-energy | AGF2 one-shot; `os_factor = ss_factor = 1`; `weight_tol = 1e-11` |
| Dyson | single extended-Fock diagonalization; \(\mu\) for \(N=2\) |
| Oracle | PySCF **2.14.0** (`agf2.RAGF2` one-shot SE + Dyson) |
| Pin date | 2026-07-23 |

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| \(E_{\mathrm{RHF}}\) | −1.116759307396 Ha | 1e-6 Ha | regression gate |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{HF}}\) | −0.578553859832 Ha | — | mean-field (info) |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{HF}}\) | +0.671143491914 Ha | — | mean-field (info) |
| HF gap | 34.005997328943 eV | — | info |
| \(\varepsilon_{\mathrm{HOMO}}^{\mathrm{GF2}}\) | **−0.591623589738** Ha | 1e-5 Ha | product |
| \(\varepsilon_{\mathrm{LUMO}}^{\mathrm{GF2}}\) | **+0.684213221821** Ha | 1e-5 Ha | product |
| **QP gap** | **34.717288266166 eV** | **1e-4 eV** | **primary gate** |
| IP \(-\varepsilon_H^{\mathrm{QP}}\) | **16.098898012595 eV** | 1e-4 eV | product |
| EA \(\varepsilon_L^{\mathrm{QP}}\) | **18.618390253572 eV** | 1e-4 eV | product |
| \(E_{\mathrm{corr}}\) (GM / MP2) | **−0.013138073590** Ha | 1e-6 Ha | Galitskii–Migdal |
| SC-AGF2 gap (info) | ≈ 34.451051 eV | — | full `RAGF2.kernel()`; **not** the gate |

Sanity: gap > 0 and finite (enforced in the regression binary).

Oracle repro:

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py gf2 \
  --geom h2 --basis sto-3g --conv-tol 1e-12
```

## Performance

Source: same REFERENCE (fair_bench 2026-07-23; median of 5; `OMP_NUM_THREADS=1`).
Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\) unless exempted.

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| one-shot GF2 | 0.2449 | 0.0174 | **0.071** | ✓ | ATV binary = RHF + one-shot GF2; PySCF = RHF+RAGF2 SE+Dyson |

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method gf2 --geom h2 --basis sto-3g --repeat 5 \
    -- ./build/regressions/reg_gf2-h2-sto3g
'
```

**Twin caveats.** (1) Method class match is **one-shot** G0F2 only — do not A/B
against full self-consistent `RAGF2.kernel()`. (2) Wall times include RHF SCF on
both sides of this fair_bench recipe; pure post-SCF QP cost is not split out.
(3) Small 2-AO system — ratio is informative for the foothold, not a large-basis
throughput claim.

## Limitations

- **Closed-shell RHF only.** No UHF / ROHF / GHF GF2.
- **One-shot product path only.** Self-consistent AGF2 is roadmap-open
  ([excited-states.md](../../roadmap/excited-states.md)).
- **Not G0W0 / ADC(2).** Bare Coulomb \(v\) in the self-energy; no screened
  \(W\); no ADC propagator.
- **CPU only.** CUDA GF2 is ○.
- **Scope of pin.** H₂/STO-3G is a minimal foothold (1 occupied, 1 virtual);
  larger systems / bases and denser auxiliary spectra are follow-on work.
- **Speed.** Current ratio \(\ll 10\times\) on the pin; no standing multi-system
  perf suite for GF2 beyond this twin.

## References

- Offline pin: `regressions/scf/gf2-h2-sto3g/REFERENCE.md`
- Perf (in REFERENCE; campaign queue records ratio **0.071**):
  (`gf2-h2-sto3g`)
- Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md`
- Pin helper: `tools/pyscf_oracle/pin_scf.py` (`gf2`)
- Roadmap: `docs/roadmap/excited-states.md` (GF2 ✓ CPU; self-consistent GF2 ○)
- Public headers: `include/atomiverse/cpu/scf/gf2.h`, `rhf.h`
- Impl: `src/cpu/scf/gf2.cpp`
- Backhouse, Holzmann & Booth, *J. Chem. Theory Comput.* **17**, 3414 (2021)
  (PySCF auxiliary GF2)
- Galitskii & Migdal, *Sov. Phys. JETP* **7**, 96 (1958)
- Related handbook: [g0w0-h2-sto3g](g0w0-h2-sto3g.md), [mp3-h2-sto3g](mp3-h2-sto3g.md),
  [adc2-h2co-sto3g](adc2-h2co-sto3g.md)
