# ΔSCF (IMOM) — N₂ / STO-3G α HOMO→LUMO

*(Technical-report draft; narrative rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `delta-scf-n2` |
| Regression | `regressions/scf/delta-scf-n2/` |
| Public API | `cpu::scf::UHF::delta_scf(frame, from, to, MomStyle)` |
| Roadmap | [Excited states](../../roadmap/excited-states.md) — Delta-SCF / MOM / IMOM ✓ |

## Summary

Closed-shell **ΔSCF** vertical electronic excitation on N₂: ground UHF (RHF
warm-start) minus a non-Aufbau UHF re-SCF with **IMOM** occupation locking on a
single α HOMO→LUMO promotion. Use when a single-determinant excited-state energy
difference is enough (MOM/IMOM foothold), not linear-response CIS/TDDFT roots.
API is concrete `cpu::scf::UHF` with members set in-place, then
`delta_scf(frame, 6, 7, MomStyle::IMOM)` → `DeltaScfResult` (`e_ground` /
`e_excited` [Ha], `delta_e_ev` [eV]). Textbook regression:
`regressions/scf/delta-scf-n2` (`reg_delta-scf-n2`), N₂ experimental \(r\),
STO-3G, vs offline PySCF 2.14.0 `scf.addons.mom_occ_`.

## Method

**Reference / target.** Neutral N₂ (\(N=14\) e), closed shell
\((n_\alpha,n_\beta)=(7,7)\). Spatial MOs: 10 AO (STO-3G); ground Aufbau fills
MO indices \(0\ldots6\); product excitation promotes α occupation \(6\to7\)
(HOMO→LUMO, 0-based).

**Vertical gap.**

\[
\Delta E = E_{\mathrm{ex}} - E_{\mathrm{gs}},\qquad
\Delta E\,[\mathrm{eV}] = \Delta E\,[\mathrm{Ha}]\times 27.211386245988
\]

(\(E_{\mathrm{gs}}\): RHF-seeded UHF; \(E_{\mathrm{ex}}\): non-Aufbau UHF with
MOM/IMOM). Same CODATA factor on both sides of the product gate
(`units::kHartreeInEv`).

**Occupation lock (Gilbert–Besley–Gill MOM; Barca–Gilbert–Gill IMOM).**

- Build non-Aufbau density via `SlaterDeterminant::with_occupation(α, from, to)`.
- Each SCF density stage scores new MOs by projection onto a reference occupied
  subspace and keeps the \(n_{\mathrm{occ}}\) largest-overlap orbitals (not pure
  energy-ordered Aufbau).
- **`MomStyle::IMOM`:** reference subspace **frozen** at the seed (PySCF twin
  `scf.addons.mom_occ_`).
- **`MomStyle::MOM`:** reference = **previous** iteration’s occupied set
  (live update).

**Approximations / exclusions.**

- Single-determinant mean-field only; no response matrix \(A/B\), no oscillator
  strengths, no ΔSCF analytic gradients / NAC on this surface.
- Spin: UHF Ms=0 single from α promotion on a closed-shell count — product is
  spin-contaminated (\(\langle S^2\rangle\approx 1.017\) on the pin; diagnostic
  only).
- Textbook minimal basis / one root: not a spectroscopic band assignment.
  Same geometry STO-3G TDA S1 is ~9.13 eV; ΔSCF IMOM pin is **8.24694 eV**.

## Implementation

Public surface: `cpu::Frame` + concrete `cpu::scf::UHF`. No `*Config` bags; no
factories. Members: `basis`, `tol`, `max_iter`, `n_alpha` / `n_beta` (ctor),
optional `MomStyle mom` / `set_mom_reference` for manual MOM runs. Product path
is `UHF::delta_scf`.

```cpp
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

// N₂, r=1.0977 Å, Z=7; box cubic 20 Å (see reg)
cpu::Frame frame = /* … */;

cpu::scf::UHF uhf(/*n_alpha=*/7, /*n_beta=*/7);
uhf.basis = cpu::scf::BasisSet::STO3G;
uhf.tol = 1e-12;
uhf.max_iter = 200;

cpu::scf::DeltaScfResult r =
    uhf.delta_scf(frame, /*from=*/6, /*to=*/7, cpu::scf::MomStyle::IMOM);
// r.e_ground, r.e_excited [Ha]; r.delta_e_ev [eV]; r.converged; r.iterations
// frame.meta energy left at the excited state after return
```

`delta_scf` sequence (closed shell): RHF warm-start → ground UHF → α
`with_occupation(from,to)` → `set_mom_reference` + re-`run` under
`MomStyle::IMOM` or `MOM`. Open-shell callers may pass a ground-state
`SCFInitialGuess*`; closed-shell without guess always RHF-seeds.

**Build / run regression**

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_delta-scf-n2 -j$(nproc)
  ./build/regressions/reg_delta-scf-n2
'
```

**File pointers**

| Role | Path |
| ---- | ---- |
| Public API | `include/atomiverse/cpu/scf/uhf.h` (`UHF::delta_scf`) |
| Result type | `include/atomiverse/cpu/scf/delta_scf.h` (`DeltaScfResult`) |
| `MomStyle` | `include/atomiverse/cpu/scf/scf_enums.h` |
| Implementation | `src/cpu/scf/uhf.cpp` (`UHF::delta_scf`); MOM in `src/cpu/scf/stages/density_stage.cpp` |
| Regression | `regressions/scf/delta-scf-n2/reg_delta-scf-n2.cpp` |
| Offline pin | `regressions/scf/delta-scf-n2/REFERENCE.md` |
| Oracle | `tools/pyscf_oracle/pin_scf.py delta-scf` (never in ctest) |

## Validation (vs PySCF)

Source: [`regressions/scf/delta-scf-n2/REFERENCE.md`](../../../regressions/scf/delta-scf-n2/REFERENCE.md)
(offline pin only; **no live PySCF in ctest**).

| Setting | Value |
| ------- | ----- |
| System | N₂, charge 0, singlet ground |
| Geometry | N (0,0,0); N (0,0,1.0977) [Å] — experimental bond length |
| Basis | STO-3G |
| Angular | Cartesian (`cart=True`; no \(d\) on STO-3G; engine-neutral) |
| Occupation | \((n_\alpha,n_\beta)=(7,7)\); α promote MO 6→7; β Aufbau |
| AO / MO | 10 AO; 7 occupied spatial MOs (14 e) |
| Ground SCF | RHF → UHF warm-start, `conv_tol=1e-12` |
| Excited SCF | UHF + IMOM (`mom_occ_`); same tol |
| Oracle | PySCF **2.14.0**, pinned 2026-07-23 (`~/work/.x86_64`) |
| Ha→eV | 27.211386245988 (CODATA; `units::kHartreeInEv`) |
| Product gates | \(\lvert\Delta E-\mathrm{pin}\rvert < 10^{-4}\) eV; energies \(10^{-6}\) Ha; \(E_{\mathrm{ex}}>E_{\mathrm{gs}}\) |

| Quantity | Atomiverse / pin | PySCF pin | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E_{\mathrm{ground}}\) [Ha] | −107.495893307834 | −107.495893307834 | \(10^{-6}\) Ha | RHF-seeded UHF |
| \(E_{\mathrm{excited}}\) [Ha] | −107.192823826964 | −107.192823826964 | \(10^{-6}\) Ha | UHF-IMOM |
| \(\Delta E\) [Ha] | 0.303069480870 | 0.303069480870 | (derived) | \(E_{\mathrm{ex}}-E_{\mathrm{gs}}\) |
| \(\Delta E\) [eV] | **8.246940703328** | **8.246940703328** | \(10^{-4}\) eV | product gate |
| \(\langle S^2\rangle\) (ex.) | ≈ 1.017 | ≈ 1.017 | diagnostic | spin-contaminated Ms=0 |

Pins copied verbatim from REFERENCE (2026-07-23). Regression hard-codes the same
three energy checks.

**Oracle repro (offline only):**

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py delta-scf \
    --geom n2 --basis sto-3g --conv-tol 1e-12
# optional: --from-mo 6 --to-mo 7  (defaults = HOMO/LUMO)
```

## Performance

Source: `regressions/scf/delta-scf-n2/REFERENCE.md` § Performance (handoff /
queue ratio **7.83**). Protocol: `tools/pyscf_oracle/fair_bench.py`; same
geom/basis; median of ≥3 runs; `OMP_NUM_THREADS=1`; gate
\(r=T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\le 10\).

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  export OMP_NUM_THREADS=1
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method delta_scf --geom n2 --basis sto-3g --repeat 3 \
    -- -- ./build/regressions/reg_delta-scf-n2
'
```

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r=T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| N₂ UHF/STO-3G ΔSCF IMOM (`delta_scf`) | **0.0283** | **0.2217** | **7.83** | ✓ | ATV path = RHF warm-start + ground UHF + IMOM UHF (**three** SCF loops); PySCF twin = RHF + `mom_occ_` UHF; micro-system walls are noisy |

Gate: \(r\le 10\) unless exempted — **within_10x = true**.
is not present in-tree; numbers above are
from the regression REFERENCE and handoff rollup.

## Limitations

- **Scope:** UHF ΔSCF / MOM / IMOM energy differences only. No ΔSCF forces,
  NAC, state-averaged SCF, or multi-root automatic search.
- **Spin purity:** closed-shell α single excitation yields a spin-contaminated
  Ms=0 determinant; not a pure singlet without spin purification / ROKS /
  spin-flip alternatives.
- **Basis / model chemistry:** STO-3G textbook pin; larger bases and multi-config
  character need separate validation. Not a substitute for CIS/TDDFT/EOM band
  structure.
- **API surface:** product helper is on `UHF` only; RHF/ROHF/RKS ΔSCF paths are
  not separate presets (manual MOM members exist on mean-field base).
- **Speed debt:** \(r\approx 7.83\) passes the 10× gate; three SCF loops vs a
  lighter PySCF twin inflate the ratio on this micro-case. ERI/SCF cost dominates.
- **Roadmap (open):** CUDA / broader MOM properties / ROKS excited states — see
  `docs/roadmap/excited-states.md` (ROKS excited ○; Delta-SCF / MOM / IMOM ✓).

## References

- Offline pin: `regressions/scf/delta-scf-n2/REFERENCE.md`
- Perf numbers: same REFERENCE § Performance; handoff
  the case `REFERENCE.md` (ratio 7.83); queue note
- Oracle: `tools/setup_pyscf_env.sh`, `tools/pyscf_oracle/pin_scf.py delta-scf`,
  `tools/pyscf_oracle/fair_bench.py --method delta_scf`
- A. T. B. Gilbert, N. A. Besley, P. M. W. Gill, *J. Phys. Chem. A* **112**,
  13164 (2008) — MOM. DOI [10.1021/jp801738f](https://doi.org/10.1021/jp801738f)
- G. M. J. Barca, A. T. B. Gilbert, P. M. W. Gill, *J. Chem. Theory Comput.*
  **14**, 1501 (2018) — IMOM. DOI
  [10.1021/acs.jctc.7b00994](https://doi.org/10.1021/acs.jctc.7b00994)
- Implementation twin: PySCF `scf.addons.mom_occ_`
- Roadmap: `docs/roadmap/excited-states.md`
- Oracle rules: `docs/tutorial/appendix-pyscf-oracle.md`
