# N₂ ΔSCF (IMOM) / STO-3G — vertical α HOMO→LUMO pin

Closed-shell **ΔSCF** electronic excitation for N₂ at the experimental bond
length, with **IMOM** (initial maximum-overlap) occupation locking on the
non-Aufbau UHF re-SCF. Product API: `UHF::delta_scf(frame, from, to,
MomStyle::IMOM)` after setting `n_alpha` / `n_beta` / `basis`.

Literature / theory:

- A. T. B. Gilbert, N. A. Besley, P. M. W. Gill, *Self-Consistent Field
  Calculations of Excited States Using the Maximum Overlap Method (MOM)*,
  J. Phys. Chem. A **112**, 13164 (2008).
  DOI [10.1021/jp801738f](https://doi.org/10.1021/jp801738f)
- G. M. J. Barca, A. T. B. Gilbert, P. M. W. Gill, *Simple Models for
  Difficult Electronic Excitations*, J. Chem. Theory Comput. **14**, 1501
  (2018). DOI [10.1021/acs.jctc.7b00994](https://doi.org/10.1021/acs.jctc.7b00994)
  (IMOM = frozen initial reference)
- Implementation twin: PySCF `scf.addons.mom_occ_(uhf, (C,C), (occ_a,occ_b))`
  on a RHF-seeded α HOMO→LUMO promotion (IMOM; reference orbitals fixed).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | N₂, charge 0, singlet ground |
| Geometry (Å) | N(0,0,0); N(0,0,1.0977) — experimental bond length |
| Basis | STO-3G |
| Angular convention | `cart=True` (no \(d\) shells; engine-neutral) |
| Ground SCF | RHF → UHF warm-start, `conv_tol=1e-12` |
| Excited SCF | UHF, α occupation promote HOMO→LUMO (indices 6→7), β Aufbau; IMOM via `mom_occ_` |
| AO / MO | 10 AO; 7 occupied spatial MOs (14 e) |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py delta-scf \
    --geom n2 --basis sto-3g --conv-tol 1e-12
# optional: --from-mo 6 --to-mo 7  (defaults = HOMO/LUMO)
```

## Pinned values

| Quantity | Pin | Tolerance |
|---|---|---|
| \(E_{\mathrm{ground}}\) (RHF/UHF) | −107.495893307834 Ha | 1e-6 Ha |
| \(E_{\mathrm{excited}}\) (UHF-IMOM) | −107.192823826964 Ha | 1e-6 Ha |
| ΔE = \(E_{\mathrm{ex}}-E_{\mathrm{gs}}\) | 0.303069480870 Ha = **8.246940703328 eV** | **1e-4 eV** (product gate) |
| ⟨S²⟩ (excited, diagnostic) | ≈ 1.017 | (spin-contaminated Ms=0 single) |

CODATA \(1\,\mathrm{Ha}=27.211386245988\,\mathrm{eV}\)
(`units::kHartreeInEv`) — both sides of the eV gate use the same factor.

This is a **textbook minimal** ΔSCF foothold (not a spectroscopic band
assignment): STO-3G N₂ TDA S1 is ~9.13 eV on the same geometry; the ΔSCF
IMOM root is a single-determinant estimate of the HOMO→LUMO promotion.

## Gates (product regression)

- ΔE: \(|\Delta E_{\mathrm{ours}}-\mathrm{pin}| < 10^{-4}\,\mathrm{eV}\).
- \(E_{\mathrm{ground}}\) and \(E_{\mathrm{excited}}\): \(10^{-6}\) Ha.
- \(E_{\mathrm{excited}} > E_{\mathrm{ground}}\).

## Performance

Fair wall-time protocol (`tools/pyscf_oracle/fair_bench.py`): same geom/basis,
median of ≥3 runs, ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\), target ≤ 10.

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT=$OPENBLASDIR
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH=$LIBRARY_PATH:$LD_LIBRARY_PATH
  ~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method delta_scf --geom n2 --basis sto-3g --repeat 3 \
    -- -- ./build/regressions/reg_delta-scf-n2
'
```

| Metric | Value |
|---|---|
| \(T_{\mathrm{pyscf}}\) median (s) | 0.0283 |
| \(T_{\mathrm{atv}}\) median (s) | 0.2217 |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **7.83** |
| within_10x | **true** |
| Notes | `OMP_NUM_THREADS=1`; ATV path = RHF warm-start + ground UHF + IMOM UHF (three SCF loops); PySCF twin = RHF + `mom_occ_` UHF |

## Public API (no Config bags)

```cpp
cpu::scf::UHF uhf(/*n_alpha=*/7, /*n_beta=*/7);
uhf.basis = cpu::scf::BasisSet::STO3G;
uhf.tol = 1e-12;
auto r = uhf.delta_scf(frame, /*from=*/6, /*to=*/7, cpu::scf::MomStyle::IMOM);
// r.delta_e_ev [eV]; r.e_ground / r.e_excited [Ha]
```

Members only: `MomStyle mom`, `set_mom_reference(det)`,
`SlaterDeterminant::with_occupation` — no `DeltaScfConfig`, no factories.
