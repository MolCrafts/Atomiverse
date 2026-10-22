# Unrestricted Hartree–Fock (UHF)

Unrestricted Hartree–Fock lets the α and β spin orbitals relax independently,
so open-shell, ionized and non-Aufbau (excited) determinants are described by
one spin-unrestricted Slater determinant. The public entry point is the
concrete method `cpu::scf::UHF{basis, n_alpha, n_beta}` (members `tol`,
`max_iter`, optional `MomStyle mom`); single-point energies run through
`run(frame)`, and the ΔSCF excited-state product path is
`UHF::delta_scf(frame, from, to, MomStyle)`.

The case also carries sections without a tutorial page — `h2_uhf_sto3g`
(H₂⁺, H₂, H₂O⁺ UHF/STO-3G, CPU and CUDA), `cuda-uhf` (O₂ and OH, CPU and
CUDA), `uhf-hessian-o2` (O₂ analytic nuclear Hessian) and `n2_ionization`
(N₂ RHF/6-31G* Koopmans vs ΔSCF ionization energies); their oracle records
live in the case `REFERENCE.md`. The page below documents the ΔSCF section.

Case: `ctest -R '^scf/uhf$'` — sources and oracle record in [`regressions/scf/uhf/`](../../../regressions/scf/uhf/). One section of the case per `##` heading below: `atv_regressions --case scf/uhf --section <name>`.

## Common setup and oracle

The case needs `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON` (CPU build
environment `buildenv-gcc/2026.03-mpich`, see the repository build notes).
Every pin is an offline PySCF **2.14.0** run (`scf.addons.mom_occ_` for MOM /
IMOM); nothing in ctest imports PySCF. Oracle rules:
`docs/tutorial/appendix-pyscf-oracle.md`.

## References

- A. T. B. Gilbert, N. A. Besley, P. M. W. Gill, *J. Phys. Chem. A* **112**,
  13164 (2008) — MOM. DOI [10.1021/jp801738f](https://doi.org/10.1021/jp801738f)
- G. M. J. Barca, A. T. B. Gilbert, P. M. W. Gill, *J. Chem. Theory Comput.*
  **14**, 1501 (2018) — IMOM. DOI
  [10.1021/acs.jctc.7b00994](https://doi.org/10.1021/acs.jctc.7b00994)
- Roadmap: [Excited states](../../roadmap/excited-states.md) — Delta-SCF / MOM / IMOM ✓

## delta-scf-n2 — ΔSCF (IMOM), N₂ / STO-3G α HOMO→LUMO {#delta-scf-n2}

### Summary

Closed-shell **ΔSCF** vertical electronic excitation on N₂: ground UHF (RHF
warm-start) minus a non-Aufbau UHF re-SCF with **IMOM** occupation locking on a
single α HOMO→LUMO promotion. Use it when a single-determinant excited-state
energy difference is enough, not linear-response CIS/TDDFT roots.
`delta_scf(frame, 6, 7, MomStyle::IMOM)` returns a `DeltaScfResult`
(`e_ground` / `e_excited` [Ha], `delta_e_ev` [eV]). The system is N₂ at the
experimental bond length in STO-3G, pinned against PySCF `scf.addons.mom_occ_`.

### Method

**Reference / target.** Neutral N₂ (\(N=14\) e), closed shell
\((n_\alpha,n_\beta)=(7,7)\). Spatial MOs: 10 AO (STO-3G); ground Aufbau fills
MO indices \(0\ldots6\); the excitation promotes α occupation \(6\to7\)
(HOMO→LUMO, 0-based).

**Vertical gap.**

\[
\Delta E = E_{\mathrm{ex}} - E_{\mathrm{gs}},\qquad
\Delta E\,[\mathrm{eV}] = \Delta E\,[\mathrm{Ha}]\times 27.211386245988
\]

(\(E_{\mathrm{gs}}\): RHF-seeded UHF; \(E_{\mathrm{ex}}\): non-Aufbau UHF with
MOM/IMOM). The same CODATA factor (`units::kHartreeInEv`) is used on both
sides of the gate.

**Occupation lock (Gilbert–Besley–Gill MOM; Barca–Gilbert–Gill IMOM).**

- The non-Aufbau density is built via `SlaterDeterminant::with_occupation(α, from, to)`.
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
- Spin: UHF Ms=0 single from α promotion on a closed-shell count — the product
  is spin-contaminated (\(\langle S^2\rangle\approx 1.017\) on the pin;
  diagnostic only).
- Textbook minimal basis / one root: not a spectroscopic band assignment.
  The same-geometry STO-3G TDA S1 is ~9.13 eV; the ΔSCF IMOM pin is **8.24694 eV**.

### Implementation

Constructor arguments: `basis`, `n_alpha` / `n_beta`; members: `tol`,
`max_iter`, optional `MomStyle mom` / `set_mom_reference` for manual MOM runs.

```cpp
// N₂, r=1.0977 Å, Z=7; box cubic 20 Å
cpu::scf::UHF uhf{cpu::scf::BasisSet::STO3G, /*n_alpha=*/7, /*n_beta=*/7};
uhf.tol = 1e-12;
uhf.max_iter = 200;

cpu::scf::DeltaScfResult r =
    uhf.delta_scf(frame, /*from=*/6, /*to=*/7, cpu::scf::MomStyle::IMOM);
// r.e_ground, r.e_excited [Ha]; r.delta_e_ev [eV]; r.converged; r.iterations
// frame.meta energy is left at the excited state after return
```

`delta_scf` sequence (closed shell): RHF warm-start → ground UHF → α
`with_occupation(from,to)` → `set_mom_reference` + re-`run` under
`MomStyle::IMOM` or `MOM`. Open-shell callers may pass a ground-state
`SCFInitialGuess*`; closed-shell without a guess always RHF-seeds.

| Role | Path |
| ---- | ---- |
| Public API | `uhf.h` (`UHF::delta_scf`) |
| Result type | `delta_scf.h` (`DeltaScfResult`) |
| `MomStyle` | `scf_enums.h` |
| Implementation | `src/cpu/scf/uhf.cpp` (`UHF::delta_scf`); MOM in `src/cpu/scf/stages/density_stage.cpp` |

### Validation (vs PySCF)

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
| Ha→eV | 27.211386245988 (CODATA; `units::kHartreeInEv`) |
| Gates | \(\lvert\Delta E-\mathrm{pin}\rvert < 10^{-4}\) eV; energies \(10^{-6}\) Ha; \(E_{\mathrm{ex}}>E_{\mathrm{gs}}\) |

| Quantity | Atomiverse / pin | PySCF pin | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E_{\mathrm{ground}}\) [Ha] | −107.495893307834 | −107.495893307834 | \(10^{-6}\) Ha | RHF-seeded UHF |
| \(E_{\mathrm{excited}}\) [Ha] | −107.192823826964 | −107.192823826964 | \(10^{-6}\) Ha | UHF-IMOM |
| \(\Delta E\) [Ha] | 0.303069480870 | 0.303069480870 | (derived) | \(E_{\mathrm{ex}}-E_{\mathrm{gs}}\) |
| \(\Delta E\) [eV] | **8.246940703328** | **8.246940703328** | \(10^{-4}\) eV | gate |
| \(\langle S^2\rangle\) (ex.) | ≈ 1.017 | ≈ 1.017 | diagnostic | spin-contaminated Ms=0 |

The section hard-codes the three energy checks above.

**Oracle repro (offline only):**

```bash
python <pyscf pin> delta-scf \
    --geom n2 --basis sto-3g --conv-tol 1e-12
# optional: --from-mo 6 --to-mo 7  (defaults = HOMO/LUMO)
```

### Performance

Protocol: PySCF twin, same geometry/basis, median of ≥3 runs,
`OMP_NUM_THREADS=1`; gate \(r=T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\le 10\).

```bash
OMP_NUM_THREADS=1 python <pyscf fair-bench> \
  --method delta_scf --geom n2 --basis sto-3g --repeat 3 -- -- <section binary>
```

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r=T_{\mathrm{ATV}}/T_{\mathrm{PySCF}}\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| N₂ UHF/STO-3G ΔSCF IMOM (`delta_scf`) | **0.0283** | **0.2217** | **7.83** | ✓ | ATV path = RHF warm-start + ground UHF + IMOM UHF (**three** SCF loops); PySCF twin = RHF + `mom_occ_` UHF; micro-system walls are noisy |

### Limitations

- **Scope:** UHF ΔSCF / MOM / IMOM energy differences only. No ΔSCF forces,
  NAC, state-averaged SCF, or multi-root automatic search.
- **Spin purity:** a closed-shell α single excitation yields a spin-contaminated
  Ms=0 determinant; not a pure singlet without spin purification / ROKS /
  spin-flip alternatives.
- **Basis / model chemistry:** STO-3G textbook pin; larger bases and
  multi-configuration character need separate validation. Not a substitute for
  CIS/TDDFT/EOM band structure.
- **API surface:** the helper is on `UHF` only; RHF/ROHF/RKS ΔSCF paths are
  not separate presets (manual MOM members exist on the mean-field base).
- **Speed:** \(r\approx 7.83\) passes the 10× gate; three SCF loops vs a
  lighter PySCF twin inflate the ratio on this micro-case. ERI/SCF cost dominates.
- **Open:** CUDA / broader MOM properties / ROKS excited states — see
  `docs/roadmap/excited-states.md` (ROKS excited ○; Delta-SCF / MOM / IMOM ✓).
