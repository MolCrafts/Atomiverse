# GPW PBE-D3 (zero damping) analytic cell stress — periodic water dimer

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d3-stress` |
| Regression | `regressions/gpw/gpw-d3-stress/` |
| Public API | `cpu::gpw::GPWEvaluator` (`d3`, `compute_stress`, `stress_use_fd=false`); free `gpw_stress` / `gpw_stress_fd` with `GpwDft::d3` |
| Roadmap / queue | `docs/roadmap/dft.md` §8 Dispersion; `docs/developer/cp2k-method-queue.md` (done, wave 2 2026-07-31) |

## Summary

Gates **Hellmann–Feynman fixed-\(P\) orthorhombic cell stress** on
**Γ-point GPW PBE** with additive **DFT-D3 zero ("original") damping**
(two-body, `CALCULATE_C9_TERM .FALSE.`) for a **periodic water dimer**
(same 12 Å cubic cell as the `gpw-d3-forces` periodic twin and
`gpw-d4-stress`). Dispersion is geometry-only; the analytic path rebuilds
\(E_{\mathrm{disp}}^{\mathrm{D3}}(\{\mathbf{R}\},\mathbf{h})\) at each
strained cell while holding the unstrained density matrix \(P\) and
energy-weighted \(W\), using the CP2K-parity `kD3Cp2kPairCutoffBohr`
(= 2 × `R_CUTOFF 20.0` Å) lattice-image cutoff.

**Oracle (binding).** Unlike `gpw-d4-stress` (Arrhenius CP2K lacks DFTD4),
CP2K 2026.1 **does** run `TYPE DFTD3`, so this slug carries a real
**CP2K `STRESS_TENSOR NUMERICAL` difference pin**: two offline runs on the
identical geometry — with and without the `&VDW_POTENTIAL` block — whose
stress difference isolates the D3-zero dispersion stress exactly (D3 does
not enter the Kohn–Sham potential, so the electronic stress cancels).
Decisive internal residual: product analytic vs multi-SCF strain-FD
(`gpw_stress_fd`) on the same `d3=true` method.

Public surface: bool / stress members on `cpu::gpw::GPWEvaluator` and
`cpu::gpw::GpwDft` (no `*Config` bag, no `make_*` / `build_*`). Frame
`kStress` is eV/Å³; pin channel `last_stress_ha_bohr3()` [Ha/Bohr³].
Product gate: `regressions/gpw/gpw-d3-stress` (`reg_gpw-d3-stress`).

## Method

### Host + dispersion

Closed-shell **Γ-point GPW** Kohn–Sham (VandeVondele et al., *Comput. Phys.
Commun.* **167**, 103 (2005)) with PBE XC and GTH PBE PPs. **DFT-D3 zero
damping** (Grimme, Antony, Ehrlich, Krieg, *J. Chem. Phys.* **132**, 154104
(2010)) with PBE reference parameters:

| Parameter | Value |
| --------- | ----: |
| \(s_6\) | 1.0 |
| \(s_8\) | 0.722 |
| \(s_{r,6}\) | 1.217 |
| \(s_{r,8}\) | 1.0 |

Periodic lattice sums to `kD3Cp2kPairCutoffBohr` = 75.589 Bohr (the CP2K
half-cutoff convention that the `gpw-d3-forces` pin settled; on this
**periodic** cell the convention moves the pin, unlike the isolated twin).

### Analytic fixed-\(P\) cell stress with D3

One SCF at the unstrained cell → \(P\), \(W\). Central FD of the
non-self-consistent Lagrangian under orthorhombic diagonal strain at fixed
fractional coordinates:

\[
E_L[P,W;\varepsilon]
  = \mathrm{Tr}(P\,T) + \mathrm{Tr}(P\,V_{\mathrm{pp}})
  + E_{\mathrm{es}}[\rho_P] + E_{\mathrm{xc}}[\rho_P]
  + E_{\mathrm{disp}}^{\mathrm{D3}}(\{\mathbf{R}\},\mathbf{h})
  - \mathrm{Tr}(W\,S),
\qquad
\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}.
\]

Sign matches CP2K `STRESS_TENSOR`; Voigt order xx,yy,zz,xy,xz,yz;
off-diagonals ~0 (orthorhombic); \(h = 2\times10^{-3}\) (`stress_strain_h`).

**Relation to other D3 / stress slugs.**

| Slug | Deliverable |
| ---- | ----------- |
| [`gpw-d3-forces`](gpw-d3-forces.md) | D3-zero **forces** (isolated) + periodic E_disp pin + cutoff convention |
| [`gpw-d4-stress`](gpw-d4-stress.md) | D4 stress, dual-gate only (no CP2K DFTD4 build) |
| [`gpw-mbd-stress`](gpw-mbd-stress.md) | Γ MBD stress |
| **`gpw-d3-stress`** | D3-zero stress with a real **CP2K stress-difference pin** |

**Key exclusions.** D3(BJ) stress (`gpw-d3bj-stress`, sibling row); ATM/C9
stress; shear / triclinic; multi-k dedicated pin (the multi-k fold shares
the same `e_disp` branch); CUDA GPW.

## Implementation

Public surface only (`GPWEvaluator` / `GpwDft` members; Frame in/out):

```cpp
cpu::gpw::GPWEvaluator eval({O, H});
eval.cutoff_ry = 200.0;
eval.tol = 1e-7;             // matches CP2K EPS_SCF twin
eval.xc = cpu::gpw::XcKind::PBE;
eval.nonperiodic = false;    // PERIODIC XYZ
eval.d3 = true;              // zero damping; exclusive vs d3bj/d4/ts/xdm/mbd
eval.compute_stress = true;  // product analytic fixed-P
eval.stress_use_fd = false;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

const double e_disp = eval.last_result().e_disp;   // [Ha]
const auto& san = eval.last_stress_ha_bohr3();     // [Ha/Bohr³]
// frame.meta.get(kStress)                          // [eV/Å³]
```

The regression isolates ATV's dispersion stress the same way the pin does:
a second warm-started analytic product with `d3=false`, subtracted.

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/gpw/gpw_evaluator.h` | `d3`, `compute_stress`, `last_stress_ha_bohr3` |
| `include/atomiverse/cpu/gpw/gpw_solver.h` | `GpwDft::d3`, `gpw_stress` / `gpw_stress_fd` |
| `src/cpu/gpw/internal/gpw_stress.cpp` | Fixed-\(P\)+Pulay Lagrangian; D3 rebuilt under strain (Γ + multi-k) |
| `include/atomiverse/cpu/gpw/internal/dftd3.h` | `dftd3_pbe`, `kD3Cp2kPairCutoffBohr` |
| `tests/cpu/gpw/test_cpu_gpw_stress.cpp` | `AnalyticMatchesFdWithD3`, `D3ChangesGammaDiagonal` |
| `regressions/gpw/gpw-d3-stress/` | Product gate + REFERENCE.md pins |
| `tools/cp2k_oracle/inputs/gpw-d3-stress/` | CP2K NUMERICAL stress twin inputs |

## Validation (CP2K 2026.1 + dual-gate)

Source:
[`regressions/gpw/gpw-d3-stress/REFERENCE.md`](../../../regressions/gpw/gpw-d3-stress/REFERENCE.md).
Offline CP2K 2026.1 (`cp2k.psmp`, git:5e54ba2), never launched from ctest.

| Quantity | Pin / gate | Tol |
| -------- | ---------- | --- |
| \(E_{\mathrm{disp}}\) | **−0.00082089467662 Ha** (CP2K `Dispersion energy`) | 1×10⁻⁸ Ha |
| \(\Delta\sigma\) diag [Ha/Bohr³] | (−1.70641×10⁻⁸, −9.0396×10⁻¹⁰, +3.7428×10⁻⁹) = σ_num(D3) − σ_num(plain) | 2×10⁻⁹ |
| max\|σ_an − σ_fd\| diag | internal multi-SCF FD twin | 2×10⁻⁵ |
| total σ diag vs CP2K | sanity only (single-grid vs multigrid, kbar-level) | 1×10⁻⁴ |

**Measured (first green, 2026-07-31):** see REFERENCE.md "Atomiverse
numbers" table.

## Performance

CP2K NUMERICAL stress twin wall **33.6 s** (np 2 × OMP 2, Arrhenius epyc);
ATV analytic product wall and the ratio are recorded in REFERENCE.md.
The regression's extra plain/FD runs serve the difference pin and dual gate,
not the product path.

## References

- Offline pin: `regressions/gpw/gpw-d3-stress/REFERENCE.md`
- Queue: `docs/developer/cp2k-method-queue.md`; loop: `docs/developer/cp2k-method-loop.md`
- S. Grimme, J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.* **132**,
  154104 (2010) (DFT-D3, zero damping)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) (GPW)
- Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985) (stress context)
- Sibling pages: [gpw-d3-forces](gpw-d3-forces.md),
  [gpw-d4-stress](gpw-d4-stress.md), [gpw-mbd-stress](gpw-mbd-stress.md)
