# GPW variable-cell relaxation (CELL_OPT)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Constrained variable-cell relaxation of a closed-shell Γ-point GPW system
toward a target pressure. The electronic structure is a `cpu::gpw::GpwDft`
bound through `cpu::gpw::GPWEvaluator` with cell stress on
(`compute_stress = true`; `stress_use_fd = false` selects the analytic
fixed-\(P\) stress, `GpwDft::stress_axes` masks the constrained edges). The
outer loop is the `cpu::minimizer::Minimize` preset with
`cpu::minimizer::AnisotropicCellRelax` as optimizer, and termination is the
criterion `cpu::StressConverged` — the optimizer never judges itself.

The case runs a proton-ordered periodic water chain (ice Ih c-axis strand
motif) in two sections: `gpw-variable-cell-ice` (compact GTH-PADE basis,
multi-SCF strain-FD stress; record in the case `REFERENCE.md`) and
`gpw-variable-cell-molopt` (MOLOPT basis, analytic stress, below). Both are
gated on offline CP2K 2026.1 `CELL_OPT` pins hard-coded in the binaries —
never live CP2K from ctest.

Case: `ctest -R '^gpw/variable-cell$'` — sources and oracle record in [`regressions/gpw/variable-cell/`](../../../regressions/gpw/variable-cell/). One section of the case per `##` heading below: `atv_regressions --case gpw/variable-cell --section <name>`.

## References

- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) (GPW / Quickstep)
- Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985) (stress context)
- CP2K Quickstep `CELL_OPT` / `STRESS_TENSOR` / `CONSTRAINT` manual
- Analytic stress: [gpw-analytic-stress](gpw-rks.md#gpw-analytic-stress);
  strain-FD stress: [gpw-stress-water](gpw-rks.md#gpw-stress-water)
- Narrative: `docs/tutorial/ch09-gpw.md`, `docs/tutorial/ch12-geometry-optimization.md`;
  roadmap: `docs/roadmap/dft.md` (variable-cell / CELL_OPT)

## gpw-variable-cell-molopt — MOLOPT CELL_OPT, ice-like water chain {#gpw-variable-cell-molopt}

Closed-shell Γ-point GPW on a periodic water chain (1 H₂O / cell, H-bonded to
its own \(x\)-images). Ionic fractional coordinates are **fixed**; only the
chain axis \(a\) relaxes under \(P_{\mathrm{target}}=0\) while the \(b=c=8\) Å
vacuum edges stay bit-identical. This is the pin-quality MOLOPT twin of
`gpw-variable-cell-ice` (PBE / DZVP-MOLOPT-SR-GTH / GTH-PBE, analytic
fixed-\(P\) stress, `stress_axes = {x}`, `AnisotropicCellRelax` with `relax_x`
only). Offline CP2K 2026.1 `CELL_OPT` pins:
\(a^\star = 3.200237\) Å, \(E^\star = -17.184075549824836\) Ha; fair wall
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 7.5\) (≤ 10×).

### Method

#### Electronic structure (GPW / Quickstep)

Gaussian-and-plane-waves Kohn–Sham: density matrix in contracted Gaussians;
Hartree / XC / local short-range GTH PP on a uniform grid; Poisson under
`PERIODIC XYZ` (`nonperiodic = false`). Public energy on Frame is eV
(`kEnergy`); the pin channel is Hartree via `last_energy_hartree()`.

**Basis choice.** Full **DZVP-MOLOPT-GTH** periodic image-sum collocation
exceeds the ≤ 60 s regression budget even for a single SCF on this chain, so
the section uses the short-range MOLOPT table **DZVP-MOLOPT-SR-GTH** (same
family as [gpw-pbe-d3bj-ice](gpw-d3.md#gpw-pbe-d3bj-ice) /
[gpw-ts-ice](gpw-tkatchenko-scheffler.md#gpw-ts-ice)) — still a MOLOPT
Gaussian basis + PBE + GTH-PBE. Full-MOLOPT CP2K `CELL_OPT` pins are kept as a
secondary oracle (not gated in the regression).

#### Cell stress (analytic fixed-\(P\))

Hellmann–Feynman fixed-\(P\) orthorhombic diagonal stress with Pulay
\(-\mathrm{Tr}(W\,S)\), not multi-SCF strain-FD. Sign matches CP2K
`STRESS_TENSOR`:

\[
\sigma_{aa}
=
-\Omega^{-1}
\frac{\partial E}{\partial\varepsilon_{aa}},
\qquad a\in\{x,y,z\},
\]

with Frame `kStress` extensive **[eV]** (σ·Ω) Voigt and pin channel
`last_stress_ha_bohr3()` **[Ha/Bohr³]**. The axes mask
`eval.method().stress_axes = {true,false,false}` forces
\(\sigma_{yy}=\sigma_{zz}=0\) in the published tensor so constrained edges
never drive residual or scale (CP2K `CONSTRAINT YZ` twin).

#### Variable-cell optimizer

`AnisotropicCellRelax` independently scales orthorhombic edges toward
\(P_{\mathrm{target}}\) (here 0). For GPW intensive stress:

\[
s_a \approx 1 - \alpha\,\frac{\sigma_{aa}-P_{\mathrm{target}}}{B},
\qquad |s_a-1|\le \texttt{max\_scale},
\]

with soft bulk modulus \(B = \texttt{bulk\_mod}\) [eV/Å³] from the offline
\(E(a)\) curvature (full-MOLOPT ENERGY scan →
\(d\sigma_{xx}/d\varepsilon \approx 0.20\,\mathrm{eV/Å}^3\)). Masked axes
(`relax_y = relax_z = false`) are never scaled and never enter the residual.
Cartesian ion positions scale affinely with the cell so **fractional**
coordinates stay fixed (CP2K `SCALED T` + `FIXED_ATOMS`). Termination:
`StressConverged{0, 5×10⁻⁴ eV/Å³, x-only}` → `Verdict::Converged` when
\(|\sigma_{xx}-0|\le\texttt{stol}\).

### System

| Knob | Value |
| ---- | ----- |
| Motif | 1 H₂O / cell, H-bonded chain along \(a\) |
| Cell | \(a_0 = 3.24\) Å start; \(b = c = 8\) Å fixed; `PERIODIC XYZ` |
| XC | **PBE** |
| Basis | **DZVP-MOLOPT-SR-GTH** (O, H) |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) |
| CUTOFF | 300 Ry |
| SCF | `tol` / EPS_SCF \(= 1\times 10^{-7}\), mix 0.3, max_iter 200 |
| Stress | analytic fixed-\(P\), \(h=2\times 10^{-3}\), axes `{x}` |
| Relax | \(\alpha=1.0\), `bulk_mod = 0.20` eV/Å³, `max_scale = 0.05` |
| Converge | `StressConverged{0, 5e-4 eV/Å³, x-only}`; budget 6 outer steps |
| Ions | forces off (`compute_forces = false`); fractions fixed |

**Fractional coordinates (fixed; CP2K `SCALED T`):**

| Atom | \(f_x\) | \(f_y\) | \(f_z\) |
| ---- | ------: | ------: | ------: |
| O | 0.000000 | 0 | 0.000000 |
| H1 | 0.319067 | 0 | 0.000000 |
| H2 | −0.079900 | 0 | 0.115830 |

Cartesian start: \(x_i = f_{x,i}\,a_0\), \(z_{\mathrm{H2}} = f_{z}\,c\).

#### Relation to `gpw-variable-cell-ice`

| | ice (GTH-PADE) | molopt (this section) |
| - | -------------------- | -------------------------- |
| XC / basis / PP | PADE / SZV+DZV-GTH-PADE | PBE / DZVP-MOLOPT-SR-GTH / GTH-PBE |
| \(a^\star\) (CP2K) | 3.211148 Å | **3.200237 Å** |
| Stress path | multi-SCF strain-FD | **analytic fixed-\(P\)** |
| Wall ATV / CP2K | ~5.1 / 3.9 s | **42.5 / 5.67 s** |
| Fair \(r\) | ~1.3 | **~7.5** |

### Code

```cpp
const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-SR-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
const cpu::gpw::Kind H{ /* DZVP-MOLOPT-SR-GTH, GTH-PBE-q1 */ };

cpu::gpw::GpwDft dft({O, H}, 300.0, cpu::gpw::XcKind::PBE);
dft.max_iter = 200; dft.tol = 1e-7; dft.mix = 0.3; dft.nonperiodic = false;

cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/false);
eval.compute_stress = true;
eval.stress_use_fd = false;                         // analytic fixed-P
eval.method().stress_strain_h = 2e-3;
eval.method().stress_axes = {true, false, false};   // chain axis only

cpu::minimizer::Minimize opt;
cpu::minimizer::AnisotropicCellRelax cell(/*P_target*/ 0.0, /*alpha*/ 1.0);
cell.bulk_mod = 0.20;          // offline E(a) curvature [eV/Å³]
cell.max_scale = 0.05;
cell.relax_y = false;
cell.relax_z = false;
opt.set_optimizer_ref(cell);
opt.add_criterion(cpu::StressConverged{0.0, /*stol=*/5e-4,
                                       /*x*/ true, /*y*/ false, /*z*/ false});
opt.on(minimizer::Phase::force).add_ref(eval);
opt.build();
opt.run(frame, 6);

// a = frame.box.length(0); E_ha = eval.last_energy_hartree();
// σ_xx = frame.meta.get(kStress)[0] / frame.box.volume();   // eV/Å³
```

Headers involved: `gpw_evaluator.h` (`compute_stress`, `stress_use_fd`,
`method()`, `last_stress_ha_bohr3()`), `gpw_solver.h`
(`GpwDft::stress_strain_h`, `GpwDft::stress_axes`),
`anisotropic_cell_relax.h` (diagonal cell optimizer + axis mask),
`criteria.h` (`StressConverged`), `minimize.h`; the analytic fixed-\(P\)
stress and its FD oracle live in `src/cpu/gpw/internal/gpw_stress.cpp`.

### Validation (vs CP2K)

Oracle: offline **CP2K 2026.1** Quickstep GPW (`cp2k.psmp`, EasyBuild
`2026.1-gmpflf-2026.03`), Arrhenius epyc login, 1 rank × 4 OMP
(`OMP_NUM_THREADS=4`, `OMP_PROC_BIND=close`). Twin `chain_cellopt.inp`
(SR-MOLOPT, `CELL_OPT`, `CONSTRAINT YZ`, `FIXED_ATOMS`, analytical stress).

#### Headline pin — SR-MOLOPT CELL_OPT

| Quantity | a₀ = 3.328 (scan start) | a₀ = 3.24 (fair twin) |
| -------- | ----------------------: | --------------------: |
| \(a^\star\) [Å] | **3.200237** | 3.200292 |
| \(E^\star\) [Ha] | **−17.184075549824836** | −17.184075557775 |
| Wall (CP2K timing table) | 9.55 s | **5.67 s** |

The regression hard-codes \(a^\star = 3.200237\),
\(E^\star = -17.184075549824836\) Ha (a₀ = 3.328 oracle run; \(a^\star\)
agrees with the a₀ = 3.24 fair twin to \(< 10^{-4}\) Å).

#### Secondary pin — full DZVP-MOLOPT-GTH CELL_OPT (not gated)

| Quantity | Value |
| -------- | ----: |
| \(a^\star\) | 3.200252 Å |
| \(E^\star\) | −17.185176472246 Ha |
| Wall (timing table) | 23.5 s |

#### \(E(a)\) curvature (full MOLOPT ENERGY scan; `bulk_mod` source)

| \(a\) [Å] | \(E\) [Ha] |
| --------: | ---------: |
| 3.16 | −17.185048204416361 |
| 3.20 | −17.185176430631692 |
| 3.24 | −17.185069177601907 |

\(d^2E/da^2 \approx 0.147\,\mathrm{Ha/Å}^2\) →
\(d\sigma_{xx}/d\varepsilon \approx 0.20\,\mathrm{eV/Å}^3\) →
`bulk_mod = 0.20`.

#### Hard gates (Release, 4 OMP)

| Gate | Tolerance | Measured |
| ---- | --------- | -------- |
| relax converged | — | `stress_converged` ✓ |
| \(\lvert a_{\mathrm{ATV}}-a^\star_{\mathrm{CP2K}}\rvert\) | ≤ 0.02 Å | **0.0012 Å** (3.201465 vs 3.200237) |
| \(b\), \(c\) untouched | bit-identical | 8.0 / 8.0 ✓ |
| residual \(\lvert\sigma_{xx}\rvert\) | ≤ \(6\times 10^{-4}\) eV/Å³ (1.2 × stol) | **5.1×10⁻⁵** eV/Å³ |
| \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}\rvert\) | ≤ \(1\times 10^{-3}\) Ha | **1.4×10⁻⁴** Ha |
| wall | ≤ 60 s | **42.5 s** |

Offline oracle sketch (host with the CP2K module; not ctest):

```bash
export CP2KROOT=/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.1-gmpflf-2026.03
export CP2K_DATA_DIR=$CP2KROOT/share/cp2k/data
export PATH=$CP2KROOT/bin:$PATH
export OMP_NUM_THREADS=4
ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
mpirun -np 1 cp2k.psmp -i chain_cellopt.inp -o cellopt.out
# CELL | Vector a  → a* ; ENERGY| Total FORCE_EVAL → E*
```

### Performance

ATV: the full outer `CELL_OPT` with analytic stress per SCF (**42.5 s**).
CP2K fair denominator: the same a₀ = **3.24 Å** `CELL_OPT` twin wall from the
timing table (**5.67 s**), not the longer a₀ = 3.328 scan start. Same host
class, 1 rank × 4 OMP.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | -----------------------: | ----: | ---- |
| Water-chain CELL_OPT, SR-MOLOPT, a₀=3.24 | **5.67** | **42.5** | **7.5** | ✓ |

`perf_exempt = false`, `within_10x = true`. Cost drivers vs CP2K: periodic
MOLOPT collocation + several outer stress-SCF cycles; the analytic
fixed-\(P\) stress removes the multi-SCF strain-FD cost that keeps the ice
section on compact GTH-PADE.

### Limitations

- **SR-MOLOPT twin** — full DZVP-MOLOPT-GTH `CELL_OPT` is pinned offline but
  does not fit the ≤ 60 s budget (periodic collocation).
- **Ions fixed** — no joint `GEO_OPT` + `CELL_OPT`; proves the cell machinery
  and the GPW stress axes mask, not a production recipe.
- **Orthorhombic diagonal only** — no triclinic / shear; masked axes are hard
  zeros on the published stress. No isotropic `IsotropicCellRelax` pin here
  (isotropic `CELL_OPT`: [pw-variable-cell-si](pw-variable-cell.md#pw-variable-cell-si)).
- **Closed-shell Γ RKS** — no UKS / k-point cell relax (k-point stress is
  [gpw-kpoint-stress](gpw-rks.md#gpw-kpoint-stress)); no D3/D4/TS/MBD on the
  cell path.
- Multi-SCF strain-FD remains available via `stress_use_fd` but is not the
  product path; no CUDA GPW.
- Public Frame energy is **eV**; Ha pins are internal / citation channels
  only. Unit tests under `tests/cpu/minimize/` exercise the axis masks.
