# Plane-wave variable-cell relaxation

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Isotropic cell relaxation (CELL_OPT) on top of the periodic plane-wave KS-DFT
engine. After each force-phase SCF the directly mounted `cpu::pw::PW` writes the
Nielsen–Martin analytic stress to `kStress` (the plan consumes stress, so one
production SCF supplies both energy and stress); `cpu::minimizer::Minimize`
drives `cpu::minimizer::IsotropicCellRelax` on it and `cpu::StressConverged`
terminates on the residual hydrostatic stress. The section covers the CPU
backend: relaxed lattice constant, residual stress and Frame energy of diamond
Si.

Case: `ctest -R '^pw/variable-cell$'` — sources and oracle record in [`regressions/pw/variable-cell/`](../../../regressions/pw/variable-cell/). One section of the case per `##` heading below: `atv_regressions --case pw/variable-cell --section <name>`.

Roadmap: [optimization.md](../../roadmap/optimization.md) § cell relax;
[dft.md](../../roadmap/dft.md) § PW CELL_OPT.

## pw-variable-cell-si — Isotropic cell relax of diamond Si (NC-UPF LDA) {#pw-variable-cell-si}

### Summary

Diamond Si (2-atom fcc primitive), ONCVPSP-LDA NC-UPF, incomplete tutorial
basis (\(E_\mathrm{cut}=15\) Ry, \(2\times2\times2\) MP). The expanded start
\(a_0=5.58\) Å shrinks toward the ATV zero-stress basin \(a^\*=5.510\) Å.
Absolute \(a^\*\) / energy are **not** a CP2K Quickstep GTH twin — the CP2K
run is methodology input only. Performance: **`perf_exempt`**.

### Method

**Level.** Same periodic plane-wave KS-DFT engine as
[pw-lda-si](pw-rks.md#pw-lda-si) / [pw-analytic-stress-si](pw-rks.md#pw-analytic-stress-si).
Cell stress comes from the Nielsen–Martin decomposition (Nielsen & Martin,
*Phys. Rev. B* **32**, 3780 (1985)); the optimizer does not run a second SCF
for stress.

**Stress convention.** PW / GPW intensive stress is **mechanical / CP2K**:

\[
\sigma_{\alpha\beta}
  = -\frac{1}{\Omega}\,
    \frac{\partial E}{\partial\varepsilon_{\alpha\beta}},
\qquad
\sigma = (\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})
\quad [\mathrm{eV}/\mathrm{\AA}^3].
\]

`kStress` holds the extensive Voigt virial σ·Ω [eV]; the regression divides by
the cell volume. Positive diagonal stress under compression drives
`IsotropicCellRelax` to **expand** the box (and vice versa). The
energy-derivative pin signs of `pw-analytic-stress-si` are flipped to this
convention on the relax path. Hydrostatic pressure for the isotropic step:

\[
P = \frac{\sigma_{xx}+\sigma_{yy}+\sigma_{zz}}{3}
\quad [\mathrm{eV}/\mathrm{\AA}^3].
\]

**Cell motion.** Fixed fractional coordinates; isotropic **full-matrix** scale
via `Box::scale_isotropic` (required for the fcc primitive whose diagonal is
zero). The conventional cubic lattice constant is \(a = 2\,\mathrm{cell}_{01}\)
(off-diagonal entries \(= a/2\)).

**Optimizer / criterion.**

| Piece | Role |
| ----- | ---- |
| `IsotropicCellRelax(P_target, alpha)` | steepest-descent scale toward \(P_\mathrm{target}\) |
| `bulk_mod`, `max_scale` | soft \(B\) and per-step \(\lvert s-1\rvert\) cap (damp flat incomplete-basis EOS) |
| `StressConverged{0, stol}` | stop when \(\max_d\lvert\sigma_{dd}-\sigma^\*\rvert \le \mathrm{stol}\) |

**XC / spin / occupations.** `XCType::LSDA`; restricted closed shell; no metallic
smearing. **BZ sampling.** Uniform Monkhorst–Pack `k_mesh`³ (\(2\times2\times2\)).

**Observables.**

| Symbol | Definition | Unit |
| ------ | ---------- | ---- |
| \(a\) | Conventional cubic lattice constant | Å |
| \(a^\*\) | Zero-stress basin pin for this twin | Å |
| \(\sigma_v\) | Intensive Voigt stress (`kStress` / Ω) | eV/Å³ |
| \(P\) | Hydrostatic \((\sigma_{xx}+\sigma_{yy}+\sigma_{zz})/3\) | eV/Å³ |
| \(E\) | Frame meta energy (`kEnergy`) | eV |

### Implementation

```cpp
constexpr double kA0 = 5.58;      // start expanded [Å]
constexpr double kStol = 1.5e-2;  // residual stress gate [eV/Å³]

cpu::Frame frame(2);
set_diamond_si(frame, kA0);       // fcc primitive: type 14, mass 28.0855,
                                  // atom 1 at (a/4.0)(1,1,1), cell (a/2)(0,1,1)…

PW method{{std::make_shared<UpfPseudopotential>(ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf")}, /*ecut Ry*/ 15.0,
          /*k_mesh 2×2×2*/ 2, cpu::scf::XCType::LSDA};
method.tol = 1e-6;
method.max_iter = 40;

cpu::minimizer::Minimize opt;
cpu::minimizer::IsotropicCellRelax cell(/*P_target*/ 0.0, /*alpha*/ 1.0);
cell.bulk_mod = 3.5;           // eV/Å³ — soft B damps incomplete-basis EOS
cell.max_scale = 0.012;        // |s−1| cap per step
opt.set_optimizer_ref(cell);
opt.add_criterion(cpu::StressConverged{0.0, kStol});
opt.on(minimizer::Phase::force).add(std::move(method));  // stress: the plan consumes it
opt.build();
opt.run(frame, 3);  // seed force + up to 2 scale steps

const double a1 = 2.0 * static_cast<double>(frame.box.cell()(0, 1));
const auto& w = frame.meta.get(kStress);      // Voigt virial σ·Ω [eV]
const double e_ev = frame.meta.get(kEnergy);  // eV
```

Headers: `cpu/minimize/minimize.h`, `cpu/minimize/cell_relax.h`
(`IsotropicCellRelax`), `cpu/driver/criteria.h` (`StressConverged`),
`cpu/planewave/pw_solver.h` (`PW`). The UPF is the in-tree
`tests/data/pw/Si_lda_nc.upf`.

### Validation

The primary gate is the **ATV residual-stress basin** on the same NC-UPF twin
(incomplete ecut/k), hard-coded in the section source. CP2K Quickstep supplies
an offline **CELL_OPT methodology** twin only (GTH ≠ ONCV).

#### Pseudopotential

| Item | Value |
| ---- | ----- |
| File | `tests/data/pw/Si_lda_nc.upf` (in-tree) |
| Family | PseudoDojo ONCVPSP-LDA-SR-noCC v0.4 |
| Generator | ONCVPSP 3.3.0 (D. R. Hamann), \(Z_\mathrm{val}=4\), no NLCC |

#### Geometry and settings

| Setting | Value |
| ------- | ----- |
| Structure | diamond Si, 2-atom fcc primitive |
| Start \(a_0\) | **5.58 Å** (expanded) |
| Basis atoms | \((0,0,0)\), \((a/4,a/4,a/4)\) |
| Cell | \((a/2)(0,1,1)\), \((a/2)(1,0,1)\), \((a/2)(1,1,0)\) |
| \(E_\mathrm{cut}\) | 15 Ry (tutorial; ONCV production ~60 Ry) |
| k-mesh | \(2\times 2\times 2\) (`k_mesh = 2`) |
| SCF tol / `max_iter` | \(10^{-6}\) / 40 |
| XC | LSDA |
| Relax | α=1, `bulk_mod`=3.5 eV/Å³, `max_scale`=0.012, budget 3 |
| Criterion | `StressConverged{0, 1.5×10^{-2} eV/Å³}` |

#### Accuracy and gates (Release, OMP=8)

| Quantity | Atomiverse / pin | Reference | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| Relaxed conventional \(a^\*\) | **5.510 Å** | ATV residual-stress basin (this ecut/k) | ≤ 0.05 Å | Hard pin |
| Measured relaxed \(a\) | **5.5097 Å** | pin 5.510 | \(\lvert a-a^\*\rvert\) = **3×10⁻⁴ Å** | |
| Shrink from expanded start | \(a_1 < a_0\) (**5.5097 < 5.58**) | qualitative | \(a_1 < 5.58-10^{-4}\) | Direction gate |
| Residual \(\max_d\lvert\sigma_{dd}\rvert\) | **~8.0×10⁻³** eV/Å³ | 0 (target) | ≤ 1.2×stol = 1.8×10⁻² | Mechanical intensive stress |
| Frame energy | **≈ −213.27 eV** | finite | finite | Public eV on Frame |
| Mechanical stress at \(a=5.43\) | σ ≈ **+0.0576** eV/Å³ | `pw-analytic-stress-si` twin | — | Compressed vs \(a^\*\); drives expansion if started there |
| Wall | **50.2 s** | — | ≤ 60 s | |

**CP2K note.** Quickstep bulk-Si CELL_OPT uses **GTH-PADE + DZVP-GTH**, not
ONCV plane-wave, so absolute \(a^\*\) / energy are **not** comparable to this
NC-UPF twin. Methodology input (KEEP_ANGLES / KEEP_SYMMETRY / FIXED_ATOMS,
\(2\times2\times2\), start \(a_0=5.55\) Å), offline wall / workflow only:

```bash
module load buildenv-gcc/2026.03-mpich
export CP2KROOT=…/CP2K/2026.1-…   # site install
export CP2K_DATA_DIR=$CP2KROOT/share/cp2k/data
export PATH=$CP2KROOT/bin:$PATH
export OMP_NUM_THREADS=4
mpirun -np 1 cp2k.psmp -i si_cellopt.inp -o cellopt.out
# wall only — absolute a* not comparable to NC-UPF
```

**EOS context (not the hard \(a^\*\) pin).**

| Quantity | Value | Source |
| -------- | ----- | ------ |
| EOS \(a_0\) | ≈ 5.468 Å | PseudoDojo `Si.djrepo` |
| Twin \(a^\*\) (this ecut/k) | 5.510 Å | ATV residual-stress pin |
| Incomplete-basis shift | \(a^\* > a_0^\mathrm{EOS}\) | Expected at 15 Ry / \(2^3\) k |

### Performance

**`perf_exempt`**: there is no fair NC-UPF PW CELL_OPT twin (CP2K Quickstep
GTH-PADE ≠ ONCV PW; absolute \(a^\*\) is not a CP2K digit twin), so
\(T_\mathrm{cp2k}\), the ratio and the ≤10× check are n/a. ATV wall
(informative, ≤3 force launches): **50.2 s** (Release, OMP=8), inside the
≤ 60 s single-rank CPU budget.

### Limitations

- Tutorial ecut/k only — \(a^\*\) is incomplete-basis relative to the
  PseudoDojo EOS \(a_0\approx 5.468\) Å; not a production lattice constant.
  Production ONCV cutoffs (~60 Ry) and dense k-meshes exceed the ≤ 60 s budget.
- Soft `bulk_mod` / small `max_scale` and the 3-iteration budget are tuned for
  the flat incomplete-basis EOS and the ≤ 60 s gate — not a general production
  CELL_OPT schedule.
- Absolute CP2K Quickstep CELL_OPT lattice is a different PP/basis world — the
  fair comparison is structural-response methodology, not digit-for-digit
  \(a^\*\).
- Isotropic full-matrix scale only: anisotropic / constrained CELL_OPT is
  `AnisotropicCellRelax` (GPW `gpw-variable-cell-ice`, axes mask); no joint
  ion+cell relax, internal coordinates or PR NPT dynamics.
- GGA (PBE) cell relax, MPI multi-rank CELL_OPT, and denser k / higher ecut
  are not covered.
- The CUDA PW surface is not covered by this section.

### References

- O. H. Nielsen and R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985)
- D. R. Hamann, *Phys. Rev. B* **88**, 085117 (2013)
- M. J. van Setten et al., *Comput. Phys. Commun.* **226**, 39 (2018)
- PseudoDojo ONCVPSP-LDA-SR-noCCv0.4 `Si.djrepo` (EOS context)
- Related sections: [pw-analytic-stress-si](pw-rks.md#pw-analytic-stress-si),
  [pw-lda-si](pw-rks.md#pw-lda-si)
- [Chapter 8 — Plane-Wave DFT](../ch08-plane-wave.md),
  [Chapter 12 — Geometry optimization](../ch12-geometry-optimization.md)
