# GPW DFT-D3 dispersion

Grimme's DFT-D3 London correction on the GPW path, on top of PBE: two-body
zero ("original") damping, two-body Becke–Johnson (rational) damping, and the
optional Axilrod–Teller–Muto (ATM) three-body C9 term. D3 is geometry-only —
it never enters the Kohn–Sham potential — so it is reported on
`GpwResult::e_disp` [Hartree], folded into `e_total`, and its exact gradient
and strain derivative are added to the analytic ionic forces and cell stress.

The switches are members of `cpu::gpw::GpwDft` (no `*Config` bag): `d3`
(zero damping), `d3bj` (BJ damping), `d3_atm` (C9 on top of whichever
two-body flag is set; ignored when neither is), and `d3_rcut_bohr` (pair /
triple hard cutoff, default `kD3Cp2kPairCutoffBohr`). `d3` and `d3bj` are
mutually exclusive with each other and with `d4` / `ts` / `mbd` / `xdm`; any
two set throws at run. Energies come from `gpw.build(frame)` +
`gpw.run(frame)`; forces from `GpwDft::forces(frame)` or
`cpu::gpw::GPWEvaluator(std::move(dft), /*compute_forces=*/true)`; stress from
`GPWEvaluator` with `compute_stress = true`, `stress_use_fd = false` on a
`cpu::Evaluate` driver (`last_stress_ha_bohr3()` [Ha/Bohr³], Frame `kStress`
= σ·Ω [eV]), or `GpwDft::stress`, with the free oracle `gpw_stress_fd` as the
multi-SCF strain-FD twin.

The sections cover the CPU backend only: dispersion energy
([gpw-d3-atm](#gpw-d3-atm), [gpw-pbe-d3bj-ice](#gpw-pbe-d3bj-ice)), analytic
ionic forces ([gpw-d3-forces](#gpw-d3-forces),
[gpw-d3bj-forces](#gpw-d3bj-forces), [gpw-d3-atm-forces](#gpw-d3-atm-forces))
and analytic cell stress ([gpw-d3-stress](#gpw-d3-stress),
[gpw-d3bj-stress](#gpw-d3bj-stress), [gpw-d3-atm-stress](#gpw-d3-atm-stress)),
so both damping functions × both boundary conditions are pinned for the
force. Build in the CPU environment (`module load buildenv-gcc/2026.03-mpich`,
see `CLAUDE.md`) with `-DATV_BUILD_GPW=ON -DATV_BUILD_REGRESSIONS=ON`.

Case: `ctest -R '^gpw/d3$'` — sources and oracle record in [`regressions/gpw/d3/`](../../../regressions/gpw/d3/). One section of the case per `##` heading below: `atv_regressions --case gpw/d3 --section <name>`.

## Method

### Two-body zero damping

DFT-D3 zero damping (Grimme et al., *J. Chem. Phys.* **132**, 154104
(2010)), matching CP2K `qs_dispersion_d3`:

\[
E_{\mathrm{disp}}^{(2)} = -\sum_{A<B}\sum_{n=6,8}
  s_n\,\frac{C_n^{AB}}{R_{AB}^{n}}\,f_{d,n}(R_{AB}),
\quad
f_{d,n}(R)=\frac{1}{1+6\bigl(R/(s_{r,n} R_0^{AB})\bigr)^{-\alpha_n}},
\]

with \(\alpha_6=14\), \(\alpha_8=16\),
\(C_8^{AB}=3\,C_6^{AB}\,r_{2r4}^{A} r_{2r4}^{B}\), and CN-interpolated
reference \(C_6\) (\(k_3=-4\)). The coordination-number dependence of
\(C_6\) makes the gradient non-trivial.

**PBE-D3 (zero) parameters** (CP2K `REFERENCE_FUNCTIONAL PBE`):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(s_6\) | 1.0 | dimensionless |
| \(s_8\) | 0.722 | dimensionless |
| \(s_{r,6}\) | 1.217 | dimensionless |
| \(s_{r,8}\) | 1.0 | dimensionless |

### Two-body Becke–Johnson damping

DFT-D3(BJ) (Grimme, Ehrlich, Goerigk, *J. Comput. Chem.* **32**, 1456
(2011)), periodic lattice sum over image vectors \(\mathbf{T}\):

\[
E_{\mathrm{disp}} = -\sum_{A<B,\,T}\left[
  \frac{s_6\,C_6^{AB}}{R_{AB,T}^6 + f^6}
  + \frac{s_8\,C_8^{AB}}{R_{AB,T}^8 + f^8}
\right],
\quad
f = a_1\sqrt{C_8^{AB}/C_6^{AB}} + a_2.
\]

**PBE-D3(BJ) parameters** (CP2K `REFERENCE_FUNCTIONAL PBE`):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(s_6\) | 1.0 | dimensionless |
| \(s_8\) | 0.7875 | dimensionless |
| \(a_1\) | 0.4289 | dimensionless |
| \(a_2\) | 4.4407 | Bohr |

### Three-body ATM / C9

CP2K `CALCULATE_C9_TERM`:

\[
C_9^{ABC} = -\sqrt{C_6^{AB}\,C_6^{BC}\,C_6^{CA}},
\quad
E_9 \;\text{contrib.}\; \propto -s_9\,f_d\,C_9\,\mathrm{ang}\big/
  (R_{AB}^{2} R_{BC}^{2} R_{CA}^{2})^{3/2},
\]

with \(s_9 = s_6\), damping `damping_d3(rabc, r0, 4/3, 16)`, angle factor
from CP2K's \(s_1\cdot s_2\cdot s_3\) form, and half-neighbor-list triple
enumeration. A smooth fifth-order cutoff zeroes each term over
\([r_{\mathrm{cut}}-1,\, r_{\mathrm{cut}}]\). In periodic cells the triple
loop covers CP2K's `doabc` lattice-image classes (AB home + C off-home; B and
C off-home with `fac0 = 1/3`) on `pbc()`-wrapped coordinates (`ANINT`
parity, `accumulate_atm_c9`). ATM contributes **no** `D3Result::virial`; the
fixed-\(P\) stress path rebuilds the energy under strain instead.

### Coordination numbers and pair cutoff

- **CN form (CP2K parity).** The GPW path uses CP2K coordination numbers
  (`cnparam_d3`, including the long-range
  \(\tfrac12(1-\tanh(R-2 r_{\mathrm{cov}}))\) factor) so \(E_{\mathrm{disp}}\)
  matches offline `TYPE DFTD3` / `TYPE DFTD3(BJ)`. Molecular `cpu::scf::D3BJ`
  keeps the original Grimme CN (PySCF / simple-dftd3 twin) — do not mix the
  two when comparing to PySCF.
- **Pair / triple cutoff.** `kD3Cp2kPairCutoffBohr` = 2 × `R_CUTOFF 20` Å =
  75.589 Bohr (≈ 75.59; CP2K prints `Cutoff Radius [Bohr]: 37.79` for
  `R_CUTOFF 20`). A periodic D3-zero CP2K run fixes the convention: 75.589
  Bohr reproduces CP2K's `E_disp` to 6.7e−11 Ha, a 40 Bohr cutoff misses by
  3.9e−7. Energy, force and stress all fold the one configured value, so they
  differentiate the same functional; `d3_rcut_bohr` matches any CP2K D3 input
  (e.g. `R_CUTOFF 10.0` → 37.79 Bohr). On an isolated cell the convention does
  not move the pin; on a periodic cell it does. Twins use
  `LONG_RANGE_CORRECTION .FALSE.`.

### Electronic host

Γ-RKS GPW Kohn–Sham (VandeVondele et al., *Comput. Phys. Commun.* **167**,
103 (2005)), PBE XC, `DZVP-MOLOPT-SR-GTH` + GTH-PBE PPs. Energy-only sections
run a short SCF just to materialize `e_disp` on a real `GpwResult`; absolute
QS \(E_{\mathrm{tot}}\) micro-Ha parity is out of scope (OT/grid/DIIS
differences).

**Martyna–Tuckerman placement.** With `nonperiodic = true` the collocation
does not wrap, so a Gaussian near a cell face is clipped: the energy stays
plausible while the forces gain a boundary term. `GpwDft::build` throws on an
off-centre nonperiodic system; at CP2K's origin-based coordinates
`max|F_atv − F_cp2k|` is **31.9 Ha/Bohr** versus 7.5e−3 with the dimer
centred. Isolated sections therefore translate the dimer to the cell centre
(E_disp is translation invariant) and assert \(\lvert\sum F\rvert\).

### Pins as a difference of two CP2K runs

Because D3 never enters the KS potential, two CP2K runs on the same geometry
that differ only in the dispersion block converge to the same density, so

\[
F_A(\mathrm{D3}) - F_A(\mathrm{plain}) = -\nabla_A E_{\mathrm{disp}},
\]

and the `STRESS_TENSOR NUMERICAL` difference is the strain derivative of
\(E_{\mathrm{disp}}\) alone: the much larger electronic force / stress (and
CP2K's FD truncation) cancels. ATV isolates its own dispersion term the same
way — the product run minus a plain twin — and the **wiring** gate asserts
that difference equals the standalone D3 kernel.

### Analytic fixed-\(P\) cell stress

One SCF at the unstrained cell → \(P\), \(W\). Central FD of the
non-self-consistent Lagrangian under orthorhombic diagonal strain at fixed
fractional coordinates, with the geometry-only dispersion rebuilt at each
strained cell (Γ + multi-k, `src/cpu/gpw/internal/gpw_stress.cpp`,
`energy_fixed_P` / `energy_fixed_P_k`):

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
Multi-SCF `gpw_stress_fd` on the same method is the internal dual-gate twin,
never the product default.

```cpp
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 200;
dft.tol = 1e-7;              // matches CP2K EPS_SCF twin
dft.mix = 0.3;
dft.nonperiodic = false;     // PERIODIC XYZ
dft.d3 = true;               // or d3bj; exclusive vs d4/ts/xdm/mbd

cpu::gpw::GPWEvaluator eval(std::move(dft));
eval.compute_stress = true;  // product analytic fixed-P
eval.stress_use_fd = false;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);

const double e_disp = eval.last_result().e_disp;   // [Ha]
const auto& san = eval.last_stress_ha_bohr3();     // [Ha/Bohr³] Voigt
// frame.meta.get(kStress)                          // [eV] (σ·Ω)
```

## Common setup

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair; not bulk ice Ih) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) (isolated runs translated to the cell centre) |
| Cells | ABC **12.0 12.0 12.0 Å** PERIODIC XYZ; 12 Å **PERIODIC NONE** (MT, `nonperiodic = true`); compressed 6 Å PERIODIC XYZ for the ATM image-triple pins |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + disp | PBE + `TYPE DFTD3` / `TYPE DFTD3(BJ)`, `REFERENCE_FUNCTIONAL PBE`; `R_CUTOFF 20.0` (Å in twin; ATV rcut = 2× ≈ 75.59 Bohr) unless stated |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |

Mounted on a driver (energy / forces; same members):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GpwDft dft{kinds, 300.0, ATV::cpu::gpw::XcKind::PBE};
dft.max_iter = 200;  dft.tol = 1e-7;  dft.mix = 0.3;
dft.nonperiodic = false;
dft.d3bj = true;     // or d3 (+ d3_atm)
ATV::cpu::gpw::GPWEvaluator eval(std::move(dft));
// bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
```

| Path | Role |
| ---- | ---- |
| `gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `d3`, `d3bj`, `d3_atm`, `d3_rcut_bohr`, `GpwDft::stress` / `gpw_stress_fd` |
| `gpw_evaluator.h` | `GPWEvaluator` (`compute_forces`, `compute_stress`, `last_stress_ha_bohr3`) |
| `dftd3.h` | `dftd3_pbe(..., calculate_c9)`, `dftd3_pbe_bj`, `kD3Cp2kPairCutoffBohr` (private) |
| `dftd3_params.h` | In-tree D3 tables (`D3Params{}`) |
| `src/cpu/gpw/internal/dftd3.cpp` | `accumulate_atm_c9` (CASE 1/2/3 image triples, pbc anchoring) |
| `src/cpu/gpw/internal/gpw_stress.cpp` | Fixed-\(P\)+Pulay Lagrangian; D3 / D3(BJ) / ATM rebuilt under strain |

No unit test in the tree compares analytic D3 / D3(BJ) / ATM forces or stress
against finite differences; the analytic-vs-FD values quoted below are recorded.

## Oracle

Offline **CP2K 2026.1** (`cp2k.psmp`, EasyBuild `2026.1-gmpflf-2026.03`,
git:5e54ba2) on the Arrhenius epyc login node (`mpirun -np 2`,
`OMP_NUM_THREADS=2`) — external engine pins, not goldens of Atomiverse's own
output; CP2K is **never** launched from ctest. Inputs and re-run recipes are
in the case `REFERENCE.md`. Pins are CP2K's printed `Dispersion energy` line,
differences of `FORCES|` tables (`ENERGY_FORCE`), and differences of
`STRESS_TENSOR NUMERICAL` diagonals (bar; converted at
1 Ha/Bohr³ = 2.94210107994716×10⁸ bar). CP2K's numerical stress uses the
`&NUMERICAL_STRESS` default `DX 0.001` Bohr.

Energy-only sections ([gpw-d3-atm](#gpw-d3-atm),
[gpw-pbe-d3bj-ice](#gpw-pbe-d3bj-ice)) are **perf_exempt**: D3 evaluation is
\(O(N^2)\) (+ \(O(N^3)\) for ATM) over a 6-atom cell (microseconds), the CP2K
wall is a full Quickstep run while the hard gate needs only a short SCF to
materialize `e_disp`, and a full twin SCF ratio depends on cutoff/OT choices.
A strict full-SCF twin applies only if product policy requires
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\). Force and stress sections
carry fair walls against the CP2K twin.

## Limitations

- **Water-dimer cells**, not bulk ice Ih — lattice-sum / image consistency
  pins, not structural benchmarks (no ice Ih supercell, no phonons).
- **Absolute GPW \(e_{\mathrm{tot}}\) vs CP2K QS** is not asserted.
- **PBE parameters only**; other reference functionals share tables but are
  not e2e-gated.
- **Orthorhombic diagonal stress only**; shear / triclinic out of scope.
- **Γ-RKS pins**; the multi-k fold shares the same `e_disp` branch but has no
  dedicated pin; no UKS.
- **D3(BJ)+ATM** is unpinned (CP2K supports it; the fold is shared).
- No CUDA GPW.
- DFT-D4 and its ATM: [gpw-d4-atm](gpw-d4.md#gpw-d4-atm),
  [gpw-d4-atm-stress](gpw-d4.md#gpw-d4-atm-stress) (no CP2K pin possible —
  Arrhenius CP2K lacks libdftd4).

## References

- S. Grimme, J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.* **132**,
  154104 (2010) (DFT-D3, zero damping, ATM)
- S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **32**, 1456 (2011)
  (D3(BJ) damping)
- B. M. Axilrod, E. Teller, *J. Chem. Phys.* **11**, 299 (1943); Y. Muto,
  *Proc. Phys.-Math. Soc. Jpn.* **17**, 629 (1943) (three-body ATM /
  triple-dipole term)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
  (Quickstep GPW)
- Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985) (stress context)
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Related: [gpw-d4-stress](gpw-d4.md#gpw-d4-stress),
  [gpw-d4-atm-forces](gpw-d4.md#gpw-d4-atm-forces),
  [gpw-mbd-stress](gpw-mbd.md#gpw-mbd-stress),
  [gpw-stress-water](gpw-rks.md#gpw-stress-water)

## gpw-d3-forces — D3 zero-damping ionic forces, isolated water dimer {#gpw-d3-forces}

D3 zero damping contributes to the **ionic forces** as the exact gradient of
the same additive \(E_{\mathrm{disp}}\) the SCF converged, including the CN
dependence of \(C_6\). Twinned to two offline CP2K `ENERGY_FORCE` runs (with
and without `&VDW_POTENTIAL`) on the isolated (MT) water dimer.

```cpp
cpu::gpw::GpwDft m{{O, H}, 300.0, cpu::gpw::XcKind::PBE};  // CP2K CUTOFF 300
m.max_iter = 120;
m.tol = 1e-7;            // CP2K EPS_SCF 1e-7
m.mix = 0.4;
m.nonperiodic = true;    // CP2K PERIODIC NONE + PSOLVER MT
m.d3 = true;
cpu::gpw::GPWEvaluator eval(std::move(m), /*compute_forces=*/true);
// plain twin: same method with d3 = false, GpwDft::forces(frame_plain)
```

### Validation

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs CP2K | 1e−8 Ha | **6.4e−11** |
| \(F_{\mathrm{disp}}\) vs CP2K difference | 5e−6 Ha/Bohr | **7.8e−07** |
| Total \(F\) vs CP2K | 1e−2 Ha/Bohr | **7.5e−03** |
| \(\lvert\sum_A F_A\rvert\) | 1e−3 Ha/Bohr | **2.2e−04** |
| Wiring \((F_{d3}-F_{plain})\) vs standalone | 1e−7 Ha/Bohr | **9.1e−11** |
| Wall | ≤ 60 s | **17.8 s** (CP2K 12.3 s → ratio 1.4) |

Analytic vs multi-SCF energy FD agreed to **2.4e−6 Ha/Bohr** on cheap MT H₂
(recorded; no unit test in the tree gates it; 6N SCFs on this cell exceed the
60 s budget). The pair-cutoff convention and the MT placement rule
([Method](#method)) are both settled on this
twin.

### Performance

CP2K `ENERGY_FORCE` 12.3 s vs Atomiverse 17.8 s on the same host → ratio
**1.4**, inside the 10× fair wall. The Atomiverse figure includes one extra
plain SCF that only the wiring gate needs.

## gpw-d3-stress — D3 zero-damping cell stress, periodic water dimer {#gpw-d3-stress}

Fixed-\(P\) orthorhombic cell stress on Γ-point GPW PBE with two-body D3 zero
damping (`CALCULATE_C9_TERM .FALSE.`) on the 12 Å periodic water dimer (same
cell as the [gpw-d4-stress](gpw-d4.md#gpw-d4-stress) dimer). Lattice sums run
to `kD3Cp2kPairCutoffBohr` = 75.589 Bohr — on this periodic cell the cutoff
convention moves the pin. CP2K runs `TYPE DFTD3`, so the gate is a real CP2K
`STRESS_TENSOR NUMERICAL` difference pin (`water_dimer_d3_stress.inp` −
`water_dimer_plain_stress.inp`); the decisive internal residual is product
analytic vs multi-SCF `gpw_stress_fd` on the same `d3=true` method. The
regression isolates ATV's dispersion stress with a second warm-started
analytic product with `d3=false`, subtracted (snippet in
[Method](#method)).

| Section | Deliverable |
| ------- | ----------- |
| [gpw-d3-forces](#gpw-d3-forces) | D3-zero **forces** (isolated) + periodic E_disp pin + cutoff convention |
| [gpw-d4-stress](gpw-d4.md#gpw-d4-stress) | D4 stress, dual-gate only (no CP2K DFTD4 build) |
| [gpw-mbd-stress](gpw-mbd.md#gpw-mbd-stress) | Γ MBD stress |
| **gpw-d3-stress** | D3-zero stress with a real **CP2K stress-difference pin** |

### Validation

CP2K printed (Hartree): total (plain) −34.338293290245112, total (D3)
−34.339114184921726, `Dispersion energy` **−0.00082089467662**. The hard pin
is σ_num(D3) − σ_num(plain).

ATV: 11 SCF iters, E_tot = −34.331152296855 Ha (single 200 Ry grid vs CP2K
NGRIDS 4 / REL_CUTOFF 40 — geometry-only pins are cutoff-independent).

| Gate | CP2K pin | Tol | ATV measured |
| ---- | -------- | --- | ------------ |
| \(E_{\mathrm{disp}}\) [Ha] | **−0.00082089467662** | 1×10⁻⁸ | −8.20894731919×10⁻⁴ (\|Δ\| = 5.53×10⁻¹¹) |
| Δσ_xx [Ha/Bohr³] | **−1.70641140246×10⁻⁸** | 2×10⁻⁹ | \|Δ\| = 2.27×10⁻¹⁰ |
| Δσ_yy [Ha/Bohr³] | **−9.03962616420×10⁻¹⁰** | 2×10⁻⁹ | \|Δ\| = 2.26×10⁻¹² |
| Δσ_zz [Ha/Bohr³] | **+3.74278507120×10⁻⁹** | 2×10⁻⁹ | \|Δ\| = 1.92×10⁻¹¹ |
| max\|σ_an − σ_fd\| diag (internal FD twin, d3=true) | — | 2×10⁻⁵ | 7.39×10⁻⁷ (xx; yy 2.46×10⁻⁸, zz 1.24×10⁻⁹) |
| max\|σ_tot − σ_cp2k\| diag (sanity: single-grid vs multigrid) | soft: (1.85034×10⁻⁵, −3.62033×10⁻⁷, 4.35412×10⁻⁵) | 1×10⁻⁴ | 4.02×10⁻⁵ |
| off-diagonal product | 0 (orthorhombic) | 1×10⁻⁸ | ~0 |

Product analytic diagonals [Ha/Bohr³]: xx −2.16997707×10⁻⁵,
yy +2.00723305×10⁻⁵, zz +5.82684354×10⁻⁵.

The Δσ residual (2.27×10⁻¹⁰ on xx) is the SCF-noise floor of the
two-full-products isolation (both analytic runs re-converge to `tol = 1e−7`);
signal separation stays decisive — xx signal 1.71×10⁻⁸ = 75× the residual,
tolerance 2×10⁻⁹ keeps 8.8× headroom.

### Performance

Same observable (this cell's stress). CP2K uses NUMERICAL strain-FD (many
SCFs); ATV's product path is one SCF + fixed-\(P\) analytic stress. Settings
caveat: ATV single 200 Ry grid, OMP 8; CP2K NGRIDS 4, np 2 × OMP 2
(Arrhenius epyc).

| Run | Wall [s] |
| --- | -------: |
| CP2K NUMERICAL stress, D3 | 33.6 |
| CP2K NUMERICAL stress, plain | 38.9 |
| ATV analytic product (SCF + fixed-P stress, OMP 8) | **18.5** |
| ATV plain twin (difference pin only) | 14.9 |
| ATV all-axes strain-FD twin (dual gate only) | 14.3 |
| Regression total (≤ 60 s budget) | 47.7 |

**Ratio (ATV product / CP2K D3 twin) = 18.5 / 33.6 ≈ 0.55**, inside the 10×
fair-wall budget. The extra plain/FD twins exist only for the difference pin
and dual gate, not the product path.

## gpw-d3-atm — D3 + ATM dispersion energy, isolated water dimer {#gpw-d3-atm}

D3 zero damping with the ATM three-body C9 term on the isolated water dimer
in a 12 Å cubic box (`PERIODIC NONE` / MT host), gating the **dispersion
energy** against CP2K `TYPE DFTD3` with `CALCULATE_C9_TERM .TRUE.`.

```cpp
GpwDft gpw{{O, H}, 200.0, XcKind::PBE};  // e_disp is geometry-only; SCF only needs to run
gpw.n_occ = 8;
gpw.max_iter = 8;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.nonperiodic = true; // isolated / MT — matches twin PERIODIC NONE
gpw.d3 = true;          // zero damping
gpw.d3_atm = true;      // CP2K CALCULATE_C9_TERM

gpw.build(frame);
const auto& res = gpw.run(frame);
// res.e_disp [Ha] within 1e-9 of the offline CP2K pin; res.e_total includes e_disp
```

### Validation

CP2K pin line: `Dispersion energy: -0.00081414118193`.

| Quantity | Atomiverse / pin | CP2K (offline) | Tol | Notes |
| -------- | ---------------: | -------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) (D3+ATM) [Ha] | −0.00081414118193 | −0.00081414118193 | \(1\times 10^{-9}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) (two-body only) [Ha] | −0.00081399213205 | (unit-test pin) | — | Context; ATM Δ ≈ −1.49×10⁻⁷ Ha |

ATM is attractive on the dimer and tiny (~10⁻⁷ Ha); the pin still locks the
three-body form against CP2K. **Tolerance rationale:** matching geometry,
PBE-D3 zero parameters, CP2K CN, the 2× pair cutoff (75.59 Bohr) and the ATM
form track CP2K dispersion to sub-nano-Hartree. Unit tests also check
analytic ATM forces vs central FD and that default `d3_atm = false` keeps the
two-body pin.

### Performance

perf_exempt (see [Oracle](#oracle)).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | ≈ 10.8 (~11 s; 1 rank × 4 OMP, pin host) | — (not fair-twinned) | **perf_exempt** | — | CP2K wall is full QS; hard gate is E_disp only |
| E_disp product gate | n/a | short SCF + D3 ATM | **perf_exempt** | — | Geometry-only deliverable |

## gpw-d3-atm-forces — ATM (C9) ionic forces, isolated and periodic {#gpw-d3-atm-forces}

The ATM term contributes its **analytic gradient** to the GPW ionic forces:
the explicit-geometry legs of every ABC triple *and* the CN chain
\(\partial C_6/\partial \mathrm{CN}\) folded into `dEdCN`. Pinned in both
boundary conditions:

- **isolated** water dimer (12 Å, MT) — home-cell triples;
- **periodic 6 Å cell** — the lattice-image triple classes of CP2K's
  `doabc` (AB home + C off-home; B and C off-home with `fac0 = 1/3`), which
  only a compressed periodic cell makes measurable.

Per boundary condition two CP2K `ENERGY_FORCE` runs differ only in
`CALCULATE_C9_TERM`, so

\[
F_A(\mathrm{C9}) - F_A(\mathrm{two\text{-}body}) = -\nabla_A E_{\mathrm{ATM}} .
\]

The electronic force cancels to ~2e−15 Ha/Bohr (the symmetry-zero
y-components of the printed tables differ only in the 15th digit), leaving
the pure ATM gradient at 4.6e−7 (iso) / 6.9e−7 (per) Ha/Bohr — ~500× above
the `FORCES|` print resolution. The product wiring runs `GPWEvaluator` with
`d3_atm = true` (cutoff 300 Ry, `tol = 1e-7`) minus a `GpwDft::forces` twin
with `d3_atm` off.

### Validation

| Gate | Tol | Measured |
| ---- | --- | -------- |
| Iso / per \(E_{\mathrm{disp}}\)(ATM) vs CP2K | 1e−9 Ha | 6.4e−11 / 2.8e−10 |
| Iso ATM force vs CP2K diff | 1e−8 Ha/Bohr | **5.0e−10** (signal 4.6e−7) |
| Per ATM force vs CP2K diff | 1e−8 Ha/Bohr | **9.8e−10** (signal 6.9e−7) |
| Product wiring | 1e−7 Ha/Bohr | 2.2e−17 |
| Analytic vs FD iso / per (recorded, no unit gate) | 1e−7 Ha/Bohr | 4.1e−14 / 2.2e−13 |
| Wall | ≤ 60 s | 34.5 s |

The analytic-vs-FD values cover the isolated and the periodic case (periodic:
FD of the ATM-only energy difference, so the two-body gradient cancels
exactly); no unit test in the tree gates them.

### Image-triple bookkeeping

- **Half neighbor list.** The triple loop's AB pairs replicate CP2K's
  *half* list (`iatom ≤ jatom`, one orientation per off-home image), as the
  two-body loop does. Taking both orientations gives every lattice-image
  triple class 6 anchored representatives against `fac0 = 1/3` and inflates
  the periodic ATM energy 8× (+6.25e−6 vs +7.85e−7 Ha) on this twin.
- **pbc anchoring.** CP2K's `doabc` runs on `pbc()`-wrapped coordinates, and
  the hard `r < r_cut` leg gates over a finite image window make the per-atom
  force split depend on the wrap while the energy is invariant (4e−19). With
  raw coordinates the two molecules' ATM forces come out exactly *swapped*
  relative to CP2K (1.8e−7 err, each value matching the other molecule to
  ~1e−10); `accumulate_atm_c9` wraps first.

### Performance

CP2K TIMING `CP2K` TOTAL TIME (`mpirun -np 2`, `OMP_NUM_THREADS=2`); the ATV
wall is the full regression (iso + per SCF + forces).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| water dimer iso 12 Å + per 6 Å, PBE-D3(ATM) forces | 10.5 | 34.5 | 3.3 | ✓ | ATV single run covers both cells; CP2K wall is the dominant iso member |

## gpw-d3-atm-stress — ATM (C9) cell stress, periodic 6 Å water dimer {#gpw-d3-atm-stress}

Pins the **three-body ATM contribution to the fixed-\(P\) cell stress** on
Γ-point GPW PBE (D3 zero damping + C9) on the periodic **6 Å** cell of
[gpw-d3-atm-forces](#gpw-d3-atm-forces). The pin is the difference of two
CP2K `STRESS_TENSOR NUMERICAL` runs (`CALCULATE_C9_TERM` `.TRUE.` −
`.FALSE.`, otherwise identical): the electronic **and two-body dispersion**
stress cancel exactly, isolating
\(-\Omega^{-1}\partial E_{\mathrm{ATM}}/\partial\varepsilon\). The fixed-\(P\)
path rebuilds \(E_{\mathrm{disp}}^{\mathrm{D3+ATM}}\) at each strained cell,
which is exactly the fold this pin certifies. The twin uses `R_CUTOFF 10.0`,
matched through `d3_rcut_bohr` (default = CP2K default 20 Å → 75.59 Bohr).

**ATM strain-FD is step-locked.** The triple sum passes hard `r < r_cut` and
empirical-scaling gates, so its included boundary set jumps (~5e−11 Ha) at
discrete strains. The ATM-only FD therefore uses **exactly CP2K's
cell-perturbation step** (`DX 0.001` Bohr ⇒ ε = DX/L): at h = 1e−4 the xx/zz
parity degrades to 1.6e−10 Ha/Bohr³; at the CP2K step it is 1.1e−13.

```cpp
cpu::gpw::GpwDft dft{{O, H}, 150.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 200;
dft.tol = 1e-6;
dft.mix = 0.3;
dft.nonperiodic = false;
dft.d3 = true;
dft.d3_atm = true;
dft.d3_rcut_bohr = 2.0 * 10.0 * 1.8897259885789; // CP2K R_CUTOFF 10.0 twin
cpu::gpw::GPWEvaluator eval(std::move(dft));
eval.compute_stress = true;
eval.stress_use_fd = false;
// … cpu::Evaluate as in Method; eval.last_stress_ha_bohr3() [Ha/Bohr³]
```

Analytic vs FD stress on cheap **periodic** H₂ (image triples engage) has no
unit test in the tree.

### Validation

Two CP2K runs (`mpirun -np 2`, OMP 2, `EPS_SCF 1e-8`, CUTOFF 300, per-6 Å
cell): `water_dimer_d3_atm_per6_stress.inp` /
`water_dimer_d3_per6_stress.inp`.

| Quantity | CP2K pin | ATV | Tol |
| -------- | -------: | --: | --- |
| \(E_{\mathrm{disp}}\)(C9) [Ha] | −0.00195809486938 | \|d\| = 2.8e−10 | 1e−9 Ha |
| \(E_{\mathrm{disp}}\)(2b) [Ha] | −0.00195887984277 | \|d\| = 2.8e−10 | 1e−9 Ha |
| \(\Delta\sigma_{xx}\) [Ha/Bohr³] | −1.7148335571e−08 | \|d\| = 1.1e−13 | 5e−12 |
| \(\Delta\sigma_{yy}\) [Ha/Bohr³] | +4.3332828109e−08 | \|d\| = 1.1e−14 | 5e−12 |
| \(\Delta\sigma_{zz}\) [Ha/Bohr³] | +9.0917867267e−09 | \|d\| = 3.7e−14 | 5e−12 |
| dual gate \(|\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}|\) (xx in e2e) | — | 2.4e−7 | 2e−5 Ha/Bohr³ |

### Performance

| Twin | \(T\) [s] |
| ---- | --------: |
| CP2K numerical-stress C9 run (18 perturbed SCFs + ref, 2×2 cores) | 54.5 |
| ATV product analytic stress pass (1 SCF + 7 fixed-\(P\) evals, OMP 8) | 40.8 |
| ATV multi-SCF FD twin, x-only, warm-started (dual-gate only) | 7.8 |

Ratio \(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) ≈ **0.75** (≤ 10; loaded login
node — settings caveats in REFERENCE.md).

## gpw-d3bj-forces — D3(BJ) ionic forces, periodic water dimer {#gpw-d3bj-forces}

The BJ sibling of [gpw-d3-forces](#gpw-d3-forces), on the **periodic** 12 Å
cell, so the dispersion lattice sum and its gradient are both exercised
(CUTOFF 300 Ry, `tol = 1e-7` = CP2K EPS_SCF, PERIODIC XYZ; product
`GPWEvaluator` with `compute_forces = true` minus a plain `GpwDft::forces`
twin).

### Validation

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs CP2K | 1e−8 Ha | **8.9e−11** |
| \(F_{\mathrm{disp}}\) vs CP2K difference | 5e−6 Ha/Bohr | **1.21e−06** |
| Total \(F\) vs CP2K | 1e−2 Ha/Bohr | **4.30e−03** |
| \(\lvert\sum_A F_A\rvert\) | 3e−3 Ha/Bohr | **1.34e−03** |
| Wiring | 1e−7 Ha/Bohr | **5.9e−12** |
| Wall | ≤ 60 s | **32.2 s** (CP2K 11.9 s → ratio 2.7) |

Analytic vs FD forces agreed to **2.4e−6 Ha/Bohr** on cheap MT H₂ (recorded;
no unit test in the tree gates it).

**Reading the tolerances.** The total-force comparison is bounded **from
below by the oracle**: CP2K's own `FORCES| Sum` on this twin is 5.43e−3
Ha/Bohr, i.e. its force field is not translation invariant at `CUTOFF 300`.
Atomiverse's own net force is 1.34e−3 — 4× tighter — and both shrink with
cutoff. A 1e−3 component gate would pin CP2K's grid noise, so the tight,
physically meaningful gates are the dispersion pins (1e−8 Ha / 5e−6
Ha/Bohr) and the net-force check.

## gpw-d3bj-stress — D3(BJ) cell stress, periodic water dimer {#gpw-d3bj-stress}

Fixed-\(P\) orthorhombic cell stress on Γ-point GPW PBE with two-body D3(BJ)
(ATM off) against a real offline CP2K `STRESS_TENSOR NUMERICAL` twin — the
pin the D4 sibling cannot carry (Arrhenius CP2K lacks libdftd4, so
[gpw-d4-stress](gpw-d4.md#gpw-d4-stress) gates on the internal dual-gate
only). D3(BJ) is native in CP2K, which makes this section the **oracle stress
pin for the whole geometry-only dispersion stress fold** in `gpw_stress.cpp`.
Same construction as the [gpw-d3bj-forces](#gpw-d3bj-forces) force pin
(D3BJ − plain); pair cutoff 2 × `R_CUTOFF 20 Å` = 75.589 Bohr.

```cpp
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 200;
dft.tol = 1e-6;
dft.mix = 0.4;
dft.nonperiodic = false;
dft.d3bj = true;
cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/false);
eval.compute_stress = true;  // analytic fixed-P (product default)
eval.stress_use_fd = false;
```

Analytic vs FD stress on cheap MT H₂ has no unit test in the tree.

### Validation

Two CP2K runs (`mpirun -np 2`, OMP 2, `EPS_SCF 1e-8`, CUTOFF 300):
`water_dimer_d3bj_stress.inp` and `water_dimer_plain_stress.inp`, both
`STRESS_TENSOR NUMERICAL`.

| Quantity | CP2K pin | ATV | Tol |
| -------- | -------: | --: | --- |
| \(E_{\mathrm{disp}}\) [Ha] | −0.00137627769481 | \|d\| = 7.6e−12 | 1e−8 Ha |
| \(\Delta\sigma_{xx}\) [Ha/Bohr³] | −1.1485894562e−07 | \|d\| = 3.9e−14 | 1e−10 |
| \(\Delta\sigma_{yy}\) [Ha/Bohr³] | −8.9319862445e−10 | \|d\| = 2.7e−14 | 1e−10 |
| \(\Delta\sigma_{zz}\) [Ha/Bohr³] | +9.6563695465e−10 | \|d\| = 3.2e−14 | 1e−10 |
| dual gate \(|\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}|\) (xx in e2e) | — | 7.4e−7 (pinning all-axes: yy 2.5e−8, zz 1.2e−9) | 2e−5 Ha/Bohr³ |

### Performance

| Twin | \(T\) [s] |
| ---- | --------: |
| CP2K numerical-stress D3BJ run (18 perturbed SCFs + ref, 2×2 cores) | 83.8 |
| ATV product analytic stress pass (1 SCF + 7 fixed-\(P\) evals, OMP 8) | 28.4 |
| ATV multi-SCF FD twin, x-only, warm-started (dual-gate only) | 4.8 |

Ratio \(T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) ≈ **0.37** (≤ 10; stress
product vs stress oracle — settings caveats in REFERENCE.md).

## gpw-pbe-d3bj-ice — D3(BJ) dispersion energy, periodic water dimer {#gpw-pbe-d3bj-ice}

D3(BJ) on Γ-point GPW PBE for the periodic 12 Å water-dimer cell
(ice-relevant H-bonded pair), gating the **dispersion energy** against CP2K
`TYPE DFTD3(BJ)`; three-body ATM off (`CALCULATE_C9_TERM .FALSE.`). The same
cell carries the D4 and TS energy pins ([gpw-d4-water](gpw-d4.md#gpw-d4-water),
[gpw-ts-ice](gpw-tkatchenko-scheffler.md#gpw-ts-ice)).

```cpp
frame.box = cpu::Box::cubic(12.0);   // Å
GpwDft gpw{kinds, 200.0, XcKind::PBE};  // e_disp is geometry-only; SCF only needs to run
gpw.n_occ = 8;
gpw.max_iter = 8;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.d3bj = true;       // mutually exclusive with gpw.d3

gpw.build(frame);
const auto& res = gpw.run(frame);
// res.e_disp [Ha] within 1e-8 of the offline CP2K pin; res.e_total includes e_disp
```

### Validation

CP2K pin line: `Dispersion energy: -0.00137627769481`.

| Quantity | Atomiverse / pin | CP2K (offline) | Tol | Notes |
| -------- | ---------------: | -------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −0.00137627769481 | −0.00137627769481 | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{tot}}\) QS [Ha] | (not gated) | −34.33962825563683 | — | Context only; OT/grid/DIIS |

**Tolerance rationale.** Matching geometry, PBE-BJ parameters, CP2K CN form,
and the 2× pair cutoff (75.59 Bohr) track CP2K dispersion to nano-Hartree.
Unit tests also check analytic BJ forces vs central FD and
\(e_{\mathrm{total}}(\mathrm{d3bj}) - e_{\mathrm{total}}(\mathrm{off}) = e_{\mathrm{disp}}\).

### Performance

perf_exempt (see [Oracle](#oracle)).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | ≈ 13.4 (~13 s; 1 rank × 4 OMP, pin host) | — (not fair-twinned) | **perf_exempt** | — | CP2K wall is full QS; hard gate is E_disp only |
| E_disp product gate | n/a | short SCF + D3 | **perf_exempt** | — | Geometry-only deliverable |
