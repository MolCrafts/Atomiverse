# GPW DFT-D4 dispersion

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Grimme / Caldeweyher DFT-D4 London correction on the Γ-point GPW path, on top
of PBE: two-body Becke–Johnson (rational) damping with EEQ2019 charges and
periodic lattice sums, plus the optional Axilrod–Teller–Muto (ATM) three-body
term at \(s_9 = 1\) (dftd4 `pbe-D4-ATM`). D4 is geometry-only — it never
enters the Kohn–Sham potential — so it is reported on `GpwResult::e_disp`
[Hartree], folded into `e_total`, and its analytic gradient and strain
derivative are added to the ionic forces and cell stress.

The switches are bool members of `cpu::gpw::GpwDft` (no `*Config` bag): `d4`
and `d4_atm` (\(s_9 = 1\); ignored unless `d4` is set; default `false` keeps
the two-body product). `d4` is mutually exclusive with `d3` / `d3bj` / `ts` /
`mbd` / `xdm`; any two set throws at run. Energies come from
`gpw.build(frame)` + `gpw.run(frame)`; forces from `GpwDft::forces(frame)` or
`cpu::gpw::GPWEvaluator` with `compute_forces = true`; stress from
`GPWEvaluator` with `compute_stress = true`, `stress_use_fd = false` on a
`cpu::Evaluate` driver (`last_stress_ha_bohr3()` [Ha/Bohr³], Frame `kStress`
= σ·Ω [eV]) or `GpwDft::stress`, with `gpw_stress_fd` as the multi-SCF
strain-FD twin. The standalone kernel is `cpu::scf::dftd4` /
`cpu::scf::dftd4_energy` (`D4Result`).

The sections cover the CPU backend only, on one periodic water-dimer cell:
two-body energy ([gpw-d4-water](#gpw-d4-water)), forces
([gpw-d4-forces](#gpw-d4-forces)) and stress
([gpw-d4-stress](#gpw-d4-stress)); ATM energy ([gpw-d4-atm](#gpw-d4-atm)),
forces ([gpw-d4-atm-forces](#gpw-d4-atm-forces)) and stress
([gpw-d4-atm-stress](#gpw-d4-atm-stress)). Multi-k D4 forces have no section.
Build in the CPU environment (`module load buildenv-gcc/2026.03-mpich`, see
`docs/developer/project-contract.md`) with `-DATV_BUILD_GPW=ON -DATV_BUILD_REGRESSIONS=ON`.

Case: `ctest -R '^gpw/d4$'` — sources and oracle record in [`regressions/gpw/d4/`](../../../regressions/gpw/d4/). One section of the case per `##` heading below: `atv_regressions --case gpw/d4 --section <name>`.

## Method

### Two-body D4 with BJ damping

Caldeweyher et al., *J. Chem. Phys.* **150**, 154122 (2019), with
**periodic** lattice sums over image vectors \(\mathbf{T}\):

\[
E_{\mathrm{disp}}^{(2)} = -\sum_{A\le B}\sum_{\mathbf{T}}{}^{\prime}
\left[
  \frac{s_6\,C_6^{AB}}{R_{AB,T}^6 + f^6}
  + \frac{s_8\,C_8^{AB}}{R_{AB,T}^8 + f^8}
\right],
\quad
f = a_1\sqrt{3\,r_4^{A} r_4^{B}} + a_2,
\]

where the prime excludes the singular \(R=0\) self term and
\(C_8^{AB} = 3\,C_6^{AB}\,r_4^{A} r_4^{B}\).

**Charge / CN model (periodic).** Partial charges from **EEQ2019**;
coordination numbers are EN-weighted erf CNs; \(C_6\) from reference
polarizability interpolation (in-tree tables). Periodic EEQ uses an
**Ewald-split A-matrix** (multicharge-style). Default cutoffs match dftd4:
CN **30 Bohr**, two-body disp **60 Bohr**, three-body **40 Bohr**.

### Three-body ATM

\[
E_{\mathrm{disp}} = E_{\mathrm{disp}}^{(2)} + s_9\, E_{\mathrm{ATM}},
\]

with \(s_9 = 1\) for `pbe-D4-ATM` (`d4_atm`) and \(s_9 = 0\) for the two-body
product (dftd4 flag `--mbdscale 0.0`). ATM C9 triples run over atoms
\(A,B,C\) and lattice images (dftd4 / simple-dftd4 `disp.f90` form; cutoff
**disp3 = 40 Bohr**; damping exponent **alp = 16**). ATM reweights C6 with
**neutral-charge** partial charges (\(q = 0\)), matching simple-dftd4; CN and
two-body \(C_6\) still use EEQ2019. `src/cpu/gpw/gpw_solver.cpp` passes
`s9 = cfg.d4_atm ? 1.0 : 0.0` into `dftd4_energy`.

**PBE-D4 parameters** (SAW190103 BJ set; dftd4 `pbe-D4` / `pbe-D4-ATM`):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(s_6\) | 1.0 | dimensionless |
| \(s_8\) | 0.95948085 | dimensionless |
| \(s_9\) | 0 (two-body) / **1.0** (ATM on) | dimensionless |
| \(a_1\) | 0.38574991 | dimensionless |
| \(a_2\) | 4.80688534 | Bohr |
| alp (ATM) | 16 | dimensionless |

### Forces

Product ionic force on atom \(A\):

\[
\mathbf{F}_A = \mathbf{F}_A^{\mathrm{elec}} + \mathbf{F}_A^{\mathrm{D4}},
\qquad
\mathbf{F}_A^{\mathrm{D4}} = -\frac{\partial E_{\mathrm{disp}}}{\partial\mathbf{R}_A}
\quad[\mathrm{Ha/Bohr}],
\]

where \(\mathbf{F}_A^{\mathrm{elec}}\) is the GPW analytic electronic force
(Hellmann–Feynman + Pulay + grid/core terms, `gpw_forces_*` — see
[gpw-forces-water](gpw-rks.md#gpw-forces-water)) and `gpw_forces.cpp` adds
`d4.forces`. `cpu::scf::dftd4` returns the **analytic** gradient: for the
two-body term the BJ pair + CN\(_\mathrm{d4}\) weight chain + EEQ
charge-response adjoint; for ATM closed-form damped Axilrod–Teller triple
derivatives plus the C9 → neutral-C6 → CN\(_\mathrm{d4}\) chain (the
neutral-charge reweight is geometry-independent, so the ATM leg has **no EEQ
adjoint**). Offline dftd4 prints the **gradient** \(g = +\partial E/\partial
R\); the comparison is \(F_{\mathrm{ATV}} \stackrel{?}{=} -g_{\mathrm{dftd4}}\).
Because D4 does not change the SCF density, product wiring is

\[
\mathbf{F}^{\mathrm{d4\,on}} - \mathbf{F}^{\mathrm{plain}}
\;\approx\;
\mathbf{F}^{\mathrm{D4\,standalone}}
\]

within SCF residual (tol \(1\times 10^{-7}\) Ha/Bohr). Multi-SCF FD of full
product forces on the 6-atom cell exceeds the ≤60 s regression budget; the
analytic-vs-FD dual gate runs on cheap MT H₂ in the unit suite.

### Analytic fixed-\(P\) cell stress

One SCF at the unstrained cell → density matrix \(P\) and energy-weighted
\(W = \sum_i 2\varepsilon_i c_i c_i^\top\). Then central FD of the
non-self-consistent Lagrangian under orthorhombic diagonal strain at
**fixed fractional coordinates** (`scale_axis` convention: cell row \(a\)
**and** Cartesian \(a\)-components scaled by \(1 \pm h\)):

\[
E_L[P,W;\varepsilon]
  = \mathrm{Tr}(P\,T) + \mathrm{Tr}(P\,V_{\mathrm{pp}})
  + E_{\mathrm{es}}[\rho_P] + E_{\mathrm{xc}}[\rho_P]
  + E_{\mathrm{disp}}^{\mathrm{D4}}(\{\mathbf{R}\},\mathbf{h})
  - \mathrm{Tr}(W\,S),
\qquad
\sigma_{aa} = -\Omega^{-1}\frac{\partial E_L}{\partial\varepsilon_{aa}},
\quad a\in\{x,y,z\}.
\]

The \(-\mathrm{Tr}(W\,S)\) term is the Pulay contribution; D4 (with
\(s_9 = 1\) when `d4_atm`) is rebuilt from the strained positions / cell.
\(E\) in Hartree, \(\Omega\) in Bohr³; sign matches CP2K `STRESS_TENSOR`;
Voigt order \((\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})\);
off-diagonals ~0 on the orthorhombic cell; default \(h = 2\times 10^{-3}\)
(`stress_strain_h`). `gpw_stress_fd` fully re-SCFs at \(\pm h\) per axis (D4
folded via the same `fold_dispersion` path as the energy) — dual-gate only,
never the product default (`stress_use_fd=false`).

### Electronic host and molecular D4

Γ-RKS GPW (VandeVondele et al., *Comput. Phys. Commun.* **167**, 103
(2005)), PBE XC, GTH PBE PPs. Energy-only sections run a short SCF just to
materialize `e_disp`; absolute QS \(E_{\mathrm{tot}}\) micro-Ha parity is out
of scope (OT/grid/DIIS differences; no DFTD4-linked CP2K on the pin host).
Non-periodic `cpu::scf::D4` ([dft-d4-h2o-pbe](scf-dispersion.md#dft-d4-h2o-pbe),
default `s9 = 0`) remains the isolated-molecule / PySCF path; GPW `d4` adds
periodic lattice CN/EEQ/dispersion on the same in-tree parameter tables.

## Common setup

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair; not bulk ice Ih), **6 atoms**, order **O H H O H H** — the cell of [gpw-pbe-d3bj-ice](gpw-d3.md#gpw-pbe-d3bj-ice) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å** (12 × 12 × 12 Å), PERIODIC XYZ (Ω = 11661.119488 Bohr³) |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) (E_disp / F_D4 are basis-independent) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Grid | CUTOFF **200** Ry |

**Atom order.** POSCAR / dftd4 JSON is **O O H H H H**; ATV regressions and
force tables use **O H H O H H**. Gradients below are already remapped.

Mounted on a driver (same members):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GpwDft dft{kinds, 300.0, ATV::cpu::gpw::XcKind::PBE};
dft.max_iter = 200;  dft.tol = 1e-7;  dft.mix = 0.3;
dft.nonperiodic = false;
dft.d4 = true;
dft.d4_atm = true;   // optional, s9 = 1
ATV::cpu::gpw::GPWEvaluator eval(std::move(dft));
// bind on Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
```

| Path | Role |
| ---- | ---- |
| `gpw_solver.h` | `GpwDft::d4`, `d4_atm`, `GpwResult::e_disp`, `GpwDft::stress` / `gpw_stress_fd` |
| `gpw_evaluator.h` | `GPWEvaluator` (`compute_forces`, `compute_stress`, `stress_use_fd`, `last_stress_ha_bohr3`) |
| `dftd4.h` / tables | `dftd4` / `dftd4_energy(..., s9)` / `D4Result`, molecular `D4`, shared parameter tables |
| `src/cpu/gpw/internal/gpw_forces.cpp` | Adds `d4.forces` into product forces |
| `src/cpu/gpw/internal/gpw_stress.cpp` | Fixed-\(P\)+Pulay + D4 rebuild; multi-SCF FD |

No unit test in the tree pins D4 \(E_{\mathrm{disp}}\) or forces (periodic +
isolated, two-body + ATM), nor compares H₂ analytic forces or stress with `d4`
against finite differences.

## Oracle

CP2K 2026.1 on Arrhenius is built **without** the DFTD4 library (`TYPE DFTD4`
aborts: `CP2K build without DFTD4`). The hard pins are offline **dftd4
3.7.0** (Arrhenius EasyBuild `dftd4/3.7.0-gfbf-2025b`) — libdftd4, the same
backend CP2K calls when DFTD4 is linked; the CP2K twin inputs (POSCAR + QS)
are retained in the case record for hosts that link libdftd4. External
engine pins — not goldens of Atomiverse's own output; dftd4 / CP2K are
**never** launched from ctest.

| Quantity | dftd4 3.7.0 [Ha] | Command |
| -------- | ---------------: | ------- |
| \(E_{\mathrm{disp}}\) two-body, periodic | −0.00099458842745846774 (`Dispersion energy: -9.9458842745847E-04 Eh`) | `dftd4 -f pbe --mbdscale 0.0 --json pin.json POSCAR` |
| \(E_{\mathrm{disp}}\) two-body, isolated nuclei (context / unit) | −9.8816586433966×10⁻⁴ | isolated twin |
| \(E_{\mathrm{disp}}\) D4+ATM, periodic | −0.00099554475834240288 (`Dispersion energy: -9.9554475834240E-04 Eh`) | `dftd4 -f pbe --json pin_periodic_atm.json POSCAR` |
| \(E_{\mathrm{disp}}\) D4+ATM, isolated nuclei (context / unit) | −9.8819741533958208×10⁻⁴ | isolated twin |

Offline re-pin (human only; never ctest):

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/bin:$PATH
export LD_LIBRARY_PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/dftd4/3.7.0-gfbf-2025b/lib:$LD_LIBRARY_PATH
# + FlexiBLAS / GCCcore from buildenv-gcc module; cd to the case's oracle input dir
dftd4 -f pbe --mbdscale 0.0 --json pin.json POSCAR                       # two-body
dftd4 -f pbe --mbdscale 0.0 --grad pin_grad.txt --json pin.json POSCAR   # + gradient
dftd4 -f pbe --json pin_periodic_atm.json POSCAR                         # s9 = 1
dftd4 -f pbe --mbdscale 0.0 --json pin_periodic_s9_0.json POSCAR         # two-body context
# remap O O H H H H → O H H O H H before comparing to ATV
```

**Performance policy.** Every section is **perf_exempt**: D4 evaluation is
\(O(N^2 \times N_{\mathrm{images}})\) (two-body, energy and force alike — not
\(6N\) energy evaluations) and \(O(N^3 \times N_{\mathrm{images}}^2)\) (ATM)
on a 6-atom cell (micro- to milliseconds); the wall is the electronic SCF +
forces / stress. A fair full QS twin ratio vs CP2K is unavailable on the pin
host (no DFTD4-linked CP2K), and a dispersion-only microbenchmark vs the
dftd4 CLI is not the product gate (same policy as
[gpw-pbe-d3bj-ice](gpw-d3.md#gpw-pbe-d3bj-ice) and
[gpw-dftu-stress](gpw-dft-plus-u.md#gpw-dftu-stress)). A strict full-SCF twin
applies only on a host with libdftd4-linked CP2K, if product policy requires
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\).

## Limitations

- **Water-dimer cell**, not bulk ice Ih — periodic lattice-sum / EEQ / stress
  consistency pins, not a structural ice benchmark (no ice Ih supercell, no
  phonons).
- **No absolute CP2K pins** on the pin host: GPW \(e_{\mathrm{tot}}\), force
  components and `STRESS_TENSOR` are not compared against CP2K QS.
- **PBE SAW190103 parameters only**; other reference functionals share tables
  but are not e2e-gated.
- **Orthorhombic diagonal stress only**; shear / triclinic out of scope.
- **Γ-RKS pins**; UKS / multi-k D4 stress not gated here; no CUDA GPW.
- **Oracle split:** the GPW pins are dftd4 3.7.0; molecular
  [dft-d4-h2o-pbe](scf-dispersion.md#dft-d4-h2o-pbe) is PySCF +
  pyscf-dispersion — do not mix geometries / BC when comparing. D3 C9 / ATM is
  `d3_atm` ([gpw-d3-atm](gpw-d3.md#gpw-d3-atm)).
- Frame forces are eV/Å; pin comparison converts to Ha/Bohr
  (`units::kEvPerAngstromInHartreePerBohr`).

## References

- E. Caldeweyher, C. Bannwarth, S. Grimme, *J. Chem. Phys.* **150**, 154122
  (2019) (DFT-D4) and follow-ons (periodic D4 / EEQ2019; ATM via \(s_9\))
- B. M. Axilrod, E. Teller, *J. Chem. Phys.* **11**, 299 (1943); Y. Muto,
  *Proc. Phys. Math. Soc. Jpn.* **17**, 629 (1943) (three-body dispersion)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005)
  (Quickstep GPW)
- Nielsen & Martin, *Phys. Rev. B* **32**, 3780 (1985) (stress context)
- dftd4 3.7.0 (CLI pin backend; the library CP2K links as `TYPE DFTD4`)
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`
- Related: [gpw-pbe-d3bj-ice](gpw-d3.md#gpw-pbe-d3bj-ice),
  [gpw-d3-atm-forces](gpw-d3.md#gpw-d3-atm-forces),
  [gpw-d3-atm-stress](gpw-d3.md#gpw-d3-atm-stress),
  [gpw-forces-water](gpw-rks.md#gpw-forces-water),
  [gpw-analytic-stress](gpw-rks.md#gpw-analytic-stress),
  [gpw-stress-water](gpw-rks.md#gpw-stress-water)

## gpw-d4-water — two-body D4 dispersion energy {#gpw-d4-water}

Gates the periodic two-body (\(s_9 = 0\)) **dispersion energy** of the
water-dimer cell against dftd4 3.7.0 (`-f pbe --mbdscale 0.0`).

```cpp
frame.box = ATV::cpu::Box::cubic(12.0); // Å
GpwDft gpw{{O, H}, 200.0, XcKind::PBE};  // e_disp is geometry-only; SCF only needs to run
gpw.n_occ = 8;
gpw.max_iter = 8;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.d4 = true;         // exclusive vs gpw.d3 / gpw.d3bj

gpw.build(frame);
const auto& res = gpw.run(frame);
// res.e_disp [Ha] within 1e-8 of the offline dftd4 3.7.0 pin; res.e_total includes e_disp
```

### Validation

| Quantity | Atomiverse / pin | dftd4 3.7.0 (offline) | Tol | Notes |
| -------- | ---------------: | --------------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −0.00099458842745846774 | −0.00099458842745846774 | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) isolated [Ha] | (unit test) | −9.8816586433966×10⁻⁴ | \(1\times 10^{-8}\) Ha | Context / unit test |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | (CP2K DFTD4 N/A on pin host) | — | Not absolute-parity gated |

**Tolerance rationale.** Matching geometry, PBE-D4 parameters, EEQ2019, and
periodic lattice sums track dftd4 to nano-Hartree on this cell. Unit tests
also check the isolated two-body pin on the same nuclei.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | n/a (CP2K lacks DFTD4 on Arrhenius) | — (not fair-twinned) | **perf_exempt** | — | Twin input retained for DFTD4-linked hosts |
| E_disp product gate | n/a | short SCF + D4 | **perf_exempt** | — | Geometry-only deliverable |

## gpw-d4-forces — two-body D4 ionic forces {#gpw-d4-forces}

Γ-point GPW PBE **analytic ionic forces** with additive two-body D4 (ATM off)
— the exact gradient of the energy pinned in [gpw-d4-water](#gpw-d4-water),
gated against the dftd4 3.7.0 printed gradient (`--grad`) and by product
wiring.

```cpp
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 120;
dft.tol = 1e-6;
dft.mix = 0.4;
dft.nonperiodic = false;
dft.d4 = true;
cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/true);
eval.compute_stress = false;
// … cpu::Evaluate driver; frame.span(kAtom_f*) [eV/Å] = elec + D4
// Standalone D4 forces [Ha/Bohr]:
//   auto r = cpu::scf::dftd4(Z, pos_bohr, cell_bohr, /*periodic=*/true);
//   r.forces[A] = −∂E_disp/∂R_A
```

### Validation

| Quantity | Offline pin | Tol (ATV gate) | Notes |
| -------- | ----------: | -------------- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | **−0.00099458842745846774** | \(1\times 10^{-8}\) Ha | Same as [gpw-d4-water](#gpw-d4-water) |
| max\|F_d4 + g_dftd4\| [Ha/Bohr] | (identity) | \(1\times 10^{-7}\) | Force = −gradient |
| max\|(F_d4_on − F_plain) − F_standalone\| [Ha/Bohr] | (wiring) | \(1\times 10^{-7}\) | SCF residual |
| Unit dual-gate max\|F_an − F_fd\| [Ha/Bohr] | H₂ MT | \(5\times 10^{-5}\) | Unit suite only |

**Remapped offline gradient** \(g = \partial E/\partial R\) [Ha/Bohr]
(force pin \(F = -g\)):

| Atom | \(g_x\) | \(g_y\) | \(g_z\) |
| ---- | ------- | ------- | ------- |
| O0 | −5.3738961271913915e−05 | ~0 | ~0 |
| H0 | −5.9550148958160894e−05 | ~0 | −1.0810592129949165e−05 |
| H1 | −5.9550148958160921e−05 | ~0 | +1.0810592129949182e−05 |
| O1 | +1.3609823922940861e−04 | ~0 | ~0 |
| H2 | +1.8370509979413634e−05 | ~0 | −8.5497124425277334e−06 |
| H3 | +1.8370509979413539e−05 | ~0 | +8.5497124425278096e−06 |

Gradient norm ≈ **1.719×10⁻⁴** Ha/Bohr.

**Measured ATV:**

| Quantity | Value |
| -------- | ----: |
| \(E\) (Γ PBE-D4 200 Ry) | **−34.3313259463** Ha (SCF 11 iters) |
| \(E_{\mathrm{disp}}\) | **−9.9458849003×10⁻⁴** Ha (\|Δ pin\| = 6.3×10⁻¹¹) |
| max\|F_d4 + g_dftd4\| | **3.0×10⁻¹¹** Ha/Bohr |
| max\|(F_d4 − F_plain) − F_standalone\| | **2.9×10⁻⁹** Ha/Bohr |
| max\|F_product\| | **0.136** Ha/Bohr |
| \(T_{\mathrm{product}}\) (OMP=8, Release) | **17.6** s |
| \(T_{\mathrm{plain}}\) twin (OMP=8) | **17.1** s |

**Tolerance rationale.** The standalone D4 path tracks dftd4 to nano-Hartree
/ sub-nHa/Bohr. Wiring tol \(10^{-7}\) Ha/Bohr absorbs SCF residual when
subtracting two full product force evaluations. The product electronic force
magnitude (~0.1 Ha/Bohr) is set by the GPW host, not by D4 (~10⁻⁴ Ha/Bohr).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS ENERGY_FORCE | n/a (CP2K lacks DFTD4 on Arrhenius) | ~17.6 (product, OMP=8) | **perf_exempt** | — | Twin input retained for DFTD4-linked hosts |
| D4 gradient pin | n/a (CLI dftd4 offline) | micro–ms (geometry-only) | **perf_exempt** | — | Primary hard gate is accuracy |

## gpw-d4-stress — two-body D4 cell stress {#gpw-d4-stress}

Fixed-\(P\) orthorhombic cell stress on Γ-point GPW PBE with two-body D4.
The hard \(E_{\mathrm{disp}}\) pin is dftd4 3.7.0; the decisive stress gate is
the **internal dual-gate** — product analytic vs multi-SCF `gpw_stress_fd` on
the same `d4=true` method — not an absolute CP2K `STRESS_TENSOR` pin (the D3
sibling [gpw-d3bj-stress](gpw-d3.md#gpw-d3bj-stress) carries the CP2K stress
pin for the shared dispersion fold).

```cpp
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 120;
dft.tol = 1e-6;
dft.mix = 0.4;
dft.nonperiodic = false;
dft.d4 = true;
cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/false);
eval.compute_stress = true;  // product analytic fixed-P (default path)
eval.stress_use_fd = false;  // multi-SCF FD is oracle only
eval.method().stress_strain_h = 2e-3;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);
const double e_disp = eval.last_result().e_disp;   // [Ha]
const auto& san = eval.last_stress_ha_bohr3();     // [Ha/Bohr³] Voigt
// frame.meta.get(kStress)  // [eV] (σ·Ω)
// Dual gate: cpu::gpw::gpw_stress_fd(frame, method, /*h=*/2e-3, …)  // [eV/Å³]
```

| Section | Deliverable |
| ------- | ----------- |
| [gpw-d4-water](#gpw-d4-water) | Periodic D4 **energy** pin only |
| [gpw-analytic-stress](gpw-rks.md#gpw-analytic-stress) | Analytic fixed-\(P\) **without** D4 |
| [gpw-stress-water](gpw-rks.md#gpw-stress-water) | Strain-FD / CP2K NUMERICAL H₂O MT (no D4) |
| **gpw-d4-stress** | Analytic fixed-\(P\) **+ D4** dual-gate + E_disp pin |

### Validation

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | -----: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −0.00099458842745846774 | dftd4 3.7.0 same | \(1\times 10^{-8}\) Ha | Hard gate (shared with [gpw-d4-water](#gpw-d4-water)) |
| max\|\(\sigma^{\mathrm{an}}-\sigma^{\mathrm{fd}}\)\| diag | **7.39×10⁻⁷** | multi-SCF FD twin | \(2\times 10^{-5}\) Ha/Bohr³ | Decisive dual-gate |
| off-diag product \(\sigma\) | ~0 | — | \(1\times 10^{-8}\) Ha/Bohr³ | Orthorhombic Voigt |
| max\|\(\sigma_{\mathrm{diag}}\)\| floor | finite | — | \(> 1\times 10^{-10}\) | Non-trivial stress |
| Frame `kStress` | eV/Å³ from Ha/Bohr³ | unit conversion | \(1\times 10^{-9}\) rel | Product path check |
| \(E_{\mathrm{tot}}\) [Ha] | −34.3313259463 (context) | — | not absolute-parity gated | 11 SCF iters |

**Measured product diagonals:**

| Component | Analytic [Ha/Bohr³] |
| --------- | ------------------: |
| \(\sigma_{xx}\) | −2.17585158×10⁻⁵ |
| \(\sigma_{yy}\) | +2.00723219×10⁻⁵ |
| \(\sigma_{zz}\) | +5.82669568×10⁻⁵ |

**Tolerance rationale.** The analytic vs multi-SCF FD residual
\(\sim 10^{-6}\)–\(10^{-7}\) Ha/Bohr³ reflects geometry-only D4 + fixed-\(P\)
consistency; the gate is **2×10⁻⁵** Ha/Bohr³ (same order as other GPW stress
dual-gates).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Product analytic (OMP=8, Release) | n/a (no CP2K DFTD4) | **21.4** | **perf_exempt** | — | One SCF + fixed-\(P\) FD |
| Multi-SCF FD twin (OMP=8) | — (internal) | **42.3** | — | — | Dual-gate only; 6 strained SCFs |
| Dual-gate wall (sum) | — | **~64** | — | — | Not a fair external ratio |
| Full QS stress vs CP2K | n/a | — | **perf_exempt** | — | Arrhenius CP2K lacks DFTD4 |

## gpw-d4-atm — D4 + ATM dispersion energy {#gpw-d4-atm}

Gates the periodic D4+ATM (\(s_9 = 1\)) **dispersion energy** against dftd4
3.7.0 (`-f pbe`, default ATM scale). On this cell the ATM shift is small
(\(\approx -9.56\times 10^{-7}\) Ha) but finite: the regression also asserts
\(\lvert E - E_{\mathrm{2body}}\rvert > 10^{-10}\) Ha against the two-body
pin −9.9458842745846774×10⁻⁴ Ha.

```cpp
frame.box = ATV::cpu::Box::cubic(12.0);  // Å
GpwDft gpw{kinds, 200.0, XcKind::PBE};  // e_disp is geometry-only; SCF only needs to run
gpw.n_occ = 8;
gpw.max_iter = 8;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.d4 = true;
gpw.d4_atm = true;     // s9 = 1 (pbe-D4-ATM)

gpw.build(frame);
const auto& res = gpw.run(frame);
// res.e_disp [Ha] within 1e-8 of the dftd4 3.7.0 ATM pin, and different from the two-body pin
```

### Validation

| Quantity | Atomiverse / pin | dftd4 3.7.0 (offline) | Tol | Notes |
| -------- | ---------------: | --------------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) D4+ATM [Ha] | −0.00099554475834240288 | −0.00099554475834240288 | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) two-body only [Ha] | (must differ) | −9.9458842745846774×10⁻⁴ | \(\Delta > 10^{-10}\) Ha | Context / anti-pin |
| \(E_{\mathrm{disp}}\) isolated ATM [Ha] | (unit test) | −9.8819741533958208×10⁻⁴ | \(1\times 10^{-8}\) Ha | Context / unit test |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | (CP2K DFTD4 N/A on pin host) | — | Not absolute-parity gated |

**Tolerance rationale.** Matching geometry, PBE-D4 parameters, EEQ2019,
periodic lattice sums, and ATM (`s9=1`, neutral-charge C6 reweight, disp3 =
40 Bohr) track dftd4 to nano-Hartree on this cell. Unit tests also pin
isolated ATM and free-function periodic ATM on the same nuclei.

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS (context) | n/a (CP2K lacks DFTD4 on Arrhenius) | — (not fair-twinned) | **perf_exempt** | — | Twin input family retained for DFTD4-linked hosts |
| E_disp product gate | n/a | short SCF + D4+ATM | **perf_exempt** | — | Geometry-only deliverable |

## gpw-d4-atm-forces — ATM (s9 = 1) ionic forces {#gpw-d4-atm-forces}

The ATM term contributes its closed-form, FD-free **analytic gradient**
(see [Forces](#forces)): product ionic force = electronic Hellmann–Feynman +
Pulay \(+\; F_{\mathrm{disp}} = -\nabla\!\left(E_2 + E_{\mathrm{ATM}}\right)\)
with \(s_9 = 1\), gated end-to-end on the periodic water dimer against dftd4
3.7.0 `--grad`.

```cpp
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 120;
dft.tol = 1e-6;
dft.mix = 0.4;
dft.nonperiodic = false;
dft.d4 = true;
dft.d4_atm = true;       // s9 = 1
cpu::gpw::GPWEvaluator eval(std::move(dft));
eval.compute_forces = true;
eval.compute_stress = false;
cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);       // forces on kAtom_f{x,y,z}; e_disp on last_result()
```

Pin commands: `dftd4 -f pbe --grad --json pin_periodic_atm_grad.json POSCAR`
(+ isolated xyz twin; + two-body `--mbdscale 0` twin; + O0 x ± 0.001 Å
displaced-energy pins). Hard energy pin \(E_{\mathrm{disp}}\)(D4+ATM,
periodic) = **−9.9554475834240310×10⁻⁴ Ha**; the ATM-only force increment vs
the two-body `--mbdscale 0` pin has max component **4.05×10⁻⁷ Ha/Bohr** — the
physics this section gates. Isolated twin
\(E = −9.8819741533958229\times 10^{-4}\) Ha, gradient pinned at 1×10⁻⁹
Ha/Bohr in unit tests.

### dftd4 3.7.0's periodic ATM gradient is non-conservative

All numbers printed by dftd4 3.7.0 (offline):

- **Isolated**: central difference of its own \(s_9 = 1\) energy
  (h = 10⁻³ Å, O0 x) matches its printed gradient to **2.6×10⁻¹⁰** Ha/Bohr —
  conservative.
- **Periodic (12 Å cell)**: the same FD gives −5.3616541919×10⁻⁵ while the
  printed \(g_x(\mathrm{O0})\) is −5.3595427767×10⁻⁵ — a **~2.4×10⁻⁸
  Ha/Bohr** discrepancy against its *own* energy.
- ATV's \(s_9 = 1\) periodic energy equals dftd4's at the base **and**
  displaced geometries to a constant 3.8×10⁻¹² Ha (Bohr-conversion constant,
  cancels in differences): the two codes share **one energy functional**, and
  ATV's analytic force is conservative against it to 3×10⁻¹² (self-FD).

A conservative gradient of the shared functional therefore cannot match the
printed periodic pin beyond that oracle defect. The **direct** gradient-pin
comparison is set at **5×10⁻⁸ Ha/Bohr** (absorbs the ~2.4×10⁻⁸ defect, still
~8× below the 4.05×10⁻⁷ ATM increment, so it fails if the ATM gradient is
dropped), plus a **sharp 1×10⁻⁹ conservative-parity** gate against the FD of
the oracle's *printed displaced energies* (O0 x ± 0.001 Å; truncation ≈
3×10⁻¹⁰).

### Validation

The regression prints each measured value on PASS (`max|F+g|`,
`max|dF_atm-pin|`, `max|wiring|`, `wall_prod`).

| Gate | Tol | Context / signal |
| ---- | --- | ---------------- |
| \(E_{\mathrm{disp}}\)(ATM) standalone + product vs pin | 1e−8 Ha | pin −9.9554475834240310e−4 Ha; ATV self-FD conservative to 3e−12 |
| Direct: max\|F_d4atm + g_pin\| (periodic) | 5e−8 Ha/Bohr | oracle defect ~2.4e−8; ATM increment 4.05e−7 |
| Conservative parity: \|F_x(O0) − FD(oracle E±)\| | 1e−9 Ha/Bohr | FD truncation ≈ 3e−10; recorded 1.7e−12 (isolated grad pin; no unit gate) |
| ATM increment: max\|(F_atm − F_2b) − Δpin\| | 5e−8 Ha/Bohr | non-vacuity floor max\|dF\| > 1e−8 |
| Product wiring: max\|(F_on − F_plain) − F_standalone\| | 1e−7 Ha/Bohr | SCF residual budget |
| Wall (product SCF + forces) | ≤ 60 s | regression budget |

Analytic vs FD of the shared \(s_9 = 1\) functional (isolated and periodic ATM
forces vs the dftd4 gradient, the periodic ATM force increment vs the dftd4
delta, ATM forces vs an energy central difference) has no unit test in the
tree.

The direct periodic gradient parity is bounded below by the oracle's own
~2.4×10⁻⁸ non-conservativeness; the sharp 1×10⁻⁹ gate is against its
printed displaced energies, not its printed gradient.

## gpw-d4-atm-stress — ATM (s9 = 1) cell stress {#gpw-d4-atm-stress}

The ATM contribution to the analytic fixed-\(P\) cell stress (`GpwDft::stress`
path, with \(E_{\mathrm{disp}}^{\mathrm{D4+ATM}}\) rebuilt at each strained
cell; see [Method](#method)). The ATM stress on this cell is **~10⁻¹⁰
Ha/Bohr³** — three orders below the analytic-vs-FD dual-gate residual
(~7×10⁻⁷) — so the decisive gates are **dedicated dftd4 energy pins at the
exact strain samples**, never the oracle's (non-conservative, see
[gpw-d4-atm-forces](#gpw-d4-atm-forces)) gradient.

```cpp
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 120;
dft.tol = 1e-6;
dft.mix = 0.4;
dft.nonperiodic = false;
dft.d4 = true;
dft.d4_atm = true;          // s9 = 1 in energy AND strained stress rebuild
cpu::gpw::GPWEvaluator eval(std::move(dft));
eval.compute_stress = true; // analytic fixed-P path
eval.method().stress_strain_h = 2e-3;
// … cpu::Evaluate; frame.meta.get(kStress) [eV] (σ·Ω); eval.last_stress_ha_bohr3() [Ha/Bohr³]
```

Pin commands: per axis a, sign ±, scale cell row a **and** Cartesian
a-components by \(1 \pm h\) (\(h = 2\times10^{-3}\)), then
`dftd4 -f pbe --json … POSCAR` and the `--mbdscale 0.0` twin. Base-cell s9=1
pin: \(E_{\mathrm{disp}} =\) **−9.9554475834240290×10⁻⁴ Ha**
(`pin_periodic_atm.json`, shared with [gpw-d4-atm](#gpw-d4-atm)). Derived
oracle dispersion stress σ_aa = −[E(+h) − E(−h)]/(2hΩ); ATM-only part
σ^ATM = σ(s9=1) − σ(2b), max |σ^ATM| = **7.47×10⁻¹⁰ Ha/Bohr³** (xx).

### Validation

| Gate | Tol | Measured |
| ---- | --- | -------- |
| strained \|E_disp − pin\| (s9=1 and 2b, 12 values) | 1e−9 Ha | 6.30e−11 (Bohr-conversion constant) |
| \|σ^disp(s9=1) − FD(oracle E)\| per axis | 1e−12 Ha/Bohr³ | 5.19e−15 |
| \|σ^ATM − Δpin\| per axis | 1e−12 Ha/Bohr³ | 3.65e−16 |
| ATM non-vacuity max\|σ^ATM_pin\| | > 5e−10 Ha/Bohr³ | pin 7.47e−10 |
| product \|e_disp − ATM base pin\| | 1e−8 Ha | 2.76e−11 |
| dual gate \|σ_xx^an − σ_xx^fd\| (multi-SCF strain-FD twin) | 2e−5 Ha/Bohr³ | 7.392e−7 (same class as [gpw-d4-stress](#gpw-d4-stress)) |
| off-diagonal product; max\|σ_diag\| floor | 1e−8; > 1e−10 | pass |
| wiring \|(σ_xx(ATM) − σ_xx(2b)) − σ^ATM_xx\| | 3e−10 Ha/Bohr³ | 7.05e−11 (dropping the s9 fold leaves 7.5e−10 → FAIL) |

Product analytic stress (Γ PBE-D4+ATM, 200 Ry; E = −34.331326946882 Ha):
σ_xx = −2.17593330×10⁻⁵, σ_yy = +2.00729249×10⁻⁵,
σ_zz = +5.82666928×10⁻⁵ Ha/Bohr³.

The strained x+h s9=1 pin agreed to 3.8×10⁻¹² against a 1×10⁻⁹ tolerance
(recorded; no unit test in the tree gates it).

The two product analytic runs (ATM vs 2b) converge reference densities
independently at SCF tol 10⁻⁶, leaving ~7×10⁻¹¹ fixed-P Lagrangian noise in
the wiring difference — hence the 3×10⁻¹⁰ wiring tolerance. The ATM signal
(~7.5×10⁻¹⁰ Ha/Bohr³) is resolved only through the strain-sample energy
pins; the dual gate alone cannot see it.

### Performance

perf_exempt (no fair QS `STRESS_TENSOR` wall on the pin host). Measured walls
(OMP=8, Release): product 24.4 s; FD twin (x, warm) 4.9 s; two-body analytic
(x, warm) 6.1 s — case total ~36 s, within the ≤60 s regression budget.

## gpw-d4-kpoint-forces — multi-k two-body D4 ionic forces {#gpw-d4-kpoint-forces}

*No section of the case runs this variant.*

**Monkhorst–Pack multi-k GPW** ionic forces with additive two-body D4 BJ
(ATM off, \(s_9 = 0\)): `GpwDft` with `k_mesh = {1,1,2}`, `d4 = true`,
`d4_atm = false`, via `GpwDft::forces(frame)` or
`GPWEvaluator(std::move(dft), /*compute_forces=*/true)` (roadmap
`docs/roadmap/dft.md` §8 Dispersion + §10 Periodic KS). Γ folds two-body D4
into `assemble_gpw_forces` ([gpw-d4-forces](#gpw-d4-forces)); on the multi-k
path the same geometry-only \(F_{\mathrm{disp}} = -\nabla E_2\) is added in
`assemble_gpw_forces_kpoints` when `d4` is set:

\[
\mathbf{F}_A^{\mathrm{tot}}
  = \mathbf{F}_A^{\mathrm{k\text{-}HF/Pulay}}
  + \mathbf{F}_A^{\mathrm{disp}},
\quad
\mathbf{F}_A^{\mathrm{disp}}
  = -\nabla_{R_A} E_2,\quad s_9 = 0.
\]

Its pins: **dftd4 3.7.0** \(E_{\mathrm{disp}}\) + standalone forces
(geometry only, k-mesh independent); **CP2K 2026.1** multi-k `ENERGY_FORCE`
Δ (D4 − plain, libdftd4 rebuild); multi-k **wiring** identity after one
shared SCF; fair wall ratio **2.426**. Multi-k ATM forces and multi-k D4
stress use the same `k_mesh` + `d4` / `d4_atm` members; plain multi-k forces
are [gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces).
