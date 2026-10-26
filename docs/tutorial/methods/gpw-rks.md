# GPW closed-shell Kohn–Sham DFT

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Gaussian-and-plane-waves (GPW, CP2K Quickstep class) closed-shell Kohn–Sham
DFT. The density matrix lives in contracted Gaussians (MOLOPT tables), the
Hartree / XC / local short-range GTH pseudopotential terms are evaluated on a
uniform real-space grid set by a density cutoff, and cores are removed by
analytic GTH pseudopotentials. The concrete method is
`cpu::gpw::GpwDft{kinds, cutoff_ry, xc}`; every knob (`tol`, `mix`,
`max_iter`, `nonperiodic`, `k_mesh`, `ngrids`, `rel_cutoff`, `smearing`,
`stress_strain_h`, …) is a public member, with no `*Config` bag and no
factories. It runs directly through `build(frame)` / `run(frame)` (energy on
`GpwResult`, plus `kEnergy` [eV] on the Frame), `forces(frame)` and
`stress(frame)`, or bound by `cpu::gpw::GPWEvaluator` into `cpu::Evaluate` /
`cpu::single_point` (`compute_forces`, `compute_stress`, `stress_use_fd`;
pin channels `last_energy_hartree()`, `last_stress_ha_bohr3()`).

The sections below cover, on the CPU backend: Γ-point energy, analytic forces
and cell stress on isolated water under Martyna–Tuckerman (MT) electrostatics;
Monkhorst–Pack multi-k energy, forces and analytic stress on diamond Si; the
k-point × `REL_CUTOFF` multigrid product; MPI k-point round-robin; metallic
smearing; and the pure-GGA functionals BLYP, PBEsol, revPBE and RPBE. The case
also carries the CPU sections `gpw-kpoints-si`, `gpw-metallic-forces` and
`gpw-metallic-stress`, and the CUDA sections `cuda-gpw-*` (public
`cuda::gpw::GpwDft` against the same-settings CPU path); their oracle records
are in `REFERENCE.md`.

Case: `ctest -R '^gpw/rks$'` — sources and oracle record in [`regressions/gpw/rks/`](../../../regressions/gpw/rks/). One section of the case per `##` heading below: `atv_regressions --case gpw/rks --section <name>`.

## Common setup

### GPW SCF cycle

Density matrix \(P\) in a small contracted Gaussian basis \(\{\chi_\mu\}\).
Each SCF cycle:

1. Collocate \(\rho(\mathbf{r}) = \sum_{\mu\nu} P_{\mu\nu}\,\chi_\mu(\mathbf{r})\chi_\nu(\mathbf{r})\)
   onto a uniform real-space grid set by the plane-wave cutoff \(E_{\mathrm{cut}}\).
2. FFT → reciprocal space; solve Poisson for the Hartree potential; evaluate
   \(v_{\mathrm{xc}}[\rho]\) on the grid — \(\mathcal{O}(N\log N)\).
3. Integrate grid potentials back against the Gaussians to form
   \(V_{\mu\nu}^{\mathrm{Hxc}}\); add kinetic + GTH local/nonlocal PP in the
   AO basis; diagonalize (or OT — not on this page); rebuild \(P\) by direct
   \(P\)-mixing or DIIS until the residual is \(\le\) `tol`.

`local_pp_on_grid = true` (product default) collocates the local
short-range GTH PP; energy and analytic gradient both read it.

### Kinds, tables and units

A `cpu::gpw::Kind{Element, GaussianBasis(Element, name), GthPotential(Element, name)}`
pairs a basis and a pseudopotential through lookup constructors. Basis/PP
tables are static in-tree headers (`cpu/gpw/basis_data/`, `gpw_tables.h`,
transcribed from CP2K `BASIS_MOLOPT` / `GTH_POTENTIALS`); an unsupported
element or name **throws** (no silent fallback). The XC choice is independent
of the PP name: every functional on this page uses the GTH-PBE rows.

Frame forces are eV/Å and Frame stress `kStress` is eV (Voigt virial σ·Ω);
the CP2K pins are in Ha, Ha/Bohr and Ha/Bohr³ (`units::kEvPerAngstromInHartreePerBohr`,
`units::kHartreePerBohrInEvPerAngstrom`, `units::kHartreePerBohr3InEvPerAngstrom3`,
`kHartreeInEv`, `kBohrInAngstrom`).

### Isolated electrostatics (MT)

A finite cubic cell with ordinary periodic Poisson couples a molecule to its
images. Martyna–Tuckerman replaces the reciprocal-space Coulomb kernel so that
the electrostatic potential matches the free-space solution for a charge
density localized inside the cell. `GpwDft` exposes this as
`nonperiodic = true` (CP2K `PERIODIC NONE` / `PSOLVER MT`).

### Water system (water sections)

CP2K getting-started `h2o.inp` geometry, cubic 10 Å cell, MT.

| Atom | \(x\) [Å] | \(y\) [Å] | \(z\) [Å] |
| ---- | ----: | ----: | ----: |
| O | 5.0000 | 5.0000 | 5.0000 |
| H1 | 5.7586 | 5.0000 | 5.5043 |
| H2 | 4.2414 | 5.0000 | 5.5043 |

| Knob | Value |
| ---- | ----- |
| Basis | DZVP-MOLOPT-GTH (O, H) |
| PP | GTH-PBE-q6 (O), GTH-PBE-q1 (H) → 8 valence e⁻, \(n_{\mathrm{occ}} = 4\) |
| Density cutoff | 400 Ry (CP2K twin: REL_CUTOFF 60 Ry, NGRIDS 4) |
| SCF | direct \(P\)-mixing \(\alpha = 0.3\), `max_iter = 200`, `tol = 1e-7` (EPS_SCF) |
| Spin / BC | closed-shell RKS, Γ only; MT (`nonperiodic = true`) |

```cpp
cpu::Frame frame(3);
const int z[3] = {8, 1, 1};
const double x[3]  = {5.0000, 5.7586, 4.2414};
const double y[3]  = {5.0000, 5.0000, 5.0000};
const double zc[3] = {5.0000, 5.5043, 5.5043};
for (int i = 0; i < 3; ++i) {
    frame.span(kAtom_type)[i] = z[i];
    frame.span(kAtom_x)[i] = x[i];
    frame.span(kAtom_y)[i] = y[i];
    frame.span(kAtom_z)[i] = zc[i];
}
frame.box = cpu::Box::cubic(10.0);

const cpu::gpw::Kind O{
    molrs::Element::O,
    cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};
const cpu::gpw::Kind H{
    molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PBE-q1")};

cpu::gpw::GpwDft dft{{O, H}, 400.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 200;  dft.tol = 1e-7;  dft.mix = 0.3;
dft.nonperiodic = true;
```

### Diamond Si BCT cell (k-point sections)

| Item | Value |
| ---- | ----- |
| Lattice | \(a = 5.43\) Å (10.26120 Bohr); orthorhombic cell \((a/\sqrt{2},\,a/\sqrt{2},\,a)\) |
| Atoms (fractional) | \((0,0,0)\), \((\tfrac12,\tfrac12,\tfrac12)\), \((\tfrac12,0,\tfrac14)\), \((0,\tfrac12,\tfrac34)\) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 → 16 valence e⁻, `n_occ = 8` |
| XC | PBE |

```cpp
const double a = 5.43, at = a / std::sqrt(2.0);   // Å
const double frac[4][3] = {
    {0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, 0.0, 0.25}, {0.0, 0.5, 0.75}};
cpu::Frame frame(4);
for (int i = 0; i < 4; ++i) {
    frame.span(kAtom_type)[i] = 14;
    frame.span(kAtom_x)[i] = frac[i][0] * at;
    frame.span(kAtom_y)[i] = frac[i][1] * at;
    frame.span(kAtom_z)[i] = frac[i][2] * a;
}
frame.box = cpu::Box::orthorhombic(at, at, a);
const cpu::gpw::Kind si{molrs::Element::Si,
    cpu::gpw::GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
    cpu::gpw::GthPotential(molrs::Element::Si, "GTH-PBE-q4")};
```

**Monkhorst–Pack sampling** (CP2K `&KPOINTS SCHEME MONKHORST-PACK`):
\(\kappa_d = (2r - N_d - 1)/(2 N_d)\), \(r = 1,\ldots,N_d\), so even \(N_d\) is a
Γ-avoiding shifted mesh. `k_time_reversal = true` (product default) folds
\(\pm\mathbf{k}\) pairs and doubles the retained weight; explicit fractional
lists (`k_points` + `k_point_weights` summing to 1) override `k_mesh`. Matrices
are complex and Bloch-phased (phased lattice sums, Kleinman–Bylander nonlocal,
per-k DIIS); \(E = \sum_{\mathbf{k}} w_{\mathbf{k}} E_{\mathbf{k}}\). The multi-k
`build()` rejects nonperiodic MT, OT and DFT+U.

### Cell stress convention

Orthorhombic diagonal strain \(\varepsilon_{aa} = \pm h\) at fixed fractional
coordinates (scale \(L_a\) and Cartesian \(r_a\) by \(1\pm h\)); CP2K
`STRESS_TENSOR` sign

\[
\sigma_{aa}
= -\Omega^{-1}\frac{\partial E}{\partial\varepsilon_{aa}}
\approx -\Omega^{-1}\frac{E(+h_a)-E(-h_a)}{2h},
\qquad a\in\{x,y,z\},
\]

with \(E\) in Hartree and \(\Omega\) in Bohr³; positive \(\sigma_{xx}\) when
expanding the box lowers the energy. Voigt order
\((\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})\);
default \(h = 2\times 10^{-3}\) (`GpwDft::stress_strain_h`, axis mask
`stress_axes`). Off-diagonal shear is **exactly 0** (orthorhombic only).

- **Product (analytic fixed-\(P\) + Pulay).** `GpwDft::stress` and
  `GPWEvaluator{compute_stress = true}` (default `stress_use_fd = false`): one
  SCF, then central FD of the non-self-consistent Lagrangian at fixed density
  matrix — see [gpw-analytic-stress](#gpw-analytic-stress).
- **Oracle (multi-SCF strain-FD).** Free function `gpw_stress_fd(frame, method, h)`
  [eV/Å³] (or `stress_use_fd = true`): **6 strained SCF** (2 per diagonal)
  **+ 1** unperturbed SCF.

CP2K prints stress in **bar**: \(1\,\mathrm{Ha/Bohr}^3 = 2.94210107994716\times 10^{8}\,\mathrm{bar}\)
(CP2K internal table; `2.94210107994716e8`).

### Pure-GGA functionals

`XcKind::{PADE, PBE, PBEsol, BLYP, RevPBE, RPBE}`. All GGAs route through the
shared pointwise engine also used by all-electron RKS,
`evaluate_xc(XCType::…, ρ, σ, …)` (`xc_functional.h`, `is_gga`), and on the
GPW grid through `xc_grid_gga(…, XCType::…)` (`src/cpu/gpw/internal/xc_grid_gga.cpp`,
`xc_grid_gga.h`): closed-shell assembly builds spectral \(\nabla\rho\) and the
divergence of the \(v_\sigma\) flux exactly as for `XcKind::PBE`; the `XcKind`
selects the kernel. Unit gates under `tests/` pin each GGA's LDA limit
(\(\sigma = 0\) ≡ PBE) and a finite-\(\sigma\) difference.

The BLYP / PBEsol / revPBE / RPBE sections carry no standing regression and
were recorded as a **dual-gate**: the same geometry / basis / PP / cutoff /
MT with `XcKind::PBE` must shift the energy by more than \(10^{-4}\) Ha (hard
functional-activation criterion), the SCF must converge with finite energy,
and a soft \(1\times 10^{-3}\) Ha absolute band freezes the first recorded
Atomiverse energy. No CP2K binary / module was on the recording host (`cp2k`,
`cp2k.psmp`, `cp2k.sopt` not on PATH; `module avail cp2k` empty), so no
absolute CP2K digits exist for them; a CP2K twin input is staged for an
offline re-pin, which then replaces the soft band (cite the CP2K version) and
tightens the tolerance toward ~1 mHa like `gpw-water-pbe-mt`. Each is
energy-only on Γ-RKS: the analytic force/stress assembly accepts the
`XcKind` through the shared GGA grid, but forces are not their observable
([gpw-forces-water](#gpw-forces-water) is the force section).

## Oracle

Offline **CP2K 2026.1** Quickstep GPW (`cp2k.psmp`, EasyBuild
`2026.1-gmpflf-2026.03`, module `CP2K/2026.1`, Arrhenius epyc). Pins are
hard-coded in the section sources — **never** live CP2K (or any third-party
process) from ctest. Fair wall: single rank, \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \le 10\)
on the same host / core count; a section without a timed CP2K twin is
`perf_exempt` and states no ratio. Mechanism unit tests live under
`tests/cpu/gpw/` (e.g. `test_cpu_gpw_multigrid.cpp`); analytic-vs-FD forces and
stress and the smearing path have no unit test in the tree. CP2K parity is
exclusively the regression case.

Offline re-run (operator only):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
OMP_NUM_THREADS=4 mpirun -np 1 cp2k.psmp -i <CP2K oracle input for the section> -o out
# ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]  ...
# STRESS| Numerical stress tensor [bar]  ...
```

## Shared limitations

- **Closed-shell RKS only** — open-shell GPW is [gpw-uks](gpw-uks.md#gpw-uks-o2-gamma);
  hybrids, meta-GGA and GAPW are not covered.
- **Grid residual dominates** the MT water agreement at \(\sim 10^{-4}\) Ha /
  Ha/Bohr; these are not sub-µHa absolute energy targets (contrast molecular
  Gaussian SCF pins). The water stack is not bit-identical to CP2K (MT
  screening / grid details).
- Stress is orthorhombic-diagonal only; off-diagonal Voigt components are hard
  zeros.

## References

- J. VandeVondele, M. Krack, F. Mohamed, M. Parrinello, T. Chassaing,
  J. Hutter, *Comput. Phys. Commun.* **167**, 103 (2005) — CP2K Quickstep;
  GPW forces/stress; MOLOPT / DZVP-MOLOPT-GTH table family.
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997);
  *Theor. Chem. Acc.* **103**, 124 (1999) — GPW / Quickstep.
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — GTH / PADE.
- M. Krack, *Theor. Chem. Acc.* **114**, 145 (2005) — GTH tables.
- J. VandeVondele, J. Hutter, *J. Chem. Phys.* **127**, 114105 (2007) — MOLOPT.
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson.
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — MP k sampling.
- O. H. Nielsen, R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) — stress.
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE form.
- CP2K manual: FORCE_EVAL / QS stress tensor section (`STRESS_TENSOR NUMERICAL` /
  `ANALYTICAL`); `&MGRID` CUTOFF / REL_CUTOFF / NGRIDS; `&KPOINTS SCHEME MONKHORST-PACK`.
- Narrative: [Ch. 9 GPW](../ch09-gpw.md); example `examples/09_gpw.cpp`,
  `examples/results/09_gpw.json`; roadmap [dft.md](../../roadmap/dft.md).
- Spec records (git): `94e64ea9` (gpw-forces-water); `79de745b`, `d5e67a44`,
  `c20e5145`, `2a52d0e4` (gpw-stress-water); `dc7ad1bc`, `6d7f0955`
  (gpw-water-pbe-mt); `7022224b` (gpw-pbesol-si); `e086865c` (gpw-revpbe-si).

## gpw-forces-water — Γ analytic forces — H₂O / PBE (MT) {#gpw-forces-water}

Analytic nuclear forces on isolated water, matching CP2K Quickstep
`ENERGY_FORCE`. `GpwDft` (water setup above) bound by
`GPWEvaluator(std::move(dft), /*compute_forces=*/true)` in `cpu::Evaluate`;
forces land in `frame.span(kAtom_f*)` [eV/Å] after one SCF + gradient.

### Method

\[
\mathbf{F}_A = -\frac{\partial E}{\partial \mathbf{R}_A}
= \mathbf{F}_A^{\mathrm{HF}} + \mathbf{F}_A^{\mathrm{Pulay}} + \mathbf{F}_A^{\mathrm{grid/core}},
\]

Hellmann–Feynman collects the explicit \(\mathbf{R}_A\)-dependence of the
Hamiltonian (local PP, nonlocal projectors, nuclear–electron terms); Pulay is
the energy-weighted density matrix contracted with \(\partial S/\partial R_A\);
grid/core terms cover the collocated short-range local PP and the multigrid
discretization consistent with the SCF functional.

`local_pp_on_grid` must be the same for energy and gradient: an energy with
grid PP and a force path that omits the collocated local SR PP gives a ZnO
analytic-vs-FD residual of order 0.53 Ha/Bohr. The product path keeps both
on grid PP (`cpu/gpw/internal/local_pp.h`).

```cpp
cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/true);
cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);
// kEnergy [eV]; eval.last_energy_hartree() [Ha]
// kAtom_fx/y/z [eV/Å] → Ha/Bohr via units::kEvPerAngstromInHartreePerBohr
```

### Validation

`RUN_TYPE ENERGY_FORCE`, Cartesian forces [Ha/Bohr], \(E_{\mathrm{tot}}\) soft check.

| Quantity | CP2K pin | Tol | Notes |
| -------- | -------: | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−17.21233715058673** | \(1\times 10^{-3}\) Ha | Soft; multigrid residual ~\(4\times 10^{-4}\) Ha |
| \(F_{\mathrm{O}}\) [Ha/Bohr] | \((-6.62274644\times 10^{-8},\,-1.03414433\times 10^{-6},\,-8.96660013\times 10^{-2})\) | max component \(1\times 10^{-3}\) | A6 |
| \(F_{\mathrm{H1}}\) [Ha/Bohr] | \((+5.16349669\times 10^{-2},\,-2.88080683\times 10^{-9},\,+4.49073908\times 10^{-2})\) | same | |
| \(F_{\mathrm{H2}}\) [Ha/Bohr] | \((-5.16349704\times 10^{-2},\,-2.33864544\times 10^{-10},\,+4.49073720\times 10^{-2})\) | same | |
| \(\|\sum_A\mathbf{F}_A\|\) [Ha/Bohr] | CP2K print residual \(\sim 1.5\times 10^{-4}\) | \(\le 5\times 10^{-4}\) | A7; numerical residual, not free-molecule machine zero |

| Measured | Value |
| ------ | ----: |
| max\|F_atv − F_cp2k\| | **3.42×10⁻⁴** Ha/Bohr |
| \|∑F\| | **2.1×10⁻⁴** Ha/Bohr |
| \|ΔE\| | **4.2×10⁻⁴** Ha |
| SCF iters | **9** |

The component tolerance \(1\times 10^{-3}\) Ha/Bohr is set by grid /
multigrid / mixing residuals, not chemical accuracy; the physical \(|F|\) on
O\(_z\) is \(\sim 0.09\) Ha/Bohr. The translational-residual gate matches
CP2K's own printed residual scale.

### Performance

Arrhenius epyc, single rank, 4 OMP threads, `OMP_PROC_BIND=close` (CP2K:
`mpirun -n 1 cp2k.psmp -i h2o_forces_mt.inp`).

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | -----------------------: | ----: | ---- |
| H₂O GPW ENERGY_FORCE | **11.2** | **39.9** | **3.56** | ✓ (`perf_exempt = false`) |

### Limitations

- Isolated MT water only; BOMD trajectories (`gpw-bomd-water`) and supercell
  phonons ([fd-phonon-supercell](response-phonons.md#fd-phonon-supercell)) are
  other cases.
- Finite-difference forces are a unit test only (analytic vs FD on a tiny system).

## gpw-stress-water — Γ cell stress vs CP2K NUMERICAL — H₂O / PBE (MT) {#gpw-stress-water}

Cell stress on isolated water against CP2K `STRESS_TENSOR NUMERICAL`.
`GPWEvaluator(std::move(dft), /*compute_forces=*/false)` with
`compute_stress = true` and `eval.method().stress_strain_h = 2e-3`; stress on
the Frame as `kStress` [eV], pin channel `last_stress_ha_bohr3()`
[Ha/Bohr³]. The section leaves `stress_use_fd` at its default, so it runs the
analytic fixed-\(P\) product path (residual ~10⁻⁸ Ha/Bohr³ vs this pin); the
multi-SCF strain-FD path (`stress_use_fd = true`) carries the measured
numbers below.

**Why NUMERICAL, not ANALYTICAL.** On this nonperiodic MT water twin, CP2K
`STRESS_TENSOR ANALYTICAL` reports diagonals of order \(-3\times 10^{5}\) bar
while `NUMERICAL` reports order \(+10^{3}\) bar. Atomiverse matches
**NUMERICAL** in magnitude and sign after the convention above; do not pin
against `ANALYTICAL` on this MT cluster. CP2K prints
`STRESS| Numerical stress tensor [bar]` with xx ~ +6.46e3, zz ~ +3.74e3,
yy ~ +3.89.

```cpp
cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/false);
eval.compute_stress = true;
eval.method().stress_strain_h = 2e-3;
// … cpu::Evaluate as in gpw-forces-water …
// kStress [eV] Voigt virial σ·Ω; eval.last_stress_ha_bohr3() [Ha/Bohr³]
```

### Validation

`RUN_TYPE ENERGY_FORCE` + `STRESS_TENSOR NUMERICAL`.

| Quantity | CP2K pin | Tol | Notes |
| -------- | -------: | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−17.212337150586624** | \(1\times 10^{-3}\) Ha | Soft (same class as forces) |
| \(\sigma_{xx}\) [Ha/Bohr³] | **2.195224914616×10⁻⁵** | max diag \(5\times 10^{-6}\) | From \(+6.45857359202\times 10^{3}\) bar |
| \(\sigma_{yy}\) [Ha/Bohr³] | **1.321554147453×10⁻⁸** | same | From \(+3.88814588443\) bar |
| \(\sigma_{zz}\) [Ha/Bohr³] | **1.270036962196×10⁻⁵** | same | From \(+3.73657711805\times 10^{3}\) bar |
| \(\sigma_{xy},\sigma_{xz},\sigma_{yz}\) | \(\sim 10^{-14}\)–\(10^{-15}\) | ATV **exactly 0**; soft \(\lvert\sigma_{\mathrm{cp2k}}\rvert\le 10^{-8}\) | Orthorhombic |

| Measured (strain-FD) | Value |
| ------ | ----: |
| max\|\(\sigma_{\mathrm{diag,atv}}-\sigma_{\mathrm{cp2k}}\)\| | **9.1×10⁻⁹** Ha/Bohr³ |
| \|ΔE\| | **4.2×10⁻⁴** Ha |
| SCF iters (unperturbed) | **9** |

The diagonal gate \(5\times 10^{-6}\) Ha/Bohr³ is generous against the
measured residual \(\sim 10^{-8}\) (sub-percent of
\(\lvert\sigma_{xx}\rvert\sim 2\times 10^{-5}\)); it absorbs strain-step / SCF
/ multigrid differences while requiring correct order and sign. Parity is
tight because both sides differentiate the same functional class.

### Performance

Arrhenius epyc, single rank, 4 OMP, `OMP_PROC_BIND=close`. The fair
denominator is the CP2K ENERGY_FORCE single-SCF wall, not the multi-SCF
NUMERICAL wall.

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| H₂O GPW ENERGY_FORCE baseline | **10.15** | **48.2** | **4.75** | ✓ | Fair gate; ATV strain-FD 6+1 SCF |
| H₂O GPW full NUMERICAL stress | **~87** | **48.2** | **~0.55** | ✓ | FD vs FD context only |

### Limitations

- Isolated MT water; periodic solid stress is
  [gpw-kpoint-stress](#gpw-kpoint-stress) /
  [pw-analytic-stress-si](pw-rks.md#pw-analytic-stress-si); variable-cell is
  [gpw-variable-cell-molopt](gpw-variable-cell.md#gpw-variable-cell-molopt).
- No UKS stress or D3/D3BJ/D4 stress contribution in this section.

## gpw-analytic-stress — Γ analytic fixed-P stress — H₂O / PBE (MT) {#gpw-analytic-stress}

**Hellmann–Feynman fixed-\(P\) cell stress** for closed-shell Γ-point GPW:
one SCF, then central FD of the non-self-consistent Lagrangian

\[
E_L[P,W;\varepsilon]
  = \mathrm{Tr}(P\,T) + \mathrm{Tr}(P\,V_{\mathrm{pp}})
  + E_{\mathrm{es}}[\rho_P] + E_{\mathrm{xc}}[\rho_P] + E_{\mathrm{disp}}
  - \mathrm{Tr}(W\,S)
\]

under orthorhombic diagonal strain at fixed fractional coordinates. The
\(-\mathrm{Tr}(W\,S)\) term is the Pulay contribution
(\(W = \sum_i 2\varepsilon_i c_i c_i^\top\)). Scope: closed-shell Γ RKS,
`local_pp_on_grid` consistent with the SCF, optional D3/D3(BJ) geometry
energy, axes mask (`stress_axes`). Not covered: closed-form term tables without
fixed-\(P\) samples, shear/triclinic, UKS / k-points / DFT+U / D4 here, and
a CP2K `ANALYTICAL` absolute pin on nonperiodic MT (see
[gpw-stress-water](#gpw-stress-water)).

```cpp
cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/false);
eval.compute_stress = true;
eval.stress_use_fd = false;          // product analytic (default)
eval.method().stress_strain_h = 2e-3;
// … cpu::Evaluate …
const auto& s = eval.last_stress_ha_bohr3();   // Ha/Bohr³ Voigt

// Oracle: multi-SCF strain-FD on a second GpwDft with the same members
const auto sfd = cpu::gpw::gpw_stress_fd(frame, method, /*h=*/2e-3);  // eV/Å³
```

### Validation

| Gate | Value | Tol |
| ---- | ----- | --- |
| \(E_{\mathrm{tot}}\) vs CP2K pin | −17.212337150586624 Ha | \(1\times 10^{-3}\) Ha |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| diag | measured ~**3×10⁻¹⁰** Ha/Bohr³ | **1×10⁻⁵** Ha/Bohr³ |
| Off-diagonal analytic | 0 | \(1\times 10^{-8}\) Ha/Bohr³ |
| Frame `kStress` | = σ·Ω in eV | unit-conversion check |

The absolute CP2K NUMERICAL stress pin is
[gpw-stress-water](#gpw-stress-water), which runs this analytic path by
default.

## gpw-water-pbe-mt — Γ-RKS energy — H₂O / PBE (MT) {#gpw-water-pbe-mt}

Total energy of isolated water (water setup above) through
`GPWEvaluator(std::move(dft))` and `cpu::single_point(frame, eval)`. The
section asserts SCF convergence, a finite `frame.meta.get(kEnergy)` [eV], and
\(|E_{\mathrm{tot}} - E_{\mathrm{CP2K}}| \le 10^{-3}\) Ha.

```cpp
cpu::gpw::GPWEvaluator eval(std::move(dft));
cpu::single_point(frame, eval);
const double e_ha = eval.last_energy_hartree();   // [Hartree]
const bool ok = eval.last_result().converged;
```

### Validation

Oracle: CP2K Quickstep getting-started `h2o.inp` family (same geometry,
DZVP-MOLOPT-GTH + GTH-PBE, CUTOFF 400, PERIODIC NONE / PSOLVER MT, PBE),
cross-recorded by `examples/09_gpw.cpp` and [Ch. 9](../ch09-gpw.md).

| Term (CP2K) | Ha |
| ---- | ---: |
| self energy | −44.34715500 |
| core Hamiltonian | +13.07138998 |
| Hartree | +18.34339895 |
| XC | −4.27997143 |
| **total** | **−17.21233714458516** |

| Quantity | Atomiverse | CP2K pin | Tol |
| -------- | ---------: | -------: | --- |
| \(E_{\mathrm{tot}}\) | **−17.2127580657 Ha** | **−17.21233714458516 Ha** | **\(1\times 10^{-3}\) Ha** |
| \(\lvert\Delta E\rvert\) | \(4.209\times 10^{-4}\) Ha | — | < 1 mHa |
| SCF | converged, 9 iters | — | `converged == true` |

The ~0.42 mHa residual on this minimal stack sits inside a **1 mHa** band
that still fails gross method / geometry / cutoff mistakes; sub-mHa / MT
kernel bit-parity (erf-screened MT tuning) is not this section's bar.

### Performance

`perf_exempt`: no CP2K binary was timed on the recording host.
\(T_{\mathrm{ATV}}\) ≈ 17.7 s (single-point SCF, OMP=1, Release).

## gpw-kpoint-stress — analytic stress at k-points — diamond Si / PBE {#gpw-kpoint-stress}

Hellmann–Feynman fixed-\(P_k\) analytic cell stress on the Si BCT cell with
MP sampling. `GpwDft::stress(frame)` [eV/Å³] (also writes `kStress` /
`kEnergy`) is the product; `gpw_stress_fd` is the dual-gate oracle. The
driver-facing equivalent is `GPWEvaluator{compute_stress = true}` in
`cpu::Evaluate`; this section calls the method directly.

### Method

After one converged SCF, store \(C_{\mathrm{occ},k}\) and \(\varepsilon_{ik}\)
→ rebuild \(P_k\) and \(W_k = \sum_i 2\varepsilon_{ik}\,c_{ik}c_{ik}^\dagger\)
per k. Under \(\varepsilon_{aa}=\pm h\) rebuild \(S_k\), \(T_k\),
\(V_{\mathrm{nl}}^k\), grid, Poisson and XC, and evaluate (without
re-converging the SCF)

\[
E_L[\{P_k\},\{W_k\};\varepsilon]
  = \sum_k w_k\Bigl(
      \mathrm{Tr}(P_k T_k)
    + \mathrm{Tr}(P_k V_{\mathrm{pp}}^k)
    - \mathrm{Tr}(W_k S_k)
    \Bigr)
    + E_{\mathrm{es}}[\rho_P]
    + E_{\mathrm{xc}}[\rho_P]
    + E_{\mathrm{disp}},
\]

then apply the stress convention above with \(E_L\).

| Knob | Value |
| ---- | ----- |
| Density cutoff | 100 Ry (`local_pp_on_grid = true`) |
| k mesh | MP \(1\times1\times2\) (\(\kappa_z = \pm 1/4\)); time reversal → 1 irreducible k |
| SCF | DIIS, `tol` \(1\times 10^{-8}\) Ha, `eps_default` \(1\times 10^{-7}\), `max_iter` 60 |
| Strain step | \(h = 2\times 10^{-3}\) |

```cpp
auto make_cfg = [&] {
    cpu::gpw::GpwDft m{{si}, 100.0, cpu::gpw::XcKind::PBE};
    m.n_occ = 8;  m.max_iter = 60;  m.tol = 1e-8;  m.eps_default = 1e-7;
    m.local_pp_on_grid = true;
    m.k_mesh[0] = 1;  m.k_mesh[1] = 1;  m.k_mesh[2] = 2;
    return m;
};
cpu::gpw::GpwDft energy = make_cfg();
energy.build(frame);
const auto& er = energy.run(frame);            // er.e_total [Ha], er.C_occ_k

cpu::gpw::GpwDft sm = make_cfg();
sm.stress_strain_h = 2e-3;
cpu::Frame sframe = frame;                     // stress writes kStress / kEnergy
sm.build(sframe);
auto san = sm.stress(sframe);                  // analytic [eV/Å³]
auto sfd = cpu::gpw::gpw_stress_fd(frame, make_cfg(), 2e-3);  // oracle [eV/Å³]
// compare in Ha/Bohr³: divide by units::kHartreePerBohr3InEvPerAngstrom3
```

### Validation

No absolute CP2K `STRESS_TENSOR` digits exist for this section (no CP2K
binary/module on the recording host; a `STRESS_TENSOR NUMERICAL` twin input is
staged). The decisive gate is analytic vs multi-SCF FD; the absolute energy
and stress values are frozen Atomiverse numbers, to be replaced by cited CP2K
numbers when the twin runs.

| Quantity | Pin / measured | Tol | Notes |
| -------- | -------------: | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−15.3483039265** | \(1\times 10^{-3}\) Ha | Frozen ATV band (100 Ry mesh) |
| Soft fold band | vs `gpw-kpoints-si` A2 \(E=-15.3483059090\) Ha @ 95 Ry | \(5\times 10^{-3}\) Ha | Mesh/cutoff soft check class |
| \(\sigma_{xx}=\sigma_{yy}\) analytic [Ha/Bohr³] | **8.83135454×10⁻⁴** | \(1\times 10^{-4}\) soft abs | Frozen ATV |
| \(\sigma_{zz}\) analytic [Ha/Bohr³] | **2.60970168×10⁻⁴** | same | Tetragonal-cell anisotropy |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| diag | **7.2×10⁻⁸** | \(1\times 10^{-4}\) | Decisive dual-gate |
| Off-diagonal | 0 | \(1\times 10^{-8}\) | Orthorhombic |
| max\|\(\sigma_{\mathrm{diag}}\)\| floor | — | \(>1\times 10^{-6}\) | Non-trivial magnitude |
| SCF iters (unperturbed) | **7** | — | |

The \(1\times 10^{-4}\) Ha/Bohr³ dual-gate absorbs mesh / \(O(h^2)\)
residual while requiring analytic and FD to agree on order and sign
(measured \(\sim 10^{-7}\)).

### Performance

`perf_exempt` (no CP2K wall twin). \(T_{\mathrm{ATV}}\) ≈ **171** s (OMP=8,
Release) for analytic pass + multi-SCF FD (6 strained SCF + reference); the
product path alone is one SCF + fixed-\(P_k\) samples. A future twin should use
an ENERGY_FORCE-class denominator as in [gpw-stress-water](#gpw-stress-water).

### Limitations

- 100 Ry, \(1\times1\times2\) is a dual-gate system, not production-converged
  bulk Si stress.
- No UKS k-stress, hybrid, DFT+U, meta-GGA, GAPW or dispersion stress here.

## gpw-multigrid-kpoints — multi-k with REL_CUTOFF multigrid — diamond Si / PBE {#gpw-multigrid-kpoints}

Closed-shell PBE with MP sampling **and** CP2K-style `CUTOFF` / `REL_CUTOFF`
multigrid product collocation (`GpwDft` members `k_mesh`, `k_time_reversal`,
`ngrids`, `rel_cutoff`, `eps_default`). Pins the Si BCT cell, MP 2×2×2
TR-folded, CUTOFF 150 Ry, NGRIDS 4, REL_CUTOFF 60 Ry against CP2K. Γ
multigrid is [gpw-uks-multigrid](gpw-uks.md#gpw-uks-multigrid); single-grid
k-point energy is `gpw-kpoints-si`.

### Method

**Multigrid (`&MGRID`).** `ngrids` geometrically spaced levels; finest =
`CUTOFF`; each primitive-pair product goes to the coarsest level that still
resolves its most-diffuse exponent (governed by `REL_CUTOFF`). Per-level
densities are **prolonged spectrally (FFT)** to the fine grid; Hartree + XC
run there; the fine potential is **restricted** (exact adjoint) back to each
level — so the KS matrix stays the gradient of the collocated energy.

**Bloch sampling over the same product list.** Each product's periodic
ket-image shift **t** becomes a phase index:

- density weight per product from the real-space stack
  \(D(t) = \sum_k w_k \,\mathrm{Re}\!\left[P^k e^{-2\pi i\,\kappa\cdot t}\right]\)
  (Γ limit: \(D(t) = P\));
- integrated potential lands in real per-cell stacks \(V_t\), phase-summed
  per k into the Hermitian KS matrix \(V^k = \sum_t e^{+2\pi i\,\kappa\cdot t}\, V_t\).

`k_time_reversal = true` folds 8 k to 4 k with doubled weights; full mesh and
folded mesh agree machine-exactly (\(\lvert\Delta\rvert\) ≤ 3e-14 Ha).

**Product screening is mass-aware.** Screening whole products at
`eps_default` as a *value* threshold discards integrated mass up to
\((\pi/p)^{3/2}\approx 70\times\) that value per product, and a periodic cell
holds thousands of image pairs at threshold: on this system a raw value
screen integrates the density to 15.910 of 16 e⁻ (~0.09 e⁻ lost) and leaves
the SCF 4.1e-2 Ha off single grid (a 4e-2 Ha error), while every MT-isolated multigrid section
stays green. The derived margin `kProductMassMargin = 1e-4`
(`internal/multigrid.cpp`, Γ and k paths alike) prevents this; do not weaken it.

Nonlocal vdW (`vdw_df` / `vdw_df2` / `rvv10`) on the k path is rejected at
`build()`.

```cpp
cpu::gpw::GpwDft m{{si}, 150.0, cpu::gpw::XcKind::PBE};   // CP2K CUTOFF
m.ngrids = 4;           // NGRIDS
m.rel_cutoff = 60.0;    // REL_CUTOFF [Ry]
m.n_occ = 8;  m.max_iter = 60;  m.tol = 1e-8;  m.eps_default = 1e-7;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
m.k_time_reversal = true;                      // 4 k, w = ¼
m.build(frame);
const auto& r = m.run(frame);   // r.e_total, r.converged, r.k_frac / r.k_weights / r.eps_k
```

### Validation

| Quantity | CP2K pin | Gate | Tol | Notes |
| -------- | -------: | ---- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha], NGRIDS 4 | **−15.802416412503845** | M5 \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{CP2K}}\rvert\) | **1e-3 Ha** | measured **4.73e-7 Ha**; ATV −15.8024168854 Ha, 7 DIIS iters |
| \(E_{\mathrm{tot}}\) [Ha], NGRIDS 1 (context) | −15.802416412655166 | — | — | CP2K mg↔sg gap 1.5e-10 Ha |
| mg vs sg (ATV, 40 Ry / REL 20) | — | M3 \(\lvert E_{\mathrm{mg}}-E_{\mathrm{sg}}\rvert\) | 1e-3 Ha | measured 1.13e-5 Ha |
| SCF | converged | M2 / M4 `r.converged`, finite, k payload shapes | — | 4 TR k-points, weights ¼ |

SCF: DIIS depth 6, tol 1e-8 Ha, `EPS_DEFAULT 1e-7` on both sides; CP2K
converges in 13 steps
(`ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]     -15.802416412503845`).

Exactness identities (unit-pinned in `test_cpu_gpw_multigrid.cpp`,
`GpwMultigridCells.*`):

| Identity | Measured |
| --- | --- |
| Γ-limit: explicit k = (0,0,0) mg k-path vs Γ mg product path (95 Ry) | 2.0e-14 Ha |
| Time reversal: full 8k vs TR-folded 4k (95 / 60 / 40 Ry mg) | ≤ 2.8e-14 Ha |
| mg vs sg at 150 Ry / REL 60 | 2.28e-5 Ha |

The mg↔sg gap is the REL_CUTOFF product-screen residual (+3.5e-5 Ha at
`eps_default 1e-7`, +2.6e-6 at 1e-9), an mHa-class budget rather than CP2K's
1e-10-class internal gap; tightening `eps_default` converges it. Full-SCF TR
/ Γ-limit twins are not re-run in the section (≈15 s per multigrid SCF; the
≤60 s budget goes to the pin).

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------: | -----------------------: | ----: | ---- | ------ |
| Si BCT4 PBE, MP 2×2×2 TR, 150 Ry, NGRIDS 4 | **23.6** | **38.6** | **1.6** | ✓ | CP2K: 1 rank × OMP 4 (`si_kpoints_mg.inp`); ATV: Release, OMP 8, main MP run alone; whole section (3 SCFs) ≈ 55 s at OMP 8 |

### Limitations

- RKS energy only: no UKS×k multigrid, no k-point multigrid forces/stress.
- No hybrids / meta-GGA / GAPW; single-rank pin, no MPI scaling claim.

## mpi-gpw-scaling-4rank — MPI k-point round-robin — Si / 4 ranks {#mpi-gpw-scaling-4rank}

MPI k-point round-robin for the multi-k RKS SCF, matching the plane-wave
pattern in `cpu::pw`:

\[
i_k \in \{\,r,\; r+P,\; r+2P,\; \ldots\,\},
\quad r = \mathrm{rank},\; P = n_{\mathrm{procs}}.
\]

Each rank collocates only its local Bloch density contribution, builds and
diagonalizes \(H_{k}\) for those points, and accumulates weighted kinetic /
nonlocal traces, followed by `MPI_Allreduce(SUM)` on \(\rho(\mathbf{r})\)
(full real-space grid), on \(E_{\mathrm{kin}}\), \(E_{\mathrm{nl}}\) and on the
Pulay \(B_{ij}\) block inside k-DIIS, and `MPI_Allreduce(MAX)` on the DIIS
residual norm — every rank obtains the **same** total energy and SCF
trajectory. With MPI not initialized or `nprocs == 1` the distributed branches
are inactive and the path is bit-identical to the serial k-path. Setup of
\(S_k\), \(T_k\), \(V_{\mathrm{nl}}^k\) and the pre-summed complex
basis-on-grid stays **replicated**; the real-space grid and FFT are not
distributed. Requires `-DATV_USE_MPI=ON`.

```cpp
cpu::gpw::GpwDft m{{si}, 80.0, cpu::gpw::XcKind::PBE};   // light twin for ≤ 60 s
m.n_occ = 8;  m.max_iter = 40;  m.tol = 1e-7;  m.eps_default = 1e-6;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
m.k_time_reversal = true;
m.build(frame);
const auto& r = m.run(frame);
// r.e_total identical on every rank; r.n_kpts == 4; sum of r.n_kpts_local == n_kpts
```

### Validation

The oracle is an exact identity (rank-count invariance), not a third-party
number.

- **Section:** the parent runs the serial SCF, self-launches
  `mpiexec -n 4 <self> --mpi-worker`, and requires
  \(|E_1 - E_4| \le 10^{-8}\) Ha and `n_kpts == 4` on both; the wall ratio is
  recorded.
- **Unit:** no MPI unit test in the tree covers the k-point split (all ranks
  agreeing on `e_total` to \(10^{-12}\) Ha at n=1/2/4;
  `sum(n_kpts_local) == n_kpts`); the section is the only gate. The PW path is
  in the same position.

### Performance

`perf_exempt`: k-point round-robin only, setup replicated, no CP2K multi-rank twin.

## gpw-blyp-water — Γ-RKS BLYP — H₂O (MT) {#gpw-blyp-water}

*No section of the case runs this variant.*

Water setup above with `XcKind::BLYP`, run directly as
`dft.build(frame); dft.run(frame)` (`GpwResult::e_total` [Ha]). BLYP (Becke,
*Phys. Rev. A* **38**, 3098 (1988) + Lee–Yang–Parr, *Phys. Rev. B* **37**, 785
(1988)) is the pure GGA

\[
E_{\mathrm{xc}}^{\mathrm{BLYP}}[\rho] = E_{\mathrm{x}}^{\mathrm{B88}}[\rho] + E_{\mathrm{c}}^{\mathrm{LYP}}[\rho],
\]

libxc `GGA_X_B88` + `GGA_C_LYP`, CP2K `XC_FUNCTIONAL BLYP`; on the grid
`xc_grid_blyp = xc_grid_gga(…, XCType::BLYP)`. This is the GPW BLYP path,
distinct from all-electron [blyp-water-sto3g](scf-rks.md#blyp-water-sto3g).

### Validation (dual-gate)

| Quantity | Atomiverse | Oracle / pin | Tol |
| -------- | ---------: | -----------: | --- |
| \(E_{\mathrm{BLYP}}\) | **−17.1982170663 Ha** | frozen ATV soft band | **\(1\times 10^{-3}\) Ha** |
| \(E_{\mathrm{PBE}}\) (twin) | −17.2127580657 Ha | CP2K −17.212337 Ha ([gpw-water-pbe-mt](#gpw-water-pbe-mt)) | 1 mHa there; residual ~0.42 mHa |
| \(\lvert E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}\rvert\) | \(1.454\times 10^{-2}\) Ha | floor \(10^{-4}\) Ha | **hard** \(> 10^{-4}\) |
| SCF (BLYP) | converged, 12 iters | — | `converged == true` |

`perf_exempt`; \(T_{\mathrm{ATV}}\) ≈ 34.4 s BLYP, ≈ 19.1 s PBE twin (OMP=4,
Release). UKS + GGA is [gpw-uks-pbe](gpw-uks.md#gpw-uks-pbe).

## gpw-kpoint-forces — analytic forces at k-points — H₂ / PADE (MP 1×1×2) {#gpw-kpoint-forces}

*No section of the case runs this variant.*

Periodic multi-k RKS with **analytic ionic forces** on a compressed H₂ bond:
`GpwDft` with `k_mesh`, then `forces(frame)` (SCF + one gradient pass,
`kAtom_f*` [eV/Å]). One SCF on the complex Bloch path builds \(S_k\), \(T_k\),
nonlocal PP and grid \(\chi^k\) (`kpoints.h`); the gradient assembles
Hellmann–Feynman + Pulay terms k-summed with weights,
\(\mathbf{F}_A = \sum_{\mathbf{k}} w_{\mathbf{k}} \mathbf{F}_A(\mathbf{k})\), with
\(P_k\) and \(W_k\) rebuilt from `GpwResult::C_occ_k` (`internal/gpw_forces.h`).
AO-centre derivatives use the Gaussian raising/lowering identity;
operator-centre derivatives follow translational invariance. At SCF
stationarity this is \(-\nabla_{\mathbf{R}}E\) of the same multi-k functional
(shown by central FD). The k-force path rejects nonperiodic MT, OT, DFT+U and
multigrid at `build()` (periodic single-grid `local_pp_on_grid` only).

| Knob | Value |
| ---- | ----- |
| Positions [Bohr] | H1 \((5.5, 6.0, 6.0)\), H2 \((6.5, 6.0, 6.0)\) (1.0 Bohr compressed bond along \(x\)) |
| Cell | cubic 12 Bohr, periodic (`nonperiodic = false`) |
| XC / basis / PP | PADE LDA / DZV-GTH-PADE (in-tree) / GTH-PADE-q1 |
| Cutoff | 100 Ry, single grid, `local_pp_on_grid = true` |
| k mesh | MP \(1\times 1\times 2\), TR on → one \(\kappa_z = \pm 1/4\) class, weight 1 |
| SCF | `tol` \(1\times 10^{-8}\), `max_iter` 60, `n_occ = 1`, `eps_default = 1e-8` |

```cpp
cpu::gpw::GpwDft m{{cpu::gpw::Kind{molrs::Element::H,
    cpu::gpw::GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
    cpu::gpw::GthPotential(molrs::Element::H, "GTH-PADE-q1")}}, 100.0, cpu::gpw::XcKind::PADE};
m.n_occ = 1;  m.tol = 1e-8;  m.max_iter = 60;  m.eps_default = 1e-8;
m.nonperiodic = false;  m.local_pp_on_grid = true;
m.k_mesh[0] = 1;  m.k_mesh[1] = 1;  m.k_mesh[2] = 2;
frame.box = cpu::Box::cubic(12.0 * units::kBohrInAngstrom);   // positions in Bohr × kBohrInAngstrom
m.build(frame);
m.forces(frame);   // kAtom_f* [eV/Å]; / units::kHartreePerBohrInEvPerAngstrom → Ha/Bohr
```

### Validation

Decisive correctness is the analytic ≡ FD identity on the k-path energy; the
absolute values are frozen Atomiverse numbers. No same-basis CP2K twin exists:
DZV-GTH-PADE is not in CP2K `BASIS_MOLOPT`, so the staged CP2K 2026.1
`ENERGY_FORCE` twin input uses DZVP-MOLOPT-GTH. An exploratory SZV-MOLOPT run differs in basis/geometry
(\(E\approx -1.026\) Ha, \|F_x\|\(\approx 0.18\) Ha/Bohr) and is not a gate.

| Check | Result | Notes |
| ----- | -----: | ----- |
| max\|F_analytic − F_fd\| | **\(5.98\times 10^{-6}\)** Ha/Bohr | Off-symmetry geometry; \(h=10^{-3}\) Bohr; 120 Ry; OMP=8 |
| Explicit-Γ vs real Γ forces | **\(1.1\times 10^{-10}\)** Ha/Bohr | Complex k machinery at \(\kappa=0\) vs `assemble_gpw_forces` |
| \|∑F\| | \(\sim 3\times 10^{-14}\) Ha/Bohr | Exact Newton-3 |

| Quantity | Atomiverse | Tol |
| -------- | ---------: | --- |
| \(E_{\mathrm{tot}}\) [Ha] | **−1.057153829426** (8 SCF iters) | \(1\times 10^{-5}\) Ha |
| \(F_{0x}\) / \(F_{1x}\) [Ha/Bohr] | **−0.23541914472** / **+0.23541914472** | \(1\times 10^{-5}\) |
| \(F_y, F_z\) | \(\sim 0\) | \(\le 10^{-6}\) |
| \|∑F\| [Ha/Bohr] | \(\sim 10^{-14}\) | \(\le 10^{-6}\) (A4) |

`perf_exempt` (no same-basis CP2K wall); \(T_{\mathrm{ATV}}\) ≈ 1 s (OMP=8,
Release). k-UKS energy is [gpw-kpoints-uks](gpw-uks.md#gpw-kpoints-uks);
Γ UKS forces are [gpw-uks-forces](gpw-uks.md#gpw-uks-forces).

## gpw-metallic-smearing — Fermi–Dirac smearing — BCC Na / multi-k {#gpw-metallic-smearing}

*No section of the case runs this variant.*

Fractional band occupations with a chemical potential \(\mu\) shared over all
k. `GpwDft` members `smearing` (`Smearing::fermi_dirac` /
`methfessel_paxton`, `mp_order`), `degauss` [Ry], `extra_bands`, `k_mesh`;
results `GpwResult::{e_total, e_fermi, n_elec_occ, e_entropy}`. Occupations
reuse the plane-wave fill (`cpu/planewave/internal/smearing.h`: Fermi
bisection + FD / MP kernels). Smearing forces/stress are the case sections
`gpw-metallic-forces` / `gpw-metallic-stress`. PW counterpart:
[pw-smearing-al](pw-smearing.md#pw-smearing-al).

### Method

Spinless \(f(x)\in[0,1]\), \(x = (\varepsilon-\mu)/\sigma\), doubled to band
occupation \(f_b\in[0,2]\); \(\sigma\) = `degauss` converted to Hartree.
Fermi–Dirac \(f = 1/(1+e^{x})\); Methfessel–Paxton series of order `mp_order`.
Fermi bisection fixes

\[
n_{\mathrm{elec}} = \sum_{k,b} w_k\, f_b(\varepsilon_{kb},\mu,\sigma) = Z_{\mathrm{val}}\cdot N_{\mathrm{atoms}}
\]

(\(Z_{\mathrm{val}}=9\) per Na → \(N_e = 18\)). The SCF total is the Mermin
free energy

\[
F = E_{\mathrm{KS}} - TS,\qquad
E_{\mathrm{entropy}} = -TS = -\sigma\sum_{k,b} w_k\, s_{kb},
\]

with FD entropy \(s = -2[f\ln f+(1-f)\ln(1-f)]\) in band form; `e_total` [Ha]
is \(F\), `e_entropy` the \(-TS\) piece, `e_fermi` \(\mu\) [Ha], `n_elec_occ`
\(\sum w f\). `extra_bands` mirrors CP2K `ADDED_MOS` (GPW diagonalizes the full
AO spectrum). `Smearing::none` keeps integer fillings. Metallic smearing is
the multi-k closed-shell RKS path only; `build()` throws for UKS
(`n_alpha` / `n_beta`), OT (`use_ot`; OT needs integer occupations),
`degauss <= 0` with smearing on, or Methfessel–Paxton with `mp_order < 1`. No
cold / Marzari–Vanderbilt or Gaussian enum on GPW. UKS + smearing is the PW
case [pw-uks-fe-smearing](pw-smearing.md#pw-uks-fe-smearing).

References: N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965); M. Methfessel,
A. T. Paxton, *Phys. Rev. B* **40**, 3616 (1989).

```cpp
const double a = 4.225;   // BCC Na conventional cubic [Å]; Na at (0,0,0), (½,½,½)
cpu::gpw::GpwDft m{{cpu::gpw::Kind{molrs::Element::Na,
    cpu::gpw::GaussianBasis(molrs::Element::Na, "DZVP-MOLOPT-SR-GTH"),
    cpu::gpw::GthPotential(molrs::Element::Na, "GTH-PBE-q9")}}, 150.0, cpu::gpw::XcKind::PADE};
m.n_occ = 9;  m.max_iter = 80;  m.tol = 1e-6;  m.eps_default = 1e-7;
m.diis_depth = 6;  m.mix = 0.4;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;  m.k_time_reversal = true;
m.smearing = cpu::gpw::Smearing::fermi_dirac;
m.degauss = 0.02;      // Ry (≈ CP2K ELECTRONIC_TEMPERATURE 3157.75 K)
m.extra_bands = 4;     // CP2K ADDED_MOS
m.build(frame);
const auto& r = m.run(frame);   // r.e_total = F [Ha], r.e_fermi, r.n_elec_occ, r.e_entropy
```

### Validation

ATV system: PADE, 150 Ry single grid, MP \(2\times2\times2\) TR, DIIS 6,
`tol = 1e-6` Ha, `eps_default = 1e-7`, `mix = 0.4` (Arrhenius x86_64).

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| Free energy \(F\) | **−92.126330608378 Ha** | **\(10^{-4}\) Ha** | frozen ATV value |
| \(N_{\mathrm{occ}}=\sum w f\) | **18.0** | \(10^{-5}\) | valence count |
| \(E_F\) | **0.055826 Ha** | \(\lvert E_F\rvert < 10^3\) Ha | finite, not NaN |
| \(-\sigma S\) | **−2.26×10⁻⁴ Ha** | finite | diagnostic |
| SCF | converged (~7 iters), \(n_{\mathrm{kpts}}>1\) | `r.converged` | |

Unit tests: the Fermi–Dirac fill recovers \(N_e\) within \(10^{-8}\) on a
synthetic spectrum.

CP2K 2026.1 (`cp2k.psmp`, source 85b8a9b49c) documents the Quickstep
counterpart on the same system class (`na_bcc_fermi_dirac.inp`: GPW, PADE,
`GTH-PBE-q9` + `DZVP-MOLOPT-SR-GTH`, `&SMEAR METHOD FERMI_DIRAC`,
\(T_e = 3157.75\) K, MP \(2\times2\times2\), CUTOFF / REL_CUTOFF 150 / 40 Ry,
NGRIDS 4, Broyden): \(E_{\mathrm{tot}}\) **−92.76593904 Ha** (8 SCF steps),
Fermi energy **0.054489 Ha**. This is not an absolute twin — grid /
multigrid / `EPS_DEFAULT` / mixer differences give a ~0.6 Ha gap; electron
count and SMEAR algorithm class are what match.

Wall: CP2K **89.51** s vs ATV **0.999** s, \(r \approx 0.011\)
(`OMP_NUM_THREADS=4`, single rank) — indicative only, not a same-multigrid
throughput claim.

## gpw-pbesol-si — Γ-RKS PBEsol — Si₂ (MT) {#gpw-pbesol-si}

*No section of the case runs this variant.*

PBEsol (Perdew et al., *Phys. Rev. Lett.* **100**, 136406 (2008)) is PBE with
solid-state parameters; PW92 LDA correlation is unchanged and PBEsol ≡ PBE at
\(\sigma = |\nabla\rho|^2 = 0\). libxc / CP2K `GGA_{X,C}_PBE_SOL` /
`XC_FUNCTIONAL PBEsol`; `XcKind::PBEsol` → `XCType::PBEsol`.

| Piece | PBE | PBEsol |
| ----- | --: | -----: |
| Exchange \(\mu\) | \(0.21951\) | \(10/81\) |
| Exchange \(\kappa\) | \(0.804\) | \(0.804\) |
| Correlation \(\beta\) | \(0.066725\) | \(0.046\) |

### Si₂ fragment (shared with gpw-revpbe-si)

| Item | Value |
| ---- | ----- |
| System | Si₂ diamond fragment in a cubic \(a = 5.43\) Å cell, MT (`nonperiodic = true`) |
| Positions [Å] | \((0,0,0)\), \((a/4,a/4,a/4) = (1.3575,1.3575,1.3575)\) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q4 (zion 4) → 8 e⁻, `n_occ = 4` |
| Cutoff | **200 Ry** |
| SCF | direct \(P\)-mixing \(\alpha=0.4\), `max_iter=60`, `tol=1e-6` |

Periodic Γ-only MOLOPT image sums for this cell exceed the ≤ 60 s budget, so
the fragment uses MT; the dual-gate still proves functional activation, and
the staged CP2K twin uses the same MT / 200 Ry deck. Bulk Si is the k-point
sections; the conventional 8-atom cell is outside the budget.

```cpp
constexpr double a = 5.43;
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 14;
    frame.span(kAtom_x)[i] = frame.span(kAtom_y)[i] = frame.span(kAtom_z)[i] = i * a / 4.0;
}
frame.box = cpu::Box::cubic(a);
cpu::gpw::GpwDft m{{si}, 200.0, cpu::gpw::XcKind::PBEsol};  // XcKind::PBE for the twin
m.n_occ = 4;  m.max_iter = 60;  m.tol = 1e-6;  m.mix = 0.4;
m.nonperiodic = true;
m.build(frame);
const auto& r = m.run(frame);   // or GPWEvaluator + cpu::single_point
```

### Validation (dual-gate)

| Quantity | Atomiverse | Oracle / pin | Tol |
| -------- | ---------: | -----------: | --- |
| \(E_{\mathrm{PBEsol}}\) | **−13.0542687637 Ha** | frozen ATV soft band | **\(1\times 10^{-3}\) Ha** |
| \(E_{\mathrm{PBE}}\) (twin) | −13.0743742822 Ha | same-host twin | — |
| \(\lvert E_{\mathrm{PBEsol}}-E_{\mathrm{PBE}}\rvert\) | \(2.011\times 10^{-2}\) Ha | floor \(>10^{-4}\) Ha | **hard** |
| SCF (PBEsol) | converged, 11 iters | — | `converged == true` |

Unit gates: PBEsol@\(\sigma=0\) ≡ PBE@\(\sigma=0\) to \(10^{-12}\); at finite
\(\sigma\) \(|e_{\mathrm{xc}}|\) differs by \(>10^{-8}\); Si GTH/MOLOPT table
digits pinned. `perf_exempt`; \(T_{\mathrm{ATV}}\) ≈ 0.7 s for both SCFs
(~0.4 s PBEsol + ~0.3 s PBE, OMP=4, Release). PW true PBE on Si:
[pw-pbe-gga-si](pw-rks.md#pw-pbe-gga-si).

## gpw-revpbe-si — Γ-RKS revPBE — Si₂ (MT) {#gpw-revpbe-si}

*No section of the case runs this variant.*

revPBE (Zhang & Yang, *Phys. Rev. Lett.* **80**, 890 (1998)) revises PBE
exchange to \(\kappa = 1.245\), keeping \(\mu = 0.21951\), \(\beta = 0.066725\)
and PW92 correlation; the larger \(\kappa\) softens the exchange enhancement at
finite \(\sigma\), and revPBE ≡ PBE at \(\sigma = 0\). libxc / CP2K
`GGA_X_PBE_R` + `GGA_C_PBE` / `XC_FUNCTIONAL revPBE`; `XcKind::RevPBE` →
`xc_grid_revpbe` → `xc_grid_gga(…, XCType::RevPBE)` (pointwise
\(\kappa=1.245\) in `src/cpu/scf/xc_functional.cpp`). System, settings and code
are the [gpw-pbesol-si](#gpw-pbesol-si) Si₂ fragment with
`XcKind::RevPBE`. Not RPBE — see [gpw-rpbe-water](#gpw-rpbe-water).

### Validation (dual-gate)

| Quantity | Atomiverse | Oracle / pin | Tol |
| -------- | ---------: | -----------: | --- |
| \(E_{\mathrm{revPBE}}\) | **−13.0900394799 Ha** | frozen ATV soft band | **\(1\times 10^{-3}\) Ha** |
| \(E_{\mathrm{PBE}}\) (twin) | −13.0743742822 Ha | same-host twin (= gpw-pbesol-si PBE) | — |
| \(\lvert E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}\rvert\) | \(1.567\times 10^{-2}\) Ha | floor \(>10^{-4}\) Ha | **hard** |
| SCF (revPBE) | converged, 10 iters | — | `converged == true` |

`perf_exempt`; \(T_{\mathrm{ATV}}\) ≈ 24 s for both SCFs (~14.8 s revPBE +
~9.2 s PBE, OMP=4, Release).

## gpw-rpbe-water — Γ-RKS RPBE — H₂O (MT) {#gpw-rpbe-water}

*No section of the case runs this variant.*

RPBE (Hammer, Hansen, Nørskov, *Phys. Rev. B* **59**, 7413 (1999)) keeps PBE's
\(\kappa = 0.804\) and \(\mu \approx 0.21951\) but uses an exponential
exchange enhancement,

\[
F_x^{\mathrm{RPBE}}(s) = 1 + \kappa\bigl(1 - e^{-\mu s^2/\kappa}\bigr),
\qquad s = \frac{|\nabla\rho|}{2\,k_{\mathrm{F}}\,\rho},
\]

with pure PBE correlation (libxc `GGA_X_RPBE` + `GGA_C_PBE`; CP2K
`XC_FUNCTIONAL RPBE`; `xc_grid_rpbe` → `xc_grid_gga(…, XCType::RPBE)`). It is
**not** revPBE (κ-revision of the rational PBE form, `XcKind::RevPBE`,
[gpw-revpbe-si](#gpw-revpbe-si)); do not cross-cite energies. Water setup
above with `XcKind::RPBE`, through `GPWEvaluator` + `cpu::single_point`; the
geometry, basis, PP, cutoff and MT match [gpw-water-pbe-mt](#gpw-water-pbe-mt)
and [gpw-blyp-water](#gpw-blyp-water), so only the functional changes.

### Validation (dual-gate)

| Quantity | Atomiverse | Oracle / pin | Tol |
| -------- | ---------: | -----------: | --- |
| \(E_{\mathrm{RPBE}}\) | **−17.2435550328 Ha** | frozen ATV soft band | **\(1\times 10^{-3}\) Ha** |
| \(E_{\mathrm{PBE}}\) (twin) | −17.2127580657 Ha (≈ −17.212758) | CP2K −17.212337 Ha on gpw-water-pbe-mt | — |
| \(\lvert E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}\rvert\) | \(3.080\times 10^{-2}\) Ha | floor \(>10^{-4}\) Ha | **hard** |
| SCF (RPBE) | converged, 9 iters; finite Ha + Frame eV | — | `converged == true` |

`perf_exempt`; \(T_{\mathrm{ATV}}\) ≈ 20.8 s RPBE, ≈ 5.9 s PBE twin (OMP=4,
Release).
