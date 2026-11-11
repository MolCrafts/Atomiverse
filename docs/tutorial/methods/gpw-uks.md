# GPW unrestricted Kohn–Sham (UKS)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Collinear spin-polarized (unrestricted) Kohn–Sham in the
Gaussian-and-plane-waves (GPW, Quickstep-family) method. Two spin densities
\(\rho_\alpha\), \(\rho_\beta\) are built from separate density matrices.
Hartree acts on the total \(\rho=\rho_\alpha+\rho_\beta\) and XC on the spin
densities, and separate \(H_\alpha\), \(H_\beta\) are diagonalized with
dual-spin DIIS. The concrete method is `cpu::gpw::GpwDft`. Occupations are
the integer members `n_alpha` / `n_beta`, not a multiplicity bag. When both
are zero, the closed-shell `n_occ` path is unchanged.

The Γ-point O₂ sections move a `GpwDft` into `cpu::gpw::GPWEvaluator` and run
it either through `cpu::single_point(frame, eval)` (energy) or through a
`cpu::Evaluate` driver (`eval.compute_forces = true` for forces;
`eval.compute_stress = true`, `eval.stress_use_fd = false` for stress). The
multi-k, GGA-stress and smearing variants call `GpwDft::build(frame)` and then
`run(frame)`, `forces(frame)` or `stress(frame)` directly, with
`gpw_stress_fd` as the multi-SCF strain-FD oracle. Frame energy is eV
(`kEnergy`), Frame forces are eV/Å, Frame `kStress` is the Voigt virial σ·Ω
[eV], and `GpwDft::stress` / `gpw_stress_fd` return σ in eV/Å³. The pins
compare in Ha, Ha/Bohr and Ha/Bohr³ (`last_energy_hartree()`,
`last_stress_ha_bohr3()`, `GpwResult::e_total`).

The sections cover: the Γ-point triplet O₂ energy and ⟨S²⟩ for PADE, PBE,
BLYP, revPBE and RPBE; PADE with CP2K-style multigrid collocation; Γ-point
UKS analytic forces and cell stress (PBE); multi-k UKS energy, analytic
forces and analytic stress (PADE and the RevPBE/RPBE/BLYP spin GGAs) on
high-spin H₂; and multi-k UKS with Fermi–Dirac smearing on BCC Na. All of
these run on the CPU only. The case also has three sections whose oracle
record lives only in
[`REFERENCE.md`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/gpw/uks/REFERENCE.md):
`gpw-uks-metallic-stress` (BCC Na UKS + Fermi–Dirac free-energy analytic
stress against multi-SCF FD), `cuda-gpw-uks-o2-gate` and
`cuda-gpw-uks-o2-forces` (Γ O₂ UKS energy, forces and stress on
`cuda::gpw::GpwDft` against a same-settings CPU twin).

Case: `ctest -R '^gpw/uks$'` — sources and oracle record in [`regressions/gpw/uks/`](../../../regressions/gpw/uks/). One section of the case per `##` heading below: `atv_regressions --case gpw/uks --section <name>`.

Build with `-DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON
-DATV_BUILD_REGRESSIONS=ON` (FFTW3), using the login-shell recipe in the
repository `docs/developer/project-contract.md` (`module load buildenv-gcc/2026.03-mpich`).
Related: closed-shell GPW on [gpw-rks](gpw-rks.md#gpw-forces-water);
narrative [Ch. 9 GPW](../ch09-gpw.md); roadmap
[`docs/roadmap/dft.md`](../../roadmap/dft.md).

## Common setup

### GPW

The wavefunction uses a contracted Gaussian AO basis. The density is
collocated on a uniform real-space grid, Hartree is solved by FFT Poisson,
XC is evaluated on the grid, and the potential is integrated back into the
AO KS matrix. Analytic **GTH** pseudopotentials remove the core electrons,
and the plane-wave cutoff in Rydberg sets the grid resolution.

### Isolated electrostatics

`nonperiodic = true` selects the **Martyna–Tuckerman** Poisson solver, the
twin of CP2K `PERIODIC NONE` / `PSOLVER MT`. The molecule must be
**centered** in the cell. Nonperiodic MT grids cover \([0,L)\), so atoms near
an edge truncate Gaussian tails and break energy, force and stress parity.
Always center isolated UKS/RKS pins.

### Spin metric

After convergence, \(\langle S^2\rangle\) (`last_s2()`) is the Löwdin
expectation value for a single unrestricted determinant. It is not
spin-projected. The ideal high-spin value for \(S=1\) is \(S(S+1)=2\).
Neither ROKS nor spin projection is provided, so the residual
\(\langle S^2\rangle\) contamination is accepted and compared with CP2K's
single-determinant value.

### O₂ triplet system (Γ-point sections)

All Γ-point O₂ sections use the same geometry and spin bookkeeping. O with
GTH-PBE-q6 has 6 valence e⁻, giving 12 valence e⁻ in total. Multiplicity 3
gives \(N_\alpha=7\), \(N_\beta=5\), not the all-electron \(9/7\).

| Item | Value |
| ---- | ----- |
| Molecule | O₂, bond along *z*, **centered** |
| \(R(\mathrm{O–O})\) | **1.2075 Å** |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE` |
| Poisson | Martyna–Tuckerman (`PSOLVER MT` / `nonperiodic`) |
| Basis / PP | DZVP-MOLOPT-GTH / GTH-PBE-q6 |
| Cutoff | 400 Ry. ATV uses a single density grid unless `ngrids` is set; the CP2K twins use `NGRIDS 4` / `REL_CUTOFF 60` |
| Spin | UKS / LSD, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| SCF | ATV: DIIS-style mix 0.4, `max_iter=200`, `tol=1e-7`. CP2K: diagonalization + Broyden α=0.4 |

```cpp
using namespace ATV;

// O₂, R = 1.2075 Å, centered in a 12 Å cubic cell (MT requires centering)
cpu::Frame frame(2);
constexpr double L = 12.0;
constexpr double R = 1.2075;
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_atomic_number)[i] = 8;
    frame.span(kAtom_mass)[i] = 15.999;
    frame.span(kAtom_x)[i] = 0.5 * L;
    frame.span(kAtom_y)[i] = 0.5 * L;
}
frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
frame.box = cpu::Box::cubic(L);

cpu::gpw::Kind O{molrs::Element::O,
                 cpu::gpw::GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
                 cpu::gpw::GthPotential(molrs::Element::O, "GTH-PBE-q6")};

cpu::gpw::GpwDft dft{{O}, 400.0, cpu::gpw::XcKind::PADE};  // XC varies per section
dft.max_iter = 200;
dft.tol = 1e-7;
dft.mix = 0.4;
dft.nonperiodic = true;
dft.n_alpha = 7;  // multiplicity 3, 12 valence e⁻
dft.n_beta = 5;

cpu::gpw::GPWEvaluator eval(std::move(dft));
cpu::single_point(frame, eval);

const double e_ha = eval.last_energy_hartree();       // [Ha]
const double e_ev = frame.meta.get(kEnergy);          // [eV]
const double s2 = eval.last_s2();                     // Löwdin ⟨S²⟩
const cpu::gpw::GpwResult& r = eval.last_result();    // converged, iters, …
```

Every O₂ energy section gates the same checks: the CP2K total energy at
**1e-3 Ha**, \(\langle S^2\rangle\) at **0.15**, SCF convergence, and Frame
energy consistency \(E_{\mathrm{eV}}/k_{\mathrm{Ha\to eV}} \equiv E_{\mathrm{Ha}}\)
(`kEnergy` vs `last_energy_hartree()`, \(10^{-9}\) Ha). The 1e-3 Ha band
covers the grid twin (ATV single density grid vs CP2K multigrid `NGRIDS 4` /
`REL_CUTOFF 60`), not a missing term. Matching the grids does not make the
SCF bit-identical: the mixers differ (DIIS vs Broyden).

### High-spin H₂ system (multi-k forces and stress)

| Item | Value |
| ---- | ----- |
| System | High-spin H₂ (\(M=3\)), compressed H–H = 1.0 Bohr along \(x\) |
| Positions [Bohr] | H1 (5.5, 6.0, 6.0), H2 (6.5, 6.0, 6.0) |
| Cell | Cubic **12 Bohr**, `PERIODIC XYZ` (not MT) |
| Basis / PP | **DZV-GTH-PADE** (in-tree table) / GTH-PADE-q1 |
| XC | PADE LSD (`XcKind::PADE`) unless stated |
| Grid | CUTOFF **100 Ry**, NGRIDS 1 (single density grid), `local_pp_on_grid = true` |
| k mesh | MONKHORST-PACK **1×1×2**; time reversal folds it to one irreducible \(\kappa_z=\pm\tfrac14\) class with weight 1 |
| Spin | \(N_\alpha=2\), \(N_\beta=0\). H with GTH-PADE-q1 has 1 valence e⁻, so 2 in total |
| SCF | `tol = 1e-8`, `max_iter = 60` |

```cpp
using namespace ATV;
using namespace ATV::cpu::gpw;

// Geometry in Bohr, converted to Å for the Frame
constexpr double b2a = units::kBohrInAngstrom;
const double L = 12.0;  // Bohr
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_atomic_number)[i] = 1;
    frame.span(kAtom_x)[i] = (0.5 * L + (i == 0 ? -0.5 : 0.5)) * b2a;
    frame.span(kAtom_y)[i] = 0.5 * L * b2a;
    frame.span(kAtom_z)[i] = 0.5 * L * b2a;
}
frame.box = cpu::Box::cubic(L * b2a);
const Kind H{molrs::Element::H,
             GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
             GthPotential(molrs::Element::H, "GTH-PADE-q1")};

auto make_uks = [&](XcKind xc) {
    GpwDft m{{H}, 100.0, xc};
    m.n_alpha = 2;
    m.n_beta = 0;
    m.max_iter = 60;
    m.tol = 1e-8;
    m.local_pp_on_grid = true;   // energy and gradient both read it
    m.k_mesh[0] = 1;
    m.k_mesh[1] = 1;
    m.k_mesh[2] = 2;             // MONKHORST-PACK 1×1×2
    return m;
};
```

DZV-GTH-PADE is an in-tree Atomiverse table that CP2K `BASIS_MOLOPT` does
not provide. The CP2K inputs for this system therefore use
**SZV-MOLOPT-GTH** and serve as protocol/documentation twins only. No
absolute same-basis CP2K value exists for them, and their fair wall is
**`perf_exempt`**. Do not compare SZV and DZV absolute energies, forces or
stresses.

### Shared limitations

- CPU only. The CUDA GPW sections are the `cuda-*` sections above.
- Hybrids, meta-GGA, GAPW and PAW are excluded. All-electron molecular UKS
  belongs to Gaussian SCF and the PySCF loop.
- No MPI scaling claim. All pins are single-rank.
- XC-kernel and grid unit tests live under `tests/cpu/gpw/`
  (`test_cpu_gpw_multigrid.cpp`, `test_cpu_gpw_xc_grid_gga.cpp`); UKS
  analytic-vs-FD forces and stress have no unit test in the tree. End-to-end
  oracle parity lives only in the regression case.
- Public parameters are constructor arguments and `GpwDft` / `GPWEvaluator`
  members. There is no `*Config` bag and no `make_*` factory.

Sources: `gpw_solver.h` (`GpwDft`, `GpwResult`, `Smearing`, `stress`,
`gpw_stress_fd`), `src/cpu/gpw/gpw_solver.cpp`, `gpw_evaluator.h`,
`src/cpu/gpw/internal/gpw_stress.cpp`, `gpw_forces.h`, `kpoints.h`,
`multigrid.h`, `xc_grid.h`, `xc_grid_gga.h`.

## Oracle

CP2K **2026.1** Quickstep (`cp2k.psmp`, git:5e54ba2), offline only. CP2K
never runs from ctest. The inputs are under `tools/cp2k_oracle/inputs/<case>/`
and the module stack is on Arrhenius (GCC 14.3 / MPICH 4.3.2):

```bash
module load buildtool-easybuild/5.2.1-hpca3ef7d197 GCC/14.3.0 MPICH/4.3.2 CP2K/2026.1
export CP2K_DATA_DIR=$EBROOTCP2K/share/cp2k/data
export OMP_NUM_THREADS=4
cd /tmp && ln -sf $CP2K_DATA_DIR/{BASIS_MOLOPT,GTH_POTENTIALS} .
mpirun -np 1 cp2k.psmp -i <input>.inp -o <input>.out
```

Fair walls are \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) on the same
settings class (CP2K 1 MPI × 4 OMP; ATV Release, OMP=4 unless stated). The
gate is \(r \le 10\) or `perf_exempt` with a stated reason. CP2K NUMERICAL
stress in bar converts with 1 Ha/Bohr³ = 2.94210107994716×10⁸ bar (CP2K
internal table, the same one used by `gpw-stress-water`).

## References

- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — Quickstep, forces
- S. Goedecker, M. Teter, J. Hutter, *Phys. Rev. B* **54**, 1703 (1996) — PADE / GTH
- J. P. Perdew, K. Burke, M. Ernzerhof, *Phys. Rev. Lett.* **77**, 3865 (1996) — PBE
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- P.-O. Löwdin, *Phys. Rev.* **97**, 1509 (1955) — \(\langle S^2\rangle\) of unrestricted determinants
- O. H. Nielsen, R. M. Martin, *Phys. Rev. B* **32**, 3780 (1985) — stress theorem, fixed-\(P\) cell stress
- Y. Zhang, W. Yang, *Phys. Rev. Lett.* **80**, 890 (1998) — revPBE
- B. Hammer, L. B. Hansen, J. K. Nørskov, *Phys. Rev. B* **59**, 7413 (1999) — RPBE
- A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988) — B88 exchange; C. Lee, W. Yang, R. G. Parr, *Phys. Rev. B* **37**, 785 (1988) — LYP correlation
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — MP mesh
- N. D. Mermin, *Phys. Rev.* **137**, A1441 (1965); M. Methfessel, A. T. Paxton, *Phys. Rev. B* **40**, 3616 (1989) — smearing
- CP2K manual: Quickstep `&MGRID` (CUTOFF / REL_CUTOFF / NGRIDS), `STRESS_TENSOR NUMERICAL` / `ANALYTICAL`

## gpw-uks-forces — Γ-point UKS analytic forces, O₂ / PBE {#gpw-uks-forces}

### Method

This section computes analytic nuclear forces on the triplet O₂ system with
spin-resolved **PBE GGA** (`XcKind::PBE`). The UKS branch of
`GpwDft::forces` assembles

\[
\mathbf{F}_A = -\frac{\partial E}{\partial \mathbf{R}_A}
=
\mathbf{F}_A^{\mathrm{HF}}
+ \mathbf{F}_A^{\mathrm{Pulay}}
+ \mathbf{F}_A^{\mathrm{grid/core}}.
\]

- **Hellmann–Feynman** collects the explicit \(\mathbf{R}_A\) dependence of
  the Hamiltonian: local PP, nonlocal projectors and nuclear–electron terms.
- **Pulay** contracts the spin-summed energy-weighted density
  \(W = W_\alpha + W_\beta\) with \(\partial S/\partial R_A\).
- **Grid/core** terms use the spin-resolved effective potentials
  \(V_{\mathrm{eff}}^{\alpha/\beta}\) of the SCF functional, including the
  collocated short-range local PP when `local_pp_on_grid` is true (the
  default).

The energy and the analytic gradient must use the same `local_pp_on_grid`
setting. If the energy uses grid PP and the force path omits the collocated
local SR PP, the ZnO analytic-vs-FD residual is of order 0.53 Ha/Bohr. The
product path keeps both sides on grid PP (`cpu/gpw/internal/local_pp.h`).

### Code

```cpp
cpu::gpw::GpwDft dft{{O}, 400.0, cpu::gpw::XcKind::PBE};
// members as in Common setup (n_alpha = 7, n_beta = 5, nonperiodic, …)
cpu::gpw::GPWEvaluator eval(std::move(dft));
eval.compute_forces = true;

cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);
// Forces: kAtom_fx/y/z [eV/Å] → Ha/Bohr via units::kEvPerAngstromInHartreePerBohr
```

### Validation

The input is `tools/cp2k_oracle/inputs/gpw-uks-forces/o2_uks_forces.inp`
(`RUN_TYPE ENERGY_FORCE`, `LSD`, `MULTIPLICITY 3`, `EPS_SCF 1e-7`).

| Quantity | CP2K pin | Tol | Notes |
| -------- | -------: | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.928797634435675** | **1e-3 Ha** | ENERGY\| FORCE_EVAL (QS) line |
| \(F_z\) O1 [Ha/Bohr] | **−3.50081576×10⁻²** | **5e-3** max component | single density grid vs CP2K multigrid |
| \(F_z\) O2 [Ha/Bohr] | **+3.50081358×10⁻²** | same | equal-and-opposite bond force |
| \(F_x,F_y\) | ~1e-12 (numerical) | same | bond along *z* |
| \(\|\sum_A\mathbf{F}_A\|\) [Ha/Bohr] | CP2K residual \(\sim 2\times 10^{-8}\) | **5e-4** (\(\le 5\times 10^{-4}\)) | translational residual |

Hard-coded pins (Ha/Bohr):

```text
O1:  2.02069184e-12,  2.05677955e-12, -3.50081576e-02
O2:  2.43439049e-12,  2.40869221e-12,  3.50081358e-02
E_tot_Ha: -31.928797634435675
```

```text
  Total energy:                                               -31.92879767674141
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.928797634435675
 FORCES|      1  2.02069184E-12  2.05677955E-12 -3.50081576E-02   3.50081576E-02
 FORCES|      2  2.43439049E-12  2.40869221E-12  3.50081358E-02   3.50081358E-02
 FORCES| Total atomic force                                       2.17692380E-08
```

Measured: max\|F_atv − F_cp2k\| **≈ 2.7×10⁻³** Ha/Bohr on a force scale of
\|F_z\| **~0.035** Ha/Bohr, and \|ΔE\| **≈ 6×10⁻⁵** Ha (~6e-5, well inside the 1e-3 Ha energy band). The force residual
comes from the single density grid against CP2K's multigrid, not from a
missing term. UKS H₂ analytic forces agree with FD within **5e-5** Ha/Bohr on
the ATV energy functional (recorded; no unit test in the tree gates it).

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| O₂ UKS-PBE, Γ, ENERGY_FORCE | **77.3** | **195** | **2.52** | ✓ |

### Limitations

- Γ only. No ROKS and no spin-projected forces. For multi-k UKS forces see
  [gpw-kpoint-uks-forces](#gpw-kpoint-uks-forces), and for multi-k RKS
  forces [gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces).
- Isolated MT O₂ only: no periodic supercell, BOMD trajectory or
  geometry-optimization gate.

## gpw-uks-stress — Γ-point UKS analytic cell stress, O₂ / PBE {#gpw-uks-stress}

### Method

This section computes Hellmann–Feynman fixed-\(P\) cell stress on the triplet
O₂ system with PBE. After one converged UKS SCF at the unstrained cell:

1. Keep \(P_\alpha,P_\beta\) and the energy-weighted density matrices
   \(W_\sigma=\sum_i^{n_\sigma}\varepsilon_i^\sigma c_i^\sigma(c_i^\sigma)^\top\).
2. Apply a dimensionless diagonal strain \(h\) at **fixed fractional
   coordinates**, scaling \(L_a\) and Cartesian \(r_a\) by \(1\pm h\).
3. Rebuild \(T\), \(V_{\mathrm{pp}}\), \(S\), the grid, Poisson and spin XC,
   and form the **non-self-consistent** Lagrangian
   \(E_L[P_\alpha,P_\beta;W_\alpha,W_\beta;\varepsilon]\) without re-running
   the SCF.
4. Take the central difference:

\[
\sigma_{aa}
=
-\Omega^{-1}
\frac{\partial E_L}{\partial\varepsilon_{aa}}
\approx
-\Omega^{-1}
\frac{E_L(+h_a)-E_L(-h_a)}{2h},
\qquad a\in\{x,y,z\},
\]

with \(E\) in Hartree, \(\Omega\) in Bohr³ and the CP2K `STRESS_TENSOR` sign.
Voigt order is
\((\sigma_{xx},\sigma_{yy},\sigma_{zz},\sigma_{xy},\sigma_{xz},\sigma_{yz})\).
The default \(h = 2\times 10^{-3}\) is set by `stress_strain_h`. The
\(-\mathrm{Tr}(W\,S)\) piece is the Pulay contribution. Shear components are
**exactly 0** because only orthorhombic cells are supported.

The oracle is the multi-SCF strain FD `gpw_stress_fd`, which re-converges a
full UKS SCF at each strained geometry (6 SCFs for the three diagonal axes).
The decisive gate is max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| on
the diagonals: a same-method twin of the energy functional that does not
depend on the CP2K multigrid residual.

### Code

```cpp
auto make_method = [&O]() {
    cpu::gpw::GpwDft m({O}, 400.0, cpu::gpw::XcKind::PBE);
    m.n_alpha = 7;
    m.n_beta = 5;
    m.max_iter = 200;
    m.tol = 1e-7;
    m.mix = 0.4;
    m.nonperiodic = true;
    m.local_pp_on_grid = true;
    return m;
};

cpu::gpw::GPWEvaluator eval(make_method());
eval.compute_forces = false;
eval.compute_stress = true;
eval.stress_use_fd = false;           // analytic fixed-P (default)
eval.method().stress_strain_h = 2e-3;
// cpu::Evaluate driver as in gpw-uks-forces
const auto& s = eval.last_stress_ha_bohr3();   // Voigt [Ha/Bohr³]
// frame.meta.get(kStress) is the Voigt virial σ·Ω [eV]

cpu::gpw::GpwDft method = make_method();
method.build(frame);
auto sfd = cpu::gpw::gpw_stress_fd(frame, method, /*h=*/2e-3,
                                   /*axes*/ {true, true, true});  // [eV/Å³]
// compare in Ha/Bohr³: divide by units::kHartreePerBohr3InEvPerAngstrom3
```

### Validation

The energy pin is the ENERGY_FORCE twin of
[gpw-uks-forces](#gpw-uks-forces). The CP2K stress twin input
(`o2_uks_stress_num.inp`, `STRESS_TENSOR NUMERICAL`) exists, but no absolute
NUMERICAL Voigt digits are recorded because the pin host had no CP2K module.
Do not invent bar-scale values.

| Quantity | Pin / check | Tol | Notes |
| -------- | ----------- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | CP2K **−31.928797634435675** | **1×10⁻³** Ha | same ENERGY pin as gpw-uks-forces |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| diag | internal dual gate | **1×10⁻⁴** Ha/Bohr³ | decisive |
| Off-diagonal \(\sigma_{xy,xz,yz}\) | \(\approx 0\) | **1×10⁻⁸** Ha/Bohr³ | orthorhombic |
| max\|\(\sigma_{\mathrm{diag}}\)\| floor | nonzero | \(> 1\times 10^{-8}\) Ha/Bohr³ | degeneracy guard |
| Frame `kStress` vs Ha/Bohr³ | unit conversion | \(10^{-9}\) (eV/Å³) | `kHartreeInEv` / \(a_0^3\) |

Measured values:

| Quantity | Value |
| -------- | ----: |
| \(E_{\mathrm{ATV}}\) [Ha] | **−31.9288547285** (60 SCF iters) |
| \|E − E_cp2k\| | **5.7×10⁻⁵** Ha |
| \(\sigma_{xx}\) analytic | \(1.34582413\times 10^{-8}\) Ha/Bohr³ |
| \(\sigma_{yy}\) analytic | \(1.34582366\times 10^{-8}\) Ha/Bohr³ |
| \(\sigma_{zz}\) analytic | \(6.74708540\times 10^{-6}\) Ha/Bohr³ |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| | **5.0×10⁻¹¹** Ha/Bohr³ |
| \(\langle S^2\rangle\) | **2.0056** |

The dual-gate tolerance (**1×10⁻⁴**) is looser than for closed-shell water
(**1×10⁻⁵**) because the O₂ UKS residual and the single-grid vs multigrid
residual are larger. The measured residual is \(\sim 5\times 10^{-11}\).

### Performance

`perf_exempt`: CP2K was not available for a fair NUMERICAL stress wall. The
dual-gate wall is dominated by the multi-SCF FD (6 SCFs), not by the
analytic stress.

| Run (OMP=8, Release) | \(T_{\mathrm{ATV}}\) [s] |
| -------------------- | ----------------------: |
| Analytic stress | **140.6** |
| FD twin | **236.5** |
| Dual-gate total | **~377** |

For context only, the gpw-uks-forces ENERGY_FORCE CP2K wall (**~77.3** s,
OMP=4) against the analytic stress run (**~140.6** s) gives an informal
**~1.8**. This is not the acceptance ratio. When re-pinned, the target is
the analytic-only run against an ENERGY_FORCE-class fair twin, with
\(r\le 10\).

### Limitations

- Orthorhombic only: shear Voigt components are hard zero and triclinic
  cells are not supported.
- No ROKS, hybrid, DFT+U or dispersion stress here. For multi-k UKS stress
  see [gpw-kpoint-uks-stress](#gpw-kpoint-uks-stress). For closed-shell
  stress see [gpw-analytic-stress](gpw-rks.md#gpw-analytic-stress) and
  [gpw-stress-water](gpw-rks.md#gpw-stress-water).

## gpw-uks-multigrid — Γ-point UKS with multigrid collocation, O₂ / PADE {#gpw-uks-multigrid}

### Method

This section is the same open-shell physics as
[gpw-uks-o2-gamma](#gpw-uks-o2-gamma) (PADE LSD, dual-spin DIIS, Löwdin
\(\langle S^2\rangle\)), but with CP2K-style **multigrid product
collocation** (`ngrids > 1`, `rel_cutoff > 0`). A single uniform grid forces
every Gaussian onto the finest spacing. Diffuse functions then span huge
bounding boxes (slow) and leave an O(mHa) residual against CP2K multigrid.
The CP2K-style mapping is:

1. Build `ngrids` geometrically spaced levels. The finest cutoff is `CUTOFF`
   and the coarsest is about `REL_CUTOFF`.
2. Assign each basis function to the **coarsest** level that still resolves
   its most-diffuse exponent.
3. Collocate density contributions per level, with compact support on the
   coarse levels for diffuse Gaussians.
4. **Prolong** the per-level densities to the fine grid and run the
   nonlinear Hartree + XC there.
5. **Restrict** the fine potential back to each level and integrate the KS
   matrix.

Inter-level transfer uses **spectral (FFT) prolong / restrict**
(`prolong_fft` / `restrict_fft`). These are exact adjoints, so the
integrated KS matrix remains the gradient of the collocated energy
(variational SCF). FFT transfer is used even under Martyna–Tuckerman
`nonperiodic`: the density lives on a uniform box grid and is well contained
for centered O₂. A real-space trilinear transfer on `periodic=false` leaves
an O(0.4 Ha) Hartree residual. Collocation applies independently to the two
spin densities. The default `ngrids = 1` bypasses multigrid entirely (a
bit-identical single-grid path). The RKS multigrid machinery is shared
(`internal/multigrid.h`).

### Code

```cpp
cpu::gpw::GpwDft dft{{O}, 400.0, cpu::gpw::XcKind::PADE};
dft.ngrids = 4;        // CP2K &MGRID NGRIDS
dft.rel_cutoff = 60.0; // CP2K REL_CUTOFF [Ry]
// remaining members as in Common setup; GPWEvaluator + cpu::single_point
```

### Validation

The input is `o2_uks_pade_mg.inp` (`&MGRID NGRIDS 4` / `REL_CUTOFF 60` /
`CUTOFF 400`), with **NGRIDS 4**, **REL_CUTOFF 60** Ry on both sides. Energy
and \(\langle S^2\rangle\) are the same physics as the gpw-uks-o2-gamma
twin.

| Quantity | CP2K pin | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.76980349931425** | **1e-3 Ha** |
| \(\langle S^2\rangle\) (single det.) | **2.004659** | **0.15** |
| Integrated spin density (CP2K) | 2.000000 | log diagnostic only |

```text
Total energy:                                               -31.76980349931425
Ideal and single determinant S**2 :                    2.000000       2.004659
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.769803499314250
```

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| O₂ UKS-PADE, Γ, NGRIDS 4 (ATV `ngrids=4`, `rel_cutoff=60`) | **21.3** | **32.8** | **1.54** | ✓ |

This is slower than the single-grid gpw-uks-o2-gamma run (\(r \approx 0.98\))
because level assignment and prolong/restrict run at every SCF density
build.

### Limitations

- Pure trilinear transfer on the `nonperiodic` path would break energy
  parity. Spectral transfer is required.
- Γ only. Multi-k UKS multigrid is not claimed (the k path throws on
  multigrid).
- Set `ngrids` and `rel_cutoff` explicitly to twin CP2K multigrid. Multigrid
  does not change the public units: Frame energy stays in eV.

## gpw-uks-o2-gamma — Γ-point UKS, O₂ / PADE {#gpw-uks-o2-gamma}

### Method

This section is the Γ-point UKS energy with spin-resolved **PADE LSD**
(`xc_grid_pade_lsd`, `XcKind::PADE`), the Goedecker–Teter–Hutter Padé
parameterization of the LDA for spin-polarized densities (CP2K
`&XC_FUNCTIONAL PADE` + `LSD`). It runs on the single density grid and the
code is the Common setup snippet unchanged. The closed-shell MT twin is
[gpw-water-pbe-mt](gpw-rks.md#gpw-water-pbe-mt).

### Validation

Input `o2_uks_pade.inp`. CP2K uses multigrid (`NGRIDS 4`, `REL_CUTOFF 60`)
and ATV a single density grid.

| Quantity | CP2K pin | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.76980349931425** | **1e-3 Ha** |
| \(\langle S^2\rangle\) (single det.) | **2.004659** | **0.15** |
| Integrated spin density (CP2K) | 2.000000 | log diagnostic only |

```text
Total energy:                                               -31.76980349931425
Ideal and single determinant S**2 :                    2.000000       2.004659
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.769803499314250
```

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| O₂ UKS-PADE, Γ (ATV single grid) | **21.3** | **20.8** | **0.98** | ✓ |

### Limitations

- Energy and \(\langle S^2\rangle\) only. Forces and stress are in
  [gpw-uks-forces](#gpw-uks-forces) and [gpw-uks-stress](#gpw-uks-stress).
- Γ only. For multi-k see [gpw-kpoints-uks](#gpw-kpoints-uks). The
  plane-wave spin path is separate:
  [pw-spin-polarized-o2](pw-collinear-spin.md#pw-spin-polarized-o2).

## gpw-uks-pbe — Γ-point UKS, O₂ / PBE {#gpw-uks-pbe}

### Method

This section is the Γ-point UKS energy with spin-resolved **PBE GGA** on
\(\rho_\alpha\), \(\rho_\beta\) and their gradients (`xc_grid_pbe_spin`,
CP2K `&XC_FUNCTIONAL PBE` with `LSD`). It is the same GGA family as the
closed-shell `XcKind::PBE` RKS pins
([gpw-water-pbe-mt](gpw-rks.md#gpw-water-pbe-mt)), on the spin path. The code
is the Common setup snippet with `cpu::gpw::XcKind::PBE`.

### Validation

Input `o2_uks_pbe.inp`. CP2K uses `EPS_SCF 1e-6` (ATV `tol=1e-7`).

| Quantity | CP2K pin | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.92879303435954** | **1e-3 Ha** |
| \(\langle S^2\rangle\) (single det.) | **2.005556** | **0.15** |
| Ideal \(S(S+1)\) (CP2K) | 2.000000 | log diagnostic only |

```text
  *** SCF run converged in    14 steps ***
  Total energy:                                               -31.92879303435954
  Ideal and single determinant S**2 :                    2.000000       2.005556
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.928793034359536
```

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| O₂ UKS-PBE, Γ (ATV single grid) | **207** | **177** | **0.85** | ✓ |

### Limitations

- No DFT+U on this pin (see
  [gpw-dftu-nio](gpw-dft-plus-u.md#gpw-dftu-nio)).
- Do not compare absolute \(E_{\mathrm{tot}}\) across XC pins (PADE:
  gpw-uks-o2-gamma). For plane-wave multi-k spin see
  [pw-uks-kmesh-fe](pw-collinear-spin.md#pw-uks-kmesh-fe) and
  [pw-uks-fe-smearing](pw-smearing.md#pw-uks-fe-smearing).

## gpw-kpoint-uks-forces — Multi-k UKS analytic forces, high-spin H₂ / PADE {#gpw-kpoint-uks-forces}

*No section of the case runs this variant.*

### Method

This variant runs one multi-k UKS SCF followed by one gradient pass, on the
high-spin H₂ system (\(N_\alpha=2\), \(N_\beta=0\)). The gradient pass works
over Bloch-phased \(S_k/T_k/V_\mathrm{nl}\), dual density matrices
\(P_\alpha^k/P_\beta^k\), the energy-weighted \(W_k^\sigma\), and
spin-resolved \(V_\mathrm{eff}^{\alpha/\beta}\) against complex \(\chi^k\).
`k_mesh[3]` (or explicit `k_points` + weights) selects multi-k sampling.
Bloch-phased complex matrices live in `internal/kpoints.h`. At the SCF
stationary point the force splits into
\(\mathbf{F}_A^{\mathrm{HF}} + \mathbf{F}_A^{\mathrm{Pulay}} +
\mathbf{F}_A^{\mathrm{grid/core}}\), k-summed over occupied Bloch states:

- **Hellmann–Feynman** takes the explicit \(\mathbf{R}_A\) dependence of
  \(T + V_\mathrm{pp} + V_\mathrm{grid}\) with dual \(P_k^\sigma\).
- **Pulay** contracts \(W_k = W_k^\alpha + W_k^\beta\) (occupancy-1 spin
  channels) with \(\partial S_k/\partial R_A\).
- **Grid** holds the converged \(V_\mathrm{eff}^{\sigma}\) fixed.

```cpp
GpwDft m = make_uks(XcKind::PADE);   // High-spin H₂ system, Common setup
m.nonperiodic = false;
m.eps_default = 1e-8;
m.build(frame);
const GpwResult& er = m.forces(frame);  // e_total [Ha], C_occ_k / C_occ_k_beta
// kAtom_f* [eV/Å]; ÷ units::kHartreePerBohrInEvPerAngstrom → Ha/Bohr
// fx[0] < 0 < fx[1] (restoring bond); |F[0]+F[1]| ~ machine eps
```

### Validation

The primary numbers are frozen Atomiverse values (Release, OMP=8). The CP2K
ENERGY_FORCE input `h2_uks_kpoint_forces.inp` (`LSD`, `MULTIPLICITY 3`,
`KPOINTS SCHEME MONKHORST-PACK 1 1 2`, `CUTOFF 100`, `NGRIDS 1`,
SZV-MOLOPT-GTH) is a protocol twin only.

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------- | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−0.528389371092** | **1e-5 Ha** | ~6 SCF iters, TR-folded \(n_k\) |
| \(F_{0,x}\) [Ha/Bohr] | **−0.554544062995** | **1e-5** | restoring; atom 0 at smaller \(x\) |
| \(F_{1,x}\) [Ha/Bohr] | **+0.554544062995** | **1e-5** | Newton-3 pair |
| \(F_{a,y}\), \(F_{a,z}\) | ~0 by symmetry | 1e-6 | bond along \(x\) |
| \(\lvert\sum_a \mathbf{F}_a\rvert\) | ~\(1.6\times10^{-14}\) | **1e-6 Ha/Bohr** | translation residual |
| Newton-3 on \(F_x\) | \(\lvert F_{0,x}+F_{1,x}\rvert\) | **1e-6** | exact pair cancellation |
| Closed-shell k-UKS vs k-RKS max\(\lvert\mathrm{d}F\rvert\) | \(7.4\times10^{-7}\) (measured) | **5e-4 Ha/Bohr** | \(N_\alpha=N_\beta=1\) vs `n_occ=1` |
| Bloch state | `C_occ_k`, `C_occ_k_beta`, `eps_k_beta` size = `n_kpts` | — | dual-spin k blocks |

A full 6N FD force check is offline only, as for
[gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces). On that closed-shell
variant's off-symmetry geometry, max\(\lvert F_{\mathrm{analytic}} -
F_{\mathrm{fd}}\rvert \approx 6\times 10^{-6}\) Ha/Bohr.

### Performance

`perf_exempt` (no same-basis CP2K wall). \(T_{\mathrm{ATV}}\) **~1–2** s
(OMP=8).

### Limitations

- The absolute pin is ATV self-consistency. A CP2K re-pin needs a data path
  with DZV-GTH-PADE, or an ATV pin rebased on SZV.
- XL-BOMD / CPMD fixed-density UKS is out of scope.

## gpw-kpoint-uks-stress — Multi-k UKS analytic cell stress, high-spin H₂ / PADE {#gpw-kpoint-uks-stress}

*No section of the case runs this variant.*

### Method

This variant applies the fixed-density-matrix strain theorem of
[gpw-uks-stress](#gpw-uks-stress) on the multi-k path. After one multi-k
UKS SCF:

1. Collect the occupied Bloch MOs `C_occ_k` / `C_occ_k_beta` and band
   energies `eps_k` / `eps_k_beta` from `GpwResult`.
2. Rebuild \(P_k^\sigma=\sum_i^{\mathrm{occ}} c_{ik}^\sigma (c_{ik}^\sigma)^\dagger\)
   and \(W_k^\sigma=\sum_i^{\mathrm{occ}} \varepsilon_{ik}^\sigma\,
   c_{ik}^\sigma (c_{ik}^\sigma)^\dagger\), with occupancy 1 per spin.
3. Under \(\pm h\) diagonal strain at fixed fractional coordinates, rebuild
   \(S_k\), \(T_k\), \(V_{\mathrm{nl}}^k\), the grid, Poisson and spin XC,
   and evaluate the non-self-consistent
   \(E_L[\{P_k^\sigma\},\{W_k^\sigma\};\varepsilon]\) without re-converging
   the SCF.
4. Take \(\sigma_{aa} \approx -\Omega^{-1}[E_L(+h_a)-E_L(-h_a)]/(2h)\). The
   \(-\mathrm{Tr}(W S)\) channel supplies the Pulay term (the GPW analogue of
   plane-wave `PW::stress`).

```cpp
GpwDft method = make_uks(XcKind::PADE);   // High-spin H₂ system, Common setup
method.build(frame);
const GpwResult& er = method.run(frame);  // er.e_total [Ha]

const double h = 2e-3;
method.stress_strain_h = h;
const auto san = method.stress(frame);             // analytic, Voigt [eV/Å³]
const auto sfd = gpw_stress_fd(frame, method, h);  // multi-SCF oracle
// san[3..5] shear (0); compare in Ha/Bohr³
```

`GPWEvaluator` (`compute_stress`) gives the same numbers on a Frame path.
The recorded check called the method directly.

### Validation

The decisive gates are the internal dual gate (analytic vs multi-SCF FD)
and the closed-shell identity (k-UKS ≡ k-RKS). The soft energy pin matches
[gpw-kpoint-uks-forces](#gpw-kpoint-uks-forces). The CP2K input
`h2_uks_kpoint_stress.inp` (`STRESS_TENSOR NUMERICAL`, SZV-MOLOPT-GTH) is a
documentation twin.

| Quantity | Pin / measured | Tol | Notes |
| -------- | -------------: | --- | ----- |
| \(E_{\mathrm{tot}}\) [Ha] | **−0.528389371092** | \(\lvert\Delta E\rvert\le 1\times 10^{-3}\) | soft; 6 SCF iters; \(n_k=1\) TR-folded |
| \(\sigma_{xx}\) analytic [Ha/Bohr³] | **4.26434770×10⁻⁴** | dual gate | compressed bond → finite \(\sigma_{xx}\) |
| \(\sigma_{yy}\) analytic [Ha/Bohr³] | **−8.07875807×10⁻⁵** | dual gate | |
| \(\sigma_{zz}\) analytic [Ha/Bohr³] | **−8.05687961×10⁻⁵** | dual gate | close to \(\sigma_{yy}\) (cubic cell, bond \(\parallel x\)) |
| \(\max_a\lvert\sigma_a^{\mathrm{an}}-\sigma_a^{\mathrm{fd}}\rvert\) | **3.86×10⁻¹⁰** (\(\sim 3.9\times 10^{-10}\)) | \(\le 1\times 10^{-4}\) | decisive dual gate |
| Closed-shell k-UKS vs k-RKS \(\max\lvert\mathrm{d}\sigma\rvert\) | **2.35×10⁻¹⁰** | \(\le 5\times 10^{-4}\) | \(N_\alpha=N_\beta=1\) ≡ `n_occ=1` |
| Off-diagonal \(\sigma_{xy,xz,yz}\) | **exactly 0** | \(\le 1\times 10^{-8}\) | orthorhombic |
| \(\max_a\lvert\sigma_a\rvert\) floor | **> 1×10⁻⁸** | magnitude sanity | |

The dual-gate tolerance \(10^{-4}\) Ha/Bohr³ is generous against the
\(\sim 10^{-10}\) residual. It absorbs SCF and grid noise while still
requiring the correct order and sign.

### Performance

`perf_exempt`. The recorded wall is **~1.0** s (OMP=8, Release) for SCF +
analytic stress + multi-SCF FD + closed-shell identity. When a same-basis
twin exists, record the single-rank analytic-only wall against it (target
\(r\le 10\)).

### Limitations

- Orthorhombic diagonal only. The analytic path costs one SCF + 6 fixed-\(P\)
  samples, and the FD oracle 6+ re-converged SCFs.
- High-spin H₂ is a cheap pin, not a bulk magnetic solid. Larger k-meshes
  and production cutoffs are outside the budget.
- For multi-k DFT+U stress see
  [gpw-dftu-kpoint-stress](gpw-dft-plus-u.md#gpw-dftu-kpoint-stress).
  Γ-UKS PBE stress is [gpw-uks-stress](#gpw-uks-stress).

## gpw-kpoint-uks-stress-gga — Multi-k UKS analytic cell stress, H₂ / RevPBE·RPBE·BLYP {#gpw-kpoint-uks-stress-gga}

*No section of the case runs this variant.*

### Method

This variant uses the same fixed-\(P_k^\sigma\) strain theorem as
[gpw-kpoint-uks-stress](#gpw-kpoint-uks-stress). The non-self-consistent
Lagrangian rebuilds **spin-GGA XC** (\(\rho^\sigma\),
\(\lvert\nabla\rho^\sigma\rvert\) and the total-density gradient terms of
each functional) under strain. The functionals are **RevPBE**, **RPBE** and
**BLYP**, given as the constructor `xc`, on the High-spin H₂ system.
\(n_\mathrm{kpts}\) on the result may be the TR-folded count.

An **activation** check requires
\(\lvert E_{\mathrm{RevPBE}}-E_{\mathrm{PBE}}\rvert > 1\times 10^{-4}\) Ha
on the same geometry and mesh, so the variant cannot pass on a
PBE-mislabelled path.

```cpp
GpwDft energy = make_uks(XcKind::RevPBE);      // RPBE, BLYP likewise
energy.build(frame);
const auto& er = energy.run(frame);            // er.e_total [Ha]

const double h = 2e-3;
GpwDft m = make_uks(XcKind::RevPBE);
m.stress_strain_h = h;
m.build(frame);
const auto san = m.stress(frame);                                    // Voigt eV/Å³
const auto sfd = gpw_stress_fd(frame, make_uks(XcKind::RevPBE), h);  // Voigt eV/Å³
```

### Validation

| Quantity | Pin / measured | Tol | Notes |
| -------- | -------------: | --- | ----- |
| \(E\) (k-UKS PBE 1×1×2, 100 Ry) | **−0.566633887639 Ha** | activation baseline | same geometry |
| \(E\) (k-UKS **RevPBE**) | **−0.574211377589 Ha** | \|E − pin\| ≤ **1×10⁻³** Ha | soft ATV pin; 6 iters |
| \(E\) (k-UKS **RPBE**) | **−0.575044480519 Ha** | same | soft pin |
| \(E\) (k-UKS **BLYP**) | **−0.558884091671 Ha** | same | soft pin |
| \(\lvert E_{\mathrm{RevPBE}}-E_{\mathrm{PBE}}\rvert\) | **7.577×10⁻³ Ha** | ≥ **1×10⁻⁴** Ha | activation |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| RevPBE | **4.63×10⁻¹⁰** Ha/Bohr³ | ≤ **1×10⁻⁴** | decisive dual gate |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| RPBE | **5.60×10⁻¹⁰** Ha/Bohr³ | same | |
| max\|\(\sigma_{\mathrm{an}}-\sigma_{\mathrm{fd}}\)\| BLYP | **1.54×10⁻⁹** Ha/Bohr³ | same | |
| Off-diagonal | ~0 | ≤ **1×10⁻⁸** | orthorhombic |
| max\|\(\sigma_{\mathrm{diag}}\)\| floor | nonzero | \> **1×10⁻⁸** Ha/Bohr³ | no trivial zero stress |

RevPBE analytic stress: \(\sigma_{xx}\) **4.22391784×10⁻⁴**, \(\sigma_{yy}\)
**−7.94777145×10⁻⁵**, \(\sigma_{zz}\) **−7.92623369×10⁻⁵** Ha/Bohr³.

The CP2K NUMERICAL documentation twins use `RUN_TYPE ENERGY_FORCE`,
`STRESS_TENSOR NUMERICAL`, `LSD`, `MULTIPLICITY 3`, MP 1 1 2, with RevPBE as
`PBE` `PARAMETRIZATION REVPBE`, RPBE and BLYP, and SZV-MOLOPT-GTH /
GTH-PBE-q1 (not DZV-GTH-PADE). Output lands in e.g.
`/tmp/cp2k_oracle_h2_uks_kpoint_stress_revpbe.out`. The absolute same-basis
twin is pending.

### Performance

`perf_exempt`. \(T_{\mathrm{ATV}}\) **~2.7** s for the dual gate (OMP=8,
Release), diagnostic only.

### Limitations

- These are three pure GGAs, not a blanket claim for every GGA. Closed-form
  term tables without fixed-\(P_k\) samples are not used.
- Bulk solids, metals with smearing, DFT+U stress
  ([gpw-dftu-kpoint-stress](gpw-dft-plus-u.md#gpw-dftu-kpoint-stress)) and
  dispersion stress (D3/D4) are not covered.

## gpw-kpoints-uks — Multi-k UKS energy, H₂ identity suite + O₂ CP2K twin {#gpw-kpoints-uks}

*No section of the case runs this variant.*

### Method

With \(N_d > 1\) on any axis, or an explicit `k_points` list, the solver
takes the **complex Bloch** path:

- Per-\(\mathbf{k}\) dual spin densities \(P_k^\alpha\), \(P_k^\beta\) and
  Hamiltonians \(H_k^\sigma\).
- k-weights from the MP mesh, using the CP2K convention
  \(\kappa_d = (2r - N_d - 1)/(2 N_d)\). Even \(N\) is Γ-avoiding (shifted).
- Optional **time reversal** (`k_time_reversal = true`) folds
  \(\pm\mathbf{k}\) pairs, which is exact for collinear RKS/UKS without SOC.
  The retained partner gets doubled weight, so an even \(2\times2\times2\)
  mesh goes from 8 to **4** irreducible k.

For fixed fillings the magnetization is \(M = N_\alpha - N_\beta\). Multi-k
does not change fixed integer fillings. The identity suite uses PADE LSD and
the O₂ product twin uses PBE. This is the multi-k counterpart of restricted
multi-k GPW (`gpw-kpoints-si`).

| Identity suite | Value |
| -------------- | ----- |
| System | High-spin H₂ toy, H–H ±0.7 Bohr about the cell centre |
| Cell | \(8\times8\times8\) Bohr, periodic |
| Basis / PP | DZV-GTH-PADE / GTH-PADE-q1 |
| Spin | \(N_\alpha=2\), \(N_\beta=0\) (\(M=2\)) |
| XC | PADE LSD |
| Cutoff | 120 Ry |
| k mesh | MONKHORST-PACK \(2\times2\times2\), TR → 4 k |

```cpp
GpwDft mp{{H}, 120.0, XcKind::PADE};   // H₂ in an 8 Bohr cube
mp.n_alpha = 2;
mp.n_beta = 0;
mp.max_iter = 80;
mp.tol = 1e-7;
mp.k_mesh[0] = mp.k_mesh[1] = mp.k_mesh[2] = 2;
mp.k_time_reversal = true;      // 8 → 4 irreducible k
mp.build(frame);
const cpu::gpw::GpwResult& r = mp.run(frame);
// r.e_total [Ha], r.converged, r.n_kpts (== 4 under TR), r.eps_k, r.k_frac
```

The explicit Γ point on the k path (for the Γ-limit identity) uses
`k_points = {{0,0,0}}` and `k_point_weights = {1.0}`. The matching UKS for a
closed-shell RKS `n_occ` uses `n_alpha = n_beta = n_occ`.

### Validation

| Gate | Result | Tol | Notes |
| ---- | -----: | --- | ----- |
| A1 Γ limit (k = Γ complex vs real Γ-UKS) | \(\lvert\Delta\rvert \approx 4\times10^{-11}\) Ha | \(1\times10^{-6}\) Ha | same occupations / cell / cutoff |
| A2 closed-shell multi-k UKS ≡ multi-k RKS | \(\lvert\Delta\rvert \approx 0\) | \(5\times10^{-6}\) Ha | \(N_\alpha=N_\beta=1\) vs `n_occ=1` |
| A3 time reversal 8 k vs 4 k | \(\lvert\Delta\rvert \approx 0\) | \(5\times10^{-6}\) Ha | `n_kpts` 8 vs 4 |
| A4 k-sampling activation \(\lvert E_{\mathrm{MP}}-E_\Gamma\rvert\) | \(2.79\times10^{-2}\) Ha | \(\ge 1\times10^{-4}\) Ha | the mesh must change the energy |
| A5 \(E_{\mathrm{MP}}\) [Ha] | **−0.719059673672** (frozen ATV) | \(1\times10^{-3}\) Ha | MP \(2\times2\times2\) TR, 120 Ry |
| \(E_\Gamma\) [Ha] | −0.691163126674 | diagnostic | |
| Payload | `n_kpts==4`, `eps_k` / `k_frac` / weights | shapes | SCF `max_iter=80`, `tol=1e-7` |

The offline CP2K product twin (`o2_uks_kpoints.inp`, libcp2k 2026.1) is
documentation only. It is not an energy gate, because an ATV O₂ MOLOPT
multi-k run exceeds the 60 s budget.

| Setting | Value |
| ------- | ----- |
| System | O₂, \(R=1.2075\) Å, cubic **8 Å**, `PERIODIC XYZ`; O (4, 4, 3.39625), O (4, 4, 4.60375) |
| Basis / PP / XC | DZVP-MOLOPT-GTH / GTH-PBE-q6 / PBE `LSD`, \(N_\alpha=7\), \(N_\beta=5\) |
| Grid | CUTOFF 200 Ry, NGRIDS 1, REL_CUTOFF 40 |
| k mesh | MONKHORST-PACK \(2\times2\times2\) |
| SCF | diagonalization + Broyden α=0.3, `EPS_SCF=1e-6` |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.92842018097085** (SCF converged in **22** steps) |

```text
  *** SCF run converged in    22 steps ***
  Total energy:                                               -31.92842018097085
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.928420180970853
```

### Performance

`perf_exempt`: H₂ has no CP2K twin, and a fair twin would need an ATV O₂
MOLOPT multi-k run. \(T_{\mathrm{ATV}}\) for A1–A5 is **~4.4** s (OMP=8).
\(T_{\mathrm{CP2K}}\) for the O₂ twin is **≈ 175** s (1 rank × OMP=4, `CP2K`
timer MAXIMUM). Once an ATV O₂ wall is recorded against the same input,
replace the exemption with a measured \(r \le 10\).

### Limitations

- Fixed occupations only on this variant. For smearing see
  [gpw-uks-smearing](#gpw-uks-smearing).
- The k path requires a periodic cell, `local_pp_on_grid` and collinear
  RKS/UKS. OT, DFT+U, multigrid and nonperiodic MT throw at `build()`.
- Noncollinear spin, SOC and hybrid k-points are out of scope. Plane-wave
  NC-UPF multi-k UKS is
  [pw-uks-kmesh-fe](pw-collinear-spin.md#pw-uks-kmesh-fe).

## gpw-uks-blyp — Γ-point UKS, O₂ / BLYP {#gpw-uks-blyp}

*No section of the case runs this variant.*

### Method

This variant uses spin-resolved **BLYP**: B88 exchange + LYP correlation
(libxc `GGA_X_B88` + `GGA_C_LYP`, CP2K `XC_FUNCTIONAL BLYP`), a pure
semi-local GGA with `hybrid_exchange_fraction(BLYP) = 0`. The spin-GGA grid
energy is

\[
E_{\mathrm{xc}}
=
\sum_g w_g\,(\rho_{\alpha,g}+\rho_{\beta,g})\,\varepsilon_{\mathrm{xc}}
(\rho_\alpha,\rho_\beta,\sigma_{\alpha\alpha},\sigma_{\alpha\beta},\sigma_{\beta\beta}),
\]

with spectral \(\nabla\rho_\sigma\) and potentials

\[
v_\alpha = v_{\rho_\alpha} - \nabla\cdot\bigl(
  2 v_{\sigma_{\alpha\alpha}}\nabla\rho_\alpha
  + v_{\sigma_{\alpha\beta}}\nabla\rho_\beta\bigr)
\]

(and the \(\beta\) twin). The grid entry point is `xc_grid_blyp_spin` →
`xc_grid_gga_spin(..., XCType::BLYP)`. At \(\sigma=0\) the GGA reduces to its
LDA limit (unit-tested). At finite \(\sigma\), BLYP and PBE differ, and the
activation gate \(|E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha on the
same geometry relies on that. The code is the Common setup snippet with
`cpu::gpw::XcKind::BLYP`. The closed-shell twin is
[gpw-blyp-water](gpw-rks.md#gpw-blyp-water).

### Validation

Input `o2_uks_blyp.inp`, CP2K `EPS_SCF 1e-6`.

| Quantity | CP2K pin | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.90397685949710** | **1e-3 Ha** |
| \(\langle S^2\rangle\) (single det.) | **2.006656** | **0.15** |
| \(\lvert E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}\rvert\) | — | \(> 10^{-4}\) Ha |

Measured (OMP=4 Release): \(E_{\mathrm{BLYP}}\) **−31.903662478892** Ha,
\(\lvert\Delta E\rvert\) \(3.14\times 10^{-4}\) Ha, \(\langle S^2\rangle\)
2.00658851, \(E_{\mathrm{PBE}}\) twin −31.928854728464 Ha,
\(\lvert E_{\mathrm{BLYP}}-E_{\mathrm{PBE}}\rvert\) \(2.52\times 10^{-2}\) Ha,
60 SCF iterations.

```text
*** SCF run converged in    19 steps ***
Total energy:                                               -31.90397685949710
Ideal and single determinant S**2 :                    2.000000       2.006656
ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.903976859497096
```

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| O₂ UKS-BLYP, Γ (ATV single grid, 60 SCF iters) | **31.7** | **78.6** | **2.48** | ✓ |

The ATV SCF count (**60** iterations, against CP2K **19** at `EPS_SCF 1e-6`)
dominates the ratio.

### Limitations

- Γ only. No hybrid B3LYP or exact exchange on the GPW path.

## gpw-uks-revpbe — Γ-point UKS, O₂ / revPBE {#gpw-uks-revpbe}

*No section of the case runs this variant.*

### Method

This variant uses spin-resolved **revPBE** (`XcKind::RevPBE`,
`xc_grid_revpbe_spin`, shared pointwise engine `xc_functional.h`
`XCType::RevPBE`). It keeps the analytic PBE form with a revised exchange
enhancement \(\kappa = 1.245\) (PBE: \(\kappa = 0.804\)) and standard PBE
correlation. At \(\sigma = |\nabla\rho|^2 = 0\) revPBE and PBE share the PW92
LDA base, and at finite gradient they differ (libxc `GGA_X_PBE_R` +
`GGA_C_PBE`). The CP2K 2026.1 input form is nested:

```text
&XC_FUNCTIONAL
  &PBE
    PARAMETRIZATION revPBE
  &END PBE
&END XC_FUNCTIONAL
```

Short `XC_FUNCTIONAL revPBE` is not a valid enumeration in CP2K 2026.1. The
closed-shell pin [gpw-revpbe-si](gpw-rks.md#gpw-revpbe-si) may use a
different stock name depending on version, so always re-pin from the twin
input. RPBE is a different exchange form (see
[gpw-uks-rpbe](#gpw-uks-rpbe)): do not interchange `XcKind::RevPBE` and
`XcKind::RPBE`. The code is the Common setup snippet with
`cpu::gpw::XcKind::RevPBE`, plus an identical-geometry UKS-PBE run for the
activation gate \(|E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\) Ha.

### Validation

Input `o2_uks_revpbe.inp`.

| Quantity | CP2K pin | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.96512377015027** | **1e-3 Ha** |
| \(\langle S^2\rangle\) (single det.) | **2.005611** | **0.15** |
| \(\lvert E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}\rvert\) | — | \(> 10^{-4}\) Ha |

Measured (OMP=4 Release): \(E_{\mathrm{revPBE}}\) **−31.965167155386 Ha**,
\(\lvert\Delta E\rvert\) \(4.34 \times 10^{-5}\) Ha, \(\langle S^2\rangle\)
2.00561329, \(E_{\mathrm{PBE}}\) twin −31.928854728432 Ha,
\(\lvert E_{\mathrm{revPBE}}-E_{\mathrm{PBE}}\rvert\) \(3.63 \times 10^{-2}\)
Ha, 59 SCF iterations.

```text
  *** SCF run converged in    15 steps ***
  Total energy:                                               -31.96512377015027
  Ideal and single determinant S**2 :                    2.000000       2.005611
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.965123770150271
 FUNCTIONAL| revPBE, Yingkai Zhang and Weitao Yang, … {spin polarized}
```

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| O₂ UKS-revPBE, Γ (revPBE SCF only) | **25.8** | **83.1** | **3.22** | ✓ |

The extra UKS-PBE activation run is not folded into \(r\).

### Limitations

- Γ only. Do not document RPBE energies against this pin.

## gpw-uks-rpbe — Γ-point UKS, O₂ / RPBE {#gpw-uks-rpbe}

*No section of the case runs this variant.*

### Method

This variant uses spin-resolved **RPBE** (Hammer–Hansen–Nørskov exchange
with PBE correlation). The exchange enhancement is the exponential form

\[
F_x(s) = 1 + \kappa\bigl(1 - e^{-\mu s^2/\kappa}\bigr)
\]

with the **same** PBE parameters \(\kappa = 0.804\), \(\mu \approx 0.21951\),
plus PBE correlation (PW92 LDA base + PBE gradient correction). This is not
revPBE, which revises \(\kappa\) in the *rational* PBE \(F_x\) (see
[gpw-uks-revpbe](#gpw-uks-revpbe) and
[gpw-revpbe-si](gpw-rks.md#gpw-revpbe-si)). The libxc / CP2K names are
`GGA_X_RPBE` + `GGA_C_PBE`. In CP2K **2026.1** the short tokens
`XC_FUNCTIONAL RPBE` and `PARAMETRIZATION RPBE` are not valid enumerations,
so the twin input uses nested `&GGA_X_RPBE` / `&GGA_C_PBE` subsections.

Atomiverse routes RPBE through the shared GGA grid path used for PBE,
PBEsol, revPBE and BLYP (`xc_grid_rpbe_spin`, `cpu::scf::XCType::RPBE`). The
code is the Common setup snippet with `cpu::gpw::XcKind::RPBE`, plus a PBE
run for the activation gate \(|E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}| > 10^{-4}\)
Ha. The closed-shell twin is [gpw-rpbe-water](gpw-rks.md#gpw-rpbe-water).

### Validation

Input `o2_uks_rpbe.inp`.

| Quantity | CP2K pin | Tol |
| -------- | -------: | --- |
| \(E_{\mathrm{tot}}\) [Ha] | **−31.97338355451434** | **1e-3 Ha** |
| \(\langle S^2\rangle\) (single det.) | **2.005630** | **0.15** |
| \(\lvert E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}\rvert\) | — | \(> 10^{-4}\) Ha |

Measured (OMP=4 Release): \(E_{\mathrm{RPBE}}\) **−31.973422072918 Ha**,
\(\lvert\Delta E\rvert\) \(3.85\times 10^{-5}\) Ha (\(\sim 4\times 10^{-5}\)),
\(\langle S^2\rangle\) 2.00563766, \(E_{\mathrm{PBE}}\) twin
−31.928854728710 Ha, \(\lvert E_{\mathrm{RPBE}}-E_{\mathrm{PBE}}\rvert\)
\(4.46\times 10^{-2}\) Ha, 59 SCF iterations.

```text
  *** SCF run converged in    15 steps ***
  Total energy:                                               -31.97338355451434
  Ideal and single determinant S**2 :                    2.000000       2.005630
 ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]            -31.973383554514335
 FUNCTIONAL| GGA_X_RPBE: … {scale=1.000, spin-polarized}
 FUNCTIONAL| GGA_C_PBE: … {scale=1.000, spin-polarized}
```

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| O₂ UKS-RPBE, Γ (59 SCF iters) | **30.4** | **82.0** | **2.70** | ✓ |

### Limitations

- Γ only. Multi-k UKS uses the same `XcKind` allow-list but has no RPBE pin.
- UGW on UKS-RPBE densities is not available: `GpwDft::g0w0` refuses UKS
  RPBE.

## gpw-uks-smearing — Multi-k UKS with Fermi–Dirac smearing, BCC Na {#gpw-uks-smearing}

*No section of the case runs this variant.*

### Method

Metals need **fractional band occupations** and a shared chemical potential
\(\mu\) that conserves the valence electron count. On `GpwDft` this uses
`n_alpha`/`n_beta` as the UKS seed, `smearing = Smearing::fermi_dirac` (or
Methfessel–Paxton with `mp_order`), `degauss` [Ry], `extra_bands` and a
multi-k mesh. Spin-orbital occupations \(f_{\sigma kb}\in[0,1]\) reuse
`cpu::pw::internal` with `spin_degen = 1`, and **one shared** \(\mu\) spans
\(\alpha\) and \(\beta\). Under smearing, `n_alpha`/`n_beta` only seed the
occupations, so the magnetization is free. With
\(x = (\varepsilon-\mu)/\sigma\) and \(\sigma =\) `degauss` (converted to
Hartree inside the SCF):

| Kind | \(f(x)\) | Public enum |
| ---- | -------- | ----------- |
| Fixed integer | step fill of `n_alpha`/`n_beta` | `Smearing::none` (default) |
| Fermi–Dirac | \(1/(1+e^{x})\) | `Smearing::fermi_dirac` |
| Methfessel–Paxton order \(N\) | MP series of order `mp_order` | `Smearing::methfessel_paxton` |

The Fermi search fixes

\[
N_{\mathrm{occ}}
  = \sum_{\sigma,k,b} w_k\, f_{\sigma kb}(\varepsilon_{\sigma kb},\mu,\sigma)
  = Z_{\mathrm{val}}\cdot N_{\mathrm{atoms}}
\]

(BCC Na with GTH-PBE-q9: \(N_e = 2\times 9 = 18\)). `extra_bands` adds
empty states above \(\max(n_\alpha,n_\beta)\), for parity with CP2K
`ADDED_MOS`. GPW still diagonalizes the full AO spectrum. With the FD entropy
\(s = -[f\ln f+(1-f)\ln(1-f)]\) per spin-orbital, the SCF total is the Mermin
free energy

\[
F = E_{\mathrm{KS}} - TS,\qquad
E_{\mathrm{entropy}} = -TS = -\sigma\sum_{\sigma,k,b} w_k\, s_{\sigma kb}.
\]

`GpwResult.e_total` [Ha] is \(F\), `e_entropy` is \(-TS\), `e_fermi` is
\(\mu\) [Ha], and `n_elec_occ` is \(N_{\mathrm{occ}}\).

**Identities.** On multi-k H₂ (\(1\times1\times2\), both \(N_e=2\)), UKS
\(n_\alpha=n_\beta=1\) + FD matches RKS \(n_{\mathrm{occ}}=1\) + FD to
\(\lvert\Delta F\rvert \lesssim 10^{-15}\) Ha on the same binary (gate
\(\le 5\times 10^{-6}\) Ha). BCC Na is non-magnetic: a ferro seed collapses
under smearing to the RKS FD free energy
([gpw-metallic-smearing](gpw-rks.md#gpw-metallic-smearing)) to
\(\sim 4\times 10^{-13}\) Ha.

```cpp
using cpu::gpw::Smearing;

// BCC conventional cubic: 2 Na at (0,0,0) and (½,½,½); a in Å
constexpr double a_ang = 4.225;
cpu::Frame frame(2);
const double frac[2][3] = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_atomic_number)[i] = 11;
    frame.span(kAtom_x)[i] = frac[i][0] * a_ang;
    frame.span(kAtom_y)[i] = frac[i][1] * a_ang;
    frame.span(kAtom_z)[i] = frac[i][2] * a_ang;
}
frame.box = cpu::Box::cubic(a_ang);

GpwDft m{{cpu::gpw::Kind{
    molrs::Element::Na,
    cpu::gpw::GaussianBasis(molrs::Element::Na, "DZVP-MOLOPT-SR-GTH"),
    cpu::gpw::GthPotential(molrs::Element::Na, "GTH-PBE-q9")}}, 150.0, XcKind::PADE};
m.n_alpha = 10;  // ferro seed; M free under smearing
m.n_beta = 8;
m.max_iter = 80;
m.tol = 1e-6;
m.eps_default = 1e-7;
m.diis_depth = 6;
m.mix = 0.4;
m.k_mesh[0] = m.k_mesh[1] = m.k_mesh[2] = 2;
m.k_time_reversal = true;
m.smearing = Smearing::fermi_dirac;
m.degauss = 0.02;   // Ry
m.extra_bands = 4;  // CP2K ADDED_MOS parity

m.build(frame);
const cpu::gpw::GpwResult& r = m.run(frame);  // frame.meta[kEnergy] = F [eV]
// r.e_total [Ha] F; r.e_fermi [Ha]; r.n_elec_occ ≈ 18; r.e_entropy [Ha] (−TS)
```

The section [`gpw-uks-metallic-stress`](https://github.com/MolCrafts/Atomiverse/blob/dev/regressions/gpw/uks/REFERENCE.md)
runs the same BCC Na settings with a 9/9 seed and gates the same free-energy
value.

### Validation

| Quantity | Pin | Tol | Notes |
| -------- | --: | --- | ----- |
| Free energy \(F\) | **−92.126330608004 Ha** | **\(10^{-4}\) Ha** | frozen ATV pin |
| \(N_{\mathrm{occ}}=\sum_{\sigma k b} w_k f\) | **18.0** | \(10^{-5}\) | valence count |
| \(E_F\) | **0.055825 Ha** | \(\lvert E_F\rvert < 10^3\) Ha | finite, not NaN |
| Entropy term \(-\sigma S\) | **−2.26×10⁻⁴ Ha** | finite | diagnostic |
| RKS Na FD identity | \(\lvert F_{\mathrm{UKS}}-F_{\mathrm{RKS}}\rvert\sim 4\times 10^{-13}\) Ha | — | non-magnetic collapse |
| H₂ closed-shell identity | \(\lvert F_{\mathrm{UKS}}-F_{\mathrm{RKS}}\rvert\) | \(\le 5\times 10^{-6}\) Ha | multi-k \(1\times1\times2\) |

The absolute free energy is an ATV-frozen value. ATV's single grid and
`EPS_DEFAULT` differ from CP2K's multigrid (`NGRIDS 4` / REL_CUTOFF) by
about 0.6 Ha. The CP2K twin documents the algorithm only: Quickstep **LSD**
+ `&SMEAR METHOD FERMI_DIRAC` + `ADDED_MOS`, PADE,
GTH-PBE-q9 + DZVP-MOLOPT-SR-GTH, \(T_e = 3157.75\) K (\(\approx 0.02\) Ry
\(k_B T\)), MP \(2\times2\times2\), CUTOFF / REL_CUTOFF 150 / 40 Ry, NGRIDS 4,
MULTIPLICITY 3 (the ferro seed collapses under smearing).

### Performance

| Twin | \(T_{\mathrm{CP2K}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× |
| ---- | ------------------------: | ----------------------: | ----: | ---- |
| Na FD twin class (`na_bcc_fermi_dirac.inp`, RKS, OMP=4) vs ATV Na UKS smearing (OMP=4, 16 SCF iters) | **89.51** (≈ 89.5) | **≈ 1.56** | **≈ 0.017** | ✓ |

The ratio compares the same system/basis/PP class, not bit-identical
Hamiltonian throughput.

### Limitations

- For a permanently magnetic metallic pin see plane-wave bulk Fe
  ([pw-uks-fe-smearing](pw-smearing.md#pw-uks-fe-smearing)). For closed-shell
  plane-wave smearing see [pw-smearing-al](pw-smearing.md#pw-smearing-al).
- OT + smearing throws at `build()`: only diagonalization SCF is supported.
- Fermi–Dirac is the gated smearing. Methfessel–Paxton (`mp_order`) is in the
  API and shared with RKS. Cold / Marzari–Vanderbilt / Gaussian smearing are
  not public on GPW.
- No force acceptance pin under UKS smearing. Stress is the
  `gpw-uks-metallic-stress` section.
