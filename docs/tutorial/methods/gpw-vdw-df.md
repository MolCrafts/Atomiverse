# GPW nonlocal vdW correlation (vdW-DF, vdW-DF2, rVV10, optB88-vdW)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Density-based nonlocal correlation evaluated with the Román–Pérez–Soler (RPS)
FFT method on the GPW uniform grid: vdW-DF (DRSLL / DF1), vdW-DF2
(LMKLL / DF2) and revised VV10 (rVV10), each additive on a semi-local host
(PBE; optB88-vdW pairs optimized-B88 exchange + PW92 correlation with DRSLL).
The nonlocal potential \(V_{\mathrm{nl}}=\delta E_c^{\mathrm{nl}}/\delta\rho\)
is added to the KS effective potential at every SCF step, the final
\(E_c^{\mathrm{nl}}\) is evaluated on the converged total density, and analytic
forces carry \(V_{\mathrm{nl}}\) in their veff term.

The method is the concrete `cpu::gpw::GpwDft` with one of the bool members
`vdw_df`, `vdw_df2` or `rvv10` (optB88-vdW: `XcKind::OptB88VdW` as the `xc`
constructor argument plus `vdw_df = true`). `run(frame)` reports
\(E_c^{\mathrm{nl}}\) on `GpwResult::e_disp` [Hartree] and folds it into
`e_total`; `forces(frame)` runs the SCF plus analytic forces. Driver work binds
the configured `GpwDft` through `cpu::gpw::GPWEvaluator`. Nonlocal flags are
periodic-only and cover closed-shell RKS (Γ and Monkhorst–Pack k, single grid
or REL_CUTOFF multigrid) and collinear UKS (Γ and multi-k; both spins receive
\(V_{\mathrm{nl}}\) of \(\rho_\alpha+\rho_\beta\)). The sections below are
CPU-only: DRSLL and rVV10 energies on the water-dimer cell, the self-consistent
rVV10 energy + force check, and DRSLL on the multigrid. Three further variants
(multi-k UKS optB88-vdW, Γ optB88-vdW, vdW-DF2) carry recorded values but no
running section. The case also carries `gpw-vdw-stress`, `gpw-vdw-uks`,
`gpw-vdw-kpoints` and `gpw-vdw-kpoint-forces`, whose oracle record lives in the
case `REFERENCE.md`.

Case: `ctest -R '^gpw/vdw-df$'` — sources and oracle record in [`regressions/gpw/vdw-df/`](../../../regressions/gpw/vdw-df/). One section of the case per `##` heading below: `atv_regressions --case gpw/vdw-df --section <name>`.

## Common setup

### Functionals

| Flag | CP2K `NON_LOCAL TYPE` | libvdwxc | Formula family | Parameters |
| ---- | --------------------- | -------- | -------------- | ---------- |
| `vdw_df` | DRSLL (vdW-DF1) | `VDWXC_DF1` | Dion \(E_c^{\mathrm{nl}}\) | \(Z_{ab}=-0.8491\) |
| `vdw_df2` | LMKLL (vdW-DF2) | `VDWXC_DF2` | same kernel | \(Z_{ab}=-1.887\) |
| `rvv10` | RVV10 | — | Sabatini revised VV10 | \(b=6.3\), \(C=0.0093\) |

**vdW-DF / DF2** (Dion et al. 2004; Lee et al. 2010):

\[
E_c^{\mathrm{nl}}
  = \tfrac12 \iint \rho(\mathbf r)\,\phi(\mathbf r,\mathbf r')\,\rho(\mathbf r')
    \,\mathrm d\mathbf r\,\mathrm d\mathbf r',
\]

with \(q_0(\rho,|\nabla\rho|^2)\) and the community RPS kernel tables shared by
DF1 and DF2 (libvdwxc 0.5.0 / Quantum ESPRESSO; in-tree float32 tables in
`vdw_df_kernel_data`). The variants differ only in the gradient-correction
prefactor \(Z_{ab}\); soft cutoff \(q_{\mathrm{cut}}\) = 5.0 Bohr⁻¹.

**rVV10** (Sabatini et al. 2013):

\[
E_c^{\mathrm{nl}}
  = \beta\int\rho\,\mathrm d\mathbf r
  + \tfrac12\iint \theta(\mathbf r)\,\phi(\mathbf r,\mathbf r')\,\theta(\mathbf r')\,
  \mathrm d\mathbf r\,\mathrm d\mathbf r'.
\]

Local \(q_0 = w_0/\kappa\) is saturated to the community q-mesh;
\(\theta_\alpha = c\,\rho^{3/4} p_\alpha(q_0)\); \(\phi_{\alpha\beta}(k)\) comes
from the rVV10 kernel table (CP2K `rVV10_kernel_table.dat`, nqs = 20,
nr = 1024, \(r_{\max}\) = 100 Bohr), embedded as float32+zlib
(`rvv10_kernel_data.cpp`) — no runtime file I/O. Defaults (CP2K
`PARAMETERS 6.3 0.0093`; ATV `kRvv10B` / `kRvv10C`):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(b\) | 6.3 | dimensionless (range separation) |
| \(C\) | 0.0093 | a.u. (gradient coefficient) |
| \(q_{\mathrm{cut}}\) | 0.5 | 1/Bohr (soft upper q) |
| \(q_{\min}\) | \(1\times 10^{-4}\) | 1/Bohr (soft lower q) |

Stock (unrevised) VV10 (Vydrov & Van Voorhis 2010) is not implemented; CP2K
2026.1 has no `TYPE VV10` either.

### Self-consistent evaluation

Following CP2K `qs_dispersion_nonloc` (Soler eq. 10 + spectral divergence of
the gradient flux), with \(|\nabla\rho|^2\) taken spectrally from the GPW FFT
gradient:

1. **Each SCF iteration:** evaluate RPS \(E_c^{\mathrm{nl}}\) and
   \(V_{\mathrm{nl}}\); add \(V_{\mathrm{nl}}\) onto the KS \(V_{\mathrm{eff}}\)
   (UKS: both spins share the total-\(\rho\) potential).
2. **After convergence:** recompute \(E_{\mathrm{disp}}=E_c^{\mathrm{nl}}\) on
   the SC total density → `GpwResult::e_disp`; fold into `e_total`.
3. **Forces:** rebuild \(V_{\mathrm{nl}}\) on the SC density and include it in
   the veff piece of `assemble_gpw_forces` (`GpwDft::forces`).

RPS costs \(O(N_g \log N_g \times n_\alpha)\) with \(n_\alpha = 20\) spline
channels per evaluation, i.e. \(O(N_{\mathrm{SCF}}\times n_\alpha)\) FFT stacks
per self-consistent solve.

CP2K twin surface (additive on the chosen XC, e.g. `&XC_FUNCTIONAL PBE`):

```text
&VDW_POTENTIAL
  POTENTIAL_TYPE NON_LOCAL
  &NON_LOCAL
    TYPE DRSLL          ! or LMKLL / RVV10 (+ PARAMETERS 6.3 0.0093)
  &END NON_LOCAL
&END VDW_POTENTIAL
```

**Sign / interpretation.** Pure \(E_c^{\mathrm{nl}}\) on a molecular density
can be **positive**; dispersion *binding* is recovered in interaction-energy
differences. Pairwise D3 / TS / MBD `e_disp` values and the different nonlocal
forms are not comparable one-to-one in magnitude.

### System

Every section uses the periodic H-bonded water dimer of
[`gpw-pbe-d3bj-ice`](gpw-d3.md#gpw-pbe-d3bj-ice) / `gpw-ts-ice` /
[`gpw-mbd-ice`](gpw-mbd.md#gpw-mbd-ice) (not bulk ice Ih):

| Item | Value |
| ---- | ----- |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| Grid | CUTOFF **200 Ry** (single grid unless stated) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| SCF | mix 0.4, tol \(10^{-7}\), max_iter 40 (unless stated) |

```cpp
frame.box = ATV::cpu::Box::cubic(12.0);  // Å; kinds = {O, H}
GpwDft gpw{kinds, 200.0, XcKind::PBE};
gpw.n_occ = 8;
gpw.max_iter = 40;  gpw.tol = 1e-7;  gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.vdw_df = true;     // or vdw_df2 / rvv10 — at most one dispersion flag
gpw.build(frame);
const auto& res = gpw.run(frame);   // res.e_disp [Ha] within 5e-5 of the pin; in e_total
```

Driver path: `ATV::cpu::gpw::GPWEvaluator eval(std::move(dft));` bound on
Evaluate / MD / Minimize; `eval.last_result().e_disp` [Hartree], while the
public Frame energy stays eV on `frame.meta.get(kEnergy)`.

### Oracle

- **Kernels:** offline **libvdwxc 0.5.0** (`VDWXC_DF1` / `VDWXC_DF2`) — the RPS
  engine family CP2K links for `TYPE DRSLL` / `LMKLL`. Kernel pins on a 16³,
  L = 8 Bohr Gaussian blob are recorded in the case `REFERENCE.md` (no unit
  test in the tree gates them) at a 1e-6 Ha tolerance (float32 tables ⇒ ~1e−7
  kernel noise).
- **Full cell:** offline **CP2K 2026.1** (Arrhenius tree install
  `work/cp2k/install`, GCC 11.5 + libxc + FFTW3) where a twin exists (rVV10,
  multigrid DRSLL, multi-k UKS optB88-vdW); otherwise a dual-frozen ATV value
  after the kernel pin.
- External offline pins only — never a golden of Atomiverse output, never live
  CP2K / libvdwxc in ctest. Build with the standard CPU environment
  (`buildenv-gcc/2026.03-mpich`) and `-DATV_BUILD_GPW=ON
  -DATV_BUILD_REGRESSIONS=ON`.

Offline kernel re-pin (human only):

```text
# 16^3 cubic L=8 Bohr, rho = exp(-0.5 r^2) blob at center, analytic sigma
E_nl DRSLL/DF1 = 1.2367739353586624e-01 Ha
```

**Performance.** On the 6-atom cell wall is dominated by the electronic SCF,
not the nonlocal kernel; the energy sections are **perf_exempt** (primary
deliverable is the \(E_{\mathrm{disp}}\) pin, same policy as `gpw-ts-ice` /
`gpw-mbd-ice` / `gpw-pbe-d3bj-ice`). The multigrid section and the multi-k UKS
optB88-vdW twin carry fair wall ratios.

### Shared limitations

- **Water dimer, not bulk ice Ih** — enough for periodic RPS / density wiring;
  no phonons.
- **Absolute \(e_{\mathrm{tot}}\)** versus CP2K QS is not asserted at micro-Ha
  (OT/grid/DIIS); where gated it is a loose integrity bound.
- **Mutual exclusion:** at most one of `d3` / `d3bj` / `d4` / `ts` / `xdm` /
  `mbd` / `vdw_df` / `vdw_df2` / `rvv10` may be true (throws at run).
- **Periodic only** — `nonperiodic` (MT) is rejected with nonlocal flags.
- Nonlocal vdW analytic **stress** is not covered by these sections; no CUDA
  GPW.

### References

- M. Dion, H. Rydberg, E. Schröder, D. C. Langreth, B. I. Lundqvist,
  *Phys. Rev. Lett.* **92**, 246401 (2004) — vdW-DF / DRSLL
- G. Román-Pérez, J. M. Soler, *Phys. Rev. Lett.* **103**, 096102 (2009) —
  RPS FFT evaluation
- K. Lee, É. D. Murray, L. Kong, B. I. Lundqvist, D. C. Langreth,
  *Phys. Rev. B* **82**, 081101(R) (2010) — vdW-DF2 / LMKLL
- R. Sabatini, T. Gorni, S. de Gironcoli, *Phys. Rev. B* **87**, 041108(R)
  (2013) — rVV10
- O. A. Vydrov, T. Van Voorhis, *J. Chem. Phys.* **133**, 244103 (2010) —
  original VV10 (not implemented)
- J. Klimeš, D. R. Bowler, A. Michaelides, *J. Phys.: Condens. Matter* **22**,
  022201 (2010) — optB88-vdW
- libvdwxc 0.5.0; CP2K Quickstep nonlocal vdW (`qs_dispersion_nonloc`)
- Narrative: [Ch. 9 GPW](../ch09-gpw.md); roadmap
  [`docs/roadmap/dft.md`](../../roadmap/dft.md) §8 Dispersion; pairwise
  siblings [gpw-pbe-d3bj-ice](gpw-d3.md#gpw-pbe-d3bj-ice),
  [gpw-d4-water](gpw-d4.md#gpw-d4-water)

## gpw-vdw-df-ice — PBE + vdW-DF (DRSLL) energy {#gpw-vdw-df-ice}

Γ-RKS PBE + `vdw_df` on the shared water dimer (CUTOFF 200 Ry, single grid);
gates \(E_c^{\mathrm{nl}}\) on `e_disp` against a dual-frozen ATV value taken
after the libvdwxc kernel pin. The pin was frozen from \(E_c^{\mathrm{nl}}\) on
the converged PBE density without \(V_{\mathrm{nl}}\) feedback (post-SCF); the
\(5\times 10^{-5}\) Ha tolerance covers that density policy, grid
discretization and the float32 tables (a looser product tol, 5×10⁻⁵ Ha, than
the 1e-6 Ha kernel pin). Matching \(q_0\), \(Z_{ab}\) and RPS
tables track libvdwxc DF1 / CP2K DRSLL on the same \((\rho,\sigma)\).

| Quantity | Atomiverse / pin | Oracle (offline) | Tol | Notes |
| -------- | ---------------: | ----------------: | --- | ----- |
| \(E_{\mathrm{nl}}\) synthetic blob [Ha] | `vdw_df_drsll` (recorded, not gated) | libvdwxc 0.5.0 **+0.12367739353586624** (\(1.2367739353586624\times 10^{-1}\)) | \(1\times 10^{-6}\) Ha | Kernel pin |
| \(E_{\mathrm{disp}}\) GPW PBE+DRSLL [Ha] | **+0.15194390480303990** (\(+1.5194390480303990\times 10^{-1}\)) | dual-frozen pin | \(5\times 10^{-5}\) Ha | Hard gate |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | (no CP2K QS wall on the pin host) | — | Not absolute-parity gated |

Performance: no full CP2K QS twin wall on the pin host (re-measurable on a host
with libvdwxc-linked CP2K); a kernel-only microbenchmark versus libvdwxc is not
the gate.

## gpw-rvv10-ice — PBE + rVV10 energy {#gpw-rvv10-ice}

Γ-RKS PBE + `rvv10` on the shared water dimer; gates \(E_c^{\mathrm{nl}}\) on
`e_disp` against a dual-frozen ATV value, with offline CP2K 2026.1 `TYPE RVV10`
(self-consistent density) as the sign-and-scale cross-check. The pin was
frozen from \(E_c^{\mathrm{nl}}\) on the converged PBE density (post-SCF); the
self-consistent value of [`gpw-vdw-scf-potential`](#gpw-vdw-scf-potential),
\(+6.11479080989958\times10^{-2}\) Ha, lies inside the tolerance.

| Item | Value |
| ---- | ----- |
| XC + nl (CP2K) | PBE + `TYPE RVV10`, PARAMETERS **6.3 0.0093**, SCALE 1.0 |
| CUTOFF / REL_CUTOFF | 200 / 40 Ry (CP2K twin) |
| CP2K pin line | `Dispersion energy = 0.06116467960721` Ha (SC) |
| ATV dual-frozen pin | \(E_c^{\mathrm{nl}} = +6.11534922097146\times 10^{-2}\) Ha |

| Quantity | Atomiverse / pin | CP2K (offline) | Tol | Notes |
| -------- | ---------------: | -------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | **+6.11534922097146×10⁻²** | **+6.116467960721×10⁻²** | \(5\times 10^{-5}\) Ha | Hard gate on the ATV pin |
| \(\Delta\) ATV − CP2K [Ha] | −1.1×10⁻⁵ | — | — | Density policy + float32 kernel |
| \(E_{\mathrm{tot}}\) QS [Ha] | (not gated) | −34.27712583714492 | — | Context only; OT/grid/DIIS |
| Sign | \(E_{\mathrm{disp}} > 0\) | \(E_{\mathrm{disp}} > 0\) | — | Asserted in the section |

The tolerance is wider than pairwise D3 nano-Hartree gates because the kernel
tables are float32 and CP2K's density is not reproduced; agreement to
~1.1×10⁻⁵ Ha confirms sign and community-kernel scale.

Performance: CP2K full self-consistent QS wall ≈ 121 s (OMP=8, pin host),
not fair-twinned against the short ATV SCF.

## gpw-vdw-scf-potential — self-consistent rVV10 energy + forces {#gpw-vdw-scf-potential}

Same settings as [`gpw-rvv10-ice`](#gpw-rvv10-ice), but through
`GpwDft::forces`: the self-consistent \(E_{\mathrm{disp}}\) is pinned to CP2K
2026.1 `TYPE RVV10` (self-consistent by default, `RUN_TYPE ENERGY_FORCE`), and
the analytic \(F_{O0,x}\) (H-bond axis, nonlocal veff included) is checked
against a central finite difference of \(E_{\mathrm{total}}\) with
\(h=5\times10^{-4}\) Bohr (three SCFs: equilibrium + two displaced). The SC
machinery is shared by DRSLL, LMKLL and rVV10; only the rVV10 twin is gated
here.

```cpp
GpwDft gpw{kinds, 200.0, XcKind::PBE};   // shared water-dimer settings
gpw.n_occ = 8;
gpw.max_iter = 40;  gpw.tol = 1e-7;  gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.rvv10 = true;
gpw.build(frame);
const GpwResult res = gpw.forces(frame);  // one SCF + analytic forces
// res.e_disp within 1e-4 of the CP2K SC pin
// frame kAtom_fx [eV/Å] ÷ units::kHartreePerBohrInEvPerAngstrom → Ha/Bohr; |ΔF| ≤ 5e-4
```

| Item | Value |
| ---- | ----- |
| CP2K grid | CUTOFF **200** Ry, REL_CUTOFF 40, NGRIDS 4 (multigrid; ATV single grid) |
| CP2K run | `RUN_TYPE ENERGY_FORCE`, OT + OUTER_SCF, `EPS_SCF 1e-7` |
| Pin line | `Dispersion energy = 0.06116467960721 Ha` |

| Quantity | Atomiverse (product) | CP2K (offline SC) | Tol | Notes |
| -------- | -------------------: | ----------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) SC [Ha] | \(+6.11479080989958\times10^{-2}\) | \(+6.116467960721\times10^{-2}\) | \(1\times10^{-4}\) Ha | Hard gate; \(\Delta\approx -1.7\times10^{-5}\) Ha (single-grid vs CP2K multigrid) |
| \(E_{\mathrm{disp}}\) post-SCF ATV [Ha] | \(+6.115349220971\times10^{-2}\) (`gpw-rvv10-ice`) | (same CP2K SC pin) | \(5\times10^{-5}\) Ha | Context: post-SCF vs SC density |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | −34.27712583714492 | — | Context only |
| \(F_{O0,x}\) analytic vs FD [Ha/Bohr] | \(\|F_{\mathrm{ana}}-F_{\mathrm{FD}}\|\approx 1\times10^{-6}\) | — (ATV FD of \(E_{\mathrm{total}}\)) | \(5\times10^{-4}\) | Non-trivial \(\|F_{\mathrm{ana}}\|>10^{-5}\) |

The residual against CP2K is dominated by single-grid vs multigrid density
(secondary OT/DIIS differences), not the kernel. The force gate is an internal
analytic-vs-FD consistency check on the SC energy surface, not a CP2K
force-vector pin.

Offline re-pin (human only):

```bash
# Host with CP2K 2026.1 + rVV10_kernel_table.dat on BASIS/potential path
# cp2k.psmp water_dimer_rvv10_force.inp
# → Dispersion energy = 0.06116467960721 Ha
# → Total energy      = -34.27712583714492 Ha
# wall ≈ 121 s (OMP=8) on pin host
```

Performance: **perf_exempt** — CP2K full QS `ENERGY_FORCE` ≈ **121 s**
(OMP=8, multigrid OT) is not fair-twinned against the ATV single-grid DIIS/mix
gate (3× SCF + forces).

Not gated here: multi-k / UKS self-consistent \(V_{\mathrm{nl}}\) forces (UKS
force assembly shares \(V_{\mathrm{nl}}\)).

## gpw-vdw-multigrid — DRSLL on the REL_CUTOFF multigrid (NGRIDS 4) {#gpw-vdw-multigrid}

Self-consistent nonlocal vdW on the CP2K-style `CUTOFF`/`REL_CUTOFF` multigrid
product path. CP2K's own default is `NGRIDS 4`, so every stock CP2K vdW input
is a multigrid input; this section combines the SC nonlocal potential
([`gpw-vdw-scf-potential`](#gpw-vdw-scf-potential)) with the multigrid
collocation ([`gpw-uks-multigrid`](gpw-uks.md#gpw-uks-multigrid) /
[`gpw-multigrid-kpoints`](gpw-rks.md#gpw-multigrid-kpoints)) on the water
dimer, PBE + DRSLL, CUTOFF 200 Ry, NGRIDS 4, REL_CUTOFF 40 Ry, versus offline
CP2K 2026.1 (`ENERGY| Total` and `Dispersion energy`, NGRIDS 4 gate + NGRIDS 1
context).

### Method

Nonlocal vdW is a functional of the finest-grid density. `build_H_and_energy`
prolongs every level's collocated density spectrally to the finest grid before
Poisson/XC and restricts the finest `veff` back per level (exact adjoint of the
prolongation). \(V_{\mathrm{nl}}\) is evaluated on that fine \(\rho\) and added
into `veff` at the same point as the Hartree and semilocal XC potentials —
**before** the per-level restriction — so the integrated KS matrix stays the
exact gradient of the collocated \(E_H + E_{xc} + E_c^{\mathrm{nl}}\) and the
SCF remains variational on both paths. The final \(E_c^{\mathrm{nl}}\)
re-collocates the SC density through the same per-level products + spectral
prolongation.

```cpp
cpu::gpw::GpwDft gpw{kinds, 200.0, cpu::gpw::XcKind::PBE};
gpw.ngrids = 4;        // CP2K NGRIDS (its default is 4)
gpw.rel_cutoff = 40.0; // CP2K REL_CUTOFF [Ry]
gpw.n_occ = 8;
gpw.max_iter = 60;  gpw.tol = 1e-7;
gpw.nonperiodic = false;
gpw.vdw_df = true;     // or vdw_df2 / rvv10
gpw.build(frame);
const cpu::gpw::GpwResult& r = gpw.run(frame);  // r.e_total, r.e_disp [Ha]
```

The section runs the same system twice (`ngrids = 1` and `ngrids = 4`).

### Pin (offline CP2K 2026.1)

| Quantity | CP2K NGRIDS 4 (gate) | CP2K NGRIDS 1 (context) |
|----------|----------------------|-------------------------|
| E_tot [Ha] | −34.186333524093456 | −34.186334128002194 |
| Dispersion energy [Ha] | +0.15193559412669 | +0.15193559998706 |

CP2K's own NGRIDS 4 ↔ 1 gap is 6.0e-7 Ha — the multigrid is a cost
optimization, not a physics change; the section pins the same statement on the
ATV side plus the CP2K E_disp / E_tot parity.

### Accuracy (OMP=8)

ATV: E_tot sg = −34.178411658610 Ha, mg = −34.178407380745 Ha;
E_disp sg = +0.151895580634 Ha, mg = +0.151895426360 Ha.

| Gate | Measured | Tolerance |
|------|----------|-----------|
| \|E_tot(mg) − E_tot(sg)\| (ATV internal consistency) | 4.28e-6 Ha | 2e-5 Ha |
| \|E_disp(mg) − CP2K NGRIDS 4\| | 4.02e-5 Ha (≈ 4.0e-5) | 1e-4 Ha |
| \|E_tot(mg) − CP2K NGRIDS 4\| | 7.93e-3 Ha | 1e-2 Ha |

The absolute E_tot offset is a systematic of this Γ dimer twin family:
identical on both grid paths (Δ_sg = 7.922e-3 vs Δ_mg = 7.926e-3 Ha), and
`eps_default = 1e-10` moves E_tot by < 1e-8 Ha, so it is not tail screening.
The capability statements are the first two rows.

### Speed (same host, OMP=8)

| Run | Wall |
|-----|------|
| CP2K 2026.1 NGRIDS 4 (30 SCF steps) | 24.7 s |
| CP2K 2026.1 NGRIDS 1 (context) | 24.2 s |
| ATV section — **both** solves (sg + mg) | 20.6 s |

Fair ratio \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx 0.83\), charging
both ATV solves against CP2K's single NGRIDS 4 run (≤ 10×).

## gpw-optb88-uks-kpoints — multi-k UKS optB88-vdW {#gpw-optb88-uks-kpoints}

*No section of the case runs this variant.*

Collinear UKS GPW on a Monkhorst–Pack mesh with optB88-vdW: optimized B88
exchange (`GGA_X_OPTB88_VDW`) + PW92 local correlation + self-consistent DRSLL
on the k-summed total density. Recorded checks: Γ-limit identity, OptB88-vs-PBE
dual gate, absolute \(E_{\mathrm{tot}}\) + tight \(E_{\mathrm{disp}}\) versus
offline CP2K, and closed-shell multi-k UKS ≡ RKS identity. Fair wall
\(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}} \approx \mathbf{0.73}\) (≤10×).

| Piece | Form |
| ----- | ---- |
| Exchange | optB88 (`XcKind::OptB88VdW` → `xc_grid_optb88_vdw_spin`) |
| Local correlation | PW92 LDA only |
| Nonlocal | DRSLL SC \(V_{\mathrm{nl}}\) on \(\rho_\alpha+\rho_\beta\) (`build_uks_k_energy`) |
| Final \(E_c^{\mathrm{nl}}\) | `fold_dispersion_k_uks` → `e_disp` [Ha] |

```cpp
ATV::cpu::gpw::GpwDft gpw{{O, H}, 300.0, ATV::cpu::gpw::XcKind::OptB88VdW};
// ngrids, rel_cutoff ...
gpw.n_alpha = 8;
gpw.n_beta = 8;
gpw.vdw_df = true;
gpw.k_mesh[0] = 2;  gpw.k_mesh[1] = 1;  gpw.k_mesh[2] = 1;
gpw.k_time_reversal = false; // 1:1 with CP2K full MP mesh
```

Offline CP2K twin: water dimer, LSD MULTIPLICITY 1, MP 2×1×1, NGRIDS 4 /
REL_CUTOFF 40, `GGA_X_OPTB88_VDW` + `PW92` + `NON_LOCAL TYPE DRSLL`.

| Quantity | CP2K 2026.1 (`git:5e54ba2`) |
|----------|----------------------------|
| E_tot [Ha] | −34.487914403936173 |
| E_disp [Ha] | +0.15206184094414 |
| wall (1×8 OMP) | 13.29 s |

## gpw-optb88-vdw — Γ optB88-vdW (optB88 exchange + DRSLL) {#gpw-optb88-vdw}

*No section of the case runs this variant.*

optB88-vdW (Klimeš–Bowler–Michaelides) on the shared water dimer: semi-local
`XcKind::OptB88VdW` (optimized B88 exchange + PW92 LDA correlation, no GGA-C)
plus `vdw_df` (DRSLL). Exchange parameters (libxc `GGA_X_OPTB88_VDW`, id 139):

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(\beta\) | 0.003368671161034469 | dimensionless |
| \(\gamma\) | 6.981317007977318 | dimensionless |

(Stock B88 uses \(\beta=0.0042\), \(\gamma=6\).) Implementation pieces:
`XCType::OptB88VdW` (`xc_functional.h`), `xc_grid_optb88_vdw` / spin
(`xc_grid_gga.h`).

```cpp
GpwDft gpw{{O, H}, 200.0, XcKind::OptB88VdW};
gpw.n_occ = 8;
gpw.max_iter = 40;  gpw.tol = 1e-7;  gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.vdw_df = true;
```

CP2K twin surface (Hutter recipe; offline input):

```
&XC_FUNCTIONAL
  &LIBXC
    FUNCTIONAL XC_GGA_X_OPTB88_VDW
  &END LIBXC
  &PW92
  &END PW92
&END XC_FUNCTIONAL
&vdW_POTENTIAL
  DISPERSION_FUNCTIONAL NON_LOCAL
  &NON_LOCAL
    TYPE DRSLL
  &END NON_LOCAL
&END vdW_POTENTIAL
```

The CP2K on the pin host is built without LIBXC (`unknown subsection LIBXC`),
so no absolute CP2K digits exist for this Γ twin. The recorded checks were
(1) a semi-local dual gate versus PBE, (2) a frozen ATV value for full
optB88-vdW, and (3) kernel-level pins of the optB88 exchange parameters and
of DRSLL versus libvdwxc. Recorded values (DRSLL evaluated post-SCF on the
converged optB88 density):

| Quantity | Value | Oracle / notes | Tol | Notes |
| -------- | ----: | -------------: | --- | ----- |
| \(\lvert E_{\mathrm{OptB88}} - E_{\mathrm{PBE}}\rvert\) [Ha] (semi-local, `vdw_df=false`) | **0.303** | dual-gate floor \(>10^{-4}\) | \(10^{-4}\) Ha floor | Fails closed if `OptB88VdW` were a PBE alias |
| \(E_{\mathrm{disp}}\) [Ha] (optB88 density + DRSLL) | **1.5207282287995946×10⁻¹** | frozen ATV value; kernel vs libvdwxc in `gpw-vdw-df-ice` | \(5\times 10^{-5}\) Ha | DRSLL pin family |
| \(E_{\mathrm{tot}}\) [Ha] (optB88-vdW) | **−34.481701207152035** | frozen until a LIBXC CP2K twin | \(1\times 10^{-3}\) Ha | Integrity bound |
| \(E_{\mathrm{OptB88}}\) semi-local [Ha] | −34.633774030031404 | context | — | no nonlocal |
| \(E_{\mathrm{PBE}}\) semi-local [Ha] | −34.330331404832023 | context | — | dual-gate baseline |
| SCF iters (full) | 11 (converged) | context | — | — |

\(E_{\mathrm{disp}}\) is close to the PBE+DRSLL pin of
[`gpw-vdw-df-ice`](#gpw-vdw-df-ice) (**1.519439×10⁻¹ Ha**) — a density-driven
difference only. Performance: **perf_exempt** (no fair CP2K QS wall without
LIBXC).

## gpw-vdw-df2-ice — PBE + vdW-DF2 (LMKLL) energy {#gpw-vdw-df2-ice}

*No section of the case runs this variant.*

PBE host + `vdw_df2` (LMKLL / libvdwxc DF2, \(Z_{ab}=-1.887\)) on the shared
water dimer, CUTOFF 200 Ry; LMKLL is additive nonlocal correlation, not a
replacement for PBE \(E_c^{\mathrm{loc}}\). The recorded value froze
\(E_c^{\mathrm{nl}}\) on the converged ATV PBE density (post-SCF) after the
libvdwxc DF2 kernel pin (`vdw_df_lmkll`, `kVdwDf2Zab`).

```cpp
GpwDft gpw{kinds, 200.0, XcKind::PBE};   // shared water-dimer settings
gpw.vdw_df2 = true;
```

| Quantity | Value | Oracle (offline) | Tol | Notes |
| -------- | ----: | ----------------: | --- | ----- |
| \(E_c^{\mathrm{nl}}\) product [Ha] | \(+1.3011875519702767\times 10^{-1}\) | dual-frozen | \(5\times 10^{-5}\) Ha | Recorded value |
| Kernel \(E_{\mathrm{nl}}\) synthetic 16³ [Ha] | recorded match | libvdwxc `VDWXC_DF2` \(+1.1428161716669384\times 10^{-1}\) | \(1\times 10^{-6}\) Ha | Fixed Gaussian blob, L=8 Bohr |
| \(E_{\mathrm{tot}}\) QS [Ha] | (not gated) | — | — | Context only |

Compare with DRSLL ([`gpw-vdw-df-ice`](#gpw-vdw-df-ice)) only via interaction
differences, not raw \(E_{\mathrm{disp}}\). Performance: **perf_exempt** (no
CP2K+libvdwxc QS wall on the pin host).
