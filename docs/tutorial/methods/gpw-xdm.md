# GPW free-atom XDM dispersion

Exchange-hole dipole moment (XDM) pairwise dispersion is an additive London
correction on top of Γ-point GPW PBE. Atomiverse implements the **free-atom**
form: the multipole moments \(M_1,M_2,M_3\) and static polarizabilities
\(\alpha\) are tabulated free-atom values — no Hirshfeld volume scaling and no
density-dependent Becke–Roussel hole moments. Because the energy is
geometry-only, it does not change the SCF density; its forces add to the
electronic forces.

The method is the member `xdm` of `cpu::gpw::GpwDft` (one `Kind` per element
from `GaussianBasis` / `GthPotential`, `cutoff_ry` and `xc` as constructor
arguments; `build(frame)` → `run(frame)`). Dispersion is reported on
`GpwResult::e_disp` [Hartree] and folded into `e_total`; with forces requested,
`assemble_gpw_forces` adds \(\mathbf F^{\mathrm{XDM}}\) to the electronic
forces. Driver work binds the configured `GpwDft` through
`cpu::gpw::GPWEvaluator` (pass `compute_forces=true` for forces). `xdm` is
mutually exclusive with `d3` / `d3bj` / `d4` / `ts` / `mbd` / `vdw_df` /
`vdw_df2` / `rvv10` — any second flag true throws at run. The sections below
are CPU-only and cover the periodic dispersion energy (`gpw-xdm-ice`) and the
analytic ionic forces of the kernel (`gpw-xdm-forces`). The case additionally
carries `gpw-xdm-stress`, `gpw-xdm-kpoint-forces` and `gpw-xdm-kpoint-stress`,
whose oracle record lives in the case `REFERENCE.md`.

Case: `ctest -R '^gpw/xdm$'` — sources and oracle record in [`regressions/gpw/xdm/`](../../../regressions/gpw/xdm/). One section of the case per `##` heading below: `atv_regressions --case gpw/xdm --section <name>`.

## Common setup

### Method

**Free-atom XDM** with Becke–Johnson combination rules and BJ damping
(Becke & Johnson 2007; postg / QE `PW/src/xdm_dispersion.f90`,
Otero-de-la-Roza & Johnson). Periodic pure real-space lattice sum over image
vectors \(\mathbf{T}\):

\[
\mathrm{fac}=\frac{\alpha_i\alpha_j}{M_{1i}\alpha_j+M_{1j}\alpha_i},\quad
C_6=\mathrm{fac}\,M_{1i}M_{1j},
\]

\[
C_8=\tfrac{3}{2}\,\mathrm{fac}\,(M_{1i}M_{2j}+M_{2i}M_{1j}),
\]

\[
C_{10}=2\,\mathrm{fac}\,(M_{1i}M_{3j}+M_{3i}M_{1j})
+\tfrac{21}{5}\,\mathrm{fac}\,M_{2i}M_{2j},
\]

\[
R_c=\frac{1}{3}\Bigl(\sqrt{C_8/C_6}+(C_{10}/C_6)^{1/4}+\sqrt{C_{10}/C_8}\Bigr),
\quad R_{\mathrm{vdW}}=a_1 R_c+a_2,
\]

\[
E_{\mathrm{disp}}=-\sum_{i\le j}\sum_{\mathbf T}{}^{\prime}
\Biggl[
\frac{C_6}{R^6+R_{\mathrm{vdW}}^6}
+\frac{C_8}{R^8+R_{\mathrm{vdW}}^8}
+\frac{C_{10}}{R^{10}+R_{\mathrm{vdW}}^{10}}
\Biggr].
\]

The prime excludes the singular \(R=0\) self term. Analytic ionic forces come
from the same kernel, \(\mathbf{F}_A^{\mathrm{XDM}} = -\partial
E_{\mathrm{disp}}/\partial \mathbf{R}_A\) [Ha/Bohr], and the product ionic force
is

\[
\mathbf{F}_A
=
\mathbf{F}_A^{\mathrm{HF+Pulay+grid/core}}
+ \mathbf{F}_A^{\mathrm{XDM}}
\]

(same wiring pattern as free-atom TS / D3(BJ) / D4 / MBD).

**PBE BJ damping (QE NC-PBE / postg `xdm.param`):**

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| \(a_1\) | 0.4283 | dimensionless |
| \(a_2\) | 2.4690 | Å (= **4.6657338220946531** Bohr, \(\approx 4.66573\) in-tree) |

**Free-atom tables** (in-tree `basis_data/xdm_tables.h`; CRC Handbook 88th Ed.
\(\alpha\); \(M_n\) inverted from literature homonuclear \(C_n\)):

| Z | \(\alpha\) [Bohr³] | \(M_1\) | \(M_2\) | \(M_3\) | \(C_n\) source |
| - | -----------------: | ------: | ------: | ------: | -------------- |
| H | 4.49979 | 2.88859 | 18.4304 | 241.636 | Yan et al., *Phys. Rev. A* **54**, 2824 (1996) |
| O | 5.41216 | 5.74631 | 28.9471 | 262.617 | Chu–Dalgarno, *J. Chem. Phys.* **121**, 4083 (2004) (\(C_6\)); consistent \(C_8,C_{10}\) |

Only H and O are populated; unsupported \(Z\) throws (no fallback).

**Pair cutoff.** Pure real-space orthorhombic lattice sum via
`pair_image_counts` with **rcut = 50 Å** (\(\approx 94.2\) Bohr default in
`xdm_pbe`). Å→Bohr is \(1/0.52917720859\).

### System

Both sections use the ice-relevant H-bonded water dimer (not a full ice Ih
supercell), matching `gpw-ts-ice` / `gpw-pbe-d3bj-ice`:

| Item | Value |
| ---- | ----- |
| Atom order | O H H O H H |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ (isolated pins: same nuclei, no lattice images) |
| XC + disp | PBE + free-atom XDM (QE NC-PBE BJ \(a_1,a_2\)) |
| Basis / PP (short SCF) | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) — E_disp and F_XDM are basis-independent |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| SCF | cutoff 200 Ry, `max_iter=8`, `tol=1e-6`, mix 0.4 |

The short Γ-RKS SCF only materializes `e_disp` on a real `GpwResult` (and in
`e_total`):

```cpp
frame.box = ATV::cpu::Box::cubic(12.0);  // Å; kinds = {O, H}
GpwDft gpw{kinds, 200.0, XcKind::PBE};   // e_disp is geometry-only
gpw.n_occ = 8;
gpw.max_iter = 8;  gpw.tol = 1e-6;  gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.xdm = true;
gpw.build(frame);
const auto& res = gpw.run(frame);   // res.e_disp within 1e-8 of the periodic pin
```

Driver path (forces land on Frame `kAtom_f*` [eV/Å], electronic + XDM):

```cpp
ATV::cpu::gpw::GpwDft dft{kinds, 300.0, ATV::cpu::gpw::XcKind::PBE};
dft.max_iter = 200;  dft.tol = 1e-7;  dft.mix = 0.3;
dft.nonperiodic = false;
dft.xdm = true;
ATV::cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/true);
```

### Oracle

Stock **CP2K 2026.1** `PAIR_POTENTIAL TYPE` accepts only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is no `TYPE XDM`. Canonical full
(density-dependent) XDM oracles are postg / critic2 / Quantum ESPRESSO
`xdm_dispersion`. The hard gates are therefore an **offline dual pure-Python
lattice sum** of the free-atom formula with the same tables (external offline
pin, not a golden of Atomiverse output; never live CP2K / postg / QE in ctest).
Matching tables, BJ combination rules and the 50 Å image enumeration track the
dual sum to machine precision on the kernel (\(\sim 10^{-16}\) Ha and
Ha/Bohr). No unit test in the tree gates the XDM kernel; the isolated
dual-Python energy and the analytic-vs-FD force check run inside
`gpw-xdm-forces`. Build with the standard CPU environment
(`buildenv-gcc/2026.03-mpich`) and `-DATV_BUILD_GPW=ON
-DATV_BUILD_REGRESSIONS=ON`.

**Performance: perf_exempt.** XDM evaluation is
\(O(N^2\times N_{\mathrm{images}})\) over a 6-atom cell (microseconds); full GPW
SCF wall is dominated by the electronic structure, and there is no stock CP2K
XDM `ENERGY` / `ENERGY_FORCE` twin to fair-bench (same policy as `gpw-ts-ice`
/ `gpw-pbe-d3bj-ice` / `gpw-ts-forces` / `gpw-mbd-forces`). The acceptance is
E_disp (+ force dual-gate) parity, not \(r = T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}
\le 10\).

### Shared limitations

- **Free-atom only** — no Hirshfeld / BR density-dependent moments (contrast
  density-scaled TS/MBD via `GpwDft::hirshfeld`, [gpw-hirshfeld.md](gpw-hirshfeld.md)).
- **H and O tables only**; other \(Z\) throws.
- **Water dimer, not bulk ice Ih** — enough for lattice-sum / pairwise force
  parity, not a structural ice benchmark; no phonons.
- **E_disp / F_XDM gates only** — absolute GPW \(e_{\mathrm{tot}}\) micro-Ha
  parity versus any QS engine is not asserted (grid/OT/DIIS residual).
- No CUDA GPW.

### References

- A. D. Becke, E. R. Johnson, *J. Chem. Phys.* **127**, 154108 (2007) — XDM
- A. Otero-de-la-Roza, E. R. Johnson, *J. Chem. Phys.* **136**, 174109 (2012) —
  postg / damping practice; QE NC-PBE \(a_1,a_2\); free-atom tables
- Z.-C. Yan et al., *Phys. Rev. A* **54**, 2824 (1996) — H \(C_n\)
- X. Chu, A. Dalgarno, *J. Chem. Phys.* **121**, 4083 (2004) — O \(C_6\)
- J. VandeVondele et al., *Comput. Phys. Commun.* **167**, 103 (2005) — GPW host
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md); roadmap
  [`docs/roadmap/dft.md`](../../roadmap/dft.md) §8 Dispersion

## gpw-xdm-ice — periodic water-dimer dispersion energy {#gpw-xdm-ice}

Gates the periodic free-atom XDM energy materialized by a short `GpwDft` SCF
with `xdm=true` (code under [Common setup](#common-setup)).

| Item | Value |
| ---- | ----- |
| R_CUTOFF | 50 Å pure real-space |
| Pin line | \(E_{\mathrm{disp}} = -2.1148157612879029\times 10^{-3}\) Ha |

| Quantity | Atomiverse / pin | Offline dual-Python | Tol | Notes |
| -------- | ---------------: | ------------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) periodic [Ha] | −2.114815761287903×10⁻³ | −2.114815761287903×10⁻³ | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) isolated [Ha] | (gated in `gpw-xdm-forces`) | −2.102043599851622×10⁻³ | tight kernel | Cross-check |
| \(E_{\mathrm{tot}}\) QS [Ha] | (not gated) | — | — | Context only; no CP2K XDM twin |

The kernel tracks the dual sum to \(\sim 10^{-16}\) Ha; the regression gate is
\(1\times 10^{-8}\) Ha. Not part of this section: k-points, UKS, cell stress.

## gpw-xdm-forces — analytic ionic forces {#gpw-xdm-forces}

Gates the geometry-only XDM energy and analytic forces of the `xdm_pbe` kernel
(`XdmResult::{energy,forces}`), isolated and periodic, against the dual
pure-Python sum plus an independent central finite difference
(\(h = 10^{-5}\) Bohr), then checks that a short `GpwDft` SCF with `xdm=true`
reports the same periodic `e_disp`. Same force dual-gate policy as
[`gpw-mbd-forces`](gpw-mbd.md#gpw-mbd-forces) / `gpw-ts-forces`.

```cpp
// Z = {8,1,1,8,1,1}; pos in Bohr (kA2B = 1.0 / 0.52917720859)
const auto xdm_iso = xdm_pbe(Z, pos, /*cell*/{0, 0, 0}, /*periodic=*/false);
const double L = 12.0 * kA2B;
const double cellp[3] = {L, L, L};
const double rcut = 50.0 * kA2B;
const auto xdm_per = xdm_pbe(Z, pos, cellp, /*periodic=*/true, rcut);
// xdm_*.energy [Ha]; xdm_*.forces[i][d] [Ha/Bohr]; central FD h=1e-5 Bohr
```

### Pins

| Quantity | Pin | Tol (ATV gate) | Notes |
| -------- | --: | -------------- | ----- |
| \(E_{\mathrm{disp}}\) isolated [Ha] | **−2.1020435998516220×10⁻³** | \(1\times 10^{-8}\) Ha | Dual pure-Python |
| \(E_{\mathrm{disp}}\) periodic [Ha] | **−2.1148157612879029×10⁻³** | \(1\times 10^{-8}\) Ha | Same as `gpw-xdm-ice` |
| max\|F_product − F_fd\| [Ha/Bohr] | ≤ **1×10⁻⁶** | dual-gate (iso + periodic) | Independent central FD |
| \(\|\sum_A\mathbf{F}_A\|\) [Ha/Bohr] | ≤ **1×10⁻⁸** | translation residual | Free-atom pairwise |
| max\|F_iso − F_dual_python sample\| [Ha/Bohr] | ≤ **1×10⁻¹⁰** | hard component pin | Offline analytic ≡ ATV tables |
| `GpwDft` `e_disp` [Ha] | = periodic pin | \(1\times 10^{-8}\) Ha | Product SCF materialization |
| Cross-check dual-Python vs ATV forces | \(\sim 10^{-16}\) Ha/Bohr | offline only | Never in ctest |

**Isolated free-atom force sample** [Ha/Bohr] (dual pure-Python ≡ ATV):

| Atom | \(F_x\) | \(F_y\) | \(F_z\) |
| ---- | ------: | ------: | ------: |
| O0 | +1.6875829936003448×10⁻⁴ | 0 | 0 |
| H0 | +6.9736686381575340×10⁻⁵ | 0 | −1.3172272218329521×10⁻⁵ |
| H1 | +6.9736686381575340×10⁻⁵ | 0 | +1.3172272218329521×10⁻⁵ |
| O1 | −1.1378113807647229×10⁻⁴ | 0 | 0 |
| H2 | −9.7225267023356434×10⁻⁵ | 0 | −1.5448899346549854×10⁻⁵ |
| H3 | −9.7225267023356434×10⁻⁵ | 0 | +1.5448899346549854×10⁻⁵ |

**Tolerance rationale.** The FD dual-gate uses \(1\times 10^{-6}\) Ha/Bohr to
cover central-difference truncation while staying tight versus the physical
\(|F|\) (\(\sim 10^{-4}\) Ha/Bohr on O0); the \(1\times 10^{-10}\) Ha/Bohr
component pin against the dual-Python sample is the hard offline force check.

### Limitations

- Ionic forces only; XDM stress is not gated by this section.
- The full GPW+XDM product force (electronic + XDM) against a multi-SCF FD is
  not gated by this section; it checks the geometry-only kernel and the SCF
  `e_disp`.
- Frame forces stay eV/Å on the public surface; pins compare in Ha/Bohr
  (`units::kEvPerAngstromInHartreePerBohr` when converting Frame columns).
