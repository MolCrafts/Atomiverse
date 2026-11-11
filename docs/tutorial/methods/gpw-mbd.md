# GPW many-body dispersion (MBD@rsSCS)

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Many-body dispersion with range-separated self-consistent screening
(MBD@rsSCS) is an additive London correction on top of GPW PBE. Free-atom
polarizabilities, \(C_6\) coefficients and van der Waals radii are screened
self-consistently and coupled through a long-range dipole Hamiltonian whose
zero-point energy shift is \(E_{\mathrm{disp}}\). With free-atom volumes
(\(V/V_{\mathrm{free}}=1\)) the energy is geometry-only; with `hirshfeld` the
parameters are rescaled from the converged density (see
[GPW Hirshfeld-scaled MBD](gpw-hirshfeld.md)).

The method is the member `mbd` of `cpu::gpw::GpwDft` (one `Kind` per element
from `GaussianBasis` / `GthPotential`, `cutoff_ry` and `xc` as constructor
arguments). `mbd_k_mesh = {0,0,0}` (default) evaluates **isolated** MBD on the
nuclear coordinates; a non-zero mesh switches to periodic Ewald dipole tensors
with Brillouin-zone k-mesh averaging. `run(frame)` reports \(E_{\mathrm{disp}}\)
on `GpwResult::e_disp` [Hartree] and folds it into `e_total`; `forces(frame)`
adds \(\mathbf F^{\mathrm{MBD}}\) to the electronic forces on the Γ and the
multi-k path; `cpu::gpw::GPWEvaluator` binds the configured method into a
driver and, with `compute_stress = true`, returns the analytic fixed-\(P\)
cell stress with \(E_{\mathrm{disp}}\) rebuilt at every strained cell. `mbd` is
mutually exclusive with `d3` / `d3bj` / `d4` / `ts` / `xdm` / `vdw_df` /
`vdw_df2` / `rvv10`. The sections are CPU-only and cover the isolated and
periodic energies, the Hirshfeld-scaled energy, Γ and multi-k ionic forces,
and Γ and multi-k cell stress.

Case: `ctest -R '^gpw/mbd$'` — sources and oracle record in [`regressions/gpw/mbd/`](../../../regressions/gpw/mbd/). One section of the case per `##` heading below: `atv_regressions --case gpw/mbd --section <name>`.

## Common setup

### Method

**MBD@rsSCS** (Ambrosetti et al. 2014; libMBD, Hermann et al. 2023):

1. **Range-separated self-consistent screening** of dynamical polarizabilities
   on a frequency grid (Fermi × Gaussian-damped dipole tensor, \(\beta\)
   damping). In the periodic path the SCS sums over lattice images without
   Ewald.
2. **Long-range MBD Hamiltonian** diagonalization → zero-point energy
   difference relative to the non-interacting atomic oscillators. The
   periodic path uses the Ewald dipole tensor and averages over a shifted
   k-mesh (libMBD `get_kpts`, shift \(\tfrac12\)):

\[
E_{\mathrm{MBD}} = \frac{1}{2N_k}\sum_{\mathbf{k}}\sum_{p}
  \sqrt{\lambda_p(\mathbf{k})}
  - \frac{3}{2}\sum_A\omega_A
\]

(schematic ZPE form; the implementation matches libMBD / pymbd `mbd_energy`).

**PBE defaults:**

| Parameter | Value | Unit / note |
| --------- | ----: | ----------- |
| \(\beta\) | 0.83 | range separation |
| Fermi \(a\) | 6 | damping steepness |
| \(V/V_{\mathrm{free}}\) | 1 | free-atom (unless `hirshfeld`) |
| Frequency grid | \(n=15\), \(L=0.6\) | libMBD `freq_grid` |

**Free-atom parameters** (libMBD `ts_vdw_params`, in-tree
`basis_data/ts_tables.h`):

| Z | \(\alpha_0\) [Bohr³] | \(C_6\) [Ha·Bohr⁶] | \(R_{\mathrm{vdW}}\) [Bohr] |
| - | -------------------: | -----------------: | --------------------------: |
| H | 4.5 | 6.5 | 3.1 |
| O | 5.4 | 15.6 | 3.19 |

Characteristic frequencies \(\omega = \tfrac{4}{3} C_6 / \alpha_0^2\). After
rsSCS, \(C_6^{\mathrm{rsscs}} = \tfrac{3}{\pi}\sum_u w(u)\,\alpha(u)^2\) and
\(R_{\mathrm{vdW}}\) is rescaled by \((\alpha(0)/\alpha_0)^{1/3}\).

**Boundary-condition switch.**

| `mbd_k_mesh` | Dispersion BC |
| ------------ | ------------- |
| `{0,0,0}` (default) | Isolated MBD@rsSCS on the nuclear coordinates, even when the GPW SCF is periodic |
| non-zero | Periodic Ewald + MBD k-mesh; requires a periodic cell (`!nonperiodic`) |

The electronic `k_mesh` (SCF Brillouin mesh) and `mbd_k_mesh` (MBD Ewald
lattice-sum mesh) are independent.

**Ionic forces.** \(\mathbf{F}_A^{\mathrm{MBD}} = -\partial
E_{\mathrm{disp}}/\partial \mathbf{R}_A\) [Ha/Bohr] is a high-order central
finite difference of the same energy kernel (product step \(h = 10^{-4}\)
Bohr), not a closed-form libMBD gradient. `assemble_gpw_forces` (Γ) and
`assemble_gpw_forces_kpoints` (multi-k) add it to the electronic force:

\[
\mathbf{F}_A
  = \mathbf{F}_A^{\mathrm{HF+Pulay+grid/core}}
  + \mathbf{F}_A^{\mathrm{MBD}}.
\]

With free-atom volumes MBD does not change the SCF density. The Hirshfeld
density chain \(\partial V/\partial R\) is not part of these forces.

**Cell stress.** The Hellmann–Feynman fixed-\(P\) path: one SCF, then a
central FD of the non-self-consistent Lagrangian under diagonal strain at
fixed fractional coordinates (strain step `method().stress_strain_h`), with
\(E_{\mathrm{disp}}^{\mathrm{MBD}}\) **rebuilt at every strained cell**
(same `mbd_rsscs_pbe` call); \(\sigma_{aa} =
-\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\) (CP2K `STRESS_TENSOR`
sign, Ha/Bohr³ internally; Frame `kStress` reports σ·Ω in eV). Holding
\(E_{\mathrm{disp}}\) fixed would differentiate a different functional than
the SCF converged — the failure mode behind a 0.53 Ha/Bohr ZnO force error.

### System

Every section except the multi-k force wiring cell uses the ice-relevant
H-bonded water dimer of [`gpw-ts-ice`](gpw-tkatchenko-scheffler.md#gpw-ts-ice)
/ [`gpw-pbe-d3bj-ice`](gpw-d3.md#gpw-pbe-d3bj-ice) (not bulk ice Ih):

| Atom | \(x\) [Å] | \(y\) [Å] | \(z\) [Å] |
| ---- | ----: | ----: | ----: |
| O0 | 0.000000 | 0.000000 | 0.000000 |
| H0 | 0.758602 | 0.000000 | 0.504284 |
| H1 | 0.758602 | 0.000000 | −0.504284 |
| O1 | 3.000000 | 0.000000 | 0.000000 |
| H2 | 3.758602 | 0.000000 | 0.504284 |
| H3 | 3.758602 | 0.000000 | −0.504284 |

Cell ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ; Å→Bohr \(1/0.52917720859\) (ATV).
Basis / PP `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H); PBE; RKS,
16 valence e⁻ → `n_occ = 8`. For free-atom MBD, \(E_{\mathrm{disp}}\) and
\(\mathbf F^{\mathrm{MBD}}\) are basis-independent; the short SCF only
materializes `e_disp` on a real `GpwResult`:

```cpp
frame.box = cpu::Box::cubic(12.0);   // Å; kinds = {O, H}
GpwDft gpw{kinds, 200.0, XcKind::PBE};
gpw.n_occ = 8;
gpw.max_iter = 8;  gpw.tol = 1e-6;  gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.mbd = true;
gpw.build(frame);
const auto& res = gpw.run(frame);   // res.e_disp within 1e-8 of the pin
```

Driver path (the same configured method bound as a component):

```cpp
ATV::cpu::gpw::GpwDft dft{kinds, 300.0, ATV::cpu::gpw::XcKind::PBE};
dft.max_iter = 200;  dft.tol = 1e-7;  dft.mix = 0.3;
dft.nonperiodic = false;
dft.mbd = true;
ATV::cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/true);
// bind on Evaluate / MD / Minimize; eval.last_result().e_disp [Hartree];
// Frame kAtom_f* [eV/Å] include F_MBD; public energy stays eV on kEnergy
```

### Oracle

Stock **CP2K 2026.1** `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — MBD exists in CP2K only through an externally
linked libMBD. The hard pins therefore come from the libMBD family:

- **Isolated:** offline **libMBD** Fortran `cmbd_mbd_scs_energy` (git
  `88069c6`), geometry converted with the ATV Å→Bohr factor.
- **Periodic / strained cells:** offline **pure-Python pymbd**
  (`libmbd/src/pymbd/pymbd.py`, `mbd_energy` / `mbd_energy_species`, same
  libmbd git `88069c6`).
- **Forces / stress:** product gradient versus an independent central FD of
  the pinned energy, and analytic stress versus multi-SCF strain-FD
  (`gpw_stress_fd`) — the analytic quantity is the derivative of an externally
  pinned functional, the same structure as
  [`gpw-d4-stress`](gpw-d4.md#gpw-d4-stress).

External engine pins only — never a golden of Atomiverse output, never live
CP2K / libMBD / pymbd in ctest. Build with the standard CPU environment
(`buildenv-gcc/2026.03-mpich`) and `-DATV_BUILD_GPW=ON
-DATV_BUILD_REGRESSIONS=ON`.

**Performance: perf_exempt** for every section — no stock CP2K MBD twin to
time against. MBD is an \(O(N^3)\) eigensolve of a \(3N\times 3N\) matrix
(\(O(N_k \cdot (3N)^3)\) periodic; microseconds to milliseconds on 6 atoms),
the product force FD costs \(6N\) energy evaluations, and full GPW SCF wall
dominates (same policy as `gpw-ts-ice` / `gpw-pbe-d3bj-ice` / `gpw-d4-water`).

### Shared limitations

- **Water dimer, not bulk ice Ih**; no phonons.
- **Absolute GPW \(e_{\mathrm{tot}}\)** versus any QS engine is not asserted
  (OT/grid/DIIS).
- **Free-atom volumes** except in [`gpw-mbd-hirshfeld`](#gpw-mbd-hirshfeld);
  do not mix free-atom and Hirshfeld pin values.
- **Mutual exclusion** with the other dispersion / nonlocal flags — any
  conflicting pair throws at run (energy and force assembly).
- No CUDA GPW; no UKS-specific MBD force / stress gate.

### References

- A. Ambrosetti, A. M. Reilly, R. A. DiStasio Jr., A. Tkatchenko,
  *J. Chem. Phys.* **140**, 18A508 (2014) — MBD@rsSCS
- A. Tkatchenko, R. A. DiStasio Jr., R. Car, M. Scheffler, *Phys. Rev. Lett.*
  **108**, 236402 (2012) — MBD
- J. Hermann, M. Stöhr, S. Góger, S. Chaudhuri, B. Aradi, R. J. Maurer,
  A. Tkatchenko, *J. Chem. Phys.* **159**, 174802 (2023) — libMBD
- A. Tkatchenko, M. Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009) —
  Hirshfeld / TS volume scaling
- Narrative: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md); roadmap
  [`docs/roadmap/dft.md`](../../roadmap/dft.md) §8 Dispersion, §10 Periodic KS;
  siblings [gpw-pbe-d3bj-ice](gpw-d3.md#gpw-pbe-d3bj-ice),
  [gpw-d4-water](gpw-d4.md#gpw-d4-water)

## gpw-mbd-ice — isolated free-atom energy {#gpw-mbd-ice}

Free-atom MBD@rsSCS (\(\beta=0.83\), \(a=6\)), isolated on the water-dimer
nuclei while the SCF cell is periodic (code under [Common setup](#common-setup)).

| Quantity | Atomiverse / pin | libMBD (offline) | Tol | Notes |
| -------- | ---------------: | ---------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −1.4928804171407961×10⁻³ | −1.4928804171407961×10⁻³ | \(1\times 10^{-8}\) Ha | Hard gate (`E_disp = -1.4928804171407961e-03` Ha) |
| pure-Python `pymbd` (same ATV Å→Bohr) | — | −1.4928804171425725×10⁻³ | — | Cross-check \(\sim 2\times 10^{-15}\) Ha |
| pure-Python (libmbd 0.529177249) | — | −1.4928806007885598×10⁻³ | — | Conversion-sensitivity note |
| \(E_{\mathrm{tot}}\) QS [Ha] | (not gated) | n/a | — | Context only |

Matching tables, \(\beta\)/\(a\), frequency grid and isolated dipole tensors
track libMBD Fortran to \(\sim 10^{-15}\) Ha; the gate is \(10^{-8}\) Ha. The
standalone `mbd_rsscs_pbe` kernel agreement was recorded at \(10^{-12}\) Ha (no
unit test in the tree gates it).

## gpw-mbd-periodic — periodic Ewald + k-mesh energy {#gpw-mbd-periodic}

Same free-atom MBD on the water dimer with `mbd_k_mesh = {2,2,2}` (libMBD
shift-½): periodic Ewald dipole tensors and k-mesh ZPE averaging.

```cpp
gpw.mbd = true;
gpw.mbd_k_mesh[0] = gpw.mbd_k_mesh[1] = gpw.mbd_k_mesh[2] = 2; // periodic
```

| Quantity | Atomiverse / pin | pymbd (offline) | Tol | Notes |
| -------- | ---------------: | --------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −1.4965720704989138×10⁻³ | −1.4965720704989138×10⁻³ | \(1\times 10^{-8}\) Ha | Hard gate (k=2×2×2) |
| \(E_{\mathrm{disp}}\) isolated [Ha] | (sibling) | −1.4928804171407961×10⁻³ | — | [`gpw-mbd-ice`](#gpw-mbd-ice) |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | — | — | Not absolute-parity gated |

**k-mesh convergence (offline context):**

| k-mesh | \(E_{\mathrm{disp}}\) [Ha] |
| ------ | -------------------------: |
| \(1\times 1\times 1\) | −1.4474832766229184×10⁻³ |
| \(2\times 2\times 2\) | **−1.4965720704989138×10⁻³** (hard pin) |
| \(3\times 3\times 3\) | −1.5008181948612531×10⁻³ |

At 12 Å the image coupling is weak — the 2×2×2 vs isolated difference is only
\(\sim 3.7\times 10^{-6}\) Ha — but the section exercises the full Ewald +
k-mesh path. The standalone periodic kernel agreement was recorded at
**1e-10 Ha**.

Offline re-pin (human only):

```python
# coords Å→Bohr with ATV factor; lattice diag(12*kA2B); free-atom alpha_0, C6, R_vdw
# k_grid=(2,2,2); beta=0.83
mbd_energy(coords, alpha_0, C6, R_vdw, beta=0.83, lattice=lattice, k_grid=(2,2,2))
# → -1.4965720704989138e-03 Ha
```

## gpw-mbd-hirshfeld — Γ Hirshfeld-scaled energy {#gpw-mbd-hirshfeld}

Hirshfeld volume ratios from the final Γ-RKS PBE density scale the free-atom
parameters (\(\alpha_A \propto v_A\), \(C_6^{A}\propto v_A^2\),
\(R_{\mathrm{vdW}}^{A}\propto v_A^{1/3}\)) before isolated MBD@rsSCS. The
Hirshfeld kernel and its exponential free-atom model are described on
[GPW Hirshfeld-scaled MBD](gpw-hirshfeld.md#common-setup); the model constants
are

| Z | \(N_{\mathrm{val}}\) | \(\alpha\) [Bohr⁻¹] |
| - | -------------------: | ------------------: |
| H | 1 | 2.0 |
| O | 6 | 2.2 |

```cpp
GpwDft gpw{kinds, 200.0, XcKind::PBE};   // pin is this fixed product setting
gpw.n_occ = 8;
gpw.max_iter = 16;  gpw.tol = 1e-6;  gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.mbd = true;
gpw.hirshfeld = true;   // post-SCF volumes → mbd_rsscs_pbe(…, vol_ratio)
// res.volume_ratios: 6 entries in (0.3, 1.5)
```

The oracle is libMBD free-atom MBD ([`gpw-mbd-ice`](#gpw-mbd-ice)) plus a dual
pure-Python Hirshfeld on the same exponential model; the gate freezes the ATV
product with `mbd` + `hirshfeld` at CUTOFF 200 Ry, max_iter 16, tol 1e-6,
mix 0.4.

| Quantity | Atomiverse / pin | Offline cross-check | Tol | Notes |
| -------- | ---------------: | ------------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) Hirshfeld-MBD [Ha] | −1.1479852469005181×10⁻³ | product dual gate | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) free-atom MBD [Ha] | −1.4928804171407961×10⁻³ | libMBD Fortran `88069c6` | \(1\times 10^{-8}\) / kernel 1e-12 | Context |
| Volume ratios \(v_A\) (O, H, H, O, H, H) | `0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784` | dual pure-Python free-atom \(v\approx 1\) | range (0.3, 1.5); free-atom O within 3e-2 | Regression bounds |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | — | — | Not absolute-parity gated |

The dual stack agrees to nano-Hartree at this fixed CUTOFF/basis; because
E_disp is density-dependent it moves at the mHa level if cutoff, basis or tol
change. Hirshfeld scaling makes \(E_{\mathrm{disp}}\) ~23% less attractive than
free-atom here, as expected for compressed molecular volumes
(\(v_A\in[0.58,0.74]\)).

## gpw-mbd-forces — Γ ionic forces (free-atom) {#gpw-mbd-forces}

Gates the isolated free-atom kernel `mbd_rsscs_pbe` (`MbdResult::{energy,
forces}`) against the libMBD energy pin and an independent central FD
(\(h = 10^{-5}\) Bohr), then checks the short `GpwDft` SCF reports the same
`e_disp`. Same force dual-gate policy as
[`gpw-ts-forces`](gpw-tkatchenko-scheffler.md#gpw-ts-forces) /
[`gpw-d4-forces`](gpw-d4.md#gpw-d4-forces).

```cpp
std::vector<int> Z = {8, 1, 1, 8, 1, 1};   // pos in Bohr, kA2B = 1.0 / 0.52917720859
const double cell0[3] = {0.0, 0.0, 0.0};
const MbdResult mbd = mbd_rsscs_pbe(Z, pos, cell0, /*periodic=*/false);
// mbd.energy [Ha] ↔ libMBD pin; mbd.forces [Ha/Bohr] ↔ independent FD
```

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | **−1.4928804171407961×10⁻³** | \(1\times 10^{-8}\) Ha | libMBD free-atom isolated |
| \(E_{\mathrm{disp}}\) via `GpwDft` [Ha] | same pin | \(1\times 10^{-8}\) Ha | `res.e_disp` after short SCF (`max_iter=8`) |
| max\|F_product − F_fd\| [Ha/Bohr] | dual-gate | \(1\times 10^{-6}\) | \(h=10^{-5}\) independent FD |
| \(\|\sum_A \mathbf{F}_A\|\) [Ha/Bohr] | translation | \(1\times 10^{-8}\) | Isolated free molecule |

The \(10^{-6}\) Ha/Bohr dual-gate is dominated by FD truncation
(\(O(h^2)\) of product \(10^{-4}\) vs independent \(10^{-5}\)), not chemical
accuracy; the \(10^{-8}\) Ha/Bohr translation residual confirms no spurious
net force.

Offline notes (human only):

```text
E_disp = -1.4928804171407961e-03 Ha (libMBD git 88069c6)
max |F_product − F_fd(h=1e-5)| ≤ 1e-6 Ha/Bohr
|sum F| ≤ 1e-8 Ha/Bohr
# Optional host with libMBD C API: request do_gradients on same coords
```

Limitations: periodic Ewald / k-mesh forces use the same switch
(`mbd_k_mesh` non-zero) but are not dual-gated here; the full GPW force with
MBD is not gated by this section; Frame forces stay eV/Å while pins compare in
Ha/Bohr (`units::kEvPerAngstromInHartreePerBohr`).

## gpw-mbd-stress — Γ analytic cell stress with periodic MBD {#gpw-mbd-stress}

Periodic MBD (`mbd_k_mesh = {2,2,2}`) folded into the Γ analytic fixed-\(P\)
stress through `GPWEvaluator` on `cpu::Evaluate`:

```cpp
cpu::gpw::GpwDft dft{{O, H}, 200.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 120;  dft.tol = 1e-6;  dft.mix = 0.4;
dft.nonperiodic = false;
dft.mbd = true;
dft.mbd_k_mesh[0] = dft.mbd_k_mesh[1] = dft.mbd_k_mesh[2] = 2;
cpu::gpw::GPWEvaluator eval(std::move(dft));
eval.compute_forces = false;
eval.compute_stress = true;
eval.stress_use_fd = false;
eval.method().stress_strain_h = 2e-3;
cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);   // eval.last_stress_ha_bohr3(), eval.last_result().e_disp
```

| Gate | Oracle |
| ---- | ------ |
| \(E_{\mathrm{disp}}\) | offline pymbd `mbd_energy`, the pin of [`gpw-mbd-periodic`](#gpw-mbd-periodic) |
| \(\sigma\) | multi-SCF strain-FD (`gpw_stress_fd_impl`), full re-SCF per strain |
| MBD actually folded | \(\sigma(\mathrm{mbd}) \neq \sigma(\mathrm{plain})\) |

CP2K ships no MBD@rsSCS stress class, so there is no external stress twin.

### Measured

| Gate | Tol | Measured |
| ---- | --- | -------- |
| \(E_{\mathrm{disp}}\) vs pymbd | 1e−8 Ha | **1.85e−10** |
| \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF FD | 2e−5 Ha/Bohr³ | **7.39e−07** |
| MBD shift \(\lvert\sigma(\mathrm{mbd})-\sigma(\mathrm{plain})\rvert\) | > 1e−10 | **1.72e−07** |
| Off-diagonals (orthorhombic) | exactly 0 | **0** |
| Wall | ≤ 60 s | **23.3 s** |

Per axis: \(\sigma_{xx} = -2.185\times10^{-5}\), \(\sigma_{yy} = 2.007\times10^{-5}\),
\(\sigma_{zz} = 5.823\times10^{-5}\) Ha/Bohr³. A cheap MT H₂ analytic-vs-FD
check with MBD was recorded at 6.1e−9 Ha/Bohr³ (no unit test in the tree gates
it). For scale, the fixed-\(P\) pass takes 23.3 s versus 23.2 s for its own
multi-SCF FD twin on this small cell; the advantage grows with atom count.

## gpw-mbd-kpoint-forces — multi-k analytic forces + MBD {#gpw-mbd-kpoint-forces}

Monkhorst–Pack multi-k GPW ionic forces with free-atom MBD folded in by
`assemble_gpw_forces_kpoints`: the multi-k SCF supplies \(P_k\), \(W_k\); the
k-summed Hellmann–Feynman / Pulay assembly is that of
[gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces); \(F_{\mathrm{MBD}}\) from
`mbd_rsscs_pbe` is added with the same BC switch as Γ. Because free-atom MBD
does not change the density, after one shared SCF

\[
\mathbf{F}^{k+\mathrm{MBD}} - \mathbf{F}^{k,\mathrm{plain}}
  \approx \mathbf{F}_{\mathrm{MBD}}^{\mathrm{geom}}
\]

must hold to tight tolerance. The section has two parts.

### A — geometry-only MBD (water dimer, isolated free-atom)

| Quantity | Atomiverse / pin | Oracle | Tol | Notes |
| -------- | ---------------: | -----: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | −1.4928804171407961×10⁻³ | libMBD Fortran (git `88069c6`) | \(1\times 10^{-8}\) Ha | Hard gate A2 |
| max\|F_product − F_fd\| [Ha/Bohr] | ≤ 1×10⁻⁶ | independent FD dual-gate, \(h = 10^{-5}\) Bohr | \(1\times 10^{-6}\) | Hard gate A3 |
| \(\|\sum_A F_A\|\) [Ha/Bohr] | ≤ 1×10⁻⁸ | translation invariance | \(1\times 10^{-8}\) | Hard gate A4 |

### B — multi-k product wiring (H₂, k-mesh 1×1×2)

Compressed H₂ of [gpw-kpoint-forces](gpw-rks.md#gpw-kpoint-forces): H (5.5,
6.0, 6.0) and H (6.5, 6.0, 6.0) Bohr in a **12 × 12 × 12 Bohr** periodic cube,
`DZV-GTH-PADE` + `GTH-PADE-q1`, PADE, 100 Ry, RKS `n_occ = 1`, electronic
**k = 1×1×2**, free-atom MBD (`mbd_k_mesh` default zero).

```cpp
GpwDft cfg{{Kind{molrs::Element::H,
                 GaussianBasis(molrs::Element::H, "DZV-GTH-PADE"),
                 GthPotential(molrs::Element::H, "GTH-PADE-q1")}}, 100.0, XcKind::PADE};
cfg.n_occ = 1;
cfg.tol = 1e-8;  cfg.max_iter = 60;  cfg.eps_default = 1e-8;
cfg.nonperiodic = false;
cfg.local_pp_on_grid = true;
cfg.k_mesh[0] = 1;  cfg.k_mesh[1] = 1;  cfg.k_mesh[2] = 2;   // electronic multi-k
cfg.mbd = true;                                              // vs a plain twin
cfg.build(frame);
const auto& scf = cfg.forces(frame);   // kAtom_f* [eV/Å] = F_k-HF/Pulay + F_MBD(geom)
```

| Quantity | Gate | Tol | Notes |
| -------- | ---- | --- | ----- |
| SCF converged; multi-k payload | `n_kpts ≥ 1`, `C_occ_k` non-empty | hard | Γ path reports `n_kpts==0` |
| max\|F^{k+MBD} − F^{k,plain} − F_MBD^geom\| | ≤ 1×10⁻⁸ Ha/Bohr | \(1\times 10^{-8}\) | Hard gate A5 wiring |
| \|F_MBD,0x^geom\| nontrivial | > 1×10⁻⁸ Ha/Bohr | hard | avoids a trivial zero-force pass |
| `e_disp` on multi-k SCF | finite and &lt; 0 | hard | geometry-only materialization |

Not covered: UKS × multi-k × MBD forces; MBD stress (next section).

## gpw-mbd-kpoint-stress — multi-k analytic cell stress + periodic MBD {#gpw-mbd-kpoint-stress}

The k-summed fixed-\(P_k\) Lagrangian (`energy_fixed_P_k`) with periodic MBD
rebuilt at every strained cell: one multi-k SCF (`k_mesh = {1,1,2}`,
time-reversal folded), central FD of the non-self-consistent
\(E_L[\{P_k\},\{W_k\};\varepsilon]\) under orthorhombic diagonal strain, MBD on
its own 2×2×2 Ewald mesh.

| Parameter | Value | Unit |
| --------- | ----: | ---- |
| MBD k-mesh | 2×2×2 | — |
| Electronic k-mesh | 1×1×2 | Monkhorst–Pack |
| Strain step \(h\) | 2×10⁻³ | dimensionless |
| XC / cutoff | PBE, 120 Ry | — |
| SCF | max_iter 120, tol 1e-6, mix 0.4 | — |

```cpp
cpu::gpw::GpwDft dft{{O, H}, 120.0, cpu::gpw::XcKind::PBE};
dft.max_iter = 120;  dft.tol = 1e-6;  dft.mix = 0.4;
dft.nonperiodic = false;
dft.k_mesh[0] = 1;  dft.k_mesh[1] = 1;  dft.k_mesh[2] = 2;   // MP 1×1×2, TR-folded
dft.mbd = true;
dft.mbd_k_mesh[0] = dft.mbd_k_mesh[1] = dft.mbd_k_mesh[2] = 2;
cpu::gpw::GPWEvaluator eval(std::move(dft));
eval.compute_forces = false;
eval.compute_stress = true;
eval.stress_use_fd = false;
eval.method().stress_strain_h = 2e-3;
// mounted on cpu::Evaluate as in gpw-mbd-stress; Frame kStress [eV] (σ·Ω)
```

Free-atom MBD does not perturb the density, so per diagonal axis

\[
\sigma_{aa}(\mathrm{mbd}) - \sigma_{aa}(\mathrm{plain})
  = -\frac{\Delta E_{\mathrm{disp}}}{2h\,\Omega},
\]

with the right-hand side pinned by pymbd at each strained cell; the total
stress is checked against multi-SCF strain-FD (`gpw_stress_fd`, full re-SCF at
each strained cell on the same 1×1×2 mesh).

### pymbd pin lines (strain h = 2×10⁻³, fixed fractional coords)

```
E_disp unstrained            = -1.4965720704989138e-03 Ha
E_disp x +h              = -1.4925684743305467e-03 Ha
E_disp x -h              = -1.5005781658326001e-03 Ha
sigma_MBD_xx (FD, pymbd)    = -1.7171785426e-07 Ha/Bohr3
E_disp y +h              = -1.4965139646472281e-03 Ha
E_disp y -h              = -1.4966306893944292e-03 Ha
sigma_MBD_yy (FD, pymbd)    = -2.5024338482e-09 Ha/Bohr3
E_disp z +h              = -1.4957889518587564e-03 Ha
E_disp z -h              = -1.4973531298823062e-03 Ha
sigma_MBD_zz (FD, pymbd)    = -3.3534037337e-08 Ha/Bohr3
```

The unstrained value is bit-identical to the
[`gpw-mbd-periodic`](#gpw-mbd-periodic) pin.

### Gates and measured results

| Gate | Quantity | Oracle | Tol | Measured |
| ---- | -------- | ------ | --- | -------- |
| A | \(E_{\mathrm{disp}}\) multi-k vs Γ pin (k-mesh independence) | pymbd | 1×10⁻⁸ Ha | **1.07×10⁻¹¹** |
| B | \(E_{\mathrm{disp}}\) at 6 strained cells | fresh pymbd pins | 1×10⁻⁸ Ha | **1.78×10⁻¹⁵** |
| C | \(\sigma_{\mathrm{diag}}\) analytic vs multi-SCF strain-FD | `gpw_stress_fd` | 2×10⁻⁵ Ha/Bohr³ | **1.36×10⁻⁹** |
| D | \(\sigma(\mathrm{mbd})-\sigma(\mathrm{plain})\) vs pymbd FD term | pymbd | 2×10⁻⁹ Ha/Bohr³ | **4.00×10⁻¹²** |
| E | off-diagonals (orthorhombic) | exact | 0 | **0** |
| E | `kStress` eV/Å³ conversion | units | 1×10⁻⁹ | exact |

Measured diagonal stress (mbd, analytic, 120 Ry):
\(\sigma_{xx} = -3.622\times10^{-5}\), \(\sigma_{yy} = -8.570\times10^{-5}\),
\(\sigma_{zz} = -3.117\times10^{-6}\) Ha/Bohr³; MBD shifts per axis
(−1.7171×10⁻⁷, −2.500×10⁻⁹, −3.353×10⁻⁸ Ha/Bohr³) match the pymbd FD terms.

Matching tables and geometry track pymbd to femto-Hartree on the strained
cells. Analytic vs multi-SCF FD carries SCF re-convergence noise, hence the Γ
MBD-stress tier 2×10⁻⁵ Ha/Bohr³ (measured four orders better). Gate D is
bounded by pin round-off \(\le 2\cdot10^{-8}/(2h\Omega) \approx 4\times10^{-10}\)
plus SCF noise.

Wall: **13.6** s analytic (FD twin 22.0 s; SCF 11 iters, nk = 1) against the
≤60 s budget — a 4.6× margin.

Limitations: RKS only (no UKS × multi-k × MBD stress); diagonal
(orthorhombic) strain only — shear components are exactly zero for this cell,
not validated for general cells; the fixed-\(P_k\) Lagrangian FD is
non-self-consistent by design, bounded by the multi-SCF FD oracle.
