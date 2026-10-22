# GPW Tkatchenko–Scheffler dispersion

Tkatchenko–Scheffler (TS) pairwise London dispersion added to Γ-point GPW DFT
on top of PBE, in the free-atom limit (\(V/V_{\mathrm{free}}=1\)). The
correction is geometry-only: it never enters the Kohn–Sham potential, is
reported on `GpwResult::e_disp` [Hartree] and folded into `e_total`, and its
analytic ionic forces and cell stress are added to the electronic terms.

The switch is the bool member `ts` on `cpu::gpw::GpwDft` (no `*Config` bag).
The method runs directly (`gpw.build(frame)`, then `gpw.run(frame)` for the
energy or `gpw.forces(frame)` for SCF + analytic forces on `kAtom_f*` [eV/Å])
or mounted through `cpu::gpw::GPWEvaluator` on an `Evaluate` / `MD` /
`Minimize` force slot. `ts` is mutually exclusive with `d3`, `d3bj`, `d4`,
`mbd`, `xdm` and nonlocal vdW: any two set throws at run.

The sections cover the CPU backend only: periodic dispersion energy
(`gpw-ts-ice`), analytic ionic forces (`gpw-ts-forces`), and Γ analytic cell
stress (`gpw-ts-stress`, pins in the case `REFERENCE.md`: unstrained and
±\(h\)-strained `E_disp` against offline libMBD pins at 1e-8 Ha, analytic
fixed-\(P\) stress against the multi-SCF strain-FD oracle `gpw_stress_fd`
at 2e-5 Ha/Bohr³, and \(\sigma(\mathrm{ts})-\sigma(\mathrm{plain})\) against
the libMBD strain-FD dispersion term at 2e-9 Ha/Bohr³). Multi-k TS forces have
no section. Build in the CPU environment (`module load
buildenv-gcc/2026.03-mpich`, see `CLAUDE.md`) with `-DATV_BUILD_GPW=ON
-DATV_BUILD_REGRESSIONS=ON`.

Case: `ctest -R '^gpw/tkatchenko-scheffler$'` — sources and oracle record in [`regressions/gpw/tkatchenko-scheffler/`](../../../regressions/gpw/tkatchenko-scheffler/). One section of the case per `##` heading below: `atv_regressions --case gpw/tkatchenko-scheffler --section <name>`.

## Method

Tkatchenko–Scheffler free-atom pairwise dispersion (Tkatchenko & Scheffler,
*Phys. Rev. Lett.* **102**, 073005 (2009)), matching libMBD's
`get_ts_energy` / `cmbd_ts_energy` real-space path with **periodic** lattice
sums over image vectors \(\mathbf{T}\):

\[
E_{\mathrm{disp}} = -\sum_{i\le j}\sum_{\mathbf{T}}{}^{\prime}
  C_{6,ij}\,f_{\mathrm{damp}}(R)\,R^{-6},
\]

\[
C_{6,ij}=\frac{2\,C_{6,i}C_{6,j}}
  {(\alpha_j/\alpha_i)\,C_{6,i}+(\alpha_i/\alpha_j)\,C_{6,j}},
\quad
R_{0,ij}=R_{\mathrm{vdW},i}+R_{\mathrm{vdW},j},
\]

\[
f_{\mathrm{damp}}=\bigl(1+\exp[-d\,(R/(s_R R_{0,ij})-1)]\bigr)^{-1}.
\]

The prime excludes the singular \(R=0\) self term. Free-atom \(\alpha_0\),
\(C_6\), \(R_{\mathrm{vdW}}\) come from libMBD `ts_vdw_params` (in-tree
`basis_data/ts_tables.h`):

| Z | \(\alpha_0\) [Bohr³] | \(C_6\) [Ha·Bohr⁶] | \(R_{\mathrm{vdW}}\) [Bohr] |
| - | -------------------: | -----------------: | -------------------------: |
| H | 4.5 | 6.5 | 3.1 |
| O | 5.4 | 15.6 | 3.19 |

**PBE-TS parameters (binding):**

| Parameter | Value | Unit / notes |
| --------- | ----: | ------------ |
| \(s_R\) | 0.94 | PBE range scaling |
| \(d\) | 20 | Fermi steepness |
| Pair cutoff (periodic) | **50 Å ≈ 94.561 Bohr** | pure real-space orthorhombic lattice sum, no Ewald (kernel default 94.2 Bohr) |
| Volume ratios | all 1 | free-atom limit |
| XC host | PBE | geometry-only disp; short Γ-RKS SCF materializes `e_disp` |

**Volume ratios.** Optional per-atom \(v = V/V_{\mathrm{free}}\) scale
\(\alpha\leftarrow\alpha_{\mathrm{free}}v\),
\(C_6\leftarrow C_{6,\mathrm{free}}v^2\),
\(R\leftarrow R_{\mathrm{free}}v^{1/3}\). Empty ratios ⇒ free-atom limit
(all \(v=1\)), which every section pins. Hirshfeld density volumes
(`GpwDft::hirshfeld`) are a separate product path
([gpw-mbd-hirshfeld](gpw-mbd.md#gpw-mbd-hirshfeld)).

**Lattice sum.** libMBD's Ewald path on the same cell differs from the pure
real-space sum by \({\sim}7\times 10^{-8}\) Ha (long-range \(1/R^6\) tail);
the hard gate is the pure-RS value.

**Electronic host.** Dispersion is geometry-only; each section still runs a
short Γ-RKS GPW (PBE, `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q*`) so `e_disp` is
materialized on a real `GpwResult` and included in `e_total`. Absolute QS
\(E_{\mathrm{tot}}\) micro-Ha parity vs CP2K is **out of scope**
(OT/grid/DIIS differences; CP2K has no stock TS).

## Common setup

| Item | Value |
| ---- | ----- |
| System | Water dimer (ice-relevant H-bond pair; not bulk ice Ih) — the same cell as [gpw-pbe-d3bj-ice](gpw-d3.md#gpw-pbe-d3bj-ice) / [gpw-d4-water](gpw-d4.md#gpw-d4-water) / [gpw-mbd-forces](gpw-mbd.md#gpw-mbd-forces) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) (E_disp / F_TS basis-independent) |
| XC + disp | PBE host + free-atom TS (\(s_R=0.94\), \(d=20\), rcut 50 Å) |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| Å→Bohr | `1/0.52917720859` (ATV) |

Mounted on a driver (same member):

```cpp
// kinds = {O, H} Kind{Element, GaussianBasis, GthPotential}
ATV::cpu::gpw::GpwDft dft{kinds, 300.0, ATV::cpu::gpw::XcKind::PBE};
dft.max_iter = 200;  dft.tol = 1e-7;  dft.mix = 0.3;
dft.nonperiodic = false;
dft.ts = true;
ATV::cpu::gpw::GPWEvaluator eval(std::move(dft), /*compute_forces=*/true);
// bind on an Evaluate / MD / Minimize force slot
// eval.last_result().e_disp  [Hartree]
// frame.span(kAtom_f*) [eV/Å] includes the TS contribution
```

| Path | Role |
| ---- | ---- |
| `gpw_solver.h` | `GpwDft`, `GpwResult::e_disp`, `ts` |
| `gpw_evaluator.h` | `GPWEvaluator` (binds the `GpwDft`; `compute_forces`) |
| `ts_dispersion.h` | `ts_pbe`, `TsResult` (private kernel) |
| `ts_tables.h` | Free-atom libMBD `ts_vdw_params` |
| `src/cpu/gpw/internal/gpw_forces.cpp` | Product force assembly folds TS |

No unit test in the tree covers TS (isolated libMBD, periodic pure-RS, FD
forces, wiring).

## Oracle

CP2K 2026.1 stock `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — there is **no** `TYPE TS`. The oracle is
offline **libMBD** (git `88069c6`), the canonical TS/MBD library and the
backend FHI-aims / Quantum ESPRESSO / DFTB+ use, plus a dual pure-Python
lattice sum / Atomiverse `ts_pbe` for the periodic value. External engine
pins — not goldens of Atomiverse's own output; live libMBD or CP2K is
**never** launched from ctest.

| Quantity | Value [Ha] | Source |
| -------- | ---------: | ------ |
| \(E_{\mathrm{disp}}\) isolated dimer | −5.9764872295816794×10⁻⁴ | libMBD `cmbd_ts_energy` (ATV Å→Bohr); `ts_pbe` matches to \({\sim}10^{-19}\) Ha |
| \(E_{\mathrm{disp}}\) periodic pure-RS (hard pin) | −6.099630773386013×10⁻⁴ | dual lattice sum / `ts_pbe`, rcut 50 Å |
| \(E_{\mathrm{disp}}\) libMBD Ewald (context) | −6.100334450881942×10⁻⁴ | Δ ≈ \(7\times 10^{-8}\) vs pure-RS; not the gate |

Offline re-pin (human only; never ctest):

```text
# Build libmbd (once); pin isolated free-atom TS on the water dimer
# Isolated: Dispersion energy = -5.9764872295816794e-04 Eh  (libMBD cmbd_ts_energy, ATV Å→Bohr)
# Periodic pure-RS hard gate: -6.099630773386013e-04 Ha  (rcut 50 Å)
# Forces: product ∇E vs independent FD; optional libMBD C gradients on host
```

## Performance

TS evaluation is \(O(N^2 \times N_{\mathrm{images}})\) over a 6-atom cell
(microseconds); full GPW SCF wall is dominated by the electronic structure.
A fair \(r=T_{\mathrm{ATV}}/T_{\mathrm{CP2K}}\) ratio is unavailable (no
stock CP2K `TYPE TS`), a dispersion-only microbenchmark vs the libMBD CLI is
not the product gate, and a full twin SCF ratio would be unfair for a
geometry-only gate. Wall is **perf_exempt** (same policy as
[gpw-pbe-d3bj-ice](gpw-d3.md#gpw-pbe-d3bj-ice) /
[gpw-d4-water](gpw-d4.md#gpw-d4-water) /
[gpw-mbd-forces](gpw-mbd.md#gpw-mbd-forces) /
[gpw-xdm-forces](gpw-xdm.md#gpw-xdm-forces)); a strict full-SCF twin applies
only on a host with an external TS-capable engine, if product policy requires
\(r \le 10\).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----: | ---- | ------ |
| Full QS ENERGY / ENERGY_FORCE | n/a (CP2K lacks stock TS) | — (not fair-twinned) | **perf_exempt** | — | — |
| E_disp / force dual-gate | n/a | short SCF + TS | **perf_exempt** | — | Geometry-only deliverable + wiring |

## Limitations

- **Cell is a water dimer**, not bulk ice Ih — sufficient for periodic TS
  energy/force parity; not a structural ice benchmark (no ice Ih supercell,
  no phonons).
- **Absolute GPW \(e_{\mathrm{tot}}\) vs CP2K QS** is not asserted (CP2K has
  no stock TS).
- **Free-atom volumes only** — no Hirshfeld \(V/V_{\mathrm{free}}\) from the
  density on these pins.
- **Pairwise only** — no MBD many-body screening
  ([gpw-mbd-ice](gpw-mbd.md#gpw-mbd-ice) /
  [gpw-mbd-forces](gpw-mbd.md#gpw-mbd-forces)); no vdW-DF nonlocal
  functionals.
- **Pure real-space lattice sum** — not Ewald; documented nano-Ha offset vs
  libMBD Ewald on this cell.
- **PBE damping parameters only** (\(s_R=0.94\)); other functionals' \(s_R\)
  are not e2e-gated.
- **Oracle split:** the TS pin is libMBD / pure-RS; D3(BJ) is CP2K; D4 is
  dftd4 — do not mix oracles when comparing.
- No UKS and no CUDA GPW.

## References

- A. Tkatchenko, M. Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009)
  (TS dispersion)
- libMBD (canonical TS/MBD library; free-atom `ts_vdw_params` tables,
  `cmbd_ts_energy` energy twin)
- Roadmap: `docs/roadmap/dft.md` §8 Dispersion
- Related GPW dispersion: [gpw-pbe-d3bj-ice](gpw-d3.md#gpw-pbe-d3bj-ice),
  [gpw-d4-water](gpw-d4.md#gpw-d4-water),
  [gpw-d4-forces](gpw-d4.md#gpw-d4-forces),
  [gpw-xdm-forces](gpw-xdm.md#gpw-xdm-forces)
- GPW analytic forces (no disp): [gpw-forces-water](gpw-rks.md#gpw-forces-water)
- GPW narrative chapter: `docs/tutorial/ch09-gpw.md`

## gpw-ts-ice — periodic water-dimer TS dispersion energy {#gpw-ts-ice}

Gates the periodic **dispersion energy** of the water-dimer cell against the
pure-RS pin; `e_disp` is folded into `e_total`.

```cpp
frame.box = ATV::cpu::Box::cubic(12.0);  // Å

GpwDft gpw{kinds, 200.0, XcKind::PBE};  // e_disp is geometry-only; SCF only needs to run
gpw.n_occ = 8;         // 16 valence e⁻ → 8 doubly occupied
gpw.max_iter = 8;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.ts = true;         // exclusive vs gpw.d3 / d3bj / d4

gpw.build(frame);
const auto& res = gpw.run(frame);
// res.e_disp [Ha] within 1e-8 of the offline pure-RS / libMBD-class pin
// res.e_total includes e_disp
```

### Validation

| Quantity | Atomiverse / pin | Oracle (offline) | Tol | Notes |
| -------- | ---------------: | ----------------: | --- | ----- |
| \(E_{\mathrm{disp}}\) periodic pure-RS [Ha] | −6.099630773386013×10⁻⁴ | −6.099630773386013×10⁻⁴ | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) isolated [Ha] | (unit test `ts_pbe`) | −5.9764872295816794×10⁻⁴ (libMBD) | \({\sim}10^{-19}\) Ha match | Unit hard pin |
| \(E_{\mathrm{disp}}\) libMBD Ewald [Ha] | (context) | −6.100334450881942×10⁻⁴ | — | Not the gate |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | (CP2K stock TS N/A) | — | Not absolute-parity gated |

**Tolerance rationale.** Matching geometry, free-atom tables, PBE Fermi
damping, and pure-RS rcut 50 Å track the dual lattice-sum pin to
nano-Hartree. Isolated libMBD and Atomiverse `ts_pbe` agree to machine
epsilon. Unit tests also check analytic TS forces vs central finite
difference and
\(e_{\mathrm{total}}(\mathrm{ts}) - e_{\mathrm{total}}(\mathrm{off}) = e_{\mathrm{disp}}\).

## gpw-ts-forces — free-atom TS ionic forces {#gpw-ts-forces}

Gates both the **dispersion energy** and the **analytic ionic forces**
\(F_A = -\partial E_{\mathrm{disp}}/\partial R_A\) [Ha/Bohr].
`GpwDft::forces(frame)` with `ts = true` **adds** these geometry-only forces
to the electronic Hellmann–Feynman / Pulay / grid-core terms.

The force oracle is a **dual-gate**: product analytic
\(\nabla E_{\mathrm{disp}}\) vs independent central finite differences
(\(h=10^{-5}\) Bohr) of the libMBD-matched energy on the isolated dimer, plus
a **product wiring** check: \(F(\mathrm{ts})-F(\mathrm{plain})\) on the same
density matches standalone periodic `ts_pbe` forces.

```cpp
// Isolated free-atom TS energy + analytic forces (libMBD pin dual-gate)
const double cell0[3] = {0.0, 0.0, 0.0};
const auto ts_iso = ts_pbe(Z, pos, cell0, /*periodic=*/false);
// |ts_iso.energy + 5.9764872295816794e-4| ≤ 1e-8 Ha
// max |F − FD(h=1e-5)| ≤ 1e-6 Ha/Bohr; |∑F| ≤ 1e-8 Ha/Bohr

// Periodic pure-RS pin (rcut 50 Å)
const double L = 12.0 * kA2B;            // kA2B = 1.0 / 0.52917720859
const double cell[3] = {L, L, L};
const auto ts_per = ts_pbe(Z, pos, cell, /*periodic=*/true, 50.0 * kA2B);
// |ts_per.energy + 6.099630773386013e-4| ≤ 1e-8 Ha

GpwDft gpw{kinds, 200.0, XcKind::PBE};
gpw.n_occ = 8;
gpw.max_iter = 80;
gpw.tol = 1e-6;
gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.local_pp_on_grid = true;
gpw.ts = true;         // exclusive vs d3 / d3bj / d4 / mbd / xdm / …

gpw.build(frame);
const auto& res = gpw.forces(frame);  // SCF + analytic forces → kAtom_f* [eV/Å]
// res.e_disp [Ha] within 1e-8 of the periodic pin; folded into res.e_total
// F(ts) − F(plain), converted to Ha/Bohr (÷ units::kHartreePerBohrInEvPerAngstrom),
// matches ts_per.forces within 1e-7 Ha/Bohr
```

### Validation

| Quantity | Atomiverse / pin | Offline source | Tol | Notes |
| -------- | ---------------: | -------------- | --- | ----- |
| \(E_{\mathrm{disp}}\) isolated [Ha] | −5.9764872295816794×10⁻⁴ | libMBD `cmbd_ts_energy` | \(1\times 10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) periodic [Ha] | −6.099630773386013×10⁻⁴ | pure-RS / `ts_pbe` ([gpw-ts-ice](#gpw-ts-ice)) | \(1\times 10^{-8}\) Ha | Hard gate; also `GpwDft::e_disp` |
| max\|F_product − F_fd\| [Ha/Bohr] | ≤ \(1\times 10^{-6}\) | FD \(h=10^{-5}\) Bohr of \(E_{\mathrm{disp}}\) | \(1\times 10^{-6}\) | Isolated dual-gate |
| \(\|\sum_A\mathbf{F}_A\|\) [Ha/Bohr] | ≤ \(1\times 10^{-8}\) | translation | \(1\times 10^{-8}\) | Isolated |
| max\|F(ts)−F(plain)−F_geom\| [Ha/Bohr] | ≤ \(1\times 10^{-7}\) | product wiring | \(1\times 10^{-7}\) | Same density; short host SCF |
| \(E_{\mathrm{tot}}\) QS [Ha] | finite; includes \(e_{\mathrm{disp}}\) | (no CP2K TS twin) | — | Not absolute-parity gated |

**Tolerance rationale.** Force agreement at \(10^{-6}\) Ha/Bohr is the
\(O(h^2)\) residual of central FD with \(h=10^{-5}\) Bohr on a smooth
pairwise energy — not chemical force accuracy vs a QS twin. Wiring at
\(10^{-7}\) Ha/Bohr asserts that force assembly adds exactly the
geometry-only TS contribution without double-counting or omission.

### Limitations

- The force pin is analytic-vs-FD of ATV's (libMBD-matched) energy — not a
  CP2K FORCE_EVAL absolute force table.
- k-point force assembly is not the gate here (see
  [gpw-ts-kpoint-forces](#gpw-ts-kpoint-forces)).
- Frame forces are eV/Å on the public path; the dual-gate comparison uses
  Ha/Bohr.

## gpw-ts-kpoint-forces — multi-k free-atom TS ionic forces {#gpw-ts-kpoint-forces}

*No section of the case runs this variant.*

**Monkhorst–Pack multi-k GPW** ionic forces with additive free-atom TS
(`GpwDft::ts` + `k_mesh`, `GpwDft::forces(frame)` or
`GPWEvaluator(std::move(dft), /*compute_forces=*/true)`; roadmap
`docs/roadmap/dft.md` §8 Dispersion + §10 Periodic KS). Γ folds TS into
`assemble_gpw_forces` ([gpw-ts-forces](#gpw-ts-forces)); multi-k stress is the
matching `stress(frame)` path. On the multi-k force path the same
geometry-only

\[
F_{\mathrm{TS}} = -\nabla E_{\mathrm{disp}}^{\mathrm{TS}}
\]

is added in `assemble_gpw_forces_kpoints` when `ts` is set:

\[
\mathbf{F}_A^{\mathrm{tot}}
  = \mathbf{F}_A^{\mathrm{k\text{-}HF/Pulay}}
  + \mathbf{F}_A^{\mathrm{TS}},
\quad
\mathbf{F}_A^{\mathrm{TS}}
  = -\nabla_{R_A} E_{\mathrm{disp}}^{\mathrm{TS}}
  \quad (V/V_{\mathrm{free}}=1).
\]

Its pins are the offline libMBD \(E_{\mathrm{disp}}\) (water dimer
free-atom) + the force dual-gate + multi-k **wiring** on compressed H₂, with
the PBE-TS parameters of [Method](#method) (\(s_R\) 0.94, Fermi \(d\) 20,
periodic pair cutoff 50 Å pure real-space). Wall is **perf_exempt** (no stock
CP2K TS twin).

```cpp
cpu::gpw::GpwDft gpw{kinds, 300.0, cpu::gpw::XcKind::PADE};   // one Kind per element
gpw.k_mesh[0] = 1; gpw.k_mesh[1] = 1; gpw.k_mesh[2] = 2;
gpw.ts = true;
gpw.build(frame);                     // positions [Å], Z in kAtom_type, box [Å]
const auto& scf = gpw.forces(frame);  // SCF + analytic forces → kAtom_f* [eV/Å]
// scf.e_disp [Ha] is geometry-only TS
```
