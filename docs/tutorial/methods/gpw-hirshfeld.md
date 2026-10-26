# GPW Hirshfeld-scaled MBD@rsSCS

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

Hirshfeld volume ratios \(v_A = V_A/V_A^{\mathrm{free}}\), partitioned from the
converged GPW valence density, rescale the free-atom polarizabilities, \(C_6\)
coefficients and van der Waals radii of the Tkatchenko–Scheffler family before
the MBD@rsSCS (or TS) dispersion energy is evaluated. The density can come from
a Γ-point or a Monkhorst–Pack multi-k SCF, restricted or collinear
unrestricted; the Hirshfeld partition always uses the **total** valence
density.

The method runs through the concrete `cpu::gpw::GpwDft`: one `Kind` per element
built from the lookup constructors `GaussianBasis` / `GthPotential`, `cutoff_ry`
and `xc` as constructor arguments, then the members `mbd` (or `ts`),
`hirshfeld`, `k_mesh` / `k_time_reversal`, `n_occ` or `n_alpha` / `n_beta`, and
the SCF knobs; `build(frame)` → `run(frame)`. Dispersion lands on
`GpwResult::e_disp` [Hartree] and is folded into `e_total`; per-atom ratios on
`GpwResult::volume_ratios`. Driver work binds the same configured `GpwDft`
through `cpu::gpw::GPWEvaluator`, which carries no method parameters of its
own. All sections are CPU-only and gate the dispersion energy: Γ-UKS O₂,
multi-k RKS water dimer, multi-k UKS O₂. The case additionally carries the
force and stress sections `gpw-hirshfeld-forces`,
`gpw-hirshfeld-kpoint-forces` and `gpw-hirshfeld-kpoint-stress`, whose oracle
record lives in the case `REFERENCE.md`.

Case: `ctest -R '^gpw/hirshfeld$'` — sources and oracle record in [`regressions/gpw/hirshfeld/`](../../../regressions/gpw/hirshfeld/). One section of the case per `##` heading below: `atv_regressions --case gpw/hirshfeld --section <name>`.

## Common setup

### Method

**GPW host (Quickstep family).** Contracted Gaussian AOs; density on a uniform
grid; FFT Poisson Hartree; XC on the grid; GTH pseudopotentials (Lippert,
Hutter, Parrinello 1997).

**Hirshfeld effective volume** (Tkatchenko & Scheffler 2009) on the uniform
orthorhombic GPW density grid:

\[
w_A(\mathbf{r})
  = \frac{\rho_A^{\mathrm{free}}(|\mathbf{r}-\mathbf{R}_A|)}
         {\sum_B\rho_B^{\mathrm{free}}(|\mathbf{r}-\mathbf{R}_B|)},
\quad
V_A = \int w_A(\mathbf{r})\,\rho(\mathbf{r})\,|\mathbf{r}-\mathbf{R}_A|^3\,
  d^3r,
\quad
v_A = V_A / V_A^{\mathrm{free}}.
\]

Free-atom densities use the in-tree exponential model of
`basis_data/hirshfeld_tables.h`:

\[
\rho_Z^{\mathrm{free}}(r)=A\exp(-\alpha_Z r),\quad
A=N_{\mathrm{val}}\alpha_Z^3/(8\pi),\quad
V_Z^{\mathrm{free}}=60\,N_{\mathrm{val}}/\alpha_Z^3
\]

(oxygen: \(N=6\), \(\alpha=2.2\) Bohr⁻¹). Distances are minimum-image when the
cell is periodic. Open-shell runs partition
\(\rho=\rho_\alpha+\rho_\beta\) (no spin-resolved volumes); multi-k runs
partition the k-summed density \(\rho=\sum_k w_k\rho_k\), once after the SCF
(not per k).

**Density-scaled MBD@rsSCS** (Ambrosetti et al. 2014). Free-atom
\(\alpha_0\) / \(C_6\) / \(R_{\mathrm{vdW}}\) from the libMBD tables scale as

\[
\alpha \propto v,\qquad C_6 \propto v^2,\qquad R \propto v^{1/3},
\]

followed by range-separated self-consistent screening MBD (PBE \(\beta=0.83\)).
With the default `mbd_k_mesh = {0,0,0}` MBD is **isolated** on the nuclear
coordinates — only the volumes come from the (possibly multi-k) density.
Periodic Ewald MBD is [`gpw-mbd-periodic`](gpw-mbd.md#gpw-mbd-periodic).

### API contract

- `mbd=true`, `hirshfeld=true` → post-SCF volumes + scaled MBD → `e_disp`,
  `volume_ratios`.
- `mbd=true`, `hirshfeld=false` → free-atom volumes (\(v=1\)); geometry-only
  MBD (same tables as [`gpw-mbd-ice`](gpw-mbd.md#gpw-mbd-ice)).
- `hirshfeld=true` without `ts` or `mbd` throws
  (`"hirshfeld requires ts or mbd"`).
- `ts` / `mbd` are mutually exclusive with D3 / D3(BJ) / D4 / XDM / vdW-DF*;
  `hirshfeld` does not apply to those.

### Oracle

Stock **CP2K 2026.1** `PAIR_POTENTIAL TYPE` is only
`DFTD2|DFTD3|DFTD3(BJ)|DFTD4` — no TS/MBD. Every hard gate therefore freezes the
**ATV product** under fixed settings (offline pin; never live CP2K / libMBD
in ctest), cross-checked against free-atom MBD (tables pinned against libMBD
Fortran on [`gpw-mbd-ice`](gpw-mbd.md#gpw-mbd-ice)) and the dual in-tree Γ
Hirshfeld kernel ([`gpw-mbd-hirshfeld`](gpw-mbd.md#gpw-mbd-hirshfeld)).
No unit test in the tree gates the Hirshfeld kernel; the regression sections
are the only gates. Build with
the standard CPU environment (`buildenv-gcc/2026.03-mpich`) and
`-DATV_BUILD_GPW=ON -DATV_BUILD_REGRESSIONS=ON` (GPW needs FFTW3).

**Performance: perf_exempt** for every section. There is no stock CP2K TS/MBD
twin, so neither a full QS wall ratio nor a dispersion-only microbenchmark is
available; the primary deliverable is \(E_{\mathrm{disp}}\) + \(v_A\). Hirshfeld
grid work is a single post-SCF \(O(N_{\mathrm{grid}} N_{\mathrm{atoms}})\)
integral and MBD is \(O(N^3)\) over a handful of atoms, so wall is dominated by
the SCF.

### Shared limitations

- **Product pin, not CP2K QS bit-identity**; absolute QS
  \(E_{\mathrm{tot}}\) micro-Ha parity (grid/DIIS vs CP2K multigrid) is out of
  scope — only \(E_{\mathrm{disp}}\) is asserted.
- **Total-density Hirshfeld only:** \(\rho_\alpha-\rho_\beta\) enters \(v_A\)
  only through the converged total \(\rho\).
- **Geometry hygiene:** off-centre molecules under Martyna–Tuckerman corrupt
  the density tails and thus the volumes; centre isolated pins.
- No CUDA GPW; analytic dispersion forces are not the gate of these sections.

### References

- A. Tkatchenko, M. Scheffler, *Phys. Rev. Lett.* **102**, 073005 (2009) — TS / Hirshfeld volumes
- A. Ambrosetti et al., *J. Chem. Phys.* **140**, 18A508 (2014) — MBD@rsSCS
- G. Lippert, J. Hutter, M. Parrinello, *Mol. Phys.* **92**, 477 (1997) — GPW
- G. J. Martyna, M. E. Tuckerman, *J. Chem. Phys.* **110**, 2810 (1999) — MT Poisson
- H. J. Monkhorst, J. D. Pack, *Phys. Rev. B* **13**, 5188 (1976) — k-mesh
- Narrative GPW: [`docs/tutorial/ch09-gpw.md`](../ch09-gpw.md)
- Roadmap: [`docs/roadmap/dft.md`](../../roadmap/dft.md) §8 Dispersion

## gpw-hirshfeld-uks — Γ-UKS O₂ / PBE / DZVP-MOLOPT-GTH {#gpw-hirshfeld-uks}

Triplet O₂ under Martyna–Tuckerman isolation, Γ-point UKS PBE, same centred
geometry as `gpw-uks-pbe` /
[`gpw-uks-o2-gamma`](gpw-uks.md#gpw-uks-o2-gamma), with post-SCF MBD on the
UKS total density instead of free-atom volumes.

### System and settings

| Item | Value |
| ---- | ----- |
| Molecule | O₂, bond along *z*, centred; \(R(\mathrm{O–O})\) = **1.2075 Å** |
| Coordinates (Å) | O (6, 6, 5.39625), O (6, 6, 6.60375) |
| Cell | **12 × 12 × 12 Å**, `PERIODIC NONE`, Martyna–Tuckerman (`nonperiodic`) |
| Basis / PP / XC | DZVP-MOLOPT-GTH / GTH-PBE-q6 / PBE |
| Cutoff | **200 Ry** |
| Spin | UKS, 12 valence e⁻, multiplicity 3 → **\(N_\alpha=7\), \(N_\beta=5\)** |
| Dispersion | `mbd=true`, `hirshfeld=true` (isolated MBD) |
| SCF | max_iter 150, tol 1e-6, mix 0.35, diis_depth 10 |

```cpp
constexpr double L = 12.0;   // Å
constexpr double R = 1.2075; // Å
cpu::Frame frame(2);
for (int i = 0; i < 2; ++i) {
    frame.span(kAtom_type)[i] = 8;
    frame.span(kAtom_x)[i] = 0.5 * L;
    frame.span(kAtom_y)[i] = 0.5 * L;
}
frame.span(kAtom_z)[0] = 0.5 * L - 0.5 * R;
frame.span(kAtom_z)[1] = 0.5 * L + 0.5 * R;
frame.box = cpu::Box::cubic(L);

GpwDft gpw{{Kind{molrs::Element::O,
                 GaussianBasis(molrs::Element::O, "DZVP-MOLOPT-GTH"),
                 GthPotential(molrs::Element::O, "GTH-PBE-q6")}}, 200.0, XcKind::PBE};
gpw.n_alpha = 7;  gpw.n_beta = 5;
gpw.max_iter = 150;  gpw.tol = 1e-6;  gpw.mix = 0.35;  gpw.diis_depth = 10;
gpw.nonperiodic = true;
gpw.mbd = true;
gpw.hirshfeld = true;
gpw.build(frame);
const auto& res = gpw.run(frame);   // res.e_disp [Ha], res.volume_ratios, res.s2
```

### Validation

| Quantity | Pin / gate | Tol | Notes |
| -------- | ---------: | --- | ----- |
| \(E_{\mathrm{disp}}\) [Ha] | **−1.0379964698969957×10⁻⁴** | **1×10⁻⁸ Ha** | PBE-MBD@rsSCS, Hirshfeld \(v_A\), CUTOFF 200 Ry |
| \(v\) (O, O) | ≈ **0.6666, 0.6666** | physical bounds (0.2, 2.0) | compressed vs free-atom \(v=1\) |
| SCF | `res.converged` | — | `max_iter=150`, `tol=1e-6` |
| \(E_{\mathrm{disp}} < 0\) | attractive | — | Hirshfeld reduces \|E_disp\| vs free-atom MBD on the same O₂ |
| Closed-shell UKS≈RKS (recorded check, not gated) | water dimer E_disp | 5×10⁻⁶ Ha | Nα=Nβ path |

Löwdin \(\langle S^2\rangle\) is reported on the result but is not gated here.

### Limitations

- **CUTOFF 200 Ry** is the pin setting (faster than the 400 Ry UKS-PADE energy
  foothold); retuning the cutoff changes both density and \(v_A\) slightly —
  re-pin before tightening the tolerance.
- Nuclear MBD is non-periodic even though the SCF cell is MT-isolated.
- Not covered: spin-projected volumes; open-shell TS+Hirshfeld forces as a
  regression gate.

## gpw-hirshfeld-kpoints — multi-k RKS water dimer {#gpw-hirshfeld-kpoints}

Extends Γ-only Hirshfeld ([`gpw-mbd-hirshfeld`](gpw-mbd.md#gpw-mbd-hirshfeld))
to the k-summed density path: multi-k RKS GPW PBE → \(\rho=\sum_k w_k\rho_k\) →
the same Hirshfeld kernel and libMBD tables. With time-reversal fold the
\(2\times2\times2\) mesh yields **4** irreducible k-points on this orthorhombic
cell (`n_kpts = 4`).

### System and settings

| Item | Value |
| ---- | ----- |
| System | Water dimer (same geometry as `gpw-mbd-ice` / `gpw-mbd-hirshfeld` / `gpw-ts-ice`) |
| Geometry [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Cell | ABC **12.0 12.0 12.0 Å**, PERIODIC XYZ |
| k-mesh | Monkhorst–Pack **2×2×2**, time-reversal → **4** irreducible k |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H) |
| XC + disp | PBE + MBD@rsSCS with Hirshfeld \(v_A\) from multi-k \(\rho\) |
| CUTOFF | **200 Ry** |
| Occupations | RKS, 16 valence e⁻ → `n_occ = 8` |
| SCF | max_iter 16, tol \(1\times10^{-6}\), mix 0.4 |

```cpp
GpwDft gpw{{O, H}, 200.0, XcKind::PBE};   // Kind O / H as in the table
gpw.n_occ = 8;
gpw.max_iter = 16;  gpw.tol = 1e-6;  gpw.mix = 0.4;
gpw.nonperiodic = false;
gpw.mbd = true;
gpw.hirshfeld = true;
gpw.k_mesh[0] = gpw.k_mesh[1] = gpw.k_mesh[2] = 2;
gpw.k_time_reversal = true;
gpw.build(frame);
const auto& res = gpw.run(frame);   // res.n_kpts; res.e_disp within 1e-8 of the pin
```

### Validation

| Quantity | Atomiverse / pin | Cross-check | Tol | Notes |
| -------- | ---------------: | ----------: | --- | ----- |
| \(E_{\mathrm{disp}}\) multi-k Hirshfeld-MBD [Ha] | **−1.1479849553026611×10⁻³** | dual product gate | \(1\times10^{-8}\) Ha | Hard gate |
| \(E_{\mathrm{disp}}\) Γ Hirshfeld-MBD [Ha] | −1.1479852469005181×10⁻³ | `gpw-mbd-hirshfeld` | context | \(\Delta\sim3\times10^{-10}\) Ha on this cell |
| \(E_{\mathrm{disp}}\) free-atom MBD [Ha] | −1.4928804171407961×10⁻³ | `gpw-mbd-ice` | context | \(v_A=1\) (no density scale) |
| \(v_A\) (O H H O H H) | ≈ 0.7403, 0.6529, 0.6529, 0.7153, 0.5784, 0.5784 | range (0.3, 1.5) | qualitative | 6 entries on `volume_ratios` |
| \(n_{\mathrm{kpts}}\) | ≥ 2 (4 with TR) | multi-k path | integer | Fails if Γ-only |

On the 12 Å molecular cell the multi-k vs Γ density difference is tiny, so
\(E_{\mathrm{disp}}\) nearly matches the Γ Hirshfeld pin; the gate freezes the
multi-k path itself (k-summed \(\rho\) → Hirshfeld → MBD), not a large
numerical shift. Free-atom MBD is more attractive (no volume contraction).

### Limitations

- The cell is a water dimer, not bulk ice — enough to gate multi-k density →
  Hirshfeld → MBD, not a condensed-phase structural benchmark.

## gpw-hirshfeld-uks-kpoints — multi-k UKS O₂ / PBE / DZVP-MOLOPT-SR-GTH {#gpw-hirshfeld-uks-kpoints}

Multi-k collinear UKS: dual complex Bloch densities \(P_k^\alpha\),
\(P_k^\beta\); the dispersion fold (`fold_dispersion_k_uks`) partitions the
k-summed total density \(\rho=\sum_k w_k(\rho_k^\alpha+\rho_k^\beta)\) and
scales isolated MBD@rsSCS (`mbd_k_mesh={0,0,0}`). Triplet O₂ (\(R\) = 1.2075 Å,
centred) on a **10 Å** periodic cube, Monkhorst–Pack **2×2×1** with time
reversal → 2 irreducible k (2×1×1 would fold to Γ). Basis
DZVP-MOLOPT-SR-GTH + GTH-PBE-q6, PBE, CUTOFF 120 Ry, `n_alpha = 7`,
`n_beta = 5`, SCF max_iter 80, tol 1e-6, mix 0.45, diis_depth 10.

```cpp
GpwDft gpw{{O}, 120.0, XcKind::PBE};   // DZVP-MOLOPT-SR-GTH + GTH-PBE-q6
gpw.n_alpha = 7;  gpw.n_beta = 5;
gpw.max_iter = 80;  gpw.tol = 1e-6;  gpw.mix = 0.45;  gpw.diis_depth = 10;
gpw.mbd = true;
gpw.hirshfeld = true;
gpw.k_mesh[0] = 2;  gpw.k_mesh[1] = 2;  gpw.k_mesh[2] = 1;
gpw.k_time_reversal = true;
```

### Validation

| Quantity | Pin / gate | Tol |
| -------- | ---------: | --- |
| \(E_{\mathrm{disp}}\) [Ha] | **E_disp = −1.0427466367435301×10⁻⁴ Ha** | 1×10⁻⁸ Ha |
| Free-atom MBD dual (\(v=1\), same O₂) | −1.1699749375715385×10⁻⁴ Ha | \(\lvert E_{\mathrm{Hirsh}}-E_{\mathrm{free}}\rvert>10^{-6}\) Ha and \(\lvert E_{\mathrm{Hirsh}}\rvert<\lvert E_{\mathrm{free}}\rvert\) |
| \(v_A\) (O, O) | two entries | (0.2, 2.0) |
| SCF / k | `converged`, `n_kpts >= 2` | — |
