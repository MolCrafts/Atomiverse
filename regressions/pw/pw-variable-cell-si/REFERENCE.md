# Reference — PW isotropic variable-cell relax (diamond Si, NC-UPF)

## Method

Productizes plane-wave **Nielsen–Martin analytic stress** (`pw-analytic-stress-si`)
into **variable-cell relaxation**:

- `cpu::minimizer::Minimize` + `IsotropicCellRelax` with
  extensive `kStress` [eV]; the relaxer divides by V
- `StressConverged` criterion (mechanical / CP2K stress convention)
- `cpu::pw::PWEvaluator{…}.with_stress(true)` bound at `force`
- Stress-only evaluator path: **one** production SCF supplies both energy
  and stress (no double SCF per cell step)

Fixed fractional coordinates; isotropic full-matrix box scale
(`Box::scale_isotropic` multiplies the full cell — required for the fcc
primitive whose diagonal is zero).

## System

| Knob | Value |
| ---- | ----- |
| Structure | diamond Si, 2-atom fcc primitive |
| Start \(a_0\) | **5.58 Å** (expanded) |
| PP | `tests/data/pw/Si_lda_nc.upf` (PseudoDojo ONCVPSP-LDA) |
| XC | LSDA |
| \(E_\mathrm{cut}\) | 15 Ry (tutorial; incomplete vs production) |
| k-mesh | \(2\times2\times2\) |
| SCF | `tol=10^{-6}`, `max_iter=40` |
| Relax | α=1, bulk_mod=3.5 eV/Å³, max_scale=0.012, budget 3 |
| Converge | `StressConverged{0, 1.5×10^{-2} eV/Å³}` |

## Offline pin (ATV zero-stress basin; never live oracle in ctest)

Incomplete ecut/k shifts the EOS minimum relative to PseudoDojo
\(a_0\approx 5.468\) Å. Pin from measured residual-stress relax on this twin
plus the mechanical stress at \(a=5.43\) (σ≈ **+0.0576** eV/Å³ from
`pw-analytic-stress-si`, mechanical convention):

| Quantity | Value |
| -------- | ----- |
| Relaxed conventional \(a^\*\) | **5.510 Å** |
| Residual \(\max_d\lvert\sigma_{dd}\rvert\) | **~8×10⁻³ eV/Å³** |
| Frame energy | **≈ −213.27 eV** |
| Direction | expanded \(a_0=5.58\) **shrinks** toward \(a^\*\) |

### CP2K note

CP2K Quickstep bulk-Si CELL_OPT uses **GTH-PADE + DZVP-GTH**, not ONCV
plane-wave. Absolute \(a^\*\) / energy are **not** comparable. Optional
methodology twin: `tools/cp2k_oracle/inputs/pw-variable-cell-si/si_cellopt.inp`
(offline wall only). Same stance as `pw-analytic-stress-si` /
`pw-lda-si`.

### PseudoDojo literature context

| Quantity | Value | Source |
| -------- | ----- | ------ |
| EOS \(a_0\) | ≈ 5.468 Å | PseudoDojo `Si.djrepo` |
| Twin \(a^\*\) (this ecut/k) | 5.510 Å | ATV residual-stress pin |

## Hard gates

| Gate | Tolerance | Measured (2026-07-25, Release, OMP=8) |
| ---- | --------- | ------------------------------------- |
| shrink from expanded start | \(a_1 < a_0\) | **5.5097 < 5.58** ✓ |
| \(\lvert a - a^\*\rvert\) | ≤ 0.05 Å | **3×10⁻⁴ Å** |
| residual \(\max\lvert\sigma_{dd}\rvert\) | ≤ 1.2×stol = 1.8×10⁻² | **8.0×10⁻³** |
| SCF / energy finite | — | E ≈ −213.27 eV ✓ |
| wall | ≤ 60 s | **50.2 s** |

## Perf gate

| Metric | Value |
| ------ | ----- |
| `perf_exempt` | **true** |
| Reason | No fair NC-UPF PW CELL_OPT twin (CP2K GTH ≠ ONCV); same class as `pw-analytic-stress-si` |
| \(T_{\mathrm{atv}}\) | **50.2 s** (informative) |
| \(T_{\mathrm{cp2k}}\) | n/a (no fair twin) |
| `within_10x` | n/a (exempt) |

## Stress convention (productization)

PW stress is **mechanical / CP2K**:
\(\sigma = -(1/\Omega)\,\partial E/\partial\varepsilon\)
(positive when compressed → `IsotropicCellRelax` expands). Matches GPW
`gpw_stress` and cell-relax drivers. Pre-product energy-derivative pin
signs in `pw-analytic-stress-si` are flipped accordingly.

## Oracle commands (offline; not ctest)

```bash
module load buildenv-gcc/2026.03-mpich
export CP2KROOT=/software/sse2/el9_epyc9005/easybuild/prefix/software/CP2K/2026.1-gmpflf-2026.03
export CP2K_DATA_DIR=$CP2KROOT/share/cp2k/data
export PATH=$CP2KROOT/bin:$PATH
export OMP_NUM_THREADS=4
cd tools/cp2k_oracle/inputs/pw-variable-cell-si
mpirun -np 1 cp2k.psmp -i si_cellopt.inp -o cellopt.out
# wall only — absolute a* not comparable to NC-UPF
```
