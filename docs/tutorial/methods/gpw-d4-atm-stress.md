# GPW DFT-D4 three-body ATM (s9 = 1) analytic cell stress

*(Technical-report page; CP2K capability loop.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d4-atm-stress` |
| Regression | `regressions/gpw/gpw-d4-atm-stress/` |
| Public API | `cpu::gpw::GPWEvaluator{d4 = true, d4_atm = true, compute_stress = true}` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

The **Axilrod–Teller–Muto three-body** term of DFT-D4 (\(s_9 = 1\), dftd4
`pbe-D4-ATM`) contributes to the GPW **analytic cell stress**. This page is
the stress companion of `gpw-d4-atm-forces` and the ATM extension of
`gpw-d4-stress`: the product stress is the Hellmann–Feynman fixed-density
path (`gpw_stress`) — one SCF → \(P, W\); central FD of the
non-self-consistent Lagrangian under orthorhombic diagonal strain at fixed
fractional coordinates — with the geometry-only dispersion energy
\(E_{\mathrm{disp}}^{\mathrm{D4+ATM}}(\{\mathbf{R}\},\mathbf{h})\)
(**s9 = 1** when `d4_atm`) rebuilt at each strained cell.
Sign convention is CP2K `STRESS_TENSOR`:
\(\sigma_{aa} = -\Omega^{-1}\,\partial E_L/\partial\varepsilon_{aa}\).

The ATM stress contribution on this cell is **~10⁻¹⁰ Ha/Bohr³** — three
orders below the analytic-vs-FD dual-gate residual (~7×10⁻⁷) — so the
decisive gates are **dedicated dftd4 energy pins at the exact strain
samples**, not a stress-tensor diff.

## Oracle and why it is dftd4, not live CP2K QS

CP2K 2026.1 on Arrhenius is built **without** the DFTD4 library
(`TYPE DFTD4` aborts). The hard pin is **dftd4 3.7.0 printed energies** —
the same libdftd4 backend CP2K calls when DFTD4 is linked; the QS input
family is retained under `tools/cp2k_oracle/inputs/gpw-d4-atm/`. Same policy
as `gpw-d4-atm` / `gpw-d4-atm-forces` / `gpw-d4-stress`.

All gates are on the oracle's **printed energies**, never its gradient:
dftd4 3.7.0's periodic ATM gradient is non-conservative by ~2.4×10⁻⁸ against
its own energy (documented in `gpw-d4-atm-forces/REFERENCE.md`).

## System / setup

| Item | Value |
| ---- | ----- |
| System | water dimer, 6 atoms (order **O H H O H H**) |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` (Ω = 11661.119488 Bohr³) |
| Coordinates [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H); 16 valence e⁻ (`n_occ = 8`) |
| XC + disp | PBE + D4 BJ (\(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)) + ATM (\(s_9=1\), alp = 16, disp3 = 40 Bohr) |
| Cutoff / strain step | 200 Ry; \(h = 2\times10^{-3}\) |
| Pin commands | per axis a, sign ±: scale cell row a **and** Cartesian a-components by \(1 \pm h\) (the `scale_axis` fixed-fractional convention), then `dftd4 -f pbe --json … POSCAR` and `--mbdscale 0.0` twin |

Base-cell s9=1 pin: \(E_{\mathrm{disp}} =\) **−9.9554475834240290×10⁻⁴ Ha**
(`pin_periodic_atm.json`, shared with `gpw-d4-atm`). Derived oracle
dispersion stress σ_aa = −[E(+h) − E(−h)]/(2hΩ); ATM-only part
σ^ATM = σ(s9=1) − σ(2b), max |σ^ATM| = **7.47×10⁻¹⁰ Ha/Bohr³** (xx) — the
physics this case gates.

## Gates (accuracy)

All hard-coded pins from dftd4 3.7.0 (REFERENCE.md); regression prints each
measured value on PASS. First-green measured values, 2026-07-31:

| Gate | Tol | Measured |
| ---- | --- | -------- |
| strained \|E_disp − pin\| (s9=1 and 2b, 12 values) | 1e−9 Ha | 6.30e−11 (Bohr-conversion constant) |
| \|σ^disp(s9=1) − FD(oracle E)\| per axis | 1e−12 Ha/Bohr³ | 5.19e−15 |
| \|σ^ATM − Δpin\| per axis | 1e−12 Ha/Bohr³ | 3.65e−16 |
| ATM non-vacuity max\|σ^ATM_pin\| | > 5e−10 Ha/Bohr³ | pin 7.47e−10 |
| product \|e_disp − ATM base pin\| | 1e−8 Ha | 2.76e−11 |
| dual gate \|σ_xx^an − σ_xx^fd\| (multi-SCF strain-FD twin) | 2e−5 Ha/Bohr³ | 7.392e−7 (same class as `gpw-d4-stress`) |
| off-diagonal product; max\|σ_diag\| floor | 1e−8; > 1e−10 | pass |
| wiring \|(σ_xx(ATM) − σ_xx(2b)) − σ^ATM_xx\| | 3e−10 Ha/Bohr³ | 7.05e−11 (dropping the s9 fold leaves 7.5e−10 → FAIL) |

Product analytic stress (Γ PBE-D4+ATM, 200 Ry; E = −34.331326946882 Ha):
σ_xx = −2.17593330×10⁻⁵, σ_yy = +2.00729249×10⁻⁵,
σ_zz = +5.82666928×10⁻⁵ Ha/Bohr³; `kStress` on Frame in eV/Å³.

Unit twin: `tests/cpu/gpw/test_cpu_gpw_dftd4.cpp`
`WaterDimerPeriodicStrainedAtmMatchesDftd4` (strained x+h s9=1 pin at
1×10⁻⁹; measured 3.8×10⁻¹²).

## Run it

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=ON -DATV_BUILD_PW=ON -DATV_BUILD_GPW=ON \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_gpw-d4-atm-stress -j$(nproc)
  ./build/regressions/reg_gpw-d4-atm-stress
'
```

Public surface (no `*Config`, no factories; Frame is the only token):

```cpp
cpu::gpw::GPWEvaluator eval({O_kind, H_kind});
eval.cutoff_ry = 200.0;
eval.xc = cpu::gpw::XcKind::PBE;
eval.nonperiodic = false;
eval.d4 = true;             // exclusive vs d3 / d3bj
eval.d4_atm = true;         // s9 = 1 in energy AND strained stress rebuild
eval.compute_stress = true; // analytic fixed-P path
eval.stress_strain_h = 2e-3;
cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);          // frame.meta.get(kStress) [eV/Å³];
                            // eval.last_stress_ha_bohr3() [Ha/Bohr³]
```

## Performance

**perf_exempt** (queue / handoff): no fair QS `STRESS_TENSOR` wall exists on
the pin host (CP2K lacks DFTD4). The wall is dominated by the one product
SCF + FD twin, not by the ATM term (geometry-only, milliseconds on 6 atoms).
Measured walls (OMP=8, Release): product 24.4 s; FD twin (x, warm) 4.9 s;
two-body analytic (x, warm) 6.1 s — case total ~36 s, within the ≤60 s
regression budget.

## Limitations

- Diagonal-strain (orthorhombic) stress only; off-diagonals gated ≈ 0 on
  this cubic cell, shear components not exercised.
- The ATM signal (~7.5×10⁻¹⁰ Ha/Bohr³ here) is resolved only through the
  dedicated strain-sample energy pins; the analytic-vs-FD dual gate alone
  cannot see it.
- The two product analytic runs (ATM vs 2b) converge reference densities
  independently at SCF tol 10⁻⁶, leaving ~7×10⁻¹¹ fixed-P Lagrangian noise
  in the wiring difference — hence the 3×10⁻¹⁰ wiring tolerance.
- PBE SAW190103 D4 parameters only; other functionals share tables, not
  gated. Water-dimer cell only; not a bulk-ice structural benchmark.
- Wall **perf_exempt**; not a speed target this wave.

## References

- Offline pins: `regressions/gpw/gpw-d4-atm-stress/REFERENCE.md`;
  `tools/cp2k_oracle/inputs/gpw-d4-atm/pin_strain_*_{atm,2b}.json`
- Related: [gpw-d4-atm](gpw-d4-atm.md) (energy),
  [gpw-d4-atm-forces](gpw-d4-atm-forces.md) (ATM forces),
  [gpw-d4-stress](gpw-d4-stress.md) (two-body stress),
  [gpw-d3-atm-stress](gpw-d3-atm-stress.md) (D3 twin)
- E. Caldeweyher, C. Bannwarth, S. Grimme, *J. Chem. Phys.* **150**, 154122
  (2019) (DFT-D4); B. M. Axilrod, E. Teller, *J. Chem. Phys.* **11**, 299
  (1943); Y. Muto, *Proc. Phys. Math. Soc. Jpn.* **17**, 629 (1943)
- dftd4 3.7.0 (CLI pin backend; the library CP2K links as `TYPE DFTD4`)
