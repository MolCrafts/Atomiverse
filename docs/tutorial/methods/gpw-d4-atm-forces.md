# GPW DFT-D4 three-body ATM (s9 = 1) analytic ionic forces

*(Technical-report page; CP2K capability loop.)*

| Field | Value |
| ----- | ----- |
| Slug | `gpw-d4-atm-forces` |
| Regression | `regressions/gpw/gpw-d4-atm-forces/` |
| Public API | `cpu::gpw::GPWEvaluator{d4 = true, d4_atm = true, compute_forces = true}` |
| Roadmap | `docs/roadmap/dft.md` §8 Dispersion |

## Summary

The **Axilrod–Teller–Muto three-body** term of DFT-D4 (\(s_9 = 1\), dftd4
`pbe-D4-ATM`) contributes its **analytic gradient** to the GPW ionic forces:
closed-form damped Axilrod–Teller triple derivatives **plus** the
C9 → neutral-C6 → CN\(_\mathrm{d4}\) chain rule. Because dftd4's ATM C6
reweight uses **neutral charges** (\(q = 0\), matching `disp.f90`), that
reweight is geometry-independent — the ATM leg has **no EEQ adjoint**, unlike
the two-body D4 gradient (`gpw-d4-forces`). The energy landed with
`gpw-d4-atm`; this capability closes the closed-form FD-free ATM force leg in
`cpu::scf::dftd4` and gates it end-to-end on the periodic water dimer.

Product ionic force = electronic Hellmann–Feynman + Pulay (analytic
`gpw_forces_*`) \(+\; F_{\mathrm{disp}} = -\nabla\!\left(E_2 +
E_{\mathrm{ATM}}\right)\) with \(s_9 = 1\).

## Oracle and why it is dftd4, not live CP2K QS

CP2K 2026.1 on Arrhenius is built **without** the DFTD4 library
(`TYPE DFTD4` aborts: `CP2K build without DFTD4`). The hard pin is
**dftd4 3.7.0 `--grad`** — the same libdftd4 backend CP2K calls when DFTD4
is linked; the QS twin input family is retained under
`tools/cp2k_oracle/inputs/gpw-d4-atm/` for hosts that link it. Same policy as
`gpw-d4-atm` / `gpw-d4-forces`.

### Oracle finding: dftd4 3.7.0's periodic ATM gradient is non-conservative

All numbers below were printed by dftd4 3.7.0 (offline, 2026-07-31):

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
printed periodic pin beyond that oracle defect. The gates are set
accordingly: the **direct** gradient-pin comparison at **5×10⁻⁸ Ha/Bohr**
(absorbs the ~2.4×10⁻⁸ defect, still ~8× below the 4.05×10⁻⁷ ATM increment,
so it fails if the ATM gradient is dropped), plus a **sharp 1×10⁻⁹
conservative-parity** gate against the FD of the oracle's *printed displaced
energies* (O0 x ± 0.001 Å; truncation ≈ 3×10⁻¹⁰).

## System / setup

| Item | Value |
| ---- | ----- |
| System | water dimer, 6 atoms (order **O H H O H H**) |
| Cell | 12 × 12 × 12 Å, `PERIODIC XYZ` |
| Coordinates [Å] | O (0,0,0); H (0.758602, 0, ±0.504284); O (3,0,0); H (3.758602, 0, ±0.504284) |
| Basis / PP | `DZVP-MOLOPT-SR-GTH` + `GTH-PBE-q6` (O) / `GTH-PBE-q1` (H); 16 valence e⁻ (`n_occ = 8`) |
| XC + disp | PBE + D4 BJ (\(s_6=1\), \(s_8=0.95948085\), \(a_1=0.38574991\), \(a_2=4.80688534\)) + ATM (\(s_9=1\), alp = 16, disp3 = 40 Bohr, neutral-charge C6 reweight) |
| Cutoff | 200 Ry |
| Pin commands | `dftd4 -f pbe --grad --json pin_periodic_atm_grad.json POSCAR` (+ isolated xyz twin; + two-body `--mbdscale 0` twin; + O0 x ± 0.001 Å displaced-energy pins) |

Hard energy pin: \(E_{\mathrm{disp}}\)(D4+ATM, periodic)
= **−9.9554475834240310×10⁻⁴ Ha**; ATM-only force increment vs the two-body
`--mbdscale 0` pin has max component **4.05×10⁻⁷ Ha/Bohr** — the physics this
case gates. Isolated twin \(E = −9.8819741533958229\times 10^{-4}\) Ha,
gradient pinned at 1×10⁻⁹ Ha/Bohr in unit tests.

## Gates (accuracy)

All hard-coded pins from dftd4 3.7.0 (REFERENCE.md); the regression prints
each measured value on PASS
(`max|F+g|`, `max|dF_atm-pin|`, `max|wiring|`, `wall_prod`).

| Gate | Tol | Context / signal |
| ---- | --- | ---------------- |
| \(E_{\mathrm{disp}}\)(ATM) standalone + product vs pin | 1e−8 Ha | pin −9.9554475834240310e−4 Ha; ATV self-FD conservative to 3e−12 |
| Direct: max\|F_d4atm + g_pin\| (periodic) | 5e−8 Ha/Bohr | oracle defect ~2.4e−8; ATM increment 4.05e−7 |
| Conservative parity: \|F_x(O0) − FD(oracle E±)\| | 1e−9 Ha/Bohr | FD truncation ≈ 3e−10; unit-test measured 1.7e−12 (isolated grad pin) |
| ATM increment: max\|(F_atm − F_2b) − Δpin\| | 5e−8 Ha/Bohr | non-vacuity floor max\|dF\| > 1e−8 |
| Product wiring: max\|(F_on − F_plain) − F_standalone\| | 1e−7 Ha/Bohr | SCF residual budget |
| Wall (product SCF + forces) | ≤ 60 s | regression budget |

Unit dual gate (`tests/cpu/gpw/test_cpu_gpw_dftd4.cpp`):
`GpwDftD4.{IsolatedAtmForcesMatchDftd4Grad, PeriodicAtmForcesMatchDftd4Grad,
PeriodicAtmForceIncrementMatchesDftd4Delta,
AtmForcesMatchEnergyCentralDifference}` — analytic vs FD of the shared
\(s_9 = 1\) functional lives there (6N SCFs on this cell would blow the ≤60 s
regression budget).

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
  cmake --build build --target reg_gpw-d4-atm-forces -j$(nproc)
  ./build/regressions/reg_gpw-d4-atm-forces
'
```

Public surface (no `*Config`, no factories):

```cpp
cpu::gpw::GPWEvaluator eval({O_kind, H_kind});
eval.cutoff_ry = 200.0;
eval.xc = cpu::gpw::XcKind::PBE;
eval.nonperiodic = false;
eval.d4 = true;          // exclusive vs d3 / d3bj
eval.d4_atm = true;      // s9 = 1
eval.compute_forces = true;
cpu::Evaluate driver;
driver.on(evaluate::Phase::evaluate).add_ref(eval);
driver.build();
driver.run(frame);       // forces on kAtom_f{x,y,z}; e_disp on last_result()
```

## Performance

**perf_exempt** (queue / handoff): ATM force evaluation is geometry-only
\(O(N^3 \times N_{\mathrm{images}}^2)\) on a 6-atom cell (milliseconds); the
wall is the electronic SCF + Pulay forces. A fair full QS `ENERGY_FORCE` twin
ratio vs CP2K is unavailable on the pin host (no DFTD4-linked CP2K) — same
policy as `gpw-d4-forces` / `gpw-d4-water`.

## Limitations

- ATM **virial/stress** is not gated here (`gpw-d4-stress` is the two-body
  stress family; a D4-ATM stress term is a separate queue row).
- Energy + force gates only on the water-dimer cell; not a bulk-ice
  structural benchmark.
- Direct periodic gradient parity is bounded below by the **oracle's own
  ~2.4×10⁻⁸ non-conservativeness**; the sharp 1×10⁻⁹ gate is against its
  printed displaced energies, not its printed gradient.
- PBE SAW190103 parameters only; other functionals share tables, not gated.
- Wall **perf_exempt**; not a speed target this wave.

## References

- Offline pins: `regressions/gpw/gpw-d4-atm-forces/REFERENCE.md`;
  `tools/cp2k_oracle/inputs/gpw-d4-atm/` (grad + displaced-energy JSON pins)
- Related: [gpw-d4-atm](gpw-d4-atm.md) (energy),
  [gpw-d4-forces](gpw-d4-forces.md) (two-body forces),
  [gpw-d3-atm-forces](gpw-d3-atm-forces.md) (D3 C9 twin),
  [gpw-d4-stress](gpw-d4-stress.md)
- E. Caldeweyher, C. Bannwarth, S. Grimme, *J. Chem. Phys.* **150**, 154122
  (2019) (DFT-D4); B. M. Axilrod, E. Teller, *J. Chem. Phys.* **11**, 299
  (1943); Y. Muto, *Proc. Phys. Math. Soc. Jpn.* **17**, 629 (1943)
- dftd4 3.7.0 (CLI pin backend; the library CP2K links as `TYPE DFTD4`)
