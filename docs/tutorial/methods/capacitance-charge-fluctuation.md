# Capacitance from electrode charge fluctuations — parallel-plate toy

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `capacitance-charge-fluctuation` |
| Regression | `regressions/md/capacitance-charge-fluctuation/` |
| Public API | `cpu::CapacitanceChargeFluctuation` (`Observer` on `cpu::md::MD`, default `end_of_step` @ +50) |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → **E10 done** (wave 2, 2026-07-24); map slice E10 ecosystem |
| Oracle | **literature** (Limmer / Scalfi–Salanne / MetalWalls charge-fluctuation \(C\); no LAMMPS or MetalWalls binary twin) |

## Summary

**Differential and integral capacitance** of a classical constant-potential
electrode cell estimated from electrode total-charge statistics. Public surface
is `cpu::CapacitanceChargeFluctuation` — a frame-immutable **Observer** that
samples \(Q = \sum_{i\in e_0} q_i\) each launch (primary electrode = type-list
index 0), accumulates moments, and publishes

\[
C_{\mathrm{diff}} = \mathrm{Var}(Q)/(k_{\mathrm{B}}T),\qquad
C_{\mathrm{int}} = \langle Q\rangle/\Delta\Psi
\]

in e/V on members and on Frame meta (`kCapacitanceDiff`,
`kCapacitanceIntegral`). It does **not** solve constant-potential charges; it
measures whatever is already on the Frame (typically after `ConstPotBO` at
`pre_force`). Architecture: Component on `cpu::md::MD` — never a new Driver
category; never charge solve inside `KSpace*`. Product gate
`reg_capacitance-charge-fluctuation` pins the closed-form literature formula on
a four-sample synthetic \(Q\) series (proxy for a short const-pot MD charge
trace). Fair wall vs an oracle binary is **`perf_exempt`**.

## Method

**Physics.** At fixed applied electrode potential, the differential capacitance
of a parallel-plate electrochemical cell follows from electrode charge
fluctuations (Limmer et al. 2013; Scalfi et al. 2020; MetalWalls analysis
tools):

\[
C_{\mathrm{diff}}
  = \frac{\langle Q^2\rangle - \langle Q\rangle^2}{k_{\mathrm{B}}T}
  = \frac{\mathrm{Var}(Q)}{k_{\mathrm{B}}T}
  \qquad[\mathrm{e}/\mathrm{V}],
\]

with \(Q\) the total charge on the **primary** electrode [e], \(T\) [K], and

\[
k_{\mathrm{B}} = 8.617333262\times 10^{-5}\,\mathrm{eV\,K^{-1}}
\]

(CODATA 2018, `units::kB_ev_per_k`) so \(k_{\mathrm{B}}T\) is in eV and
\(1\,e\cdot 1\,\mathrm{V} = 1\,\mathrm{eV}\). Population variance uses the
running moments \(\sum Q\), \(\sum Q^2\) over \(N\) samples:

\[
\mathrm{Var}(Q) = \frac{1}{N}\sum_s Q_s^2 - \Bigl(\frac{1}{N}\sum_s Q_s\Bigr)^2.
\]

When a fixed potential difference \(\Delta\Psi\) [V] is known and nonzero, the
integral capacitance is

\[
C_{\mathrm{int}} = \frac{\langle Q\rangle}{\Delta\Psi}
  \qquad[\mathrm{e}/\mathrm{V}].
\]

If \(\Delta\Psi = 0\), the integral route is disabled (`C_int` = NaN; meta key
`kCapacitanceIntegral` is not written).

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| Geometry | Parallel-plate toy: 4 left wall atoms (type 1), 4 right (type 2); no electrolyte |
| Box | cubic \(L = 20\) Å |
| \(T\) | \(300\) K |
| \(\Delta\Psi\) | \(1.0\) V |
| \(Q_L\) samples [e] | \(0.4,\;0.6,\;0.4,\;0.6\) (equal per-atom share on each wall; \(Q_R = -Q_L\)) |
| \(N\) | \(4\) |
| Primary electrode | type list index 0 → type **1** (left wall) |
| Binding | standalone `launch` loop (regression); MD default `end_of_step` @ +50 |

Closed form (REFERENCE):

\[
\langle Q\rangle = 0.5\,\mathrm{e},\qquad
\mathrm{Var}(Q) = 0.01\,\mathrm{e}^2,
\]

\[
k_{\mathrm{B}}T = 8.617333262\times 10^{-5}\times 300 = 0.025851999786\,\mathrm{eV},
\]

\[
C_{\mathrm{diff}}^{\mathrm{pin}} = \frac{0.01}{0.025851999786}
  = 0.38681727072485284\,\mathrm{e/V},\qquad
C_{\mathrm{int}}^{\mathrm{pin}} = 0.5\,\mathrm{e/V}.
\]

**Key exclusions (non-goals of this slug)**

- Solving electrode charges (that is `ConstPotBO` / E1–E2); this Observer only
  **reads** `kAtom_charge`.
- Live MetalWalls post-process scripts or LAMMPS `compute` twins inside ctest.
- Area-normalized areal capacitance [µF/cm²], series/parallel multi-electrode
  decomposition, or free-energy thermodynamic integration of \(C\).
- Block averaging / bootstrap error bars; finite-sample bias corrections.
- Public `*Config` bags or `make_*` / `build_*` factories.

## Implementation

Public surface only (`cpu::Frame` + ctor args; no Config bag). Minimal sequence
matching the regression:

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/observe/capacitance_charge_fluctuation.h"

using namespace ATV;

// 4 left (type 1) + 4 right (type 2); set wall totals each sample.
cpu::Frame frame(8);
// … positions, types, masses, tags; frame.box = cpu::Box::cubic(20.0);

// After charge solve so samples see updated Q (default end_of_step @ +50).
cpu::md::ConstPotBO conp(/*eta [1/Å]=*//*…*/);
// … electrodes, electroneutrality, …

cpu::CapacitanceChargeFluctuation cap(
    /*electrode_types=*/{1, 2},  // index 0 = primary Q
    /*temperature_K=*/300.0,
    /*delta_psi_V=*/1.0);

cpu::md::MD sim;
// … neighbor, potentials, integrator …
sim.add_ref(conp);
sim.add_ref(cap);   // or add(std::move(cap))
sim.build();
sim.run(frame, /*nsteps=*//*…*/);

// Members [e], [e²], [e/V]
const double mean_Q = cap.mean_charge();
const double var_Q  = cap.charge_variance();
const double C_diff = cap.differential_capacitance();    // Var/(kB T)
const double C_int  = cap.integral_capacitance();        // mean_Q / ΔΨ
// Frame meta: kCapacitanceDiff, kCapacitanceIntegral, kElectrodeCharge
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** Capacitance is an Observer Component; never a
  new Driver category.
- Parameters are **constructor arguments** — no `CapacitanceChargeFluctuationConfig`.
- Frame is the sole I/O currency; outputs land on meta keys and component
  members.
- MPI: per-electrode totals are all-reduced when a domain decomposition is
  active (`owned_count` + `MPI_Allreduce`).

Build and run the regression (MD-only configure; CPU login pattern from
`AGENTS.md` / metalwalls loop):

```bash
bash -lc '
  module load buildenv-gcc/2026.03-mpich
  export OPENBLAS_ROOT="$OPENBLASDIR"
  export RUSTUP_TOOLCHAIN=stable-x86_64-unknown-linux-gnu
  export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
        -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
        -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
        -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
  cmake --build build --target reg_capacitance-charge-fluctuation -j"$(nproc)"
  ./build/regressions/reg_capacitance-charge-fluctuation
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/observe/capacitance_charge_fluctuation.h` | Public `CapacitanceChargeFluctuation` |
| `src/cpu/observe/capacitance_charge_fluctuation.cpp` | Sum \(Q\), moments, meta write-back |
| `include/atomiverse/core/frame_schema.h` | `kCapacitanceDiff`, `kCapacitanceIntegral`, `kElectrodeCharge` |
| `regressions/md/capacitance-charge-fluctuation/reg_capacitance-charge-fluctuation.cpp` | Product gate |
| `tests/cpu/observe/test_capacitance_charge_fluctuation.cpp` | Unit surface (not the e2e pin) |
| `tools/metalwalls_oracle/pin_notes_capacitance-charge-fluctuation.md` | Offline pin provenance |

## Validation (vs literature)

Source:
[`regressions/md/capacitance-charge-fluctuation/REFERENCE.md`](../../../regressions/md/capacitance-charge-fluctuation/REFERENCE.md)
(offline pin; **never** live MetalWalls or LAMMPS in ctest). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md),
[`tools/metalwalls_oracle/pin_notes_capacitance-charge-fluctuation.md`](../../../tools/metalwalls_oracle/pin_notes_capacitance-charge-fluctuation.md).

**Oracle:** **literature** — Limmer et al., *J. Chem. Phys.* **139**, 184704
(2013) (charge-fluctuation \(C\)); Scalfi et al., *Phys. Chem. Chem. Phys.*
**22**, 10480 (2020); MetalWalls JOSS **5**(53), 2373 (2020) analysis tools.
Offline pin = closed-form evaluation of that formula on the synthetic \(Q\)
series above (proxy for a short const-pot MD charge time series).

**Why literature (not LAMMPS / live MetalWalls):**

- LAMMPS core has no built-in electrode charge-fluctuation capacitance compute;
  the pin host also lacked ELECTRODE.
- MetalWalls analysis tools implement this formula; no MW binary was installed
  for a live twin.
- Binding oracle priority for this loop is **LAMMPS when the feature overlaps**;
  here it does not — formula identity is textbook for const-pot electrodes.

### Accuracy

| Quantity | Atomiverse gate / pin | Literature closed form | Tol | Notes |
| -------- | --------------------: | ---------------------: | --- | ----- |
| \(N\) samples | **4** | 4 | exact | synthetic series length |
| \(\langle Q\rangle\) [e] | **0.5** | 0.5 | **1e-12** | mean of \(\{0.4,0.6,0.4,0.6\}\) |
| \(\mathrm{Var}(Q)\) [e²] | **0.01** | 0.01 | **1e-12** | population variance |
| \(C_{\mathrm{diff}}\) [e/V] | **0.38681727072485284** | \(0.01/(k_B\cdot 300)\) | **1e-12** | `units::kB_ev_per_k` |
| \(C_{\mathrm{int}}\) [e/V] | **0.5** | \(0.5/1.0\) | **1e-12** | \(\Delta\Psi = 1\) V |
| meta `kCapacitanceDiff` | same as \(C_{\mathrm{diff}}\) | same | **1e-12** | Frame write-back |
| meta `kCapacitanceIntegral` | same as \(C_{\mathrm{int}}\) | same | **1e-12** | Frame write-back |

Tolerance is machine-eps scale: exact rational moments + the shared \(k_B\)
constant. Catches wrong electrode selection, double-counting, unit mistakes in
\(k_B T\), or inverted \(C=\mathrm{Var}/(kT)\) algebra.

Recorded offline constants (REFERENCE):

```text
PIN_MEAN_Q      0.5
PIN_VAR_Q       0.01
PIN_C_DIFF      0.38681727072485284
PIN_C_INT       0.5
```

Related unit surface (not the e2e pin):
`tests/cpu/observe/test_capacitance_charge_fluctuation.cpp`.

### Offline pin repro (never ctest)

```bash
# Closed form (Python / CODATA kB matching ATV) — no MW/LAMMPS binary required
python3 - <<'PY'
kB = 8.617333262e-5
Q = [0.4, 0.6, 0.4, 0.6]
mean = sum(Q)/len(Q)
var = sum(q*q for q in Q)/len(Q) - mean*mean
print(mean, var, var/(kB*300), mean/1.0)
PY
# → 0.5  0.01  0.38681727072485284  0.5
```

## Performance

Source: REFERENCE.md dual-gate summary; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | --------------------------: | -----------------------: | ----: | ---- | ------ |
| four-sample moment update | — | — | **perf_exempt** | n/a | No MetalWalls or LAMMPS twin binary for this named capacitance post-process |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | Analysis Observer is \(O(N_{\mathrm{electrode}})\) charge sums + \(O(1)\) moment updates over four synthetic samples. No MetalWalls or LAMMPS twin binary for this named capacitance post-process on the agent host; fair wall vs an oracle binary is not applicable. |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. A fair twin would require either a
MetalWalls analysis-script harness or a custom LAMMPS post-process with the
same \(Q\) series and \(k_B\); neither shipped as a binary pin for this slug.

## Limitations

- **Observer only.** Does not solve or enforce constant potential; pair with
  `ConstPotBO` (or an equivalent charge model) for physical MD trajectories.
- **Primary electrode only for moments.** Capacitance moments use type-list
  index 0; other electrodes are summed for diagnostics (`total_charge(e)`,
  `kElectrodeCharge` pack ≤3) but do not enter \(\mathrm{Var}(Q)\).
- **Synthetic pin, not a production trajectory.** The e2e gate uses four
  hand-assigned \(Q\) samples — it validates algebra and electrode selection,
  not sampling quality, equilibration, or areal \(C\) of a real electrolyte.
- **Population variance, no block error.** No running SEM, bootstrap, or
  decorrelation window; long correlated const-pot series need external analysis
  or a later extension.
- **No areal normalization.** Results are in e/V (total device), not µF/cm².
- **No fair speed ratio** until a documented oracle binary twin exists.
- **E10 ecosystem slice.** Thermodynamic integration of capacitance, potential
  profiles, and related paper workflows are sibling / follow-on queue items.

## References

- Offline pin: [`regressions/md/capacitance-charge-fluctuation/REFERENCE.md`](../../../regressions/md/capacitance-charge-fluctuation/REFERENCE.md)
- Product gate: [`regressions/md/capacitance-charge-fluctuation/reg_capacitance-charge-fluctuation.cpp`](../../../regressions/md/capacitance-charge-fluctuation/reg_capacitance-charge-fluctuation.cpp)
- Public header: [`include/atomiverse/cpu/observe/capacitance_charge_fluctuation.h`](../../../include/atomiverse/cpu/observe/capacitance_charge_fluctuation.h)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md), [`tools/metalwalls_oracle/pin_notes_capacitance-charge-fluctuation.md`](../../../tools/metalwalls_oracle/pin_notes_capacitance-charge-fluctuation.md)
- Capability map: [`.claude/notes/metalwalls-capability-map.md`](../../../.claude/notes/metalwalls-capability-map.md) (E10)
- Literature: Limmer et al., *J. Chem. Phys.* **139**, 184704 (2013); Scalfi et al., *Phys. Chem. Chem. Phys.* **22**, 10480 (2020); Marin-Laflèche et al., *J. Open Source Softw.* **5**(53), 2373 (2020) (MetalWalls)
