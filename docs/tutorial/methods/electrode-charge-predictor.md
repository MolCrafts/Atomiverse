# Electrode charge predictor — Kolafa ASPC warm-start for const-pot CG

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `electrode-charge-predictor` |
| Regression | `regressions/md/electrode-charge-predictor/` |
| Public API | `cpu::electrodes::ChargeASPC`; `cpu::md::ConstPotBO` members `use_cg`, `predictor_order`, `last_iters()`, `last_used_predictor()` |
| Roadmap / queue | `docs/developer/metalwalls-method-queue.md` → **E2 done** (wave 2, 2026-07-24); map slice E2 |
| Oracle | **MetalWalls** (algorithm pin: Kolafa ASPC + MW const-pot CG/BO warm-start). LAMMPS ELECTRODE has **no** ASPC twin. |

## Summary

In constant-potential electrode MD the electrode charge vector \(Q(t)\) is
re-solved every step as the electrolyte moves. With a dense Cholesky path that
cost is one factorization of the elastance \(A\); with iterative CG (MetalWalls
path, large multi-site walls) cold-starting from zero (or a stale \(Q\)) burns
unnecessary Krylov iterations. Atomiverse ships Kolafa’s Always-Stable
Predictor-Corrector (**ASPC**) applied to the electrode charge vector:
`cpu::electrodes::ChargeASPC` extrapolates \(Q^{\mathrm{pred}}\) from a short
history, and `cpu::md::ConstPotBO` with `use_cg = true` and
`predictor_order = K` warm-starts CG from that guess, then records the
converged unconstrained solution. This is a **Component** feature on
`cpu::md::MD` (`pre_force`), not a Driver category and not inside `KSpace*`.
The product gate `reg_electrode-charge-predictor` pins the \(K=5\)
coefficients that MetalWalls c8ad175d hard-codes
(`src/electrode_charge.F90:572-577`), the \(K=2\) row of the same family, CG vs
dense charge agreement on the madelung twin, and a **strict** reduction of mean
CG iterations with ASPC vs cold start on an 18-site plate. Offline oracle is the
**real MetalWalls binary** (release 20.05, git c8ad175d); fair wall is
**`perf_exempt`** — see REFERENCE for why the two codes' charge solves are not
comparable work.

## Method

**Physics context.** The BO electrode problem is the SPD elastance system

\[
A\,Q = \Psi - b
\]

(with optional electroneutrality projection \(\sum_i Q_i = 0\) as in
`const-pot-bo-parallel-plate`). Here \(A_{ij}\) is the Gaussian potential
response among electrode sites, \(\Psi\) the applied potentials, and \(b\) the
electrolyte contribution. Over MD, \(A\) is often quasi-static (frozen walls)
while \(b\) (and thus \(Q\)) changes smoothly with ion motion. Iterative solvers
benefit from a good initial guess \(Q^{(0)}\).

**ASPC predictor (Kolafa).** With extrapolation order \(K\), store the last
\(K+1\) converged charge vectors \(\{Q_{n-K},\ldots,Q_n\}\) (oldest → newest).
The Lagrange coefficients \(\omega_k\) for predicting step \(n+1\) are

\[
\omega_k = \prod_{\substack{j=0 \\ j\neq k}}^{K}
  \frac{(K+1)-j}{k-j},\qquad k=0,\ldots,K
\]

so

\[
Q^{\mathrm{pred}}_{n+1} = \sum_{k=0}^{K} \omega_k\, Q_{n-K+k}.
\]

with Kolafa's coefficients (J. Comput. Chem. **25**, 335 (2004), Eq. 10)

\[
B_j = (-1)^{j+1}\, j\, \frac{\binom{2K+2}{K+1-j}}{\binom{2K}{K}},
\qquad \omega_{K+1-j} = B_j .
\]

For \(K=2\):

\[
\omega = \{1/2,\ -2,\ 5/2\}
\qquad\Rightarrow\qquad
Q^{\mathrm{pred}}_{n+1} = \tfrac12 Q_{n-2} - 2 Q_{n-1} + \tfrac52 Q_n,
\]

and for MetalWalls' \(K=5\),
\(B = \{22/7,\,-55/14,\,55/21,\,-22/21,\,5/21,\,-1/42\}\) newest → oldest.

These are **not** Lagrange (polynomial) extrapolation coefficients. ASPC is
exact only through degree 1 — \(\sum_j B_j = 1\) and \(\sum_j j B_j = 0\), so a
linear history \(Q_m = m\) predicts \(Q_{n+1} = n+1\) exactly, while a quadratic
history \(Q_m = m^2\) is deliberately damped (\(K=2\) gives 8, not the Lagrange
9). That damping is what makes the predictor-corrector pair unconditionally
stable; plain polynomial extrapolation is the scheme Kolafa 2004 replaces.

**Corrector blend (API surface).** `ChargeASPC::correct` implements Kolafa’s
time-reversibility blend

\[
Q_{\mathrm{corr}} =
  \frac{K+2}{2K+3}\,Q_{\mathrm{scf}}
  + \frac{K+1}{2K+3}\,Q_{\mathrm{pred}}
\]

for callers that want the full ASPC corrector. The `ConstPotBO` CG path
warm-starts from `predict` and records the unconstrained SCF solution
(history fill); the regression foothold gates **prediction + CG iter
reduction**, not the corrector blend.

**What this pin includes**

| Item | Setting |
| ---- | ------- |
| ASPC order | \(K=5\) coeffs == MetalWalls `B1..B6` (rel ≤ 1e-15); \(K=2\) coeffs \(\{1/2,-2,5/2\}\) (exact) |
| CG vs dense | 4-atom madelung family, \(xy\_image\_range=0\), \(\eta=2\) Å⁻¹, \(\Psi=(0,1)\) V |
| CG iter comparison | 3×3 atoms/wall (**18** electrode sites) + 4 ions, \(L_{xy}=6\) Å, \(L_z=20\) Å |
| Drift protocol | electrolyte \(z\) shifts each launch so \(b\) changes; primary cell only |
| Neutrality | on; \(\lvert\sum Q_i\rvert \le 10^{-8}\,e\) |
| Binding | standalone `build`/`launch` of `ConstPotBO` (same component MD uses) |

**Key exclusions (non-goals of this slug)**

- LAMMPS ELECTRODE twin (package has no ASPC path) — oracle is MetalWalls /
  Kolafa algorithm literature.
- Reciprocal-space electrode matrix / full `ewald/electrode` (E2 slab Ewald is
  a sibling slug).
- Fair wall-clock vs a live MetalWalls binary (`perf_exempt`).
- Charge solve inside PME/`KSpace*` — architecture-forbidden.
- Public `*Config` bags or `make_*` / `build_*` factories.
- Live MetalWalls or LAMMPS inside ctest.

## Implementation

Public surface only: `ChargeASPC` (order is a ctor arg) and `ConstPotBO`
members. No Config bag.

### Standalone ASPC (matches regression ac-002 / ac-003)

```cpp
#include "atomiverse/cpu/electrodes/charge_aspc.h"
#include <vector>

using namespace ATV;

cpu::electrodes::ChargeASPC aspc(/*order K=*/2);
// coefficient(k): k=0 oldest in the K+1 window → {1/2, -2, 5/2}
assert(aspc.coefficient(0) == 0.5);
assert(aspc.coefficient(1) == -2.0);
assert(aspc.coefficient(2) == 2.5);
// K=5 reproduces MetalWalls' hard-coded B1..B6 exactly:
cpu::electrodes::ChargeASPC mw(/*order K=*/5);
assert(mw.coefficient(5) == 22.0 / 7.0);   // B1

for (int n = 0; n <= 2; ++n)
    aspc.record(std::vector<double>{static_cast<double>(n)});

std::vector<double> pred;
const bool ok = aspc.predict(pred);  // needs K+1 history points
// pred[0] == 3.0 for linear history {0,1,2}
```

### ConstPotBO CG + ASPC warm-start (matches multi-site gate)

```cpp
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/dynamics/const_pot_bo.h"
#include "atomiverse/cpu/dynamics/md.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

// frame: multi-site parallel plate (see regression make_multi)
// Component defaults to pre_force order −50.
cpu::md::ConstPotBO conp(/*eta [1/Å]=*/2.0);
conp.add_electrode(/*type=*/1, /*Psi [V]=*/0.0);
conp.add_electrode(/*type=*/2, /*Psi [V]=*/1.0);
conp.electroneutrality = true;
conp.xy_image_range = 0;       // primary cell (foothold)
conp.use_cg = true;            // iterative path required for ASPC gain
conp.cg_tol = 1e-10;
conp.max_cg_iters = 500;
conp.predictor_order = 2;      // K=2; 0 = cold start every step

cpu::md::MD sim;
sim.set_neighbor(cpu::md::NeighborList<>{/*cutoff=*/12.0, /*skin=*/0.5});
sim.set_evaluator(cpu::PairCoulCut{/*cutoff=*/12.0});
sim.set_integrator(cpu::md::VelocityVerlet{/*dt [fs]=*/0.5});
sim.add_ref(conp);   // or add(std::move(conp)) to own
sim.build();
sim.run(frame, /*nsteps=*/16);   // ASPC history fills after K+1 steps
// after history fill (K+1 successful records):
//   conp.last_used_predictor() == true
//   conp.last_iters()  — CG iterations of last A Q = rhs solve
//   conp.neutrality_residual() <= 1e-8
```

Architecture rules:

- **Driver stays `cpu::md::MD`.** ASPC is owned by the electrode Component;
  never a new Driver category.
- Parameters are **constructor arguments and public members** — no
  `ChargeASPCConfig` / `ConstPotBOConfig`.
- Frame is the sole I/O currency; solved charges land on `kAtom_charge`.

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
  cmake --build build --target reg_electrode-charge-predictor -j"$(nproc)"
  ./build/regressions/reg_electrode-charge-predictor
'
```

| Path | Role |
| ---- | ---- |
| `include/atomiverse/cpu/electrodes/charge_aspc.h` | Public `ChargeASPC` |
| `src/cpu/electrodes/charge_aspc.cpp` | Lagrange \(\omega\), predict / correct / record |
| `include/atomiverse/cpu/dynamics/const_pot_bo.h` | `ConstPotBO` CG + `predictor_order` |
| `src/cpu/dynamics/const_pot_bo.cpp` | CG solve, ASPC warm-start / record |
| `regressions/md/electrode-charge-predictor/reg_electrode-charge-predictor.cpp` | Product gate |
| `tools/metalwalls_oracle/inputs/metalwalls/electrode-charge-predictor/README.md` | Offline oracle notes |

## Validation (vs MetalWalls / Kolafa)

Source:
[`regressions/md/electrode-charge-predictor/REFERENCE.md`](../../../regressions/md/electrode-charge-predictor/REFERENCE.md)
(offline pin; **never** live MetalWalls in ctest). Oracle tooling:
[`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md),
[`tools/metalwalls_oracle/inputs/metalwalls/electrode-charge-predictor/README.md`](../../../tools/metalwalls_oracle/inputs/metalwalls/electrode-charge-predictor/README.md).

**Oracle:** **metalwalls** (algorithm MetalWalls const-pot BO/CG + Kolafa ASPC).
Cited algorithm sources in REFERENCE: Kolafa, *J. Comput. Chem.* **25**, 335
(2004); Coretti et al., *J. Chem. Phys.* **157**, 184801 (2022);
Marin-Laflèche et al., JOSS MetalWalls.

**Why MetalWalls (not LAMMPS):** LAMMPS ELECTRODE implements const-pot BO but
has **no** ASPC charge-history warm-start. Binding oracle priority for this
loop is LAMMPS when the feature overlaps; here the feature is **MW-only**
(ASPC predictor) → MetalWalls binary c8ad175d (source-constant pin).

### Accuracy

| Quantity | Atomiverse gate / pin | Oracle / theory | Tol | Notes |
| -------- | --------------------: | --------------: | --- | ----- |
| ASPC \(K=5\) coeffs | \(\{22/7,-55/14,55/21,-22/21,5/21,-1/42\}\) | MetalWalls c8ad175d `electrode_charge.F90:572-577` | rel ≤ **1e-15** | `ChargeASPC::coefficient` |
| ASPC \(K=2\) coeffs \(\omega\) | \(\{1/2,-2,5/2\}\) | Kolafa 2004 Eq. 10 | **exact** | `ChargeASPC::coefficient` |
| Linear history predict | \(Q_k=k\) → \(Q^{\mathrm{pred}}=3\) after \(\{0,1,2\}\) | degree-1 exact | abs ≤ **1e-12** | regression ac-003 |
| Quadratic history damped | \(Q_k=k^2\) → 8 (Lagrange would give 9) | ASPC-vs-Lagrange discriminator | abs ≤ **1e-12** | regression ac-003 |
| CG \(Q_{\mathrm{top}}\) vs dense Cholesky | same madelung RHS | dense BO twin | abs ≤ **1e-8** e | `use_cg` on/off |
| Mean CG iters, post history fill | warm **&lt;** cold | MW ASPC intent | **strict inequality** | observed cold≈6, warm≈5 on 18-site plate (REFERENCE) |
| ASPC engaged after fill | `last_used_predictor() == true` for \(s\ge 6\) at \(K=2\) and \(K=5\) | needs \(K+1\) records | exact | 16-step drift protocol |
| \(\lvert\sum_i Q_i\rvert\) [e] | \(\le 10^{-8}\) | neutrality on | **1e-8** | every warm launch |
| Polarity at \(\Delta\Psi=+1\) V | \(Q_{\mathrm{top}}>0\), \(Q_{\mathrm{bot}}<0\) | capacitive | sign only | multi-site end state |

Geometry, \(\eta\), \(\Psi\), CG tolerances, and gates: **Method** table and
REFERENCE (match `reg_electrode-charge-predictor`).

Related unit / sibling surfaces (not this e2e pin):
`const-pot-bo-parallel-plate` (dense BO vs LAMMPS madelung),
`const-pot-cg-large-electrode` (CG path at scale).

### Offline pin repro (never ctest)

Read the constants straight out of the MetalWalls tree the pinned binary was
built from (`sed -n '556,596p' src/electrode_charge.F90` at c8ad175d), or
evaluate Kolafa's closed form at \(K=5\). A twin MetalWalls run of this
regression's 18-site geometry lives at
`tools/metalwalls_oracle/inputs/metalwalls/electrode-charge-predictor/`.
MetalWalls has **no switch to disable its predictor** (it turns on at step 7,
`src/system.F90:1430`) and its non-predictor branch initialises from the
previous step rather than from zero, so the "warm &lt; cold" iteration claim is
not A/B-testable there — see REFERENCE. Never invoked from ctest.

## Performance

Source: REFERENCE.md dual-gate summary; handoff
[`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md);
queue row `electrode-charge-predictor`.

| Twin | \(T_{\mathrm{MetalWalls}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | ------------------------------: | -----------------------: | ----: | ---- | ------ |
| ASPC CG warm-start vs cold | — | — | **perf_exempt** | n/a | No MetalWalls twin binary on PATH for fair wall; ASPC is MW-only algorithm pin |

| Field | Value |
| ----- | ----- |
| `perf_exempt` | **true** |
| `perf_exempt_reason` | no MetalWalls twin binary on PATH for fair wall; ASPC is MW-only algorithm pin |
| `ratio_atv_over_oracle` | **n/a** |
| `within_10x` | exempt |

Do **not** invent wall-clock ratios. When a MetalWalls binary is available for
a matching const-pot CG demo, re-run offline fair bench, record version +
walls, and replace `perf_exempt` in REFERENCE.

**Internal cost note (not a fair oracle ratio):** the regression only asserts
mean CG iterations warm &lt; cold after history fill (observed ≈5 vs ≈6 on the
18-site plate). That is an accuracy/algorithm gate, not \(T_{\mathrm{ATV}} /
T_{\mathrm{oracle}}\).

## Limitations

- **Coefficient pin is a source-level oracle read.** The \(K=5\) row is
  bit-compared against MetalWalls' hard-coded constants; the binary exposes no
  diagnostic that prints its predictor output, so there is no black-box
  charge-by-charge parity against a MetalWalls log. CG vs dense and the strict
  iteration reduction remain internal-consistency gates.
- **ASPC gain requires `use_cg = true`.** Dense Cholesky ignores
  `predictor_order` for iteration counts (`last_iters()` is 0 on dense).
- **History fill latency.** Prediction needs \(K+1\) successful `record`
  calls; first \(K\) steps after `build` / `reset` are cold.
- **v1 elastance.** Same real-space Gaussian \(A\) as `ConstPotBO` v1 (primary
  cell or 2D \(xy\) images) — not full electrode Ewald. ASPC is orthogonal to
  that matrix approximation.
- **Unconstrained record.** History stores the unconstrained CG solution used
  for warm-start; electroneutrality projection still applies to write-back
  charges — see `const_pot_bo` implementation notes if extending the API.
- **No fair speed ratio** until MW is on the pin host with a matching demo.
- **Scope of E2 slice:** large-electrode CG scale-out is sibling
  `const-pot-cg-large-electrode`; this page is the predictor foothold only.

## References

- Offline pin: [`regressions/md/electrode-charge-predictor/REFERENCE.md`](../../../regressions/md/electrode-charge-predictor/REFERENCE.md)
- Product gate: [`regressions/md/electrode-charge-predictor/reg_electrode-charge-predictor.cpp`](../../../regressions/md/electrode-charge-predictor/reg_electrode-charge-predictor.cpp)
- Public headers: [`include/atomiverse/cpu/electrodes/charge_aspc.h`](../../../include/atomiverse/cpu/electrodes/charge_aspc.h), [`include/atomiverse/cpu/dynamics/const_pot_bo.h`](../../../include/atomiverse/cpu/dynamics/const_pot_bo.h)
- Queue / handoff: [`docs/developer/metalwalls-method-queue.md`](../../developer/metalwalls-method-queue.md), [`docs/developer/metalwalls-method-handoff.md`](../../developer/metalwalls-method-handoff.md), [`docs/developer/metalwalls-method-loop.md`](../../developer/metalwalls-method-loop.md)
- Oracle notes: [`tools/metalwalls_oracle/README.md`](../../../tools/metalwalls_oracle/README.md), [`tools/metalwalls_oracle/inputs/metalwalls/electrode-charge-predictor/README.md`](../../../tools/metalwalls_oracle/inputs/metalwalls/electrode-charge-predictor/README.md)
- Literature: J. Kolafa, *J. Comput. Chem.* **25**, 335 (2004) (ASPC); A. Coretti et al., *J. Chem. Phys.* **157**, 184801 (2022) (const-pot / electrode dynamics context); A. Marin-Laflèche et al., MetalWalls (JOSS) — const-pot CG electrode charge extrapolation practice
- Sibling BO foothold (LAMMPS oracle): [`const-pot-bo-parallel-plate.md`](const-pot-bo-parallel-plate.md)
