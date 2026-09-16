# XL-BOMD on the Gaussian SCF methods — RHF/STO-3G H₂O NVE

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `xl-bomd-rhf-water` |
| Regression | `regressions/dynamics/xl-bomd-rhf-water/` |
| Public API | `cpu::md::MD` + `VelocityVerlet` + `cpu::electronic::XlBomdMethod<cpu::scf::{RHF,UHF,RKS,UKS,…}>{method, k}` bound with `set_evaluator_ref`; witness `xl_advances()`; propagator `cpu::electronic::XlBomd` |
| Spec | `esmm-cpmd-05-scf-xlbomd` (ac-001 ladder, ac-002 exclusivity, ac-003 history integrity, ac-004 the XL signature) |
| Dual gate | Correctness = `test_cpu_scf_xl_bomd` + `reg_xl-bomd-rhf-water`; speed **`perf_exempt`** |
| GPW sibling | [xl-bomd-water](xl-bomd-water.md) — same propagator behind GPW forces |
| CP sibling | [cpmd-rhf-h2](cpmd-rhf-h2.md) — the *other* answer to "what is the density now" on this seam |

## Summary

**Extended-Lagrangian Born–Oppenheimer MD** (Niklasson, *PRL* **100**, 123004
(2008); dissipative form Niklasson et al., *JCP* **130**, 214109 (2009))
propagates an auxiliary AO density matrix \(P\) alongside the nuclei and hands
it to the SCF as the initial guess each MD step:

\[
P_{n+1} = 2P_n - P_{n-1} + \kappa\,(D_n - P_n)
          + \alpha \sum_{k=0}^{K} c_k P_{n-k},
\]

with \(D_n\) the SCF-converged density at step \(n\) and \((\kappa,\alpha,c_k)\)
the tabulated row for the dissipation order \(K \in [3,9]\) (default \(K=5\):
\(\kappa=1.82\), \(\alpha=0.018\), \(c=\{-6,14,-8,-3,4,-1\}\)).

Reusing the last converged density is *not* a time-reversible operation — the
guess always lags the nuclei, and the one-sided error pumps energy into the
trajectory. The auxiliary density has its own (almost time-reversible)
dynamics, so warm steps converge in fewer iterations **and** without ASPC's
corrector bias. This is the **SCF-converged** form: the SCF still runs to `tol`
every step; only the guess changes. (The linearized / zero-SCF shadow variant
stays GPW-only — see *Honest non-goals*.)

The propagator is the same method-agnostic dense algebra GPW already uses
(`cpu::electronic::XlBomd`); this page is about the **SCF seam**: which member
turns it on, where the guess enters, and when the auxiliary density advances.

## API

| Public surface | Meaning |
| --- | --- |
| `cpu::electronic::XlBomdMethod<Method>{Method method, int k = 5}` | Move a configured `RHF` / `UHF` / `RKS` / `UKS` in and engage XL-BOMD auxiliary-density propagation with dissipation order \(K\) (history depth \(K+1\)). Table I covers \(K \in [3,9]\); anything else throws `std::invalid_argument` at build (lookup table, no fallback) |
| `int xl_advances() const` | Auxiliary propagations taken — one per **converged nested** run; 0 under plain BOMD. The public witness that the history is on the nuclear clock |

XL-BOMD is a **type**, not a flag: the mode bits live in `protected` storage on
`MeanFieldMethod`, and the wrapper (`cpu/electronic/aimd_method.h`) is the only
public way to set them. It publicly inherits `Method`, so `basis` / `tol` /
`max_iter` are still ordinary members on the object you move in. There is no
`XlConfig` and no `make_xl_*`: a plain BOMD user constructs the method
directly, and an XL user gets the same dissipation order the GPW evaluator
ships.

### The guess ladder

`drive_scf_` resolves **one** ordered chain per step — the entire contract:

| Rank | Source | When it wins |
| --- | --- | --- |
| 1 | the caller's **explicit** `SCFInitialGuess` | whenever `run(frame, &guess)` supplies one |
| 2 | **XL** `aux()` | `xl_bomd` and the history is seeded |
| 3 | **ASPC** `predict()` | `predictor(n)` set and the polynomial is filled |
| 4 | **last-P** reuse | a previous converged density exists |

Rank 1 beating rank 3 is a **behaviour change**: before `esmm-cpmd-05` the ASPC
branch overwrote an explicit caller guess (measured divergence 2.8e-5 Ha). The
ladder rewrite fixed that; `MeanFieldMethodGuessLadder.ExplicitGuessOutranksAspcPrediction`
pins it.

### Exclusivity — one guess owner at a time

`xl_bomd` is mutually exclusive with **both** other propagation schemes, in
both directions, and each pair throws `std::invalid_argument` on the first
nested entry:

- `xl_bomd` × `predictor(...)` — ASPC *extrapolates* a polynomial through past
  densities; XL propagates a *dynamical variable*. Two answers to "what is the
  density now".
- `xl_bomd` × `cpmd` — Car–Parrinello propagates the *coefficients* and never
  runs the SCF that XL-BOMD would be guessing for.

`GPWEvaluator::build` refuses the same pairs, so the two backends agree.

### History integrity

The auxiliary density is a dynamical variable on the **nuclear** clock:

- it advances exactly **once per converged nested run** (`xl_advances()`
  increments in lockstep with the MD steps);
- a **standalone** `run()` / `forces()` is allowed but leaves the history
  untouched — a single point is somewhere else in configuration space, and
  GPW takes the same line;
- an **unconverged** nested run *drops* the history rather than propagating an
  SCF residual into every later guess.

Channel convention: RHF/RKS propagate the factor-2 total density in one
channel; UHF/UKS/ROHF/ROKS propagate α and β independently; GHF propagates the
single 2n-dimensional spinor density.

## Usage

```cpp
cpu::scf::RHF rhf_base;                 // or UHF / RKS / UKS
rhf_base.basis = cpu::scf::BasisSet::STO3G;
rhf_base.tol = 1e-6;                    // Ha, |dE| SCF exit
// XL-BOMD is a wrapper TYPE around the method, not an `xl_bomd = true` flag:
// the dissipation order K (Niklasson 2009 Table I) is a constructor argument.
cpu::electronic::XlBomdMethod<cpu::scf::RHF> rhf{std::move(rhf_base), /*k=*/5};

cpu::md::MD sim;
sim.set_integrator(cpu::md::VelocityVerlet{0.25});   // fs — see below
sim.set_evaluator_ref(rhf);             // borrow: diagnostics stay readable
sim.build();
sim.run(frame, 20);                     // frame.meta.get(kEnergy) [eV]

rhf.xl_advances();                      // == 20: one advance per converged step
```

Step 1 is a cold seed (nothing has been propagated yet); every later step is a
normal SCF that starts from `aux()`.

### Choosing the timestep — κ = ω²Δt² is not decoration

The auxiliary density is a harmonic oscillator whose Verlet coupling
\(\kappa = \omega^2\Delta t^2\) is **fixed by the Table-I row**, so raising
\(\Delta t\) does not "cost accuracy" gradually — it moves the auxiliary
oscillator toward the nuclear frequency and the guess stops tracking. Measured
on H₂O/STO-3G (guess error = one-shot energy minus the converged energy at the
same geometry [Ha]; full table in the regression's `REFERENCE.md`):

| Δt (fs) | steps per O–H period | err(XL) / err(last-P reuse) |
| --- | --- | --- |
| 0.25 | ≈ 36 | **128–507× better** |
| 0.5 | ≈ 18 | 3.9–7.9× better (one sampled step: 8× *worse*) |
| 1.0 | ≈ 9 | **0.4–1.8× — mostly worse than plain reuse** |

Rule of thumb: XL-BOMD wants Δt well under the fastest vibrational period —
here ≈ 0.25 fs with hydrogens moving. That is why this case runs at 0.25 fs
while its Car–Parrinello siblings run at 0.05 fs. The GPW XL twin runs at
0.5 fs because it pins only the drift envelope — no iteration margin — and so
never depends on the tracking regime.

The *iteration count* is a coarser meter than the guess error, because the SCF
exits on \(|\Delta E| < \)`tol`: once a guess is "good enough" the counter
saturates. Warm-step mean over steps 9–20 at Δt = 0.25 fs, `tol` = 1e-6 Ha:

| Guess owner | mean SCF iterations |
| --- | --- |
| XL-BOMD (K = 5) | **5.67** |
| plain BOMD (last-P reuse) | 6.58 |
| ASPC(3) | 6.00 |
| cold start (fresh method each geometry) | 8.25 |

## Pin

- **Unit** (`test_cpu_scf_xl_bomd`, 14 tests): member-surface defaults; both
  exclusivity throws; \(K \notin [3,9]\) throws; advance-on-converged-nested-only
  (standalone and unconverged runs leave `xl_advances()` alone); restricted and
  unrestricted advance conventions; and the four-rung ladder order, probed with
  a one-shot energy fingerprint. The Table-I coefficients themselves are pinned
  in `test_cpu_electronic_xl_bomd`.
- **E2E** (`reg_xl-bomd-rhf-water`, 0.41 s): RHF/STO-3G H₂O NVE, 20 steps at
  Δt = 0.25 fs, three legs on one trajectory (XL / plain-BOMD twin / cold-start
  twin). Frozen: warm-mean iteration gain ≥ 1.50 vs cold (measured 2.58) and
  ≥ 0.40 vs reuse (measured 0.92); relative energy-drift envelope ≤ 2.0e-5
  (measured 1.955e-6); `xl_advances() == 20` and the twin's `== 0`; the two MD
  legs stay within 1e-6 Å. See
  `regressions/dynamics/xl-bomd-rhf-water/REFERENCE.md` for the full measured
  landscape (dt scan, tol scan, guess-error table) and why each margin was
  frozen where it is.

## Source

| What | Path |
| ---- | ---- |
| XL wrapper (`XlBomdMethod`) | `include/atomiverse/cpu/electronic/aimd_method.h` |
| Mode storage, ladder, advance hook, exclusivity gates | `include/atomiverse/cpu/scf/mean_field_method.h`, `src/cpu/scf/mean_field_method.cpp` |
| Propagator | `include/atomiverse/cpu/electronic/xl_bomd.h` |
| Unit tests | `tests/cpu/scf/test_scf_xl_bomd.cpp` |
| Regression | `regressions/dynamics/xl-bomd-rhf-water/` |

## Honest non-goals

- **Linearized / zero-SCF shadow XL-BOMD** — needs the shadow-energy force
  assembly; stays GPW-only ([xl-bomd-linearized-forces](xl-bomd-linearized-forces.md))
- Multi-k XL on the AO methods, and CUDA: out of scope
- **Long-time drift is not demonstrated here.** Time reversibility is
  XL-BOMD's headline property, and a 20-step case can only guard an
  *envelope*; a thousand-step demonstration is production usage, not a
  ≤ 60 s regression
- No third-party XL-BOMD twin runs at ctest time (`perf_exempt`): SCF
  iteration counts are not comparable across programs, so the pinned
  observable is a relative property of three runs of this code
