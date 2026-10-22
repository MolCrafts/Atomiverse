# mc/metropolis

One case; every section below is run by `atv_regressions --case mc/metropolis` and keeps its own oracle record.


## Reference — NPT (isothermal–isobaric) MC of the truncated-and-shifted LJ vapor

### What is checked

Three independent statements about the same Markov chain.

#### 1. The ensemble mean

⟨V⟩ from NPT Monte Carlo against the **exact finite-N ensemble average** plus
the second-virial correction:

⟨V⟩ = (N+1)·k_BT/P + N·B₂(T) + O(N·B₂²·ρ)

- Frenkel & Smit, *Understanding Molecular Simulation* (2nd ed. 2002), §5.4:
  the NPT partition function with volume sampled **uniformly in V** weights
  configurations by V^N e^{−βPV} (their eq. 5.4.7), and the acceptance rule
  for a volume move is exp(−β[ΔU + P·ΔV − N·k_BT·ln(V'/V)]) (eq. 5.4.11).
  For the ideal gas ∫V^N e^{−βPV}dV gives exactly ⟨V⟩ = (N+1)k_BT/P — the
  (N+1), not N, is the fingerprint of the correct V^N weight.
- The interacting correction comes from the virial equation of state
  P = ρk_BT(1 + B₂ρ + …) (McQuarrie, *Statistical Mechanics*, eq. 12-25),
  which at first order shifts the volume by +N·B₂.
- B₂(T) = −2π ∫₀^∞ [e^{−βu(r)} − 1] r² dr (McQuarrie eq. 12-30), evaluated by
  quadrature for **exactly** the sampled cut-and-shifted potential (see the
  gcmc_lj REFERENCE.md for why an analytic, convention-exact reference is
  used instead of a NIST table computed with long-range corrections).

Measured: ⟨V⟩ = 241250.7 Å³ against a prediction of 241370.6 Å³ — **0.05 %**,
or 0.17 blocked standard errors.

#### 2. The ensemble width

The same partition function fixes the *width*, not just the mean: for the
ideal gas Var V = (N+1)(k_BT/P)², so σ_V = √(N+1)·k_BT/P exactly.

Measured: σ_V = 22267 Å³ against 22282 Å³ — **0.07 %**.

This gate is what a *frozen* chain fails hardest: a Markov chain that has
stopped moving has σ_V = 0 exactly while its mean can sit anywhere plausible.
It is also what a wrong volume Jacobian reshapes first.

#### 3. The V^N weight, with no tolerance at all

The (N+1) is the whole fingerprint, and it is worth exactly k_BT/P = 1985 Å³
here. So the case simply asks which of the two candidate predictions the
sampled mean is closer to:

```
|⟨V⟩ − [(N+1)kT/P + N·B₂]| = 120 Å³      <-- must win
|⟨V⟩ − [ N   kT/P + N·B₂]| = 1865 Å³
```

No tolerance is negotiated: one model is right and the other is 1/(N+1) away.

#### 4. The sampler's reference energy still describes the frame

`NptMetropolis` carries the last accepted energy forward rather than
recomputing it. If anything mutates the configuration behind the sampler's
back, the two part company silently and every later trial is judged against a
state that no longer exists. The case therefore re-evaluates the final
configuration through a fresh `cpu::Evaluate` and requires bitwise agreement
(measured |ΔE| = 0).

That is not a hypothetical. Until 2026-07-28 the MC plan let `atom_modify
sort` renumber frame rows *inside* a trial; `mc::Transaction` records its
changeset by row, so a rejected move then spliced a saved row over an
unrelated atom. Two atoms landed on top of each other, the chain's reference
energy stayed at −0.109 eV while the configuration it held was worth
+20 091 eV, and every subsequent trial was rejected against an overlap the
sampler could not see. The chain froze at trial ~15 000 of 340 000 with no
error, no NaN, and a plausible-looking ⟨V⟩ 1.24 % off the prediction. See
`tests/cpu/montecarlo/test_mc_row_order.cpp`.

### State point

| Quantity | Value |
|---|---|
| Potential | LJ 12-6, cut AND shifted at r_c = 2.5σ |
| ε/k_B, σ | 120 K, 3.4 Å (argon-flavoured) |
| T* | 1.5 (supercritical) |
| N | 125 (5³ simple-cubic start) |
| P* = Pσ³/ε | 0.0297 (targets ρσ³ ≈ 0.02) |
| Moves | 50% single-particle translation (0.4σ), 50% volume (δV = ⟨V⟩/√N) |
| Chain | 20 000 equilibration + 400 000 production trials (~28 s) |

At ρσ³ ≈ 0.02 the O(B₃ρ²) term is ~10⁻⁴ relative — the second-order
prediction is exact at the quoted tolerance.

Half the trials are volume moves because **volume is the observable**: the
error bar on ⟨V⟩ is set by the volume autocorrelation time, and translations
do not shorten it.

### Tolerance, and how the error bar is estimated

3× the **blocked** standard error of ⟨V⟩ plus 0.2 % of ⟨V⟩ for the
virial-series truncation.

The error bar comes from **batch means** (Flyvbjerg & Petersen,
*J. Chem. Phys.* **91**, 461 (1989); Allen & Tildesley §6.4.1): 40 batches of
1000 samples, the spread of the batch means being the honest standard error
once the batch is long compared with the autocorrelation time. Consecutive MC
samples are correlated, and the naive √(Var/n) understates the error here by a
factor of ~6.

Blocking is also the only numerically viable route: ⟨V²⟩ − ⟨V⟩² at V ≈ 2.4×10⁵
subtracts two ~5.9×10¹⁰ numbers and returns a **negative** variance in double
precision — the previous version of this case printed `se = -nan`, `tol =
-nan`, and therefore failed every comparison it made. Both accumulators are
now shifted by the prediction before squaring.

Fixed seeds make the run deterministic; the check is against the analytic
value, never against our own previous output.


## Reference — Widom hard-sphere free volume

### Formula

Widom (1963), *J. Chem. Phys.* **39**, 2808, DOI 10.1063/1.1734110:

\[
\mu_{\mathrm{ex}} = -k_B T \ln\langle e^{-\beta\Delta U}\rangle
\]

Fixed host: one hard sphere of diameter \(\sigma\) at the origin in a cubic
cell of edge \(L>2\sigma\) (exclusion ball does not wrap into itself).
Test particle overlaps if \(r<\sigma\):

\[
P_{\mathrm{insert}} = 1 - \frac{4\pi\sigma^3}{3L^3}\,,
\qquad
\mu_{\mathrm{ex}} = -k_B T\ln P_{\mathrm{insert}}.
\]

This is a **snapshot** free-volume golden (not bulk HS fluid CS μ).

### Parameters

| Quantity | Value |
|---|---|
| σ | 2.0 Å |
| L | 12.0 Å |
| T | 298.15 K |
| n_trials | 500000 |

### Pass

| Quantity | Tolerance |
|---|---|
| ⟨e⟩ vs \(P_{\mathrm{insert}}\) | 0.005 absolute |
| μ_ex vs analytic | \(0.002\,k_B T\) |


## Reference — μVT (grand-canonical) MC of the truncated-and-shifted LJ vapor

### What is checked

⟨N⟩ from grand-canonical Monte Carlo at two activities, against the **exact
Mayer activity expansion** of the same potential:

ρ(z) = z + 2·b₂·z² + O(z³) = z − 2·B₂(T)·z² + O(z³)

- T. L. Hill, *An Introduction to Statistical Thermodynamics* (Dover 1986),
  §15-3, eq. (15-24): the activity expansion ρ = Σ_j j·b_j·z^j with b₁ = 1 and
  b₂ = −B₂.
- D. A. McQuarrie, *Statistical Mechanics* (University Science Books 2000),
  eq. (12-30): B₂(T) = −2π ∫₀^∞ [e^{−βu(r)} − 1] r² dr.
- The μVT acceptance rules themselves are Frenkel & Smit, *Understanding
  Molecular Simulation* (2nd ed. 2002), §5.6.1, Algorithms 13–14, with the
  activity z ≡ e^{βμ}/Λ³ (their eq. 5.6.9) and Λ = h/√(2πmk_BT).

Matching ⟨N⟩(z) at **two** activities pins the full μVT machinery
quantitatively — the V/(Λ³(N+1)) and Λ³N/V prefactors, the μ convention, the
thermal de Broglie wavelength, the variable-N frame bookkeeping, and
insert/delete rollback. This is the "⟨N⟩ response to μ must satisfy the
analytic μVT relation" detailed-balance criterion of the spec.

### Why an analytic reference and not a NIST state point

The engine's `PairLJCut` is **cut-and-shifted** (energy shift at r_c, no tail
correction). The NIST SRSW μVT benchmarks are computed for r_c = 3σ **with
long-range corrections** (or with *linear-force* shifting), i.e. a different
Hamiltonian — those numbers do not apply to this potential. Rather than
silently compare across conventions, B₂ is evaluated by quadrature **for
exactly the sampled potential** (u_s(r) = u_LJ(r) − u_LJ(r_c), zero beyond
r_c), making the reference convention-exact. Simpson quadrature with 2·10⁵
panels is converged far below the statistical tolerance.

### State points

| Quantity | Value |
|---|---|
| Potential | LJ 12-6, cut AND shifted at r_c = 2.5σ |
| ε/k_B, σ | 120 K, 3.4 Å (argon-flavoured) |
| T* | 1.5 (supercritical — no phase boundary anywhere near) |
| Box | L = 12σ, V = 1728σ³ |
| z·σ³ | 0.010 and 0.030 (⟨N⟩ ≈ 18 and 55) |
| Moves | 50% single-particle translation, 50% insert/delete |

### Tolerance

5× the standard error of the block-averaged ⟨N⟩ (correlation-inflated:
samples 10 sweeps apart decorrelate over ~⟨N⟩ sweeps) **plus** the magnitude
of the first *neglected* term of the activity expansion, `3·b₃·z³·V`. The
third cluster integral obeys `b₃ = 2b₂² − B₃/2` (Hill §15-3); at T* = 1.5 the
LJ third virial coefficient B₃ is positive and partially cancels, so
`|b₃| ≤ 2.5·b₂²` is a safe bound (b₂*² ≈ 3.17, B₃* ≈ +2). At zσ³ = 0.03 this
allows ~1.1 atoms on ⟨N⟩ ≈ 57 — and the *measured* excess over the
second-order prediction, ≈ +1.1 atoms, is exactly this third-order physics
showing up with the predicted sign and size. At zσ³ = 0.01 the allowance
shrinks to 0.04 atoms and the statistical term dominates. Fixed seeds make
the run deterministic; the check is against the analytic value, never
against our own previous output.
