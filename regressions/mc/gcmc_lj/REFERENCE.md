# Reference — μVT (grand-canonical) MC of the truncated-and-shifted LJ vapor

## What is checked

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

## Why an analytic reference and not a NIST state point

The engine's `PairLJCut` is **cut-and-shifted** (energy shift at r_c, no tail
correction). The NIST SRSW μVT benchmarks are computed for r_c = 3σ **with
long-range corrections** (or with *linear-force* shifting), i.e. a different
Hamiltonian — those numbers do not apply to this potential. Rather than
silently compare across conventions, B₂ is evaluated by quadrature **for
exactly the sampled potential** (u_s(r) = u_LJ(r) − u_LJ(r_c), zero beyond
r_c), making the reference convention-exact. Simpson quadrature with 2·10⁵
panels is converged far below the statistical tolerance.

## State points

| Quantity | Value |
|---|---|
| Potential | LJ 12-6, cut AND shifted at r_c = 2.5σ |
| ε/k_B, σ | 120 K, 3.4 Å (argon-flavoured) |
| T* | 1.5 (supercritical — no phase boundary anywhere near) |
| Box | L = 12σ, V = 1728σ³ |
| z·σ³ | 0.010 and 0.030 (⟨N⟩ ≈ 18 and 55) |
| Moves | 50% single-particle translation, 50% insert/delete |

## Tolerance

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
