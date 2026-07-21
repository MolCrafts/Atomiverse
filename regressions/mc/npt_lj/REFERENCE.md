# Reference — NPT (isothermal–isobaric) MC of the truncated-and-shifted LJ vapor

## What is checked

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

Any error in the P·ΔV term, the N·k_BT·ln(V'/V) Jacobian term, the
uniform-in-V proposal, or the box+coordinate rollback moves ⟨V⟩ off this
value — the ln-term alone is a ~1% effect at N = 128, five times the
statistical tolerance.

## State point

| Quantity | Value |
|---|---|
| Potential | LJ 12-6, cut AND shifted at r_c = 2.5σ |
| ε/k_B, σ | 120 K, 3.4 Å (argon-flavoured) |
| T* | 1.5 (supercritical) |
| N | 128 |
| P* = Pσ³/ε | 0.0297 (targets ρσ³ ≈ 0.02) |
| Moves | 80% single-particle translation, 20% volume (δV = ⟨V⟩/√N) |

At ρσ³ ≈ 0.02 the O(B₃ρ²) term is ~10⁻⁴ relative — the second-order
prediction is exact at the quoted tolerance.

## Tolerance

5× the standard error of the block-averaged ⟨V⟩ (correlation-inflated for the
20% volume-move fraction) **plus** 0.2% of ⟨V⟩ for the virial-series
truncation. Fixed seeds make the run deterministic; the check is against the
analytic value, never against our own previous output.
