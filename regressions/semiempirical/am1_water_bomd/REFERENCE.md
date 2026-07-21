# Reference — AM1 water BOMD smoke

## Method

AM1 (Dewar et al., *J. Am. Chem. Soc.* **107**, 3902 (1985)).

## Check

`cpu::md::MD` + `VelocityVerlet` + `AM1Evaluator` on force phase, 20 steps
at `dt = 0.5` fs, mild `InitVelocity` (50 K):

- no throw
- PE finite and negative (bound molecule)
- KE finite and non-negative
- no energy explosion

Workflow pin only — not NVE conservation or literature trajectory.
