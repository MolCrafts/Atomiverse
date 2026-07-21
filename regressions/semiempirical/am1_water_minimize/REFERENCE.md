# Reference — AM1 water Minimize

## Method

AM1 (Dewar et al., *J. Am. Chem. Soc.* **107**, 3902 (1985)).

## Check

Public `cpu::minimizer::Minimize` + `AM1Evaluator` on a distorted water:

1. Start energy is finite and negative; residual `max|F|` is non-trivial.
2. After ≤15 L-BFGS iterations, **energy decreases** and/or **max|F|** drops
   by ≥5% (no throw; forces from NDDO SCF analytic gradient).

This is a workflow pin (SE force drives Optimize), not a literature geometry
oracle.
