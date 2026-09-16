# qmmm_opt_h2_ar

**What:** Geometry optimization of mechanical QM/MM H₂ (DFTB2) + Ar with
H–Ar 12-6 LJ cross terms, composed as `Minimize.set_evaluator(QmmmMechanical)`.

**Physics pin:** At \(r < r^\*=2^{1/6}\sigma\), the LJ force is **repulsive**.
Starting Ar at 2.2 Å with \(\sigma=3.4\) Å must increase the H–Ar distance under
L-BFGS. Absolute energy is method-dependent (DFTB2 + shifted LJ); the regression
gates motion + residual force / `ForceConverged`, not a literature total energy.

**Composition:** demonstrates QM/MM geometry optimization without a dedicated
`QmmmDriver` (harness `prefer-composition`).
