# Reference — PIMD free-particle skeleton

## What this checks

Infrastructure for path-integral MD:

1. `IdentityReplication` expands P beads from one Frame (second consumer after REMD).
2. Closed-chain harmonic springs with
   \(k = m\,P/(\beta^2\hbar^2)\) in Atomiverse energy units.
3. Nested `cpu::md::MD` per bead; collapse writes the **centroid**.
4. Free particle (no external potential): thermal init opens the polymer
   (mean bead–centroid spread > 0).

This is **not** a full quantum estimator regression (no RPMD \(c_{vv}\), no
harmonic-oscillator energy vs \(\hbar\omega\)).

## Literature

- Chandler & Wolynes, *J. Chem. Phys.* **74**, 4078 (1981)
- Tuckerman, *Statistical Mechanics: Theory and Molecular Simulation*, ch. on PIMD
- Ceriotti et al., i-PI / ring-polymer MD reviews

## Parameters

| Quantity | Value |
|---|---|
| P | 4 |
| T | 300 K |
| m | 1 amu |
| dt | 0.25 fs |
| blocks × steps | 15 × 20 |
