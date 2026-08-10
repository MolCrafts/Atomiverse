# cuda-gpw-stress

## Twin

Same-settings CPU `gpw_stress` (fixed-P analytic at converged density) vs
public `cuda::gpw::gpw_stress` after CUDA Γ RKS PADE SCF (H₂).

| Setting | Value |
| --- | --- |
| System | H₂, DZV-GTH-PADE / GTH-PADE-q1, 12 Bohr cube |
| CUTOFF | 80 Ry, `ngrids=1`, periodic |
| XC | Padé LDA |
| Strain `h` | 2e-3 (CPU default) |

First cut: CUDA SCF density + host fixed-P assembly (`cpu::gpw::gpw_stress` at
state). Fail-closed multi-k.

## Tolerances

| Quantity | Tol |
| --- | ---: |
| \|E\| dual-gate | 5e-6 Ha |
| max\|Δσ_diag\| | 5e-5 Ha/Bohr³ |
| off-diag | ~0 (1e-8) |

**perf_exempt** small cell.
