# gpw-d4-stress — offline pin notes

Stress productization of DFT-D4 on the same periodic water dimer as
`gpw-d4-water`.

- Geometry / POSCAR / CP2K QS input: reuse `../gpw-d4-water/`.
- Hard E_disp pin: dftd4 3.7.0 (CP2K Arrhenius lacks DFTD4).
- Decisive stress gate: ATV analytic fixed-P vs multi-SCF strain-FD
  (internal dual-gate); no absolute CP2K STRESS_TENSOR pin on this host.

Never launched from ctest.
