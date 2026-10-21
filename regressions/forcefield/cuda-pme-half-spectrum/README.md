# PME half-spectrum representation

This regression evaluates the public CUDA force pipeline against an independent
full discrete Fourier sum. The reference evaluates the cardinal spline
from its closed truncated-power polynomial, explicitly includes every full-grid
mode, and differentiates its energy analytically. It uses neither cuFFT nor the
device spline recurrence.
The inverse cell uses the motion-state precision: FP64 in mixed/double mode,
matching the canonical cell used to construct the reciprocal solver.

Four order-6 cases combine even/odd final grid dimensions (9×10×12 and 9×10×15)
with orthorhombic/skew cells. A deliberately large alpha makes Nyquist terms
visible. No real-space pair is within the 0.1 Å cutoff. This checks the discrete
PME functional and its force gradient, including the real projection of
triclinic Nyquist modes. A fifth, order-4 skew case checks the runtime-order
kernel in addition to the specialized orders. Physical Ewald accuracy is checked separately by the
NaCl and OpenMM regressions.

Reference: U. Essmann et al., *J. Chem. Phys.* **103**, 8577–8593 (1995),
[doi:10.1063/1.470117](https://doi.org/10.1063/1.470117).

Run `timeout 10s build/regressions/reg_cuda-pme-half-spectrum` in an Arrhenius
GPU batch allocation. Energy and every force component must agree within
5e-8 in double mode or 5e-5 in mixed mode (eV and eV/Å respectively).
