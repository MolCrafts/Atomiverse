# longrange/ewald-pme

One case; every section below is run by `atv_regressions --case longrange/ewald-pme` and keeps its own oracle record.


## CUDA EWALD rock-salt NaCl

oracle: literature

J. M. Ziman, *Principles of the Theory of Solids*, 2nd ed., Cambridge
University Press (1972), §2.3, gives the NaCl Madelung lattice sum.
The same reference and constant are used by `md/npt_pme_nacl`.

The 64 unit-charge ions alternate sign on a cubic lattice with nearest-neighbor
spacing d = 2.82 Å and box edge 4d. The lattice energy is

    E = −N M k_e / (2d),  M = 1.747564594633182.

The case composes CUDA CoulLong with KSpaceEwald and performs exactly one
VelocityVerlet step with zero initial velocities and forces. Positions do not
move before the force evaluation, so the energy is the input lattice's energy.
The cutoff is 1.9d, strictly below half the periodic box edge.

The original unit-test assertion is preserved: abs(E/E_reference − 1) ≤ 2e-3.
No oracle executable is invoked. Run on Arrhenius; the case is designed to
finish within 10 seconds on a GH200, including CUDA initialization.


## CUDA PME rock-salt NaCl

oracle: literature

J. M. Ziman, *Principles of the Theory of Solids*, 2nd ed., Cambridge
University Press (1972), §2.3, gives the NaCl Madelung lattice sum.
The same reference and constant are used by `md/npt_pme_nacl`.

The 64 unit-charge ions alternate sign on a cubic lattice with nearest-neighbor
spacing d = 2.82 Å and box edge 4d. The lattice energy is

    E = −N M k_e / (2d),  M = 1.747564594633182.

The case composes CUDA CoulLong with KSpacePME and performs exactly one
VelocityVerlet step with zero initial velocities and forces. Positions do not
move before the force evaluation, so the energy is the input lattice's energy.
The cutoff is 1.9d, strictly below half the periodic box edge.

The original unit-test assertion is preserved: abs(E/E_reference − 1) ≤ 3e-3.
No oracle executable is invoked. Run on Arrhenius; the case is designed to
finish within 10 seconds on a GH200, including CUDA initialization.


## PME half-spectrum representation

This regression evaluates the public CUDA force pipeline against an independent
full discrete Fourier sum. The reference evaluates the cardinal spline
from its closed truncated-power polynomial, explicitly includes every full-grid
mode, and differentiates its energy analytically. It uses neither cuFFT nor the
device spline recurrence.

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


## Reference — slab 2D Ewald foothold (Yeh–Berkowitz / LAMMPS slab)

oracle: lammps  
oracle_version: develop / patch_30Mar2026-1074-g7f680de296 (`lmp` at pin time)

### System

Parallel-plate electrostatic toy (planar electrodes + vacuum gap):

| Quantity | Value |
|---|---|
| Box | \(12 \times 12 \times 20\) Å orthorhombic |
| Boundary (LAMMPS) | `p p f` (non-periodic z; ATV uses 3D min-image; vacuum + cutoff make them equivalent here) |
| Atoms | 4× charge \(+1\,e\) at \(z=5\) Å on a 6 Å xy grid; 4× \(-1\,e\) at \(z=15\) Å |
| Pair | `coul/long` cutoff **5.9 Å** (no same-plane pairs at 6 Å; real-space \(E=0\)) |
| KSpace | `ewald` accuracy `1e-6` |
| α (`gewald`) | **0.5** Å⁻¹ (fixed; matches ATV) |
| Slab | `kspace_modify slab 3.0` → \(L_z' = 60\) Å, \(V' = 8640\) Å³ |

Dipole: \(M_z = 4\cdot(+1)\cdot 5 + 4\cdot(-1)\cdot 15 = -40\) e·Å.

### Offline pin (LAMMPS `run 0`)

Input: `tools/metalwalls_oracle/inputs/lammps/slab-2d-ewald-foothold/in.lammps`

```text
E_TOTAL = 63.0848970842656   eV
E_LONG  = 63.0848970842656   eV   (E_COUL = 0)
mean Fz (+ plane) = +2.51351 eV/Å
mean Fz (− plane) = −2.51351 eV/Å
```

Physics: 3D Ewald on the z-extended cell + Yeh–Berkowitz \(k=0\) correction
(Yeh & Berkowitz, *J. Chem. Phys.* **111**, 3155 (1999)); LAMMPS
`Ewald::slabcorr`.

### What Atomiverse asserts

1. **Energy** of `CoulLong` + `KSpaceEwald` with `slab_factor = 3.0` matches
   \(E_{\mathrm{pin}}\) to relative \(10^{-4}\) or absolute \(5\times 10^{-3}\) eV.
2. **Mean \(F_z\)** on each plane matches the pin to \(10^{-3}\) eV/Å.
3. Without slab (`slab_factor = 0`) the energy **differs** from the slab
   energy by more than 1 eV (guards a no-op implementation).

ctest / the regression binary **never** launches LAMMPS.

### Perf (fair wall)

Same 8-atom geometry, `OMP_NUM_THREADS=1`, 2000 force evaluations after warm-up.
LAMMPS: `run 2000` with `fix setforce 0` (loop time). ATV: hot-path
`CoulLong`+`KSpaceEwald` launch only (build once; no Evaluate re-init).

| Engine | Wall (s) | Notes |
|---|---|---|
| LAMMPS `lmp` | **0.205141** | loop time, 2000 steps, develop/30Mar2026 |
| Atomiverse | **1.330859** | hot-path 2000 force evals, half-sphere Ewald |

ratio \(T_{\mathrm{atv}}/T_{\mathrm{oracle}} = 1.330859 / 0.205141 \approx **6.49**\)  
`within_10x`: **true** (≤ 10.0)

One optimize applied for the gate: EwaldSum reciprocal sum over the positive
hemisphere only with prefactor \(4\pi/V\) (LAMMPS-equivalent; was full sphere
at \(2\pi/V\)).
