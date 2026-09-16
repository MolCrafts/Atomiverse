# Reference — finite-field + single electrode (3D, Dufils-style)

oracle: metalwalls
oracle_version: **MetalWalls git c8ad175d** (release branch, 2026-06-18) — the
real binary, run `mpirun -n 1`. Binary self-check: upstream
`tests/external_field/electric_field_matrix_inversion` reproduces its shipped
`charges.ref` to 1.9e-12 absolute (max |q| = 0.213 e) and its
`energies_breakdown.ref` to 6e-13 relative.

verdict: **not_expressible** — the *system this regression simulates* has no
MetalWalls counterpart; MW refuses it at input parsing, by design. The pinned
numbers and their 1e-10 / 1e-12 tolerances are **unchanged**. What the real
binary confirmed is every **convention** the pin rests on; what it exposed is a
separate fidelity gap in Atomiverse's 3D-PBC elastance (last section). No
Atomiverse code was changed.

Supersedes the wave-1 header `oracle_version: literature analytic` and the
"Why not LAMMPS" note: the reason this slug is MetalWalls-shaped is still
correct (LAMMPS ELECTRODE `ffield on` needs exactly two electrode groups), but
the fallback rationale "no MW binary available" no longer applies.

## Unit conversion used throughout

MetalWalls is entirely atomic units; the Atomiverse public surface is eV / Å /
e / V. With Bohr = 0.529177210903 Å and Ha = 27.211386245988 eV:

| quantity | ATV (public) | MW (a.u.) |
| --- | --- | --- |
| box edge | 20 Å | 37.794522492515405 bohr |
| electrode \(z\) | 10 Å | 18.897261246257703 bohr |
| \(\eta\) | 2.0 Å⁻¹ | 1.058354421806 bohr⁻¹ |
| \(\Psi\) | 0 V | 0 Ha/e |
| \(E_z\) | 0.1 V/Å | 1.944690381148888e-03 a.u. |
| elastance | 22.978508853084353 eV/e | 0.844444624958137 Ha/e² |
| polarization | 1 e/Å² | 0.529177210903² = 0.2800283 e/bohr² |

Dimensional check of the field factor (not a number match): 1 a.u. of field is
Ha/(e·bohr) = (27.211386245988 eV/Ha)/(0.529177210903 Å/bohr) = 51.4220674763
V/Å, so 0.1 V/Å = 1.9446903811e-3 a.u. Independent confirmation: MW's own
`example/electric-field` — the input distributed *as* the Dufils PRL setup —
uses `amplitude 0.00194469`, the same 0.1 V/Å to six digits.

## What this regression hard-codes

`reg_finite-field-single-electrode.cpp`, four assertions, all derived in-file
from one closed form:

| # | quantity | pinned value | tol |
| --- | --- | --- | --- |
| 1 | electrode charge \(Q\) (standalone `ConstPotBO`) | **0.043518924852505053** e | 1e-10 |
| 2 | \(P_z\) (`ClassicalPolarization`) | **5.4398656065631316e-05** e/Å² | 1e-12 |
| 3 | \(F_z\) on a \(q=+1\) ion (`FixEfield`) | **0.1** eV/Å | 1e-12 |
| 4 | \(Q\) and \(P_z\) through the MD binding path (1 VV step) | same as #1/#2 | 1e-10 |

System: cubic \(L=20\) Å, 3D PBC, **no electrolyte**, ONE Gaussian electrode
site (type 1) at \((0,0,10)\) Å, \(\eta = 2.0\) Å⁻¹, \(\Psi = 0\),
\(\mathbf E = (0,0,0.1)\) V/Å, **electroneutrality off** (free total \(Q\)),
`xy_image_range = 0` (primary cell only — no lattice sum).

Wave-1 provenance: literature analytic (Dufils et al., PRL **123**, 195501
(2019); Reed/Siepmann–Sprik Gaussian kernels),
\(A_{ii}=k_e\eta\sqrt{2/\pi}\), \(Q=(\Psi+E_z z)/A_{ii}\), \(P_z=Qz/V\).

## Conventions CONFIRMED against the real binary

Read from MetalWalls c8ad175d source, and reproduced numerically where a number
exists:

1. **External potential on the BO right-hand side.**
   `src/external_field.f90:264-297` (`Efield_elec_potential`) sets
   \(V_i=-\sum_\alpha r_{i\alpha}E_\alpha\); `src/electrode_charge.F90:88-96`
   (`setup_b`, matrix-inversion branch) forms
   \(b_i = \Psi_i-(\Phi^{\mathrm{melt}}_i+V_i)\) and solves \(AQ=b\). That is
   exactly Atomiverse's
   \((AQ)_i=\Psi_i-\Phi^{\mathrm{elyt}}_i+\mathbf E\cdot\mathbf r_i\)
   (`src/cpu/dynamics/const_pot_bo.cpp:444-447`). Sign, placement and the use of
   the **raw** (unwrapped) coordinate all match.
2. **Force on an electrolyte ion.** `src/external_field.f90:181-193`
   (`melt_forces`, `FIELD_TYPE_ELECTRIC`) is `force(i,α) += q_i*E(α)` — no
   prefactor, no cell dependence. Assertion #3 is therefore MW-exact by
   construction: \(q=1\,e\), \(E_z = 1.944690381148888\times10^{-3}\)
   Ha/(e·bohr) → \(F_z = 1.944690381148888\times10^{-3}\) Ha/bohr =
   **0.1 eV/Å** (1 Ha/bohr = 51.4220674763 eV/Å).
3. **Cell polarization.** `src/external_field.f90:448-484`
   (`elec_polarization`) is \(P_\alpha=\frac1V\sum_i q_i r_{i\alpha}\) —
   identical to `ClassicalPolarization`. Numerically, on the `neutral-2site`
   run below MW's `polarization.out` gives
   \(P_z = 8.332287442536274\times10^{-6}\) e/bohr², and
   \(\sum_i q_i z_i/V = 0.02380411088555103\times18.897261246257701/
   37.794522492515405^3 = 8.3322874425\times10^{-6}\) e/bohr² — 12 digits.
4. **External-field energy.** `src/external_field.f90:244-248` is
   \(H=-V\,\mathbf E\cdot\mathbf P=-\mathbf E\cdot\sum_i q_i\mathbf r_i\). MW
   reports −8.747849400381311e-04 Ha for `neutral-2site`; the closed form gives
   −8.747849400381312e-04 Ha.

So the *formulas* behind the pin are MetalWalls' formulas. The *number* is
another matter.

## Why MetalWalls cannot express THIS system (`mw_expressible = false`)

The literal twin is checked in at
`tools/metalwalls_oracle/inputs/metalwalls/finite-field-single-electrode/exact-atv-system/`
(one electrode site, `charge_neutrality false`, `external_field type E`, same
box/η/Ψ/E). MetalWalls **aborts before the first step**:

```
[error]...cannot use external fields without charge neutrality
          subroutine: read_system_parameters
          file: configuration.f90
```

`src/configuration.F90:624-630`:

```fortran
! External fields need charge neutrality
if ((system%field%field_type.eq.FIELD_TYPE_ELECTRIC      .or.  &
     system%field%field_type.eq.FIELD_TYPE_DISPLACEMENT) .and. &
     .not.system%electrode_charge_neutrality) then
   call MW_errors_runtime_error(..., "cannot use external fields without charge neutrality")
```

This is a **well-posedness** guard, not a missing feature — and the binary lets
us demonstrate why. Two runs of the same MW-legal system differing *only* in the
Ewald splitting (`neutral-2site`: `rcut 18`, `rtol/ktol 1e-12`;
`neutral-2site-alpha2`: `rcut 15`, `rtol/ktol 1e-14`) print these
`hessian_matrix.out` entries:

| case | \(A_{11}\) [Ha/e²] | \(A_{12}\) [Ha/e²] | \(A_{11}-A_{12}\) [Ha/e²] | solved \(Q\) [e] |
| --- | --- | --- | --- | --- |
| `neutral-2site` | 0.7701350687795038 | −0.001776185380416242 | 0.771911254159920 | 2.380411088555103e-02 |
| `neutral-2site-alpha2` | 0.7698164089820876 | −0.002094845177845544 | 0.771911254159933 | 2.380411088555063e-02 |

The **diagonal alone moves by 4.1e-4 relative** with an arbitrary numerical
parameter, while the **neutral combination is invariant to 13 digits** and so is
the solved charge. In 3D PBC only neutral combinations of \(A\) are physical. A
single-site solve with free total \(Q\) reads the diagonal directly
(\(Q=E_z z/A_{11}\)) and would inherit both that \(\alpha\) dependence *and* an
origin dependence — \(Q\propto z\), so translating the cell by one lattice
vector (a symmetry of the periodic system) would triple \(Q\). MetalWalls
refuses to emit such a number.

Dufils' actual single-electrode product keeps the *system* neutral (electrode +
electrolyte); "one electrode instead of two" removes the second wall, not
electroneutrality.

Two escape routes were tried and closed:

- **Neutrality on, one site.** The constraint then fixes \(Q\) completely
  (\(Q=-\sum q_{\mathrm{elyt}}\), field-independent). Case
  `single-site-elastance/` uses the regression's exact geometry and returns
  \(Q = +0.000000000000000\)E+000 exactly. There is no MW configuration in
  which a *single* electrode site carries a field-driven charge.
- **2D PBC.** MW also refuses a finite field with `num_pbc 2`
  (`src/configuration.F90:632-…`, "External fields needs 3D periodic boundary
  conditions"), so a slab geometry is not a way around the guard either.

**Conclusion.** 0.043518924852505053 e is a correct and exactly reproducible
property of the model this regression *declares* — an **isolated** Gaussian
conductor (`xy_image_range = 0`, no lattice sum), used to check that
\(\mathbf E\cdot\mathbf r\) reaches the BO right-hand side with the right sign
and magnitude, that the charge is written to the frame, that
`ClassicalPolarization` reads it, and that the MD binding path reproduces it. It
is **not** a 3D-periodic electrochemical observable and must not be cited as a
MetalWalls-reproducible one. Pin and tolerances kept; the claim attached to them
is narrowed.

## Real-MW cross-check on the nearest legal system (advisory, not asserted)

`tools/metalwalls_oracle/inputs/metalwalls/finite-field-single-electrode/neutral-2site/`
— same cell, same \(\eta\), same \(\Psi\), same \(E_z\), but **two** electrode
sites (\(z=5\) and \(15\) Å) and `charge_neutrality true`.

Raw MW `charges.out` (already in e — no conversion needed):

```
# Electrode atom charges in atomic unit
# step        0
-2.380411088555103E-002
+2.380411088555103E-002
```

Atomiverse `ConstPotBO(2.0)` with `electroneutrality = true` on the identical
6-atom frame (out-of-library consumer, single thread):

```
ATV neutral-2site Q(z=5)  = -2.321419645652141e-02 e
ATV neutral-2site Q(z=15) = +2.321419645652141e-02 e
ATV neutral-2site Pz      =  2.901774557065177e-05 e/A^2   (MW: 2.975513861e-05)
```

**relative_discrepancy = −2.478 %** (ATV low), same sign and size on \(P_z\).

Origin of the gap: `ConstPotBO::build_matrix_`
(`src/cpu/dynamics/const_pot_bo.cpp:305-372`) assembles the elastance as a pure
**real-space** sum — isolated Gaussian self term \(k_e\eta\sqrt{2/\pi}\) plus
\(k_e\,\mathrm{erf}(\eta_{ij}r)/r\) over the primary cell (or, with
`xy_image_range > 0`, a planar \(xy\) image lattice). There is no reciprocal
space and no background term anywhere on the electrode path. For a **3D**
periodic cell — the only geometry in which MetalWalls permits a finite field —
that is not the periodic elastance:

| | ATV | MW (c8ad175d) | ATV/MW |
| --- | --- | --- | --- |
| \(A_{11}-A_{12}\) (2-site, physical) | 0.791526905626131 Ha/e² | 0.771911254159920 Ha/e² | +2.54 % |
| \(A_{11}\) (single site) | 0.844444624958137 Ha/e² = 22.978508853084353 eV/e | 0.7701350687795038 Ha/e² = 20.956442818139610 eV/e | +9.65 % (MW value is α-dependent) |

This is a **capability gap on the E3 finite-field path, not a defect in what
this regression asserts** — the regression declares the isolated model and
reproduces it to machine precision. Closing it means giving `ConstPotBO` a
3D-Ewald elastance (reciprocal-space + background, with the α-cancellation the
table above demonstrates). That is new work, deliberately not attempted in a
re-pin, and is reported to the loop for the queue.

## Perf

```
perf_exempt: true
perf_exempt_reason: the regression's own workload (single electrode site, free total charge, finite field) aborts in MetalWalls c8ad175d during input parsing, so it has no wall-clock twin at all; on the nearest legal twin the two codes are not doing the same work (MW: 3D Ewald elastance + Ewald melt potential every step; ATV finite-field path: real-space min-image erf only, no reciprocal space), so the ratio measures the fidelity gap, not speed
ratio_atv_over_oracle: 0.0026 (advisory only)
within_10x: advisory pass
```

Measured for the record on the `neutral-2site` geometry (6 atoms, 1000 steps,
single rank, same 48-core epyc login node, `OMP_NUM_THREADS=1`):

| code | wall | notes |
| --- | --- | --- |
| MetalWalls `mpirun -n 1` | 2.37 s | 0.16 s of that is the 0-step startup baseline → ~2.2 ms/step |
| Atomiverse `md::MD` + `ConstPotBO(rebuild_matrix_every_step)` + `FixEfield` | 0.0062 s | ~6.2 µs/step |

ratio ATV/MW = 0.0026 total wall (0.0028 startup-corrected). Read this as a
measurement of the missing Ewald work, not as a speed claim.

## Repro

MetalWalls — run each case in a **clean** scratch copy (MW *reads*
`hessian_matrix.inpt` when the file exists, silently reusing a stale matrix):

```bash
bash -lc 'module load buildenv-gcc/2026.03-mpich
          export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
          cd <case-dir>
          mpirun -n 1 /…/work/opt/metalwalls/mw > run.stdout 2>&1'
```

Atomiverse pin (offline only — never ctest):

```python
import math
coulomb = 14.399645
eta, Ez, z, L = 2.0, 0.1, 10.0, 20.0
Aii = coulomb * eta * math.sqrt(2 / math.pi)   # 22.978508853084353 V/e
Q = (Ez * z) / Aii                             # 0.04351892485250505 e
Pz = Q * z / (L ** 3)                          # 5.439865606563132e-05 e/Å²
```
