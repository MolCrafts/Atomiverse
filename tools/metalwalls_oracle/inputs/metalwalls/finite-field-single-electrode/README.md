# MetalWalls oracle — finite-field-single-electrode

Real-binary re-pin (2026-07-29). Oracle binary:
`/…/work/opt/metalwalls/mw`, MetalWalls git **c8ad175d** (release, 2026-06-18).

Run any case with:

```bash
bash -lc 'module load buildenv-gcc/2026.03-mpich
          export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
          cd <case-dir>
          mpirun -n 1 /…/work/opt/metalwalls/mw > run.stdout 2>&1'
```

**Gotcha:** MetalWalls *reads* `hessian_matrix.inpt` when it exists. Always run
in a clean scratch copy, never re-run in a directory that already holds a
previous run's output.

All four cases share the Atomiverse regression's cell and electrode:
cubic \(L=20\) Å = 37.794522492515405 bohr, 3D PBC, one Gaussian electrode
species with \(\eta = 2.0\) Å⁻¹ = 1.058354421806 bohr⁻¹, \(\Psi = 0\),
\(E_z = 0.1\) V/Å = 1.944690381148888e-3 a.u.
(Bohr = 0.529177210903 Å, Ha = 27.211386245988 eV, 1 a.u. field = 51.4220674763 V/Å.)

| case | what it is | MW outcome |
| --- | --- | --- |
| `exact-atv-system/` | literal twin of the regression: **1** electrode site at \(z=10\) Å, `charge_neutrality false`, finite field | **MW refuses to start** — `[error]...cannot use external fields without charge neutrality` (`src/configuration.F90:624-630`) |
| `neutral-2site/` | nearest MW-legal analogue: **2** electrode sites at \(z=5,15\) Å, `charge_neutrality true`, finite field | runs; \(Q = \mp2.380411088555103\times10^{-2}\) e |
| `neutral-2site-alpha2/` | same system, different Ewald split (`rcut 15`, `rtol/ktol 1e-14`) | runs; **identical** \(Q\) to 13 digits, **different** \(A_{ij}\) entries |
| `single-site-elastance/` | exact regression geometry (1 site at \(z=10\) Å), no field, neutrality on | runs; prints MW's periodic \(A_{11}\) via `write_matrix true` |

`neutral-2site` vs `neutral-2site-alpha2` is the decisive experiment: MetalWalls'
electrode elastance **diagonal** is Ewald-\(\alpha\)-dependent
(0.7701350687795038 vs 0.7698164089820876 Ha/e²) while the neutral combination
\(A_{11}-A_{12}\) is \(\alpha\)-independent to 13 digits
(0.771911254159920 vs 0.771911254159933 Ha/e²). A non-neutral single-site
constant-potential solve in 3D PBC would therefore depend on an arbitrary
numerical parameter — which is exactly why MW guards against it.

The `X` species in the legal cases is a **charge-0, mass-1** dummy melt species
(4 copies). It is required only because MW divides by the ionic degrees of
freedom when it prints a step (`system.F90:2655`, NaN with zero mobile ions);
with \(q=0\) it contributes nothing to \(b_i\), to the polarization, or to the
external-field energy.

Full analysis, raw MW output and conversions:
`regressions/md/finite-field-single-electrode/REFERENCE.md`.

## Offline analytic pin (Atomiverse's own model, unchanged)

The regression's model is the **isolated** Gaussian self-capacitance
(`xy_image_range = 0`, no lattice sum), not a periodic elastance:

```python
import math
coulomb = 14.399645
eta, Ez, z, L = 2.0, 0.1, 10.0, 20.0
Aii = coulomb * eta * math.sqrt(2 / math.pi)   # 22.978508853084353 V/e
Q = (Ez * z) / Aii                             # 0.04351892485250505 e
Pz = Q * z / (L ** 3)                          # 5.439865606563132e-05 e/Å²
```
