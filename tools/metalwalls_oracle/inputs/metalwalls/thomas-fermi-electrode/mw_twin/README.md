# MetalWalls twin — `thomas-fermi-electrode`

The oracle input behind `regressions/md/thomas-fermi-electrode`. It is the
MetalWalls transcription of the *exact* system that regression simulates: six Au
electrode sites (3 + 3) stacked at \(x=y=0\), \(L_x=L_y=4.07\) Å, first-plane gap
20 Å, \(\Delta\Psi = 1\) V, \(\eta = 1/0.56\) Å⁻¹, Voronoi volume \(a_0^3/4\),
Thomas–Fermi length 0 or 2 Å, electroneutral, matrix inversion.

`runtime.inpt` carries one placeholder, `@LTF@` — the Thomas–Fermi length **in
bohr**. Run it twice:

```bash
module load buildenv-gcc/2026.03-mpich          # never pipe `module load`
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"

for L in 0.0 3.779452249251540; do              # 0 A and 2 A, in bohr
  d=/tmp/tf_$L && mkdir -p $d && cp data.inpt $d/
  sed "s/@LTF@/$L/" runtime.inpt > $d/runtime.inpt
  (cd $d && mpirun -n 1 /path/to/metalwalls/mw > run.stdout 2>&1)
  cat $d/total_charges.out
done
```

Expected `total_charges.out` (columns: step, \(Q_{\mathrm{bot}}\),
\(Q_{\mathrm{top}}\), in e):

```
l_tf = 0 bohr                +0 -4.354124119406108E-003 +4.354124119406114E-003
l_tf = 3.779452249251540     +0 -3.707555065364344E-003 +3.707555065364343E-003
```

Conversions and the full breakdown (charges per site, Coulomb and Thomas–Fermi
energies, the dimensional cross-checks) live in
`regressions/md/thomas-fermi-electrode/REFERENCE.md`.

## Two things that will bite you

**`coulomb_rcut` must be < min(a,b,c)/2.** MetalWalls hard-errors otherwise
(`system.F90`, "Invalid value for ewald rcut"). The lateral cell here is 7.691
bohr, so `coulomb_rcut 3.84` is near the ceiling — that is why `coulomb_rtol` is
1e−9 (small \(r_{\mathrm{cut}}\) forces a large \(\alpha\); MetalWalls picks
\(\alpha=1.146\) bohr⁻¹, \(k_{\max}=14\), which is cheap at this size).

**Two `Xx` probes with `charge point 0.0` are load-bearing.** MetalWalls'
velocity initialisation removes the centre-of-mass velocity
(`MW_velocity::global_scaling`); with a single mobile ion that zeroes the kinetic
energy and the rescale factor becomes `sqrt(T/0)`, so the run aborts with
"Temperature is NaN or meaningless" *before* printing step 0. Two probes give a
finite temperature. Being neutral, they contribute nothing to the electrode RHS,
nothing to `total_ions_charge`, and nothing to the elastance — the capacitor
stays empty, matching the Atomiverse regression.

## Sanity check of the oracle itself

`tests/thomas_fermi/LiCl-Al-inv` from the MetalWalls tree (same TF +
`matrix_inversion` + `charge_neutrality` path) reproduces its shipped
`energies_breakdown.ref` Hamiltonian to 1.0e−12 relative and `charges.ref` to
1.2e−11 with this binary. Run that first if these numbers ever move.
