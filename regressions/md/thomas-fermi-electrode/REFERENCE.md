# Reference — Thomas–Fermi finite metallicity electrode (gold capacitor toy)

oracle: metalwalls
oracle_version: MetalWalls git **c8ad175d** (release branch, 2026-06-18), binary
`/nobackup/proj/.../work/opt/metalwalls/mw`, run `mpirun -n 1 mw`
oracle_inputs: `tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/mw_twin/`
verdict: **repinned** — the wave-1..5 pin was a self-golden and was 6.13 % high.

## Science

Thomas–Fermi electrode metallicity (map **E7**): Scalfi, Dufils, Reeves,
Rotenberg, Salanne, *J. Chem. Phys.* **153**, 174704 (2020),
DOI [10.1063/5.0028232](https://doi.org/10.1063/5.0028232) (arXiv:1910.13341).
**Not** available as a LAMMPS ELECTRODE feature (perfect conductor only) →
oracle binds to **MetalWalls**.

MetalWalls implements it in `src/thomas_fermi.f90` as a diagonal add to the
elastance (Hessian) in atomic units,

```fortran
gradQ_V(i,i) = gradQ_V(i,i) + fourpi_ltfsquarei / voronoii   ! 4π l_TF² / v_i
h            = h + fourpi_ltfsquarei * 0.5 * q(i)**2 / voronoii
```

Atomiverse writes the same thing in metal units (`k_e = units::coulomb`,
`d = 1/v`), `src/cpu/dynamics/const_pot_bo.cpp`:

\[
A_{ii}\;\mathrel{+}= 4\pi\,k_e\,l_{\mathrm{TF}}^{2}\,d .
\]

The two are the *same* expression: \(4\pi l^2/v\) [Ha/e²] with \(l,v\) in bohr
times 27.2114 eV/Ha equals \(4\pi k_e l^2/v\) [eV/e²] with \(l,v\) in Å, because
\(1\,\mathrm{Ha}\cdot a_0 = 14.3996\ \mathrm{eV\,Å}\). Perfect metal is
\(l_{\mathrm{TF}}=0\) (MetalWalls skips the term when `ltf <= 0`).

## System (gold capacitor toy)

Empty parallel-plate capacitor (no electrolyte):

| Quantity | Atomiverse (eV, Å, V) | MetalWalls twin (Ha, bohr, e) |
| --- | --- | --- |
| Au lattice \(a_0\) | 4.07 Å | — |
| Site density \(d = 4/a_0^3\) | 0.059330329963 Å⁻³ | `voronoi_volume 113.741732075823890` bohr³ = \(1/d\) |
| Plane spacing \(a_0/2\) | 2.035 Å | — |
| Planes per electrode | 3 (1 site per plane, all at \(x=y=0\)) | 3 `Au1` + 3 `Au2` |
| \(L_x = L_y\) | 4.07 Å | 7.691185327226886 bohr |
| \(L_z\) | 80 Å | 151.178089970061620 bohr (`num_pbc 2` ⇒ z aperiodic) |
| First-plane gap | 20.0 Å | — |
| \(\Psi_{\mathrm{bot}},\Psi_{\mathrm{top}}\) | 0 V, 1 V | `potential 0.0`, `potential 0.03674932217565499` Ha/e |
| Gaussian \(\eta = 1/0.56\) | 1.785714285714 Å⁻¹ | `charge gaussian 0.944959305183928` bohr⁻¹ |
| \(l_{\mathrm{TF}}\) | 0 and 2.0 Å | `thomas_fermi_length` 0.0 and 3.779452249251540 bohr |
| Electroneutrality | `electroneutrality = true` | `charge_neutrality true` |
| Solver | dense Cholesky + projector | `electrode_charges matrix_inversion` |

Atom \(z\) [Å]: bot `{5.0, 7.035, 9.07}`, top `{29.07, 31.105, 33.14}`
(types 1 / 2) → bohr `{9.448630623128851, 13.294223286742294,
17.139815950355736}` and `{54.934338442871145, 58.779931106484582,
62.625523770098027}`.

Conversions used (explicit, no library): 1 bohr = 0.529177210903 Å,
1 Ha = 27.211386245988 eV, so 1 V = 1/27.211386245988 = 0.03674932217565499 Ha/e
and \(\eta\,[\mathrm{bohr}^{-1}] = \eta\,[\mathrm{Å}^{-1}]\times 0.529177210903\).
`voronoi_volume` \(= a_0^3/4 = 16.854785750\ \mathrm{Å^3} = 113.741732076\
\mathrm{bohr^3}\) — which reproduces MetalWalls' own gold value
`113.74173222606741` from `example/thomas-fermi-gold-capacitor`, an independent
check that the two codes mean the same thing by "atomic density".

MetalWalls needs ≥ 2 mobile non-electrode atoms (velocity init removes the COM
velocity; with one mobile ion that zeroes the kinetic energy and the rescale
factor becomes `sqrt(T/0)`). The twin therefore carries two `Xx` probes with
`charge point 0.0`, which enter neither the RHS, nor `total_ions_charge`, nor
the elastance — the capacitor stays empty.

## Raw MetalWalls output (verbatim)

`mw_twin` with `thomas_fermi_length 0.0` — `total_charges.out`:

```
# Total charges on electrodes in atomic unit
# ------------------------------------------
# Charge:   1 au = 1.602176565e-19 C = 1 e
      +0 -4.354124119406108E-003 +4.354124119406114E-003
```

`charges.out` (per site, bottom plane 1→3, top plane 4→6):

```
-2.142725238543552E-004
-5.786374304750878E-004
-3.561214165076665E-003
+3.561214165076682E-003
+5.786374304750617E-004
+2.142725238543711E-004
```

`energies_breakdown.out` (`Hamiltonian / Potential / Coulomb / Thomas-Fermi`):

```
      0   2.770123908402422E-003  -8.000555502842363E-005  -8.000555502842363E-005   0.000000000000000E+000
```

`mw_twin` with `thomas_fermi_length 3.779452249251540` (= 2 Å):

```
      +0 -3.707555065364344E-003 +3.707555065364343E-003
```

```
-7.494127825241582E-004
-1.061846055705252E-003
-1.896296227134934E-003
+1.896296227134934E-003
+1.061846055705251E-003
+7.494127825241580E-004
```

```
      0   2.782004395640317E-003  -6.812506779052898E-005  -7.646570025981265E-005   8.340632469283672E-006
```

### Conversion of the MW numbers

**Charge needs no conversion.** MetalWalls' atomic unit of charge *is* the
elementary charge (its own header: `1 au = 1.602176565e-19 C = 1 e`), so
\(Q_{\mathrm{top}}\) is directly the Atomiverse `total_charge(1)` in e. All the
unit work is on the **inputs** (table above).

**Capacitance** does need one. MetalWalls' \(\Delta\Psi\) is 0.03674932217565499
Ha/e, so

\[
C_{\mathrm{MW}}=\frac{4.354124119406114\times10^{-3}\,e}
{3.674932217565499\times10^{-2}\,\mathrm{Ha}/e}
=1.184816\times10^{-1}\ e^2/\mathrm{Ha},
\]

and \(1\,e^2/\mathrm{Ha} = (1/27.211386245988)\,e^2/\mathrm{eV} = 3.674932\times
10^{-2}\ e/\mathrm{V}\), giving
\(C = 1.184816\times10^{-1}\times3.674932\times10^{-2} = 4.354124\times10^{-3}\
e/\mathrm{V}\) — identical to \(|Q|\) because the twin's \(\Delta\Psi\) is
exactly 1 V by construction. (Dimensional sanity: e/V = e²/eV; dividing a charge
by a potential *must* leave one power of e over energy, which it does.)

**Thomas–Fermi energy** as an independent check of the convention:
\(E_{\mathrm{TF}} = \sum_i (4\pi l^2/2v)\,q_i^2\) with \(l = 3.779452249\) bohr,
\(v = 113.741732076\) bohr³ gives \(4\pi l^2/2v = 0.789050\) and
\(\sum q_i^2 = 1.0570148\times10^{-5}\), i.e. \(8.3404\times10^{-6}\) Ha —
matching MW's printed `8.340632469283672E-006` Ha (= \(2.2696\times10^{-4}\) eV).

**Electrostatic energy** cross-check: for a capacitor at fixed \(\Psi\) the
printed potential energy is the grand potential \(-\tfrac12\Psi Q\):
\(-\tfrac12\times3.674932\times10^{-2}\times4.354124\times10^{-3} =
-8.0004\times10^{-5}\) Ha vs MW's `-8.000555E-005` Ha.

## What changed and why (old pin vs new pin)

| \(l_{\mathrm{TF}}\) [Å] | old pin \(Q_{\mathrm{top}}\) [e] | **MetalWalls c8ad175d** [e] | relative |
| --- | --- | --- | --- |
| 0.0 | 4.620945546104e−03 | **4.354124119406114e−03** | **+6.128 %** |
| 2.0 | 3.934601758679e−03 | **3.707555065364343e−03** | **+6.124 %** |

The old pin came from
`tools/metalwalls_oracle/.../pin_gold_cap.py`, a Python script that
**re-implemented `ConstPotBO::build_matrix_`'s truncated \(xy\)-image sum at
`xy_image_range = 40`**. It was a golden snapshot of Atomiverse's own numerics
wearing an oracle's name, so it agreed to ~1e−12 by construction and could not
detect the error. That script is deleted; the pin is now MetalWalls output.

**The discrepancy is truncation, not physics.** `ConstPotBO`'s elastance is a
direct square-image lattice sum truncated at half-range \(R\); the exact
2D-periodic answer (MetalWalls `num_pbc 2`, Ewald) is the \(R\to\infty\) limit.
Each pair misses a lattice tail
\(\int_{RL}^{\infty}2\pi\rho\,\mathrm{d}\rho\,[\,(\rho^2+z^2)^{-1/2}-\rho^{-1}]/A
\simeq -\pi z^2/(R L A)\), i.e. the error decays only as \(O(1/R)\) and grows as
\(z^2\). This toy has a 20 Å gap over a 4.07 Å lateral cell (\(z/L\approx 6\)),
which is the worst possible ratio, so \(R=40\) is 6 % off:

```
l_tf = 0.0 A     MetalWalls c8ad175d Q_top = 4.354124119406e-03 e
  truncated R=   40  Q_top=4.620945546104e-03 e  (+6.1280 % vs MW)
  truncated R=  160  Q_top=4.417905982930e-03 e  (+1.4649 % vs MW)
  truncated R=  320  Q_top=4.385778337944e-03 e  (+0.7270 % vs MW)
  truncated R=  640  Q_top=4.369898601631e-03 e  (+0.3623 % vs MW)
  truncated R= 1280  Q_top=4.362004736725e-03 e  (+0.1810 % vs MW)
  truncated R= 2560  Q_top=4.358069301166e-03 e  (+0.0906 % vs MW)
  2D Ewald (exact, alpha-independent)  4.354141528379e-03 e  (+0.0004 % vs MW)

l_tf = 2.0 A     MetalWalls c8ad175d Q_top = 3.707555065364e-03 e
  truncated R=   40  Q_top=3.934601758679e-03 e  (+6.1239 % vs MW)
  truncated R=  640  Q_top=3.721124876667e-03 e  (+0.3660 % vs MW)
  truncated R= 1280  Q_top=3.714340036935e-03 e  (+0.1830 % vs MW)
  2D Ewald (exact, alpha-independent)  3.707577384657e-03 e  (+0.0006 % vs MW)
```

(`tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/tf_lattice_convergence.py`;
the "2D Ewald" column is a third, independent implementation — real-space erfc +
reciprocal `erfcx` + the \(k=0\) slab term — which lands on MetalWalls to 4e−6
relative and confirms both codes, not just one.)

**Atomiverse's model is correct.** The Gaussian elastance
(\(\eta_{ij}=\eta_i\eta_j/\sqrt{\eta_i^2+\eta_j^2}\), self
\(k_e\eta\sqrt{2/\pi}\)), the TF hardness, the sign convention
(\(\Psi_{\mathrm{top}}=+1\)V ⇒ \(Q_{\mathrm{top}}>0\)) and the
\(S=A^{-1}-A^{-1}\mathbf{1}\mathbf{1}^\mathsf{T}A^{-1}/(\mathbf{1}^\mathsf{T}A^{-1}\mathbf{1})\)
neutrality projector all agree with MetalWalls exactly — the projector is
literally the same algebra as `MW_matrix_inversion::solve`. Only the lattice-sum
truncation differed. The **ratio** \(C(l_{\mathrm{TF}}{=}2)/C(0)\) was already
right to 4e−5 (0.851471 old pin vs 0.851504 MW) because the truncation error is
very nearly a common factor on the whole elastance — which is exactly why a
dual-gate-only test never caught it.

**Regression change:** `xy_image_range` 40 → **1280**. That is `ConstPotBO`'s
*convergence knob*, not a property of the system, and the tolerance was **not**
touched (still 1 %). At R = 1280 Atomiverse sits 0.18 % from MetalWalls.

## What Atomiverse asserts

Public API: `cpu::md::ConstPotBO` members `l_tf`, `site_density`,
`capacitance()`; still a Component on `md::Phase::pre_force`.

| Check | Tol |
| --- | --- |
| \(\lvert\sum Q_i\rvert\) | \(\le 10^{-10}\) e |
| \(Q_{\mathrm{top}}(l_{\mathrm{TF}}=0)\) vs MW | **≤ 1%** rel |
| \(Q_{\mathrm{top}}(l_{\mathrm{TF}}=2)\) vs MW | **≤ 1%** rel |
| \(C(l_{\mathrm{TF}}=2) < C(l_{\mathrm{TF}}=0)\) | exact |
| \(C\) dual-gate vs MW at both \(l_{\mathrm{TF}}\) | **≤ 1%** rel |
| \(\mathrm{sign}(Q_{\mathrm{top}})=+1\) at \(\Delta\Psi=+1\) V | exact |
| `kAtom_charge` write-back sum == `total_charge(1)` | \(10^{-12}\) e |

## Perf

```
perf_exempt: false
ratio_atv_over_oracle: 4.4
within_10x: yes
```

Same system, same answer, single rank each, `/usr/bin/time` wall clock,
`OMP_NUM_THREADS=1`:

| | wall | per elastance solve |
| --- | --- | --- |
| MetalWalls `mpirun -n 1 mw`, `mw_twin/ltf2` (1 solve) | 0.24 / 0.20 / 0.19 s | **0.20 s** |
| `reg_thomas-fermi-electrode` at `xy_image_range = 1280` (4 solves) | 3.54 s | **0.885 s** |

\(\Rightarrow\) `ratio_atv_over_mw = 4.4`. The gap is algorithmic and expected:
MetalWalls sums the 2D-periodic elastance by Ewald (converged, \(k_{\max}=14\)
here), while Atomiverse evaluates \(21\times2561^2 \approx 1.4\times10^{8}\)
`erf` images per solve just to reach 0.18 %. Atomiverse's 3.54 s is essentially
all lattice sum (≈6 ns per image term, no measurable fixed overhead), whereas
MetalWalls' 0.20 s is dominated by MPI startup and I/O — so the ratio *flatters*
Atomiverse and the true kernel gap is larger. Closing it means giving
`ConstPotBO` a 2D Ewald elastance; the truncated sum is the foothold and the
\(O(1/R)\) tail above is the argument for replacing it.

## Repro (offline only — never ctest)

```bash
# oracle
cp tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/mw_twin/data.inpt /tmp/tf/
sed 's/@LTF@/0.0/' .../mw_twin/runtime.inpt > /tmp/tf/runtime.inpt   # or 3.779452249251540
module load buildenv-gcc/2026.03-mpich
export LD_LIBRARY_PATH="$LIBRARY_PATH:$LD_LIBRARY_PATH"
cd /tmp/tf && mpirun -n 1 /path/to/metalwalls/mw    # -> total_charges.out

# convergence diagnosis
python3 tools/metalwalls_oracle/inputs/metalwalls/thomas-fermi-electrode/tf_lattice_convergence.py
```
