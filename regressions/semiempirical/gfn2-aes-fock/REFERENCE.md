# Reference — off-eq water GFN2-xTB AES-in-Fock forces vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

## Method

GFN2-xTB nuclear gradient of the Bannwarth 2019 SCC energy with anisotropic
electrostatics **inside the Fock** (tblite `get_potential` + `add_pot_to_h1`):

\[
E = \operatorname{Tr}(P H^{0})
  + E_{\mathrm{IES}}
  + E^{(3)}
  + E_{\mathrm{rep}}
  + E_{\mathrm{D4}}
  + E_{\mathrm{AES}}
\]

\(E_{\mathrm{AES}}\) is charge–dipole, dipole–dipole, charge–quadrupole +
dipole/quadrupole AXC. CAMM moments enter the SCC Hamiltonian so the density
is stationary with respect to AES. This is the deepen of `gfn2-xtb-forces`,
which kept AES as a post-SCC energy (angular \(F_{H1,y}\) hole).

Reference: Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI 10.1021/acs.jctc.8b01176. Formula twin: tblite
`coulomb/multipole.f90` `get_potential`, `scf/potential.f90` `add_vmp_to_h1`.
**Not** DFTB+; **not** OpenMOPAC; **not** GFN0/1.

## Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host). Command:

```text
XTBPATH=.../share/xtb OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 2 --chrg 0 --uhf 0 --acc 1.0 --grad
```

Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/`
(`mol.xyz` + `xtb.out.txt` + `gradient.txt`). Nothing is executed at ctest
time. xtb prints \(\nabla E\) [Eh/a0]; ATV stores \(F=-\nabla E\) [eV/Å]
with CODATA \(a_0\) / Hartree from `units.h`.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−5.058625058504** | Eh |
| \(\|\nabla E\|\) | 0.109121448851 | Eh/a0 |
| Wall (xtb, 1 thread) | 0.018 | s |

### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| O | −7.5243103995119e−2 | −1.7310256877367e−2 | ~0 |
| H1 | +7.5511039446358e−2 | +1.5529339054184e−2 | ~0 |
| H2 | −2.6793545123835e−4 | +1.7809178231828e−3 | ~0 |

## Geometry

Gas-phase water, charge 0, singlet, **off equilibrium** (O–H stretched
0.96 → 1.10 Å) so \(|F|\neq 0\). Cubic box 20 Å (isolation only). Same
geometry as `reg_gfn2-xtb-forces`.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 1.10  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

## Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E\) | **−5.058625058504** | Eh | \(1\times10^{-2}\) eV |
| \(F_{O,x}\) | **+3.869155970641** | eV/Å | \(0.35\) |
| \(F_{H1,y}\) | **−0.7986** | eV/Å | printed, **not gated** |

\(F = -\nabla E \times\) `units::kHartreePerBohrInEvPerAngstrom`.

**Tolerance rationale.** AES-in-Fock changes the SCC density; stretch
\(F_{O,x}\) residual is **0.306** eV/Å (was \(\sim 0.09\) post-SCC).
Off-eq energy residual **9.19 meV**. Analytic \(F_{H1,y}=+0.136\) eV/Å vs
xtb \(-0.799\); ATV energy FD stays \(\sim+0.06\): that slope is a
**CAMM-vs-\(R\)** hole (xtb `mmompop` two-centre quadrupole on H; ATV skips
s-only \(\Theta\)), not a missing Fock term. Naive two-centre \(\Theta\) on
H is not xtb `mmompop` (eq-H₂O residual 23 meV). Putting AES in the Fock
does not tilt ATV's energy surface onto xtb's angular slope. Dual-gated
anyway (assign wave 11).

## Perf

xtb wall **0.018 s** (1 thread, archived transcript). ATV wall is measured
in the regression; fail if \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} > 10\).

## Gaps

- Angular \(F_{H1,y}\) vs xtb (CAMM-vs-\(R\) / H quadrupole from two-centre
  overlap; AES-in-Fock does not close it — ATV energy FD stays \(\sim+0.06\)
  eV/Å)
- Frozen-CAMM \(G\) vs energy FD \(\sim 0.19\) eV/Å on \(F_{O,x}\) (quadrupole
  \(\partial Q/\partial R\))
- Dipole–quadrupole / quadrupole–quadrupole pair AES ranks
- Elements beyond H/C/N/O
- ALPB, spin, PBC, IPEA, sTDA
