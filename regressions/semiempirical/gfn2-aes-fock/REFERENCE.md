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
is stationary with respect to AES.

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

**Current model state.** ATV GFN2 reproduces xtb 6.7.1 / tblite 0.4.0
term by term: H⁰ (double-exponential CN', Π radius O = 0.64 Å), isotropic
ES, third order (in the shell *charge*), repulsion, and AES/AXC on the
cumulative multipoles of every atom (≤ 1e−8 Eh at the same density). AES
and the self-consistent D4 are in the Fock as \(\partial E/\partial P\), so
the analytic gradient is the derivative of the energy: analytic vs central
FD ≤ 3e−7 eV/Å here. Residuals vs the table: \(E\) 11 µeV (D4 term
−1.40847e−4 vs xtb −1.41270e−4 Eh), \(F_{O,x}\) 1.5e−5 eV/Å.

**xtb 6.7.1 \(\partial E/\partial y\) artifact.** At this orientation (O at
the origin, H1 on the x axis) the y-gradients xtb prints for O and H1 are
not the derivative of its own energy. Central FD of xtb 6.7.1 energies
(h = 1e−3 Å) gives \(g_{O,y}=-1.942\times10^{-3}\),
\(g_{H1,y}=+1.620\times10^{-4}\) Eh/a0; the same molecule rotated 0.5 rad
about z and translated makes xtb print \(-1.943\times10^{-3}\) /
\(+1.620\times10^{-4}\) (rotated back); tblite 0.4.0 prints
\(-1.9425\times10^{-3}\) / \(+1.6183\times10^{-4}\). ATV:
\(-1.9428\times10^{-3}\) / \(+1.6203\times10^{-4}\). The tabled
\(-1.731\times10^{-2}\) / \(+1.553\times10^{-2}\) are not GFN2 gradients.

## Perf

xtb wall **0.018 s** (1 thread, archived transcript). ATV wall is measured
in the regression; fail if \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} > 10\).

## Gaps

- Elements beyond H/C/N/O
- ALPB, spin, PBC, IPEA, sTDA
