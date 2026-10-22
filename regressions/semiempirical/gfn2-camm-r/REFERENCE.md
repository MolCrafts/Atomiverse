# Reference — off-eq water GFN2-xTB two-centre CAMM quadrupole vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

## Method

GFN2-xTB nuclear gradient of the Bannwarth 2019 SCC energy with anisotropic
electrostatics on the cumulative atomic multipoles (charge, dipole,
two-centre quadrupole of every atom) in the Fock, plus frozen-\(P\)
\(\partial\mu/\partial R\), \(\partial\Theta/\partial R\) in the analytic \(G\):

\[
E = \operatorname{Tr}(P H^{0})
  + E_{\mathrm{IES}}
  + E^{(3)}
  + E_{\mathrm{rep}}
  + E_{\mathrm{D4}}
  + E_{\mathrm{AES}}
\]

\(E_{\mathrm{AES}}\) is charge–dipole, dipole–dipole, charge–quadrupole +
dipole/quadrupole AXC; \(E_{\mathrm{D4}}\) is self-consistent in the SCC
charges.

Reference: Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI 10.1021/acs.jctc.8b01176. Formula twin: xtb 6.7.1
`src/aespot.f90` `mmompop` + `aniso_grad` / `setdvsdq`. **Not** DFTB+;
**not** OpenMOPAC; **not** GFN0/1.

## Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host). Command:

```text
XTBPATH=.../share/xtb OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 2 --chrg 0 --uhf 0 --acc 1.0 --grad
```

Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/`
(`mol.xyz` + `xtb.out.txt` + `gradient.txt`). Re-run 2026-08-25 reproduced
the archived TOTAL ENERGY and \(\nabla E\) to all printed digits. Nothing
is executed at ctest time. xtb prints \(\nabla E\) [Eh/a0]; ATV stores
\(F=-\nabla E\) [eV/Å] with CODATA \(a_0\) / Hartree from `units.h`.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−5.058625058504** | Eh |
| \(\|\nabla E\|\) | 0.109121448851 | Eh/a0 |
| Wall (xtb, 1 thread) | 0.015 | s |

xtb internal wall on the re-run is 0.015 s (archived sibling transcript
0.018 s). Gate uses **0.015 s**.

### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| O | −7.5243103995119e−2 | −1.7310256877367e−2 | ~0 |
| H1 | +7.5511039446358e−2 | +1.5529339054184e−2 | ~0 |
| H2 | −2.6793545123835e−4 | +1.7809178231828e−3 | ~0 |

## Geometry

Gas-phase water, charge 0, singlet, **off equilibrium** (O–H stretched
0.96 → 1.10 Å) so \(|F|\neq 0\). Cubic box 20 Å (isolation only). Same
geometry as `reg_gfn2-aes-fock` / `reg_gfn2-xtb-forces`.

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
| \(F_{H1,y}\) | **−0.008321573206** (tblite 0.4.0, see below) | eV/Å | \(0.35\) (**gated**) |

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

### \(F_{H1,y}\) pin provenance (tblite 0.4.0)

xtb 6.7.1's printed \(\partial E/\partial y\) of O and H1 at this input is
not the derivative of xtb's own energy (evidence above and below), so the
\(F_{H1,y}\) pin is taken from tblite 0.4.0 — same GFN2-xTB Hamiltonian,
total energy within 2.7e−9 Eh of xtb 6.7.1 — at the **identical** input.
E and \(F_{O,x}\) stay the xtb 6.7.1 pins.

| Field | Value |
| ----- | ----- |
| Binary | `/opt/homebrew/Caskroom/miniconda/base/envs/tblite-oracle/bin/tblite` (conda-forge `tblite 0.4.0 hb3fdf34_2`; `tblite --version` → `tblite version 0.4.0`) |
| Input | `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/mol.xyz` (sha256 `ea3fd42d92d5ad482dbac0f46ad7e1894b3566ef692652ec2aff92e27f3f7eb9`) |
| Command | `OMP_NUM_THREADS=1 tblite run --method gfn2 --grad tblite.grad mol.xyz` |
| Printed total energy | `total energy             -5.0586250605191E+00 Eh` |
| Output line (tblite.grad, `gradient :real:2:3,3`, row 2 = H1) | `  7.5510829209522637E-02  1.6182883370639375E-04  1.6456648816321770E-17` |
| Pin | \(\partial E/\partial y_{H1} = +1.6182883370639375\times10^{-4}\) Eh/a0 → \(F_{H1,y} = -\partial E/\partial y \times\) `units::kHartreePerBohrInEvPerAngstrom` = −0.008321573206 eV/Å |

Evidence that the replaced xtb value (+1.5529339054184e−2 Eh/a0) is an
orientation artifact of xtb 6.7.1's gradient (all Eh/a0):

| Source | \(g_{O,y}\) | \(g_{H1,y}\) |
| ------ | ------------- | -------------- |
| xtb 6.7.1 `--grad`, this input (old pin) | −1.7310e−2 | +1.5529e−2 |
| central FD of xtb 6.7.1 energies, this input (h = 1e−3 Å, `--acc 0.0001`) | −1.94179e−3 | +1.6202e−4 |
| xtb 6.7.1 `--grad`, same molecule rotated 0.5 rad about z and translated, rotated back | −1.94296e−3 | +1.6203e−4 |
| tblite 0.4.0 `--grad`, this input | −1.94249e−3 | +1.61829e−4 |
| ATV GFN2 (analytic; analytic vs FD ≤ 3e−7 eV/Å) | −1.94283e−3 | +1.62025e−4 |

The x components and H2 agree across all rows (e.g. \(g_{O,x}\): xtb
−7.52431e−2, rotated xtb −7.52431e−2, tblite −7.52427e−2), so only the
O/H1 y pair of the symmetric-orientation xtb gradient is affected.

## Perf

xtb wall **0.015 s** (1 thread, this re-run). ATV wall is measured in the
regression; fail if \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} > 10\).

## Gaps

- Elements beyond H/C/N/O
- ALPB, spin, PBC, sTDA
