# semiempirical/gfn2-xtb

One case; every section below is run by `atv_regressions --case semiempirical/gfn2-xtb` and keeps its own oracle record.


## H2O GFN2-xTB energy

```text
oracle: xtb
oracle_version: 6.7.1 (edcfbbe)
```

### Method

GFN2-xTB (Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI [10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)).

Family-locked numerical oracle: **xtb** (never OpenMOPAC / DFTB+).

Native static in-tree tables. No `libxtb` wrap.

### Geometry

Water, slightly off-equilibrium (Å) — the same ATV H₂O used by the SE
pins. Archived xtb 6.7.1 transcript
`tools/tb_oracle/inputs/xtb/h2o_gfn2/xtb.out` (`mol.xyz --gfn 2 --chrg 0
--uhf 0`); input twin `mol.xyz`.

| Atom | x    | y    | z   |
|------|------|------|-----|
| O    | 0.00 | 0.00 | 0.0 |
| H    | 0.96 | 0.00 | 0.0 |
| H    |−0.24 | 0.93 | 0.0 |

Cubic box 20 Å (isolated). Transcript: 6 AO, 4 shells, 8 electrons,
HOMO–LUMO 14.519 eV, Mulliken q(O)=−0.563, q(H)=0.282 / 0.281.

### Pin

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E_{\mathrm{tot}}\) | **−5.070383095219** | Eh | — |
| \(E_{\mathrm{tot}}\) | −5.070383095219 × `units::kHartreeInEv` | eV | \(2\times10^{-3}\) eV |

xtb summary (same transcript):

| Piece | Eh |
| ----- | -- |
| SCC | −5.104517627658 |
| isotropic ES | 0.031400392671 |
| anisotropic ES | 0.000610908598 |
| anisotropic XC | −0.000805309300 |
| dispersion | −0.000141270483 |
| repulsion | 0.034134532439 |
| **total** | **−5.070383095219** |

Wall (xtb, 1 thread): 0.155 s (SCC 0.115 s).

### Gates

- Finite, non-zero total energy [eV] on `frame.meta[kEnergy]`
- SCC residual < `scc_tol`
- Hard-coded energy pin vs table (tol \(2\times10^{-3}\) eV ≈ 0.07 mEh)

Measured ATV residual: **11 µeV** (4.2e−7 Eh). H⁰, isotropic ES, third
order, repulsion and AES/AXC reproduce xtb 6.7.1 / tblite 0.4.0 to
≤ 1e−8 Eh at the same density; the remainder is the self-consistent D4
term (ATV −1.40847e−4 vs xtb −1.41270e−4 Eh).

### Perf

`perf_exempt: true` — no xtb binary on the SE build host for a fair wall
ratio. Archived xtb wall is 0.155 s. Revisit when xtb is on PATH.

### Gaps

- Analytic forces: `gfn2-xtb-forces`
- Elements beyond H/C/N/O
- ALPB, spin, PBC, IPEA, sTDA


## Reference — off-eq water GFN2-xTB analytic forces vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

### Method

GFN2-xTB nuclear gradient of the Bannwarth 2019 SCC energy

\[
E = \operatorname{Tr}(P H^{0})
  + E_{\mathrm{IES}}
  + E^{(3)}
  + E_{\mathrm{rep}}
  + E_{\mathrm{D4}}
  + E_{\mathrm{AES}}
\]

already shipped by `reg_gfn2-xtb-water` / `reg_gfn2-quadrupole-aes`. Forces
are the **closed-form** \(\mathbf{F}=-\nabla E\) (Hellmann–Feynman + Pulay +
CN chain + IES \(\partial\gamma/\partial R\) + pair repulsion + D4 + AES
kernel at frozen CAMM). There is no 6N re-SCC product path.

Reference: Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI 10.1021/acs.jctc.8b01176. Gradient module boundary: tblite
`get_hamiltonian_gradient`, `get_multipole_gradient_0d`.
**Not** DFTB+; **not** OpenMOPAC; **not** GFN0/1.

### Offline oracle run

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

#### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| O | −7.5243103995119e−2 | −1.7310256877367e−2 | ~0 |
| H1 | +7.5511039446358e−2 | +1.5529339054184e−2 | ~0 |
| H2 | −2.6793545123835e−4 | +1.7809178231828e−3 | ~0 |

### Geometry

Gas-phase water, charge 0, singlet, **off equilibrium** (O–H stretched
0.96 → 1.10 Å) so \(|F|\neq 0\). Cubic box 20 Å (isolation only).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 1.10  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

### Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E\) | **−5.058625058504** | Eh | \(1\times10^{-2}\) eV |
| \(F_{O,x}\) | **+3.869155970641** | eV/Å | \(0.35\) |

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
\(F_{H1,y}\) is printed, not gated.

### Perf

xtb wall **0.018 s** (1 thread, archived transcript). ATV wall is measured
in the regression; fail if \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} > 10\).

### Gaps

- Elements beyond H/C/N/O
- ALPB, spin, PBC, sTDA


## Reference — off-eq water GFN2-xTB AES-in-Fock forces vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

### Method

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

### Offline oracle run

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

#### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| O | −7.5243103995119e−2 | −1.7310256877367e−2 | ~0 |
| H1 | +7.5511039446358e−2 | +1.5529339054184e−2 | ~0 |
| H2 | −2.6793545123835e−4 | +1.7809178231828e−3 | ~0 |

### Geometry

Gas-phase water, charge 0, singlet, **off equilibrium** (O–H stretched
0.96 → 1.10 Å) so \(|F|\neq 0\). Cubic box 20 Å (isolation only). Same
geometry as `reg_gfn2-xtb-forces`.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 1.10  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

### Pins

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

### Perf

xtb wall **0.018 s** (1 thread, archived transcript). ATV wall is measured
in the regression; fail if \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} > 10\).

### Gaps

- Elements beyond H/C/N/O
- ALPB, spin, PBC, IPEA, sTDA


## Reference — off-eq water GFN2-xTB two-centre CAMM quadrupole vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

### Method

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

### Offline oracle run

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

#### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| O | −7.5243103995119e−2 | −1.7310256877367e−2 | ~0 |
| H1 | +7.5511039446358e−2 | +1.5529339054184e−2 | ~0 |
| H2 | −2.6793545123835e−4 | +1.7809178231828e−3 | ~0 |

### Geometry

Gas-phase water, charge 0, singlet, **off equilibrium** (O–H stretched
0.96 → 1.10 Å) so \(|F|\neq 0\). Cubic box 20 Å (isolation only). Same
geometry as `reg_gfn2-aes-fock` / `reg_gfn2-xtb-forces`.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 1.10  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

### Pins

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

#### \(F_{H1,y}\) pin provenance (tblite 0.4.0)

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

### Perf

xtb wall **0.015 s** (1 thread, this re-run). ATV wall is measured in the
regression; fail if \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} > 10\).

### Gaps

- Elements beyond H/C/N/O
- ALPB, spin, PBC, sTDA


## H2O GFN2-xTB energy with quadrupole AES

```text
oracle: xtb
oracle_version: 6.7.1 (edcfbbe)
```

### Method

GFN2-xTB (Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI [10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)).

Family-locked numerical oracle: **xtb** (never OpenMOPAC / DFTB+).

Native static in-tree tables. No `libxtb` wrap.

This slug adds **charge–quadrupole AES** and the on-site **qkernel AXC**
on top of the `gfn2-xtb-water` foothold (charge–dipole + dipole–dipole +
dipole AXC). Kernel transcribed from tblite `damped_multipole` molecular
0d (`amat_sq`, packed `(xx,xy,yy,xz,yz,zz)`, traceless Buckingham map).

### Geometry

Water, slightly off-equilibrium (Å) — same ATV H₂O as `gfn2-xtb-water`.
Archived xtb 6.7.1 transcript
`tools/tb_oracle/inputs/xtb/h2o_gfn2/xtb.out` (`mol.xyz --gfn 2 --chrg 0
--uhf 0`); input twin `mol.xyz`.

| Atom | x    | y    | z   |
|------|------|------|-----|
| O    | 0.00 | 0.00 | 0.0 |
| H    | 0.96 | 0.00 | 0.0 |
| H    |−0.24 | 0.93 | 0.0 |

Cubic box 20 Å (isolated). Transcript: 6 AO, 4 shells, 8 electrons,
HOMO–LUMO 14.519 eV, Mulliken q(O)=−0.563, q(H)=0.282 / 0.281.

### Pin

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E_{\mathrm{tot}}\) | **−5.070383095219** | Eh | — |
| \(E_{\mathrm{tot}}\) | −5.070383095219 × `units::kHartreeInEv` | eV | \(5\times10^{-4}\) eV |

xtb summary (same transcript):

| Piece | Eh |
| ----- | -- |
| SCC | −5.104517627658 |
| isotropic ES | 0.031400392671 |
| anisotropic ES | 0.000610908598 |
| anisotropic XC | −0.000805309300 |
| dispersion | −0.000141270483 |
| repulsion | 0.034134532439 |
| **total** | **−5.070383095219** |

Wall (xtb, 1 thread): 0.155 s (SCC 0.115 s).

### Gates

- Finite, non-zero total energy [eV] on `frame.meta[kEnergy]`
- SCC residual < `scc_tol`
- Hard-coded energy pin vs table (tol \(5\times10^{-4}\) eV)

Measured residual **11 µeV**: AES/AXC on the cumulative multipoles of every
atom (H included) is −1.94391e−4 Eh vs xtb anisotropic ES + XC
−1.94401e−4 Eh; the remainder is the D4 term (see `reg_gfn2-xtb-water`).

### Perf

`perf_exempt: true` — no xtb binary on the SE build host for a fair wall
ratio. Archived xtb wall is 0.155 s. Revisit when xtb is on PATH.

### Gaps

- Elements beyond H/C/N/O
- ALPB, spin, PBC, sTDA


## Reference — water GFN1/GFN2-xTB FD Hessian vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

### Method

GFN1-xTB and GFN2-xTB nuclear Hessian of the Grimme 2017 / Bannwarth 2019
SCC energies, assembled as a **6N central finite difference of the
already-shipped analytic gradient** (\(h = 10^{-3}\) Å,
`cpu::vib::Hessian` + `cpu::semiempirical::GFN1` / `GFN2` with
`with_forces = true`). Not a closed-form \(\partial^2 E/\partial R_i\partial R_j\)
kernel; not GFN0 (already `reg_gfn0-xtb-analytic-hessian`); not `libxtb`.

References: Grimme, Bannwarth, Shushkov, *J. Chem. Theory Comput.* **13**,
1989 (2017), DOI 10.1021/acs.jctc.7b00118; Bannwarth, Ehlert, Grimme,
*J. Chem. Theory Comput.* **15**, 1652 (2019), DOI 10.1021/acs.jctc.8b01176.
xtb 6.7.1 `--gfn 1 --hess` / `--gfn 2 --hess` prints an **analytical**
Hessian (xtb log: “analytical hessian”); ATV stays FD of analytic \(G\)
(same shape as GFN0). **Not** DFTB+; **Not** OpenMOPAC.

Primary pin is **GFN1**; GFN2 is gated on the same shape.

### Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host). Command:

```text
XTBPATH=.../share/xtb OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess
## and --gfn 2 on the same xyz
```

Twins: `tools/tb_oracle/inputs/xtb/h2o_gfn1_hess/` and
`tools/tb_oracle/inputs/xtb/h2o_gfn2_hess/`
(`mol.xyz` + `xtb.out.txt` + `hessian.txt` + `vibspectrum.txt`). Nothing
is executed at ctest time. xtb prints \(H\) [Eh/a0²] and projected ω
[cm⁻¹]; ATV stores \(H\) [eV/Å²] with CODATA \(a_0\) /
\(1\,\mathrm{Ha}=27.211386245988\) eV from `units.h`.

#### GFN1 (`--gfn 1 --hess`)

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−5.768641651774** | Eh |
| \(\omega\) (projected) | **1474.96, 3657.78, 3761.56** | cm⁻¹ |
| \(H_{O_x,O_x}\) | **0.5466593694** | Eh/a0² |
| xtb Hessian path | analytical | — |
| Wall (xtb printed, 1 thread) | 0.020 | s |
| Wall (`/usr/bin/time`) | 0.03 | s |

#### GFN2 (`--gfn 2 --hess`)

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−5.070383095219** | Eh |
| \(\omega\) (projected) | **1565.10, 3634.95, 3643.62** | cm⁻¹ |
| \(H_{O_x,O_x}\) | **0.5210291876** | Eh/a0² |
| xtb Hessian path | analytical | — |
| Wall (xtb printed, 1 thread) | 0.012 | s |
| Wall (`/usr/bin/time`) | 0.02 | s |

xtb reports “incompletely optimized geometry” (same 0.96 Å water as
`reg_gfn0-xtb-analytic-hessian` / `reg_gfn1-xtb-forces`); all three
vibrational eigenvalues are real. GFN2 energy matches
`reg_gfn2-xtb-water` (this geometry is **not** the 1.10 Å stretch of
`reg_gfn2-xtb-forces`).

### Geometry

Gas-phase water, charge 0, singlet. Cubic box 20 Å (isolation only).
Same Cartesian as `reg_gfn1-xtb-forces` / `reg_gfn0-xtb-analytic-hessian`.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 0.96  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

### Pins

\(H[\mathrm{eV}/\mathrm{Å}^2] = H[\mathrm{Eh}/a_0^2]\times
27.211386245988 / (0.529177210903)^2\).

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| GFN1 \(E\) | **−5.768641651774** | Eh | \(5\times10^{-3}\) eV |
| GFN1 \(\omega_1\) (bend) | **1474.96** | cm⁻¹ | 10 |
| GFN1 \(\omega_2\) (sym stretch) | **3657.78** | cm⁻¹ | 10 |
| GFN1 \(\omega_3\) (asym stretch) | **3761.56** | cm⁻¹ | 10 |
| GFN1 \(H_{O_x,O_x}\) | **53.12087** | eV/Å² | 0.2 |
| GFN2 \(E\) | **−5.070383095219** | Eh | \(1\times10^{-3}\) eV |
| GFN2 \(\omega\) | **1565.10, 3634.95, 3643.62** | cm⁻¹ | 150 |
| GFN2 \(H_{O_x,O_x}\) | **50.63029** | eV/Å² | 1.0 |

**Tolerance rationale.** Energy floors match the family TB pins
(`reg_gfn1-xtb-forces` 5 meV; `reg_gfn2-xtb-water` 1 meV). GFN1
frequency / Cartesian-\(H\) tols absorb FD-step vs xtb **analytical**
Hessian plus IUPAC-2013 vs xtb isotope masses (the GFN1 angular-\(F\)
kernel hole does **not** move these ω). GFN2 ω / \(H\) tols are unchanged from
the first landing. Structural gates: 9×9 finite symmetric Hessian,
nonlinear rotor, 3 real modes, \(n_{\mathrm{imag}}=0\).

Measured on the pin host (Release, `OMP_NUM_THREADS=1`):

| | GFN1 | GFN2 |
| - | ---- | ---- |
| \(\|E_{\mathrm{ATV}}-E_{\mathrm{pin}}\|\) | \(2.767\times10^{-3}\) eV | \(1.14\times10^{-5}\) eV |
| \(\Delta\omega\) | 0.38 / 0.40 / 5.48 cm⁻¹ | 0.05 / 5.4 / 0.8 cm⁻¹ |
| \(\|H_{O_x,O_x}-H_{\mathrm{pin}}\|\) | \(0.104\) eV/Å² | \(0.104\) eV/Å² |

The GFN2 analytic gradient is the exact derivative of its SCC energy
(Fock = \(\partial E/\partial P\) with AES and self-consistent D4), so the FD
Hessian here differs from xtb only by FD step and masses, as for GFN1.

### Perf

xtb GFN1 process wall **0.03 s** (1 thread; `/usr/bin/time` of `--hess`).
ATV `Hessian::run` wall **~3.2 ms** →
\(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.11\le 10\).
xtb GFN2 process wall **0.02 s**; ATV **~1.4 ms** → ratio **≈ 0.07**.
