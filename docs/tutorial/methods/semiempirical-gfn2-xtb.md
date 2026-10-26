# GFN2-xTB

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

GFN2-xTB (Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019)) is the third-generation extended tight-binding Hamiltonian. Beyond
the isotropic shell-resolved SCC of GFN1 it carries anisotropic electrostatics
(AES): cumulative atomic dipoles **and** traceless quadrupoles interact
through a damped multipole kernel, with on-site anisotropic XC (AXC) kernels
\(d_{\mathrm{kernel}}\) and \(q_{\mathrm{kernel}}\), plus a self-consistent D4
dispersion. AES and D4 enter the SCC Fock as \(\partial E/\partial P\), so the
converged energy is stationary in \(P\) and the closed-form gradient is its
exact derivative. GFN-xTB is **not** DFTB3++ — different Hamiltonian, tables,
and oracle (xtb only).

The concrete method is `cpu::semiempirical::GFN2{charge}` (charge defaults to
0) with members `scc_tol` (1e-8), `max_scc` (50) and `with_forces`; there is
no public `*Config` and no `make_*` / `build_*`, and AES-in-Fock is not a
switch — it is the production SCC. `GFN2` is mounted on `cpu::Evaluate` at
`evaluate::Phase::evaluate` (energy on `kEnergy` [eV], forces on `kAtom_f*`
[eV/Å]) and, with `with_forces = true`, on `cpu::vib::Hessian` at
`vib::Phase::force` for the FD Hessian. Parameters are static in-tree tables —
no `libxtb` wrap and no runtime parameter file. The sections below cover CPU
energy with quadrupole AES, analytic forces, the AES/D4 Fock potential, and the
GFN1/GFN2 FD Hessian on water. Two further sections of the case have no
section here and are documented only in the case's `REFERENCE.md`:
`gfn2-xtb-water` (near-eq H₂O energy, tolerance \(2\times10^{-3}\) eV) and
`gfn2-camm-r` (two-centre CAMM quadrupole on the stretched H₂O). See also
[Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md) (SE product surface)
and [Ch. 14 Vibrations](../ch14-vibrations.md).

Case: `ctest -R '^semiempirical/gfn2-xtb$'` — sources and oracle record in [`regressions/semiempirical/gfn2-xtb/`](../../../regressions/semiempirical/gfn2-xtb/). One section of the case per `##` heading below: `atv_regressions --case semiempirical/gfn2-xtb --section <name>`.

## Method

**GFN2-xTB energy** (Bannwarth et al., *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI 10.1021/acs.jctc.8b01176) — closed-shell molecular SCC:

\[
E = \operatorname{Tr}(P H^{0})
  + E_{\mathrm{IES}}
  + E^{(3)}
  + E_{\mathrm{rep}}
  + E_{\mathrm{D4}}
  + E_{\mathrm{AES}}.
\]

| Term | Role |
| ---- | ---- |
| \(\operatorname{Tr}(P H^{0})\) | Occupied density on the extended-Hückel \(H^0\) (CN'-shifted self-energies, distance polynomial \(\Pi(R)\)); \(S\) from STO-\(n\)G |
| \(E_{\mathrm{IES}}\) | Shell-resolved (DFTB2-style) isotropic electrostatics, \(\gamma_{AB}\) with \(g_{\exp}=2\) |
| \(E^{(3)}\) | On-site cubic \(\frac13\sum_l \Gamma_l q_l^3\) in the shell charge \(q_l\) |
| \(E_{\mathrm{rep}}\) | Screened \(Z_{\mathrm{eff}}\) pair repulsion |
| \(E_{\mathrm{D4}}\) | Self-consistent Grimme D4 (\(C_6\) scaled by the SCC Mulliken charges) — **Fock potential** \(\partial E/\partial q\); GFN2 damping \(s_6=1\), \(s_8=2.7\), \(a_1=0.52\), \(a_2=5\), \(s_9=5\) |
| \(E_{\mathrm{AES}}\) | Anisotropic electrostatics (charge–dipole \(f_3\), charge–quadrupole and dipole–dipole \(f_5\)) + dipole/quadrupole AXC on the cumulative multipoles of every atom (H included) — **Fock potential** |

Parameter path: static tables → double-exponential CN' (\(k_a=10\),
\(k_b=20\), 2 Bohr shift, \(4/3\)-scaled Pyykkö radii) → \(H^0\) with
\(\Pi(R)=\prod\bigl(1+k\,(R/(R_A+R_B))^{1/2}\bigr)\) (\(\Pi\) radius
O = 0.64 Å) and \(S\) → generalized eigen → Mulliken shell \(q\) and
cumulative multipoles \(\boldsymbol{\mu}, \Theta\) → SCC mix of the IES,
\(E^{(3)}\), AES and D4 potentials → band + IES + \(E^{(3)}\) + repulsion +
D4 + AES. In ATV this is
`Gfn2ElementRow → CN' → H/S → eigen → P → SCC(γ, Γ, AES, D4) → E/F`
with parameters as members of `GFN2` and tables in
`basis_data/gfn2_xtb_tables.h`. Unsupported \(Z\) throws.

\(H^{0}\) is the CN-shifted STO-\(n\)G valence Hamiltonian (tblite
`get_hamiltonian` scales \(k_{s}=1.85\), \(k_{p}=2.23\),
\(w_{\mathrm{exp}}=0.5\)). Occupation is restricted integer filling (8 valence
electrons on H₂O).

### Anisotropic electrostatics {#anisotropic-electrostatics}

Moments are cumulative atomic multipoles of \(P\) on **every** atom (H
included), with the operator centred on the bra atom \(A\) of \(\kappa\):

\[
q_A = -\sum_{l\in A}\Delta n_l,
\qquad
\mu_{A,\alpha}
  = -\sum_{\kappa\in A}\sum_{\nu} P_{\kappa\nu}
    \langle\kappa|r_\alpha-R_{A,\alpha}|\nu\rangle,
\]

\[
\Theta_{A,\alpha\beta}
  = -\sum_{\kappa\in A}\sum_{\nu} P_{\kappa\nu}
    \bigl\langle\kappa\bigr|
      \tfrac32(r-R_A)_\alpha(r-R_A)_\beta
      -\tfrac12\delta_{\alpha\beta}\lvert r-R_A\rvert^2
    \bigl|\nu\bigr\rangle,
\]

with \(\Delta n_l=\mathrm{pop}_l-n^0_l\) the Mulliken shell population minus
its reference occupation. The \(R_A\)-centred integrals come from origin-0
\(S\), \(D^\alpha\), \(Q_{\alpha\beta}\): \(D^\alpha-R_{A,\alpha}S\) and the
Cartesian origin shift
\(q^{\mathrm{shift}}_{\alpha\beta}
 = Q_{\alpha\beta}-R_{\alpha}D_{\beta}-R_{\beta}D_{\alpha}+R_{\alpha}R_{\beta}S\),
mapped to the **traceless** \(3/2\) form
\(\tfrac32 q^{\mathrm{shift}}_{\alpha\beta}
 -\tfrac12\delta_{\alpha\beta}\operatorname{Tr}q^{\mathrm{shift}}\)
(tblite packed \((xx,xy,yy,xz,yz,zz)\)).

Pair damping uses a CN'-dependent AES radius (tblite `damped_multipole` 0d):

\[
R_{A}
  = R_{\mathrm{mp},A}
  + \frac{R_{\max}-R_{\mathrm{mp},A}}
        {1+\exp\bigl(-k_{\mathrm{aes}}(\mathrm{CN}_{A}
          -\mathrm{CN}_{\mathrm{mp},A}-\mathrm{shift})\bigr)},
\]

with \(k_{\mathrm{aes}}=4\), \(\mathrm{shift}=1.2\), \(R_{\max}=5\) Bohr, and
the same double-exponential CN' as \(H^0\). For pair \(AB\) with
\(r=\lvert\mathbf{r}\rvert\), \(\mathbf{r}=\mathbf{R}_{A}-\mathbf{R}_{B}\),
\(R_{\mathrm{avg}}=(R_{A}+R_{B})/2\),

\[
f_{n}(r)
  = \bigl(1+6(R_{\mathrm{avg}}/r)^{n}\bigr)^{-1},
\qquad n=3\text{ (odd), }n=5\text{ (even)},
\qquad g_n=f_n/r^n.
\]

The energy assembled in `solve_gfn2` (tblite `get_energy` / `amat_sq` /
`amat_sd`) is, for every pair \(A\neq B\),

\[
\begin{aligned}
E_{q\mu}
  &= g_{3}\,(q_{A}\boldsymbol{\mu}_{B}-q_{B}\boldsymbol{\mu}_{A})\!\cdot\mathbf{r},\\
E_{\mu\mu}
  &= g_{5}\bigl(
        (\boldsymbol{\mu}_{A}\!\cdot\boldsymbol{\mu}_{B})\,r^{2}
      - 3(\boldsymbol{\mu}_{A}\!\cdot\mathbf{r})(\boldsymbol{\mu}_{B}\!\cdot\mathbf{r})
      \bigr),\\
E_{q\Theta}
  &= g_{5}\bigl(
        q_{A}\,\mathbf{r}\!\cdot\Theta_{B}\!\cdot\mathbf{r}
      + q_{B}\,\mathbf{r}\!\cdot\Theta_{A}\!\cdot\mathbf{r}
      \bigr).
\end{aligned}
\]

\(\Theta\) is already the traceless \(3/2\) form, so no extra factor enters;
charge–quadrupole is even in \(\mathbf{r}\). On-site AXC (off-diagonal packed
quadrupole scaled by 2, tblite):

\[
E_{\mathrm{AXC}}
  = \sum_{A}
    \Bigl(
      d_{A}\lvert\boldsymbol{\mu}_{A}\rvert^{2}
    + q_{A}\bigl(
        \Theta_{xx}^{2}+2\Theta_{xy}^{2}+\Theta_{yy}^{2}
        +2\Theta_{xz}^{2}+2\Theta_{yz}^{2}+\Theta_{zz}^{2}
      \bigr)
    \Bigr).
\]

Dipole–quadrupole and quadrupole–quadrupole pair ranks are not GFN2 terms and
are not assembled.

### Tables

| Piece | Source in ATV | Oracle twin |
| ----- | ------------- | ----------- |
| H/C/N/O element rows (ζ, ε, Hubbard, \(d/q_{\mathrm{kernel}}\), AES radii) | `basis_data/gfn2_xtb_tables.h` | xtb `param_gfn2-xtb.txt` / tblite `gfn2.f90` |
| Global AES damp (\(d_{\mathrm{mp3}}=3\), \(d_{\mathrm{mp5}}=4\)) | same header | tblite 0d molecular kernel |
| D4 damping | self-consistent `scf::dftd4_fixed_charges` at the SCC charges + GFN2 \(s_{6,8,9},a_{1,2}\) | xtb GFN2 D4 |
| Other elements | **unsupported** (throw) | xtb is all-element |

O `qkernel` is tabulated and non-zero (\(-3.10828\times10^{-3}\) Eh); the
unit test `test_gfn2_params` asserts that.

| Path | Role |
| ---- | ---- |
| `gfn2.h` | Public `GFN2` (`scc_tol`, `charge`, `with_forces`) |
| `src/cpu/semiempirical/gfn2.cpp` | SCC with AES + self-consistent D4 in the Fock (`aes_eval`, `build_fock`) + analytic \(\nabla E\) (HF+Pulay, CN', \(\Pi\), IES, rep, D4, AES kernel + multipole-integral derivatives) |
| `src/cpu/scf/dftd4.cpp` | `dftd4_fixed_charges`: D4 at given charges, \(\partial E/\partial q\), fixed-\(q\) forces |
| `tests/cpu/semiempirical/test_gfn2_params.cpp` | Unit: tables, analytic-vs-FD (tol \(10^{-4}\) eV/Å) + net-force invariance |

## Common setup

Two water geometries (Å), both closed shell, charge 0, in a cubic 20 Å cell
(isolation only; min-image present, images not engaged). Masses 15.999 (O) /
1.008 (H) amu.

| Atom | \(Z\) | near-eq \((x,y,z)\) | stretched \((x,y,z)\) |
| ---- | ----: | ------------------- | --------------------- |
| O | 8 | \((0.00, 0.00, 0.00)\) | \((0.00, 0.00, 0.00)\) |
| H1 | 1 | \((0.96, 0.00, 0.00)\) | \((\mathbf{1.10}, 0.00, 0.00)\) |
| H2 | 1 | \((-0.24, 0.93, 0.00)\) | \((-0.24, 0.93, 0.00)\) |

The **near-eq 0.96 Å** geometry is used by `gfn2-xtb-water`,
`gfn2-quadrupole-aes` and `gfn-xtb-analytic-hessian`. The **stretched**
geometry (O–H 0.96 → **1.10** Å, so \(\|F\|\neq 0\)) is used by
`gfn2-xtb-forces`, `gfn2-aes-fock` and `gfn2-camm-r`.

Units: \(E\) [eV] on `kEnergy`; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å];
internals in Hartree / Bohr (tblite convention). Conversions:
\(a_0=1/1.8897261245650618\) Å (CODATA); Ha → eV public write
\(27.211386245988\) (CODATA-2018, `units::kHartreeInEv`); Ha → eV for table
ε \(27.21138505\) (xtb/tblite `kGfn2EvToEh`); forces
\(F=-\nabla E\times\) `units::kHartreePerBohrInEvPerAngstrom`.

```cpp
cpu::Frame frame(3);                 // one of the geometries above, kAtom_type = Z
// ... kAtom_x/y/z, kAtom_mass, kAtom_tag, zeroed kAtom_f*
frame.box = cpu::Box::cubic(20.0);
frame.meta.get(kEnergy) = 0;
```

Build with the SE-only module closure (`module load buildenv-gcc/2026.03-mpich`;
`-DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_EVALUATE=ON`, plus `-DATV_BUILD_VIB=ON`
for the Hessian section; MD/MINIMIZER/HF/PW/GPW/CUDA stay OFF).

## Oracle {#oracle}

**xtb 6.7.1** (`edcfbbe`) only, `OMP_NUM_THREADS=1`; the GFN family is locked
to xtb — never OpenMOPAC or DFTB+ as reference. Offline pins are hard-coded
in the section binaries; xtb is never run from ctest. Formula twins: tblite
`get_hamiltonian_gradient`, `get_multipole_gradient_0d`, `get_potential`,
`add_vmp_to_h1`.

```text
oracle: xtb
oracle_version: 6.7.1 (edcfbbe)
```

**Near-eq H₂O** (`xtb mol.xyz --gfn 2 --chrg 0 --uhf 0`): 6 AO, 4 shells, 8
electrons; HOMO–LUMO 14.519 eV; Mulliken \(q(\mathrm{O})=-0.563\),
\(q(\mathrm{H})=0.282/0.281\).

| Quantity | Value | Unit |
| -------- | ----: | ---- |
| Total energy | **−5.070383095219** | Eh |
| SCC | −5.104517627658 | Eh |
| isotropic ES | +0.031400392671 | Eh |
| anisotropic ES | +0.000610908598 | Eh |
| anisotropic XC | −0.000805309300 | Eh |
| dispersion | −0.000141270483 | Eh |
| repulsion | +0.034134532439 | Eh |

**Stretched H₂O** (`xtb mol.xyz --gfn 2 --chrg 0 --uhf 0 --acc 1.0 --grad`,
twin `mol.xyz`, `xtb.out.txt`, `gradient.txt`): 6 AO, 4 shells, 8 electrons;
HOMO–LUMO 10.606 eV; Mulliken \(q(\mathrm{O})=-0.549\),
\(q(\mathrm{H})=0.271/0.277\).

| Quantity | Value | Unit |
| -------- | ----: | ---- |
| Total energy | **−5.058625058504** | Eh |
| SCC | −5.080060971331 | Eh |
| isotropic ES | +0.031320691734 | Eh |
| anisotropic ES | +0.001306928885 | Eh |
| anisotropic XC | −0.000150463849 | Eh |
| dispersion | −0.000143706605 | Eh |
| repulsion | +0.021435912826 | Eh |
| HOMO–LUMO | 10.605534434298 | eV |
| \(\|\nabla E\|\) | 0.109121448851 | Eh/\(a_0\) |
| Wall (xtb, 1 thread) | 0.018 | s |

xtb \(\nabla E\) [Eh/\(a_0\)] (`gradient.txt`):

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| O | −7.5243103995119×10⁻² | −1.7310256877367×10⁻² | ~0 |
| H1 | +7.5511039446358×10⁻² | +1.5529339054184×10⁻² | ~0 |
| H2 | −2.6793545123835×10⁻⁴ | +1.7809178231828×10⁻³ | ~0 |

**Residual vs xtb (shared).** On both geometries the ATV energy differs from
xtb by **11 µeV**, and that remainder is the self-consistent D4 term (ATV
\(-1.40847\times10^{-4}\) vs xtb \(-1.41270\times10^{-4}\) Eh). AES + AXC on
the near-eq geometry is \(-1.94391\times10^{-4}\) Eh vs xtb anisotropic ES +
XC \(-1.94401\times10^{-4}\) Eh.

**xtb y-gradient orientation artifact.** At the stretched orientation (O at
the origin, H1 on the \(x\) axis) the O / H1 \(y\)-gradients xtb 6.7.1 prints
are not the derivative of its own energy. Central FD of xtb 6.7.1 energies
(\(h=10^{-3}\) Å) gives \(g_{O,y}=-1.942\times10^{-3}\),
\(g_{H1,y}=+1.620\times10^{-4}\) Eh/\(a_0\); xtb on the same molecule rotated
0.5 rad about \(z\) and translated prints \(-1.943\times10^{-3}\) /
\(+1.620\times10^{-4}\) (rotated back); tblite 0.4.0 prints
\(-1.9425\times10^{-3}\) / \(+1.6183\times10^{-4}\); ATV gives
\(-1.9428\times10^{-3}\) / \(+1.6203\times10^{-4}\). The tabled
\(-1.731\times10^{-2}\) / \(+1.553\times10^{-2}\) are not GFN2 gradients, so
\(F_{H1,y}\) is printed, not gated, against xtb; ATV matches FD of xtb’s own
energies, rotated-frame xtb and tblite 0.4.0 to \(\le 1\times10^{-6}\)
Eh/\(a_0\).

## Limitations

- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; not a
  silent skip.
- **Closed-shell even-electron only.** Odd / non-integer valence count throws.
- **No dipole–quadrupole / quadrupole–quadrupole pair ranks** (not GFN2
  terms).
- No ALPB/GBSA, CPCM-X, spin, PBC, sTDA. GFN-FF is a classical force field and
  outside this method.
- CHNO energy + AES-in-Fock + analytic \(G\) + FD Hessian do not make the
  whole GFN family closed; solvent, PBC and all-element tables remain open.

Sibling methods: GFN0 is [`gfn0-xtb-forces`](semiempirical-gfn0-xtb.md#gfn0-xtb-forces) /
[`gfn0-xtb-analytic-hessian`](semiempirical-gfn0-xtb.md#gfn0-xtb-analytic-hessian);
GFN1 [`gfn1-xtb-organic`](semiempirical-gfn1-xtb.md#gfn1-xtb-organic) /
[`gfn1-xtb-forces`](semiempirical-gfn1-xtb.md#gfn1-xtb-forces).
IPEA energy / \(G\) / FD Hessian / VIP/VEA are [`ipea-xtb-small`](semiempirical-ipea-xtb.md#ipea-xtb-small) /
[`ipea-xtb-forces`](semiempirical-ipea-xtb.md#ipea-xtb-forces) /
[`ipea-xtb-analytic-hessian`](semiempirical-ipea-xtb.md#ipea-xtb-analytic-hessian) /
[`ipea-xtb-vip`](semiempirical-ipea-xtb.md#ipea-xtb-vip).

## References

- C. Bannwarth, S. Ehlert, S. Grimme, *J. Chem. Theory Comput.* **15**, 1652
  (2019), DOI [10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
- tblite `coulomb/multipole.f90` `get_potential`; `scf/potential.f90`
  `add_vmp_to_h1`; `get_hamiltonian_gradient`; `get_multipole_gradient_0d`;
  `damped_multipole` molecular 0d
- Offline twins under `tools/tb_oracle/inputs/xtb/` (`h2o_gfn2/`,
  `h2o_gfn2_offeq_grad/`, `h2o_gfn2_hess/`); roadmap
  [`docs/roadmap/semiempirical.md`](../../roadmap/semiempirical.md) §3.

## gfn2-xtb-forces — analytic nuclear forces, stretched H₂O {#gfn2-xtb-forces}

### Summary

Closed-form nuclear gradient of GFN2-xTB on the stretched water. Because AES
and the self-consistent D4 are in the SCC Fock ([`gfn2-aes-fock`](#gfn2-aes-fock)),
\(\mathbf{F}=-\nabla E\) is the exact derivative of the converged energy —
Hellmann–Feynman + Pulay on \(\partial S\), CN' and \(\Pi(R)\) chains through
\(H^0\), IES \(\partial\gamma/\partial R\), pair repulsion, fixed-charge D4,
and the AES kernel plus frozen-\(P\) \(\partial\boldsymbol{\mu}/\partial R\),
\(\partial\Theta/\partial R\) — written to `frame.span(kAtom_f*)` in eV/Å.
This is the **xtb `--gfn 2 --grad` / tblite `get_hamiltonian_gradient` +
`get_multipole_gradient_0d` module boundary**, **not** a \(6N\) re-SCC
finite-difference. Fair wall asserted \(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\le 10\)
(xtb archived wall 0.018 s).

Analytic vs central FD of ATV’s own energy agrees to
\(\le 3\times10^{-7}\) eV/Å. Against the xtb 6.7.1 table the residuals are
11 µeV on \(E\) (the D4 term) and \(1.5\times10^{-5}\) eV/Å on \(F_{O,x}\).

### Ionic forces

`solve_gfn2(..., want_grad=true)` in `src/cpu/semiempirical/gfn2.cpp`. xtb
prints \(\nabla E\) [Eh/\(a_0\)]; ATV stores \(\mathbf{F}=-\nabla E\) [eV/Å]:

\[
\mathbf{F}_A = -\frac{\partial E}{\partial\mathbf{R}_A}
=
\mathbf{F}_A^{H^{0}}
+ \mathbf{F}_A^{\mathrm{IES}}
+ \mathbf{F}_A^{\mathrm{rep}}
+ \mathbf{F}_A^{\mathrm{D4}}
+ \mathbf{F}_A^{\mathrm{AES}},
\]

with pairwise accumulation (antisymmetrized bond contributions) plus CN
chains:

| Term | Contraction (schematic) |
| ---- | ----------------------- |
| Electronic \(H^0\) + Pulay | \(2\,\mathrm{Tr}\!\bigl((P\odot H^0-W)\,\partial S/\partial R\bigr)\) plus shell-polynomial \(\partial\Pi/\partial R\); \(W=\sum_i n_i\varepsilon_i\,c_ic_i^\top\) |
| Self-energy CN' chain | \(\sum_A (\partial E_{H^0}/\partial\varepsilon_A)\,(\partial\varepsilon_A/\partial\mathrm{CN}'_A)\,(\partial\mathrm{CN}'_A/\partial R)\) |
| IES \(\gamma\) (fixed \(q\)) | \(\sum_{ij} q_i q_j\,(\partial\gamma_{ij}/\partial R)\) |
| Repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) of the GFN2 \(Z_{\mathrm{eff}} e^{-\alpha r^{k_{\exp}}}/r\) pair (\(k_{\exp}=1.5\), light–light \(1.0\)) |
| D4 | Two-body + ATM D4 forces at the fixed SCC charges (`scf::dftd4_fixed_charges`); the charge response is carried by the Fock |
| AES kernel + multipoles | tblite `get_multipole_gradient_0d`: explicit charge–dipole, dipole–dipole, charge–quadrupole geometry + CN'\(\to R_{\mathrm{mp}}\) damping chain, plus frozen-\(P\) \(\partial\boldsymbol{\mu}/\partial R\), \(\partial\Theta/\partial R\) through the dipole / quadrupole integrals (incl. AXC) |

Because the SCC energy is stationary in \(P\), no response term appears:
\(\mathrm{d}E/\mathrm{d}R=\partial E/\partial R|_P-\operatorname{Tr}(W\,\partial S/\partial R)\).
The \(6N\) re-SCC path is **not** the production force; the section compares
two selected components against a **central energy-only** FD of ATV’s own
\(E\) (\(h=10^{-4}\) Å) as an identity gate.

### Implementation

```cpp
cpu::semiempirical::GFN2 gfn2;       // charge 0
gfn2.with_forces = true;
cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(gfn2);
eval.build();
eval.run(frame);                     // stretched geometry
// Energy: frame.meta.get(kEnergy) [eV]; forces: kAtom_fx / fy / fz [eV/Å]
```

### Validation (xtb)

xtb pins: the stretched-geometry energy and gradient tables under
[Oracle](#oracle).

| Quantity | Atomiverse | xtb 6.7.1 | Tol | Notes |
| -------- | ---------: | --------: | --- | ----- |
| \(E\) | `kEnergy` | **−5.058625058504 Eh** | \(1\times10^{-2}\) eV | measured residual 11 µeV (D4 term) |
| \(F_{O,x}\) | `kAtom_fx[0]` | **+3.869155970641 eV/Å** | \(0.35\) eV/Å | \(-\nabla E\) converted; stretch component; residual \(1.5\times10^{-5}\) eV/Å |
| \(\lvert\sum_i F_{i,d}\rvert\) | identity | 0 | \(1\times10^{-10}\) eV/Å | translation invariance, each Cartesian |
| \(F_{O,x}\), \(F_{H1,y}\) vs energy FD | identity | — | \(0.25\) eV/Å | \(h=10^{-4}\) Å; measured \(\le 3\times10^{-7}\) eV/Å |
| \(F_{H1,y}\) vs xtb | printed, **not gated** | **−0.7986 eV/Å** | — | xtb 6.7.1 orientation artifact; ATV \(-0.0083\) eV/Å |

**Tolerance rationale.** The energy floor and the \(F_{O,x}\) tol sit far
above the measured residuals (11 µeV; \(1.5\times10^{-5}\) eV/Å). The FD
identity is the gradient-consistency gate against ATV’s own \(E\), angular
component included; it holds to \(\le 3\times10^{-7}\) eV/Å because the SCC
energy is stationary in \(P\).

### Performance

Fair wall asserted in the section (\(r\le 10\); the section fails if the ratio
exceeds 10). The analytic path is one SCC + one gradient (replaces 18 energy
solves of a \(6N\) FD).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN2 `--gfn 2 --grad` | **0.018** (1 thread) | measured in the section | **asserted \(\le 10\)** | ✓ | xtb cold wall from archived `xtb.out.txt`; no xtb binary on the SE tutorial host for a fresh wall; ATV tables are static |

## gfn2-aes-fock — AES and D4 in the SCC Fock, stretched H₂O {#gfn2-aes-fock}

### Summary

AES — charge–dipole, dipole–dipole, charge–quadrupole, and dipole/quadrupole
AXC on the cumulative atomic multipoles of every atom (H included) — and the
self-consistent D4 enter the **SCC Fock** as \(\partial E/\partial P\). The SCC
energy is therefore stationary in \(P\), and the closed-form
\(\mathbf{F}=-\nabla E\) of [`gfn2-xtb-forces`](#gfn2-xtb-forces) is the exact
derivative of the energy. This is the tblite `get_potential` +
`add_pot_to_h1` / `add_vmp_to_h1` module boundary: multipoles \(\to\)
potential \(\{v, \mathbf{v}^\mu, \mathbf{v}^\Theta\}\) \(\to\) \(H\) every SCC
iteration. The section runs the same stretched-H₂O twin, API and gates as
`gfn2-xtb-forces`; analytic vs central FD of ATV’s own energy agrees to
\(\le 3\times10^{-7}\) eV/Å, and against xtb 6.7.1 the residuals are 11 µeV on
\(E\) and \(1.5\times10^{-5}\) eV/Å on \(F_{O,x}\).

### AES and D4 potential in the Fock

tblite `coulomb/multipole.f90` `get_potential` differentiates
\(E_{\mathrm{AES}}+E_{\mathrm{AXC}}\) w.r.t. \(\{q, \boldsymbol{\mu},
\Theta\}\) at frozen geometry / damping radii (pair kernel and AXC under
[Anisotropic electrostatics](#anisotropic-electrostatics)); the
self-consistent D4 adds \(\partial E_{\mathrm{D4}}/\partial q_A\) to the
charge channel. Per pair (and symmetrically for \(B\) with
\(\mathbf{R}\to-\mathbf{R}\), \(\mathbf{R}=\mathbf{R}_A-\mathbf{R}_B\)):

\[
\begin{aligned}
v_A
  &\mathrel{+}= g_3\,\boldsymbol{\mu}_B\!\cdot\mathbf{R}
              + g_5\,\mathbf{R}\!\cdot\Theta_B\!\cdot\mathbf{R},\\
\mathbf{v}^\mu_A
  &\mathrel{+}= -g_3\,q_B\,\mathbf{R}
              + g_5\bigl(\boldsymbol{\mu}_B\,r^2
                - 3\,\mathbf{R}\,(\boldsymbol{\mu}_B\!\cdot\mathbf{R})\bigr),\\
\mathbf{v}^\Theta_A
  &\mathrel{+}= g_5\,q_B\,\mathbf{R}\mathbf{R},
\end{aligned}
\]

and on site \(v_A \mathrel{+}= \partial E_{\mathrm{D4}}/\partial q_A\),
\(\mathbf{v}^\mu_A \mathrel{+}= 2d_A\boldsymbol{\mu}_A\),
\(\mathbf{v}^\Theta_A \mathrel{+}= \partial E_{\mathrm{AXC}}/\partial\Theta_A\).
AXC on packed \(\Theta\) uses the tblite off-diagonal scale 2 in the
energy, hence 4 in the potential (diag \(2 q^{\mathrm{kern}}\Theta_{ii}\)).

tblite `scf/potential.f90` `add_vmp_to_h1` then folds those channels into
the SCC Hamiltonian (ATV: `build_fock` in `solve_gfn2`), for
\(\kappa\in A\) in shell \(l\), \(\nu\in B\) in shell \(l'\):

\[
\begin{aligned}
H_{\kappa\nu}
  &= H^0_{\kappa\nu}
  + \tfrac12 S_{\kappa\nu}\,(v_l+v_{l'})\\
  &\quad - \tfrac12\sum_\alpha
      \bigl[
        v^\mu_{A,\alpha}\langle\kappa|r_\alpha-R_{A,\alpha}|\nu\rangle
      + v^\mu_{B,\alpha}\langle\nu|r_\alpha-R_{B,\alpha}|\kappa\rangle
      \bigr]\\
  &\quad - \tfrac12\sum_k
      \bigl[
        v^\Theta_{A,k}\,\theta^{A}_{k,\kappa\nu}
      + v^\Theta_{B,k}\,\theta^{B}_{k,\nu\kappa}
      \bigr],
\end{aligned}
\]

with \(\theta^A\) the traceless \(R_A\)-centred quadrupole integrals and
\(v_l=\partial E/\partial\Delta n_l
=\sum_{l''}\gamma_{ll''}\Delta n_{l''}-\Gamma_l\Delta n_l^2-v_A\) (IES,
\(E^{(3)}\) in the shell charge, AES + D4 charge channel). Every atom
carries all three channels. After the generalized eigen, \(\Delta n\),
\(\boldsymbol{\mu}\) and \(\Theta\) are rebuilt from the new \(P\), mixed
together, and the loop repeats until their largest change is below
`scc_tol`. \(H=\partial E/\partial P\), so the converged energy is
stationary in \(P\), and
\(\mathrm{d}E/\mathrm{d}R=\partial E/\partial R|_P-\operatorname{Tr}(W\,\partial S/\partial R)\).

### Implementation

Same as [`gfn2-xtb-forces`](#gfn2-xtb-forces): one `GFN2` with
`with_forces = true` is both the Evaluate energy source and the force source.

### Validation (xtb)

Gates (identical to `gfn2-xtb-forces`, same stretched twin):

| Quantity | Atomiverse | xtb 6.7.1 | Tol | Notes |
| -------- | ---------: | --------: | --- | ----- |
| \(E\) | `kEnergy` | **−5.058625058504 Eh** | \(1\times10^{-2}\) eV | measured residual **11 µeV** (D4 term) |
| \(F_{O,x}\) | `kAtom_fx[0]` | **+3.869155970641 eV/Å** | \(0.35\) eV/Å | \(-\nabla E\) converted; stretch; residual \(1.5\times10^{-5}\) eV/Å |
| \(\lvert\sum_i F_{i,d}\rvert\) | identity | 0 | \(1\times10^{-10}\) eV/Å | translation invariance, each Cartesian |
| \(F_{O,x}\), \(F_{H1,y}\) vs energy FD | identity | — | \(0.25\) eV/Å | \(h=10^{-4}\) Å; measured \(\le 3\times10^{-7}\) eV/Å |
| \(F_{H1,y}\) vs xtb | printed, **not gated** | **−0.7986 eV/Å** | — | xtb 6.7.1 orientation artifact; ATV \(-0.0083\) eV/Å |

The identity gate holds to \(\le 3\times10^{-7}\) eV/Å because AES and D4 are
in the Fock as \(\partial E/\partial P\).

### Performance

Fair wall asserted in the section (\(r\le 10\), xtb 0.018 s). AES / D4 in the
Fock is one potential fill per SCC iteration on the same closed-form gradient
(replaces 18 energy solves of a \(6N\) FD).

## gfn2-quadrupole-aes — quadrupole AES energy, near-eq H₂O {#gfn2-quadrupole-aes}

### Summary

Pins the GFN2 total energy of near-eq water on the public Evaluate surface
(eV on `frame.meta[kEnergy]`), isolating the charge–quadrupole AES
\(E_{q\Theta}\) + quadrupole-\(q_{\mathrm{kernel}}\) AXC terms; charge–dipole,
dipole–dipole and dipole AXC are the same kernel as `gfn2-xtb-water`. The
measured residual against archived xtb is **11 µeV**: AES + AXC is
\(-1.94391\times10^{-4}\) Eh vs xtb anisotropic ES + XC
\(-1.94401\times10^{-4}\) Eh, and the remainder is the self-consistent D4
term.

### Implementation

```cpp
cpu::Evaluate eval;
cpu::semiempirical::GFN2 gfn2;
gfn2.with_forces = false;
gfn2.scc_tol = 1e-8;
eval.on(evaluate::Phase::evaluate).add_ref(gfn2);
eval.build();
eval.run(frame);                     // near-eq geometry

const double E_eV = frame.meta.get(kEnergy);
const double pin_eV = -5.070383095219 * units::kHartreeInEv;
// gate: |E_eV − pin_eV| ≤ 5e−4 eV
```

### Validation (vs xtb)

xtb pin: the near-eq decomposition under [Oracle](#oracle)
(`xtb mol.xyz --gfn 2 --chrg 0 --uhf 0`; public pin
−5.070383095219 × `units::kHartreeInEv` eV).

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E_{\mathrm{tot}}\) | `kEnergy` | **−5.070383095219 Eh** | **5×10⁻⁴ eV** | hard-coded in the section |
| Host residual | \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert\) | — | — | **11 µeV** (D4 term) |
| SCC residual | `last_scc_residual()` | — | \(<\) `scc_tol` \(=10^{-8}\) | must converge, not exhaust `max_scc` |
| Energy finite / non-zero | `kEnergy` | — | \(\lvert E\rvert>1\) eV | sanity gate |

Tolerance is \(5\times10^{-4}\) eV (0.5 meV) — tighter than `gfn2-xtb-water`
(\(2\times10^{-3}\) eV) because this pin isolates the quadrupole AES.

### Performance

Informative only — **not** asserted (`perf_exempt: true`).

| Twin | \(T_{\mathrm{oracle}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) | ≤10× | Caveat |
| ---- | --------------------------: | ----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN2 Evaluate (forces off) | 0.155 (xtb 6.7.1, 1 thread; SCC 0.115 s) | — | **perf_exempt** | — | no xtb binary on the SE build host for a fair wall |

## gfn-xtb-analytic-hessian — GFN1/GFN2 FD Hessian, near-eq H₂O vs xtb `--hess` {#gfn-xtb-analytic-hessian}

### Summary

Nuclear Hessian of **GFN1-xTB** (primary pin) and **GFN2-xTB** on the near-eq
water, assembled as a **\(6N\) central finite difference of the analytic
gradient** (\(h=10^{-3}\) Å) through `cpu::vib::Hessian` with
`cpu::semiempirical::GFN1` / `GFN2` (`with_forces = true`) on
`vib::Phase::force` — the same shape as
[`gfn0-xtb-analytic-hessian`](semiempirical-gfn0-xtb.md#gfn0-xtb-analytic-hessian),
**not** a closed-form \(\partial^2 E/\partial R_i\partial R_j\) kernel and
**not** `libxtb`. xtb 6.7.1 `--gfn 1 --hess` / `--gfn 2 --hess` prints an
**analytical** Hessian (log: “analytical hessian”), a different algorithm
class; the gate is still \(\omega\) and one Cartesian \(H\) element vs the xtb
printout, with tols that absorb the path mismatch. Because the GFN2 gradient
is the exact derivative of its SCC energy, the GFN2 FD Hessian differs from
xtb only by FD step and masses, as for GFN1. After mass-weighting (IUPAC-2013
from \(Z\)) and TR projection the three vibrational wavenumbers are written
on `HessianResult::frequencies` [cm⁻¹]; the Cartesian matrix is
`HessianResult::hessian` [eV/Å²]. Fair wall
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.11\le 10\) (GFN1 primary).

### Method

The GFN1 energy (Grimme, Bannwarth, Shushkov, *J. Chem. Theory Comput.*
**13**, 1989 (2017), DOI
[10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118)) and its
analytic gradient are described on the [GFN1-xTB page](semiempirical-gfn1-xtb.md#gfn1-xtb-forces):

\[
E = \mathrm{Tr}(P H_0)
  + \tfrac12 q\,\gamma\,q
  + \tfrac13 \Gamma q^3
  + E_{\mathrm{rep}}
  + E_{\mathrm{D3}},
\]

Klopman–Ohno SCC (\(g_{\exp}=2\), harmonic-mean hardness), frozen-\(q\)
\(\partial E_\Gamma/\partial R=0\), pair
\(Z^{\mathrm{eff}}_A Z^{\mathrm{eff}}_B\,e^{-\sqrt{\alpha_A\alpha_B}\,r^{1.5}}/r\),
D3(BJ) GFN1 set \(s_6=1\), \(s_8=2.4\), \(a_1=0.63\), \(a_2=5.0\) (xtb
\(s_9=0\)); hydrogen double-ζ (1s + orthogonalized 2s polarization); tables in
`basis_data/gfn1_tables.h` (tblite `gfn1.f90` / xtb `param_gfn1-xtb.txt`).
The GFN2 energy and gradient are [above](#gfn2-xtb-forces) (analytic vs
central FD \(\le 3\times10^{-7}\) eV/Å).

Central FD of the analytic forces, then symmetrized. With \(g=-F\) and
\(h=10^{-3}\) Å (`cpu::vib::kHessianStepAngstrom`):

\[
H_{ij}
  \approx
  \frac{g_i(+h\,e_j)-g_i(-h\,e_j)}{2h},
  \qquad
  H\leftarrow\tfrac12(H+H^\top).
\]

For \(N=3\) this is a fixed **\(6N=18\)** force budget plus one finalize
pass (restore \(R_0\), mass-weight, TR-project, diagonalize). The driver
is `cpu::vib::Hessian` (`Driver<vib::Phase>`); stages
`setup → displace → force → accumulate → diagonalize → thermochem →
end_of_step`. Masses for \(\omega\) come from IUPAC-2013 `mass_amu(Z)`
(`kAtom_type`), **not** `kAtom_mass`. (The GFN0 twin is analytic \(G\),
numerical \(H\) at step \(0.005\,a_0\).)

**Not included.** Closed-form \(\partial^2 E/\partial R_i\partial R_j\)
(CPHF / GFN second-derivative kernel). IR intensities from an APT (the FD
driver does not solve CPHF).

### Settings

| Knob | Value |
| ---- | ----- |
| System | Near-eq H₂O (same xyz as `gfn1-xtb-forces` / `gfn0-xtb-analytic-hessian`; **not** the 1.10 Å stretch of `gfn2-xtb-forces`) |
| GFN1 | `--gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess` |
| GFN2 | `--gfn 2 --chrg 0 --uhf 0 --acc 0.01 --hess` (twin `h2o_gfn2_hess/mol.xyz`) |
| Hessian | `cpu::vib::Hessian`, \(h=10^{-3}\) Å central FD of closed-form \(G\) |
| Units | \(H\) [eV/Å²]; \(\omega\) [cm⁻¹] |
| Conversion | \(a_0=0.529177210903\) Å (CODATA `units::kBohrInAngstrom`); \(H[\mathrm{eV}/\mathrm{Å}^2]=H[\mathrm{Eh}/a_0^2]\times 27.211386245988/(0.529177210903)^2\) |
| Thermo ctor | \(T=298.15\) K, \(P=101325\) Pa, multiplicity 1, \(\sigma=1\); empty recorder paths |

xtb reports “incompletely optimized geometry” on both twins; all three
vibrational eigenvalues are real.

```cpp
cpu::semiempirical::GFN1 gfn1;
gfn1.with_forces = true;
// Empty recorder paths: do not write frequencies.tsv / mode_*.xyz.
cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
hess.on(vib::Phase::force).add(std::move(gfn1));
hess.build();
const cpu::vib::HessianResult res = hess.run(frame);
// res.hessian: [9,9] dense, eV/Å²; res.frequencies: 3 modes, cm⁻¹
// Frame geometry is restored to R0; leftover kEnergy is the last displacement.
```

GFN2 is the same shape with `cpu::semiempirical::GFN2` in place of `GFN1`.
The section re-evaluates the energy after restore with a fresh method on
`cpu::Evaluate` (leftover `kEnergy` from the last displacement is not the
pin).

| Path | Role |
| ---- | ---- |
| `hessian.h` | `cpu::vib::Hessian` FD preset |
| `hessian_result.h` | `HessianResult` (\(H\), \(\omega\), \(n_{\mathrm{imag}}\)) |
| `hessian_workspace.h` | \(h=10^{-3}\) Å (`kHessianStepAngstrom`) |
| `tests/cpu/semiempirical/test_xtb_gfn_hessian.cpp` | Unit: `Hessian::build` accepts GFN1 and GFN2 force kinds (no `run`) |

### Validation (xtb)

Command: `xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --hess`
(`OMP_NUM_THREADS=1`); GFN2 replaces `--gfn 1`. Twins carry `mol.xyz`,
`xtb.out.txt`, `hessian.txt`, `vibspectrum.txt`. Oracle literature: GFN1
(Grimme et al. 2017), GFN2 (Bannwarth et al. 2019), xtb (Bannwarth et al.
2021).

**xtb projected \(\omega\) [cm⁻¹]** (`vibspectrum.txt` modes 7–9; six TR
zeros projected out):

| Mode | Assignment | GFN1 xtb 6.7.1 | GFN2 xtb 6.7.1 |
| ---- | ---------- | -------------: | -------------: |
| \(\omega_1\) | bend | **1474.96** | **1565.10** |
| \(\omega_2\) | symmetric stretch | **3657.78** | **3634.95** |
| \(\omega_3\) | antisymmetric stretch | **3761.56** | **3643.62** |

xtb also printed TOTAL ENERGY **−5.768641651774 Eh** (GFN1) /
**−5.070383095219 Eh** (GFN2) and Cartesian \(H_{O_x,O_x}=0.5466593694\)
/ \(0.5210291876\) Eh/\(a_0^2\) (`hessian.txt`). IR intensities in the
twins (GFN1 233.62 / 45.82 / 45.96 km mol⁻¹; GFN2 131.85 / 12.89 / 7.38
km mol⁻¹) are **not** gated (the ATV FD Hessian has no APT).

**Hard-coded gates:**

| Quantity | Atomiverse | xtb 6.7.1 | Tol | Notes |
| -------- | ---------: | --------: | --- | ----- |
| GFN1 \(E\) | `kEnergy` (fresh Evaluate after restore) | **−5.768641651774 Eh** | \(5\times10^{-3}\) eV | family TB energy floor (`gfn1-xtb-forces`) |
| GFN1 \(\omega_1\) (bend) | `res.frequencies(0)` | **1474.96 cm⁻¹** | 10 cm⁻¹ | projected, ascending; **primary pin** |
| GFN1 \(\omega_2\) (sym stretch) | `res.frequencies(1)` | **3657.78 cm⁻¹** | 10 cm⁻¹ | |
| GFN1 \(\omega_3\) (asym stretch) | `res.frequencies(2)` | **3761.56 cm⁻¹** | 10 cm⁻¹ | |
| GFN1 \(H_{O_x,O_x}\) | `res.hessian(0,0)` | **53.12087 eV/Å²** | 0.2 eV/Å² | from \(0.5466593694\) Eh/\(a_0^2\) |
| GFN2 \(E\) | `kEnergy` (fresh Evaluate after restore) | **−5.070383095219 Eh** | \(1\times10^{-3}\) eV | family TB energy floor |
| GFN2 \(\omega_1\) (bend) | `res.frequencies(0)` | **1565.10 cm⁻¹** | 150 cm⁻¹ | projected, ascending |
| GFN2 \(\omega_2\) (sym stretch) | `res.frequencies(1)` | **3634.95 cm⁻¹** | 150 cm⁻¹ | |
| GFN2 \(\omega_3\) (asym stretch) | `res.frequencies(2)` | **3643.62 cm⁻¹** | 150 cm⁻¹ | |
| GFN2 \(H_{O_x,O_x}\) | `res.hessian(0,0)` | **50.63029 eV/Å²** | 1.0 eV/Å² | from \(0.5210291876\) Eh/\(a_0^2\) |
| Hessian shape | identity | \(9\times 9\) | hard | finite, \(\lvert H-H^\top\rvert\le 10^{-8}\) eV/Å² |
| Rotor | identity | nonlinear | hard | `res.is_linear == false`; 3 modes |
| \(n_{\mathrm{imag}}\) | identity | 0 | hard | all three \(\omega>100\) cm⁻¹ |
| Geometry restore | identity | \(R_0\) | \(10^{-12}\) Å | `Hessian::run` restores displacements |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**

| | GFN1 | GFN2 |
| - | ---- | ---- |
| \(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert\) | \(2.767\times10^{-3}\) eV | \(1.14\times10^{-5}\) eV |
| \(\Delta\omega\) | 0.38 / 0.40 / 5.48 cm⁻¹ | 0.05 / 5.4 / 0.8 cm⁻¹ |
| \(\lvert H_{O_x,O_x}-H_{\mathrm{pin}}\rvert\) | \(0.104\) eV/Å² | \(0.104\) eV/Å² |

**Tolerance rationale.** Energy floors match the family TB pins (GFN1 5 meV;
GFN2 1 meV). GFN1 frequency / Cartesian-\(H\) tols absorb FD-of-\(G\) vs xtb
**analytical** Hessian plus IUPAC-2013 vs xtb isotope masses (the GFN1
angular-\(F\) kernel hole does **not** move these \(\omega\); measured residual
sits well under 10 cm⁻¹ / 0.2 eV/Å²). Measured GFN2 residuals
(\(\Delta\omega\le 5.4\) cm⁻¹, \(0.104\) eV/Å²) sit far inside the GFN2
\(\omega\) / \(H\) tols. Structural gates: 9×9 finite symmetric Hessian,
nonlinear rotor, 3 real modes, \(n_{\mathrm{imag}}=0\).

### Performance

Fair wall asserted in the section (\(r\le 10\)). Same *product* (projected
\(\omega\) of H₂O); **not** the same Hessian algorithm class (ATV: analytic
\(G\), numerical \(H\), 18 force evaluations + finalize; xtb 6.7.1:
analytical Hessian). Primary ratio is GFN1.

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN1 `--gfn 1 --hess` | **0.03** (1 thread, `/usr/bin/time`) | **\(\sim 3.2\times10^{-3}\)** | **≈ 0.11** | ✓ | xtb printed wall 0.020 s; the section pins the process wall 0.03 s; ATV tables are static; xtb Hessian is analytical |
| H₂O GFN2 `--gfn 2 --hess` | **0.02** (1 thread, `/usr/bin/time`) | **\(\sim 1.4\times10^{-3}\)** | **≈ 0.07** | ✓ | xtb printed wall 0.012 s; xtb Hessian is analytical |

### Limitations

- **FD of analytic \(G\), not closed-form \(H\).** Cost is
  \(O(N_{\mathrm{atoms}})\) GFN1/GFN2 force evaluations; \(\omega\) tols
  absorb the path difference to xtb's analytical Hessian.
- **Off-eq geometry.** xtb warns “incompletely optimized”; frequencies are
  still all real. Do not cite these \(\omega\) as a GFN1/GFN2 minimum of
  water.
- **GFN2 tols are loose.** The 150 cm⁻¹ / 1.0 eV/Å² GFN2 tols sit far
  above the measured \(\Delta\omega\le 5.4\) cm⁻¹ / \(0.104\) eV/Å²; the gate
  does not resolve GFN2 \(\omega\) at the GFN1 level.
- **GFN1 energy kernel hole.** GFN1 \(\lvert\Delta E\rvert=2.767\) meV on
  this water (CH₄ energy residual 10.2 meV on
  [`gfn1-xtb-organic`](semiempirical-gfn1-xtb.md#gfn1-xtb-organic)) is
  kernel/D3/intcut, not a missing Hessian term. It does not move GFN1
  \(\omega\) past 10 cm⁻¹.
- No IR intensities.
