# GFN0-xTB

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

GFN0-xTB (Pracht, Caldeweyher, Ehlert, Grimme, ChemRxiv 2019) is the
non-self-consistent member of the GFN-xTB family: one electronegativity
equilibration (EEQ) solve and one extended-Hückel diagonalization of a CN- and
charge-shifted Hamiltonian, plus classical repulsion, two-body D4(BJ) and a
polar-bond short-range correction. It is **not** SCC-DFTB and not a DFTB3
successor.

The concrete method is `cpu::semiempirical::GFN0{charge}` (charge defaults to
0) with the member `with_forces`; there is no public `*Config` and no
`make_*` / `build_*`. It is mounted on `cpu::Evaluate` at
`evaluate::Phase::evaluate` for energy and closed-form analytic forces, and on
`cpu::vib::Hessian` at `vib::Phase::force` for the finite-difference Hessian.
Energies land on `kEnergy` [eV], forces on `kAtom_f*` [eV/Å]. The sections
below cover CPU analytic forces and the FD Hessian on off-equilibrium water; a
third section of the case, `gfn0-beyond-toy`, pins the methanol total energy
(−7.597234498685 Eh, tolerance \(10^{-3}\) eV) and is documented only in the
case's `REFERENCE.md`. See also [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md)
(SE product surface; GFN-xTB is a sibling family) and
[Ch. 14 Vibrations](../ch14-vibrations.md) (`cpu::vib::Hessian` FD driver).

Case: `ctest -R '^semiempirical/gfn0-xtb$'` — sources and oracle record in [`regressions/semiempirical/gfn0-xtb/`](../../../regressions/semiempirical/gfn0-xtb/). One section of the case per `##` heading below: `atv_regressions --case semiempirical/gfn0-xtb --section <name>`.

## Method

**GFN0-xTB energy** (Pracht et al., ChemRxiv 2019,
DOI 10.26434/chemrxiv.8326202.v1) — non-SCC, one EHT diagonalization:

\[
E = E_{\mathrm{H0}} + E_{\mathrm{rep}} + E_{\mathrm{IES}} + E_{\mathrm{D4}} + E_{\mathrm{SRB}}.
\]

| Term | Role |
| ---- | ---- |
| \(E_{\mathrm{H0}}\) | Occupied eigenvalues of a CN- and \(q\)-shifted extended-Hückel \(H\), \(S\) from STO-\(n\)G |
| \(E_{\mathrm{rep}}\) | Classical pair repulsion \(Z^{\mathrm{eff}}_A Z^{\mathrm{eff}}_B\,e^{-\alpha r^{3/2}}/r\) |
| \(E_{\mathrm{IES}}\) | Electronegativity-equilibration (EEQ) electrostatics with Gaussian-smeared Coulomb |
| \(E_{\mathrm{D4}}\) | Two-body DFT-D4(BJ); \(s_6=1\), \(s_8=2.85\), \(a_1=0.80\), \(a_2=4.60\), \(s_9=0\) |
| \(E_{\mathrm{SRB}}\) | Polar heteronuclear short-range bond correction; **both** atoms must have \(Z\in[5,9]\) |

Parameter path: static tables → erf coordination numbers → EEQ charges →
shifted atomic levels \(\varepsilon_{A\ell}\) → \(H/S\) → generalized eigen →
band energy + classical terms. In ATV this is
`Gfn0Element → CN → EEQ → self-energy → EHT → repulsion/D4/SRB → E/F`
with parameters as members of `GFN0` and static H/C/N/O tables (transcribed
from pprcht/gfn0 `gfn0_param.f90`) in `basis_data/gfn0_tables.h`.
Unsupported \(Z\) throws.

**Coordination number** (Pyykkö radii, \(k_{\mathrm{CN}}=7.5\), cut at
\(\mathrm{CN}_{\max}=8\)):

\[
\mathrm{CN}_A^{\mathrm{raw}}
  = \sum_{B}\tfrac12\bigl(1+\mathrm{erf}[-k_{\mathrm{CN}}(r_{AB}-r_0)/r_0]\bigr),\quad
r_0=\tfrac43(R_A^{\mathrm{cov}}+R_B^{\mathrm{cov}}),
\]

then \(\mathrm{CN}_A=\ln(1+e^{\mathrm{CN}_{\max}})-\ln(1+e^{\mathrm{CN}_{\max}-\mathrm{CN}^{\mathrm{raw}}_A})\).

**Shell self-energy** (feeds the EHT diagonal):

\[
\varepsilon_{A\ell}
  = \varepsilon^0_{A\ell}
  - k_{\mathrm{CN},\ell}\,\mathrm{CN}_A
  - k_{q,\ell}\,q_A
  - k_{q,\mathrm{atom}}\,q_A^2.
\]

## Common setup

Both sections use the same isolated, **off-equilibrium** water (so
\(|F|\neq 0\)), closed shell, `charge = 0` (even valence-electron count), in a
cubic 20 Å cell (isolation only; min-image present, images not engaged).
xtb accuracy `--acc 1.0` (default GFN0). Water has **no SRB** term (H is
outside \(Z\in[5,9]\)); methanol SRB + C–O is covered by `gfn0-beyond-toy`.

| Atom | \(Z\) | \(x\) [Å] | \(y\) [Å] | \(z\) [Å] |
| ---- | ----: | ----: | ----: | ----: |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H1 | 1 | 0.96 | 0.00 | 0.00 |
| H2 | 1 | −0.24 | 0.93 | 0.00 |

Masses `kAtom_mass` 15.999 (O) / 1.008 (H) amu. Units: \(E\) [eV] on
`kEnergy`; \(\mathbf{R}\) [Å]; \(\mathbf{F}\) [eV/Å].
Conversion: \(a_0=0.52917726\) Å (GFN0 table, `kGfn0AutoAA`);
\(1\,\mathrm{Ha}=27.211386245988\) eV.

```cpp
cpu::Frame frame(3);                 // off-eq water above, kAtom_type = Z
// ... kAtom_x/y/z, kAtom_mass, kAtom_tag, zeroed kAtom_f*
frame.box = cpu::Box::cubic(20.0);
frame.meta.get(kEnergy) = 0;

cpu::semiempirical::GFN0 gfn0{/*charge=*/0};
gfn0.with_forces = true;
```

Build with the SE-only module closure (`module load buildenv-gcc/2026.03-mpich`;
`-DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_EVALUATE=ON`, plus `-DATV_BUILD_VIB=ON`
for the Hessian section; MD/MINIMIZER/HF/PW/GPW/CUDA stay OFF).

## Oracle

**xtb 6.7.1** only (`xtb/6.7.1-gcc-2025b-eb`, `OMP_NUM_THREADS=1`); the GFN
family is locked to xtb — never DFTB+ or OpenMOPAC as a GFN0 reference.
Offline pins are hard-coded in the section binaries; xtb is never run from
ctest. xtb printed TOTAL ENERGY **−4.366816412151 Eh** for the water geometry
in both twins; the energy gate is the family \(10^{-3}\) eV TB floor.

```text
oracle: xtb
oracle_version: 6.7.1
```

## Limitations

- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; not a
  silent skip.
- **Closed-shell even-electron only.** Odd / non-integer valence count throws.
- **Water has no SRB.** Polar-bond SRB is gated on methanol energy
  (`gfn0-beyond-toy`) plus the unit FD identity; no absolute xtb SRB force
  pin, and the Hessian does not exercise \(\partial^2 E_{\mathrm{SRB}}\).
- **No PBC / spin / solvent (ALPB/GBSA, CPCM-X) / sTDA.** GFN-FF is a
  classical force field and outside this method.
- Beyond-CHNO elements are not tabulated.
- GFN0 energy + \(G\) + FD \(H\) do not make the whole GFN family closed.

Sibling methods: GFN1 forces are [`gfn1-xtb-forces`](semiempirical-gfn1-xtb.md#gfn1-xtb-forces),
GFN2 forces [`gfn2-xtb-forces`](semiempirical-gfn2-xtb.md#gfn2-xtb-forces), AES-in-Fock
[`gfn2-aes-fock`](semiempirical-gfn2-xtb.md#gfn2-aes-fock), GFN1/2 Hessian
[`gfn-xtb-analytic-hessian`](semiempirical-gfn2-xtb.md#gfn-xtb-analytic-hessian).
IPEA energy / \(G\) / FD Hessian / VIP/VEA are [`ipea-xtb-small`](semiempirical-ipea-xtb.md#ipea-xtb-small) /
[`ipea-xtb-forces`](semiempirical-ipea-xtb.md#ipea-xtb-forces) /
[`ipea-xtb-analytic-hessian`](semiempirical-ipea-xtb.md#ipea-xtb-analytic-hessian) /
[`ipea-xtb-vip`](semiempirical-ipea-xtb.md#ipea-xtb-vip).

## References

- P. Pracht, E. Caldeweyher, S. Ehlert, S. Grimme, ChemRxiv (2019),
  DOI 10.26434/chemrxiv.8326202.v1
- C. Bannwarth et al., *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
- Offline twins under `tools/tb_oracle/inputs/xtb/`; roadmap
  `docs/roadmap/semiempirical.md`.

## gfn0-xtb-forces — analytic nuclear forces, off-eq H₂O {#gfn0-xtb-forces}

### Summary

Closed-form nuclear gradient of GFN0-xTB on off-equilibrium water. After one
EEQ solve and one extended-Hückel diagonalization, forces are
\(\mathbf{F}=-\nabla E\) at that density — Hellmann–Feynman + Pulay on
\(\partial S\), EEQ/CN chains through the self-energy, pair-repulsion,
two-body D4(BJ), and the polar-bond short-range correction (SRB is identically
zero on H₂O) — written to `frame.span(kAtom_f*)` in eV/Å. This is the
**xtb `--gfn 0 --grad` module boundary** (pprcht `build_dSH0`,
`chargeEquilibration`, `gfn0_repulsion`, `gfn0_shortranged`), **not** a
\(6N\) energy finite-difference. Fair wall
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.092\le 10\).

### Ionic forces

`energy_gfn0(..., want_grad=true)` in `src/cpu/semiempirical/gfn0.cpp`. xtb
prints \(\nabla E\) [Eh/\(a_0\)]; ATV stores \(\mathbf{F}=-\nabla E\) [eV/Å]:

\[
\mathbf{F}_A = -\frac{\partial E}{\partial\mathbf{R}_A}
=
\mathbf{F}_A^{\mathrm{H0}}
+ \mathbf{F}_A^{\mathrm{IES}}
+ \mathbf{F}_A^{\mathrm{rep}}
+ \mathbf{F}_A^{\mathrm{D4}}
+ \mathbf{F}_A^{\mathrm{SRB}},
\]

with pairwise accumulation (antisymmetrized bond contributions) plus CN/\(q\)
chains:

| Term | Contraction (schematic) |
| ---- | ----------------------- |
| Electronic \(H_0\) + Pulay | \(2\,\mathrm{Tr}\!\bigl((P\odot H_0-W)\,\partial S/\partial R\bigr)\) plus shell-polynomial \(\partial\Pi/\partial R\); \(W=\sum_i n_i\varepsilon_i\,c_ic_i^\top\) |
| Self-energy chain | \(\sum_A (\partial E_{\mathrm{H0}}/\partial\varepsilon_A)\,(\partial\varepsilon_A/\partial\mathrm{CN}_A\,\partial\mathrm{CN}_A/\partial R + \partial\varepsilon_A/\partial q_A\,\partial q_A/\partial R)\) |
| EEQ / IES | \(\partial(q^\top J q/2 - q^\top x)/\partial R\) from \(\partial J/\partial R\) and \(\partial\chi/\partial\mathrm{CN}\) |
| Repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) of the \(e^{-\alpha r^{3/2}}/r\) pair |
| D4 | Analytic two-body D4(BJ) forces (`cpu::scf::dftd4`) |
| SRB | Pair \(e^{-\mathrm{pre}\,(r-r_0(\mathrm{CN}))^2}\) plus \(\partial r_0/\partial\mathrm{CN}\) |

There is no \(6N\) central-FD product path. The section compares two selected
components against a **central energy-only** FD of ATV’s own \(E\)
(\(h=10^{-4}\) Å) as an identity gate, not as the production force.

### Implementation

```cpp
cpu::semiempirical::GFN0 gfn0;       // charge 0
gfn0.with_forces = true;             // closed-form path
cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(gfn0);
eval.build();
eval.run(frame);
// Energy: frame.meta.get(kEnergy) [eV]; forces: kAtom_fx / fy / fz [eV/Å]
```

| Path | Role |
| ---- | ---- |
| `gfn0.h` | Public `GFN0` (`charge`, `with_forces`) |
| `gfn0_tables.h` | Static H/C/N/O GFN0 rows + global scales |
| `src/cpu/semiempirical/gfn0.cpp` | EEQ + EHT + analytic \(\nabla E\) (HF+Pulay, CN/\(q\), rep, D4, SRB) |
| `tests/cpu/semiempirical/test_xtb_gfn0.cpp` | Unit analytic-vs-FD + net-force invariance |

### Validation (xtb)

Command: `xtb mol.xyz --gfn 0 --chrg 0 --uhf 0 --acc 1.0 --grad`
(`OMP_NUM_THREADS=1`); twin `h2o_gfn0_offeq_grad/` (`mol.xyz`, `xtb.out.txt`,
`gradient.txt`). Conversion
\(F=-\nabla E\times(27.211386245988\,\mathrm{eV/Eh})/0.52917726\,\mathrm{Å}\).

**xtb \(\nabla E\) [Eh/\(a_0\)]** (`gradient.txt`):

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| O | −3.4755123224283×10⁻⁴ | −9.4449534348479×10⁻⁴ | ~0 |
| H1 | −3.8500960780473×10⁻³ | +3.4420757341719×10⁻³ | ~0 |
| H2 | +4.1976473102901×10⁻³ | −2.4975803906871×10⁻³ | ~0 |

xtb also printed \(\|\nabla E\|=0.007179285426\) Eh/\(a_0\).

**Hard-coded gates:**

| Quantity | Atomiverse | xtb 6.7.1 | Tol | Notes |
| -------- | ---------: | --------: | --- | ----- |
| \(E\) | `kEnergy` | **−4.366816412151 Eh** | \(1\times10^{-3}\) eV | family TB energy floor |
| \(F_{O,x}\) | `kAtom_fx[0]` | **+0.017871801258 eV/Å** | \(1\times10^{-4}\) eV/Å | \(-\nabla E\) converted |
| \(F_{H1,y}\) | `kAtom_fy[1]` | **−0.176998634239 eV/Å** | \(1\times10^{-4}\) eV/Å | dual-component gate |
| \(\lvert\sum_i F_{i,d}\rvert\) | identity | 0 | \(1\times10^{-10}\) eV/Å | translation invariance, each Cartesian |
| \(F_{O,x}\), \(F_{H1,y}\) vs energy FD | identity | — | \(3\times10^{-5}\) eV/Å | \(h=10^{-4}\) Å; unit test same |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=3.4\times10^{-8}\) eV,
\(\lvert F_{O,x}-F_{\mathrm{pin}}\rvert=2.6\times10^{-6}\) eV/Å,
analytic vs energy FD \(6.5\times10^{-7}\) eV/Å.

**Tolerance rationale.** Force tol \(10^{-4}\) eV/Å sits well above xtb’s
printed \(10^{-16}\) Eh/\(a_0\) noise on \(F_z\). The unit FD identity on
this geometry covers EHT + EEQ + repulsion + D4.

### Performance

Fair wall asserted in the section (\(r\le 10\)). The analytic path is one
EEQ + one diagonalization (replaces 18 energy solves of a \(6N\) FD).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN0 `--gfn 0 --grad` | **0.007** (1 thread) | **6.45×10⁻⁴** | **≈ 0.092** | ✓ | xtb cold wall from archived `xtb.out.txt`; ATV tables are static |

## gfn0-xtb-analytic-hessian — FD Hessian, off-eq H₂O vs xtb `--hess` {#gfn0-xtb-analytic-hessian}

### Summary

Nuclear Hessian of GFN0-xTB on the same off-equilibrium water, assembled as a
**\(6N\) central finite difference of the analytic gradient**
(\(h=10^{-3}\) Å) through `cpu::vib::Hessian` with
`cpu::semiempirical::GFN0` (`with_forces = true`) on `vib::Phase::force`.
This is the **xtb `--gfn 0 --hess` module boundary** (analytic \(G\),
numerical \(H\); xtb step \(0.005\,a_0\)), **not** a closed-form
\(\partial^2 E/\partial R_i\partial R_j\) kernel. After mass-weighting
(IUPAC-2013 from \(Z\)) and TR projection the three vibrational wavenumbers
are written on `HessianResult::frequencies` [cm⁻¹]; the Cartesian matrix is
`HessianResult::hessian` [eV/Å²]. Fair wall
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.06\le 10\).

### Method

Central FD of the analytic forces of [`gfn0-xtb-forces`](#gfn0-xtb-forces),
then symmetrized. With \(g=-F\) and \(h=10^{-3}\) Å
(`cpu::vib::kHessianStepAngstrom`):

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
(`kAtom_type`), **not** `kAtom_mass`. xtb `--hess` is the same algorithm
class (analytic \(G\), numerical \(H\)) at step \(0.005\,a_0\approx 0.00265\) Å.

**Not included.** Closed-form \(\partial^2 E/\partial R_i\partial R_j\)
(CPHF / GFN0 second-derivative kernel). IR intensities from an APT (the FD
driver does not solve CPHF).

Additional settings: \(H\) [eV/Å²]; \(\omega\) [cm⁻¹]; thermo ctor
\(T=298.15\) K, \(P=101325\) Pa, multiplicity 1, \(\sigma=1\), empty
recorder paths. xtb reports “incompletely optimized geometry” (RMS
\(\|\nabla E\|=0.00718\) Eh/\(a_0\)); all three vibrational eigenvalues are
real.

### Implementation

```cpp
cpu::semiempirical::GFN0 gfn0;
gfn0.with_forces = true;
// Empty recorder paths: do not write frequencies.tsv / mode_*.xyz.
cpu::vib::Hessian hess(298.15, 101325.0, 1, 1, "", "");
hess.on(vib::Phase::force).add(std::move(gfn0));
hess.build();
const cpu::vib::HessianResult res = hess.run(frame);
// res.hessian: [9,9] dense, eV/Å²; res.frequencies: 3 modes, cm⁻¹
// Frame geometry is restored to R0; leftover kEnergy is the last displacement.
```

The section needs `VIB` in addition to `SEMIEMPIRICAL`.

| Path | Role |
| ---- | ---- |
| `hessian.h` | `cpu::vib::Hessian` FD preset |
| `hessian_result.h` | `HessianResult` (\(H\), \(\omega\), \(n_{\mathrm{imag}}\)) |
| `hessian_workspace.h` | \(h=10^{-3}\) Å (`kHessianStepAngstrom`) |
| `tests/cpu/semiempirical/test_xtb_gfn0_hessian.cpp` | Unit: `Hessian::build` accepts GFN0 force kind (no `run`) |

### Validation (xtb)

Command: `xtb mol.xyz --gfn 0 --chrg 0 --uhf 0 --acc 1.0 --hess`
(`OMP_NUM_THREADS=1`); twin files `mol.xyz`, `xtb.out.txt`, `hessian.txt`,
`vibspectrum.txt`. Conversion
\(H[\mathrm{eV}/\mathrm{Å}^2]=H[\mathrm{Eh}/a_0^2]\times 27.211386245988/(0.52917726)^2\).

**xtb projected \(\omega\) [cm⁻¹]** (`vibspectrum.txt` modes 7–9; six TR
zeros projected out):

| Mode | Assignment | xtb 6.7.1 |
| ---- | ---------- | --------: |
| \(\omega_1\) | bend | **1563.14** |
| \(\omega_2\) | symmetric stretch | **3497.87** |
| \(\omega_3\) | antisymmetric stretch | **3505.03** |

xtb also printed Cartesian \(H_{O_x,O_x}=0.4793058695\) Eh/\(a_0^2\)
(`hessian.txt`). IR intensities in the twin (17.01 / 18.60 / 16.02 km mol⁻¹)
are **not** gated (the ATV FD Hessian has no APT).

**Hard-coded gates:**

| Quantity | Atomiverse | xtb 6.7.1 | Tol | Notes |
| -------- | ---------: | --------: | --- | ----- |
| \(E\) | `kEnergy` (fresh Evaluate after restore) | **−4.366816412151 Eh** | \(1\times10^{-3}\) eV | family TB energy floor |
| \(\omega_1\) (bend) | `res.frequencies(0)` | **1563.14 cm⁻¹** | 10 cm⁻¹ | projected, ascending |
| \(\omega_2\) (sym stretch) | `res.frequencies(1)` | **3497.87 cm⁻¹** | 10 cm⁻¹ | |
| \(\omega_3\) (asym stretch) | `res.frequencies(2)` | **3505.03 cm⁻¹** | 10 cm⁻¹ | |
| \(H_{O_x,O_x}\) | `res.hessian(0,0)` | **46.57588 eV/Å²** | 0.2 eV/Å² | from \(0.4793058695\) Eh/\(a_0^2\) |
| Hessian shape | identity | \(9\times 9\) | hard | finite, \(\lvert H-H^\top\rvert\le 10^{-8}\) eV/Å² |
| Rotor | identity | nonlinear | hard | `res.is_linear == false`; 3 modes |
| \(n_{\mathrm{imag}}\) | identity | 0 | hard | all three \(\omega>100\) cm⁻¹ |
| Geometry restore | identity | \(R_0\) | \(10^{-12}\) Å | `Hessian::run` restores displacements |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=3.4\times10^{-8}\) eV,
\(\Delta\omega = 0.05/2.73/1.87\) cm⁻¹,
\(\lvert H_{O_x,O_x}-H_{\mathrm{pin}}\rvert=3.8\times10^{-2}\) eV/Å².

**Tolerance rationale.** Frequency / Cartesian-\(H\) tols absorb the FD-step
mismatch (xtb \(0.005\,a_0 \approx 0.00265\) Å vs ATV \(10^{-3}\) Å) plus
IUPAC-2013 vs xtb isotope masses (\(\sim 0.3\,\mathrm{cm}^{-1}\) on H
stretches). The 10 cm⁻¹ / 0.2 eV/Å² gates sit well above the measured
residual and still catch a sign, unit, or mass-weighting error on the stiff
stretches (\(\sim 3500\) cm⁻¹).

### Performance

Fair wall asserted in the section (\(r\le 10\)). Same algorithm class:
analytic \(G\), numerical \(H\) (18 force evaluations + finalize vs xtb
`--hess`).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN0 `--gfn 0 --hess` | **0.02** (1 thread, `/usr/bin/time`) | **\(\sim 1.2\times10^{-3}\)** | **≈ 0.06** | ✓ | xtb printed wall 0.007 s; the section pins the process wall 0.02 s; ATV tables are static |

### Limitations

- **FD of analytic \(G\), not closed-form \(H\).** Cost is \(O(N_{\mathrm{atoms}})\)
  GFN0 force evaluations.
- **Off-eq geometry.** xtb warns “incompletely optimized”; frequencies are
  still all real. Do not cite these \(\omega\) as a GFN0 minimum of water.
- No IR intensities.
