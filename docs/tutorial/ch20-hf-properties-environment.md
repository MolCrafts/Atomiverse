# Chapter 20. Properties and Environment

Extract observables from a mean-field wave function, and place the molecule in
an external field, continuum solvent, or scalar-relativistic one-electron
Hamiltonian. Each topic uses **Theory → Practice → Results**.

See also [Ch. 2](ch02-rhf.md), [Ch. 14](ch14-vibrations.md), and the
[Appendix](appendix-pyscf-oracle.md).

> **Build flag.** Everything in this chapter lives behind `ATV_BUILD_HF`, which
> is **OFF by default** — with it off the `cpu::scf::…` headers, sources, tests
> and regressions are excluded from the build entirely.
>
> ```bash
> cmake -B build -DCMAKE_BUILD_TYPE=Release \
>       -DATV_BUILD_HF=ON -DBLA_VENDOR=OpenBLAS
> cmake --build build -j$(nproc)
> ```

---

## 20.1 Multipole moments

### Theory

For a single determinant, \(\langle\hat O\rangle=\operatorname{Tr}(P O)\). Dipole
and traceless quadrupole (Buckingham):

\[
\boldsymbol\mu = \sum_A Z_A \mathbf R_A - \operatorname{Tr}(P\,\mathbf r),\qquad
\Theta_{ab}=\tfrac12\sum q\,(3r_ar_b-r^2\delta_{ab}).
\]

Origin shift \(\mathbf d\): \(\boldsymbol\mu'=\boldsymbol\mu-q_{\mathrm{tot}}\mathbf d\).
Neutral systems are origin-independent; charged systems must report the origin.
This engine uses the **centre of nuclear charge** (`kMultipoleOrigin`).

### Practice

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Pople631Gs;
rhf.run(frame);

cpu::MultipoleObserver obs{rhf.result_handle()};
// meta: kDipole, kQuadrupole, kMultipoleOrigin
```

### Results

| System | Quantity | Engine | PySCF 2.14.0 | Tol. |
|---|---|---:|---:|---|
| H₂O RHF/6-31G* | \(\mu_z\) / a.u. | −0.87578587 | −0.87578587 | \(10^{-6}\) |

HF overestimates the water dipole vs experiment (~2.23 D vs ~1.85 D)—method
error, not an implementation bug. Regression: `h2o_rhf_dipole`.

---

## 20.2 Atomic charges

### Theory

Atomic charges are conventions, not observables.

| Method | Idea |
|---|---|
| Mulliken | Split overlap populations |
| Löwdin | Symmetrically orthogonalise, then partition |
| MK / CHELPG | Fit exterior electrostatic potential (ESP) |

Löwdin uses unit-normalised AOs here so Cartesian d/f norms cancel. ESP
fitting is ill-conditioned for buried atoms; the engine falls back to an SVD
minimum-norm solution when needed.

### Practice

```cpp
auto q = cpu::lowdin_charge(rhf.result_handle(), frame);

cpu::MKCharges mk;
auto q_mk = mk.fit(rhf.result_handle(), frame);

cpu::ChelpgCharges chelpg;
auto q_chelpg = chelpg.fit(rhf.result_handle(), frame);
```

### Results

H₂O RHF/6-31G* (oxygen charge / \(e\)):

| Method | Engine | PySCF | Tol. |
|---|---:|---:|---|
| Löwdin | −0.7284 | −0.7284 | \(10^{-6}\) |
| MK | −0.8071 | −0.8071 | \(10^{-3}\) |
| CHELPG | −0.8050 | −0.8050 | \(10^{-3}\) |

Regression: `h2o_charges_pyscf`.

---

## 20.3 Analytic Hessian and vibrations

### Theory

Second energy derivatives need orbital response \(U^x\) (CPHF):

\[
{}^1(A+B)\,U^x = -B^x.
\]

\({}^1(A+B)\) is the same orbital Hessian used in stability analysis and SOSCF.
Analytic Hessians are currently **RHF only**.

### Practice

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Pople631Gs;
rhf.tol = 1e-12;
cpu::vib::HessianResult H = rhf.hessian(frame); // eV/Å²
// mass-weight → project TR → diagonalise → frequencies [cm⁻¹]
```

### Results

H₂O RHF/6-31G* harmonic frequencies / cm⁻¹:

| Mode | Engine | PySCF | Tol. |
|---|---:|---:|---|
| Bend | 1843.71 | 1843.71 | 1 |
| Sym. stretch | 4036.17 | 4036.17 | 1 |
| Asym. stretch | 4146.38 | 4146.38 | 1 |

Regression: `h2o_rhf_freq`.

---

## 20.4 TDHF / TDA excitation energies

### Theory

Excitation energies are poles of the frequency-dependent polarisability.
TDHF/RPA:

\[
\begin{pmatrix}A&B\\-B&-A\end{pmatrix}
\begin{pmatrix}X\\Y\end{pmatrix}
=\omega\begin{pmatrix}X\\Y\end{pmatrix}.
\]

**TDA** drops \(B\) (CIS on HF). \(\omega_{\mathrm{RPA}}\le\omega_{\mathrm{TDA}}\)
for corresponding roots. Triplet-unstable references make RPA \(\omega^2\)
non-physical; `tdhf()` runs a stability check first. Oscillator strengths use
the length-gauge transition dipole.

### Practice

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::Pople631Gs;
rhf.tol = 1e-12;
rhf.run(frame); // formaldehyde geometry in regressions

auto tda  = rhf.tda(3);
auto tdhf = rhf.tdhf(3);
```

### Results

CH₂O RHF/6-31G* singlets (eV; \(f\) dimensionless):

| Root | Assignment | TDA \(E\) | TDHF \(E\) | TDHF \(f\) |
|---|---|---:|---:|---:|
| 1 | \(n\to\pi^*\) (dark) | 4.6690 | 4.4999 | 0 |
| 2 | \(\sigma\to\pi^*\) | 9.9505 | 9.7075 | ~0.001 |
| 3 | \(\pi\to\pi^*\) (bright) | 10.3002 | 9.7559 | 0.180 |

Matches PySCF TDA/TDHF to \(10^{-4}\) eV / \(10^{-3}\) in \(f\). Regression:
`ch2o_tdhf`. Single-excitation theories are semi-quantitative, not
spectroscopic standards.

---

## 20.5 External field and point-charge embedding

### Theory

Environments modify the **one-electron** Hamiltonian only:

- Point charges: \(h\leftarrow h-\sum_C q_C\langle 1/|\mathbf r-\mathbf R_C|\rangle\),
  plus classical nucleus–charge and charge–charge terms.
- Uniform field: \(h\leftarrow h+\mathbf F\cdot\langle\mathbf r-\mathbf R_0\rangle\),
  \(\mathbf R_0\) = nuclear-charge centre (same as multipoles).

Identity \(E(\mathbf F)=E(0)-\boldsymbol\mu\cdot\mathbf F\) checks the field
implementation. Strong fields / diffuse bases can “ionise” the SCF; keep
\(|\mathbf F|\sim 10^{-4}\) a.u. Analytic forces are available for fixed
point charges (`embed_h2o_grad`), uniform field (`field_h2o_grad`), and
C-PCM free energy (`pcm-grad-h2o`).

### Practice

```cpp
cpu::scf::RHF rhf;
rhf.basis = cpu::scf::BasisSet::STO3G;
rhf.point_charges = {{0.0, 0.0, 3.0, 0.5}, {0.0, 0.0, -3.0, -0.5}};
rhf.external_field = {0.0, 0.0, 1e-4};
rhf.run(frame);
```

### Results

| Check | Observation |
|---|---|
| H₂O STO-3G finite-field \(\mu_z\) vs analytic | agree to \(10^{-5}\) a.u. |
| H₂O STO-3G + ±0.5 \(e\) charges vs PySCF `qmmm.mm_charge` | \(10^{-8}\) Ha after MM–MM bookkeeping |

Regression: `h2o_embedding_field`. PySCF omits MM–MM self-energy by default;
this engine includes it in nuclear repulsion.

---

## 20.6 Continuum Solvation

### Theory

Dielectric continuum \(\varepsilon\); cavity of scaled Bondi spheres; apparent
surface charges from \(S\mathbf q=-f(\varepsilon)\mathbf V\) with
\(f=(\varepsilon-1)/(\varepsilon+x)\). \(x=0\): C-PCM; \(x=1/2\): COSMO.
IEF-PCM (`solvent_model = SolventModel::IEFPCM`) uses
\(f=(\varepsilon-1)/(\varepsilon+1)\) and the \(K q = R V\) system with the
double-layer operator \(D\).
Only electrostatic \(\Delta G_{\mathrm{el}}=\frac12\mathbf q\cdot\mathbf V\).
No cavitation/dispersion; cavity uses hard bury-delete (not SWIG-smoothed).
Free-energy forces: vacuum HF gradient on the solvent density plus
\(\partial\Delta G_{\mathrm{el}}/\partial R|_P\) (fixed-density rebuild of
the cavity).

### Practice

```cpp
cpu::scf::RHF sol;
sol.basis = cpu::scf::BasisSet::Pople631Gs;
sol.solvent_eps = 78.4;
sol.solvent_radii_scale = 1.2;
sol.solvent_x = 0.0;
sol.run(frame);
double dg_kcal = sol.e_solvent() * 627.509474;
// forces of G = E_HF + ΔG_el:
auto g = sol.forces(frame); // eV/Å on frame fx/fy/fz
```

`solvent_eps = 0` is the gas-phase path.

### Results

| System | Quantity | Engine | PySCF | Tol. |
|---|---|---:|---:|---|
| H₂O RHF/6-31G* C-PCM \(\varepsilon=78.4\) | \(\Delta G_{\mathrm{el}}\) / kcal mol⁻¹ | −8.02 | −7.96 | 0.1 |
| H₂O RHF/6-31G* IEF-PCM \(\varepsilon=78.4\) | \(\Delta G_{\mathrm{el}}\) / Ha | −0.012691 | −0.012658 | \(10^{-4}\) Ha |
| H₂O RHF/6-31G* C-PCM | forces vs FD of \(G\) | — | — | \(10^{-5}\) Ha/Bohr |

Regressions: `h2o_rhf_631gs_cpcm`, `ief-pcm-h2o`, `pcm-grad-h2o`.

---

## 20.7 Scalar Relativity

### Theory

x2c-1e replaces the one-electron Hamiltonian by a spin-free exact
two-component operator built from \(\{S,T,V,W\}\)
(\(W_{\mu\nu}=\langle\nabla\mu|V|\nabla\nu\rangle\)). Two-electron integrals
stay non-relativistic. **Mutually exclusive with ECP** (double-counting).
Property operators are not picture-change transformed. Forces with `x2c=true`
are not implemented.

### Practice

```cpp
cpu::scf::RHF x2c;
x2c.basis = cpu::scf::BasisSet::Def2TZVP;
x2c.x2c = true;
x2c.run(ar_frame);
```

### Results

| System | Quantity | Engine | PySCF `x2c1e` (`xuncontract=False`) | Tol. |
|---|---|---:|---:|---|
| Ar def2-TZVP | \(E_{\mathrm{x2c}}\) / Ha | (reg) | −528.64422317 | \(10^{-5}\) |
| Ar def2-TZVP | \(E_{\mathrm{x2c}}-E_{\mathrm{nr}}\) / Ha | \(\approx -1.84\) | −1.84146 | sign + 5% |

Regression: `x2c_argon`.

---

## Further reading

- Jensen, *Introduction to Computational Chemistry* (properties, solvation, relativity).
- Szabo & Ostlund, *Modern Quantum Chemistry*, §§3, 7.
- Dreuw & Head-Gordon, *Chem. Rev.* **105**, 4009 (2005); Barone & Cossi, *J. Phys. Chem. A* **102**, 1995 (1998); Peng & Reiher, *Theor. Chem. Acc.* **131**, 1081 (2012).

External pinning rules: [Appendix](appendix-pyscf-oracle.md).
