# Electronic Excitations

Status tables: [Excited States roadmap](../roadmap/excited-states.md).

Absorbing light without changing electron count and removing an electron
are different experiments. An orbital gap does not generally equal either
measured energy. Defining the target quantity is therefore the first step
in an excited-state calculation.

**Learning goals.** Distinguish vertical excitation, adiabatic relaxation,
and ionization; interpret response eigenvectors and oscillator strengths.

**Before you begin.** [Hartree–Fock Theory](ch02-rhf.md),
[Density Functional Theory](ch05-kohn-sham-dft.md), and
[Charges and Multipoles](ch20-hf-properties-environment.md).

## Response to a weak field

A vertical excitation compares states at the same geometry. An adiabatic
energy difference allows nuclear relaxation; zero-point and thermal
contributions are additional choices. Linear-response methods find
resonances in the electronic response to a weak time-dependent field.
Occupied–virtual orbital pairs are coupled, so excitation energies are
not generally bare orbital-energy differences.

For real TDHF matrices, the response problem can be written

$$\begin{pmatrix}A&B\\-B&-A\end{pmatrix}
\begin{pmatrix}X\\Y\end{pmatrix}
=\Omega\begin{pmatrix}X\\Y\end{pmatrix}.$$

$A$ contains gaps and interaction terms; $B$ couples excitation and
de-excitation amplitudes. The Tamm–Dancoff approximation neglects $B$.
TDDFT uses the corresponding exchange–correlation response kernel.
An unstable reference can produce unphysical response even after its SCF
iterations have converged.

The electric-dipole oscillator strength in atomic units is
$f_{0n}=\tfrac23\Omega_{0n}|\langle0|\hat{\boldsymbol\mu}|n\rangle|^2$.
A transition may exist but be weak or dark. Applying line broadening to
energy sticks is a plotting model, not a computed lifetime unless a
separate lifetime calculation has been performed.

Ionization instead compares different electron counts. A vertical
state-specific value is $I=E_{N_e-1}(\mathbf R_0)-E_{N_e}(\mathbf R_0)$.
The frozen-orbital Koopmans estimate omits relaxation and correlation changes.

## Practice plan

Use formaldehyde at an explicit ground-state geometry. Compare TDA and TDHF
at the same basis, spin channel, and converged stable reference. Record
transition character and intensity where available, not only root number.
States can exchange energy order when numerical or model settings change.

Test polarization functions for valence states and diffuse functions for
Rydberg character. Do not expect basis refinement to repair missing
double-excitation physics. The following response examples and result data
are relocated unchanged from the former combined properties chapter.

## TDHF / TDA excitation energies

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
non-physical; `TDHF<RHF>::run` runs a stability check first. Oscillator strengths use
the length-gauge transition dipole.

### Practice

```cpp
cpu::scf::RHF rhf{cpu::scf::BasisSet::Pople631Gs};
rhf.tol = 1e-12;
rhf.run(frame); // formaldehyde geometry in regressions

auto tda  = cpu::scf::TDA<cpu::scf::RHF>{rhf}.run(3);
auto tdhf = cpu::scf::TDHF<cpu::scf::RHF>{rhf}.run(3);
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

## Further reading

- Dreuw & Head-Gordon, *Chem. Rev.* **105**, 4009 (2005).
- Szabo & Ostlund, *Modern Quantum Chemistry*, chapter 7.
- [Reference provenance](appendix-pyscf-oracle.md) describes the requirements
  for reproducing the retained response data.

## Questions for reflection

Why is a neutral optical excitation not an ionization energy? What should be followed when two computed roots exchange their energy ordering?
