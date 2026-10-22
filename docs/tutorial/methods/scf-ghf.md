# Generalized Hartree–Fock (GHF)

Generalized Hartree–Fock lets each spinor mix \(\alpha\) and \(\beta\)
components, so noncollinear spin arrangements are reachable that no UHF
(collinear) determinant can represent. The real two-component method is
`cpu::scf::GHF{basis, n_electrons}` (members `tol`, `max_iter`,
`guess_spin_rotation` for a per-atom Néel guess); `run(frame)` gives the
energy and `forces(frame)` returns a `cpu::scf::GradientResult` (energy [eV],
forces [eV/Å]). Complex spinor variants are `cpu::scf::ComplexGHF` and
Dirac–Hartree–Fock `cpu::scf::DHF`.

All sections run on CPU. Besides the gradient section documented below, the
case carries `h3_ghf_sto3g` (the H₃ GHF energy and the collinear-trap
checks) and `hf-complex-spinor` (DHF and complex x2c1e GHF energies on H₂ and
H₂O); their oracle records live in the case `REFERENCE.md`.

Case: `ctest -R '^scf/ghf$'` — sources and oracle record in [`regressions/scf/ghf/`](../../../regressions/scf/ghf/). One section of the case per `##` heading below: `atv_regressions --case scf/ghf --section <name>`.

## Common setup and oracle

The case needs `-DATV_BUILD_HF=ON -DATV_BUILD_REGRESSIONS=ON`. Pins are
offline PySCF **2.14.0** runs (`docs/tutorial/appendix-pyscf-oracle.md`);
nothing in ctest imports PySCF.

## References

- H. Fukutome, Int. J. Quantum Chem. **20**, 955 (1981), DOI 10.1002/qua.560200502
- T. Helgaker, P. Jørgensen, J. Olsen, *Molecular Electronic-Structure Theory*, §9–10
- P. Pulay, Mol. Phys. **17**, 197 (1969) (Pulay forces)

## ghf-grad-h3-sto3g — GHF analytic gradient, equilateral H₃ / STO-3G {#ghf-grad-h3-sto3g}

Real two-component GHF **nuclear forces** on the noncollinear Néel solution of
frustrated H₃; the energy pin is shared with the `h3_ghf_sto3g` section.

### Theory

On the spin-frustrated equilateral H₃ triangle the lowest GHF solution is a
coplanar 120° Néel arrangement. This section pins the **analytic**
\(\partial E/\partial R\) of that density (spinor energy-weighted density +
unsymmetrized exchange blocks). A UHF gradient at a collinear saddle is a
different surface and is not the oracle.

PySCF’s `GHF.nuc_grad_method()` is not implemented; the pin is the analytic
gradient evaluated with PySCF AO/dAO integrals on the pinned density,
cross-checked against central FD of `scf.GHF.kernel` (h = 1e-4 Bohr, max
|Δ| ≈ 1e-7 Ha/Bohr).

### Practice

```cpp
// equilateral side 1.0 Å in the xy-plane; 20 Å cubic box
cpu::scf::GHF ghf{cpu::scf::BasisSet::STO3G, /*n_electrons=*/3};
ghf.tol = 1e-12;
ghf.max_iter = 200;
ghf.guess_spin_rotation = 2.0 * M_PI / 3.0;   // 120° Néel guess
const cpu::scf::GradientResult res = ghf.forces(frame);  // energy [eV], forces [eV/Å]
```

### Results

#### Accuracy

Pinned analytic gradient dE/dR [Ha/Bohr] (from the case `REFERENCE.md`):

| Atom | ∂E/∂x | ∂E/∂y | ∂E/∂z |
|---|---|---|---|
| H (0) | +9.473114183e-02 | +5.469305020e-02 | 0 |
| H (1) | −9.473114183e-02 | +5.469305020e-02 | 0 |
| H (2) | 0 | −1.093861004e-01 | 0 |

| Check | Tol |
| ----- | --- |
| Element-wise vs analytic pin | **1e-5 Ha/Bohr** |
| Translational invariance \(\lvert\sum_A F_A\rvert\) | 1e-8 Ha/Bohr |
| Energy \(E_{\mathrm{GHF}}\) | −1.34044034354119 Ha |

#### Speed

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| proxy `uhf` H₂ | 0.32 s | 0.028 s | **0.088** | ✓ |

The twin is imperfect (different system); treat the ratio as a rough
order-of-magnitude only.
