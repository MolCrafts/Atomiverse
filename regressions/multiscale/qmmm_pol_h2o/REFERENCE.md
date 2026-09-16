# qmmm_pol_h2o

## Physics

Induced-dipole polarizable QM/MM (Applequist / Thole damping). QM = H₂O RHF/STO-3G;
one polarizable MM site on the C₂ axis. Permanent field from QM nuclei;
\(E_{\mathrm{pol}}=-\frac12 k_e\sum_i\boldsymbol{\mu}_i\cdot\mathbf{E}_i^{\mathrm{perm}}\).

Refs: Thompson & Schenter, *J. Phys. Chem.* **99**, 6374 (1995); Thole, *Chem. Phys.* **59**, 341 (1981).

## Gate

Finite energies; \(E_{\mathrm{pol}}< -10^{-4}\) eV; total polarizable energy below fixed-charge `QmmmElectrostatic` by \(\ge 10^{-4}\) eV.
