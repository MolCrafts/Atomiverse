# qmmm_link_ch3oh

## Physics

Additive electrostatic QM/MM with an **H link atom** on a covalent O–C cut of a
methanol-like fragment (Senn & Thiel, *Angew. Chem. Int. Ed.* **48**, 1198
(2009), DOI 10.1002/anie.200802019).

- QM: RHF/STO-3G on O + H(O) + link H
- MM: C + three methyl H as LJ environment; link C excluded from embedding charges
- Placement: \(\mathbf{r}_H = \mathbf{r}_O + g(\mathbf{r}_C-\mathbf{r}_O)\), \(g=0.709\)

## Gate

Finite `last_qm_energy` for \(g\in\{0.709,0.55\}\);
\(|E_{\mathrm{QM}}(0.709)-E_{\mathrm{QM}}(0.55)| > 10^{-3}\) eV;
\(E_{\mathrm{QM}}(0.709) < -100\) eV (LJ total is not gated — short contacts).
