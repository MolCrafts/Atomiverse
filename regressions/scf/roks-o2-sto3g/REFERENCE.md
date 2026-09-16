# O2 triplet ROKS/LSDA/STO-3G — reference values

## Literature

- C. C. J. Roothaan, *Self-Consistent Field Theory for Open Shells of
  Electronic Systems*, Rev. Mod. Phys. **32**, 179 (1960),
  DOI [10.1103/RevModPhys.32.179](https://doi.org/10.1103/RevModPhys.32.179) —
  single effective Fock for open shells; the spin-purity identity
  ⟨S²⟩ = S(S+1) this case asserts.
- M. F. Guest & V. R. Saunders, Mol. Phys. **28**, 819 (1974) — coupling
  coefficients used by PySCF's canonical Roothaan Fock
  (`get_roothaan_fock`), shared by this engine's `RohfCoupleStage`.
- T. Tsuchimochi & G. E. Scuseria, *Communication: ROHF theory made simple*,
  J. Chem. Phys. **133**, 141102 (2010),
  DOI [10.1063/1.3503173](https://doi.org/10.1063/1.3503173) — constrained-UHF
  reading of restricted open-shell (ROKS is the KS analogue).
- Slater exchange + VWN5 correlation (Atomiverse `XCType::LSDA`); matches
  PySCF `xc="LDA,VWN"` / `lda_x,lda_c_vwn` for this system.

## Pinned oracle (PySCF 2.14.0, run offline 2026-07-23)

```bash
# tools/setup_pyscf_env.sh  # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py roks \
    --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN --conv-tol 1e-12
```

```python
import pyscf
from pyscf import gto, dft
mol = gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g",
            spin=2, cart=True, unit="Angstrom")
mf = dft.ROKS(mol); mf.xc = "LDA,VWN"; mf.conv_tol = 1e-12
mf.kernel()          # -147.193715448518 Ha
mf.spin_square()     # (2.0, 3.0)
```

| Quantity | Pin | Tolerance |
|---|---|---|
| E(ROKS-LSDA, O2, (9,7), STO-3G cart) | −147.193715448518 Ha | 1e-6 Ha |
| ROKS ⟨S²⟩ − S(S+1) | 0 (analytic identity) | 1e-10 |
