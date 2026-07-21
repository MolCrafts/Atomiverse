# O2 triplet ROHF/STO-3G — reference values

## Literature

- C. C. J. Roothaan, *Self-Consistent Field Theory for Open Shells of
  Electronic Systems*, Rev. Mod. Phys. **32**, 179 (1960),
  DOI [10.1103/RevModPhys.32.179](https://doi.org/10.1103/RevModPhys.32.179) —
  the single-effective-Fock open-shell SCF and the single-MO-set spin-purity
  identity ⟨S²⟩ = S(S+1) this case asserts.
- M. F. Guest & V. R. Saunders, Mol. Phys. **28**, 819 (1974) — the coupling
  coefficient family used by PySCF's canonical ROHF (`get_roothaan_fock`),
  the convention this engine binds to.
- T. Tsuchimochi & G. E. Scuseria, *Communication: ROHF theory made simple*,
  J. Chem. Phys. **133**, 141102 (2010),
  DOI [10.1063/1.3503173](https://doi.org/10.1063/1.3503173) — the modern
  constrained-UHF reading of ROHF, and the origin of the UHF-vs-ROHF
  contamination/energy trade this case contrasts.

**Convention pitfall (why only PySCF numbers are pinned):** Roothaan /
Guest–Saunders / McWeeny–Diercksen coupling coefficients all give the SAME
total energy but DIFFERENT orbital energies. Total energy is the portable
observable; orbital energies are convention-bound and are only compared under
the PySCF canonical convention (in the unit tests, not here).

## Pinned oracle (PySCF 2.14.0, run offline 2026-07-21)

The spec named PySCF 2.13.1; the pinned oracle venv (`~/work/.x86_64`) ships
2.14.0 — recorded as-is.

```python
import pyscf
from pyscf import gto, scf
mol = gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g",
            spin=2, cart=True, unit="Angstrom")
mf = scf.ROHF(mol); mf.conv_tol = 1e-12
mf.kernel()          # -147.632166990682 Ha
mf.spin_square()     # (2.0, 3.0)

mfu = scf.UHF(mol); mfu.conv_tol = 1e-12
mfu.kernel()         # -147.633946785450 Ha
mfu.spin_square()    # (2.003410857681, ...)  -> contamination 3.410858e-3
```

| Quantity | Pin | Tolerance |
|---|---|---|
| E(ROHF, O2, (9,7), STO-3G cart) | −147.632166990682 Ha | 1e-8 Ha |
| ROHF ⟨S²⟩ − S(S+1) | 0 (analytic identity) | 1e-10 |
| UHF ⟨S²⟩ − 2 | 3.410858e-3 | asserted > 1e-3 |
| E(UHF) < E(ROHF) | variational ordering | strict |
