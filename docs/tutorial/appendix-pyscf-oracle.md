# Appendix. External References

Rules for the **Results** block of every tutorial chapter: how to cite PySCF
(or other references) without making them build dependencies.

## 1. External codes are not build dependencies

Reference energies, properties, and basis parameters are computed offline by
developers and written into tables, regression comments, or `REFERENCE.md`
files. CMake, ctest, and unit tests must not import, invoke, or read PySCF or
its data files. Version strings and provenance in comments are allowed.

Optional local setup for offline comparison:

```bash
bash tools/setup_pyscf_env.sh
```

## 2. Every tabulated reference should be independently reproducible

Each pinned value should record:

| Item | Example |
|---|---|
| Geometry and units | H₂O; O(0, 0, 0.1173), H(0, ±0.7572, −0.4692) Å |
| Basis and angular convention | 6-31G* (Cartesian d) / def2-SVP (spherical) |
| Convergence settings | `conv_tol=1e-12` |
| Program version | PySCF 2.14.0 |
| Call sequence | `mf = scf.RHF(mol); mf.kernel()` |

## 3. Units and tolerances

- Energy: Hartree internally; many user-facing paths convert to eV.
- Property tolerances depend on the quantity—for example
  \(10^{-8}\)–\(10^{-6}\,\mathrm{Ha}\) for energies,
  \(10^{-6}\)–\(10^{-5}\) a.u. for dipoles, \(\sim 1\,\mathrm{cm}^{-1}\) for
  frequencies, \(\sim 0.1\,\mathrm{kcal\,mol}^{-1}\) for solvation energies.
  Use the tolerances stated in each chapter table.

## 4. Tables and figures

Scalar results belong in tables. Continuous data (convergence histories, basis
ladders) may use figures when a figure clarifies the argument; figures should
not merely restate numbers already in a table.

## 5. Basis tables and the generator

Consistency of static basis headers with the PySCF basis library is checked
with the generator:

```bash
python tools/gen_basis_data.py --check \
    include/atomiverse/cpu/scf/basis_data/sto3g.h sto-3g sto3g H He …
```

When editing tables, prefer regenerating them and keep the generator command
and version in the header comment.

### Further reading

- Sun et al., *WIREs Comput. Mol. Sci.* **8**, e1340 (2018) (PySCF).
