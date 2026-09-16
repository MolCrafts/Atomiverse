# mp2-d3bj-forces-h2o-sto3g

## What is gated

Public product path:

\[
E = E_{\mathrm{RMP2/STO\text{-}3G}} + E_{\mathrm{D3(BJ)}}
\quad,\quad
\mathbf{F} = -\nabla E
\]

on a **water dimer** (not a monomer). The monomer intramolecular D3 force is
\(\sim 7\times 10^{-5}\) eV/Å — smaller than a comfortable FD tolerance — so a
missing `add_dispersion_forces` leg is invisible. The dimer carries a material
intermolecular D3 contribution (\(\sim 10^{-2}\) eV/Å class on the oxygens).

## Checks

1. Materiality: \(|F_{\mathrm{D3}}|_{\max} > 10^{-3}\) eV/Å (bare `D3BJ{}.forces`).
2. Analytic `RHF::mp2_forces` with `d3bj=true` vs central FD of `mp2().e_tot`
   on both oxygens, all three Cartesian components; tol \(5\times 10^{-4}\) eV/Å.
3. Translational invariance of the total analytic force,
   \(|\sum_A F_A| < 10^{-5}\) eV/Å.

## What is NOT claimed

- Not a literature absolute pin of \(E_{\mathrm{MP2}}\) or \(E_{\mathrm{D3}}\).
- Not a PySCF dual-gate (bare MP2 gradient is already pinned by
  `mp2-grad-h2o-sto3g`).
- Grimme D3(BJ) model: Grimme et al., *J. Chem. Phys.* **132**, 154104 (2010);
  *J. Comput. Chem.* **32**, 1456 (2011). PBE-parameter set used by `scf::D3BJ`.

## Geometry

Water dimer (Å), same as `dftb2-dispersion-d3` / simple-dftd3 water-dimer pin:

| Atom | x | y | z |
|------|---|---|---|
| O | 0 | 0 | 0 |
| H | 0.757 | 0 | 0.586 |
| H | −0.757 | 0 | 0.586 |
| O | 0 | 0 | 2.900 |
| H | 0.757 | 0 | 3.486 |
| H | −0.757 | 0 | 3.486 |

Cubic box 20 Å (isolated).
