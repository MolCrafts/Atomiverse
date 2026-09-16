# Reference — H₂O / STO-3G restricted BLYP total energy

## Method

Restricted Kohn–Sham (RKS) with the pure GGA functional **BLYP**
(Becke-1988 exchange + Lee–Yang–Parr correlation), minimal STO-3G Gaussian
basis, single-point total electronic + nuclear-repulsion energy on a fixed
water geometry. Driven by the concrete `cpu::scf::RKS` method with
`XCType::BLYP` through the public Frame API.

BLYP is pure semi-local GGA: `hybrid_exchange_fraction(BLYP) = 0` (no exact
HF exchange). Building blocks `b88_exchange_spin` / `lyp_correlation` are
shared with the B3LYP semi-local mix; BLYP is the unit-weight composite
(B88 + LYP), not the hybrid recipe.

## Reference value

**E_BLYP/STO-3G(H₂O) = −75.277447482665 Hartree**

at the pin geometry below (cart=True, conv_tol=1e-12).

Source: offline PySCF **2.14.0** pin (`dft.RKS`, `xc='blyp'`). This is an
external-engine oracle value, not a golden snapshot of Atomiverse's own
output. The STO-3G basis is from W. J. Hehre, R. F. Stewart, J. A. Pople,
*J. Chem. Phys.* **51**, 2657 (1969), doi:10.1063/1.1672392. B88:
A. D. Becke, *Phys. Rev. A* **38**, 3098 (1988). LYP: C. Lee, W. Yang,
R. G. Parr, *Phys. Rev. B* **37**, 785 (1988); B. Miehlich et al.,
*Chem. Phys. Lett.* **157**, 200 (1989).

## Geometry

PySCF oracle preset `water` from `tools/pyscf_oracle/pin_scf.py` [Å]:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.000000 | 0.117300 |
| H | 0.000000 | 0.757200 | −0.469200 |
| H | 0.000000 | −0.757200 | −0.469200 |

Derived metrics: R(O–H) ≈ 0.9578 Å, ∠H–O–H ≈ 104.48°. The molecule sits in
a 20 Å cubic box (gas-phase; the box only supplies a non-degenerate cell
and does not enter the isolated-molecule SCF).

## Basis / angular convention

- Basis: **STO-3G**
- Angular: **Cartesian** (`cart=True` in PySCF; STO-3G has no d shells, so
  cart/sph is a no-op here but is recorded for protocol consistency)

## Convergence

- Ground-state SCF: `conv_tol = 1e-12` (oracle); Atomiverse uses
  `tol = 1e-10` with default MatchPySCF grid.

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**).

### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py rks \
  --geom water --basis sto-3g --xc blyp --conv-tol 1e-12
```

### Pinned JSON excerpt (2026-07-23)

```json
{
  "basis": "sto-3g",
  "cart": true,
  "conv_tol": 1e-12,
  "converged": true,
  "e_tot_ha": -75.27744748266491,
  "geom": "water",
  "method": "rks",
  "pyscf": "2.14.0",
  "xc": "blyp"
}
```

## Tolerance and its rationale

**1 × 10⁻⁶ Ha.** Job gate for this pure-GGA total-energy case: with the
MatchPySCF grid and the same geometry/basis, Atomiverse's native B88+LYP
path is expected to track the PySCF total energy to micro-Hartree. The
window is the regression acceptance band, not a claim of machine-precision
identity of every intermediate.

## What is checked

`cpu::scf::RKS{XCType::BLYP, basis = STO3G}.run(frame)` converges
(`rks.converged()`) and the total energy `rks.energy()` [Hartree] (or
equivalently `frame.meta.get(kEnergy)` [eV] converted via
`units::kHartreeInEv`) lies within 1 × 10⁻⁶ Ha of the pinned
−75.277447482665 Ha. Non-convergence, a non-finite energy, or an
out-of-tolerance value exits non-zero with `got / want / tol`.
