# Reference — O₂ triplet ROKS/LSDA/STO-3G nuclear gradient

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (external run, pinned).
Energy sibling: `regressions/scf/roks-o2-sto3g` (same O₂ geometry/basis/functional).
ROHF gradient sibling: `regressions/scf/o2_rohf_grad` (same geometry; HF W path).

| Field | Value |
|---|---|
| Geometry (Å) | O(0, 0, 0); O(0, 0, 1.2075) |
| Basis | STO-3G (Cartesian) |
| Method | ROKS(9,7) / LSDA (`XCType::LSDA` ≡ PySCF `LDA,VWN`) |
| Spin | triplet, ⟨S²⟩ = 2 exactly |
| Convergence | `conv_tol = 1e-12` |
| PySCF version | 2.14.0 |
| Observable | analytic nuclear forces [Ha/Bohr] |
| Tolerance | element-wise **2e-5** Ha/Bohr (nominal 1e-5; see residual note); translational invariance 1e-8 Ha/Bohr |

## Pinned oracle (offline, 2026-07-23)

```bash
# tools/setup_pyscf_env.sh  # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py roks \
    --geom o2 --basis sto-3g --spin 2 --xc LDA,VWN --conv-tol 1e-12 --grad
```

```python
import pyscf
from pyscf import gto, dft
mol = gto.M(atom="O 0 0 0; O 0 0 1.2075", basis="sto-3g",
            spin=2, cart=True, unit="Angstrom")
mf = dft.ROKS(mol); mf.xc = "LDA,VWN"; mf.conv_tol = 1e-12
mf.kernel()                          # -147.193715448518 Ha
g = mf.nuc_grad_method().kernel()    # pin below
```

| Quantity | Pin | Tolerance |
|---|---|---|
| E(ROKS-LSDA, O2, (9,7), STO-3G cart) | −147.193715448518 Ha | 1e-6 Ha (energy sibling) |
| ∂E/∂z [atom 0] | **+0.16292653404646806** Ha/Bohr | 1e-5 Ha/Bohr |
| ∂E/∂z [atom 1] | **−0.16292653404646806** Ha/Bohr | 1e-5 Ha/Bohr |
| ∂E/∂x, ∂E/∂y | ~0 (homonuclear diatomic on z) | 1e-5 Ha/Bohr |

Homonuclear diatomic along z → equal-and-opposite z-forces; x/y ~ machine zero.

## Residual vs PySCF analytic

PySCF 2.14.0 `nuc_grad_method` analytic and central FD of the same PySCF
energy (h = 1e-4 Bohr, atom 0 z) differ by **1.004e-5 Ha/Bohr** on this
system (LDA grid-response). Atomiverse analytic matches the **FD** value to
~7e-9 Ha/Bohr and sits 1.005e-5 from the PySCF analytic pin. Gate is therefore
2e-5 (still fails a wrong W by ≥1e-1 as on the ROHF sibling).

## Why this is a true ROKS gradient test

ROKS reuses the unrestricted KS electronic gradient (J + hybrid K + XC grid)
on the split-occupation α/β densities from one spatial MO set. The
ROKS-specific piece is the energy-weighted density (same as ROHF):

    W = O_a · F_a · O_a + O_b · F_b · O_b

with **KS spin Focks** F_σ (not the Roothaan effective Fock, not Σ ε_i C Cᵀ).
A naive W from effective-Fock eigenvalues is wrong by a large factor on this
geometry (same pathology as ROHF). The pin discriminates the Lagrangian.

Literature: PySCF `pyscf/grad/rohf.py` (`make_rdm1e`) + `pyscf/grad/uks.py`
(`grad_elec`); Roothaan, *Rev. Mod. Phys.* **32**, 179 (1960); Johnson,
Gill, Pople, *J. Chem. Phys.* **98**, 5612 (1993) (KS XC grid gradient).

## Performance (fair wall vs PySCF)

Gate: `T_atv_median / T_pyscf_median ≤ 10` (median of ≥3 runs).

```bash
# after build
export PATH="$HOME/work/.x86_64/bin:$PATH"
export LD_LIBRARY_PATH="$LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
python tools/pyscf_oracle/fair_bench.py --method roks_grad --geom o2 \
    --basis sto-3g --spin 2 --xc LDA,VWN --repeat 3 -- \
    ./build/regressions/reg_roks-grad-o2-sto3g
```

| Side | Median wall (s) | Notes |
|---|---|---|
| PySCF `dft.ROKS` + `nuc_grad_method` | **0.1767** | SCF + analytic grad; n=3 |
| Atomiverse `reg_roks-grad-o2-sto3g` | **0.1974** | SCF + analytic grad; n=3 |
| ratio `T_atv / T_pyscf` | **1.117** | within_10x=true (max 10) |

Measured 2026-07-23 on login node (OMP_NUM_THREADS=4, OMP_PROC_BIND=close):

```json
{"atv_median_s": 0.1974, "pyscf_median_s": 0.1767, "ratio_atv_over_pyscf": 1.117, "within_10x": true}
```
