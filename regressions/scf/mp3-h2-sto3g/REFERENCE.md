# H₂ MP3/STO-3G — e_corr pin

Closed-shell third-order Møller–Plesset (RMP3) on H₂ / STO-3G. Product gate
is the total correlation energy \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}\)
(observable `e_corr`, tolerance **1e-6 Ha**).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py mp3`.

PySCF 2.14 does **not** ship `mp.MP3`. The pin rebuilds the Szabo–Ostlund /
Psi4NumPy closed-shell MP3 diagram sum on **PySCF RHF orbitals + MO ERIs**
(`ao2mo`), which is the same formula Atomiverse implements
(`RHF::mp3` → spin-orbital residual form of the same diagrams). Cross-check:
\(E^{(2)}\) matches `mp.MP2` to machine precision; \(E_{\mathrm{corr}}\) lies
between MP2 and FCI/CISD on this 2-electron system.

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — pin_scf `h2` preset |
| Basis | STO-3G (`sto-3g`) |
| Angular convention | Cartesian (`cart=True`; STO-3G has no \(d\), cart is a no-op) |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | Canonical RMP3 (no frozen core) |
| AO / MO count | 2 / 2 |
| Occupied spatial MOs | 1 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py mp3 \
    --geom h2 --basis sto-3g --conv-tol 1e-12
```

Equivalent explicit script (oracle formula):

```python
import numpy as np
from pyscf import gto, scf, ao2mo, mp

mol = gto.M(
    atom="""
H  0.000000  0.000000  0.000000
H  0.000000  0.000000  0.740000
""",
    basis="sto-3g",
    cart=True,
    unit="Angstrom",
)
mf = scf.RHF(mol)
mf.conv_tol = 1e-12
e_hf = mf.kernel()                 # -1.116759307396 Ha
print("MP2 check", mp.MP2(mf).run().e_corr)

C, eps = mf.mo_coeff, mf.mo_energy
nmo, ndocc = C.shape[1], mol.nelectron // 2
eri_chem = ao2mo.restore(1, ao2mo.full(mol, C), nmo)
# Physicist <pq|rs> = (pr|qs)
MO = np.einsum("prqs->pqrs", eri_chem)
o, v = slice(0, ndocc), slice(ndocc, nmo)
eocc, evirt = eps[:ndocc], eps[ndocc:]
Dinv = 1.0 / (
    eocc.reshape(-1, 1, 1, 1)
    + eocc.reshape(-1, 1, 1)
    - evirt.reshape(-1, 1)
    - evirt
)
# Szabo:1996 / Psi4NumPy MP3.py — 12 spatial diagrams (E3 only) + MP2
# ... (see tools/pyscf_oracle/pin_scf.py method "mp3")
```

## Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396 | 1e-7 Ha (reference only) |
| \(E^{(2)}\) (RMP2) | −0.013138073590 | 1e-6 Ha (diagnostic; matches `mp.MP2`) |
| \(E^{(3)}\) | −0.004836072637 | 1e-6 Ha (diagnostic) |
| \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}\) | **−0.017974146227** | **1e-6 Ha** (product gate) |
| \(E_{\mathrm{tot}}^{\mathrm{MP3}}\) | −1.134733453623 | 1e-6 Ha |

Cross-check: FCI/CISD correlation on the same geometry is −0.020524527092 Ha
(PySCF `fci.FCI` / `ci.CISD`); MP3 recovers ~87.6% of the correlation and
sits between MP2 and FCI as expected.

## Performance

Fair wall-time vs the offline oracle (same geom/basis/method class; median of
≥3 runs). Protocol: `tools/pyscf_oracle/fair_bench.py --method mp3`.

```bash
export OMP_NUM_THREADS=1
~/work/.x86_64/bin/python tools/pyscf_oracle/fair_bench.py \
    --method mp3 --geom h2 --basis sto-3g --repeat 5 \
    -- ./build/regressions/reg_mp3-h2-sto3g
```

Recorded 2026-07-23 (login node, `OMP_NUM_THREADS=1`, median of 5):

| Side | Median wall (s) | Notes |
|---|---|---|
| PySCF oracle (RHF + MP3 diagrams) | 0.01602 | `fair_bench.py --method mp3` |
| Atomiverse `reg_mp3-h2-sto3g` | 0.01241 | includes RHF + MP3 + asserts |
| **ratio** \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **0.774** | gate ≤ 10.0 — **within_10x=true** |

## Notes

- Product regression `reg_mp3-h2-sto3g.cpp` asserts the hard-coded
  \(E_{\mathrm{corr}}\) (and \(E_{\mathrm{tot}}\), \(E^{(2)}\), \(E^{(3)}\))
  pins — no live PySCF, no reading this file at ctest time. Public path:
  ```cpp
  cpu::scf::RHF rhf;
  rhf.basis = cpu::scf::BasisSet::STO3G;
  rhf.tol = 1e-12;
  auto r = rhf.mp3(frame);   // r.e_corr = E2+E3; r.e_corr_mp2 / e_corr_mp3
  ```
  Members / ctor args only — no Config bag (iron law).
- Literature: Szabo & Ostlund, *Modern Quantum Chemistry*, Table 6.2 /
  pp. 364–368 (third-order diagrams). Psi4NumPy
  `Moller-Plesset/MP3.py` is the spatial reduction used by the pin.
- Domain path under `regressions/scf/`; slug = queue id `mp3-h2-sto3g`.
