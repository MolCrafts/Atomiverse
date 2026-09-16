# H₂ MP4(SDTQ)/STO-3G — e_corr pin

Closed-shell fourth-order Møller–Plesset MP4(SDTQ) on H₂ / STO-3G. Product gate
is the total correlation energy
\(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}+E^{(4)}\)
(observable `e_corr`, tolerance **1e-6 Ha**).

Oracle protocol: `docs/tutorial/appendix-pyscf-oracle.md` (offline pin only;
**never** import/invoke PySCF from CMake, ctest, or Atomiverse C++).
Pin helper: `tools/pyscf_oracle/pin_scf.py mp4`.

PySCF 2.14 does **not** ship `mp.MP4`. The pin rebuilds intermediate-normalization
Rayleigh–Schrödinger PT through order 4 on the full spin-orbital determinant
basis from **PySCF RHF orbitals + MO ERIs** (`ao2mo`). Cross-checks:
\(E^{(2)}\) matches `mp.MP2` to machine precision; \(E^{(2)}+E^{(3)}\) matches
the offline MP3 pin; \(E_{\mathrm{corr}}\) lies between MP3 and FCI/CISD on
this 2-electron system. Atomiverse implements the equivalent spin-orbital
SDTQ residual form (`RHF::mp4`).

## Geometry and settings

| Field | Value |
|---|---|
| Molecule | H₂, charge 0, singlet (\(2S=0\)) |
| Geometry (Å) | H(0,0,0); H(0,0,0.74) — pin_scf `h2` preset |
| Basis | STO-3G (`sto-3g`) |
| Angular convention | Cartesian (`cart=True`; STO-3G has no \(d\), cart is a no-op) |
| SCF | RHF, `conv_tol=1e-12` |
| Post-HF | Canonical RMP4(SDTQ) (no frozen core) |
| AO / MO count | 2 / 2 |
| Occupied spatial MOs | 1 |
| PySCF version | 2.14.0 |
| Pinned | 2026-07-23, venv `~/work/.x86_64` |

## Repro command

```bash
bash tools/setup_pyscf_env.sh   # once
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py mp4 \
    --geom h2 --basis sto-3g --conv-tol 1e-12
```

## Pinned values

| Quantity | Pin (Ha) | Tolerance |
|---|---|---|
| \(E_{\mathrm{RHF}}\) | −1.116759307396 | 1e-7 |
| \(E^{(2)}\) | −0.013138073590 | 1e-6 |
| \(E^{(3)}\) | −0.004836072637 | 1e-6 |
| \(E^{(4)}\) | −0.001711078792 | 1e-6 |
| \(E_{\mathrm{corr}}=E^{(2)}+E^{(3)}+E^{(4)}\) | **−0.019685225019** | **1e-6** (product gate) |
| \(E_{\mathrm{tot}}=E_{\mathrm{RHF}}+E_{\mathrm{corr}}\) | −1.136444532416 | 1e-6 |

Notes for this 2e system: triples and true quadruple excitations vanish
(\(n_{\mathrm{occ}}^{\mathrm{SO}}=2\)); \(E^{(4)}\) is carried by the doubles
residual channel plus renormalization \(E_Q^{(4)}=-E^{(2)}\|\Psi^{(1)}\|^2\).
FCI / CCSD correlation on the same geometry is −0.020524527092 Ha.

## Fair wall twin

```bash
python tools/pyscf_oracle/fair_bench.py --method mp4 --geom h2 --basis sto-3g \
    --repeat 3 ./build/regressions/reg_mp4-h2-sto3g
```

| Field | Value |
|---|---|
| \(T_{\mathrm{pyscf}}\) median (s) | 0.747 (det-space RSPT-4 twin) |
| \(T_{\mathrm{atv}}\) median (s) | 0.023 |
| ratio \(T_{\mathrm{atv}}/T_{\mathrm{pyscf}}\) | **0.031** |
| within_10x | **true** |
| perf_exempt | false |
| Fair-bench date | 2026-07-23 |
