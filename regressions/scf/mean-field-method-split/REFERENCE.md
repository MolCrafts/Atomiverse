# Reference — mean-field method split on H₂/STO-3G and H₂O/6-31G

Out-of-library consumer. `RHF::run` / `RHF::forces` stay on the mean-field
type. TDHF, CIS(D), and spectral G0W0 are not members of `RHF`
(`TDHF<RHF>::run`, `CISD<RHF>::run`, `G0W0<RHF>::run` —
`pyscf.tdscf.rhf.TDHF`, Head-Gordon CIS(D) on `tdscf.TDA` amplitudes,
`pyscf.gw.gw_exact.GWExact`).

## Oracle

PySCF **2.14.0**, `tools/pyscf_oracle/pin_scf.py`. `cart=True`. Not an
Atomiverse measurement.

```bash
python tools/pyscf_oracle/pin_scf.py rhf --geom h2 --basis sto-3g --conv-tol 1e-12
python tools/pyscf_oracle/pin_scf.py tdhf --geom h2 --basis sto-3g --nstates 1 --conv-tol 1e-12
python tools/pyscf_oracle/pin_scf.py cisd --geom h2 --basis sto-3g --nstates 1 --conv-tol 1e-12
python tools/pyscf_oracle/pin_scf.py g0w0 --geom h2 --basis sto-3g --conv-tol 1e-12
python tools/pyscf_oracle/pin_scf.py rhf --geom water --basis 6-31g --grad --conv-tol 1e-12
```

CIS(D) is not a PySCF product class. The pin evaluates Head-Gordon
Chem. Phys. Lett. **219**, 21 (1994) on PySCF TDA amplitudes (same
offline helper as `cisd-h2co-sto3g`). G0W0 is `gw.gw_exact.GWExact` on
`dft.RKS(xc='hf')`, which builds `tdscf.dRPA` inside `kernel`.

## Geometry

| System | Basis | Coordinates [Å] |
| --- | --- | --- |
| H₂ | STO-3G | H (0,0,0), H (0,0,0.74) |
| H₂O | 6-31G | O (0,0,0.1173), H (0, ±0.7572, −0.4692) |

20 Å cubic box.

## Pins (PySCF 2.14.0)

| Observable | Value | Gate (existing) |
| --- | --- | --- |
| H₂ E_RHF | −1.1167593073964255 Ha | 1×10⁻⁸ Ha |
| H₂ TDHF S1 | 25.33200837852095 eV | 1×10⁻⁴ eV |
| H₂ TDHF f(S1) | 0.8857897216216963 | 1×10⁻³ |
| H₂ CIS S1 | 25.807465778338745 eV | 1×10⁻⁴ eV |
| H₂ CIS(D) S1 | 26.16497097331174 eV | 1×10⁻⁴ eV |
| H₂ G0W0 gap | 35.007583892679726 eV | 1×10⁻⁴ eV |
| H₂ G0W0 IP | 16.2440458258512 eV | 1×10⁻⁴ eV |
| H₂ G0W0 EA | 18.763538066828527 eV | 1×10⁻⁴ eV |
| H₂ QP HOMO | −0.5969576734903094 Ha | 1×10⁻⁵ Ha |
| H₂ QP LUMO | +0.6895473055730482 Ha | 1×10⁻⁵ Ha |
| H₂O ∇E (O, z) | +0.02381868594901171 Ha/Bohr | 1×10⁻⁶ Ha/Bohr |
| H₂O ∇E (H, y) | ∓0.004413880349737553 Ha/Bohr | 1×10⁻⁶ Ha/Bohr |
| H₂O ∇E (H, z) | −0.011909342974508963 Ha/Bohr | 1×10⁻⁶ Ha/Bohr |

H₂O forces are `grad_ha_bohr` from `pin_scf.py rhf --grad`. The full 3×3
atom-major matrix is in the regression. Energy and excitation gates match
`ch2o_tdhf`, `cisd-h2co-sto3g`, and `g0w0-h2-sto3g`.
