# Reference — H₂O molecular DFT-D3 (BJ) dispersion energy (PBE params)

## Method

Grimme **DFT-D3 with Becke–Johnson (rational) damping** evaluated as a
geometry-only additive correction on the fixed water geometry below, using
the **PBE** reference-functional parameters. Basis STO-3G is recorded for
protocol consistency with the molecular KS path; E_disp does not depend on
the electronic density or the basis.

Public API: `cpu::scf::D3BJ` (`include/atomiverse/cpu/scf/dftd3.h`). Optional
fold-in on mean-field methods via `method.d3bj = true` then
`method.energy_disp()` after `run()`.

Formula (Grimme et al., *J. Comput. Chem.* **32**, 1456 (2011)):

\[
E_{\mathrm{disp}} = -\sum_{A<B}\left[
  \frac{s_6\,C_6^{AB}}{R_{AB}^6 + f(R_0^{AB})^6}
  + \frac{s_8\,C_8^{AB}}{R_{AB}^8 + f(R_0^{AB})^8}
\right],
\quad
f(R_0) = a_1 R_0 + a_2,\quad
R_0^{AB}=\sqrt{C_8^{AB}/C_6^{AB}}.
\]

PBE-D3(BJ) parameters: \(s_6=1\), \(s_8=0.7875\), \(a_1=0.4289\),
\(a_2=4.4407\) Bohr. Coordination numbers use the original Grimme form
(simple-dftd3 / PySCF `disp='d3bj'`); three-body ATM is off.

## Reference value

**E_disp (PBE-D3(BJ), H₂O) = −0.000359468656 Hartree**

at the pin geometry below.

Source: offline PySCF **2.14.0** + `pyscf-dispersion` (simple-dftd3 backend)
with `mf.disp = 'd3bj'`, `mf.xc = 'pbe'`. External-engine oracle — not a
golden snapshot of Atomiverse's own output. Primary literature: S. Grimme,
J. Antony, S. Ehrlich, H. Krieg, *J. Chem. Phys.* **132**, 154104 (2010)
(D3); S. Grimme, S. Ehrlich, L. Goerigk, *J. Comput. Chem.* **32**, 1456
(2011) (BJ damping).

## Geometry

PySCF oracle preset `water` from `tools/pyscf_oracle/pin_scf.py` [Å]:

| Atom | x | y | z |
|---|---|---|---|
| O | 0.000000 | 0.000000 | 0.117300 |
| H | 0.000000 | 0.757200 | −0.469200 |
| H | 0.000000 | −0.757200 | −0.469200 |

Gas-phase isolated molecule (no periodic images). A 20 Å cubic box may be
set on the Frame for SCF consumers; D3BJ ignores the box.

## Basis / angular convention

- Basis: **STO-3G** (protocol; E_disp is basis-independent)
- Angular: **Cartesian** (`cart=True`)

## Oracle: PySCF 2.14.0 (offline — nothing in CMake/ctest imports PySCF)

Interpreter: `~/work/.x86_64/bin/python` (pyscf **2.14.0**,
`pyscf-dispersion`).

### Exact command (pin helper)

```bash
~/work/.x86_64/bin/python tools/pyscf_oracle/pin_scf.py d3bj \
  --geom water --basis sto-3g --xc pbe
```

### Pinned JSON excerpt (2026-07-23)

```json
{
  "basis": "sto-3g",
  "cart": true,
  "e_disp_ha": -0.0003594686557095473,
  "e_tot_no_disp_ha": -75.22556669754539,
  "e_tot_with_disp_ha": -75.22592616620109,
  "geom": "water",
  "method": "d3bj",
  "pyscf": "2.14.0",
  "xc": "pbe"
}
```

## Tolerance and its rationale

**1 × 10⁻⁶ Ha.** Job gate for the pure dispersion energy. With matching
geometry, PBE-BJ parameters, and the in-tree Grimme C6 tables, Atomiverse is
expected to track the simple-dftd3 / PySCF E_disp to micro-Hartree or better.
The window is the regression acceptance band.

## What is checked

`cpu::scf::D3BJ{}.energy(frame)` returns E_disp [Hartree] within 1 × 10⁻⁶ Ha
of the pinned −0.000359468656 Ha. Non-finite or out-of-tolerance exits
non-zero with `got / want / tol`.
