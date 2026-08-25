# Reference — off-eq water GFN2-xTB analytic forces vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

## Method

GFN2-xTB nuclear gradient of the Bannwarth 2019 SCC energy

\[
E = \operatorname{Tr}(P H^{0})
  + E_{\mathrm{IES}}
  + E^{(3)}
  + E_{\mathrm{rep}}
  + E_{\mathrm{D4}}
  + E_{\mathrm{AES}}
\]

already shipped by `reg_gfn2-xtb-water` / `reg_gfn2-quadrupole-aes`. Forces
are the **closed-form** \(\mathbf{F}=-\nabla E\) (Hellmann–Feynman + Pulay +
CN chain + IES \(\partial\gamma/\partial R\) + pair repulsion + D4 + AES
kernel at frozen CAMM). There is no 6N re-SCC product path.

Reference: Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI 10.1021/acs.jctc.8b01176. Gradient module boundary: tblite
`get_hamiltonian_gradient`, `get_multipole_gradient_0d`.
**Not** DFTB+; **not** OpenMOPAC; **not** GFN0/1.

## Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host). Command:

```text
XTBPATH=.../share/xtb OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 2 --chrg 0 --uhf 0 --acc 1.0 --grad
```

Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/`
(`mol.xyz` + `xtb.out.txt` + `gradient.txt`). Nothing is executed at ctest
time. xtb prints \(\nabla E\) [Eh/a0]; ATV stores \(F=-\nabla E\) [eV/Å]
with CODATA \(a_0\) / Hartree from `units.h`.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−5.058625058504** | Eh |
| \(\|\nabla E\|\) | 0.109121448851 | Eh/a0 |
| Wall (xtb, 1 thread) | 0.018 | s |

### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| O | −7.5243103995119e−2 | −1.7310256877367e−2 | ~0 |
| H1 | +7.5511039446358e−2 | +1.5529339054184e−2 | ~0 |
| H2 | −2.6793545123835e−4 | +1.7809178231828e−3 | ~0 |

## Geometry

Gas-phase water, charge 0, singlet, **off equilibrium** (O–H stretched
0.96 → 1.10 Å) so \(|F|\neq 0\). Cubic box 20 Å (isolation only).

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 1.10  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

## Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E\) | **−5.058625058504** | Eh | \(1\times10^{-2}\) eV |
| \(F_{O,x}\) | **+3.869155970641** | eV/Å | \(0.15\) |

\(F = -\nabla E \times\) `units::kHartreePerBohrInEvPerAngstrom`.

**Tolerance rationale.** AES is still a **post-SCC energy** (not in the Fock),
so the offeq stretch residual is **8.9 meV** vs 26 µeV on the 0.96 Å energy
pin — energy floor \(10^{-2}\) eV. The stretch force \(F_{O,x}\) matches xtb
to \(\sim 0.09\) eV/Å (analytic vs ATV energy FD \(\sim 0.02\) eV/Å). The
angular \(F_{H1,y}\) is an AES-in-Fock / quadrupole-CAMM hole (xtb
\(-0.80\) eV/Å vs ATV energy FD \(+0.06\)) and is **not** a hard pin; it
is the next deepen (`gfn2` AES potential), not a defect of the H0/IES/rep/D4
closed-form path.

## Perf

xtb wall **0.018 s** (1 thread, archived transcript). ATV wall is measured
in the regression; fail if \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} > 10\).

## Gaps

- AES as an SCC Fock potential (post-energy kernel gradient only here)
- Elements beyond H/C/N/O
- ALPB, spin, PBC, IPEA, sTDA
