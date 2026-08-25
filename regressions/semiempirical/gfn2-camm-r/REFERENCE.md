# Reference — off-eq water GFN2-xTB two-centre CAMM quadrupole vs xtb

```text
oracle: xtb
oracle_version: 6.7.1
```

## Method

GFN2-xTB nuclear gradient of the Bannwarth 2019 SCC energy with anisotropic
electrostatics in the Fock **and** xtb `mmompop` two-centre CAMM quadrupole
on every centre (including s-only H), plus frozen-\(P\)
\(\partial\Theta/\partial R\) in the analytic \(G\):

\[
E = \operatorname{Tr}(P H^{0})
  + E_{\mathrm{IES}}
  + E^{(3)}
  + E_{\mathrm{rep}}
  + E_{\mathrm{D4}}
  + E_{\mathrm{AES}}
\]

\(E_{\mathrm{AES}}\) is charge–dipole, dipole–dipole, charge–quadrupole +
dipole/quadrupole AXC. This is the deepen of `gfn2-aes-fock`: AES-in-Fock
does not close angular \(F_{H1,y}\) (ATV energy FD \(\sim+0.06\) eV/Å vs
xtb \(-0.80\)). Frozen-CAMM \(\partial Q/\partial R\) on \(F_{O,x}\) is the
same CAMM completeness hole.

Reference: Bannwarth, Ehlert, Grimme, *J. Chem. Theory Comput.* **15**, 1652
(2019), DOI 10.1021/acs.jctc.8b01176. Formula twin: xtb 6.7.1
`src/aespot.f90` `mmompop` + `aniso_grad` / `setdvsdq`. **Not** DFTB+;
**not** OpenMOPAC; **not** GFN0/1.

## Offline oracle run

Binary: xtb 6.7.1 (`xtb/6.7.1-gcc-2025b-eb` on the pin host). Command:

```text
XTBPATH=.../share/xtb OMP_NUM_THREADS=1 \
  xtb mol.xyz --gfn 2 --chrg 0 --uhf 0 --acc 1.0 --grad
```

Twin: `tools/tb_oracle/inputs/xtb/h2o_gfn2_offeq_grad/`
(`mol.xyz` + `xtb.out.txt` + `gradient.txt`). Re-run 2026-08-25 reproduced
the archived TOTAL ENERGY and \(\nabla E\) to all printed digits. Nothing
is executed at ctest time. xtb prints \(\nabla E\) [Eh/a0]; ATV stores
\(F=-\nabla E\) [eV/Å] with CODATA \(a_0\) / Hartree from `units.h`.

| Quantity | Value | Unit |
| -------- | ----- | ---- |
| Total energy | **−5.058625058504** | Eh |
| \(\|\nabla E\|\) | 0.109121448851 | Eh/a0 |
| Wall (xtb, 1 thread) | 0.015 | s |

xtb internal wall on the re-run is 0.015 s (archived sibling transcript
0.018 s). Gate uses **0.015 s**.

### xtb \(\nabla E\) [Eh/a0] (`gradient` file)

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------- | ------------------------- | ------------------------- |
| O | −7.5243103995119e−2 | −1.7310256877367e−2 | ~0 |
| H1 | +7.5511039446358e−2 | +1.5529339054184e−2 | ~0 |
| H2 | −2.6793545123835e−4 | +1.7809178231828e−3 | ~0 |

## Geometry

Gas-phase water, charge 0, singlet, **off equilibrium** (O–H stretched
0.96 → 1.10 Å) so \(|F|\neq 0\). Cubic box 20 Å (isolation only). Same
geometry as `reg_gfn2-aes-fock` / `reg_gfn2-xtb-forces`.

| Atom | x [Å] | y [Å] | z [Å] |
|------|-------|-------|-------|
| O    | 0.00  | 0.00  | 0.00  |
| H    | 1.10  | 0.00  | 0.00  |
| H    | −0.24 | 0.93  | 0.00  |

## Pins

| Quantity | Value | Unit | Tol |
| -------- | ----- | ---- | --- |
| \(E\) | **−5.058625058504** | Eh | \(1\times10^{-2}\) eV |
| \(F_{O,x}\) | **+3.869155970641** | eV/Å | \(0.35\) |
| \(F_{H1,y}\) | **−0.7986** | eV/Å | \(0.35\) (**gated**) |

\(F = -\nabla E \times\) `units::kHartreePerBohrInEvPerAngstrom`.

**Tolerance rationale.** AES-in-Fock left angular \(F_{H1,y}\) unmatched
(ATV energy FD \(\sim+0.06\) vs xtb \(-0.80\); analytic \(+0.136\)). This
slug gates both stretch and angular components at the same 0.35 eV/Å
envelope used for \(F_{O,x}\) in `reg_gfn2-aes-fock`. Energy floor 10 meV
covers remaining mixer / D4 / intcut residual (sibling 9.19 meV). Dual
gate includes analytic vs ATV energy FD.

**Wave-12 finding (not closed).** Analytic \(G\) now chains frozen-\(P\)
\(\partial\Theta/\partial R\) on p-centres: off-eq \(F_{O,x}\) residual
**0.236** eV/Å (was 0.306 in `reg_gfn2-aes-fock`), energy 9.19 meV,
analytic-vs-FD holds, \(T_{\mathrm{atv}}/T_{\mathrm{xtb}}\approx 0.14\).
Angular \(F_{H1,y}\) stays **+0.112** vs xtb **−0.799** (0.91 eV/Å).
Naive two-centre \(\Theta\) on H (same origin-shift as O) is the
documented 23 meV eq-H₂O residual and does **not** tilt \(F_{H1,y}\)
(stays \(\sim+0.12\)). Flipping the \(R\otimes R\,S\) sign to the xtb
`opab4` convention breaks SCC (residual stuck \(7\times10^{-6}\),
\(\Delta E\sim 0.67\) eV). s–p-only H \(\Theta\) is 32 meV and still
\(F_{H1,y}=+0.10\). xtb `mmompop` on H is **not** that dump — do not
bless it. Dipole–quadrupole / quadrupole–quadrupole pair ranks wait.

## Perf

xtb wall **0.015 s** (1 thread, this re-run). ATV wall is measured in the
regression; fail if \(T_{\mathrm{atv}}/T_{\mathrm{xtb}} > 10\).

## Gaps

- Dipole–quadrupole / quadrupole–quadrupole pair AES ranks
- Elements beyond H/C/N/O
- ALPB, spin, PBC, sTDA
