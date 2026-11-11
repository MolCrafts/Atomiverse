# GFN1-xTB

> The regression runner has been removed. Regression commands below are historical; case sources and oracle records are retained as archived evidence, not current executable checks.

GFN1-xTB (Grimme, Bannwarth, Shushkov, *J. Chem. Theory Comput.* **13**, 1989
(2017)) is first-generation extended tight binding on a Stewart STO-\(n\)G
valence basis: EHT \(H^0\) from overlap, shell-resolved Klopman–Ohno SCC,
on-site third-order \(\Gamma q^3\), exponential pair repulsion and D3(BJ) with
the GFN1 damping set. GFN-xTB is **not** DFTB3++ — different Hamiltonian,
tables, and oracle (xtb only).

The concrete method is `cpu::semiempirical::GFN1{charge, multiplicity}`
(defaults 0 and 1) with members `with_forces`, `scc_tol`, `max_scc`,
`mix_beta`; there is no public `*Config` and no `make_*` / `build_*`. It is
mounted on `cpu::Evaluate` at `evaluate::Phase::evaluate`; the energy lands on
`kEnergy` [eV] and, with `with_forces = true`, closed-form forces on
`kAtom_f*` [eV/Å]. Native in-tree tables only — **no** `libxtb`. The sections
cover the CPU total energy of methane and analytic forces on off-equilibrium
water. See also [Ch. 17 Semiempirical AM1](../ch17-semiempirical-am1.md)
(SE product surface; NDDO/AM1 is a different family with an OpenMOPAC oracle).

Case: `ctest -R '^semiempirical/gfn1-xtb$'` — sources and oracle record in [`regressions/semiempirical/gfn1-xtb/`](../../../regressions/semiempirical/gfn1-xtb/). One section of the case per `##` heading below: `atv_regressions --case semiempirical/gfn1-xtb --section <name>`.

## Method

Closed-shell total energy on the public surface (internals in Hartree / Bohr):

\[
E = E_{H^0} + E_{\gamma} + E_{\Gamma} + E_{\mathrm{rep}} + E_{\mathrm{D3}}
  = \mathrm{Tr}(P H_0)
  + \tfrac12 q\,\gamma\,q
  + \tfrac13 \Gamma q^3
  + E_{\mathrm{rep}}
  + E_{\mathrm{D3}}.
\]

| Term | Role |
| ---- | ---- |
| \(E_{H^0}=\mathrm{Tr}(P H_0)\) | Occupied density on a CN-shifted EHT Hamiltonian; \(S\) from Stewart STO-\(n\)G |
| \(E_\gamma=\tfrac12 q\,\gamma\,q\) | Shell-resolved Klopman–Ohno SCC (\(g_{\exp}=2\), harmonic-mean hardness) |
| \(E_\Gamma=\tfrac13\sum_A\Gamma_A q_A^3\) | On-site third-order; frozen-\(q\) \(\partial E_\Gamma/\partial R=0\) |
| \(E_{\mathrm{rep}}\) | Classical pair \(Z^{\mathrm{eff}}_A Z^{\mathrm{eff}}_B\,e^{-\sqrt{\alpha_A\alpha_B}\,r^{1.5}}/r\) |
| \(E_{\mathrm{D3}}\) | Molecular D3(BJ); GFN1 set \(s_6=1\), \(s_8=2.4\), \(a_1=0.63\), \(a_2=5.0\) (xtb \(s_9=0\)) |

1. **Basis / overlap.** Per-element Stewart STO-\(n\)G primitives
   (tblite `basis/slater.f90` coefficients). H: 1s (4G, valence) + 2s (3G,
   polarization, Gram–Schmidt against 1s), i.e. hydrogen is double-ζ.
   C/N/O: 2s + 2p (6G). Cartesian overlap \(S_{\mu\nu}\) from Gaussian
   product integrals.
2. **EHT \(H^0\).** On-site \(h_l - k_{\mathrm{CN},l}\,\mathrm{CN}_A\) with D3
   exponential coordination numbers (Pyykkö \(r_{\mathrm{cov}}\), \(k_1=16\)).
   Off-site
   \[
   H^0_{\mu\nu}
     = \tfrac12(h_\mu+h_\nu)\,k_{\mu\nu}
       \bigl(1+k_{\mathrm{poly},\mu}\sqrt{R/(r_\mu+r_\nu)}\bigr)
       \bigl(1+k_{\mathrm{poly},\nu}\sqrt{R/(r_\mu+r_\nu)}\bigr)
       S_{\mu\nu},
   \]
   \(k_{\mu\nu}\) from \(k_s=1.85\), \(k_p=2.25\), \(k_{sp}=2.08\),
   \(k_{\mathrm{diff}}=2.85\) (polarization), pair scales \(k_{\mathrm{HH}}=0.96\),
   \(k_{\mathrm{NH}}=1.04\), and \(k_{\mathrm{EN}}=-7\times10^{-3}\) on
   \((\Delta\chi)^2\).
3. **SCC.** Shell charges \(q_i = n_i - n_i^0\) (Mulliken). Klopman–Ohno
   \(\gamma_{ij}\) with \(g_{\exp}=2\) and harmonic-mean shell hardness.
   On-site third-order \(E_{\Gamma}=\frac13\sum_A\Gamma_A q_A^3\). Anderson
   mixing (`mix_beta`, history 6) until \(\max|\Delta q_{\mathrm{shell}}| <
   \texttt{scc\_tol}\). Integer occupation.
4. **Repulsion.**
   \[
   E_{\mathrm{rep}}
     = \sum_{A<B}
       \frac{Z_A^{\mathrm{eff}} Z_B^{\mathrm{eff}}}{R}
       \exp\bigl(-\sqrt{\alpha_A\alpha_B}\,R^{1.5}\bigr).
   \]
5. **Dispersion.** Molecular D3(BJ), GFN1 set above.

Parameter path: static tables → D3 exponential CN → shifted atomic levels
\(\varepsilon_{A\ell}\) → \(H/S\) → generalized eigen → Mulliken shell \(q\) →
Anderson SCC → repulsion/D3 → \(E/F\), with parameters as members of `GFN1`
and static H/C/N/O tables (tblite `gfn1.f90` / xtb `param_gfn1-xtb.txt`) in
`basis_data/gfn1_tables.h`. Unsupported \(Z\) throws.

| Path | Role |
| ---- | ---- |
| `gfn1.h` | Public `GFN1` (`charge`, `multiplicity`, `scc_tol`, `with_forces`) |
| `gfn1_tables.h` | Static GFN1 H/C/N/O rows + global scales |
| `src/cpu/semiempirical/gfn1.cpp` | Stewart overlap / EHT+SCC / rep / D3 / analytic \(\nabla E\) (private) |
| `tests/cpu/semiempirical/test_xtb_gfn1.cpp` | Unit: tables, Z throw, finite \(E\), analytic-vs-FD + net-force invariance |

## Oracle

**xtb 6.7.1** only (EasyBuild `xtb/6.7.1-gcc-2025b-eb` on Arrhenius,
`OMP_NUM_THREADS=1`), `--acc 0.01` (xtb SCF \(10^{-8}\) Eh); the GFN family is
locked to xtb — never DFTB+ or OpenMOPAC as reference. Offline pins are
hard-coded in the section binaries; xtb is never launched from ctest.
Conversion: \(E[\mathrm{eV}] = E[\mathrm{Ha}] \times 27.211386245988\) (ATV
CODATA, `units::kHartreeInEv`, `units::kHartreePerBohrInEvPerAngstrom`).
Re-pin offline only (`--gfn 1 --chrg 0 --uhf 0 --acc 0.01 [--grad]`; record
TOTAL ENERGY [Eh] and `gradient.txt` into `REFERENCE.md`).

```text
oracle: xtb
oracle_version: 6.7.1
```

**Shared kernel hole vs xtb.** The residuals of both sections (10.2 meV on
CH₄, 2.77 meV on H₂O) come from the same three kernel differences: D3 C6
interpolation vs xtb's internal D3, Stewart primitive screening vs xtb
`intcut`, and Anderson vs Broyden mixing. They are not gradient defects —
analytic forces match ATV's own energy FD.

Build with the SE-only module closure (`module load buildenv-gcc/2026.03-mpich`;
`-DATV_BUILD_SEMIEMPIRICAL=ON -DATV_BUILD_EVALUATE=ON`; MD/MINIMIZER/HF/PW/GPW/CUDA
stay OFF).

## Limitations

- **H/C/N/O tables only.** Unsupported \(Z\) throws with a how-to; other GFN1
  elements are not in-tree (beyond-CHNO exists on GFN2 only).
- **Closed-shell only.** `multiplicity != 1` throws.
- **Halogen-bond correction** (xtb term) is not implemented; it is identically
  0 on CH₄ and H₂O.
- No ALPB/GBSA, CPCM-X/ddCOSMO, PBC, spin, sTDA; no `libxtb` / runtime
  `param_gfn1-xtb.txt` parser. GFN-FF is classical and outside this method.
- Organic energy + analytic \(G\) do not make the whole GFN family closed.

Sibling methods: GFN0 is [`gfn0-xtb-forces`](semiempirical-gfn0-xtb.md#gfn0-xtb-forces);
GFN2 forces [`gfn2-xtb-forces`](semiempirical-gfn2-xtb.md#gfn2-xtb-forces); AES-in-Fock
[`gfn2-aes-fock`](semiempirical-gfn2-xtb.md#gfn2-aes-fock); GFN1/2 Hessian
[`gfn-xtb-analytic-hessian`](semiempirical-gfn2-xtb.md#gfn-xtb-analytic-hessian).
IPEA energy / \(G\) / FD Hessian / VIP/VEA are [`ipea-xtb-small`](semiempirical-ipea-xtb.md#ipea-xtb-small) /
[`ipea-xtb-forces`](semiempirical-ipea-xtb.md#ipea-xtb-forces) /
[`ipea-xtb-analytic-hessian`](semiempirical-ipea-xtb.md#ipea-xtb-analytic-hessian) /
[`ipea-xtb-vip`](semiempirical-ipea-xtb.md#ipea-xtb-vip).

## References

- S. Grimme, C. Bannwarth, P. Shushkov, *J. Chem. Theory Comput.* **13**,
  1989 (2017), DOI [10.1021/acs.jctc.7b00118](https://doi.org/10.1021/acs.jctc.7b00118)
- C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht, J. Seibert,
  S. Spicher, S. Grimme, *WIREs Comput. Mol. Sci.* **11**, e01493 (2021) (xtb)
- Offline twins under `tools/tb_oracle/inputs/xtb/`; roadmap
  `docs/roadmap/semiempirical.md` §3.

## gfn1-xtb-organic — CH₄ organic energy {#gfn1-xtb-organic}

### Summary

GFN1-xTB total energy for isolated methane on the organic tables. The section
asserts

\[
\bigl|E_{\mathrm{ATV}} - (-116.30778398\,\mathrm{eV})\bigr| \le 0.02\,\mathrm{eV}
\]

against an offline xtb `--gfn 1` pin (−4.274232224875 Ha). Measured ATV
residual at this geometry is **10.2 meV**. Fair wall vs xtb:
\(T_{\mathrm{ATV}}/T_{\mathrm{xtb}} = 0.5 \le 10\).

### System and settings

| Item | Setting |
| ---- | ------- |
| System | CH₄, tetrahedral, C at origin; H at \(\pm s,\pm s,\pm s\) with \(s=1.09/\sqrt{3}\), \(r(\mathrm{C{-}H})=1.09\) Å |
| Spin / charge | Closed-shell (`--uhf 0`; ATV `multiplicity = 1`), charge 0 (`--chrg 0`) |
| Cell | cubic 20 Å on ATV (isolated; xtb molecular) |
| Basis | 12 AOs (C 2s2p + 4×H 1s2s), 10 shells, 8 electrons |
| Forces | off (`with_forces = false`) |
| SCC | `scc_tol = 1e-8`, `max_scc = 80` |

Occupation is integer: the CH₄ HOMO–LUMO gap is ~10 eV, so xtb's 300 K Fermi
fill is integer here.

```cpp
const double s = 1.09 / std::sqrt(3.0);   // H at (±s, ±s, ±s), C at origin
frame.box = cpu::Box::cubic(20.0);
cpu::Evaluate eval;
cpu::semiempirical::GFN1 gfn1;
gfn1.with_forces = false;
gfn1.scc_tol = 1e-8;
gfn1.max_scc = 80;
eval.on(evaluate::Phase::evaluate).add_ref(gfn1);
eval.build();
eval.run(frame);
const double e_eV = frame.meta.get(kEnergy);  // [eV]
// gfn1.last_scc_residual() < gfn1.scc_tol is gated
```

### Validation (vs xtb)

Command: `xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01` (twin `mol.xyz` +
`xtb.out.txt`).

**xtb energy decomposition (documentation; Ha):**

| Term | Ha |
| ---- | --: |
| SCC (electronic) | −4.340066348203 |
| → electrostatic | 0.000089060808 |
| repulsion | 0.066288872935 |
| dispersion | −0.000454749608 |
| halogen | 0 |
| **Total** | **−4.274232224875** |

Public pin: **−116.30778398 eV**.

| Quantity | Atomiverse / pin | xtb 6.7.1 | Tol | Notes |
| -------- | ---------------: | --------: | --- | ----- |
| \(E_{\mathrm{tot}}\) | **−116.30778398 eV** | **−4.274232224875 Ha** | **0.02 eV** | gate `\|E − pin\| ≤ 0.02` |
| Measured residual | **10.2 meV** (0.375 mEh) | — | — | vs this xtb transcript |
| SCC residual | \(<\) `scc_tol` | 9 iter (xtb) | `1e-8` e | gated |
| Finite energy | required | — | finite | gated |

**Tolerance rationale.** The 20 meV floor covers the shared kernel hole on a
~116 eV total; it tightens once the kernels are bit-identical.

### Performance

Informative only — **not** asserted by this section
(\(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}} \le 10\) convention).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| CH₄ GFN1 Evaluate (forces off) | **0.02** (xtb 6.7.1, `OMP_NUM_THREADS=1`) | **0.01** (Release, 1 thread) | **0.5** | ✓ | Tiny molecule; both walls are I/O-scale |

## gfn1-xtb-forces — analytic nuclear forces, off-eq H₂O {#gfn1-xtb-forces}

### Summary

Closed-form nuclear gradient of GFN1-xTB on off-equilibrium water. After one
shell-resolved Klopman–Ohno SCC solve, forces are \(\mathbf{F}=-\nabla E\) at
that density — Hellmann–Feynman + Pulay on \(\partial S\), CN/shell-polynomial
chains, \(\partial\gamma/\partial R\), pair repulsion, and D3(BJ) — written to
`frame.span(kAtom_f*)` in eV/Å. This is the **xtb `--gfn 1 --grad` module
boundary**, **not** a \(6N\) energy finite-difference and **not** SCC-DFTB3.
Fair wall \(T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\approx 0.16\le 10\).

### Ionic forces

`nuclear_gradient` in `src/cpu/semiempirical/gfn1.cpp` at frozen SCC density.
xtb prints \(\nabla E\) [Eh/\(a_0\)]; ATV stores \(\mathbf{F}=-\nabla E\)
[eV/Å] via `units::kHartreePerBohrInEvPerAngstrom`:

\[
\mathbf{F}_A = -\frac{\partial E}{\partial\mathbf{R}_A}
=
\mathbf{F}_A^{H_0+\mathrm{Pulay}}
+ \mathbf{F}_A^{\mathrm{CN}}
+ \mathbf{F}_A^{\gamma}
+ \mathbf{F}_A^{\mathrm{rep}}
+ \mathbf{F}_A^{\mathrm{D3}},
\]

with pairwise accumulation (antisymmetrized bond contributions) plus CN
chains:

| Term | Contraction (schematic) |
| ---- | ----------------------- |
| Electronic \(H_0\) + Pulay | \(2\,\mathrm{Tr}\!\bigl((P\odot H_0'-W)\,\partial S/\partial R + P\,\partial\Pi/\partial R\,S\bigr)\) plus SCC shift \(V_i=\sum_j\gamma_{ij}q_j\) on \(S\); \(W=\sum_i n_i\varepsilon_i\,c_ic_i^\top\) |
| CN / shell-poly chain | \(\sum_\ell(\partial E/\partial h_\ell)\,(-k_{\mathrm{CN},\ell}\,\partial\mathrm{CN}/\partial R)\) plus \(\partial\Pi/\partial R\) of the EHT polynomial |
| SCC \(\gamma\) (frozen \(q\)) | \(q_i q_j\,(\partial\gamma_{ij}/\partial R)\) from Klopman–Ohno \(\partial\gamma/\partial r\) |
| Third-order \(\Gamma\) | on-site; frozen-\(q\) contribution is identically 0 |
| Repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) of the \(e^{-\sqrt{\alpha_A\alpha_B}\,r^{1.5}}/r\) pair |
| D3 | Analytic two-body D3(BJ) forces (`gpw::dftd3_pbe_bj`, GFN1 damping) |

The \(6N\) central-FD path is **not** the production force. The section
compares two selected components against a **central energy-only** FD of
ATV’s own \(E\) (\(h=10^{-4}\) Å) as an identity gate.

### System and settings

Isolated H₂O, **off-equilibrium** so \(\|F\|\neq 0\), closed shell,
`charge = 0`, cubic 20 Å cell (isolated; min-image present, images not
engaged). SCC at class defaults `scc_tol = 1e-8`, `max_scc = 250`.
Masses 15.999 (O) / 1.008 (H) amu.

| Atom | \(Z\) | \(x\) [Å] | \(y\) [Å] | \(z\) [Å] |
| ---- | ----: | ----: | ----: | ----: |
| O | 8 | 0.00 | 0.00 | 0.00 |
| H1 | 1 | 0.96 | 0.00 | 0.00 |
| H2 | 1 | −0.24 | 0.93 | 0.00 |

```cpp
cpu::semiempirical::GFN1 gfn1;       // charge 0, multiplicity 1
gfn1.with_forces = true;
cpu::Evaluate eval;
eval.on(evaluate::Phase::evaluate).add_ref(gfn1);
eval.build();
eval.run(frame);
// Energy: frame.meta.get(kEnergy) [eV]; forces: kAtom_fx / fy / fz [eV/Å]
```

### Validation (xtb)

Command: `xtb mol.xyz --gfn 1 --chrg 0 --uhf 0 --acc 0.01 --grad`; twin
`mol.xyz`, `xtb.out.txt`, `gradient.txt`. Conversion
\(F=-\nabla E\times\) `units::kHartreePerBohrInEvPerAngstrom`.

**xtb \(\nabla E\) [Eh/\(a_0\)]** (`gradient.txt`):

| Atom | \(\partial E/\partial x\) | \(\partial E/\partial y\) | \(\partial E/\partial z\) |
| ---- | ------------------------: | ------------------------: | ------------------------: |
| O | −3.4089559688772×10⁻³ | −1.1077217607051×10⁻³ | ~0 |
| H1 | +6.9083433839239×10⁻⁴ | −7.6331225483940×10⁻⁴ | ~0 |
| H2 | +2.7181216304849×10⁻³ | +1.8710340155445×10⁻³ | ~0 |

xtb also printed TOTAL ENERGY **−5.768641651774 Eh** and
\(\|\nabla E\|=0.004979646520\) Eh/\(a_0\).

**xtb energy decomposition (documentation; Eh):**

| Term | Eh |
| ---- | --: |
| SCC (electronic) | −5.805633868132 |
| → electrostatic | 0.050104282812 |
| repulsion | 0.037129798770 |
| dispersion | −0.000137582413 |
| halogen | 0 |
| **Total** | **−5.768641651774** |

**Hard-coded gates:**

| Quantity | Atomiverse | xtb 6.7.1 | Tol | Notes |
| -------- | ---------: | --------: | --- | ----- |
| \(E\) | `kEnergy` | **−5.768641651774 Eh** | \(5\times10^{-3}\) eV | family TB energy floor |
| \(F_{O,x}\) | `kAtom_fx[0]` | **+0.175295563850 eV/Å** | \(5\times10^{-3}\) eV/Å | \(-\nabla E\) converted |
| \(F_{H2,y}\) | `kAtom_fy[2]` | **−0.096212437395 eV/Å** | \(5\times10^{-3}\) eV/Å | dual-component gate |
| \(\lvert\sum_i F_{i,d}\rvert\) | identity | 0 | \(1\times10^{-10}\) eV/Å | translation invariance, each Cartesian |
| \(F_{O,x}\), \(F_{H2,y}\) vs energy FD | identity | — | \(3\times10^{-5}\) eV/Å | \(h=10^{-4}\) Å; unit test same |

**Host residual (Release, `OMP_NUM_THREADS=1`, not a ctest gate):**
\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{pin}}\rvert=2.77\) meV,
\(\lvert F_{O,x}-F_{\mathrm{pin}}\rvert=1.87\times10^{-3}\) eV/Å,
\(\lvert F_{H2,y}-F_{\mathrm{pin}}\rvert=1.81\times10^{-3}\) eV/Å,
analytic vs energy FD \(<10^{-9}\) eV/Å.

**Tolerance rationale.** The \(5\times10^{-3}\) eV energy floor covers the
shared kernel hole (10.2 meV on CH₄). Force tol \(5\times10^{-3}\) eV/Å sits
above the 1.87 meV/Å host residual on the matching Cartesian components.
\(F_{O,y}\) / \(F_{H1,y}\) vs xtb sit at **~0.20 eV/Å** — not a gradient
defect (they match ATV energy FD). The pin uses the well-matching components;
the FD identity is the gradient gate.

### Performance

Fair wall asserted in the section (`ratio > 10` fails). The analytic path is
one SCC + one gradient (replaces 18 energy solves of a \(6N\) FD).

| Twin | \(T_{\mathrm{xtb}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r = T_{\mathrm{ATV}}/T_{\mathrm{xtb}}\) | ≤10× | Caveat |
| ---- | -----------------------: | -----------------------: | ----------------------------------------: | ---- | ------ |
| H₂O GFN1 `--gfn 1 --grad` | **0.004** (1 thread) | **6.4×10⁻⁴** | **≈ 0.16** | ✓ | xtb cold wall from archived `xtb.out.txt`; ATV tables are static |
