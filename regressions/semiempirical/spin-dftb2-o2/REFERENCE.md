# spin-dftb2-o2 — collinear spin-polarized SCC-DFTB2 on O₂

```text
oracle: dftb+
oracle_version: 24.1
```

Offline twin: `tools/tb_oracle/inputs/dftbplus/spin-dftb2-o2/`
(`geo.gen`, `dftb_in.hsd`, archived `dftb.out.txt` = DFTB+ 24.1 `detailed.out`
verbatim). SK: DFTB.org mio-1-1 `O-O.skf`. `SpinConstants { ShellResolvedSpin
= No; O = { -0.028 } }` — the published PBE table is
\(\{W_{ss},W_{sp},W_{ps},W_{pp}\}=\{-0.035,-0.030,-0.030,-0.028\}\) Ha
(DFTB+ order, \(l'\) fastest); DFTB+ `ShellResolvedSpin = No` accepts a
single constant and fills \(W_{ll'}\) with it, which is ATV's HOAO collapse
(\(W_{\mathrm{pp}}\) for O). `UnpairedElectrons = 2`. Cluster (`C`) geometry.
ctest never launches DFTB+.

Public API: `cpu::semiempirical::DFTB2` with `multiplicity = 3` and
`d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;` (`shell_resolved_spin`
stays default `false`). `sk_set` stays DFTB2's default `MioSketch` (transcribed
mio-1-1).

## Method

Collinear unrestricted SCC-DFTB2:

- Charge SCC: Elstner \(\gamma_{AB}\) + Hubbard \(U\) (Elstner et al.,
  *Phys. Rev. B* **58**, 7260 (1998),
  DOI [10.1103/PhysRevB.58.7260](https://doi.org/10.1103/PhysRevB.58.7260)).
- Spin channel: HOAO-collapsed \(W_A\) (Köhler / DFTB+ `ShellResolvedSpin =
  No`). Published PBE constants: DFTB+ manual appendix `app:spinconst`
  (Köhler, PhD thesis, Univ. Paderborn 2004; Köhler et al., *J. Phys. Chem. A*
  **111**, 5622 (2007), DOI 10.1021/jp068802p).
- Energy assembled as \(\mathrm{Tr}(\rho H^0)+E^{(2)}+E_{\mathrm{spin}}+E_{\mathrm{rep}}\)
  [eV], **not** a band sum.

## Geometry

O₂, bond length 1.21 Å (near experimental), isolated cluster:

| Atom | Z | x [Å] | y | z |
|------|---|-------|---|---|
| O    | 8 | 0.00  | 0 | 0 |
| O    | 8 | 1.21  | 0 | 0 |

Charge 0, multiplicity **3** (\(n_\alpha=7\), \(n_\beta=5\)).
`scc_tol = 10^{-10}`, max 80 iterations. Forces off (energy + spin gate).

## Why \(\Delta E_{\mathrm{total}} = \Delta E_{\mathrm{spin}}\) exactly

This is an identity, not an estimate. Four steps, each named in
`src/cpu/semiempirical/dftb2.cpp`:

1. **Total energy is not a band sum.** The assembly is
   \(E=\mathrm{Tr}(\rho H^0)+E^{(2)}+E_{\mathrm{spin}}+E_{\mathrm{rep}}\).
   The spin shift never enters the first term.
2. **The shift is the same constant on both atoms.** O₂ is homonuclear ⇒
   \(\Delta q_A\equiv 0\) ⇒ Coulomb shift vanishes; \(\varepsilon^\sigma_A=\sigma W_A m_A\)
   is one constant \(c\).
3. **`apply_shift` then gives \(H^\sigma=H^0\pm cS\).** Equal on-site shifts
   make `half = ½(ε_a+ε_b)` identically \(c\).
4. **The generalised eigenproblem has the same eigenvectors.**
   \((H^0+cS)C=\varepsilon SC \iff H^0 C=(\varepsilon-c)SC\): eigenvectors
   unchanged, eigenvalues translated by \(c\). Therefore \(\rho^\alpha,\rho^\beta\)
   are unchanged ⇒ \(m_A\) stays self-consistently 1.0, \(\mathrm{Tr}(\rho H^0)\),
   \(E^{(2)}\) and \(E_{\mathrm{rep}}\) are unchanged.

The only term that moves is \(E_{\mathrm{spin}}\). For O₂ triplet,
\(m_A=1\) by symmetry so \(E_{\mathrm{spin}}=W_{\mathrm{pp}}(\mathrm{O},\mathrm{PBE})\).
Sketch \(W=-1\) eV → published \(W=-0.028\) Ha \(=-0.761918814888\) eV ⇒
\(\Delta E_{\mathrm{spin}}=+0.238081185112\) eV. **1e-4 eV is numerical-noise
margin on that equality.** A measured deviation beyond it is an iron-law-9
stop (`path:line` + the measured value) — never a reason to widen the
tolerance.

## Pins (hard-coded gates)

| Quantity | Value | Tol | Unit |
| -------- | ----- | --- | ---- |
| \(E_{\mathrm{spin}}=W_{\mathrm{pp}}(\mathrm{O},\mathrm{PBE})\) | **−0.761918814888** | \(10^{-9}\) | eV |
| Total energy \(E\) | **−177.004501172488** | \(10^{-4}\) | eV |
| \(E-E_{\mathrm{old}}\) vs superseded sketch-W pin −177.2425823576 | **+0.238081185112** | \(10^{-4}\) | eV |
| \(\sum_A m_A\) (Mulliken spin) | 2.0 | 0.25 | e |
| Per-atom \(\lvert m\rvert\) | > 0.3 each | — | e |
| \(\lvert m_0-m_1\rvert\) | < 0.2 | — | e |

Measured ATV (2026-09-04, Release, `OMP_NUM_THREADS=1`):
\(E=-177.004501172479\) eV, \(m=[1,1]\), \(\Delta q=[0,0]\),
\(E-E_{\mathrm{old}}=+0.238081185121\) eV
(\(|\mathrm{shift}-0.238081185112|=9\times10^{-12}\) eV).

## Offline DFTB+ 24.1

From archived `dftb.out.txt` (1 Ha = 27.211386245988 eV):

| Quantity | DFTB+ | Converted |
| -------- | ----- | --------- |
| Total energy | −6.5047954387 H | **−177.004501133607 eV** |
| Energy SPIN | −0.0280000000 H | **−0.761918814888 eV** |
| Energy SCC | 0 (Δq = 0) | 0 |
| Gross atomic spins | 1.000, 1.000 | e |

\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{DFTB+}}\rvert=3.9\times10^{-8}\) eV
(≤ \(10^{-3}\) eV ⇒ `oracle: dftb+`). Spin populations and \(E_{\mathrm{spin}}\)
are bit-identical to the published \(W_{\mathrm{pp}}\).

## Perf

```text
perf_exempt: true
perf_exempt_reason: DFTB+ 24.1 wall is a cold process + I/O (~0.3 s); ATV
  Evaluate+DFTB2 is ~0.02 s. Not a fair kernel-vs-kernel wall.
```

## Units

Public ATV: energy **eV**, positions **Å**, Mulliken populations **e**.
