# Reference — collinear spin-DFTB2 analytic nuclear forces (O₂)

```text
oracle: dftb+
oracle_version: 24.1
```

Offline twin: `tools/tb_oracle/inputs/dftbplus/dftb2-spin-analytic-forces/`
(`geo.gen`, `dftb_in.hsd`, archived `dftb.out.txt`). Same spin block as
`spin-dftb2-o2` (PBE \(W_{\mathrm{pp}}(\mathrm{O})=-0.028\) Ha,
`ShellResolvedSpin = No`). Cluster O₂, r = 1.30 Å. ctest never launches DFTB+.

The **module-boundary** gate remains analytic \(F\) vs re-SCC central FD of
the same ATV energy. Absolute \(E\) and \(F_x\) are additionally pinned
against DFTB+ 24.1.

## Method

Collinear unrestricted SCC-DFTB2 (Elstner SCC + Köhler spin channel). Nuclear
forces are the **closed-form** gradient at the SCC density (dual α/β
Hellmann–Feynman + Pulay + spin-dependent Hamiltonian shifts on \(\partial S\)
+ fixed-\(\Delta q\) SCC \(\gamma\) + repulsive). \(W_A\) is
geometry-independent; fixed-\(m\) spin energy contributes no extra nuclear
term. This is the DFTB+ collinear spin force module boundary
(Hourahine et al., *J. Chem. Phys.* **152**, 124101 (2020)).

Energy: \(\mathrm{Tr}(\rho H^0)+E^{(2)}+E_{\mathrm{spin}}+E_{\mathrm{rep}}\).

## Hamiltonian / tables

- ATV class: `cpu::semiempirical::DFTB2`, `sk_set = MioSketch` (transcribed
  mio-1-1), `boundary = cluster`, `multiplicity = 3`,
  `spin_w_ha = basis_data::kDftbSpinConstantsPbeHa`.

## Geometry (Å)

O₂ triplet, **stretched** so \(|F|\neq 0\):

| atom | Z | x | y | z |
| ---- | - | - | - | - |
| O | 8 | 0.00 | 0.00 | 0.00 |
| O | 8 | 1.30 | 0.00 | 0.00 |

Isolated cluster. `charge = 0`, `multiplicity = 3`, `scc_tol = 1e-10`.

## Pins (hard-coded gates)

1. **Energy:** finite, \(|E| > 10^{-3}\) eV.
2. **Forces:** all components finite; \(|F|^2 > 10^{-8}\) at off-eq geometry.
3. **Translation:** \(|\sum_i F_i| \le 10^{-6}\) eV/Å per Cartesian.
4. **Action–reaction:** \(F_{0,x}+F_{1,x}=0\) (within \(10^{-6}\)).
5. **FD identity:** analytic \(F\) vs central **re-SCC** FD of the same \(E\)
   (\(h=10^{-4}\) Å) on \(F_{0,x}\) and \(F_{1,x}\) within **0.05 eV/Å**.

Measured ATV (2026-09-04, Release, 1 thread):

```text
E = -176.77425642 eV
F[0] = ( 4.91147291, 0, 0) eV/Å
F[1] = (-4.91147291, 0, 0) eV/Å
F_0_x analytic=4.91147291  fd=4.91147232  |d|=5.9e-07 eV/Å
```

## Offline DFTB+ 24.1

From archived `dftb.out.txt` (1 Ha = 27.211386245988 eV;
1 Ha/Bohr = 51.42206747467393 eV/Å):

| Quantity | DFTB+ | Converted |
| -------- | ----- | --------- |
| Total energy | −6.4963341184 H | **−176.774256878772 eV** |
| Energy SPIN | −0.0280000000 H | −0.761918814888 eV |
| \(F_{0,x}\) | +0.095512870310 Ha/Bohr | **+4.9114692618 eV/Å** |
| \(F_{1,x}\) | −0.095512870310 Ha/Bohr | −4.9114692618 eV/Å |

\(\lvert E_{\mathrm{ATV}}-E_{\mathrm{DFTB+}}\rvert=4.6\times10^{-7}\) eV,
\(\lvert F_{0,x}^{\mathrm{ATV}}-F_{0,x}^{\mathrm{DFTB+}}\rvert=3.6\times10^{-6}\) eV/Å
(both ≤ \(10^{-3}\) ⇒ `oracle: dftb+`).

Homonuclear O₂ has \(\Delta q\equiv 0\), so this gate does not exercise the
charge-dependent \(\gamma\) gradient. That discriminator is the NO doublet in
`dftb2-spin-analytic-stress`.

## Perf

```text
perf_exempt: true
perf_exempt_reason: DFTB+ 24.1 cold wall includes I/O; the analytic path
  replaces 6N re-SCC force evaluations with one post-SCC contraction
```

## Units

Public ATV: energy **eV**, positions **Å**, forces **eV/Å**.
