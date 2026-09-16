# Reference — Gaussian electrode charge kernels (E2)

oracle: literature  
oracle_version: Reed JCP 126 084704 (2007) Gaussian-charge Coulomb; MetalWalls JOSS 5 2373 (2020) / Scalfi constant-potential electrode form  
perf_exempt: true  
perf_exempt_reason: No MetalWalls binary and no LAMMPS ELECTRODE package on agent host; pin is analytic closed-form kernel (not a wall-time twin).

## Physics

Electrode atoms carry normalized 3D Gaussian charges of width \(\eta\) [Å⁻¹]:

\[
\rho_i(\mathbf{r}) = q_i \frac{\eta_i^3}{\pi^{3/2}}
  e^{-\eta_i^2|\mathbf{r}-\mathbf{R}_i|^2},
\quad
\eta_{ij}=\frac{\eta_i\eta_j}{\sqrt{\eta_i^2+\eta_j^2}},
\quad
G_{ij}(r)=\frac{\mathrm{erf}(\eta_{ij} r)}{r},\quad
G_{ii}=\eta_i\sqrt{\frac{2}{\pi}}.
\]

Energy (ATV metal units, `units::coulomb = 14.399645` eV·Å/e²):

\[
U=\texttt{coulomb}\Big[
  \sum_{i<j} q_i q_j G_{ij}(r_{ij})
  +\tfrac12\sum_i q_i^2 G_{ii}\Big].
\]

Gaussian–point: \(\eta_j\to 0\) flag (point) ⇒ \(\eta_{ij}=\eta_i\).  
Point–point: \(G=1/r\), no self.

These are the **same kernels MetalWalls uses** for electrode Gaussians. Offline
pin = direct evaluation of the published formula (no live MW run).

## Point-charge \(A\) is an approximation

Constant-potential BO solves \(A Q = \Psi\) (plus neutrality). Using
**point-charge** \(A_{ij}=\texttt{coulomb}/r_{ij}\) (no self diagonal) is
**not** the MetalWalls electrode model: it drops \(G_{ii}\) and replaces
\(\mathrm{erf}(\eta r)/r\) by \(1/r\). The toy two-site solve below shows
\(Q\) changes by \(O(1)\) (and can flip sign) under that replacement.
Production electrode MD must use the Gaussian kernel for \(A\).

## Geometry — fixed-\(Q\) energy pin

Cubic box \(L=100\) Å (min-image irrelevant at this spacing), cutoff 40 Å.

| Atom | type | \(\eta\) [Å⁻¹] | \(q\) [e] | \((x,y,z)\) [Å] |
| ---- | ---- | -------------- | --------- | --------------- |
| L0 | 1 | 1.0 | +0.25 | (0, 0, 0) |
| L1 | 1 | 1.0 | +0.25 | (2, 0, 0) |
| R0 | 1 | 1.0 | −0.25 | (0, 0, 10) |
| R1 | 1 | 1.0 | −0.25 | (2, 0, 10) |
| ion | 0 | point | +1 | (1, 0, 5) |

**Pin:** \(E = 1.938689655050537\) eV  
**Tol:** \(|E - E_{\mathrm{pin}}| < 10^{-9}\) eV  
**API:** `cpu::Evaluate` + `cpu::PairGaussianCoul{40, {0, 1.0}, /*self=*/true}`

## Geometry — toy \(Q\) vs point-charge \(A\)

Two Gaussian sites at \(z=0\) and \(z=L=10\) Å, \(\eta=1.0\) Å⁻¹,
\(\Psi=(+0.5,-0.5)\) eV/e, neutrality \(Q_1+Q_2=0\):

\[
Q_1=\frac{\Psi_1-\Psi_2}{2(A_{11}-A_{12})},\quad
A_{11}=\texttt{coulomb}\,G_{ii},\quad
A_{12}=\texttt{coulomb}\,G(L).
\]

| Model | \(A_{11}\) | \(A_{12}\) | \(Q_1\) [e] |
| ----- | ---------- | ---------- | ----------- |
| Gaussian (this feature) | \(\texttt{coulomb}\,\eta\sqrt{2/\pi}\) | \(\texttt{coulomb}\,\mathrm{erf}(\eta_{\mathrm{eff}} L)/L\) | **+0.0497547591576** |
| Point-charge \(A\) (approx) | 0 (no self) | \(\texttt{coulomb}/L\) | −0.347230782426 |

**Tol:** \(|Q_1^{\mathrm{gauss}} - Q_{\mathrm{pin}}| < 10^{-9}\) e  
**Requirement:** \(|Q_1^{\mathrm{gauss}} - Q_1^{\mathrm{point}}| > 10^{-6}\) e (documents the approximation).

## Public API (no Config bags)

```cpp
cpu::PairGaussianCoul pot(/*cutoff [Å]=*/40.0,
                          /*eta_by_type=*/{0.0, 1.0}, // type0 point, type1 Gaussian
                          /*include_self=*/true);
cpu::Evaluate eval;
eval.set_evaluator(std::move(pot));
eval.set_neighbor(cpu::md::NeighborList<>{40.0, /*skin [Å]=*/1.0});
eval.build();
eval.run(frame); // E on frame.meta[kEnergy] [eV]
```

`PairGaussianCoul` is a `NeighborConsumer` (it gained a neighbour-listed bulk
path in the wave-4 electrochem drop), so `Evaluate::build()` requires the list
instead of letting the style fall back to its O(N²) sweep unannounced. At
L = 100 Å with r_c = 40 Å every pair of this 5-atom cell is inside the cutoff,
so the list selects which loop runs and never which pairs are summed: the
energy reproduces the analytic pin to |ΔE| = 0.

Math-only: `cpu::GaussianChargeKernel::{eta_eff,greens,self_greens,dgreens_dr}`.

## Build (MD-only)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DATV_BUILD_MD=ON -DATV_BUILD_KSPACE=ON \
  -DATV_BUILD_TESTS=ON -DATV_BUILD_REGRESSIONS=ON \
  -DATV_BUILD_HF=OFF -DATV_BUILD_PW=OFF -DATV_BUILD_GPW=OFF \
  -DATV_BUILD_CUDA=OFF -DBLA_VENDOR=OpenBLAS
```

## Perf

**perf_exempt:** no oracle twin binary for wall timing. Kernel is \(O(N^2)\)
all-pairs for the foothold; neighbor-listed path is future work.
