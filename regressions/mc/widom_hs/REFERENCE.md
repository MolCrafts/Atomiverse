# Reference — Widom hard-sphere free volume

## Formula

Widom (1963), *J. Chem. Phys.* **39**, 2808, DOI 10.1063/1.1734110:

\[
\mu_{\mathrm{ex}} = -k_B T \ln\langle e^{-\beta\Delta U}\rangle
\]

Fixed host: one hard sphere of diameter \(\sigma\) at the origin in a cubic
cell of edge \(L>2\sigma\) (exclusion ball does not wrap into itself).
Test particle overlaps if \(r<\sigma\):

\[
P_{\mathrm{insert}} = 1 - \frac{4\pi\sigma^3}{3L^3}\,,
\qquad
\mu_{\mathrm{ex}} = -k_B T\ln P_{\mathrm{insert}}.
\]

This is a **snapshot** free-volume golden (not bulk HS fluid CS μ).

## Parameters

| Quantity | Value |
|---|---|
| σ | 2.0 Å |
| L | 12.0 Å |
| T | 298.15 K |
| n_trials | 500000 |

## Pass

| Quantity | Tolerance |
|---|---|
| ⟨e⟩ vs \(P_{\mathrm{insert}}\) | 0.005 absolute |
| μ_ex vs analytic | \(0.002\,k_B T\) |
