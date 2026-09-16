# C-PCM free-energy forces — H₂O / 6-31G*

*(Technical-report draft; narrative house style TBD — rewrite later.)*

Nuclear gradient of the **C-PCM solvation free energy**
\(G = E_{\mathrm{HF}}[P] + \Delta G_{\mathrm{el}}[P,R]\).
Regression: `regressions/scf/pcm-grad-h2o`. Energy-only C-PCM: `h2o_rhf_631gs_cpcm`.

## Theory

Conductor-like PCM adds surface charges and a free-energy term \(\Delta G_{\mathrm{el}}\).
Forces require consistent derivatives of both the vacuum HF pieces and the
solvent contribution. The product gate is **analytic vs finite-difference of
\(G\)** (not PySCF PCM forces at 1e-5 — surface discretizations differ; see
REFERENCE.md).

## Practice

```cpp
// See reg: RHF + C-PCM solvent model, then analytic free-energy gradient
// Public surface follows mean-field + solvent members used in the regression.
```

```bash
./build/regressions/reg_pcm-grad-h2o
```

## Results

### Accuracy

| Check | Criterion |
| ----- | --------- |
| Analytic vs FD of \(G\) | \(\le 10^{-5}\) Ha/Bohr element-wise |
| PySCF PCM force table | documented in REFERENCE; **not** the hard product gate |

### Speed (honest)

| Twin | \(T_{\mathrm{PySCF}}\) | \(T_{\mathrm{ATV}}\) | \(r\) | ≤10× |
| ---- | ---------------------: | -------------------: | ----: | ---- |
| vacuum `rhf` | 0.67 s | 42.4 s | **62.9** | **✗** |

This is the campaign’s known **speed debt**: the regression does full
solvent + FD-consistent work while the twin is vacuum RHF. Optimizing PCM
forces is the priority follow-up; do not treat \(r\sim 63\) as “pass.”
