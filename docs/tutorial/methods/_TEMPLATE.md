# &lt;Method name&gt; — &lt;system / basis&gt;

*(Technical-report draft; narrative house style TBD — rewrite later.)*

| Field | Value |
| ----- | ----- |
| Slug | `<slug>` |
| Regression | `regressions/scf/<slug>/` |
| Public API | `…` |
| Roadmap | link / row name |

## Summary

What the method computes, typical use, and where it sits relative to RHF/RKS/MP2/….
Three to six sentences. No pedagogy.

## Method

Equations, spin treatment, kernel/response level, important approximations.
Use display math where useful. State what is **not** included.

## Implementation

Public surface only (`cpu::Frame`, concrete method members; no `*Config` bags).

```cpp
// minimal call sequence — match regression
```

```bash
cmake --build build --target reg_<slug>
./build/regressions/reg_<slug>
```

Relevant headers / source paths if known.

## Validation (vs PySCF)

Source: `regressions/scf/<slug>/REFERENCE.md` (offline pin; never live in ctest).

| Quantity | Atomiverse / pin | PySCF | Tol | Notes |
| -------- | ---------------: | ----: | --- | ----- |
| … | … | … | … | … |

Geometry, basis, `cart`/spherical, `conv_tol`, PySCF version — all from REFERENCE.

## Performance

```bash
python tools/pyscf_oracle/fair_bench.py --method … -- \
  ./build/regressions/reg_<slug>
```

| Twin | \(T_{\mathrm{PySCF}}\) [s] | \(T_{\mathrm{ATV}}\) [s] | \(r\) | ≤10× | Caveat |
| ---- | -------------------------: | -----------------------: | ----: | ---- | ------ |
| … | … | … | … | ✓/✗ | … |

Gate: \(r = T_{\mathrm{ATV}}/T_{\mathrm{PySCF}} \le 10\) unless exempted.

## Limitations

Scope, known failures, speed debt, follow-ons.

## References

- Offline pin: `regressions/scf/<slug>/REFERENCE.md`
- Literature / specs …
