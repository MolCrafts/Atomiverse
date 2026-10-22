# <Method name>

What the method computes and where it sits relative to its neighbours
(RHF/RKS/MP2/…, NVE/NPT/…). Two to four sentences, no pedagogy — the tutorial
chapters own the teaching narrative.

Public API: the concrete type and the members a reader calls, constructor
arguments only (no `*Config` bags), exactly as the case sections use them.

```cpp
// minimal call sequence shared by the sections
```

Case: `ctest -R '^<domain>/<method>$'` — sources and oracle record in
[`regressions/<domain>/<method>/`](../../../regressions/<domain>/<method>/).
One section of the case per `##` heading below:
`atv_regressions --case <domain>/<method> --section <name>`.

## Common setup

State once what every section shares: oracle program and version (PySCF,
CP2K, Quantum ESPRESSO, LAMMPS, MetalWalls, OpenMOPAC, DFTB+, xtb), shared
settings, shared limitations, shared literature. Omit when the sections share
nothing.

## <section> — <system / quantity> {#<section>}

One `##` per section of the case; the heading anchor is the section name, so
other pages link as `<domain>-<method>.md#<section>`. A variant documented
here but not run by the case carries the line
`*No section of the case runs this variant.*` directly under its heading.

### Method

Variant-specific equations, spin treatment, approximations; what is **not**
included.

### Validation

| Quantity | Atomiverse | Oracle | Tol | Notes |
| -------- | ---------: | -----: | --- | ----- |
| … | … | … | … | … |

System, basis/cutoff, convergence settings, oracle version and the output line
the pin came from. Never a number Atomiverse printed.

### Performance

Fair wall \(r = T_{\mathrm{ATV}}/T_{\mathrm{oracle}}\) on the same settings
class (gate \(r \le 10\)), or `perf_exempt` with the reason.

### Limitations

Scope specific to this variant.

## References

Literature and oracle citations shared by the sections; variant-specific ones
stay in their section.
