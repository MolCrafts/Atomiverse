# CP2K offline oracle (CP2K capability loop)

Mirrors `tools/pyscf_oracle/`: produce **hard-coded pins** for regressions.
**ctest never launches CP2K.** Binding law:
[`.claude/notes/oracle-pin-provenance.md`](../../.claude/notes/oracle-pin-provenance.md)
and CLAUDE.md § Oracle capability loops.

## Contract (strict)

1. **Install CP2K** on the pin host (module, local prefix, or source build).
   Missing binary is an install task, not an excuse for literature / exempt.
2. Human or agent runs CP2K **offline** with a fixed input that matches the
   **same algorithm / representation class** ATV is claiming (GPW↔GPW GTH,
   etc.). **Forbidden:** unequal Hamiltonians as twins (e.g. GTH-GPW numbers
   anchoring NC-UPF plane-wave ATV) — those have **no comparison value**.
3. Record energy / forces / stress / frequencies / walls into
   `regressions/.../REFERENCE.md`: binary path, version/sha, input path,
   key output lines, date.
4. Regression asserts Atomiverse vs those numbers within stated tol.
5. **Perf is mandatory** for `oracle: cp2k` pins: fair wall on the same twin
   job; write \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}}\) (gate typically ≤ 10).
   `perf_exempt` is not a green path for "no binary" or mismatched models.
6. If CP2K does not implement the algorithm (or the scientific standard is a
   published number): `oracle: literature` + real citation — never a fake
   `oracle: cp2k` window.

## Suggested layout

```text
tools/cp2k_oracle/
  README.md          ← this file
  inputs/            ← CP2K .inp twins (committed with the pin)
  pin_notes.md       ← optional host install notes
  fair_bench.py      ← optional wall twin helper
```

## Version pin

Prefer a single CP2K release string in REFERENCE (e.g. `CP2K 2026.1` + git
sha). Do not mix versions across a regression without noting it.

## Relation to tutorials

- GPW narrative: `docs/tutorial/ch09-gpw.md`
- Loop product pages: `docs/tutorial/methods/<slug>.md` from `cp2k-method-loop`
