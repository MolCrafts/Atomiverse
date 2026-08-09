# Quantum ESPRESSO offline oracle (parity-gap loop, `oracle: qe`)

Mirrors `tools/cp2k_oracle/`: produce **hard-coded pins** for
`regressions/pw/` cases. **ctest never launches QE.** Binding law:
[`.claude/notes/oracle-pin-provenance.md`](../../.claude/notes/oracle-pin-provenance.md)
and CLAUDE.md § Oracle capability loops.

## Contract (strict)

1. **Install QE** on the pin host (module, local prefix, or source build).
   Missing binary is an install task, not an excuse for literature /
   `perf_exempt`. The current pin-host install is recorded in
   [`pin_notes.md`](pin_notes.md); runtime env =
   `source /nobackup/proj/disk/teoroo/personal/jicli594/work/qe/install/env.sh`.
2. Human or agent runs `pw.x` (or `bands.x` / `dos.x` / …) **offline** with a
   fixed input in the **same algorithm / representation class** ATV claims.
   For ATV `cpu::pw::PW` this is exact by construction: QE consumes the **same
   NC UPF file** from `tests/data/pw/` (ONCVPSP norm-conserving), same
   `ecutwfc`, same k-mesh, same XC, same smearing. **Forbidden:** unequal
   Hamiltonians as twins — in particular CP2K GTH-GPW numbers anchoring
   NC-UPF plane-wave ATV have **no comparison value**.
3. Record energy / forces / stress / bands / walls into
   `regressions/.../REFERENCE.md`: binary path, version, input path, key
   output lines, date. Commit the twin input under `inputs/<case>/`.
4. The regression asserts Atomiverse vs those numbers within stated tol.
5. **Perf is mandatory** for `oracle: qe` pins: fair wall on the same twin job
   (same process/thread shape, `OMP_NUM_THREADS` pinned and recorded); write
   \(T_{\mathrm{atv}}/T_{\mathrm{qe}}\). `perf_exempt` is not a green path for
   "no binary" or mismatched models.
6. If QE does not implement the algorithm (or the scientific standard is a
   published number): `oracle: literature` + real citation — never a fake
   `oracle: qe` window.

## Layout

```text
tools/qe_oracle/
  README.md          ← this file (contract + env pointer; no binaries)
  pin_notes.md       ← pin-host install record: version, build line, BLAS/FFT
  inputs/            ← QE input twins (committed with the pin)
```

## Version pin

Prefer a single QE release string in REFERENCE (currently
`Program PWSCF v.7.6`). Do not mix versions across a regression without
noting it.
