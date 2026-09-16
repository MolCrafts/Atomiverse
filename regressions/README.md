# regressions/ — out-of-library textbook end-to-end

The **only** home of end-to-end in this repository.

```bash
cmake -B build -DATV_BUILD_REGRESSIONS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target regressions     # builds AND runs every case
```

## Why a third directory

| | runs | tests | hard prohibition |
|---|---|---|---|
| `tests/` | **whole suite in seconds** — the fast-iteration loop | **one function or method** | **e2e is strictly forbidden** |
| `regressions/` | one case ≤ 60 s | **end-to-end, against the literature** | never in the default `ctest` |
| `benchmarks/` | at acceptance time | parity vs an external **engine** (LAMMPS/PySCF/CP2K), and speed | never in `ctest` |

`benchmarks/` asks *"are we as fast and as accurate as LAMMPS?"* — it needs a
third-party engine installed. `regressions/` asks *"are we still as correct as
the **textbook**?"* — it needs nothing, so it can run every cycle.

## The rules

1. **Out-of-library consumer.** Include only public `atomiverse/…` headers; link
   only the `atomiverse` target. No `internal/` headers, no friends, no reaching
   into private implementation. If a case cannot be written against the public
   API, that is not a reason to reach inside — **it is the defect the case just
   found.** Report it.

2. **Check the literature, not ourselves.** The expected value comes from a
   textbook or paper, and `REFERENCE.md` cites it (source, section/equation/page,
   the value, and the *physical* justification for the tolerance). Freezing our
   own current output as a "golden value" is forbidden — that regression-tests our
   bugs.

3. **As small as possible.** ≤ 60 s. This is a regression, not a production run.

   **Declare your thread width at registration.** `atv_add_regression(<domain>
   <case> [THREADS <n>])` defaults to **1**, and the aggregate target runs the
   binary under `OMP_NUM_THREADS=<n>`. Almost every case is a handful of atoms,
   where OpenMP fork/join costs more than the physics — on a 48-core node
   `mc_bias_es` takes 7 s at 1 thread, 231 s at 8, and does not finish at the
   node default, with bit-identical results at every count. Cases with the
   opposite shape (GPW/OT self-consistency on a real grid, DFPT response,
   multi-k plane-wave) declare `THREADS 8` next to the case, so the requirement
   is visible in one file instead of hidden in a `.cpp`. Raise a case only on a
   **measurement**, not a hunch, and never above 8: at 16 threads on a shared
   node the OpenMP barriers start losing to descheduling.

4. **Never resolve input data relative to the current directory.** The absolute
   data root is compiled in: `ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf"`. A case must
   give the same verdict from the build tree, the source tree, or a bare
   `./reg_<case>` under a debugger. No `../..` probe chains, no `getenv`
   escape hatch, and **no substituting a different model when an input is
   missing** — `reg_pw-smearing-al` used to fall back to a toy `V_loc` and print
   `PASS` against a pin recorded for the real pseudopotential. A missing input
   is a failure. Enforced by review (CLAUDE.md test iron law), not by a gate.

5. **Not in the default `ctest`.** Built behind `ATV_BUILD_REGRESSIONS=ON`, run
   through the `regressions` target.

6. **Non-zero exit on failure**, printing `got / want / tol`.

7. **One flat `main()` per case.** Code that runs exactly once is **not** carved
   into `setup_frame()` / `run_md()` / `check_energy()` helpers with one call site
   each. A reader must be able to read the case top to bottom. (CLAUDE.md: *"Extract
   on the second caller, not the first."*)

**Every delivered feature ships exactly one case here.** Not in `tests/`.

## Cases

| Domain | Case | What it pins | Reference |
|---|---|---|---|
| `md` | `lj_argon` | NVE energy conservation of the LJ argon fluid | Rahman (1964); Frenkel & Smit §4.3 |
| `md` | `pe_bonded_nve` | Bonded-only PE UA-chain NVE energy conservation | Frenkel & Smit §4.3; Tschopp 2010 (OPLS-AA UA) |
| `md` | `dihedral_charmm` | CHARMM periodic torsion \(V=K(1+\cos(n\phi-\delta))\) at \(\phi=\pi/2\) | MacKerell et al. (1998); LAMMPS `dihedral_style charmm` |
| `md` | `improper_harmonic` | Harmonic improper \(V=K(\chi-\chi_0)^2\) at \(\chi=\pi/2\) | LAMMPS `improper_style harmonic` |
| `pw` | `h_force_fd` | Plane-wave FD ionic force vanishes for one H (symmetry) | Martin *Electronic Structure*; FD oracle |
| `pw` | `h_force_analytic` | Plane-wave analytic ionic force vanishes for one H (symmetry) | Martin; QE force_lc/ew; default evaluator path |
| `pw` | `h_stress_cubic` | Plane-wave cell stress off-diagonals vanish for cubic H | strain-FD of SCF energy; cubic symmetry |
| `semiempirical` | `rm1_water_energy` | RM1 water converges and differs from AM1 | Rocha et al. JCC 27, 1101 (2006) |
| `scf` | `h2o_rhf_sto3g` | RHF/STO-3G water total energy via the driver-ized `cpu::scf::RHF` | LibreTexts PChem (Peverati) §28.2; Hehre/Stewart/Pople JCP 51, 2657 (1969) |
