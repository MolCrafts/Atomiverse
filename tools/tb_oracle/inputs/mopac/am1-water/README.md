# OpenMOPAC offline twin — AM1 water

First entry in `tools/tb_oracle/inputs/mopac/`. Backs
`regressions/semiempirical/am1-water-mopac-energy/`.

```text
oracle: mopac
oracle_version: MOPAC 7.00 (public domain), archived transcript
```

**Offline twin — never executed in ctest.** `mopac` is not installed on the
Atomiverse build hosts and no regression may shell out to an oracle binary
(`tools/tb_oracle/README.md`). The regression hard-codes the numbers below as
transcribed constants; these files exist so the pin is auditable and
reproducible by hand on a host that *does* have MOPAC.

## Files

| File | What it is |
| ---- | ---------- |
| `mopac_am1_water.mop` | The input deck. **Reconstructed** from the geometry table echoed in the transcript — the archive publishes the `.out`, not the `.mop`. Keywords and Z-matrix values are verbatim from that echo; only the numeric optimisation flags (`1`/`0`) are inferred from the printed `*` markers. |
| `mopac_am1_water.out.txt` | The transcript, **verbatim**. Explanatory prose between output blocks removed; no numeric line altered. |

Source: <https://www.cup.uni-muenchen.de/ch/compchem/energy/MOPAC_output.html>
(H. Zipse, LMU München computational-chemistry course archive, page dated
2004-11-29). AM1 H and O parameters are Dewar, Zoebisch, Healy & Stewart,
*J. Am. Chem. Soc.* **107**, 3902 (1985) — the citations MOPAC prints for this
very run.

## Pinned quantities

| Quantity | Value | Unit | Geometry |
| -------- | ----- | ---- | -------- |
| `FINAL HEAT OF FORMATION` | −59.24079 | kcal/mol | AM1 optimum |
| `TOTAL ENERGY` | −348.56254 | eV | AM1 optimum |
| `ELECTRONIC ENERGY` | −493.26959 | eV | AM1 optimum |
| `CORE-CORE REPULSION` | +144.70704 | eV | AM1 optimum |
| `NET ATOMIC CHARGES` O / H | −0.3827 / +0.1914 | e | AM1 optimum |
| `IONIZATION POTENTIAL` (Koopmans) | 12.46417 | eV | AM1 optimum |
| `CYCLE 1 HEAT` | −57.64871 | kcal/mol | input (1.0 Å, 105°) |

Geometries (C2v, O at origin, H1 on +x, H2 in the xy plane):

| Structure | r(O–H) [Å] | ∠HOH [°] | r(H–H) [Å] |
| --------- | ---------- | -------- | ---------- |
| input | 1.000000 | 105.00000 | 1.586707 |
| AM1 optimum | 0.961267 | 103.53103 | 1.510121 |

## Reproducing on a MOPAC host

```sh
mopac mopac_am1_water.mop      # writes mopac_am1_water.out
grep "FINAL HEAT OF FORMATION" mopac_am1_water.out
```

A modern OpenMOPAC will print a slightly different last digit than the archived
7.00 run (updated physical constants, different optimiser defaults); the
regression's tolerances are sized for that — see the parent `REFERENCE.md`
§ *Tolerance rationale*.
