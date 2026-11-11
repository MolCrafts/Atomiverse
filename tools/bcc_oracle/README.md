# AM1-BCC / ABCG2 offline oracle pins

Offline references for `cpu::semiempirical::AM1BCCCharges`, the GAFF (AM1-BCC)
and GAFF2 (ABCG2) charge model. The oracle is **AmberTools antechamber**. It
computes the AM1 base charges with **sqm**, then averages them over equivalent
atoms and applies the bond charge increments. Never launch either binary from
ctest. The regression hard-codes the numbers captured here.

```text
oracle: ambertools
oracle_version: AmberTools 26.1 (antechamber; SQM version 19)
```

This family covers the *charge* quantity only. NDDO energies and heats of
formation stay locked to OpenMOPAC (`tools/tb_oracle/README.md`).

## Re-running

```bash
python3 tools/bcc_oracle/make_inputs.py tools/bcc_oracle/inputs   # geometries
tools/bcc_oracle/run_antechamber.sh                                # pins → outputs/
```

The AmberTools Lmod module needs an easybuild toolchain stack that does not
load on the login node, so the script calls the install prefix directly:
`/software/sse2/el9_epyc9005/easybuild/prefix/software/AmberTools/26.1-gmpich-2026.03`.
Set `AMBERHOME` to use a different install.

Each case runs

```bash
antechamber -i inputs/<mol>.mol2 -fi mol2 -o <mol>_<m>.ac -fo ac -c <m> -nc <q> -pf y \
            -ek "qm_theory='AM1', maxcyc=0, scfconv=1.d-10"
```

with `<m>` ∈ {`bcc`, `abcg2`}:

- `maxcyc=0` means sqm runs one SCF at the input geometry and does not optimise.
- antechamber's other defaults are left alone: equivalencing `-eq 1`, and full
  bond-type perception for the BCC tables.

## Inputs

`make_inputs.py` writes idealised textbook geometries (standard bond lengths
and angles), not optimised by any code:

| Molecule | Net charge | What it exercises |
| -------- | ---------- | ----------------- |
| phenol (C₆H₅OH, 13 atoms) | 0 | aromatic bond types 7/8, ortho/meta equivalence classes |
| acetate (CH₃COO⁻, 7 atoms) | −1 | charged path, equivalent carboxylate O and methyl H |

Coordinates are rounded to 1e-3 Å. antechamber stores coordinates at that
precision before it writes `sqm.in`, so with pre-rounded inputs both codes see
bit-identical geometry. You can check this against the `sqm.in` captured in
each output directory.

## Pinned quantities

| File | Quantity | Print precision |
| ---- | -------- | --------------- |
| `outputs/<mol>_<m>/<mol>_<m>.ac`, `ATOM` column 9 | final AM1-BCC / ABCG2 charge [e] | 1e-6 (values are on a 1e-4 grid except where averaged) |
| `outputs/<mol>_bcc/sqm.out`, `Mulliken Charge` table | AM1 base charge [e] | **1e-3** |

sqm prints its Mulliken charges to 3 decimals, and that rounded table is what
antechamber reads into the BCC stage. Each oracle charge therefore carries up
to ±5e-4 e of rounding from its AM1 input. This rounding is the floor for any
comparison against a full-precision AM1. See
`regressions/semiempirical/am1-bcc/REFERENCE.md`.
