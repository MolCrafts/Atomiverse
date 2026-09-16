# Offline twin — DFTB2 Γ-PBC lattice-sum D3(BJ)

Inputs that reproduce the pins in
`regressions/semiempirical/dftb2-dispersion-d3-periodic/REFERENCE.md`.

**Never executed by ctest.** Run by hand when re-deriving a pin.

```bash
export PATH=/software/sse2/el9_epyc9005/easybuild/prefix/software/Simple-DFTD3/1.2.1-gfbf-2025b/bin:$PATH
s-dftd3 --bj dftb3 dimer_pbc12.gen   # -1.6959579071183E-03 Eh  (lattice sum)
s-dftd3 --bj dftb3 dimer_mol.xyz     # -1.6892886213669E-03 Eh  (cluster)
```

| file | what |
| ---- | ---- |
| `dimer_pbc12.gen` | water dimer, DFTB+ `.gen` supercell format, cubic 12 Å cell |
| `dimer_mol.xyz` | same geometry, isolated cluster |

`s-dftd3` switches to a periodic evaluation automatically for a periodic input
format (`.gen` with an `S` header and lattice vectors); there is no `--periodic`
flag. It also exposes **no cutoff flag** — it truncates the two-body sum at its
compiled-in 60 Bohr default, which is why the regression sets
`DFTB2::d3_cutoff = 60 Bohr` to compare like for like. See the REFERENCE's
"The cutoff is part of the pinned quantity".

Family note: DFTB+ is the oracle for the *electronic* DFTB ladder. D3 is a
geometry-only additive term and simple-dftd3 is the engine DFTB+ itself calls,
so it is the correct pin here — this is not a substitution of oracle families.
