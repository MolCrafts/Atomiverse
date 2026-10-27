# AmberTools GAFF2 DMSO atom-order dependence

These raw files preserve an independently reproduced AmberTools 26.1 reference
anomaly found while qualifying the GAFF parameterization benchmark. They are
not timed samples and do not substitute for a rebuilt-native campaign.

GAFF2 explicitly declares `h1-c3-s4-c3` with zero torsional coefficient and
periodicity one. Its generic `X-c3-s4-X` row has `PK/IDIVF = 1.2/6 = 0.2`
kcal/mol and periodicity three. The source table rows and SHA-256, program
hashes, canonical atom permutation and raw-artifact hashes are in
[`evidence.json`](evidence.json).

In `original/input.mol2`, DMSO's methyl carbons are atom rows 0 and 3 (indices
here are zero-based). The native exact-row interpretation assigns the zero
coefficient to all six C–S–C–H quartets and the generic 0.2 coefficient to all
six O–S–C–H quartets. The raw `original/ligand.prmtop` instead adds a further
0.2, periodicity-three component on only three C–S–C–H quartets:
`[3,1,0,4]`, `[3,1,0,5]`, `[3,1,0,6]`. The other methyl's corresponding
quartets retain only the zero coefficient. `ligand.frcmod` is empty.

The `permuted` case swaps only the two methyl carbon atom rows, remaps their
bond indices and preserves coordinates and chemical connectivity. After
mapping the output indices back to the original physical atoms, the extra
three components move to `[0,1,3,7]`, `[0,1,3,8]`, `[0,1,3,9]`. The six
O–S–C–H terms remain unchanged. Thus the difference is in tleap's actual
numbering-dependent parameter assembly, not the prmtop reader or a different
molecular geometry. The resulting force field breaks the chemical equivalence
of the two methyl groups.

The benchmark keeps this numerical mismatch visible for GAFF2 DMSO under both
charge corrections and suppresses its speedup. It does not copy the asymmetric
reference behavior into the native force field or relax the parameter check.

Both directories retain exact input/output MOL2, fixed-geometry sqm input and
output, command logs, frcmod, tleap input and complete prmtop/inpcrd. Reproduce
in a fresh directory by copying its `input.mol2` and running the commands in
the method's AmberTools adapter: `antechamber -at gaff2 -c bcc -nc 0 -eq 1`
with `qm_theory='AM1', maxcyc=0, scfconv=1.d-10`, followed by `parmchk2 -s
gaff2` and tleap using the retained `build.leap` text. Set `OMP_NUM_THREADS=1`
and `OPENBLAS_NUM_THREADS=1`, and use the recorded AmberTools installation.
