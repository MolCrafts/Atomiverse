# SCF convergence layer — initial guesses + far-field mixers (H2O/def2-SVP)

End-to-end case for `hf-capability-gaps-07-guess`:

1. **Guess quality** (H2O/def2-SVP RHF, equilibrium geometry): iteration
   counts must order `Hueckel < Hcore` and `SAP < Hcore`, and every guess
   must converge to the pinned RHF energy.
2. **Far-field mixer contrast** (2.6×-stretched H2O/def2-SVP UHF): plain
   CDIIS stalls within a 60-iteration budget on the *unstable*
   spin-symmetric solution, while `ADIIS→CDIIS` converges to the pinned
   broken-symmetry UHF minimum within 1e-6 Ha.

## Literature

- S. Lehtola, *Assessment of Initial Guesses for Self-Consistent Field
  Calculations. Superposition of Atomic Potentials: Simple yet Efficient*,
  J. Chem. Theory Comput. **15**, 1593 (2019).
  DOI [10.1021/acs.jctc.8b01089](https://doi.org/10.1021/acs.jctc.8b01089)
  — extended-Hückel and SAP initial guesses.
- K. N. Kudin, G. E. Scuseria, E. Cancès, *A black-box self-consistent
  field convergence algorithm: One step closer*, J. Chem. Phys. **116**,
  8255 (2002). DOI
  [10.1063/1.1470195](https://doi.org/10.1063/1.1470195) — EDIIS.
- X. Hu, W. Yang, *Accelerating self-consistent field convergence with the
  augmented Roothaan–Hall energy function*, J. Chem. Phys. **132**, 054109
  (2010). DOI [10.1063/1.3304922](https://doi.org/10.1063/1.3304922)
  — ADIIS.

## Oracle (PySCF 2.14.0, external — never a build/test dependency)

Interpreter: `~/work/.x86_64/bin/python` (pyscf 2.14.0, def2-SVP spherical).

### Part 1 — equilibrium H2O iteration counts + energy pin

```python
from pyscf import gto, scf
mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692",
            basis="def2-svp", verbose=0)
for ig in ["hcore", "minao", "huckel", "sap"]:
    mf = scf.RHF(mol)
    mf.init_guess = ig
    mf.conv_tol = 1e-9
    e = mf.kernel()
    print(f"init_guess={ig:7s} cycles={mf.cycles:3d} converged={mf.converged} E={e:.11f}")
mf = scf.RHF(mol); mf.conv_tol = 1e-12
print("RHF equilibrium E(conv_tol=1e-12) =", f"{mf.kernel():.11f}")
```

Output (2026-07-21):

```
init_guess=hcore   cycles= 10 converged=True E=-75.96098398708
init_guess=minao   cycles=  8 converged=True E=-75.96098398708
init_guess=huckel  cycles=  8 converged=True E=-75.96098398708
init_guess=sap     cycles=  8 converged=True E=-75.96098398708
RHF equilibrium E(conv_tol=1e-12) = -75.96098398708
```

Pinned: **E_RHF(eq) = −75.96098398708 Ha**; PySCF guess ordering
hcore(10) > minao(8) = huckel(8) = sap(8). (PySCF's `huckel` is Lehtola's
parameter-free on-the-fly variant; Atomiverse implements the static-VSIP
extended Hückel from the same paper, so only the *ordering vs. hcore* is
comparable, not the exact count.)

### Part 2 — stretched-H2O UHF broken-symmetry pin

Geometry = regression geometry at stretch 2.6:
`hx = 0.7572·2.6 = 1.96872`, `hz = 0.1173 − 0.5865·2.6 = −1.40760` Å.

```python
from pyscf import gto, scf
mol = gto.M(atom="O 0 0 0.1173; H 0 1.96872 -1.40760; H 0 -1.96872 -1.40760",
            basis="def2-svp", spin=0, verbose=0)
mf = scf.UHF(mol)
mf.conv_tol = 1e-12
mf.max_cycle = 500
mf.kernel()
print("first solution:", f"{mf.e_tot:.11f}", mf.converged)
for it in range(8):
    mo1 = mf.stability()[0]
    if mo1 is mf.mo_coeff:
        print("stable after", it, "rotations"); break
    mf.kernel(dm0=mf.make_rdm1(mo1, mf.mo_occ))
print("final UHF E =", f"{mf.e_tot:.11f}", "converged:", mf.converged)
print("<S^2> =", mf.spin_square()[0])
```

Output (2026-07-21):

```
first solution: -75.35728186304 True
  restarted -> -75.71896632261 True
stable after 1 rotations
final UHF E = -75.71896632261 converged: True
<S^2> = 1.9709626405956904
```

Pinned: **E_UHF(broken symmetry) = −75.71896632261 Ha** (⟨S²⟩ ≈ 1.97).
Note that PySCF's own default-guess UHF also lands first on the *unstable*
spin-symmetric solution (−75.35728186304) — the very state the plain-CDIIS
engine run oscillates around without converging, which is what makes this a
genuine far-from-convergence case rather than an engine artifact.

## Engine evidence (Atomiverse, this machine, 2026-07-21)

```
H2O/def2-SVP RHF (tol=1e-9), iterations per guess:
  hcore    iters= 16   minao iters= 16   hueckel iters= 14   sap iters= 14
2.6x-stretched H2O/def2-SVP UHF (tol=1e-9, hcore guess, damping 0,
diis_start_iter 1, budget 60):
  CDIIS alone : 60 iterations, NOT converged, E ~ -75.357 (symmetric state)
  ADIIS->CDIIS: 38 iterations, converged, E = -75.71896632 Ha
```

## Acceptance

- Every equilibrium guess converges to −75.96098398708 Ha within 1e-7.
- `iters(Hueckel) < iters(Hcore)` and `iters(SAP) < iters(Hcore)`.
- Stretched UHF: CDIIS-alone **not** converged in 60 iterations;
  ADIIS→CDIIS converged with |E − (−75.71896632261)| ≤ 1e-6 Ha.
- Runtime ≤ 60 s, single process.
