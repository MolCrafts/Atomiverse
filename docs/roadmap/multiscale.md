# Multiscale Methods and Properties

Tutorial: [Environment](../tutorial/molecular-environment.md) · [charges](../tutorial/ch20-hf-properties-environment.md).

Charges, continuum solvent, QM/MM, embedding, relativity, and intermolecular
corrections. Legend: [Roadmap index](index.md). Trajectory observers:
[Observables](observables.md).

## 1. Atomic charges

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Mulliken | ✓ | ✓ | |
| Löwdin | ✓ | ○ | |
| Merz–Kollman / CHELPG | ✓ | ○ | |
| AM1-BCC (GAFF) / ABCG2 (GAFF2) | ✓ | ○ | `AM1BCCCharges`; ATV AM1 + molpy BCC typing; matches AmberTools antechamber at fixed geometry |
| NPA / Hirshfeld / RESP / Gasteiger | ○ | ○ | |
| QEq (population analysis) | ○ | ○ | Classical QEq MD: [Force Fields](force-fields.md) |

## 2. Implicit solvent

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| C-PCM / COSMO / IEF-PCM | ✓ | ○ | `WithSolvent`; energy in the Fock build |
| C-PCM free-energy forces | ✓ | ○ | |
| SMD / GB / PB / nonequilibrium PCM | ○ | ○ | |

## 3. QM/MM

Compose under one outer Driver: MM evaluator + `Qmmm*.set_engine(QM…)`.
QM region = `kAtomQmZ` mask; link cuts derived from mask × bonds.

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Mechanical embedding | ◐ | ○ | `QmmmMechanical`; no external pin, self-pinned case removed |
| Electrostatic embedding | ✓ | ○ | `QmmmElectrostatic` |
| Polarizable embedding | ✓ | ○ | `QmmmPolarizable` |
| Two-layer ONIOM (subtractive) | ✓ | ○ | `QmmmOniom` |
| Link atoms | ✓ | ○ | |
| QM/MM MD | ✓ | ○ | Same composition as AIMD |
| QM/MM geometry opt | ◐ | ○ | Same composition as Minimize; no external pin, self-pinned case removed |
| 3-layer ONIOM / periodic / PME QM/MM | ○ | ○ | |
| Excited-state QM/MM | ○ | ○ | |

## 4. Embedding and relativity

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Fixed point-charge embedding | ✓ | ○ | + analytic forces |
| Uniform external E-field | ✓ | ○ | + analytic forces |
| ECP | ✓ | ○ | Semi-numerical forces |
| x2c-1e | ✓ | ○ | Analytic nuclear forces. No device path |
| DKH / ZORA / SOC / 4c | ○ | ○ | |
| FDE / DMET / PBE embedding | ○ | ○ | |

## 5. Intermolecular

| Capability | CPU | CUDA | Notes |
| ---------- | :-: | :--: | ----- |
| Counterpoise correction | ✓ | ○ | |
| SAPT / EDA families | ○ | ○ | |
