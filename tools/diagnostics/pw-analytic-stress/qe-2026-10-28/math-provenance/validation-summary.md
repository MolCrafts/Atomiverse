# Scratch closed-form PW tensor verification

Status: the limited Gamma RKS fixed-occupation LSDA ordinary-GTH Si2 slice
passes. No production file or public stress default was changed.

The accepted SCF is 23 iterations at tolerance 1e-10. A single six-band
unstrained auxiliary solve has gap 0.92604477eV and maximum H*C residual
5.23e-12Ha. The original accepted result and reference density are retained.
The auxiliary full functional differs from the accepted energy by 3.64e-14Ha.
All nine raw W entries are checked before symmetrizing; maximum antisymmetry is
2.98e-16Ha. Nonlocal energy is 1.66658467Ha and the ionic background is
-0.407636404Ha, so neither contribution is a zero-fixture escape.

| Reference tolerance | SCF iterations | rho_orb-reference L2 | Harris-minus-orbital D, Ha | max closed stress error vs QE, eV/A3 |
|---|---:|---:|---:|---:|
| 1e-6 | 8 | 4.94e-7 | -2.11e-11 | 3.79e-6 |
| 1e-8 | 16 | 2.47e-9 | 1.69e-14 | 2.26e-7 |
| 1e-10 | 23 | 3.23e-11 | 1.15e-14 | 2.24e-7 |

At the strictest tolerance the independent QE energy error is 1.055e-7Ha,
maximum full six-force-component error is 3.83e-7Ha/Bohr, and maximum physical
six-stress error is 2.24e-7eV/A3. All three original absolute gates remain
1e-5Ha / 1e-4Ha/Bohr / 2e-3eV/A3. The actual QE7.6 model is Slater+VWN5,
not the historical PW/PZ-labelled model. Its separate fixed-charge six-band
probe measures a 0.92604499eV occupied/empty gap.

The existing 12-NSCF stress differs from the closed tensor by approximately
1.294e-5eV/A3 in every reference-tolerance run. This weak dependence on density
residual is consistent with its fixed h=0.002 truncation floor; it is not
presented as a strict identity at finite residual. This old path is explicitly
counted as a separate numerical oracle, not a component of the closed tensor.

| Fixed-basis full-SCF strain h | max central error, eV/A3 | max five-point error, eV/A3 |
|---|---:|---:|
| 3e-5 | 1.09e-9 | 6.49e-10 |
| 1e-5 | 1.67e-9 | 2.05e-9 |
| 3e-6 | 6.40e-9 | 8.64e-9 |

The roundoff/SCF floor grows at the smallest h as expected; no h or comparison
gate was selected after seeing the numbers. Every lane and every +/-h,+/-2h
sample passes the unchanged 2e-3eV/A3 gate. All 72 fixed-basis samples have
exactly the same ordered density/wave Miller labels and FFT shapes as the
reference. An additional setup-only audit proves positional equality, not only
set membership, with zero SCF or band calls.

The full-SCF oracle uses the original compiled SCF loop object from the
unmodified library. A scratch-only GNU linker wrapper changes its setup binding
to the existing fixed-basis setup function. The loop, mixing, convergence and
energy statements are neither copied nor modified. The 72 full-SCF samples and
12 fresh-recut cutoff-crossing controls are separately counted oracle work.
Each physical closed-form evaluation remains one reference SCF, one unstrained
band sweep and zero strained band solves. Component/frozen-C/f energy finite
differences use no SCF or band solves.

The earlier fresh-recut qualification intentionally failed: h=3e-5 has a
five-point endpoint at 6e-5 that changes a density cutoff set. Its good numerical
agreement was not promoted to a fixed-subspace proof. `validation-2/receipt.json`
and its explanatory companion remain untouched. The successful explicit
fixed-basis receipt is `validation-3/receipt.json`. The original failed dimer
SCF is separately retained in the QE reference provenance.

The shared final-band extractor used by this binary is an immutable local
snapshot before the reviewer's later FFT-buffer reuse optimization. Its source,
header and exact compile/link command hashes are in the receipt; updated generic
helper results are not claimed by this binary. See `helper-source-provenance.json`
and `production-loop-snapshot/archive-manifest.json`.

This does not qualify radial gauge-rescaled admitted NC operators, raw-negative
XC domains, spin, smearing or general k-point total stress. Those are explicit
follow-up scopes; no original gate or public convention was weakened.
