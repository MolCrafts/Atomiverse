# Spec — gpw-bomd-smearing

## Summary

Multi-k Born–Oppenheimer MD under **metallic occupation smearing** (Fermi–Dirac
/ Methfessel–Paxton): free-energy / Mermin trajectory with fractional
occupations carried across BOMD steps (warm-start \(P_k\) consistent with
smear).

## Domain basis

- Product energy under smear: `reg_gpw-metallic-smearing` ✓
- Multi-k RKS BOMD (integer occ): `reg_gpw-bomd-kpoints` ✓
- Free-energy forces under smear: `reg_gpw-metallic-forces` ✓
- Residual: AIMD/BOMD path still assumes fixed integer fillings when advancing
  ionic steps under smear (no dedicated free-energy BOMD pin).

## Design

1. BOMD / `GPWEvaluator` multi-k path respects `Smearing` + \(kT\): each ionic
   step uses free-energy SCF (`F = E_{\mathrm{KS}}-TS`) and stores fractional
   `f_k` / `P_k` for the next warm start (same payload rules as metallic forces).
2. Public surface: members on evaluator / `GpwDft` only (no `*Config`, no factories).
3. Frame I/O for MD; electronic workspace private.
4. Offline twin: CP2K 2026.1 Quickstep AIMD + `SMEAR` + k-mesh, GTH-GPW class
   (matched algorithm — not NC-UPF PW).

## Files (expected)

| Path | Role |
| --- | --- |
| `src/cpu/gpw/gpw_evaluator.cpp` / BOMD warm-start | carry fractional \(P_k\) under smear |
| `regressions/gpw/gpw-bomd-smearing/` | dual-gate + REFERENCE |
| `tools/cp2k_oracle/inputs/gpw-bomd-smearing/` | CP2K `.inp` twin |

## Tasks

1. Spec acceptance dual-gate + fair wall criteria (this package).
2. Implement warm-start / force consistency under smear if still missing.
3. Offline CP2K twin run → pins in REFERENCE.md (binary, version, input, key lines).
4. Regression: short NVE-ish trajectory envelope + energy/force consistency.
5. Fair wall \(T_{\mathrm{atv}}/T_{\mathrm{cp2k}} \le 10\) on matched single-step or short traj twin.

## Testing

- Unit: only payload/warm-start helpers if extracted (no e2e in `tests/`).
- Regression: `reg_gpw-bomd-smearing` ≤ 60 s, literature/CP2K cited pins.

## Out of scope

- Variable-cell BOMD under smear; CPMD fictitious-mass dynamics; NC-UPF PW twins.
