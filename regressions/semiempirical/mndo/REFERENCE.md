# MNDO: neutral closed-shell CHNO molecules

The external oracle is the unmodified OpenMOPAC 23.2.5 release, executed on
three fixed structures: distorted water, tilted CO and nonplanar ammonia.
Exact inputs, raw outputs, extracted pins, source/binary hashes and independent
gradient qualification are archived in
[`mndo-pm3-23.2.5`](../../../tools/tb_oracle/inputs/mopac/mndo-pm3-23.2.5/README.md).
No oracle executable is needed during the regression.

Acceptance uses total energy `1e-6 eV`, HOF `1e-6 kcal/mol`, charges `1e-7 e`
and every force component `1e-5 eV/Å`. These budgets are fixed from the archive:
total-energy print uncertainty is at most `2.1682e-7 eV`; the native MOPAC
numerical gradients were independently qualified by refined re-SCF energy
differences. The largest native-gradient difference over all six archived
MNDO/PM3 structures is `4.69e-7 eV/Å`. This is an independent numerical-gradient
oracle, not an analytic-gradient proof. Atomiverse uses analytic integral
derivatives. No tolerance is fitted to an Atomiverse discrepancy.

Both backends consume the same parameter source and modern physical constants.
The case also exercises force-preserving energy evaluation, additive Evaluate,
SCF failure/retry, accepted electronic history, and (when enabled) short MD and
actual Minimize line-search acceptance/rejection. Those internal comparisons
supplement the external scientific pins; they do not replace them or claim
long-run energy conservation, minimization convergence, or performance.

Scope remains CHNO, isolated molecules, neutral closed shells. Unsupported
method identities/elements, charge and multiplicity are rejected. No evidence
here extends to heavy-element pair corrections, open shells, periodicity or
solvent. Backend completion is recorded only after actual runs in the roadmap.
