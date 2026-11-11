# Historical QE Si input model correction

The archived `si_lsda.in` bytes are preserved. Its `SLA PW` input uses
Slater/PW92, and the comment identifying that model as current
`XCType::LSDA` is incorrect: the current CPU evaluator uses Slater/VWN5.
The associated 2026-08-09 fair-wall receipt therefore does not establish a
matched current-LSDA comparison. Do not silently edit or relabel the old input.
A new matched reference must use explicit `SLA+VWN+NOGX+NOGC`, retain its
actual converged raw output and verify the printed XC model.

See the [dated audit](../pw-analytic-stress-si/model-audit-2026-10-28.json)
and the [historical receipt erratum](../../../../regressions/pw/rks/REFERENCE.md).
