# Interrupted contended diagnostic

This partial campaign was stopped after 96 observations because every allowed
CPU was 93–99% busy with unrelated workloads. Only this campaign controller and
its worker session were interrupted. No unrelated job was changed.

`resource-selection.json` records the one-second per-core sample, selected
CPU 7 and affinity policy. `results.json` and raw jobs retain all completed
scientific observations and raw latencies. All derived speedup ratios are
explicitly null and are ineligible for a controlled performance claim.

This is not the final isolated campaign. Do not re-enable ratios by re-running
the generic reporter, which does not yet consume the contamination metadata.
