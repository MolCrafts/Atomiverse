# MP1 Gamma PBE UKS diagnostic

This opt-in diagnostic qualifies one matched H3 branch with PBE, Gamma, two initially populated spin channels, MP order 1 and width 0.04 Ry. It calls the public standalone and nested methods; it contains no SCF or band iteration implementation. It uses the shared Recipe, ensemble, fixture, parent/context audit and five GNU link wrappers.

The three-H geometry, skew cell, GTH-PBE-q1 potential, 6 Ry cutoff, 48×42×42 PBE quadrature, shared chemical potential, extra three bands, mixing 0.35, 250-step budget and native tolerance 1e-9 are unchanged. Both spin channels have five bands. The actual QE7.6 reference uses the same model, grid and filling. The energy gate is 1e-5 Ha, every force component 1e-4 Ha/Bohr, and all six stress components 2e-3 eV/Å³. Auxiliary completion uses the existing advertised tolerance times 0.1, with no budget increase.

## Build and run

Root registration should add opt-in target `pw-mp-ensemble` from `mp_ensemble.cpp` and `general_trace.cpp`, using the existing `_pw_final_state_wraps` list and library dependencies. Do not add a CTest SCF. Commands are `pw-mp-ensemble`, `pw-mp-ensemble 0`, and `pw-mp-ensemble --mpi`; MPI2 is one owner rank and one empty rank. The thin entry uses the existing CLI implementation.

`SharedBands` adds MP observation fields at its tail. **Recompile all source consumers of general_trace.h and Recipe, including all existing standalone diagnostic main objects, before linking the new trace object.** Neither the production FinalBandState layout nor solver ABI changes in this package. `Recipe::mp()` adds methods only; no Recipe data field is added. Unchanged fixture/audit/pin copies in the private package exist solely to compile it in isolation and are excluded from copy-list.json.

Each extract clears prior pointers, payload, force observations and MP context. `trace::reset()` clears the entire aggregate and counters. Assembly observes data only after real production admission succeeds. Its existing MPI filling SUM also checks uniform MP branch selection; an MP-only MIN reduction computes finite global occupation extrema even on an empty rank. Density reconstruction is observation only and is labeled `PW_DENSITY_DOMAIN`; it does not clamp, project, or replace the production input.

## Independent reference and protocol

Run `python3 export_mp_pins.py --check` after installation. Private validation additionally uses `--raw-root PATH`. The default reference root is `results/2026-10-28-mp-gamma-pbe/private-candidate`, root's byte-preserved 233-file capture (ready SHA256 b4e68a9f3d9a450bb68f8e2ab84259d4788c24c38d9a14dbaf309bf48fe22b33). The exporter verifies actual QE input/stdout/XML/potential/binary-record hashes against this capture, convergence, model/filling/grid, signed occupations, particle count/spin, units and printed totals. It reproduces exact C++ literal tokens; whitespace is immaterial. No old 72³ grid reference enters these pins.

Every rank emits `PASS PW_MP h3-gamma-uks-pbe-mp1-004 ...`, including actual SCF iterations, local Davidson count, extract/assembly counts, zero legacy calls, single energy/virial publication, preserved reference metadata and `same_C_f=1`. `MP_FORCE` gives all nine force comparisons; `MP_STRESS` gives all six stress comparisons. `MP_STATE` reports independent signed Σwf, density G0 counts, auxiliary Ne/M/S/mu, actual local completion residual, target, order and width. `MP_REFERENCE`, `MP_BAND` and `MP_AUX` retain the accepted reference, signed occupations, internal functional energy and auxiliary free energy E−width*S. Entropy contributes to energy, not an extra frozen-f stress tensor.

Negative dN/dmu is qualified by the immutable private `spectrum-stationarity-1` thirteen-spectrum actual-QE replay, not inferred from this leaf's markers. The new leaf does not print or claim a slope measurement.

## Scope and retained failures

This H3 branch has negative occupations, occupations above one, negative generalized entropy, and 38 negative alpha-density/raw-|zeta|>1 quadrature points. Its passing stress includes the existing clipped point-law extension; it is not a positive-density-only qualification. The independent prepared 9-lane/two-step primal checks, actual scalar stationarity replay, and twelve fixed-basis full-SCF strain samples are in the immutable private capture. They are additional oracle calculations, not part of the claimed one-SCF/one-allspin-final-sweep analytic call.

The original LSDA compressed-strain MP nonconvergence, previous 72³ quadrature controls and all raw-domain limitations remain failed or unqualified. There is no qualification for RKS, other XC models, non-Gamma k points, other MP orders, empty spin, DFPT or arbitrary thermodynamic branches. Gaussian and FD retain their existing CLI, pins, gates and filling policy. No point-law, SCF loop, iteration budget or production hook is supplied by this diagnostic package.
