# GPW accepted-state force residency

This opt-in diagnostic covers the Γ RKS single-grid SCF-to-force handoff.
Configure with `-DCMAKE_PROJECT_Atomiverse_INCLUDE=$repo/tools/diagnostics/gpw-residency/register.cmake`
and build `gpw-residency`. The hook retains the preceding Driver, PBE-family
and independent periodic-oracle targets. Complete solves remain outside CTest.

`main.cpp` keeps the archived Padé water geometry, basis, potentials, cutoff,
SCF settings and p-shell gates: `5e-5` Ha and `5e-4` Ha/Bohr. Four additional
fully asymmetric water cases exercise PBE/PBEsol/revPBE/RPBE, p-shell AO Pulay,
nonlocal operator-center derivatives and every Cartesian force. Each uses
two geometries on the same public method. A nonzero off-plane ionic energy
difference has a fixed `1e-4` Å step and `1e-3` eV/Å gate. The new asymmetric
cases use `tol=1e-11`; the archived Padé settings remain unchanged.

The native library preceding the handoff change is frozen separately and
linked to this same diagnostic, providing an actual before/after record.
These CPU/device twins and energy differences establish consistency within
their stated envelope; they are not independent-engine qualification.

For an actual transfer trace, set `ATV_GPW_PROFILE_FORCE=1` and run one case:

```sh
nsys profile --sample=none --cpuctxsw=none --trace=cuda,cusolver,cublas \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=force env ATV_GPW_PROFILE_FORCE=1 \
  build-gpu-job/bin/gpw-residency --case water-asym-PBE --profile-only
nsys export --type sqlite --output force.sqlite force.nsys-rep
python3 tools/diagnostics/gpw-residency/trace.py force.sqlite force-audit.json
```

The capture includes physical-H eigen/energy-weighted-density preparation
and force assembly. Geometry-table uploads precede it; final force/last-result
downloads follow it. `trace.py` inspects real CUPTI memcpy directions and
sizes, kernel activity and CUDA runtime statuses. D2D copies and four-byte
eigensolver status downloads are permitted. CUDA 13 also retrieves an
eight-byte matrix maximum for eigen scaling. Its allowance requires an
enclosing cuSOLVER API range on the same host thread and the executed
`lansy_M_stage2` producer; a separate CUPTI numerical probe verifies maxima
7 and 21, distinct from the matrices' other norms. This is a scalar exception,
not a byte-size exemption for arbitrary fields. H2D fields or other D2H
payloads inside the capture fail. Required kernel substrings can be supplied with
`--required-kernel`; source inspection must also verify the interval and
that each permitted scalar is a solver status. No self-reported transfer
counter substitutes for this trace.

This scope does not qualify UKS, multigrid, multi-k setup/payload, cold-start
science, or cell-stress device residency. Formal source/build qualification
remains false. Small water timings do not establish a GPU speedup.
