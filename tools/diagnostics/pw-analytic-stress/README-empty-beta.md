# Fixed empty-beta LSDA tangent

New public admission is restricted to actual Gamma, LSDA, fixed occupations, Na>0/Nb=0. The scientific fixture is stretched tilted H2 triplet (2 angstrom, skew 6.2/6.0/5.8 angstrom cell), GTH-PBE-q1, 6 Ry, native base FFT 22x20x20 and LSDA quadrature 34x30x30. It retains midpoint inversion symmetry. It does not qualify arbitrary raw-spin variations, GGA, other k meshes, Gaussian, or empty alpha.

The independent QE 7.6 input/output/XML, PP, exit status, binary/module/host provenance and exact hashes are in empty-beta-rawrefs/manifest.json. export_empty_beta_pins.py --check verifies full force/stress XML against printed total rows, actual convergence, input/grid/model and exact C++ literal tokens. Eight failed historical controls remain explicitly false, including exit-zero nonconverged QE runs and native/QE energy mismatches. No control becomes a pin.

Opt-in targets (not CTest):

- pw-empty-beta --mpi [--rotate-z90]: default standalone stress plus real nested combined force/stress; per-rank complete E/full6F/6sigma, trace counts, metadata/output/context assertions. Two SCFs total (one per independent public invocation), each one active alpha final sweep, beta canonical 0xNg block.
- pw-empty-beta-component [--mpi]: zero SCFs, LSDA/PBE prepared-state components, nine collective malformed-input rejections, synthetic false reference retained, signed zero accepted. PBE component success is not external GGA qualification.
- pw-empty-beta-zero-paths [--mpi]: zero SCFs, nonlocal Si null-data zero-band T/NL and legal empty-beta force paths, extra-zero-f alpha gap-band shape retains canonical zero-row beta.
- pw-empty-beta-tangent: serial only, one SCF plus default active sweep, all 9 affine lanes x two h (3e-5/1e-5); 36 independent strained fixed-C/f primal evaluations with zero added SCF/band solves. Original 1e-8 Ha + 1e-7 relative W gate, global z90 primal rotation and immutable cache assertions.

Link dependencies: normal atomiverse; empty public entry uses shared general_trace.cpp and existing five GNU-wrap symbols. Component and zero paths wrap Davidson only; tangent wraps Davidson/extract/assembly/legacy stress. register-empty-beta.cmake is an optional include after register.cmake defines the shared wrap list.

ABI closure: FinalBandState gains optional typed fixed-spin metadata. Recompile pw_final_state.cpp, pw_solver.cpp, pw_forces.cpp and frozen_functional_virial.cpp together, plus every caller using this layout. pw_forces.cpp has no source changes but its object must be rebuilt. Diagnostic Recipe gains two trailing fields (Geometry, rotate_z90); SharedBands also grows. Recompile every old/new diagnostic main.cpp and general_trace.cpp together; old main.o must never be linked to new trace.o. Existing four-field Recipe aggregate/default semantics and FD/Gaussian/fixed protocols are retained. No production SCF loop, numerical threshold or point law changes.

Fixed reference PWResult optional Ne remains 0; actual sum(wf), rho G0, typed Na/Nb and auxiliary Ne/M/S0 are audited independently. Reference-result markers (including synthetic false markers in component tests) are not mathematical occupancy truth.
