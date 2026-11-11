/// Closed-form orbital-functional PW virial assembly; no SCF or band solves.
#pragma once

#include "atomiverse/cpu/planewave/internal/frozen_operator_virial.h"
#include "atomiverse/cpu/xc/xc_functional.h"

namespace ATV::cpu::pw::internal {
struct FinalBandState;
struct PWIons;
struct PWDensityView;
class XCBuilder;
class EwaldSummation;

struct FrozenFunctionalVirial {
    FrozenOperatorVirial kinetic, hartree, local, nonlocal, xc, ionic;
    FrozenOperatorVirial total;
    // Maximum |W_ab-W_ba| before any symmetric physical-tensor projection.
    AccReal max_antisymmetric_ha = 0;
    AccReal density_mismatch_l2 = 0;
};

// Public slices: LSDA/PBE-family RKS/UKS fixed, Fermi or Gaussian filling on
// supported k meshes. Fixed filling means per-k band-index occupations; it
// does not assert a global insulating gap.
// Fixed LSDA at actual Gamma additionally admits declared Na>0, Nb=0 with
// canonical zero-row beta blocks and exactly zero beta densities. Other empty
// fixed/Gaussian channels retain the earlier public path.
// The explicit
// orbital_density_G in bands is used consistently with its C/f/w for every
// electronic term. The accepted problem, density, operators and result remain
// unchanged; total.energy_ha is an auxiliary internal energy, not a replacement
// for the reference free energy. An auxiliary free energy also needs -width*S.
// No derivative consumer needs to rerun SCF/bands.
//
// bands must be the collective extract_final_bands result for this exact problem.
// Each rank owns its round-robin k points for every spin. Ranks may be empty.
// A zero-row beta block requires the producer fixed-spin declaration; missing
// blocks, noncanonical occupations or nonzero empty-channel payloads reject.
// T/NL E+W and weighted electron count share one MPI SUM. The collective
// electron count must match the reference before replicated terms are built.
// Replicated H/local/XC/ionic terms are
// each evaluated once per rank and never summed across ranks. This function is
// collective when MPI is active, including synchronized admission failures.
//
// ionic is prepared for the same geometry and valence charges as problem;
// its energy and full real/reciprocal/self/background virial are consumed.
// xc_scratch is exclusive to this evaluation. Its prepare/step operate on a new
// workspace with an owned FFT; no accepted workspace or cache is borrowed.
FrozenFunctionalVirial assemble_frozen_functional_virial(
    const PWIons& ions, const GVectorList& density_modes,
    std::span<const GVectorList> modes, std::span<const AccReal> weights,
    PWDensityView density, const FinalBandState& bands,
    cpu::xc::XCType functional, const EwaldSummation& ionic,
    XCBuilder& xc_scratch);

} // namespace ATV::cpu::pw::internal
