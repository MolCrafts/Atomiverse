/// @file cpu/planewave/internal/pw_forces.h
/// @brief Analytic ionic forces for plane-wave DFT (local HF + Ewald + KB).
///
/// After one SCF, F_I = F_I^loc + F_I^nl + F_I^Ewald. Plane waves are
/// cell-fixed (no AO Pulay). See `pw-breadth-01-analytic-forces` spec.
#pragma once

#include "atomiverse/cpu/planewave/internal/pw_setup.h"

#include <array>
#include <vector>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

struct FinalBandState;
class EwaldSummation;

/// Forces from one completed unstrained final-band state; no SCF or band solve.
/// The state and optional ionic terms must belong to this exact geometry.
/// Inputs are immutable; only local nonlocal-band contributions are MPI summed.
std::vector<std::array<double, 3>> pw_forces_from_bands(
    const cpu::Frame& frame, const PWIons& ions,
    const GVectorList& density_modes, std::span<const GVectorList> modes,
    std::span<const KBProjectors> projectors, const FinalBandState& bands,
    const EwaldSummation* ionic = nullptr);

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
