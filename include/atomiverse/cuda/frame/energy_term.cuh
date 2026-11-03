/// @file cuda/frame/energy_term.cuh
/// @brief EnergyTerm -- the device energy sink of one force-field term: the
///        system total `SystemScalars::energy` plus the term's own field.
///
/// A CPU style writes its energy on one line into `kEnergy` and the term key
/// it belongs to (`kEnergyPair`, `kEnergyBond`, …), so `energy` equals the
/// sum of the terms. A CUDA force kernel takes one `EnergyTerm` by value and
/// adds each block's reduced energy to both device scalars, which keeps the
/// same per-term breakdown in `cuda::to_cpu` and every recorder. A step
/// without eflag (LAMMPS `ev_set`) passes an empty sink: both pointers null,
/// uniform across the grid, so the reduction is skipped.
#pragma once

#include <cuda_runtime.h>

#include "atomiverse/core/system_scalars.h"
#include "atomiverse/cuda/frame/types.cuh"

namespace ATV {
namespace cuda {

/// @brief Which `SystemScalars` per-term field a force-field style feeds
///        (LAMMPS thermo keywords `epair`, `ebond`, `eangle`, `edihed`,
///        `eimp`). Reciprocal-space and tail energies count as `Pair`, as on
///        CPU and in LAMMPS `epair`.
enum class EnergyKind { Pair, Bond, Angle, Dihedral, Improper };

/// @brief Device sink of one style's energy: total + term [eV].
struct EnergyTerm {
    acc_t* total = nullptr; ///< `SystemScalars::energy` [device], or null.
    acc_t* term = nullptr;  ///< The style's per-term field [device], or null.

    /// @brief Whether this step accumulates energy (eflag).
    __host__ __device__ explicit operator bool() const {
        return total != nullptr;
    }

    /// @brief Add one reduced contribution [eV] to the total and the term.
    ///        Call only when the sink is non-empty.
    __device__ void add(acc_t v) const {
        atomicAdd(total, v);
        atomicAdd(term, v);
    }
};

/// @brief The sink of `kind` on a device `SystemScalars`.
/// @param scalars   Device scalars of the Frame (address arithmetic only;
///                  never dereferenced on the host).
/// @param kind      The term the style feeds.
/// @param requested Whether this step evaluates energy; false gives an empty
///                  sink.
/// @return The total/term pointer pair, or an empty sink.
inline EnergyTerm energy_term(SystemScalars* scalars, EnergyKind kind,
                              bool requested) {
    if (!requested || scalars == nullptr)
        return {};
    acc_t* term = nullptr;
    switch (kind) {
    case EnergyKind::Pair:
        term = &scalars->energy_pair;
        break;
    case EnergyKind::Bond:
        term = &scalars->energy_bond;
        break;
    case EnergyKind::Angle:
        term = &scalars->energy_angle;
        break;
    case EnergyKind::Dihedral:
        term = &scalars->energy_dihedral;
        break;
    case EnergyKind::Improper:
        term = &scalars->energy_improper;
        break;
    }
    return {&scalars->energy, term};
}

} // namespace cuda
} // namespace ATV
