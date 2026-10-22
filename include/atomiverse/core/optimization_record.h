/// @file core/optimization_record.h
/// @brief Backend-independent geometry-optimization diagnostic record.
#pragma once
#include "atomiverse/core/precision.h"
namespace ATV::minimizer {
/// @brief Per-step optimization record for diagnostics.
struct OptRecord {
    BigInt step = 0;      ///< Optimization step number.
    double energy = 0;    ///< Total energy [eV].
    double max_force = 0; ///< Maximum atomic force magnitude [eV/Angstrom].
    double rms_force = 0; ///< RMS of atomic force magnitudes [eV/Angstrom].
    double max_displacement =
        0; ///< Maximum displacement from previous step [Angstrom].
    double rms_displacement =
        0; ///< RMS displacement from previous step [Angstrom].
    double energy_change = 0; ///< Energy change from previous step [eV].
};

} // namespace ATV::minimizer
