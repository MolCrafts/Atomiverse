/// @file cpu/gpw/internal/gpw_workspace.h
/// @brief Mutable GPW SCF potentials retained for accepted-state derivatives.
#pragma once

#include <vector>

#include "atomiverse/cpu/linalg/matrix.h"

namespace ATV::cpu::gpw::internal {

struct GpwWorkspace {
    std::vector<double> veff;   ///< Restricted fine-grid potential [Ha].
    std::vector<double> veff_a; ///< Collinear alpha potential [Ha].
    std::vector<double> veff_b; ///< Collinear beta potential [Ha].
    std::vector<double> v_es;   ///< Electrostatic potential [Ha].
    /// Final Bloch density (total for RKS, alpha for UKS).
    std::vector<linalg::CplxMatrix> accepted_P_k;
    std::vector<linalg::CplxMatrix> accepted_P_k_beta;
};

} // namespace ATV::cpu::gpw::internal
