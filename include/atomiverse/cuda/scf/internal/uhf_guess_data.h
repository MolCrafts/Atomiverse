/// @file cuda/scf/internal/uhf_guess_data.h
/// @brief Geometry-independent MINAO constants packed from the shared tables.
#pragma once

#include <vector>

namespace ATV::cuda::scf::internal {

/// Empty when any element lacks a MINAO table; UHF then starts from Hcore.
struct UhfMinaoData {
    UhfMinaoData(const int* atomic_numbers, int natm);
    std::vector<double> alpha, coefficient, occupation;
    std::vector<int> primitive_offset{0}, angular, atom, ao_shell, ao_cart;
};

} // namespace ATV::cuda::scf::internal
