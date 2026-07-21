/// @file cpu/semiempirical/nddo_fock.h
/// @brief NDDO reference-shaped one-center Fock and electronic energy helpers.
#pragma once

#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"

namespace ATV {
namespace cpu {
namespace semiempirical {

struct NDDOOneCenterIntegrals {
    int atomic_number = 0;
    int dimension = 0;
    std::vector<double> values;

    double at(int row, int col) const;
};

/// @brief Builds NDDO reference `wstore` one-center electron-repulsion matrices.
class NDDOOneCenterIntegralBuilder {
  public:
    NDDOOneCenterIntegrals build(const NDDOAtom& atom) const;
};

/// @brief Builds Fock matrices using NDDO reference `fock1` one-center accumulation.
class NDDOFockBuilder {
  public:
    std::vector<double>
    add_one_center(const NDDOSystem& system,
                   const std::vector<double>& packed_hcore,
                   const std::vector<double>& total_density,
                   const std::vector<double>& spin_density) const;

    std::vector<double>
    add_two_center(const NDDOSystem& system, const cpu::Frame& frame,
                   const std::vector<double>& one_center_fock,
                   const std::vector<double>& total_density,
                   const std::vector<double>& spin_density) const;

    /// @brief Two-center Fock with caller-hoisted coords [Å] (SCF / force loops).
    std::vector<double>
    add_two_center(const NDDOSystem& system, const StorageReal* sp_x,
                   const StorageReal* sp_y, const StorageReal* sp_z,
                   const std::vector<double>& one_center_fock,
                   const std::vector<double>& total_density,
                   const std::vector<double>& spin_density) const;

    std::vector<double> build(const NDDOSystem& system, const cpu::Frame& frame,
                              const std::vector<double>& packed_hcore,
                              const std::vector<double>& total_density,
                              const std::vector<double>& spin_density) const;

    /// @brief Full Fock with caller-hoisted coords [Å].
    std::vector<double> build(const NDDOSystem& system, const StorageReal* sp_x,
                              const StorageReal* sp_y, const StorageReal* sp_z,
                              const std::vector<double>& packed_hcore,
                              const std::vector<double>& total_density,
                              const std::vector<double>& spin_density) const;
};

/// @brief Port of NDDO reference `helect`: electronic energy in eV.
double electronic_energy_ev(int n_orbitals, const std::vector<double>& density,
                            const std::vector<double>& hcore,
                            const std::vector<double>& fock);

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
