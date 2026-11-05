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
///
/// The two-center pair loop runs on the OpenMP threads (thread-private Fock
/// accumulators summed once per build); the builder retains those
/// accumulators and, optionally, every two-center block of one geometry, so
/// an SCF should keep one builder across iterations.
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
                   const std::vector<double>& spin_density);

    /// @brief Two-center Fock with caller-hoisted coords [Å] (SCF / force loops).
    std::vector<double>
    add_two_center(const NDDOSystem& system, const StorageReal* sp_x,
                   const StorageReal* sp_y, const StorageReal* sp_z,
                   const std::vector<double>& one_center_fock,
                   const std::vector<double>& total_density,
                   const std::vector<double>& spin_density);

    std::vector<double> build(const NDDOSystem& system, const cpu::Frame& frame,
                              const std::vector<double>& packed_hcore,
                              const std::vector<double>& total_density,
                              const std::vector<double>& spin_density);

    /// @brief Full Fock with caller-hoisted coords [Å].
    std::vector<double> build(const NDDOSystem& system, const StorageReal* sp_x,
                              const StorageReal* sp_y, const StorageReal* sp_z,
                              const std::vector<double>& packed_hcore,
                              const std::vector<double>& total_density,
                              const std::vector<double>& spin_density);

    /// @brief Compute and keep every two-center block at these coordinates
    ///        (OpenMP over atom pairs; each block lands in its own slot).
    /// @return false, keeping no cache, when the blocks exceed the memory
    ///         budget; `build_cached` must then not be used.
    bool cache_two_center(const NDDOSystem& system, const StorageReal* sp_x,
                          const StorageReal* sp_y, const StorageReal* sp_z);
    /// @brief Drop the cached blocks.
    void release_two_center() noexcept;
    /// @brief Whether `cache_two_center` holds blocks for this system size.
    bool has_two_center_cache(const NDDOSystem& system) const noexcept {
        return cached_atoms_ == static_cast<int>(system.atoms.size());
    }
    /// @brief Full Fock from the cached two-center blocks of the geometry
    ///        last passed to `cache_two_center`.
    /// @throws std::logic_error without a cache for this system.
    std::vector<double> build_cached(const NDDOSystem& system,
                                     const std::vector<double>& packed_hcore,
                                     const std::vector<double>& total_density,
                                     const std::vector<double>& spin_density);

  private:
    template <typename BlockFn>
    void accumulate_pairs_(const NDDOSystem& system,
                           const std::vector<double>& total_density,
                           const std::vector<double>& spin_density,
                           std::vector<double>& fock, BlockFn&& block);

    std::vector<std::vector<double>> thread_fock_;
    std::vector<double> pair_values_;      ///< Two-center blocks, pair-major.
    std::vector<std::size_t> pair_offset_; ///< Start of pair (i,j), i > j.
    int cached_atoms_ = -1;
};

/// @brief Port of NDDO reference `helect`: electronic energy in eV.
double electronic_energy_ev(int n_orbitals, const std::vector<double>& density,
                            const std::vector<double>& hcore,
                            const std::vector<double>& fock);

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
