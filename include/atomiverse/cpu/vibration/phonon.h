/// @file cpu/vibration/phonon.h
/// @brief cpu::vib::Phonon — finite-displacement supercell phonon driver.
///
/// Crystal counterpart of `cpu::vib::Hessian`. Builds an optional supercell,
/// assembles Cartesian force constants by central differences, mass-weights,
/// and diagonalizes the full \(3N\) dynamical matrix at Γ.
///
/// @code{.cpp}
///   cpu::vib::Phonon phonon(/*na=*/1, /*nb=*/1, /*nc=*/1);
///   phonon.set_neighbor(cpu::md::NeighborList<>{rc, 0.3, 1, 0, true});
///   phonon.on(ATV::vib::Phase::force).add(cpu::StillingerWeber{pot});
///   phonon.build();
///   cpu::vib::PhononResult r = phonon.run(unit_cell);
/// @endcode
#pragma once

#include <memory>
#include <stdexcept>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/bias.h"
#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/path_coupling.h"
#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/component/semiempirical.h"
#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/vibration/displace_step.h"
#include "atomiverse/cpu/vibration/hessian_accumulator.h"
#include "atomiverse/cpu/vibration/hessian_workspace.h"
#include "atomiverse/cpu/vibration/mass_weighting.h"
#include "atomiverse/cpu/vibration/phonon_diagonalizer.h"
#include "atomiverse/cpu/vibration/phonon_result.h"
#include "atomiverse/cpu/vibration/reference_geometry.h"
#include "atomiverse/cpu/vibration/supercell.h"
#include "atomiverse/cpu/vibration/symmetrize.h"

namespace ATV {
namespace cpu {
namespace vib {

/// @brief Finite-displacement Γ-point phonon preset (supercell-aware).
///
/// Termination is a fixed \(6N_\mathrm{sc}\) displacement budget + one finalize
/// pass (same budget shape as `Hessian`). Parameters are constructor args and
/// public members — no `PhononConfig` bag.
class Phonon : public Driver<ATV::vib::Phase> {
  public:
    /// @brief Supercell repeats along lattice vector a (≥1).
    int na = 1;
    /// @brief Supercell repeats along lattice vector b (≥1).
    int nb = 1;
    /// @brief Supercell repeats along lattice vector c (≥1).
    int nc = 1;

    /// @brief Construct with 1×1×1 (phonons of the given cell at Γ).
    Phonon() : Phonon(1, 1, 1) {}

    /// @brief Construct with explicit supercell factors.
    /// @param na_ Repeats along a [dimensionless].
    /// @param nb_ Repeats along b [dimensionless].
    /// @param nc_ Repeats along c [dimensionless].
    Phonon(int na_, int nb_, int nc_) : na(na_), nb(nb_), nc(nc_) {
        if (na < 1 || nb < 1 || nc < 1)
            throw std::invalid_argument(
                "Phonon: supercell factors na,nb,nc must be ≥ 1");
        // Structural FD chain (no molecular TR diagonalizer / thermochem).
        this->bind_defaults(ReferenceGeometry{workspace_.get()});
        this->bind_defaults(DisplaceStep{workspace_.get()});
        this->bind_defaults(HessianAccumulator{workspace_.get()});
        this->bind_defaults(Symmetrize{workspace_.get()});
        this->bind_defaults(MassWeighting{workspace_.get()});
        this->bind_defaults(PhononDiagonalizer{workspace_.get()});
        // Clean force accumulator each displacement (classical pair styles).
        this->on(ATV::vib::Phase::force).add(cpu::ZeroForces{}, -1000);
        this->on(ATV::vib::Phase::force).add(cpu::ZeroEnergy{}, -999);
    }

    /// @brief Run FD phonons on `unit` (expanded by `na×nb×nc` internally).
    ///
    /// The input unit-cell Frame is not permanently mutated when the supercell
    /// is larger than 1×1×1. For 1×1×1 the geometry is restored to R0 on exit.
    ///
    /// @param unit Unit-cell system (positions [Å]).
    /// @return Γ-point spectrum of the supercell.
    PhononResult run(cpu::Frame& unit);

    /// @brief Most recent result from `run()`.
    const PhononResult& last_result() const { return last_result_; }

  protected:
    void install_defaults() override {
        const auto s = this->scan_pending_infra_();
        this->throw_if_missing_neighbor_list_(
            s, "Phonon",
            "Add `phonon.set_neighbor(cpu::md::NeighborList<>{cutoff, "
            "skin})`.");
    }

    void validate_components() const override {
        this->require_force();
        this->require_hessian();
    }

  private:
    std::unique_ptr<HessianWorkspace> workspace_{
        std::make_unique<HessianWorkspace>()};
    PhononResult last_result_;
};

} // namespace vib
} // namespace cpu
} // namespace ATV
