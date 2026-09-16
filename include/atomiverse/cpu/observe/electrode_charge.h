/// @file cpu/observe/electrode_charge.h
/// @brief ElectrodeChargeObserver — per-electrode total charge \(Q_{\mathrm{tot}}\).
///
/// E1 map §8: electrode charge time series. Frame-immutable Observer that
/// sums `kAtom_charge` over owned atoms whose `kAtom_type` matches each
/// electrode type id. Does **not** solve constant-potential charges — it
/// measures whatever is already on the Frame (fixed charges or a prior
/// const-pot Component).
///
/// Units: elementary charge [e] on the public surface.
///
/// Refs: MetalWalls electrode charge diagnostics (JOSS 10.21105/joss.02373);
/// LAMMPS `compute reduce sum` on group charge (oracle twin for fixed-q).
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/observer.h"

namespace ATV {
namespace cpu {

/// @brief Per-electrode total charge observer for classical MD electrodes.

///
/// Construct with one atom type id per electrode (order = electrode index).
/// Bind at MD `end_of_step` (default) after the charge source has written
/// `kAtom_charge`. Parameters are ctor args / members — no public Config bag.
class ElectrodeChargeObserver : public Observer {
  public:
    /// @brief Construct with one type id per electrode.
    /// @param electrode_types Atom type ids (one entry per electrode); must be
    ///        non-empty. Parallel-plate foothold: `{left_type, right_type}`.
    explicit ElectrodeChargeObserver(std::vector<int> electrode_types)
        : electrode_types_(std::move(electrode_types)) {
        if (electrode_types_.empty())
            throw std::invalid_argument(
                "ElectrodeChargeObserver: need at least one electrode type");
        totals_.assign(electrode_types_.size(), 0.0);
    }

    /// @brief Component identifier.
    /// @return `"electrode_charge"`.
    std::string name() const override { return "electrode_charge"; }

    /// @brief Default phase: end of MD step (after force / charge update).
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +40)};
    }

    /// @brief Sum owned charges per electrode; publish members + meta.
    /// @param ctx Launch context (frame + optional domain decomposition).
    void launch(const LaunchContext& ctx) override;

    /// @brief Number of electrodes (size of the type list).
    std::size_t n_electrodes() const { return electrode_types_.size(); }

    /// @brief Atom type id for electrode `e`.
    /// @param e Electrode index in \([0, n_{\mathrm{electrodes}})\)).
    /// @return Type id.
    int electrode_type(std::size_t e) const { return electrode_types_.at(e); }

    /// @brief Total charge on electrode `e` from the last launch [e].
    /// @param e Electrode index.
    /// @return \(Q_{\mathrm{tot},e}\) [e].
    double total_charge(std::size_t e) const { return totals_.at(e); }

    /// @brief All electrode totals from the last launch [e].
    /// @return Reference to length-`n_electrodes` vector.
    const std::vector<double>& total_charges() const { return totals_; }

  private:
    std::vector<int> electrode_types_;
    std::vector<double> totals_;
};

} // namespace cpu
} // namespace ATV
