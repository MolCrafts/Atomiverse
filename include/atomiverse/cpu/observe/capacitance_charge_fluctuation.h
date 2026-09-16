/// @file cpu/observe/capacitance_charge_fluctuation.h
/// @brief CapacitanceChargeFluctuation — \(C\) from electrode charge moments.
///
/// E10 map: post-process / on-the-fly capacitance estimate for classical
/// constant-potential electrodes. Frame-immutable Observer that samples total
/// electrode charge \(Q = \sum_{i\in e_0} q_i\) each launch and estimates
///
/// \[
/// C_{\mathrm{diff}} = \frac{\langle Q^2\rangle - \langle Q\rangle^2}{k_B T}
///   \qquad[\mathrm{e}/\mathrm{V}],
/// \]
///
/// and (when \(\Delta\Psi \ne 0\))
///
/// \[
/// C_{\mathrm{int}} = \frac{\langle Q\rangle}{\Delta\Psi}
///   \qquad[\mathrm{e}/\mathrm{V}].
/// \]
///
/// Does **not** solve constant-potential charges — it measures whatever is
/// already on the Frame. Primary electrode is type list index 0 (parallel-plate
/// left wall). Units: \(Q\) [e], \(T\) [K], \(\Psi\) [V], \(k_B\) from
/// `units::kB_ev_per_k` so \(k_B T\) [eV] and \(1\,e\cdot 1\,\mathrm{V}=1\,\mathrm{eV}\).
///
/// Refs: Limmer et al., J. Chem. Phys. 139, 184704 (2013); Scalfi et al.,
/// Phys. Chem. Chem. Phys. 22, 10480 (2020); MetalWalls JOSS 5, 2373 (2020)
/// charge-fluctuation analysis tools.
#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/observer.h"

namespace ATV {
namespace cpu {

/// @brief Running capacitance estimate from electrode charge fluctuations.
///
/// Construct with electrode type ids (index 0 = primary \(Q\)), temperature
/// [K], and optional applied potential difference \(\Delta\Psi\) [V] for the
/// integral route. Parameters are ctor args — no public Config bag.
class CapacitanceChargeFluctuation : public Observer {
  public:
    /// @brief Construct capacitance / charge-fluctuation observer.
    /// @param electrode_types Atom type ids (non-empty); index 0 is primary \(Q\).
    /// @param temperature_K Temperature \(T\) [K] for \(k_B T\) (must be \(> 0\)).
    /// @param delta_psi_V Applied electrode potential difference \(\Delta\Psi\)
    ///        [V]; 0 disables integral capacitance (returns NaN).
    /// @throws std::invalid_argument if types empty or \(T \le 0\).
    CapacitanceChargeFluctuation(std::vector<int> electrode_types,
                                 double temperature_K, double delta_psi_V = 0.0)
        : electrode_types_(std::move(electrode_types)),
          temperature_K_(temperature_K), delta_psi_V_(delta_psi_V) {
        if (electrode_types_.empty())
            throw std::invalid_argument("CapacitanceChargeFluctuation: need at "
                                        "least one electrode type");
        if (!(temperature_K_ > 0.0))
            throw std::invalid_argument(
                "CapacitanceChargeFluctuation: temperature_K must be > 0");
        last_totals_.assign(electrode_types_.size(), 0.0);
    }

    /// @brief Component identifier.
    /// @return `"capacitance_charge_fluctuation"`.
    std::string name() const override {
        return "capacitance_charge_fluctuation";
    }

    /// @brief Default phase: end of MD step (after charge update).
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::end_of_step, +50)};
    }

    /// @brief Sample electrode \(Q\), update moments, publish members + meta.
    /// @param ctx Launch context (frame + optional domain decomposition).
    void launch(const LaunchContext& ctx) override;

    /// @brief Clear accumulated moments (keeps parameters).
    void reset() {
        n_samples_ = 0;
        sum_q_ = 0.0;
        sum_q2_ = 0.0;
        last_c_diff_ = 0.0;
        last_c_int_ = std::numeric_limits<double>::quiet_NaN();
        last_totals_.assign(electrode_types_.size(), 0.0);
    }

    /// @brief Number of charge samples accumulated.
    /// @return Sample count \(N\).
    std::size_t n_samples() const { return n_samples_; }

    /// @brief Temperature used for \(C_{\mathrm{diff}}\) [K].
    /// @return \(T\) [K].
    double temperature() const { return temperature_K_; }

    /// @brief Applied \(\Delta\Psi\) [V] (0 if integral route disabled).
    /// @return \(\Delta\Psi\) [V].
    double delta_psi() const { return delta_psi_V_; }

    /// @brief Number of electrodes (size of the type list).
    std::size_t n_electrodes() const { return electrode_types_.size(); }

    /// @brief Atom type id for electrode `e`.
    /// @param e Electrode index.
    /// @return Type id.
    int electrode_type(std::size_t e) const { return electrode_types_.at(e); }

    /// @brief Total charge on electrode `e` from the last launch [e].
    /// @param e Electrode index.
    /// @return \(Q_e\) [e] last sample.
    double total_charge(std::size_t e) const { return last_totals_.at(e); }

    /// @brief Running mean \(\langle Q\rangle\) of primary electrode [e].
    /// @return Mean charge; 0 if no samples.
    double mean_charge() const {
        return n_samples_ == 0 ? 0.0 : sum_q_ / static_cast<double>(n_samples_);
    }

    /// @brief Population variance \(\langle Q^2\rangle-\langle Q\rangle^2\) [e²].
    /// @return \(\mathrm{Var}(Q)\); 0 if fewer than 1 sample.
    double charge_variance() const {
        if (n_samples_ == 0)
            return 0.0;
        const double n = static_cast<double>(n_samples_);
        const double mean = sum_q_ / n;
        return sum_q2_ / n - mean * mean;
    }

    /// @brief Differential capacitance \(C_{\mathrm{diff}}=\mathrm{Var}(Q)/(k_B T)\) [e/V].
    /// @return Last published \(C_{\mathrm{diff}}\) (0 before any sample).
    double differential_capacitance() const { return last_c_diff_; }

    /// @brief Integral capacitance \(C_{\mathrm{int}}=\langle Q\rangle/\Delta\Psi\) [e/V].
    /// @return Last published \(C_{\mathrm{int}}\); NaN if \(\Delta\Psi=0\) or no samples.
    double integral_capacitance() const { return last_c_int_; }

  private:
    std::vector<int> electrode_types_;
    double temperature_K_ = 300.0;
    double delta_psi_V_ = 0.0;

    std::size_t n_samples_ = 0;
    double sum_q_ = 0.0;
    double sum_q2_ = 0.0;
    double last_c_diff_ = 0.0;
    double last_c_int_ = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> last_totals_;
};

} // namespace cpu
} // namespace ATV
