/// @file cpu/electrodes/charging_work.h
/// @brief ChargingWork — electrode charging-work energy \(-\sum_i \Psi_i Q_i\).
///
/// MetalWalls total energy under constant potential (JOSS 5, 2373 (2020) eq. 6):
/// \(V(Q) = V_c(Q) - \sum_i \Psi_i Q_i\). This Component books the second term
/// only. It does **not** solve for electrode charges (const-pot Component) and
/// writes no nuclear forces.
///
/// Units: \(\Psi\) [V], \(Q\) [e], energy [eV] with \(1\,e\cdot 1\,\mathrm{V} =
/// 1\,\mathrm{eV}\).
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "atomiverse/core/role.h"
#include "atomiverse/cpu/component/potential.h"

namespace ATV {
namespace cpu {
namespace electrodes {

/// @brief Potential bookkeeping \(E_{\mathrm{charge}} = -\sum_i \Psi_i Q_i\) [eV].
///
/// Select electrode atoms by **type**; each listed type has a fixed applied
/// potential \(\Psi\) [V]. Types not listed contribute zero. Owned atoms only
/// (`nlocal`); MPI ranks accumulate locally and the existing energy `Reduce`
/// gathers the global total when multi-rank.
///
/// Bind via `sim.set_evaluator(ChargingWork{types, potentials})` on
/// `cpu::md::MD` or `cpu::Evaluate`. Prefer ordering **after** any const-pot
/// charge solve so \(Q\) on the Frame is current.
class ChargingWork : public cpu::Potential {
  public:
    /// @param electrode_types Atom types that belong to electrodes.
    /// @param potential_V     Applied potential \(\Psi\) [V] for each type
    ///                        (same length as `electrode_types`).
    /// @throws std::invalid_argument if lengths differ or a type is repeated.
    ChargingWork(std::vector<int> electrode_types,
                 std::vector<double> potential_V);

    /// @brief Component name.
    /// @return `"charging_work"`.
    std::string name() const override { return "charging_work"; }

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    /// @brief Charging work from the last `launch` [eV].
    /// @return \(-\sum \Psi_i Q_i\) over owned electrode atoms last step.
    double last_work() const { return last_work_; }

  private:
    std::vector<int> types_;
    std::vector<double> psi_V_;
    std::unordered_map<int, double> psi_of_type_;
    double last_work_ = 0.0;
};

} // namespace electrodes
} // namespace cpu
} // namespace ATV
