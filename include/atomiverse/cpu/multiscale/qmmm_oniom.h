/// @file cpu/multiscale/qmmm_oniom.h
/// @brief Two-layer subtractive ONIOM connection evaluator.
#pragma once

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/multiscale/internal/classical_nonbonded.h"
#include "atomiverse/cpu/multiscale/internal/qm_subsystem.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

namespace ATV {
namespace cpu {

/// @brief Two-layer ONIOM (subtractive) energy/forces.
///
/// \f[
///   E = E_{\mathrm{high}}(\mathrm{model})
///     + E_{\mathrm{low}}(\mathrm{real})
///     - E_{\mathrm{low}}(\mathrm{model})
/// \f]
///
/// - **model** — QM atoms from Frame `kAtomQmZ` (high also gets topology-derived
///   link H via `QmSubsystem`).
/// - **high** — any vacuum `QmEngine` (`set_engine`).
/// - **low** — classical nonbonded (LJ table + optional Coulomb) on the full
///   Frame and on the model atom set.
///
/// Compose under `MD` / `Minimize` / `Evaluate` like other QM/MM evaluators.
/// Do **not** also bind the same classical low table via a second evaluator
/// (double-counts \(E_{\mathrm{low}}(\mathrm{real})\)).
///
/// v1 is mechanical ONIOM (no electrostatic embedding of high in the MM
/// environment). Svensson et al., *J. Phys. Chem.* **100**, 19357 (1996).
class QmmmOniom : public Potential {
  public:
    /// @param low_lj Low-level LJ diagonals/pairs (Lorentz–Berthelot fill).
    /// @param cutoff Low-level nonbonded cutoff [Å].
    /// @param with_coulomb Include classical Coulomb among low pairs.
    QmmmOniom(std::vector<LJTypeCoeff> low_lj, double cutoff,
              bool with_coulomb = false);

    std::string name() const override { return "qmmm_oniom"; }

    /// @brief Link-H scale for high-level model only [dimensionless].
    double& link_scale() { return sub_.link_scale; }
    bool& reset_history_each_launch() {
        return sub_.reset_history_each_launch();
    }

    template <typename E,
              std::enable_if_t<!std::is_lvalue_reference_v<E>, int> = 0>
    QmmmOniom& set_engine(E&& e) {
        sub_.set_engine(std::forward<E>(e));
        return *this;
    }
    template <typename E>
    QmmmOniom& set_engine_ref(E& e) {
        sub_.set_engine_ref(e);
        return *this;
    }
    template <typename E>
    QmmmOniom& set_engine(E&) = delete;

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    double last_high_energy() const { return e_high_; }
    double last_low_real_energy() const { return e_low_real_; }
    double last_low_model_energy() const { return e_low_model_; }
    int last_n_links() const { return sub_.last_n_links(); }

  private:
    multiscale::internal::QmSubsystem sub_;
    multiscale::internal::ClassicalNonbonded low_;
    double e_high_ = 0;
    double e_low_real_ = 0;
    double e_low_model_ = 0;
};

} // namespace cpu
} // namespace ATV
