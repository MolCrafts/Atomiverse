/// @file cpu/multiscale/qmmm_electrostatic.h
/// @brief Electrostatic QM/MM coupling: engine slot + mask region + QM–MM LJ.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/multiscale/internal/qm_mm_vdw.h"
#include "atomiverse/cpu/multiscale/internal/qm_subsystem.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

namespace ATV {
namespace cpu {

/// @brief Additive electrostatic QM/MM coupling (Senn–Thiel embedding).
///
/// Owns only QM-involving cross terms:
/// - E_QM[ρ; MM sites] via any embedding-capable `QmEngine`
/// - E_QM–MM^vdW via per-type 12-6 LJ
///
/// MM–MM LJ/Coulomb/bonded belong to the user's ordinary MM styles. QM region
/// comes from the `kAtomQmZ` mask column (see `qm_mask.h`), not a ctor list.
class QmmmElectrostatic : public Potential {
  public:
    /// @param cross_lj QM–MM LJ diagonals/off-diagonals (Lorentz–Berthelot fill).
    /// @param lj_cutoff LJ cutoff [Å].
    /// @param embed_cutoff Embedding site collection cutoff [Å].
    QmmmElectrostatic(std::vector<LJTypeCoeff> cross_lj, double lj_cutoff,
                      double embed_cutoff);

    std::string name() const override { return "qmmm_electrostatic"; }

    /// @brief Link-H scale g ∈ (0,1). Region cuts come from Frame bonds×mask.
    double& link_scale() { return sub_.link_scale; }
    bool& reset_history_each_launch() {
        return sub_.reset_history_each_launch();
    }

    /// @brief Own an embedding-capable engine (heap-stable).
    template <typename E,
              std::enable_if_t<!std::is_lvalue_reference_v<E>, int> = 0>
    QmmmElectrostatic& set_engine(E&& e) {
        sub_.set_engine(std::forward<E>(e));
        return *this;
    }

    /// @brief Borrow an embedding-capable engine.
    template <typename E>
    QmmmElectrostatic& set_engine_ref(E& e) {
        sub_.set_engine_ref(e);
        return *this;
    }

    template <typename E>
    QmmmElectrostatic& set_engine(E&) = delete;

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    double last_qm_energy() const { return e_qm_; }
    double last_vdw_energy() const { return e_vdw_; }
    int last_n_links() const { return sub_.last_n_links(); }
    std::array<double, 3> last_force_residual() const {
        return sub_.last_force_residual();
    }

  private:
    multiscale::internal::QmSubsystem sub_;
    multiscale::internal::QmMmVdw vdw_;
    double e_qm_ = 0;
    double e_vdw_ = 0;
};

} // namespace cpu
} // namespace ATV
