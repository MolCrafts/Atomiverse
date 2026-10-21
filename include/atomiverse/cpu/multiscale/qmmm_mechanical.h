/// @file cpu/multiscale/qmmm_mechanical.h
/// @brief Mechanical QM/MM: vacuum QM engine + classical QM–MM LJ/Coulomb.
#pragma once

#include <array>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/multiscale/internal/qm_mm_vdw.h"
#include "atomiverse/cpu/multiscale/internal/qm_subsystem.h"
#include "atomiverse/cpu/potentials/lj_type_coeff.h"

namespace ATV {
namespace cpu {

/// @brief Mechanical embedding: E_QM(vacuum) + E_QM–MM^{LJ} + E_QM–MM^{Coul}.
///
/// Any `QmEngine` (embedding never probed). QM region from `kAtomQmZ`.
/// Classical cross-Coulomb reads both sides' `kAtom_charge` (QM charges may
/// be nonzero — unlike electrostatic embedding).
class QmmmMechanical : public Potential {
  public:
    /// @param cross_lj QM–MM LJ table (Lorentz–Berthelot fill).
    /// @param lj_cutoff LJ and classical Coulomb cutoff [Å].
    QmmmMechanical(std::vector<LJTypeCoeff> cross_lj, double lj_cutoff);

    std::string name() const override { return "qmmm_mechanical"; }

    /// @brief Link-H scale g ∈ (0,1). Region cuts come from Frame bonds×mask.
    double& link_scale() { return sub_.link_scale; }
    bool& reset_history_each_launch() {
        return sub_.reset_history_each_launch();
    }

    template <typename E,
              std::enable_if_t<!std::is_lvalue_reference_v<E>, int> = 0>
    QmmmMechanical& set_engine(E&& e) {
        sub_.set_engine(std::forward<E>(e));
        return *this;
    }
    template <typename E>
    QmmmMechanical& set_engine_ref(E& e) {
        sub_.set_engine_ref(e);
        return *this;
    }
    template <typename E>
    QmmmMechanical& set_engine(E&) = delete;

    void build(const BuildContext& ctx) override;
    void launch(const LaunchContext& ctx) override;

    double last_qm_energy() const { return e_qm_; }
    double last_mm_energy() const { return e_vdw_ + e_coul_; }
    double last_coul_energy() const { return e_coul_; }
    int last_n_links() const { return sub_.last_n_links(); }

  private:
    multiscale::internal::QmSubsystem sub_;
    multiscale::internal::QmMmVdw vdw_;
    double coul_cutoff_ = 0;
    double e_qm_ = 0, e_vdw_ = 0, e_coul_ = 0;
};

} // namespace cpu
} // namespace ATV
