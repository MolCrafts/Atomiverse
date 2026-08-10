/// @file cpu/gpw/stages/density_stage.h
/// @brief DensityStage — `gpw::Phase::density`: P from occupied orbitals.
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/linalg/density.h"

namespace ATV {
namespace cpu {
namespace gpw {

/// @brief Density-build stage for the GPW Driver loop (Γ RKS closed-shell).
///
/// DIIS path: `P = dens(C)` with occupation 2. Linear-mix path: damped
/// `P ← (1-α)P + α dens(C)`.
class DensityStage : public cpu::Component {
  public:
    /// @param use_diis     True → replace P; false → linear mix with `mix`.
    /// @param mix          Linear density-mixing parameter (dimensionless).
    /// @param n_occ        Number of doubly-occupied orbitals.
    /// @param n_basis      AO dimension.
    /// @param C            MO coefficients (read).
    /// @param P            Density matrix (read/write).
    /// @param skip_update  When true, leave P unchanged (converged break).
    DensityStage(bool use_diis, double mix, int n_occ, int n_basis,
                 const linalg::RealMatrix& C, linalg::RealMatrix& P,
                 const bool& skip_update)
        : use_diis_(use_diis), mix_(mix), n_occ_(n_occ), n_basis_(n_basis),
          C_(C), P_(P), skip_update_(skip_update) {}

    /// @brief Component name.
    /// @return `"gpw_density_stage"`.
    std::string name() const override { return "gpw_density_stage"; }

    /// @brief Default binding: `gpw::Phase::density` @ 0.
    /// @return Single binding vector.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::gpw::Phase::density, 0)};
    }

    /// @brief Rebuild or mix the density from C.
    /// @param ctx Launch context; self-guards on `gpw::Phase::density`.
    void launch(const cpu::LaunchContext& ctx) override {
        if (!ctx.at(ATV::gpw::Phase::density))
            return;
        if (skip_update_)
            return;
        linalg::RealMatrix P_new =
            linalg::build_density_matrix(C_, n_occ_, n_basis_, 2.0);
        if (use_diis_)
            P_ = std::move(P_new);
        else
            P_ = (1.0 - mix_) * P_ + mix_ * P_new;
    }

  private:
    bool use_diis_;
    double mix_;
    int n_occ_;
    int n_basis_;
    const linalg::RealMatrix& C_;
    linalg::RealMatrix& P_;
    const bool& skip_update_;
};

} // namespace gpw
} // namespace cpu
} // namespace ATV
