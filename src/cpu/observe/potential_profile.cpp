/// @file cpu/observe/potential_profile.cpp
/// @brief PotentialProfile implementation — planar Poisson \(\Psi(z)\).

#include "atomiverse/cpu/observe/potential_profile.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <cmath>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {

void PotentialProfile::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const StorageReal* __restrict q = f.span(kAtom_charge);
    const StorageReal* __restrict z = f.span(kAtom_z);

    const std::size_t nb = static_cast<std::size_t>(n_bins_);
    q_bin_.assign(nb, 0.0);

    for (int i = 0; i < n; ++i) {
        const double zi = static_cast<double>(z[i]);
        if (zi < z_min_ || zi >= z_max_)
            continue;
        int bin = static_cast<int>((zi - z_min_) / dz_);
        if (bin < 0)
            bin = 0;
        if (bin >= n_bins_)
            bin = n_bins_ - 1;
        q_bin_[static_cast<std::size_t>(bin)] += static_cast<double>(q[i]);
    }

#ifdef ATV_HAS_MPI
    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        std::vector<double> global(nb, 0.0);
        MPI_Allreduce(q_bin_.data(), global.data(), n_bins_, MPI_DOUBLE,
                      MPI_SUM, ctx.runtime.md.dd->cart_comm);
        q_bin_.swap(global);
    }
#endif

    const double Lx = f.box.length(0);
    const double Ly = f.box.length(1);
    const double area = Lx * Ly;
    const double inv_area_dz = 1.0 / (area * dz_);
    const double four_pi_ke = 4.0 * units::pi * units::coulomb;

    for (std::size_t i = 0; i < nb; ++i)
        rho_[i] = q_bin_[i] * inv_area_dz;

    // Piecewise-constant ρ: edge field / potential, then bin-center samples.
    double E_left = 0.0;
    double psi_left = 0.0;
    for (std::size_t i = 0; i < nb; ++i) {
        const double rho_i = rho_[i];
        const double dE = four_pi_ke * rho_i * dz_;
        // E at bin center: E_left + 4π ke ρ (dz/2)
        field_[i] = E_left + four_pi_ke * rho_i * (0.5 * dz_);
        // Ψ at bin center: integrate E from left edge to center
        psi_[i] = psi_left
                  - (E_left * (0.5 * dz_)
                     + 0.5 * four_pi_ke * rho_i * (0.5 * dz_) * (0.5 * dz_));
        // Advance to right edge of bin
        psi_left =
            psi_left - (E_left * dz_ + 0.5 * four_pi_ke * rho_i * dz_ * dz_);
        E_left += dE;
    }
    potential_drop_ = psi_left; // Ψ(z_max) − Ψ(z_min), Ψ(z_min)=0

    if (!f.meta.contains(kPotentialDrop.key))
        f.meta.insert(kPotentialDrop, potential_drop_);
    else
        f.meta.get(kPotentialDrop) = potential_drop_;
}

} // namespace cpu
} // namespace ATV
