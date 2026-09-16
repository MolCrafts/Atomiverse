/// @file cpu/observe/species_density_profile.cpp
/// @brief SpeciesDensityProfile implementation — per-species ρ_s(z).

#include "atomiverse/cpu/observe/species_density_profile.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {

void SpeciesDensityProfile::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx); // ghosts would double-count owners
    const StorageReal* __restrict z = f.span(kAtom_z);
    const int* __restrict type = f.span(kAtom_type);

    const std::size_t ns = types_.size();
    const std::size_t nb = static_cast<std::size_t>(n_bins_);
    std::fill(density_.begin(), density_.end(), 0.0);
    counts_.assign(ns, 0);

    for (int i = 0; i < n; ++i) {
        const int ti = type[i];
        for (std::size_t s = 0; s < ns; ++s) {
            if (ti != types_[s])
                continue;
            const double zi = static_cast<double>(z[i]);
            if (zi >= z_min_ && zi < z_max_) {
                int bin = static_cast<int>((zi - z_min_) / dz_);
                if (bin < 0)
                    bin = 0;
                if (bin >= n_bins_)
                    bin = n_bins_ - 1;
                density_[s * nb + static_cast<std::size_t>(bin)] += 1.0;
                ++counts_[s];
            }
            break; // one species per type; types are exclusive
        }
    }

#ifdef ATV_HAS_MPI
    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        std::vector<double> global(density_.size(), 0.0);
        MPI_Allreduce(density_.data(), global.data(),
                      static_cast<int>(density_.size()), MPI_DOUBLE, MPI_SUM,
                      ctx.runtime.md.dd->cart_comm);
        density_.swap(global);
        std::vector<long> gcount(ns, 0);
        MPI_Allreduce(counts_.data(), gcount.data(), static_cast<int>(ns),
                      MPI_LONG, MPI_SUM, ctx.runtime.md.dd->cart_comm);
        counts_.swap(gcount);
    }
#endif

    const double area = f.box.length(0) * f.box.length(1);
    const double inv_vol = 1.0 / (area * dz_);
    for (std::size_t c = 0; c < density_.size(); ++c) {
        density_[c] *= inv_vol;
        mean_sum_[c] += density_[c];
    }
    ++n_samples_;
}

} // namespace cpu
} // namespace ATV
