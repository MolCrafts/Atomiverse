/// @file cpu/observe/electrode_charge.cpp
/// @brief ElectrodeChargeObserver implementation.

#include "atomiverse/cpu/observe/electrode_charge.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {

void ElectrodeChargeObserver::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx); // ghosts would double-count owners' charge
    const StorageReal* __restrict q = f.span(kAtom_charge);
    const int* __restrict type = f.span(kAtom_type);

    const std::size_t ne = electrode_types_.size();
    totals_.assign(ne, 0.0);

    for (int i = 0; i < n; ++i) {
        const int ti = type[i];
        for (std::size_t e = 0; e < ne; ++e) {
            if (ti == electrode_types_[e]) {
                totals_[e] += static_cast<double>(q[i]);
                break; // one type per electrode; types are exclusive
            }
        }
    }

#ifdef ATV_HAS_MPI
    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        std::vector<double> global(ne, 0.0);
        MPI_Allreduce(totals_.data(), global.data(), static_cast<int>(ne),
                      MPI_DOUBLE, MPI_SUM, ctx.runtime.md.dd->cart_comm);
        totals_.swap(global);
    }
#endif

    // Frame meta: first ≤3 electrodes for thermo / write-back.
    MetaF64x3 pack{{0.0, 0.0, 0.0}};
    for (std::size_t e = 0; e < ne && e < 3; ++e)
        pack[e] = totals_[e];
    if (!f.meta.contains(kElectrodeCharge.key))
        f.meta.insert(kElectrodeCharge, pack);
    else
        f.meta.get(kElectrodeCharge) = pack;
}

} // namespace cpu
} // namespace ATV
