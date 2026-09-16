/// @file cpu/observe/capacitance_charge_fluctuation.cpp
/// @brief CapacitanceChargeFluctuation implementation.

#include "atomiverse/cpu/observe/capacitance_charge_fluctuation.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/communication/domain.h"
#include "atomiverse/cpu/component/owned_range.h"

#include <limits>

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

namespace ATV {
namespace cpu {

void CapacitanceChargeFluctuation::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = owned_count(ctx);
    const StorageReal* __restrict q = f.span(kAtom_charge);
    const int* __restrict type = f.span(kAtom_type);

    const std::size_t ne = electrode_types_.size();
    last_totals_.assign(ne, 0.0);

    for (int i = 0; i < n; ++i) {
        const int ti = type[i];
        for (std::size_t e = 0; e < ne; ++e) {
            if (ti == electrode_types_[e]) {
                last_totals_[e] += static_cast<double>(q[i]);
                break;
            }
        }
    }

#ifdef ATV_HAS_MPI
    if (ctx.runtime.md.dd && ctx.runtime.mpi_size > 1
        && ctx.runtime.md.dd->cart_comm != MPI_COMM_NULL) {
        std::vector<double> global(ne, 0.0);
        MPI_Allreduce(last_totals_.data(), global.data(), static_cast<int>(ne),
                      MPI_DOUBLE, MPI_SUM, ctx.runtime.md.dd->cart_comm);
        last_totals_.swap(global);
    }
#endif

    // Primary electrode (index 0) drives capacitance moments.
    const double Q = last_totals_[0];
    ++n_samples_;
    sum_q_ += Q;
    sum_q2_ += Q * Q;

    const double kT = units::kB_ev_per_k * temperature_K_; // eV; 1 e·V = 1 eV
    const double var = charge_variance();
    last_c_diff_ = var / kT;

    if (delta_psi_V_ != 0.0) {
        last_c_int_ = mean_charge() / delta_psi_V_;
    } else {
        last_c_int_ = std::numeric_limits<double>::quiet_NaN();
    }

    if (!f.meta.contains(kCapacitanceDiff.key))
        f.meta.insert(kCapacitanceDiff, last_c_diff_);
    else
        f.meta.get(kCapacitanceDiff) = last_c_diff_;

    if (delta_psi_V_ != 0.0) {
        if (!f.meta.contains(kCapacitanceIntegral.key))
            f.meta.insert(kCapacitanceIntegral, last_c_int_);
        else
            f.meta.get(kCapacitanceIntegral) = last_c_int_;
    }

    // Keep electrode charge meta in sync for thermo (first ≤3 electrodes).
    MetaF64x3 pack{{0.0, 0.0, 0.0}};
    for (std::size_t e = 0; e < ne && e < 3; ++e)
        pack[e] = last_totals_[e];
    if (!f.meta.contains(kElectrodeCharge.key))
        f.meta.insert(kElectrodeCharge, pack);
    else
        f.meta.get(kElectrodeCharge) = pack;
}

} // namespace cpu
} // namespace ATV
