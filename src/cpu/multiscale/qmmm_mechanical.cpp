/// @file src/cpu/multiscale/qmmm_mechanical.cpp
#include "atomiverse/cpu/multiscale/qmmm_mechanical.h"

#include <cmath>
#include <unordered_set>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {

namespace {

/// Classical QM–MM Coulomb pair sum (min-image, cutoff), both sides from
/// kAtom_charge. Returns energy [eV]; optionally accumulates forces [eV/Å].
double classical_qmmm_coulomb(Frame& frame, const std::vector<int>& qm_rows,
                              double cutoff, bool forces_requested) {
    if (qm_rows.empty() || !(cutoff > 0.0))
        return 0.0;
    const int n = static_cast<int>(frame.nlocal());
    const auto charge = frame.span(kAtom_charge);
    const StorageReal* x = frame.span(kAtom_x);
    const StorageReal* y = frame.span(kAtom_y);
    const StorageReal* z = frame.span(kAtom_z);
    AccReal* fx = nullptr;
    AccReal* fy = nullptr;
    AccReal* fz = nullptr;
    if (forces_requested) {
        fx = frame.span(kAtom_fx);
        fy = frame.span(kAtom_fy);
        fz = frame.span(kAtom_fz);
    }
    std::unordered_set<int> qm_set(qm_rows.begin(), qm_rows.end());
    const double rc2 = cutoff * cutoff;
    const double ke = units::coulomb; // eV·Å/e²
    double e = 0.0;
    for (int iq : qm_rows) {
        const double qi = static_cast<double>(charge[iq]);
        if (qi == 0.0)
            continue;
        for (int j = 0; j < n; ++j) {
            if (qm_set.count(j))
                continue;
            const double qj = static_cast<double>(charge[j]);
            if (qj == 0.0)
                continue;
            double dx = x[j] - x[iq];
            double dy = y[j] - y[iq];
            double dz = z[j] - z[iq];
            frame.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= rc2 || r2 == 0.0)
                continue;
            const double r = std::sqrt(r2);
            e += ke * qi * qj / r;
            if (fx) {
                const double fpair = ke * qi * qj / (r2 * r);
                fx[iq] -= fpair * dx;
                fy[iq] -= fpair * dy;
                fz[iq] -= fpair * dz;
                fx[j] += fpair * dx;
                fy[j] += fpair * dy;
                fz[j] += fpair * dz;
            }
        }
    }
    return e;
}

} // namespace

QmmmMechanical::QmmmMechanical(std::vector<LJTypeCoeff> cross_lj,
                               double lj_cutoff)
    : vdw_(std::move(cross_lj), lj_cutoff), coul_cutoff_(lj_cutoff) {}

void QmmmMechanical::build(const BuildContext& ctx) {
    // Any QmEngine; embedding never required.
    sub_.build(ctx.frame, ctx.runtime.mpi_size, /*require_embedding=*/false);
}

void QmmmMechanical::launch(const LaunchContext& ctx) {
    sub_.launch(ctx.frame, ctx.forces_requested, e_qm_);
    e_vdw_ = vdw_.launch(ctx.frame, sub_.qm_rows(), ctx.forces_requested);
    e_coul_ = classical_qmmm_coulomb(ctx.frame, sub_.qm_rows(), coul_cutoff_,
                                     ctx.forces_requested);
    ctx.frame.meta.get(kEnergy) += e_vdw_ + e_coul_;
    // Classical terms are not electronic.
}

} // namespace cpu
} // namespace ATV
