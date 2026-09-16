/// @file src/cpu/multiscale/internal/qm_mm_vdw.cpp
#include "atomiverse/cpu/multiscale/internal/qm_mm_vdw.h"

#include <cmath>
#include <iterator>
#include <stdexcept>
#include <unordered_set>

#include "atomiverse/core/frame_schema.h"

namespace ATV {
namespace cpu {
namespace multiscale {
namespace internal {

QmMmVdw::QmMmVdw(std::vector<LJTypeCoeff> cross_lj, double cutoff)
    : cutoff_(cutoff), cutoff2_(cutoff * cutoff) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("QmMmVdw: cutoff must be > 0");

    // Seed table from explicit rows.
    std::unordered_map<int, PairParam> diag;
    for (const auto& c : cross_lj) {
        if (!(c.epsilon >= 0.0) || !(c.sigma > 0.0))
            throw std::invalid_argument("QmMmVdw: invalid LJ coefficients");
        table_[key_(c.type_i, c.type_j)] = {c.epsilon, c.sigma};
        if (c.type_i == c.type_j)
            diag[c.type_i] = {c.epsilon, c.sigma};
    }
    // Lorentz–Berthelot fill for missing off-diagonals among known diagonals.
    for (auto it = diag.begin(); it != diag.end(); ++it) {
        for (auto jt = std::next(it); jt != diag.end(); ++jt) {
            const auto k = key_(it->first, jt->first);
            if (table_.count(k))
                continue;
            PairParam p;
            p.eps = std::sqrt(it->second.eps * jt->second.eps);
            p.sigma = 0.5 * (it->second.sigma + jt->second.sigma);
            table_[k] = p;
        }
    }
}

QmMmVdw::PairParam QmMmVdw::lookup_(int ti, int tj) const {
    auto it = table_.find(key_(ti, tj));
    if (it != table_.end())
        return it->second;
    return {}; // ε=0 → no interaction
}

double QmMmVdw::launch(Frame& frame, const std::vector<int>& qm_rows,
                       bool forces_requested) const {
    const int n = static_cast<int>(frame.nlocal());
    if (qm_rows.empty() || n == 0)
        return 0.0;

    const auto type = frame.span(kAtom_type);
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
    double e = 0.0;

    for (int iq : qm_rows) {
        const int ti = type[iq];
        for (int j = 0; j < n; ++j) {
            if (qm_set.count(j))
                continue; // skip QM–QM
            const int tj = type[j];
            const PairParam p = lookup_(ti, tj);
            if (!(p.eps > 0.0) || !(p.sigma > 0.0))
                continue;
            double dx = x[j] - x[iq];
            double dy = y[j] - y[iq];
            double dz = z[j] - z[iq];
            frame.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= cutoff2_ || r2 == 0.0)
                continue;
            const double sig2 = p.sigma * p.sigma;
            const double r2inv = 1.0 / r2;
            const double s2 = sig2 * r2inv;
            const double s6 = s2 * s2 * s2;
            const double s2c = sig2 / cutoff2_;
            const double s6c = s2c * s2c * s2c;
            const double vshift = 4.0 * p.eps * (s6c * s6c - s6c);
            e += 4.0 * p.eps * (s6 * s6 - s6) - vshift;
            if (fx) {
                const double fpair =
                    24.0 * p.eps * r2inv * (2.0 * s6 * s6 - s6);
                // Force on j from i→j vector: +fpair * r̂ direction
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

} // namespace internal
} // namespace multiscale
} // namespace cpu
} // namespace ATV
