/// @file src/cpu/multiscale/internal/classical_nonbonded.cpp
#include "atomiverse/cpu/multiscale/internal/classical_nonbonded.h"

#include <cmath>
#include <iterator>
#include <stdexcept>
#include <unordered_set>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"

namespace ATV {
namespace cpu {
namespace multiscale {
namespace internal {

ClassicalNonbonded::ClassicalNonbonded(std::vector<LJTypeCoeff> lj,
                                       double cutoff, bool with_coulomb)
    : cutoff_(cutoff), cutoff2_(cutoff * cutoff), with_coulomb_(with_coulomb) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("ClassicalNonbonded: cutoff must be > 0");

    std::unordered_map<int, PairParam> diag;
    for (const auto& c : lj) {
        if (!(c.epsilon >= 0.0) || !(c.sigma > 0.0))
            throw std::invalid_argument(
                "ClassicalNonbonded: invalid LJ coefficients");
        table_[key_(c.type_i, c.type_j)] = {c.epsilon, c.sigma};
        if (c.type_i == c.type_j)
            diag[c.type_i] = {c.epsilon, c.sigma};
    }
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

ClassicalNonbonded::PairParam ClassicalNonbonded::lookup_(int ti,
                                                          int tj) const {
    auto it = table_.find(key_(ti, tj));
    if (it != table_.end())
        return it->second;
    return {};
}

double ClassicalNonbonded::launch(Frame& frame, const std::vector<int>& rows,
                                  bool forces_requested,
                                  double force_sign) const {
    const int n = static_cast<int>(frame.nlocal());
    if (n == 0)
        return 0.0;

    std::vector<int> atoms;
    if (rows.empty()) {
        atoms.resize(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            atoms[static_cast<std::size_t>(i)] = i;
    } else {
        atoms = rows;
    }
    const int m = static_cast<int>(atoms.size());
    if (m < 2)
        return 0.0;

    const auto type = frame.span(kAtom_type);
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

    const double ke = units::coulomb;
    double e = 0.0;
    for (int a = 0; a < m; ++a) {
        const int i = atoms[static_cast<std::size_t>(a)];
        const int ti = type[i];
        const double qi = static_cast<double>(charge[i]);
        for (int b = a + 1; b < m; ++b) {
            const int j = atoms[static_cast<std::size_t>(b)];
            const int tj = type[j];
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double dz = z[j] - z[i];
            frame.box.min_image(dx, dy, dz);
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= cutoff2_ || r2 == 0.0)
                continue;
            const double r = std::sqrt(r2);
            const double r2inv = 1.0 / r2;

            const PairParam p = lookup_(ti, tj);
            double fpair = 0.0;
            if (p.eps > 0.0 && p.sigma > 0.0) {
                const double sig2 = p.sigma * p.sigma;
                const double s2 = sig2 * r2inv;
                const double s6 = s2 * s2 * s2;
                const double s2c = sig2 / cutoff2_;
                const double s6c = s2c * s2c * s2c;
                const double vshift = 4.0 * p.eps * (s6c * s6c - s6c);
                e += 4.0 * p.eps * (s6 * s6 - s6) - vshift;
                fpair += 24.0 * p.eps * r2inv * (2.0 * s6 * s6 - s6);
            }
            if (with_coulomb_ && qi != 0.0) {
                const double qj = static_cast<double>(charge[j]);
                if (qj != 0.0) {
                    e += ke * qi * qj / r;
                    fpair += ke * qi * qj / (r2 * r);
                }
            }
            if (fx && fpair != 0.0) {
                const double s = force_sign * fpair;
                fx[i] -= s * dx;
                fy[i] -= s * dy;
                fz[i] -= s * dz;
                fx[j] += s * dx;
                fy[j] += s * dy;
                fz[j] += s * dz;
            }
        }
    }
    return e;
}

} // namespace internal
} // namespace multiscale
} // namespace cpu
} // namespace ATV
