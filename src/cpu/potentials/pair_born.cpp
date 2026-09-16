/// @file cpu/potentials/pair_born.cpp
#include "atomiverse/cpu/potentials/pair_born.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/force_scratch.h"

namespace ATV {
namespace cpu {

namespace {

double shift_of(double A, double inv_rho, double sigma, double C, double D,
                double cutoff) {
    const double rc2 = cutoff * cutoff;
    const double rc6 = rc2 * rc2 * rc2;
    const double rc8 = rc6 * rc2;
    return A * std::exp((sigma - cutoff) * inv_rho) - C / rc6 + D / rc8;
}

} // namespace

PairBorn::PairBorn(double A, double rho, double sigma, double C, double D,
                   double cutoff)
    : Pair(cutoff), A_(A), rho_(rho), sigma_(sigma), C_(C), D_(D) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairBorn: cutoff must be > 0");
    if (!(rho > 0.0))
        throw std::invalid_argument("PairBorn: rho must be > 0");
    if (!(A >= 0.0))
        throw std::invalid_argument("PairBorn: A must be ≥ 0");
    if (!(C >= 0.0))
        throw std::invalid_argument("PairBorn: C must be ≥ 0");
    cutoff_sq_ = cutoff * cutoff;
    inv_rho_ = 1.0 / rho;
    v_shift_ = shift_of(A, inv_rho_, sigma, C, D, cutoff);
}

PairBorn::PairBorn(std::vector<BornTypeCoeff> coeffs, double cutoff)
    : Pair(cutoff), multi_(true) {
    if (!(cutoff > 0.0))
        throw std::invalid_argument("PairBorn: cutoff must be > 0");
    if (coeffs.empty())
        throw std::invalid_argument(
            "PairBorn: multi-type constructor needs at least one coefficient");

    cutoff_sq_ = cutoff * cutoff;

    int tmax = 0;
    for (const auto& c : coeffs) {
        if (c.type_i < 0 || c.type_j < 0)
            throw std::invalid_argument("PairBorn: types must be ≥ 0");
        if (!(c.rho > 0.0))
            throw std::invalid_argument("PairBorn: rho must be > 0");
        if (!(c.A >= 0.0))
            throw std::invalid_argument("PairBorn: A must be ≥ 0");
        if (!(c.C >= 0.0))
            throw std::invalid_argument("PairBorn: C must be ≥ 0");
        tmax = std::max(tmax, std::max(c.type_i, c.type_j));
    }
    n_types_ = tmax + 1;
    table_.assign(static_cast<std::size_t>(n_types_)
                      * static_cast<std::size_t>(n_types_),
                  Entry{});

    for (const auto& c : coeffs) {
        Entry e;
        e.A = c.A;
        e.inv_rho = 1.0 / c.rho;
        e.sigma = c.sigma;
        e.C = c.C;
        e.D = c.D;
        e.v_shift = shift_of(c.A, e.inv_rho, c.sigma, c.C, c.D, cutoff);
        e.set = true;
        const std::size_t ij = static_cast<std::size_t>(c.type_i)
                                   * static_cast<std::size_t>(n_types_)
                               + static_cast<std::size_t>(c.type_j);
        const std::size_t ji = static_cast<std::size_t>(c.type_j)
                                   * static_cast<std::size_t>(n_types_)
                               + static_cast<std::size_t>(c.type_i);
        table_[ij] = e;
        table_[ji] = e;
    }
}

const PairBorn::Entry& PairBorn::entry_for_(int ti, int tj) const {
    static const Entry kUnset{};
    if (ti < 0 || tj < 0 || ti >= n_types_ || tj >= n_types_)
        return kUnset;
    return table_[static_cast<std::size_t>(ti)
                      * static_cast<std::size_t>(n_types_)
                  + static_cast<std::size_t>(tj)];
}

void PairBorn::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const int n = static_cast<int>(f.nlocal());

    double energy = 0.0;
    // Pair virial, Voigt xx yy zz xy xz yz [eV] — LAMMPS sign (Σ Δr ⊗ f,
    // positive under repulsion). Without it a Born–Mayer–Huggins salt run at
    // constant pressure is driven by a pressure that is missing its entire
    // short-range repulsion, and the barostat collapses the cell.
    md::Voigt6 vir{};
    const StorageReal* __restrict x = f.span(kAtom_x);
    const StorageReal* __restrict y = f.span(kAtom_y);
    const StorageReal* __restrict z = f.span(kAtom_z);
    AccReal* __restrict fx = f.span(kAtom_fx);
    AccReal* __restrict fy = f.span(kAtom_fy);
    AccReal* __restrict fz = f.span(kAtom_fz);

    // Δ = r_i − r_j is `-(dx,dy,dz)` here (callers pass x_j − x_i) and the
    // force on i is `-fpair·(dx,dy,dz)` = `fpair·Δ`, so every virial component
    // is `fpair·Δ_a·Δ_b = fpair·d_a·d_b` — independent of the sign convention
    // the caller used for the separation.
    auto add_virial = [&](double fpair, double dx, double dy, double dz) {
        vir.v[0] += fpair * dx * dx;
        vir.v[1] += fpair * dy * dy;
        vir.v[2] += fpair * dz * dz;
        vir.v[3] += fpair * dx * dy;
        vir.v[4] += fpair * dx * dz;
        vir.v[5] += fpair * dy * dz;
    };

    auto born_pair = [&](int i, int j, double dx, double dy, double dz,
                         double A, double inv_rho, double sigma, double C,
                         double D, double v_shift) {
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 >= cutoff_sq_ || r2 == 0.0)
            return;
        const double r = std::sqrt(r2);
        const double r6 = r2 * r2 * r2;
        const double r8 = r6 * r2;
        const double expterm = std::exp((sigma - r) * inv_rho);
        const double v = A * expterm - C / r6 + D / r8 - v_shift;
        const double dUdr =
            -A * inv_rho * expterm + 6.0 * C / (r6 * r) - 8.0 * D / (r8 * r);
        const double fpair = -dUdr / r;
        fx[i] -= fpair * dx;
        fy[i] -= fpair * dy;
        fz[i] -= fpair * dz;
        fx[j] += fpair * dx;
        fy[j] += fpair * dy;
        fz[j] += fpair * dz;
        energy += v;
        add_virial(fpair, dx, dy, dz);
    };

    if (ctx.runtime.md.pair_list && !multi_) {
        const auto& pairs = *ctx.runtime.md.pair_list;
        const double inv_rho = inv_rho_;
        for (int ii = 0; ii < pairs.inum; ++ii) {
            const int i = pairs.ilist[static_cast<std::size_t>(ii)];
            const int jnum = pairs.numneigh[static_cast<std::size_t>(i)];
            const int off = pairs.firstneigh[static_cast<std::size_t>(i)];
            for (int jj = 0; jj < jnum; ++jj) {
                const int j =
                    pairs.neighbors[static_cast<std::size_t>(off + jj)];
                if (j <= i)
                    continue;
                double dx = x[j] - x[i];
                double dy = y[j] - y[i];
                double dz = z[j] - z[i];
                f.box.min_image(dx, dy, dz);
                born_pair(i, j, dx, dy, dz, A_, inv_rho, sigma_, C_, D_,
                          v_shift_);
            }
        }
        f.meta.get(kEnergy) += energy;
        f.meta.get(kEnergyPair) += energy;
        vir.flush_to(f.meta.get(kStress));
        return;
    }

    if (!multi_) {
        const double inv_rho = inv_rho_;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                double dx = x[j] - x[i];
                double dy = y[j] - y[i];
                double dz = z[j] - z[i];
                f.box.min_image(dx, dy, dz);
                born_pair(i, j, dx, dy, dz, A_, inv_rho, sigma_, C_, D_,
                          v_shift_);
            }
        }
    } else {
        const int* __restrict type = f.span(kAtom_type);
        for (int i = 0; i < n; ++i) {
            const int ti = type[i];
            for (int j = i + 1; j < n; ++j) {
                const int tj = type[j];
                const Entry& e = entry_for_(ti, tj);
                if (!e.set)
                    continue;

                double dx = x[j] - x[i];
                double dy = y[j] - y[i];
                double dz = z[j] - z[i];
                f.box.min_image(dx, dy, dz);
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= cutoff_sq_ || r2 == 0.0)
                    continue;

                const double r = std::sqrt(r2);
                const double r6 = r2 * r2 * r2;
                const double r8 = r6 * r2;
                const double expterm = std::exp((e.sigma - r) * e.inv_rho);
                const double v =
                    e.A * expterm - e.C / r6 + e.D / r8 - e.v_shift;
                const double dUdr = -e.A * e.inv_rho * expterm
                                    + 6.0 * e.C / (r6 * r)
                                    - 8.0 * e.D / (r8 * r);
                const double fpair = -dUdr / r;

                fx[i] -= fpair * dx;
                fy[i] -= fpair * dy;
                fz[i] -= fpair * dz;
                fx[j] += fpair * dx;
                fy[j] += fpair * dy;
                fz[j] += fpair * dz;
                energy += v;
                add_virial(fpair, dx, dy, dz);
            }
        }
    }
    f.meta.get(kEnergy) += energy;
    f.meta.get(kEnergyPair) += energy;
    vir.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV
