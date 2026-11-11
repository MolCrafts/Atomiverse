/// @file cpu/potentials/pair_born.cpp
#include "atomiverse/cpu/potentials/pair_born.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_loop.h"

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

    // Born–Mayer–Huggins pair, LAMMPS `pair_style born`, shifted to zero at
    // the cutoff. The virial is LAMMPS-signed (Σ Δr ⊗ f, positive under
    // repulsion): without it a salt run at constant pressure is driven by a
    // pressure missing its entire short-range repulsion, and the barostat
    // collapses the cell.
    struct Kernel {
        const PairBorn* self;
        const int* type; ///< per-type form only
        double rc_sq;

        double cutoff_sq() const { return rc_sq; }
        int row(int i) const { return type ? type[i] : 0; }

        md::PairTerm operator()(int ti, const md::PairSite& p,
                                const md::PairChunk&) const {
            Entry single;
            const Entry* e = &single;
            if (type) {
                e = &self->entry_for_(ti, type[p.j]);
                if (!e->set)
                    return {};
            } else {
                single = {self->A_, self->inv_rho_, self->sigma_, self->C_,
                          self->D_, self->v_shift_, true};
            }
            const double r2 = p.r2;
            const double r = std::sqrt(r2);
            const double r6 = r2 * r2 * r2;
            const double r8 = r6 * r2;
            const double expterm = std::exp((e->sigma - r) * e->inv_rho);
            const double v =
                e->A * expterm - e->C / r6 + e->D / r8 - e->v_shift;
            const double dUdr = -e->A * e->inv_rho * expterm
                                + 6.0 * e->C / (r6 * r) - 8.0 * e->D / (r8 * r);
            // F_i = −dU/dr · r̂ = (−dU/dr / r)·(r_i − r_j).
            return {-dUdr / r, v};
        }
    };
    const Kernel kernel{this, multi_ ? f.span(kAtom_type).data() : nullptr,
                        cutoff_sq_};
    // The per-type table walks every owned pair (minimum image) as it always
    // has; the single-type form takes the published half list when there is
    // one.
    const md::PairTotals sums = multi_
                                    ? loop_.all_pairs(ctx, kernel, "PairBorn")
                                    : loop_.run(ctx, kernel, "PairBorn");
    f.meta.get(kEnergy) += sums.energy;
    f.meta.get(kEnergyPair) += sums.energy;
    if (ctx.stress_requested)
        sums.virial.flush_to(f.meta.get(kStress));
}

} // namespace cpu
} // namespace ATV
