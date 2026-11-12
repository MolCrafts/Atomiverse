/// @file cpu/potentials/pair_born.cpp
#include "atomiverse/cpu/potentials/pair_born.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include <xsimd/xsimd.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/neighbor/pair_list.h"
#include "atomiverse/cpu/potentials/pair_curvature.h"
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

/// Homogeneous-tail prefactors of one ordered type pair (class comment):
/// energy 2π N_i N_j ∫ r² U and virial −(2π/3) N_i N_j ∫ r³ U' over [r_c, ∞).
struct BornTail {
    double energy = 0.0; ///< [eV·Å³]
    double virial = 0.0; ///< [eV·Å³]
};

BornTail born_tail(double A, double inv_rho, double sigma, double C, double D,
                   double rc, double n_i, double n_j) {
    const double rho = 1.0 / inv_rho;
    const double rho2 = rho * rho;
    const double rc2 = rc * rc;
    const double rc3 = rc2 * rc;
    const double rc5 = rc3 * rc2;
    const double ex = A * std::exp((sigma - rc) * inv_rho);
    const double int_r2_u = ex * rho * (rc2 + 2.0 * rho * rc + 2.0 * rho2)
                            - C / (3.0 * rc3) + D / (5.0 * rc5);
    const double int_r3_du =
        -ex * (rc3 + 3.0 * rho * rc2 + 6.0 * rho2 * rc + 6.0 * rho2 * rho)
        + 2.0 * C / rc3 - 8.0 * D / (5.0 * rc5);
    const double pairs = n_i * n_j;
    return {2.0 * units::pi * pairs * int_r2_u,
            -(2.0 / 3.0) * units::pi * pairs * int_r3_du};
}

} // namespace

PairBorn& PairBorn::set_energy_shift(bool on) & {
    energy_shift_ = on;
    return *this;
}

PairBorn PairBorn::set_energy_shift(bool on) && {
    energy_shift_ = on;
    return std::move(*this);
}

PairBorn& PairBorn::set_tail_correction(bool on) & {
    tail_ = on;
    tail_ready_ = false;
    return *this;
}

PairBorn PairBorn::set_tail_correction(bool on) && {
    set_tail_correction(on);
    return std::move(*this);
}

// LAMMPS PairBorn::init_one tail sum over ordered type pairs with the global
// type counts. PairBorn requires the full frame, so the owned rows are every
// atom and no reduction is needed.
void PairBorn::prepare_tail_(const Frame& frame) {
    const int n = frame.nlocal();
    const double rc = std::sqrt(cutoff_sq_);
    BornTail sum;
    if (!multi_) {
        const double count = static_cast<double>(n);
        sum = born_tail(A_, inv_rho_, sigma_, C_, D_, rc, count, count);
    } else {
        const auto type = frame.span(kAtom_type);
        std::vector<double> count(static_cast<std::size_t>(n_types_), 0.0);
        for (int i = 0; i < n; ++i) {
            const int t = type[i];
            if (t >= 0 && t < n_types_)
                count[static_cast<std::size_t>(t)] += 1.0;
        }
        for (int ti = 0; ti < n_types_; ++ti)
            for (int tj = 0; tj < n_types_; ++tj) {
                const Entry& e = entry_for_(ti, tj);
                if (!e.set)
                    continue;
                const BornTail t =
                    born_tail(e.A, e.inv_rho, e.sigma, e.C, e.D, rc,
                              count[static_cast<std::size_t>(ti)],
                              count[static_cast<std::size_t>(tj)]);
                sum.energy += t.energy;
                sum.virial += t.virial;
            }
    }
    etail_pref_ = sum.energy;
    ptail_pref_ = sum.virial;
    tail_ready_ = true;
}

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

const md::PairList* PairBorn::covering_list_(const LaunchContext& ctx) const {
    const md::PairList* list = ctx.runtime.md.pair_list;
    return list != nullptr && list->cutoff * list->cutoff >= cutoff_sq_
               ? list
               : nullptr;
}

void PairBorn::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;

    // Born–Mayer–Huggins pair, LAMMPS `pair_style born`, shifted to zero at
    // the cutoff. The virial is LAMMPS-signed (Σ Δr ⊗ f, positive under
    // repulsion): without it a salt run at constant pressure is driven by a
    // pressure missing its entire short-range repulsion, and the barostat
    // collapses the cell.
    struct Kernel {
        const Entry* table; ///< dense n_types × n_types; null: single type
        const int* type;    ///< per-type form only
        int n_types;
        Entry single; ///< the single-type coefficients
        double rc_sq;
        double shift; ///< 1 with the energy shift, 0 without

        /// Row i's slice of the table (null: type out of range, no pairs),
        /// or the single-type entry.
        struct Row {
            const Entry* coeff;
        };

        double cutoff_sq() const { return rc_sq; }
        Row row(int i) const {
            if (table == nullptr)
                return {&single};
            const int ti = type[i];
            return {ti >= 0 && ti < n_types
                        ? table
                              + static_cast<std::size_t>(ti)
                                    * static_cast<std::size_t>(n_types)
                        : nullptr};
        }
        bool skip(const Row& ri) const { return ri.coeff == nullptr; }

        /// The coefficients of the pair (row i, partner j), or null when
        /// the pair does not interact.
        const Entry* coeff(const Row& ri, int j) const {
            if (table == nullptr)
                return ri.coeff;
            const int tj = type[j];
            if (tj < 0 || tj >= n_types || !ri.coeff[tj].set)
                return nullptr;
            return ri.coeff + tj;
        }

        /// r and e^{(σ−r)/ρ} of every survivor into chunk.a / chunk.b: the
        /// root and the exponential one xsimd batch at a time, the per-pair
        /// exponent in between (`r2` is padded with 1 past `n`).
        void prepare(const Row& ri, const double* r2, const int* j, int n,
                     md::PairChunk& c) const {
            using B = xsimd::batch<double>;
            constexpr int W = static_cast<int>(B::size);
            for (int k = 0; k < n; k += W)
                xsimd::sqrt(B::load_unaligned(r2 + k)).store_aligned(c.a + k);
            for (int k = 0; k < n; ++k) {
                const Entry* e = coeff(ri, j[k]);
                c.b[k] = e ? (e->sigma - c.a[k]) * e->inv_rho : 0.0;
            }
            for (int k = n; k % W != 0; ++k)
                c.b[k] = 0.0;
            for (int k = 0; k < n; k += W)
                xsimd::exp(B::load_aligned(c.b + k)).store_aligned(c.b + k);
        }

        md::PairTerm operator()(const Row& ri, const md::PairSite& p,
                                const md::PairChunk& c) const {
            const Entry* e = coeff(ri, p.j);
            if (e == nullptr)
                return {};
            // One division: 1/r = r·r⁻², as LAMMPS `born` (r2inv).
            const double r = c.a[p.k];
            const double inv_r2 = 1.0 / p.r2;
            const double inv_r6 = inv_r2 * inv_r2 * inv_r2;
            const double inv_r8 = inv_r6 * inv_r2;
            const double ae = e->A * c.b[p.k];
            const double v =
                ae - e->C * inv_r6 + e->D * inv_r8 - shift * e->v_shift;
            // F_i = −dU/dr · r̂ = (−dU/dr / r)·(r_i − r_j), with
            // −dU/dr / r = (A/ρ) e^{(σ−r)/ρ} / r − 6C/r⁸ + 8D/r¹⁰.
            const double fpair = ae * e->inv_rho * (r * inv_r2)
                                 - 6.0 * e->C * inv_r8
                                 + 8.0 * e->D * inv_r8 * inv_r2;
            return {fpair, v};
        }
    };
    const Kernel kernel{multi_ ? table_.data() : nullptr,
                        multi_ ? f.span(kAtom_type).data() : nullptr,
                        n_types_,
                        {A_, inv_rho_, sigma_, C_, D_, v_shift_, true},
                        cutoff_sq_,
                        energy_shift_ ? 1.0 : 0.0};
    const md::PairList* list = covering_list_(ctx);
    const md::PairTotals sums =
        list ? loop_.run(ctx, md::PairRows::of(*list), kernel)
             : loop_.all_pairs(ctx, kernel, "PairBorn");
    f.meta.get(kEnergy) += sums.energy;
    f.meta.get(kEnergyPair) += sums.energy;
    if (ctx.stress_requested)
        sums.virial.flush_to(f.meta.get(kStress));

    // Tail (LAMMPS pair_modify tail yes): constant prefactors over the fixed
    // type counts, scaled by the current volume because barostats move it.
    if (tail_) {
        if (!tail_ready_)
            prepare_tail_(f);
        const double vol = f.box.volume();
        f.meta.get(kEnergy) += etail_pref_ / vol;
        f.meta.get(kEnergyPair) += etail_pref_ / vol;
        if (ctx.stress_requested)
            for (int d = 0; d < 3; ++d)
                f.meta.get(kStress)[d] += ptail_pref_ / vol;
    }
}

// The pairs of compute(): the covering half list, else every owned pair at
// minimum image.
void PairBorn::add_curvature(const LaunchContext& ctx,
                             const CurvatureRows& rows) {
    const int* type = multi_ ? ctx.frame.span(kAtom_type).data() : nullptr;
    const bool use_list = covering_list_(ctx) != nullptr;
    const Entry single{A_, inv_rho_, sigma_, C_, D_, v_shift_, true};
    md::for_each_curvature_pair(
        ctx, cutoff_sq_, use_list, "PairBorn",
        [&](int i, int j, int, double dx, double dy, double dz, double r2) {
            const Entry& e = type ? entry_for_(type[i], type[j]) : single;
            if (!e.set)
                return;
            const double r = std::sqrt(r2);
            const double inv_r = 1.0 / r;
            const double inv_r2 = inv_r * inv_r;
            const double inv_r6 = inv_r2 * inv_r2 * inv_r2;
            const double inv_r8 = inv_r6 * inv_r2;
            const double ae = e.A * std::exp((e.sigma - r) * e.inv_rho);
            const double k = e.inv_rho;
            const double d1 = -ae * k + 6.0 * e.C * inv_r6 * inv_r
                              - 8.0 * e.D * inv_r8 * inv_r;
            const double d2 =
                ae * k * k - 42.0 * e.C * inv_r8 + 72.0 * e.D * inv_r8 * inv_r2;
            const double d3 = -ae * k * k * k + 336.0 * e.C * inv_r8 * inv_r
                              - 720.0 * e.D * inv_r8 * inv_r2 * inv_r;
            md::RadialCurvature c;
            c.d1 = d1;
            c.d2 = d2;
            c.lap = d2 + 2.0 * d1 * inv_r;
            c.dlap = d3 + 2.0 * d2 * inv_r - 2.0 * d1 * inv_r2;
            md::add_pair_curvature(rows, i, j, dx, dy, dz, r, c);
        });
}

} // namespace cpu
} // namespace ATV
