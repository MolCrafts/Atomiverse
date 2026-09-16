/// @file cpu/potentials/bond_fene.cpp
#include "atomiverse/cpu/potentials/bond_fene.h"
#include "atomiverse/cpu/potentials/resolve_view.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV {
namespace cpu {

BondFene::BondFene(std::vector<double> K, std::vector<double> R0)
    : K_(std::move(K)), R0_(std::move(R0)) {
    if (K_.size() != R0_.size() || K_.empty())
        throw std::invalid_argument(
            "BondFene: K and R0 must match and be non-empty");
    for (std::size_t t = 0; t < K_.size(); ++t) {
        if (!(K_[t] > 0.0) || !(R0_[t] > 0.0))
            throw std::invalid_argument("BondFene: K and R0 must be > 0");
    }
}

void BondFene::build(const BuildContext& ctx) {
    int n_threads = 1;
#ifdef _OPENMP
    n_threads = omp_get_max_threads();
#endif
    scratch_.build(n_threads, ctx.frame.capacity());
}

void BondFene::compute(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    const auto num_bond = f.span(kAtom_num_bond);
    const auto bond_atom = f.span(kAtom_bond_atom);
    const auto bond_type = f.span(kAtom_bond_type);

    const md::ResolveView rv(ctx);
    const int nlocal = rv.n_local();
    const int n_force = rv.n_force();

    if (cache_.stale(ctx)) {
        cache_.flat.clear();
        cache_.dropped_at_build = 0;
        for (int i = 0; i < nlocal; ++i) {
            const int nb = num_bond[i];
            for (int m = 0; m < nb; ++m) {
                const int j = rv.resolve(bond_atom(i, m));
                if (j < 0) {
                    ++cache_.dropped_at_build;
                    continue;
                }
                cache_.flat.push_back(i);
                cache_.flat.push_back(j);
                cache_.flat.push_back(bond_type(i, m));
            }
        }
        cache_.mark_built(ctx);
        drops_.dropped += cache_.dropped_at_build;
    }
    const int n_tuples = static_cast<int>(cache_.flat.size()) / 3;
    if (n_tuples == 0)
        return;
    const int* __restrict tup = cache_.flat.data();

    double energy_total = 0;
    md::Voigt6 vir;
    int bad_type = 0;
    int overstretched = 0;

    const StorageReal* __restrict x_ptr = f.span(kAtom_x);
    const StorageReal* __restrict y_ptr = f.span(kAtom_y);
    const StorageReal* __restrict z_ptr = f.span(kAtom_z);

    scratch_.bind(f);

#pragma omp parallel if (scratch_.n_threads() > 1) reduction(+ : energy_total) \
    reduction(+ : bad_type) reduction(+ : overstretched)
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const md::ThreadForce tf = scratch_.begin_thread(tid, n_force);
        md::Voigt6 vir_local;

#pragma omp for schedule(static)
        for (int t = 0; t < n_tuples; ++t) {
            const int i = tup[3 * t + 0];
            const int j = tup[3 * t + 1];
            const int type = tup[3 * t + 2];
            if (type < 0 || static_cast<std::size_t>(type) >= K_.size()) {
                ++bad_type;
                continue;
            }

            double dx = x_ptr[i] - x_ptr[j];
            double dy = y_ptr[i] - y_ptr[j];
            double dz = z_ptr[i] - z_ptr[j];
            f.box.min_image(dx, dy, dz);

            const double r2 = dx * dx + dy * dy + dz * dz;
            const double r = std::sqrt(r2);
            const double R0 = R0_[type];
            const double K = K_[type];
            if (!(r < R0)) {
                ++overstretched;
                continue;
            }

            const double rr = r / R0;
            const double rr2 = rr * rr;
            const double one_m = 1.0 - rr2;
            // U = −½ K R0² ln(1 − (r/R0)²)
            energy_total += -0.5 * K * R0 * R0 * std::log(one_m);
            // dU/dr = K r / (1 − (r/R0)²)
            const double dUdr = K * r / one_m;
            const double fpair = -dUdr / r;

            const double fx_val = fpair * dx;
            const double fy_val = fpair * dy;
            const double fz_val = fpair * dz;

            tf.fx[i] += fx_val;
            tf.fx[j] -= fx_val;
            tf.fy[i] += fy_val;
            tf.fy[j] -= fy_val;
            tf.fz[i] += fz_val;
            tf.fz[j] -= fz_val;

            vir_local.add_pair(dx, dy, dz, fx_val, fy_val, fz_val);
        }
#pragma omp critical
        vir += vir_local;
    }

    if (bad_type > 0)
        throw std::runtime_error("BondFene: bond type out of range");
    if (overstretched > 0)
        throw std::runtime_error("BondFene: bond length ≥ R0");

    scratch_.reduce_into(n_force);

    f.meta.get(kEnergy) += energy_total;
    f.meta.get(kEnergyBond) += energy_total;
    vir.flush_to(f.meta.get(kStress));

    drops_.evaluated += static_cast<std::uint64_t>(n_tuples);
}

void BondFene::destroy(const DestroyContext& /*ctx*/) {
    drops_.report("BondFene");
    drops_.reset();
    cache_.clear();
    scratch_.clear();
}

} // namespace cpu
} // namespace ATV
