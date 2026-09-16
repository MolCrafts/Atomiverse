/// @file src/cpu/potentials/pim_xl_dipole.cpp
/// @brief Extended-Lagrangian PIM dipoles (iEL/SCF) — warm-started SCC +
///        Niklasson auxiliary propagation (see header for references).
#include "atomiverse/cpu/potentials/pim_xl_dipole.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include "atomiverse/cpu/frame/frame.h"

namespace ATV {
namespace cpu {

PimXlDipole::PimXlDipole(double cutoff, std::vector<double> alpha_by_type,
                         double thole_a, int xl_order, double pol_tol,
                         int pol_max_iter)
    : pim_(cutoff, std::move(alpha_by_type), thole_a, pol_tol, pol_max_iter),
      xl_(xl_order) {}

void PimXlDipole::build(const BuildContext& ctx) { pim_.build(ctx); }

void PimXlDipole::reset() {
    xl_.reset();
    aux_err_ = 0.0;
    last_warm_ = false;
    n_prev_ = -1;
}

void PimXlDipole::launch(const LaunchContext& ctx) {
    const long n = static_cast<long>(ctx.frame.nlocal());
    if (n != n_prev_) {
        // Atom count changed (first launch, deposition, MPI migration, …):
        // the auxiliary history no longer matches — cold restart.
        xl_.reset();
        n_prev_ = n;
    }

    aux_err_ = 0.0;
    last_warm_ = false;

    // Warm start: hand the auxiliary dipoles to the SCC as its guess. Keep a
    // copy — `PimInducedDipole::launch` consumes `mu_guess`, and we still
    // need the guess afterwards for the aux-vs-SCC deviation metric.
    std::vector<double> guess;
    if (n > 0 && xl_.ready()) {
        const auto& aux = xl_.aux(); // n×3
        guess.resize(static_cast<std::size_t>(3 * n));
        for (long i = 0; i < n; ++i)
            for (long c = 0; c < 3; ++c)
                guess[static_cast<std::size_t>(3 * i + c)] =
                    aux(static_cast<std::size_t>(i),
                        static_cast<std::size_t>(c));
        pim_.mu_guess = guess;
        last_warm_ = true;
    }

    // Converge the SCC (energy + Hellmann–Feynman forces onto Frame).
    pim_.launch(ctx);
    if (n <= 0)
        return;

    // Advance the auxiliary dynamics with this step's converged dipoles:
    //   μ_aux(n+1) = 2 μ_aux(n) − μ_aux(n−1) + κ (μ_scc − μ_aux(n)) + diss.
    const std::vector<double>& mu = pim_.last_mu();
    linalg::RealMatrix d = linalg::RealMatrix::from_shape(
        {static_cast<std::size_t>(n), std::size_t{3}});
    for (long i = 0; i < n; ++i)
        for (long c = 0; c < 3; ++c)
            d(static_cast<std::size_t>(i), static_cast<std::size_t>(c)) =
                mu[static_cast<std::size_t>(3 * i + c)];

    if (last_warm_) {
        double err = 0.0;
        for (std::size_t k = 0; k < guess.size(); ++k)
            err = std::max(err, std::fabs(guess[k] - mu[k]));
        aux_err_ = err;
    }

    xl_.advance(d);
}

} // namespace cpu
} // namespace ATV
