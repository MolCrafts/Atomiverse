/// @file cpu/planewave/internal/smearing.cpp
/// @brief Metallic occupation smearing implementations.
#include "atomiverse/cpu/planewave/internal/smearing.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

namespace {

constexpr AccReal kSqrtPi = AccReal(1.77245385090551602729);

// Gaussian-weighted, coefficient-normalized Hermite recurrence. Keeping
// A_n H_m exp(-x^2) together avoids factorial/polynomial overflow. At |x|>40
// every relevant double-precision tail is zero; never cap the exponential
// at exp(-200) and then multiply a polynomial that grows without bound.
struct MPTerms {
    AccReal correction = 0;
    AccReal even_last = 0;
    AccReal delta = 0;
    AccReal delta_odd = 0;
};

MPTerms mp_terms(AccReal x, int order) {
    MPTerms out;
    if (std::abs(x) > AccReal(40))
        return out;
    AccReal even = std::exp(-x * x) / kSqrtPi;
    AccReal odd = 0;
    out.delta = even;
    out.delta_odd = AccReal(2) * x * even;
    for (int n = 1; n <= order; ++n) {
        // odd = A_n H_(2n-1) exp; even = A_n H_(2n) exp.
        const AccReal dn = AccReal(n);
        odd = -x * even / (AccReal(2) * dn) + (dn - AccReal(1)) / dn * odd;
        even = AccReal(2) * x * odd + (AccReal(1) - AccReal(0.5) / dn) * even;
        out.correction += odd;
        out.delta += even;
        out.delta_odd += AccReal(2) * x * even - AccReal(4) * dn * odd;
    }
    out.even_last = even;
    return out;
}

AccReal fd_entropy(AccReal f) {
    // Exact endpoint limits, including f rounded to one. 1-1e-30 == 1
    // in double, so an epsilon clamp does not protect 0*log(0).
    if (f <= AccReal(0) || f >= AccReal(1))
        return AccReal(0);
    return -(f * std::log(f) + (AccReal(1) - f) * std::log1p(-f));
}

} // namespace

AccReal smear_f_spinless(AccReal x, SmearKind kind, int mp_order) {
    switch (kind) {
    case SmearKind::fermi_dirac: {
        // f = 1 / (1 + exp(x)); stable limits
        if (x > AccReal(36))
            return AccReal(0);
        if (x < AccReal(-36))
            return AccReal(1);
        return AccReal(1) / (AccReal(1) + std::exp(x));
    }
    case SmearKind::gaussian: {
        return AccReal(0.5) * std::erfc(static_cast<double>(x));
    }
    case SmearKind::methfessel_paxton: {
        if (mp_order < 1)
            throw std::invalid_argument("smear_f_spinless: mp_order must be >= "
                                        "1 for Methfessel-Paxton");
        // MP occupations are quasi-occupations: their signed tails may
        // lie outside [0,1]. Clamping destroys s'(x)=x f'(x).
        return AccReal(0.5) * std::erfc(static_cast<double>(x))
               + mp_terms(x, mp_order).correction;
    }
    }
    throw std::invalid_argument("smear_f_spinless: unknown kind");
}

AccReal smear_f_band(AccReal eig, AccReal mu, AccReal degauss, SmearKind kind,
                     int mp_order) {
    if (!(degauss > AccReal(0)))
        throw std::invalid_argument("smear_f_band: degauss must be > 0");
    const AccReal x = (eig - mu) / degauss;
    return AccReal(2) * smear_f_spinless(x, kind, mp_order);
}

AccReal smear_f_spinor(AccReal eig, AccReal mu, AccReal degauss, SmearKind kind,
                       int mp_order) {
    if (!(degauss > AccReal(0)))
        throw std::invalid_argument("smear_f_spinor: degauss must be > 0");
    const AccReal x = (eig - mu) / degauss;
    return smear_f_spinless(x, kind, mp_order);
}

AccReal smear_entropy_band(AccReal eig, AccReal mu, AccReal degauss,
                           SmearKind kind, int mp_order) {
    if (!(degauss > AccReal(0)))
        throw std::invalid_argument("smear_entropy_band: degauss must be > 0");
    const AccReal x = (eig - mu) / degauss;
    switch (kind) {
    case SmearKind::fermi_dirac: {
        return AccReal(2) * fd_entropy(smear_f_spinless(x, kind, mp_order));
    }
    case SmearKind::gaussian: {
        return std::abs(x) > AccReal(40) ? AccReal(0)
                                         : std::exp(-x * x) / kSqrtPi;
    }
    case SmearKind::methfessel_paxton: {
        if (mp_order < 1)
            throw std::invalid_argument(
                "smear_entropy_band: mp_order must be >= 1");
        // s_spinless=.5*A_N*H_(2N)*exp(-x^2); band degeneracy is two.
        // This is -2*QE w1gauss(-x,N), and satisfies s'=x f'.
        return mp_terms(x, mp_order).even_last;
    }
    }
    throw std::invalid_argument("smear_entropy_band: unknown kind");
}

AccReal smear_entropy_spinor(AccReal eig, AccReal mu, AccReal degauss,
                             SmearKind kind, int mp_order) {
    // Half the closed-shell band entropy (one spin channel).
    return AccReal(0.5) * smear_entropy_band(eig, mu, degauss, kind, mp_order);
}

namespace {

AccReal nelec_of_mu(const std::vector<AccReal>& evals,
                    const std::vector<AccReal>& weights, AccReal mu,
                    AccReal degauss, SmearKind kind, int mp_order,
                    AccReal spin_degen) {
    AccReal n = 0;
    const std::size_t m = evals.size();
    for (std::size_t i = 0; i < m; ++i)
        n += weights[i] * spin_degen
             * smear_f_spinless((evals[i] - mu) / degauss, kind, mp_order);
    return n;
}

} // namespace

AccReal find_fermi_level(const std::vector<AccReal>& evals,
                         const std::vector<AccReal>& weights, AccReal n_elec,
                         AccReal degauss, SmearKind kind, int mp_order,
                         AccReal spin_degen) {
    if (evals.empty())
        throw std::invalid_argument("find_fermi_level: empty spectrum");
    if (evals.size() != weights.size())
        throw std::invalid_argument(
            "find_fermi_level: evals/weights size mismatch");
    if (!(degauss > AccReal(0)))
        throw std::invalid_argument("find_fermi_level: degauss must be > 0");
    if (!(n_elec > AccReal(0)))
        throw std::invalid_argument("find_fermi_level: n_elec must be > 0");
    if (!(spin_degen > AccReal(0)))
        throw std::invalid_argument("find_fermi_level: spin_degen must be > 0");

    AccReal emin = evals[0];
    AccReal emax = evals[0];
    for (AccReal e : evals) {
        emin = std::min(emin, e);
        emax = std::max(emax, e);
    }
    AccReal capacity = 0;
    for (std::size_t i = 0; i < evals.size(); ++i) {
        if (!std::isfinite(evals[i]) || !std::isfinite(weights[i])
            || weights[i] < AccReal(0))
            throw std::invalid_argument(
                "find_fermi_level: spectrum/weights must be finite and weights "
                "nonnegative");
        capacity += weights[i] * spin_degen;
    }
    if (n_elec > capacity)
        throw std::invalid_argument(
            "find_fermi_level: electron count exceeds weighted band capacity");
    if (kind == SmearKind::methfessel_paxton && mp_order < 1)
        throw std::invalid_argument("find_fermi_level: mp_order must be >= 1");

    // QE efermig uses an absolute 1e-10 electron-count stopping criterion.
    constexpr AccReal count_tol = AccReal(1e-10);
    AccReal lo = emin - AccReal(10) * degauss;
    AccReal hi = emax + AccReal(10) * degauss;
    const auto count = [&](AccReal mu, SmearKind mode) {
        return nelec_of_mu(evals, weights, mu, degauss, mode, mp_order,
                           spin_degen);
    };
    const auto bisect = [&](SmearKind mode) {
        AccReal lower = lo, upper = hi;
        for (int expand = 0;
             expand < 20
             && (count(lower, mode) > n_elec || count(upper, mode) < n_elec);
             ++expand) {
            lower -= AccReal(20) * degauss;
            upper += AccReal(20) * degauss;
        }
        if (count(lower, mode) > n_elec || count(upper, mode) < n_elec)
            throw std::runtime_error(
                "find_fermi_level: cannot bracket electron count");
        for (int it = 0; it < 200; ++it) {
            const AccReal mid = AccReal(0.5) * lower + AccReal(0.5) * upper;
            const AccReal residual = count(mid, mode) - n_elec;
            // Stop on N, including an exactly filled gap plateau. Shrinking
            // only the energy bracket drives mu to a gap edge and creates
            // artificial entropy even when N was already correct.
            if (std::abs(residual) <= count_tol)
                return mid;
            if (mid == lower || mid == upper)
                break;
            if (residual < AccReal(0))
                lower = mid;
            else
                upper = mid;
        }
        throw std::runtime_error(
            "find_fermi_level: electron count did not converge");
    };
    if (kind != SmearKind::methfessel_paxton)
        return bisect(kind);

    // MP N(mu) need not be monotone. QE first obtains a Gaussian root,
    // accepts an already-correct plateau, then minimizes [N(mu)-Ntarget]^2.
    AccReal mu = bisect(SmearKind::gaussian);
    AccReal residual = count(mu, kind) - n_elec;
    for (int it = 0; it < 100; ++it) {
        if (std::abs(residual) <= count_tol)
            return mu;
        AccReal first = 0, second = 0;
        for (std::size_t i = 0; i < evals.size(); ++i) {
            const auto terms = mp_terms((evals[i] - mu) / degauss, mp_order);
            const AccReal factor = weights[i] * spin_degen;
            first += factor * terms.delta / degauss;
            second += factor * terms.delta_odd / (degauss * degauss);
        }
        const AccReal denominator = std::abs(first * first + residual * second);
        if (!(denominator > AccReal(0)) || !std::isfinite(denominator))
            break;
        AccReal step = residual * first / denominator;
        bool accepted = false;
        for (int backtrack = 0; backtrack < 30; ++backtrack) {
            const AccReal trial = mu - step;
            if (std::isfinite(trial) && trial >= lo && trial <= hi) {
                const AccReal r = count(trial, kind) - n_elec;
                if (std::abs(r) < std::abs(residual)) {
                    mu = trial;
                    residual = r;
                    accepted = true;
                    break;
                }
            }
            step *= AccReal(0.5);
        }
        if (!accepted)
            break;
    }
    // Sign-bracket bisection still guarantees a root of continuous MP N(mu)
    // when minimization fails; never return a spectrum midpoint with bad N.
    return bisect(kind);
}

SmearFill fill_occupations(const std::vector<AccReal>& evals,
                           const std::vector<AccReal>& weights, AccReal mu,
                           AccReal degauss, SmearKind kind, int mp_order,
                           AccReal spin_degen) {
    if (evals.size() != weights.size())
        throw std::invalid_argument(
            "fill_occupations: evals/weights size mismatch");
    if (!(degauss > AccReal(0)) || !(spin_degen > AccReal(0)))
        throw std::invalid_argument(
            "fill_occupations: width and spin degeneracy must be positive");
    SmearFill out;
    out.occ.resize(evals.size());
    for (std::size_t i = 0; i < evals.size(); ++i) {
        out.occ[i] =
            spin_degen
            * smear_f_spinless((evals[i] - mu) / degauss, kind, mp_order);
        out.n_occ += weights[i] * out.occ[i];
        out.entropy +=
            weights[i] * spin_degen
            * smear_entropy_spinor(evals[i], mu, degauss, kind, mp_order);
    }
    return out;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
