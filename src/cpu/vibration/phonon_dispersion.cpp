/// @file src/cpu/vibration/phonon_dispersion.cpp
/// @brief Fourier dynamical matrix + \(\omega(\mathbf{q})\) [cm⁻¹].

#include "atomiverse/cpu/vibration/phonon_dispersion.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/vibration/atomic_masses.h"
#include "atomiverse/cpu/vibration/nist_constants.h"

namespace ATV {
namespace cpu {
namespace vib {

namespace {

constexpr double kTwoPi = 6.28318530717958647692;

/// @brief Encode home-cell unit atom → supercell index at (ia,ib,ic).
int encode_sc(int ia, int ib, int ic, int iu, int n_unit, int nb, int nc) {
    return ((ia * nb + ib) * nc + ic) * n_unit + iu;
}

/// @brief Mass-weighted Hermitian dynamical matrix at one q [Hartree/Bohr²/amu].
///
/// Builds \(D(\mathbf{q})\) from the supercell Cartesian Hessian (eV/Å²) using
/// only home-cell rows and every supercell image of the partner atom
/// (phonopy-style Fourier sum).
xt::xtensor<std::complex<double>, 2>
dynamical_matrix_q(const xt::xtensor<double, 2>& hessian_sc,
                   const std::vector<double>& mass, int n_unit, int na, int nb,
                   int nc, const std::array<double, 3>& q_frac) {
    const int dim = 3 * n_unit;
    const int n_sc = n_unit * na * nb * nc;
    const double uconv = nist::EV_PER_ANG2_TO_HARTREE_PER_BOHR2;

    xt::xtensor<std::complex<double>, 2> D =
        xt::zeros<std::complex<double>>({dim, dim});

    for (int i = 0; i < n_unit; ++i) {
        const int I = encode_sc(0, 0, 0, i, n_unit, nb, nc);
        const double inv_sqrt_mi =
            1.0 / std::sqrt(mass[static_cast<std::size_t>(i)]);
        for (int ja = 0; ja < na; ++ja) {
            for (int jb = 0; jb < nb; ++jb) {
                for (int jc = 0; jc < nc; ++jc) {
                    // phase = q · R, R in fractional cell indices of the
                    // unit lattice → 2π (q_a ja + q_b jb + q_c jc).
                    const double phase =
                        kTwoPi
                        * (q_frac[0] * static_cast<double>(ja)
                           + q_frac[1] * static_cast<double>(jb)
                           + q_frac[2] * static_cast<double>(jc));
                    const std::complex<double> w(std::cos(phase),
                                                 std::sin(phase));
                    for (int j = 0; j < n_unit; ++j) {
                        const int J = encode_sc(ja, jb, jc, j, n_unit, nb, nc);
                        if (I >= n_sc || J >= n_sc)
                            continue;
                        const double inv_sqrt_mj =
                            1.0 / std::sqrt(mass[static_cast<std::size_t>(j)]);
                        const double scale = uconv * inv_sqrt_mi * inv_sqrt_mj;
                        for (int a = 0; a < 3; ++a) {
                            for (int b = 0; b < 3; ++b) {
                                const double phi =
                                    hessian_sc(3 * I + a, 3 * J + b);
                                D(3 * i + a, 3 * j + b) += scale * phi * w;
                            }
                        }
                    }
                }
            }
        }
    }

    // Enforce Hermiticity (numerical FD noise).
    for (int p = 0; p < dim; ++p) {
        for (int q = p; q < dim; ++q) {
            const auto s = 0.5 * (D(p, q) + std::conj(D(q, p)));
            D(p, q) = s;
            D(q, p) = std::conj(s);
        }
    }
    return D;
}

/// @brief Eigenvalues of a Hermitian matrix via the real 2n embedding.
///
/// For \(H = A + iB\) (A symmetric, B skew), the real symmetric block matrix
/// \(\begin{bmatrix}A & -B\\ B & A\end{bmatrix}\) has each eigenvalue of H
/// with multiplicity two. Returns the unique eigenvalues (ascending), taking
/// the even-index of the paired spectrum after sort.
std::vector<double>
hermitian_eigenvalues(const xt::xtensor<std::complex<double>, 2>& H) {
    const int n = static_cast<int>(H.shape(0));
    if (n == 0)
        return {};
    if (static_cast<int>(H.shape(1)) != n)
        throw std::invalid_argument(
            "hermitian_eigenvalues: matrix must be square");

    xt::xtensor<double, 2> R = xt::zeros<double>({2 * n, 2 * n});
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const double a = H(i, j).real();
            const double b = H(i, j).imag();
            // [ A, -B ; B, A ]
            R(i, j) = a;
            R(i, j + n) = -b;
            R(i + n, j) = b;
            R(i + n, j + n) = a;
        }
    }

    auto [evals, evecs] = xt::linalg::eigh(R);
    // Paired spectrum: after ascending sort, take every other value by
    // averaging each consecutive pair (they should match to roundoff).
    std::vector<double> lam(static_cast<std::size_t>(n));
    for (int k = 0; k < n; ++k) {
        const double e0 = evals(2 * k);
        const double e1 = evals(2 * k + 1);
        lam[static_cast<std::size_t>(k)] = 0.5 * (e0 + e1);
    }
    return lam;
}

std::vector<double> lambdas_to_cm(const std::vector<double>& lambdas) {
    const double au2cm = nist::au2wavenumber();
    std::vector<double> w(lambdas.size());
    for (std::size_t k = 0; k < lambdas.size(); ++k) {
        const double lambda = lambdas[k];
        if (lambda < 0.0)
            w[k] = -std::sqrt(-lambda) * au2cm;
        else
            w[k] = std::sqrt(lambda) * au2cm;
    }
    std::sort(w.begin(), w.end());
    return w;
}

std::vector<double> omega_at_q(const xt::xtensor<double, 2>& hessian_sc,
                               const std::vector<double>& mass, int n_unit,
                               int na, int nb, int nc,
                               const std::array<double, 3>& q_frac) {
    auto D = dynamical_matrix_q(hessian_sc, mass, n_unit, na, nb, nc, q_frac);
    auto lam = hermitian_eigenvalues(D);
    return lambdas_to_cm(lam);
}

void validate_inputs(const xt::xtensor<double, 2>& hessian_sc,
                     const cpu::Frame& unit, int na, int nb, int nc) {
    if (na < 1 || nb < 1 || nc < 1)
        throw std::invalid_argument(
            "PhononDispersion: supercell factors na,nb,nc must be ≥ 1");
    const int n_unit = unit.nlocal();
    if (n_unit <= 0)
        throw std::invalid_argument("PhononDispersion: unit cell has no atoms");
    const int n_sc = n_unit * na * nb * nc;
    const int dim = 3 * n_sc;
    if (static_cast<int>(hessian_sc.shape(0)) != dim
        || static_cast<int>(hessian_sc.shape(1)) != dim) {
        throw std::invalid_argument(
            "PhononDispersion: hessian shape ["
            + std::to_string(hessian_sc.shape(0)) + ","
            + std::to_string(hessian_sc.shape(1)) + "] does not match 3 N_sc = "
            + std::to_string(dim) + " (N_unit=" + std::to_string(n_unit)
            + ", supercell " + std::to_string(na) + "x" + std::to_string(nb)
            + "x" + std::to_string(nc) + ")");
    }
}

std::vector<double> masses_from_frame(const cpu::Frame& unit) {
    const int n = unit.nlocal();
    std::vector<double> mass(static_cast<std::size_t>(n));
    const auto type = unit.span(kAtom_type);
    for (int i = 0; i < n; ++i)
        mass[static_cast<std::size_t>(i)] = mass_amu(static_cast<int>(type[i]));
    return mass;
}

std::vector<std::array<double, 3>> lerp_segment(const std::array<double, 3>& a,
                                                const std::array<double, 3>& b,
                                                int n_seg, bool include_start) {
    std::vector<std::array<double, 3>> pts;
    if (n_seg < 2)
        n_seg = 2;
    const int start = include_start ? 0 : 1;
    for (int i = start; i < n_seg; ++i) {
        const double t =
            static_cast<double>(i) / static_cast<double>(n_seg - 1);
        pts.push_back({a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]),
                       a[2] + t * (b[2] - a[2])});
    }
    return pts;
}

} // namespace

void PhononDispersion::set_diamond_path(int n_seg) {
    // FCC/diamond primitive-cell path (fractional), same as Bands.
    const std::array<double, 3> G = {0.0, 0.0, 0.0};
    const std::array<double, 3> X = {0.5, 0.0, 0.5};
    const std::array<double, 3> W = {0.5, 0.25, 0.75};
    const std::array<double, 3> L = {0.5, 0.5, 0.5};
    const std::array<double, 3> K = {0.375, 0.375, 0.75};

    qfrac.clear();
    auto append = [&](const std::array<double, 3>& a,
                      const std::array<double, 3>& b, bool include_start) {
        auto seg = lerp_segment(a, b, n_seg, include_start);
        qfrac.insert(qfrac.end(), seg.begin(), seg.end());
    };
    append(G, X, true);
    append(X, W, false);
    append(W, L, false);
    append(L, G, false);
    append(G, K, false);
}

PhononDispersionResult
PhononDispersion::compute(const xt::xtensor<double, 2>& hessian_sc,
                          const cpu::Frame& unit, int na, int nb,
                          int nc) const {
    validate_inputs(hessian_sc, unit, na, nb, nc);
    const int n_unit = unit.nlocal();
    const auto mass = masses_from_frame(unit);

    std::vector<std::array<double, 3>> qs = qfrac;
    if (qs.empty()) {
        PhononDispersion tmp;
        tmp.set_diamond_path(4);
        qs = tmp.qfrac;
    }

    PhononDispersionResult out;
    out.qfrac = qs;
    out.n_atoms_unit = n_unit;
    out.n_branches = 3 * n_unit;
    out.na = na;
    out.nb = nb;
    out.nc = nc;
    out.frequencies_cm.resize(qs.size());

    for (std::size_t iq = 0; iq < qs.size(); ++iq) {
        out.frequencies_cm[iq] =
            omega_at_q(hessian_sc, mass, n_unit, na, nb, nc, qs[iq]);
    }
    return out;
}

std::vector<double> PhononDispersion::frequencies_at_q(
    const xt::xtensor<double, 2>& hessian_sc, const cpu::Frame& unit, int na,
    int nb, int nc, const std::array<double, 3>& q_frac) const {
    validate_inputs(hessian_sc, unit, na, nb, nc);
    const int n_unit = unit.nlocal();
    const auto mass = masses_from_frame(unit);
    return omega_at_q(hessian_sc, mass, n_unit, na, nb, nc, q_frac);
}

} // namespace vib
} // namespace cpu
} // namespace ATV
