/// @file cpu/scf/wannier.cpp
/// @brief Foster–Boys localization implementation. @see wannier.h

#include "atomiverse/cpu/scf/wannier.h"

#include <cmath>
#include <stdexcept>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/one_electron.h"

namespace ATV {
namespace cpu {
namespace scf {

namespace {

constexpr double kBohr2ToAng2 = units::kBohrInAngstrom * units::kBohrInAngstrom;

/// Occupied spatial columns: integer occupations f_i == 1 (per spin).
std::vector<int> occupied_columns(const SlaterDeterminant& det) {
    if (!det.restricted() || det.generalized())
        throw std::invalid_argument(
            "WannierCenters: Foster–Boys foothold requires a restricted "
            "(closed-shell) non-generalized determinant");
    const auto& f = det.occupations(SlaterDeterminant::Spin::alpha);
    std::vector<int> cols;
    cols.reserve(static_cast<std::size_t>(f.size()));
    for (std::size_t i = 0; i < f.size(); ++i) {
        if (f(i) > 0.5)
            cols.push_back(static_cast<int>(i));
    }
    if (cols.empty())
        throw std::invalid_argument(
            "WannierCenters: no occupied orbitals (f_i > 0.5)");
    return cols;
}

xt::xtensor<double, 2> to_ao_basis(const xt::xtensor<double, 2>& M_cart,
                                   const MeanFieldOutputs& outs) {
    if (!outs.c2s || outs.c2s->size() == 0)
        return M_cart;
    const auto& C = *outs.c2s;
    return xt::xtensor<double, 2>(xt::linalg::dot(
        xt::transpose(C), xt::xtensor<double, 2>(xt::linalg::dot(M_cart, C))));
}

/// Build occupied C [n_ao × n_occ] from determinant columns.
RealMatrix occ_coeff(const SlaterDeterminant& det,
                     const std::vector<int>& cols) {
    const auto& C = det.coefficients(SlaterDeterminant::Spin::alpha);
    const auto n_ao = C.shape()[0];
    const auto n_occ = cols.size();
    RealMatrix Co = xt::zeros<double>({n_ao, n_occ});
    for (std::size_t j = 0; j < n_occ; ++j) {
        const auto src = static_cast<std::size_t>(cols[j]);
        for (std::size_t mu = 0; mu < n_ao; ++mu)
            Co(mu, j) = C(mu, src);
    }
    return Co;
}

/// MO multipole block R^a = C^T r^a C  [n_occ × n_occ], Bohr.
void mo_first(const RealMatrix& Co, const xt::xtensor<double, 2> r_ao[3],
              xt::xtensor<double, 2> R[3]) {
    for (int a = 0; a < 3; ++a) {
        const xt::xtensor<double, 2> tmp = xt::linalg::dot(r_ao[a], Co);
        R[a] = xt::linalg::dot(xt::transpose(Co), tmp);
    }
}

/// Diagonal second-moment sum ⟨i|r²|i⟩ from packed xx,xy,xz,yy,yz,zz AO mats.
void mo_r2_diag(const RealMatrix& Co, const xt::xtensor<double, 2> m2_ao[6],
                std::vector<double>& r2) {
    const auto n = Co.shape()[1];
    r2.assign(n, 0.0);
    // ⟨r²⟩ = ⟨xx⟩ + ⟨yy⟩ + ⟨zz⟩ → packed indices 0, 3, 5.
    constexpr int kDiag[3] = {0, 3, 5};
    for (int k : kDiag) {
        const auto tmp = xt::linalg::dot(m2_ao[k], Co);
        const auto M = xt::linalg::dot(xt::transpose(Co), tmp);
        for (std::size_t i = 0; i < n; ++i)
            r2[i] += M(i, i);
    }
}

void fill_result_moments(const RealMatrix& Co,
                         const xt::xtensor<double, 2> r_ao[3],
                         const xt::xtensor<double, 2> m2_ao[6],
                         WannierResult& out, bool as_initial) {
    xt::xtensor<double, 2> R[3];
    mo_first(Co, r_ao, R);
    std::vector<double> r2;
    mo_r2_diag(Co, m2_ao, r2);
    const auto n = Co.shape()[1];
    std::vector<std::array<double, 3>> centers(n);
    std::vector<double> spreads(n);
    double tot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        std::array<double, 3> c_bohr = {R[0](i, i), R[1](i, i), R[2](i, i)};
        const double r2i = r2[i];
        const double mag2 = c_bohr[0] * c_bohr[0] + c_bohr[1] * c_bohr[1]
                            + c_bohr[2] * c_bohr[2];
        const double omega_bohr2 = r2i - mag2;
        centers[i] = {c_bohr[0] * units::kBohrInAngstrom,
                      c_bohr[1] * units::kBohrInAngstrom,
                      c_bohr[2] * units::kBohrInAngstrom};
        spreads[i] = omega_bohr2 * kBohr2ToAng2;
        tot += spreads[i];
    }
    if (as_initial) {
        out.total_spread_initial = tot;
    } else {
        out.centers = std::move(centers);
        out.spreads = std::move(spreads);
        out.total_spread = tot;
    }
}

} // namespace

double boys_pair_angle(double A, double B) {
    // γ = (1/4) atan2(B, A) ∈ (-π/4, π/4]
    return 0.25 * std::atan2(B, A);
}

WannierResult WannierCenters::compute(const MeanFieldOutputs& outs,
                                      const cpu::Frame& frame) const {
    (void)frame;
    if (max_sweeps < 1)
        throw std::invalid_argument("WannierCenters: max_sweeps must be >= 1");
    if (!(tol > 0.0))
        throw std::invalid_argument("WannierCenters: tol must be > 0");
    if (!outs.ready || !*outs.ready)
        throw std::invalid_argument(
            "WannierCenters: mean-field result is not ready — run() first");
    if (!outs.result || !outs.shells)
        throw std::invalid_argument(
            "WannierCenters: empty MeanFieldOutputs handles");

    const auto& det = *outs.result;
    const auto cols = occupied_columns(det);
    const auto n_occ = cols.size();

    const auto& shells = *outs.shells;
    const int n_shell = static_cast<int>(shells.size());
    if (n_shell <= 0)
        throw std::invalid_argument("WannierCenters: empty shell list");

    const double origin[3] = {0.0, 0.0, 0.0};
    const MultipoleIntegrals ints{shells.data(), n_shell, origin};

    xt::xtensor<double, 2> r_ao[3];
    xt::xtensor<double, 2> m2_ao[6];
    for (int a = 0; a < 3; ++a)
        r_ao[a] = to_ao_basis(ints.dipole(a), outs);
    for (int k = 0; k < 6; ++k)
        m2_ao[k] = to_ao_basis(ints.second(k), outs);

    RealMatrix Co = occ_coeff(det, cols);

    WannierResult result;
    fill_result_moments(Co, r_ao, m2_ao, result, /*as_initial=*/true);

    if (n_occ == 1) {
        fill_result_moments(Co, r_ao, m2_ao, result, /*as_initial=*/false);
        result.sweeps = 0;
        result.converged = true;
        return result;
    }

    bool converged = false;
    int sweeps = 0;
    for (int sw = 0; sw < max_sweeps; ++sw) {
        ++sweeps;
        double max_abs_gamma = 0.0;
        // Rebuild MO multipoles each sweep (cheap for small n_occ).
        xt::xtensor<double, 2> R[3];
        mo_first(Co, r_ao, R);
        for (std::size_t i = 0; i < n_occ; ++i) {
            for (std::size_t j = i + 1; j < n_occ; ++j) {
                double A = 0.0;
                double B = 0.0;
                for (int a = 0; a < 3; ++a) {
                    const double dii = R[a](i, i);
                    const double djj = R[a](j, j);
                    const double dij = R[a](i, j);
                    const double Aa = dii - djj;
                    const double Ba = 2.0 * dij;
                    A += Aa * Aa - Ba * Ba;
                    B += 2.0 * Aa * Ba;
                }
                const double gamma = boys_pair_angle(A, B);
                max_abs_gamma = std::max(max_abs_gamma, std::fabs(gamma));
                if (std::fabs(gamma) < 1e-16)
                    continue;
                const double c = std::cos(gamma);
                const double s = std::sin(gamma);
                // Co'[:,i] = c Ci + s Cj; Co'[:,j] = -s Ci + c Cj
                const auto n_ao = Co.shape()[0];
                for (std::size_t mu = 0; mu < n_ao; ++mu) {
                    const double ci = Co(mu, i);
                    const double cj = Co(mu, j);
                    Co(mu, i) = c * ci + s * cj;
                    Co(mu, j) = -s * ci + c * cj;
                }
                // n_occ is small — rebuild MO multipoles after each rotation.
                mo_first(Co, r_ao, R);
            }
        }
        if (max_abs_gamma < tol) {
            converged = true;
            break;
        }
    }

    fill_result_moments(Co, r_ao, m2_ao, result, /*as_initial=*/false);
    result.sweeps = sweeps;
    result.converged = converged;
    return result;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
