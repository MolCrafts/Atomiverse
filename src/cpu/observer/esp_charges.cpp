/// @file cpu/observer/esp_charges.cpp
/// @brief MK / CHELPG ESP-fitted charges implementation. @see esp_charges.h

#include "atomiverse/cpu/observer/esp_charges.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include <xtensor-blas/xlinalg.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"

namespace ATV {
namespace cpu {

namespace {

using scf::MeanFieldOutputs;
using scf::RealMatrix;
using scf::RealVector;
using scf::native::Atom;
using scf::native::ShellBlock;

constexpr double kPi = 3.14159265358979323846;

/// @brief Bondi vdW radius [Å] (Bondi, J. Phys. Chem. 68, 441 (1964)).
/// @param Z Atomic number. @return Radius [Å].
/// @throws std::invalid_argument for elements outside the table (no
///         fallback values).
double bondi_radius(int Z) {
    switch (Z) {
    case 1: return 1.20;
    case 2: return 1.40;
    case 6: return 1.70;
    case 7: return 1.55;
    case 8: return 1.52;
    case 9: return 1.47;
    case 10: return 1.54;
    case 15: return 1.80;
    case 16: return 1.80;
    case 17: return 1.75;
    case 35: return 1.85;
    default:
        throw std::invalid_argument(
            "esp_charges: no Bondi vdW radius tabulated for Z="
            + std::to_string(Z));
    }
}

/// @brief Atom positions [Å] and atomic numbers from the Frame.
struct Sites {
    std::vector<std::array<double, 3>> pos_ang;
    std::vector<int> z;

    explicit Sites(const Frame& frame) {
        const int natm = frame.nlocal();
        const auto x = frame.span(kAtom_x);
        const auto y = frame.span(kAtom_y);
        const auto zc = frame.span(kAtom_z);
        const auto type = frame.span(kAtom_type);
        pos_ang.resize(static_cast<std::size_t>(natm));
        z.resize(static_cast<std::size_t>(natm));
        for (int i = 0; i < natm; ++i) {
            pos_ang[static_cast<std::size_t>(i)] = {x[i], y[i], zc[i]};
            z[static_cast<std::size_t>(i)] = static_cast<int>(type[i]);
        }
    }
};

double dist(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/// @brief QM ESP V_k = Σ_A Z_A/|R_A−r_k| − Tr(P·⟨1/|r−r_k|⟩) [Hartree/e]
///        at each grid point, via `nuclear_block` with a unit point charge
///        at r_k (spherical packs sandwich with c2s).
std::vector<double> qm_esp(const MeanFieldOutputs& outputs, const Sites& sites,
                           const std::vector<std::array<double, 3>>& grid_ang) {
    const auto& shells = *outputs.shells;
    const auto blocks = scf::blocks_from_shells(shells);
    const int n_blocks = static_cast<int>(blocks.size());
    const int n_cart = static_cast<int>(shells.size());
    const int natm = static_cast<int>(sites.z.size());

    const bool sph = outputs.c2s && outputs.c2s->size() > 0;
    const RealMatrix P = outputs.result->density_total();

    std::vector<double> V(grid_ang.size(), 0.0);
    RealMatrix N_cart = xt::zeros<double>({n_cart, n_cart});
    std::vector<double> buf;
    for (std::size_t k = 0; k < grid_ang.size(); ++k) {
        const auto& p = grid_ang[k];
        // Nuclear term [Hartree/e], distances in Bohr.
        double v_nuc = 0.0;
        for (int A = 0; A < natm; ++A)
            v_nuc += static_cast<double>(sites.z[static_cast<std::size_t>(A)])
                     / (dist(p, sites.pos_ang[static_cast<std::size_t>(A)])
                        * units::kAngstromInBohr);

        // ⟨μ|−1/|r−r_k||ν⟩ via a unit fake nucleus at r_k.
        const Atom probe{1,
                         {p[0] * units::kAngstromInBohr,
                          p[1] * units::kAngstromInBohr,
                          p[2] * units::kAngstromInBohr}};
        N_cart.fill(0.0);
        for (int i = 0; i < n_blocks; ++i) {
            for (int j = 0; j <= i; ++j) {
                const ShellBlock& A = blocks[static_cast<std::size_t>(i)];
                const ShellBlock& B = blocks[static_cast<std::size_t>(j)];
                buf.assign(static_cast<std::size_t>(A.n_cart * B.n_cart), 0.0);
                scf::native::nuclear_block(A, B, &probe, 1, buf.data());
                for (int ia = 0; ia < A.n_cart; ++ia)
                    for (int ib = 0; ib < B.n_cart; ++ib) {
                        const double v = buf[ia * B.n_cart + ib];
                        N_cart(A.ao_start + ia, B.ao_start + ib) = v;
                        N_cart(B.ao_start + ib, A.ao_start + ia) = v;
                    }
            }
        }

        // Electron term: −Tr(P·⟨1/r⟩) = +Tr(P·N) since N carries the −Z=−1.
        double tr = 0.0;
        if (sph) {
            const auto& C = *outputs.c2s;
            const RealMatrix N_sph = xt::linalg::dot(
                xt::transpose(C), RealMatrix(xt::linalg::dot(N_cart, C)));
            for (std::size_t i = 0; i < N_sph.shape()[0]; ++i)
                for (std::size_t j = 0; j < N_sph.shape()[1]; ++j)
                    tr += P(i, j) * N_sph(i, j);
        } else {
            for (int i = 0; i < n_cart; ++i)
                for (int j = 0; j < n_cart; ++j)
                    tr += P(i, j) * N_cart(i, j);
        }
        V[k] = v_nuc + tr;
    }
    return V;
}

/// @brief Lagrange-constrained least squares (shared by MK and CHELPG):
///        KKT system [[A, 1], [1ᵀ, 0]]·[q; λ] = [b; q_tot], solved with
///        xt::linalg::solve; SVD minimum-norm solution when degenerate.
std::vector<double>
constrained_fit(const std::vector<double>& V,
                const std::vector<std::array<double, 3>>& grid_ang,
                const Sites& sites, double q_tot) {
    if (V.size() != grid_ang.size())
        throw std::invalid_argument(
            "esp_charges: potential samples (" + std::to_string(V.size())
            + ") != grid points (" + std::to_string(grid_ang.size()) + ")");
    const int natm = static_cast<int>(sites.z.size());
    const std::size_t npts = grid_ang.size();

    // Design matrix column A: 1/|r_k − R_A| [1/Bohr].
    RealMatrix rinv = xt::zeros<double>({static_cast<int>(npts), natm});
    for (std::size_t k = 0; k < npts; ++k)
        for (int A = 0; A < natm; ++A)
            rinv(k, A) = 1.0
                         / (dist(grid_ang[k],
                                 sites.pos_ang[static_cast<std::size_t>(A)])
                            * units::kAngstromInBohr);

    // Normal equations + the multiplier row.
    const int n = natm + 1;
    RealMatrix K = xt::zeros<double>({n, n});
    RealVector rhs = xt::zeros<double>({n});
    for (int a = 0; a < natm; ++a) {
        for (int b = 0; b < natm; ++b) {
            double s = 0.0;
            for (std::size_t k = 0; k < npts; ++k)
                s += rinv(k, a) * rinv(k, b);
            K(a, b) = s;
        }
        double s = 0.0;
        for (std::size_t k = 0; k < npts; ++k)
            s += rinv(k, a) * V[k];
        rhs(a) = s;
        K(a, natm) = 1.0;
        K(natm, a) = 1.0;
    }
    rhs(natm) = q_tot;

    // Condition probe: SVD of the KKT matrix. Well-conditioned → direct
    // solve; degenerate (buried / coincident atoms) → SVD minimum-norm
    // pseudo-inverse from the same factors (must not diverge).
    auto svd = xt::linalg::svd(K, /*full_matrices=*/false);
    const RealMatrix& U = std::get<0>(svd);
    const RealVector& s = std::get<1>(svd);
    const RealMatrix& Vt = std::get<2>(svd);
    const double s_max = s(0);
    const double cutoff = s_max * 1e-12;

    RealVector sol;
    if (s(n - 1) > cutoff) {
        sol = xt::linalg::solve(K, rhs);
    } else {
        const RealVector Ub = xt::linalg::dot(xt::transpose(U), rhs);
        RealVector w = xt::zeros<double>({n});
        for (int i = 0; i < n; ++i)
            w(i) = (s(i) > cutoff) ? Ub(i) / s(i) : 0.0;
        sol = xt::linalg::dot(xt::transpose(Vt), w);
    }

    std::vector<double> q(static_cast<std::size_t>(natm));
    for (int a = 0; a < natm; ++a)
        q[static_cast<std::size_t>(a)] = sol(a);
    return q;
}

/// @brief Handle guard + total charge q_tot = Σ Z − N_el for the fit entries.
double check_handle_q_tot(const MeanFieldOutputs& outputs, const Sites& sites) {
    if (!outputs.result || !outputs.shells || !outputs.ready)
        throw std::invalid_argument(
            "esp_charges: null MeanFieldOutputs (did you forget "
            "method.result_handle()?)");
    if (!*outputs.ready)
        throw std::invalid_argument(
            "esp_charges: MeanFieldOutputs not ready — run the method first");
    using Spin = scf::SlaterDeterminant::Spin;
    double q_tot = 0.0;
    for (const int z : sites.z)
        q_tot += static_cast<double>(z);
    q_tot -= static_cast<double>(outputs.result->n_electrons(Spin::alpha)
                                 + outputs.result->n_electrons(Spin::beta));
    return q_tot;
}

} // namespace

std::vector<std::array<double, 3>> MKCharges::grid(const Frame& frame) const {
    const Sites sites{frame};
    const int natm = static_cast<int>(sites.z.size());
    const double golden = kPi * (3.0 - std::sqrt(5.0));

    std::vector<std::array<double, 3>> pts;
    for (int a = 0; a < natm; ++a) {
        for (const double s : shell_scales) {
            const double r =
                s * bondi_radius(sites.z[static_cast<std::size_t>(a)]);
            const int n = static_cast<int>(
                std::ceil(density_per_ang2 * 4.0 * kPi * r * r));
            for (int k = 0; k < n; ++k) {
                const double z = 1.0 - 2.0 * (k + 0.5) / n;
                const double phi = k * golden;
                const double rho = std::sqrt(std::max(0.0, 1.0 - z * z));
                const auto& c = sites.pos_ang[static_cast<std::size_t>(a)];
                const std::array<double, 3> p = {
                    c[0] + r * rho * std::cos(phi),
                    c[1] + r * rho * std::sin(phi), c[2] + r * z};
                bool keep = true;
                for (int b = 0; b < natm && keep; ++b) {
                    if (b == a)
                        continue;
                    const double rb =
                        s * bondi_radius(sites.z[static_cast<std::size_t>(b)]);
                    if (dist(p, sites.pos_ang[static_cast<std::size_t>(b)])
                        < rb)
                        keep = false;
                }
                if (keep)
                    pts.push_back(p);
            }
        }
    }
    return pts;
}

std::vector<double> MKCharges::fit(const scf::MeanFieldOutputs& outputs,
                                   const Frame& frame) const {
    const Sites sites{frame};
    const double q_tot = check_handle_q_tot(outputs, sites);
    const auto g = grid(frame);
    const auto V = qm_esp(outputs, sites, g);
    return constrained_fit(V, g, sites, q_tot);
}

std::vector<double>
MKCharges::fit_potential(const std::vector<double>& V,
                         const std::vector<std::array<double, 3>>& grid_ang,
                         const Frame& frame, double q_tot) const {
    return constrained_fit(V, grid_ang, Sites{frame}, q_tot);
}

std::vector<std::array<double, 3>>
ChelpgCharges::grid(const Frame& frame) const {
    const Sites sites{frame};
    const int natm = static_cast<int>(sites.z.size());

    std::array<double, 3> lo = sites.pos_ang[0];
    std::array<double, 3> hi = sites.pos_ang[0];
    for (int a = 1; a < natm; ++a)
        for (int d = 0; d < 3; ++d) {
            lo[static_cast<std::size_t>(d)] =
                std::min(lo[static_cast<std::size_t>(d)],
                         sites.pos_ang[static_cast<std::size_t>(a)]
                                      [static_cast<std::size_t>(d)]);
            hi[static_cast<std::size_t>(d)] =
                std::max(hi[static_cast<std::size_t>(d)],
                         sites.pos_ang[static_cast<std::size_t>(a)]
                                      [static_cast<std::size_t>(d)]);
        }

    // Pre-resolve radii (throws before any point is emitted).
    std::vector<double> rvdw(static_cast<std::size_t>(natm));
    for (int a = 0; a < natm; ++a)
        rvdw[static_cast<std::size_t>(a)] =
            bondi_radius(sites.z[static_cast<std::size_t>(a)]);

    std::array<std::vector<double>, 3> axes;
    for (int d = 0; d < 3; ++d) {
        const double start = lo[static_cast<std::size_t>(d)] - band_ang;
        const double stop = hi[static_cast<std::size_t>(d)] + band_ang;
        for (int i = 0;; ++i) {
            const double c = start + i * spacing_ang;
            if (c > stop + 1e-9)
                break;
            axes[static_cast<std::size_t>(d)].push_back(c);
        }
    }

    std::vector<std::array<double, 3>> pts;
    for (const double x : axes[0])
        for (const double y : axes[1])
            for (const double z : axes[2]) {
                const std::array<double, 3> p = {x, y, z};
                bool outside_all = true;
                bool near_one = false;
                for (int a = 0; a < natm; ++a) {
                    const double d =
                        dist(p, sites.pos_ang[static_cast<std::size_t>(a)]);
                    if (d < rvdw[static_cast<std::size_t>(a)]) {
                        outside_all = false;
                        break;
                    }
                    if (d <= rvdw[static_cast<std::size_t>(a)] + band_ang)
                        near_one = true;
                }
                if (outside_all && near_one)
                    pts.push_back(p);
            }
    return pts;
}

std::vector<double> ChelpgCharges::fit(const scf::MeanFieldOutputs& outputs,
                                       const Frame& frame) const {
    const Sites sites{frame};
    const double q_tot = check_handle_q_tot(outputs, sites);
    const auto g = grid(frame);
    const auto V = qm_esp(outputs, sites, g);
    return constrained_fit(V, g, sites, q_tot);
}

std::vector<double>
ChelpgCharges::fit_potential(const std::vector<double>& V,
                             const std::vector<std::array<double, 3>>& grid_ang,
                             const Frame& frame, double q_tot) const {
    return constrained_fit(V, grid_ang, Sites{frame}, q_tot);
}

} // namespace cpu
} // namespace ATV
