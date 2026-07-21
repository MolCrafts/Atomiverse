/// @file cpu/scf/internal/pcm.cpp
/// @brief C-PCM / COSMO cavity + conductor solve + AO reaction field.

#include "atomiverse/cpu/scf/internal/pcm.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/core/xmath.hpp>
#include <xtensor/generators/xbuilder.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"
#include "atomiverse/cpu/scf/lebedev_data.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

constexpr double kPi = 3.14159265358979323846;
/// Klamt–Schüürmann diagonal prefactor (dimensionless) for the classical
/// point-charge S matrix. Prefer the Gaussian-smeared form below (Lange–
/// Herbert / PySCF) when a Lebedev order is tabulated.
constexpr double kKlamtDiag = 1.0694;

/// Lange & Herbert XI table (JCP 133, 244111 (2010) TABLE II) for the
/// Gaussian charge width ξ = XI / (R √w) on a Lebedev sphere of R points.
double lebedev_xi_factor(int n_leb) {
    switch (n_leb) {
    case 6: return 4.84566077868;
    case 14: return 4.86458714334;
    case 26: return 4.85478226219;
    case 38: return 4.90105812685;
    case 50: return 4.89250673295;
    case 86: return 4.89741372580;
    case 110: return 4.90101060987;
    case 146: return 4.89825187392;
    case 170: return 4.90685517725;
    case 194: return 4.90337644248;
    case 302: return 4.90498088169;
    case 350: return 4.86879474832;
    case 434: return 4.90567349080;
    case 590: return 4.90624071359;
    case 770: return 4.90656435779;
    default: return 0.0; // fall back to Klamt point-charge S
    }
}

const lebedev::Point* resolve_lebedev(int n, int& n_pts) {
    if (n == 6) {
        n_pts = lebedev::n_6;
        return lebedev::grid_6;
    }
    if (n == 38) {
        n_pts = lebedev::n_38;
        return lebedev::grid_38;
    }
    if (n == 50) {
        n_pts = lebedev::n_50;
        return lebedev::grid_50;
    }
    if (n == 86) {
        n_pts = lebedev::n_86;
        return lebedev::grid_86;
    }
    if (n == 110) {
        n_pts = lebedev::n_110;
        return lebedev::grid_110;
    }
    if (n == 194) {
        n_pts = lebedev::n_194;
        return lebedev::grid_194;
    }
    if (n == 266) {
        n_pts = lebedev::n_266;
        return lebedev::grid_266;
    }
    if (n == 302) {
        n_pts = lebedev::n_302;
        return lebedev::grid_302;
    }
    if (n == 590) {
        n_pts = lebedev::n_590;
        return lebedev::grid_590;
    }
    if (n == 770) {
        n_pts = lebedev::n_770;
        return lebedev::grid_770;
    }
    throw std::invalid_argument(
        "Pcm: unsupported Lebedev order n=" + std::to_string(n)
        + " (supported: 6,38,50,86,110,194,266,302,590,770)");
}

} // namespace

double bondi_radius_ang(int Z) {
    // Classical Bondi (J. Phys. Chem. 68, 441 (1964)) with the common PCM
    // "modified Bondi" hydrogen radius of 1.1 Å (PySCF / Q-Chem default;
    // classic Bondi H is 1.20 Å). Unsupported Z throws — no fallback.
    switch (Z) {
    case 1: return 1.10; // modified Bondi H (PCM convention)
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
            "Pcm: no Bondi vdW radius tabulated for Z=" + std::to_string(Z));
    }
}

Pcm::Pcm(std::vector<native::Atom> atoms, double eps, double radii_scale,
         double x, int lebedev_n)
    : atoms_(std::move(atoms)), eps_(eps), x_(x) {
    if (atoms_.empty())
        throw std::invalid_argument("Pcm: empty atom list");
    if (!(eps_ > 0.0) || !std::isfinite(eps_))
        throw std::invalid_argument(
            "Pcm: dielectric eps must be finite and > 0 (got "
            + std::to_string(eps_) + ")");
    if (!std::isfinite(radii_scale) || !(radii_scale > 0.0))
        throw std::invalid_argument(
            "Pcm: radii_scale must be finite and > 0");
    if (!std::isfinite(x_) || x_ < 0.0)
        throw std::invalid_argument(
            "Pcm: solvent_x must be finite and >= 0 (0=C-PCM, 0.5=COSMO)");

    f_eps_ = (eps_ - 1.0) / (eps_ + x_);

    const int nat = static_cast<int>(atoms_.size());
    std::vector<double> R(static_cast<std::size_t>(nat));
    for (int a = 0; a < nat; ++a) {
        // Bondi [Å] → Bohr (atom positions already in Bohr).
        R[static_cast<std::size_t>(a)] =
            radii_scale * bondi_radius_ang(atoms_[static_cast<std::size_t>(a)].Z)
            * units::kAngstromInBohr;
    }

    int n_leb = 0;
    const lebedev::Point* leb = resolve_lebedev(lebedev_n, n_leb);

    // Tesserae: sphere points, hard bury-delete when inside another sphere.
    // Also retain per-tessera Gaussian width ξ (Lange–Herbert) for the S
    // matrix; ξ = XI / (R √w) with w = 4π · leb_weight (unit-sphere sum=1).
    std::vector<double> charge_exp; // ξ per kept tessera
    const double xi_tab = lebedev_xi_factor(n_leb);
    for (int a = 0; a < nat; ++a) {
        const double Ra = R[static_cast<std::size_t>(a)];
        const auto& A = atoms_[static_cast<std::size_t>(a)];
        for (int g = 0; g < n_leb; ++g) {
            const double px = A.pos[0] + Ra * leb[g].x;
            const double py = A.pos[1] + Ra * leb[g].y;
            const double pz = A.pos[2] + Ra * leb[g].z;
            bool buried = false;
            for (int b = 0; b < nat; ++b) {
                if (b == a)
                    continue;
                const auto& B = atoms_[static_cast<std::size_t>(b)];
                const double dx = px - B.pos[0];
                const double dy = py - B.pos[1];
                const double dz = pz - B.pos[2];
                const double r2 = dx * dx + dy * dy + dz * dz;
                const double Rb = R[static_cast<std::size_t>(b)];
                // Strict inside other sphere → discard. On-surface equality
                // kept (rare). Cavity surface is discontinuous — documented.
                if (r2 + 1e-14 < Rb * Rb) {
                    buried = true;
                    break;
                }
            }
            if (buried)
                continue;
            // Weights sum to 1 over the unit sphere → area = w * 4π R².
            const double w4pi = leb[g].w * (4.0 * kPi);
            const double ak = w4pi * Ra * Ra;
            pos_.push_back({px, py, pz});
            area_.push_back(ak);
            if (xi_tab > 0.0)
                charge_exp.push_back(xi_tab / (Ra * std::sqrt(w4pi)));
            else
                charge_exp.push_back(0.0);
        }
    }
    if (pos_.empty())
        throw std::runtime_error(
            "Pcm: cavity has zero exposed tesserae (check radii / geometry)");

    const int nt = static_cast<int>(pos_.size());
    v_nuc_.assign(static_cast<std::size_t>(nt), 0.0);
    for (int k = 0; k < nt; ++k) {
        double vn = 0.0;
        for (int a = 0; a < nat; ++a) {
            const auto& A = atoms_[static_cast<std::size_t>(a)];
            const double dx = A.pos[0] - pos_[static_cast<std::size_t>(k)][0];
            const double dy = A.pos[1] - pos_[static_cast<std::size_t>(k)][1];
            const double dz = A.pos[2] - pos_[static_cast<std::size_t>(k)][2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-12)
                throw std::runtime_error(
                    "Pcm: tessera coincides with a nucleus (singular 1/r)");
            vn += static_cast<double>(A.Z) / r;
        }
        v_nuc_[static_cast<std::size_t>(k)] = vn;
    }

    // S matrix. Prefer Lange–Herbert Gaussian-smeared Coulomb (erf(ξr)/r)
    // matching PySCF PCM when XI is tabulated; else classical point-charge
    // S with Klamt–Schüürmann diagonal.
    S_ = xt::zeros<double>({static_cast<std::size_t>(nt),
                            static_cast<std::size_t>(nt)});
    const bool use_gauss = (xi_tab > 0.0);
    for (int k = 0; k < nt; ++k) {
        if (use_gauss) {
            const double xik = charge_exp[static_cast<std::size_t>(k)];
            S_(k, k) = xik * std::sqrt(2.0 / kPi); // switch_fun = 1 (hard)
        } else {
            S_(k, k) =
                kKlamtDiag
                * std::sqrt(4.0 * kPi / area_[static_cast<std::size_t>(k)]);
        }
        for (int l = 0; l < k; ++l) {
            const double dx = pos_[static_cast<std::size_t>(k)][0]
                              - pos_[static_cast<std::size_t>(l)][0];
            const double dy = pos_[static_cast<std::size_t>(k)][1]
                              - pos_[static_cast<std::size_t>(l)][1];
            const double dz = pos_[static_cast<std::size_t>(k)][2]
                              - pos_[static_cast<std::size_t>(l)][2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r < 1e-12)
                throw std::runtime_error(
                    "Pcm: two tesserae coincide (singular S)");
            double s_kl;
            if (use_gauss) {
                const double xik = charge_exp[static_cast<std::size_t>(k)];
                const double xil = charge_exp[static_cast<std::size_t>(l)];
                const double xij = xik * xil / std::sqrt(xik * xik + xil * xil);
                s_kl = std::erf(xij * r) / r;
            } else {
                s_kl = 1.0 / r;
            }
            S_(k, l) = s_kl;
            S_(l, k) = s_kl;
        }
    }

    // Cache S^{-1} once (nt ≲ 10³ for typical small molecules).
    S_inv_ = xt::linalg::inv(S_);
}

double Pcm::f_epsilon() const { return f_eps_; }

void Pcm::prepare(const std::vector<native::Shell>& shells,
                  const xt::xtensor<double, 2>& c2s) {
    V_unit_.clear();
    n_ao_ = static_cast<int>(shells.size());
    if (n_ao_ == 0)
        return;

    const auto blocks = blocks_from_shells(shells);
    const int n_blocks = static_cast<int>(blocks.size());
    const int nt = static_cast<int>(pos_.size());
    V_unit_.resize(static_cast<std::size_t>(nt));
    std::vector<double> buf;

    for (int k = 0; k < nt; ++k) {
        xt::xtensor<double, 2> V =
            xt::zeros<double>({static_cast<std::size_t>(n_ao_),
                               static_cast<std::size_t>(n_ao_)});
        const native::Atom probe{
            1,
            {pos_[static_cast<std::size_t>(k)][0],
             pos_[static_cast<std::size_t>(k)][1],
             pos_[static_cast<std::size_t>(k)][2]}};
        for (int i = 0; i < n_blocks; ++i) {
            for (int j = 0; j <= i; ++j) {
                const auto& A = blocks[static_cast<std::size_t>(i)];
                const auto& B = blocks[static_cast<std::size_t>(j)];
                const int nab = A.n_cart * B.n_cart;
                buf.assign(static_cast<std::size_t>(nab), 0.0);
                native::nuclear_block(A, B, &probe, 1, buf.data());
                for (int ia = 0; ia < A.n_cart; ++ia) {
                    for (int ib = 0; ib < B.n_cart; ++ib) {
                        const double v =
                            buf[static_cast<std::size_t>(ia * B.n_cart + ib)];
                        const int row = A.ao_start + ia;
                        const int col = B.ao_start + ib;
                        V(row, col) += v;
                        if (i != j)
                            V(col, row) += v;
                    }
                }
            }
        }
        if (c2s.size() > 0) {
            // Sandwich to spherical AO basis (same as point-charge embedding).
            const auto ct = xt::transpose(c2s);
            V = xt::linalg::dot(ct, xt::linalg::dot(V, c2s));
            n_ao_ = static_cast<int>(c2s.shape(1));
        }
        V_unit_[static_cast<std::size_t>(k)] = std::move(V);
    }
    if (c2s.size() > 0)
        n_ao_ = static_cast<int>(c2s.shape(1));
}

Pcm::Result Pcm::solve_and_contract_(const std::vector<double>& V) const {
    const int nt = static_cast<int>(pos_.size());
    xt::xtensor<double, 1> rhs = xt::zeros<double>({static_cast<std::size_t>(nt)});
    for (int k = 0; k < nt; ++k)
        rhs(k) = -f_eps_ * V[static_cast<std::size_t>(k)];

    // q = S^{-1} (−f V)
    const xt::xtensor<double, 1> q = xt::linalg::dot(S_inv_, rhs);
    last_q_.resize(static_cast<std::size_t>(nt));
    double dg = 0.0;
    for (int k = 0; k < nt; ++k) {
        last_q_[static_cast<std::size_t>(k)] = q(k);
        dg += q(k) * V[static_cast<std::size_t>(k)];
    }
    dg *= 0.5;

    Result out;
    out.delta_g_el = dg;
    if (n_ao_ > 0 && !V_unit_.empty()) {
        out.V_pcm = xt::zeros<double>({static_cast<std::size_t>(n_ao_),
                                       static_cast<std::size_t>(n_ao_)});
        for (int k = 0; k < nt; ++k)
            out.V_pcm += q(k) * V_unit_[static_cast<std::size_t>(k)];
    } else {
        out.V_pcm = xt::zeros<double>({0, 0});
    }
    return out;
}

Pcm::Result Pcm::update_nuclear_only() const {
    return solve_and_contract_(v_nuc_);
}

Pcm::Result Pcm::update(const xt::xtensor<double, 2>& P_total) const {
    const int nt = static_cast<int>(pos_.size());
    std::vector<double> V = v_nuc_;
    if (n_ao_ > 0 && !V_unit_.empty()) {
        if (static_cast<int>(P_total.shape(0)) != n_ao_
            || static_cast<int>(P_total.shape(1)) != n_ao_)
            throw std::invalid_argument(
                "Pcm::update: P_total shape does not match prepared AO "
                "dimension n_ao="
                + std::to_string(n_ao_));
        for (int k = 0; k < nt; ++k) {
            // V_unit = −⟨1/r⟩ ⇒ Tr[P V_unit] = −v_e; total V = v_nuc − v_e
            // = v_nuc + Tr[P V_unit].
            const double tr =
                xt::sum(P_total * V_unit_[static_cast<std::size_t>(k)])();
            V[static_cast<std::size_t>(k)] += tr;
        }
    }
    return solve_and_contract_(V);
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
