/// @file cpu/scf/internal/point_charges.cpp
/// @brief PointChargeEmbedding: nuclear-attraction machinery with charge
///        scaling, classical QM–MM / MM–MM sums, and analytic gradients.

#include "atomiverse/cpu/scf/internal/point_charges.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/cart.h"
#include "atomiverse/cpu/scf/internal/integral_derivatives.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

bool finite3(double x, double y, double z) {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

/// Decode `sh.cart` into (l_x, l_y, l_z) — same as integral_derivatives.cpp.
void cart_exponents(const native::Shell& sh, int a_out[3]) {
    int comps[kMaxNCart][3];
    fill_cart_comps(sh.l, comps);
    a_out[0] = comps[sh.cart][0];
    a_out[1] = comps[sh.cart][1];
    a_out[2] = comps[sh.cart][2];
}

} // namespace

PointChargeEmbedding::PointChargeEmbedding(
    const std::vector<cpu::EmbeddingSite>& sites) {
    pos_bohr_.reserve(sites.size());
    q_.reserve(sites.size());
    for (const auto& s : sites) {
        if (!finite3(s.x, s.y, s.z) || !std::isfinite(s.q))
            throw std::invalid_argument(
                "PointChargeEmbedding: non-finite position or charge "
                "(positions [Å], charge [e])");
        pos_bohr_.push_back({s.x * units::kAngstromInBohr,
                             s.y * units::kAngstromInBohr,
                             s.z * units::kAngstromInBohr});
        q_.push_back(s.q);
    }
}

xt::xtensor<double, 2> PointChargeEmbedding::core_matrix(
    const std::vector<native::Shell>& shells) const {
    const int n = static_cast<int>(shells.size());
    xt::xtensor<double, 2> V = xt::zeros<double>({n, n});
    if (q_.empty() || n == 0)
        return V;

    const auto blocks = blocks_from_shells(shells);
    const int n_blocks = static_cast<int>(blocks.size());
    std::vector<double> buf;

    // V_unit(Z=1) already carries −⟨1/r⟩; scale by q → −q⟨1/r⟩.
    for (std::size_t c = 0; c < q_.size(); ++c) {
        if (q_[c] == 0.0)
            continue;
        const native::Atom probe{
            1, {pos_bohr_[c][0], pos_bohr_[c][1], pos_bohr_[c][2]}};
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
                            q_[c]
                            * buf[static_cast<std::size_t>(ia * B.n_cart + ib)];
                        const int row = A.ao_start + ia;
                        const int col = B.ao_start + ib;
                        V(row, col) += v;
                        if (i != j)
                            V(col, row) += v;
                    }
                }
            }
        }
    }
    return V;
}

double PointChargeEmbedding::nuclear_energy(
    const std::vector<native::Atom>& atoms) const {
    double e = 0.0;
    for (const auto& A : atoms) {
        for (std::size_t c = 0; c < q_.size(); ++c) {
            const double dx = A.pos[0] - pos_bohr_[c][0];
            const double dy = A.pos[1] - pos_bohr_[c][1];
            const double dz = A.pos[2] - pos_bohr_[c][2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r == 0.0)
                throw std::invalid_argument(
                    "PointChargeEmbedding::nuclear_energy: charge coincides "
                    "with a nucleus (singular 1/r)");
            e += static_cast<double>(A.Z) * q_[c] / r;
        }
    }
    return e;
}

double PointChargeEmbedding::self_energy() const {
    double e = 0.0;
    const int n = static_cast<int>(q_.size());
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dx = pos_bohr_[static_cast<std::size_t>(i)][0]
                              - pos_bohr_[static_cast<std::size_t>(j)][0];
            const double dy = pos_bohr_[static_cast<std::size_t>(i)][1]
                              - pos_bohr_[static_cast<std::size_t>(j)][1];
            const double dz = pos_bohr_[static_cast<std::size_t>(i)][2]
                              - pos_bohr_[static_cast<std::size_t>(j)][2];
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (r == 0.0)
                throw std::invalid_argument(
                    "PointChargeEmbedding::self_energy: two charges coincide "
                    "(singular 1/r)");
            e += q_[static_cast<std::size_t>(i)]
                 * q_[static_cast<std::size_t>(j)] / r;
        }
    }
    return e;
}

void PointChargeEmbedding::accumulate_qm_center_gradient(
    xt::xtensor<double, 2>& grad, const std::vector<native::Shell>& shells,
    const xt::xtensor<double, 2>& P_cart, const std::vector<int>& shell_to_atom,
    const std::vector<native::Atom>& atoms) const {
    const int n = static_cast<int>(shells.size());
    if (q_.empty() || n == 0)
        return;
    if (static_cast<int>(shell_to_atom.size()) != n)
        throw std::invalid_argument(
            "PointChargeEmbedding::accumulate_qm_center_gradient: "
            "shell_to_atom size mismatch");
    if (static_cast<int>(P_cart.shape(0)) != n
        || static_cast<int>(P_cart.shape(1)) != n)
        throw std::invalid_argument(
            "PointChargeEmbedding::accumulate_qm_center_gradient: "
            "P_cart must be n_cart × n_cart");

    // Electron–MM: Tr[P · dV/dR_A] with V = Σ_C −q_C ⟨1/r_C⟩.
    // Bare nuclear_deriv omits −Z; we multiply by −q (same as Z=q nucleus).
    for (std::size_t c = 0; c < q_.size(); ++c) {
        if (q_[c] == 0.0)
            continue;
        const double C[3] = {pos_bohr_[c][0], pos_bohr_[c][1], pos_bohr_[c][2]};
        const double mq = -q_[c]; // −q prefactor
        for (int i = 0; i < n; ++i) {
            int a_ang[3];
            cart_exponents(shells[static_cast<std::size_t>(i)], a_ang);
            const int atom_i = shell_to_atom[static_cast<std::size_t>(i)];
            for (int j = 0; j < n; ++j) {
                int b_ang[3];
                cart_exponents(shells[static_cast<std::size_t>(j)], b_ang);
                const int atom_j = shell_to_atom[static_cast<std::size_t>(j)];
                const double Pij = P_cart(i, j);

                double dV_dA[3] = {0, 0, 0};
                double dV_dB[3] = {0, 0, 0};
                const auto& si = shells[static_cast<std::size_t>(i)];
                const auto& sj = shells[static_cast<std::size_t>(j)];
                for (int p = 0; p < si.n_prim; ++p) {
                    for (int q = 0; q < sj.n_prim; ++q) {
                        const double cij = si.coeff[p] * sj.coeff[q];
                        double dA[3], dB[3];
                        nuclear_deriv_prim_center_general(
                            a_ang, b_ang, si.alpha[p], si.center, sj.alpha[q],
                            sj.center, C, dA);
                        // d/dB: raise on B (swap roles)
                        nuclear_deriv_prim_center_general(
                            b_ang, a_ang, sj.alpha[q], sj.center, si.alpha[p],
                            si.center, C, dB);
                        for (int d = 0; d < 3; ++d) {
                            dV_dA[d] += cij * mq * dA[d];
                            dV_dB[d] += cij * mq * dB[d];
                        }
                    }
                }
                for (int d = 0; d < 3; ++d) {
                    grad(atom_i, d) += Pij * dV_dA[d];
                    grad(atom_j, d) += Pij * dV_dB[d];
                }
            }
        }
    }

    // Nucleus–MM classical: ∂/∂R_A of Σ Z_A q_C / |R_A−R_C|
    //   = − Z_A q_C (R_A − R_C) / r³
    for (std::size_t a = 0; a < atoms.size(); ++a) {
        const double* ap = atoms[a].pos;
        const double Za = static_cast<double>(atoms[a].Z);
        for (std::size_t c = 0; c < q_.size(); ++c) {
            if (q_[c] == 0.0)
                continue;
            const double dx = ap[0] - pos_bohr_[c][0];
            const double dy = ap[1] - pos_bohr_[c][1];
            const double dz = ap[2] - pos_bohr_[c][2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 == 0.0)
                throw std::invalid_argument(
                    "PointChargeEmbedding::accumulate_qm_center_gradient: "
                    "charge coincides with a nucleus");
            const double r3 = r2 * std::sqrt(r2);
            const double zq = Za * q_[c];
            grad(static_cast<int>(a), 0) += -zq * dx / r3;
            grad(static_cast<int>(a), 1) += -zq * dy / r3;
            grad(static_cast<int>(a), 2) += -zq * dz / r3;
        }
    }
}

void PointChargeEmbedding::accumulate_site_gradient(
    xt::xtensor<double, 2>& site_grad, const std::vector<native::Shell>& shells,
    const xt::xtensor<double, 2>& P_cart,
    const std::vector<native::Atom>& atoms) const {
    const int n = static_cast<int>(shells.size());
    const int nc = static_cast<int>(q_.size());
    if (nc == 0 || n == 0)
        return;
    if (static_cast<int>(site_grad.shape(0)) != nc
        || static_cast<int>(site_grad.shape(1)) != 3)
        throw std::invalid_argument(
            "PointChargeEmbedding::accumulate_site_gradient: "
            "site_grad must be [n_charges, 3]");
    if (static_cast<int>(P_cart.shape(0)) != n
        || static_cast<int>(P_cart.shape(1)) != n)
        throw std::invalid_argument(
            "PointChargeEmbedding::accumulate_site_gradient: "
            "P_cart must be n_cart × n_cart");

    // Electron–MM: Tr[P · dV/dR_C], dV/dC bare * (−q)
    for (std::size_t c = 0; c < q_.size(); ++c) {
        if (q_[c] == 0.0)
            continue;
        const double C[3] = {pos_bohr_[c][0], pos_bohr_[c][1], pos_bohr_[c][2]};
        const double mq = -q_[c];
        for (int i = 0; i < n; ++i) {
            int a_ang[3];
            cart_exponents(shells[static_cast<std::size_t>(i)], a_ang);
            const auto& si = shells[static_cast<std::size_t>(i)];
            for (int j = 0; j < n; ++j) {
                int b_ang[3];
                cart_exponents(shells[static_cast<std::size_t>(j)], b_ang);
                const auto& sj = shells[static_cast<std::size_t>(j)];
                const double Pij = P_cart(i, j);
                double dV_dC[3] = {0, 0, 0};
                for (int p = 0; p < si.n_prim; ++p) {
                    for (int q = 0; q < sj.n_prim; ++q) {
                        const double cij = si.coeff[p] * sj.coeff[q];
                        double dC[3];
                        nuclear_deriv_prim_nuc_general(
                            a_ang, b_ang, si.alpha[p], si.center, sj.alpha[q],
                            sj.center, C, dC);
                        for (int d = 0; d < 3; ++d)
                            dV_dC[d] += cij * mq * dC[d];
                    }
                }
                for (int d = 0; d < 3; ++d)
                    site_grad(static_cast<int>(c), d) += Pij * dV_dC[d];
            }
        }
    }

    // Nucleus–MM classical on the site: ∂/∂R_C = + Z_A q_C (R_A − R_C) / r³
    // = − ∂/∂R_A of the same term.
    for (std::size_t a = 0; a < atoms.size(); ++a) {
        const double* ap = atoms[a].pos;
        const double Za = static_cast<double>(atoms[a].Z);
        for (std::size_t c = 0; c < q_.size(); ++c) {
            if (q_[c] == 0.0)
                continue;
            const double dx = ap[0] - pos_bohr_[c][0];
            const double dy = ap[1] - pos_bohr_[c][1];
            const double dz = ap[2] - pos_bohr_[c][2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 == 0.0)
                throw std::invalid_argument(
                    "PointChargeEmbedding::accumulate_site_gradient: "
                    "charge coincides with a nucleus");
            const double r3 = r2 * std::sqrt(r2);
            const double zq = Za * q_[c];
            site_grad(static_cast<int>(c), 0) += zq * dx / r3;
            site_grad(static_cast<int>(c), 1) += zq * dy / r3;
            site_grad(static_cast<int>(c), 2) += zq * dz / r3;
        }
    }
}

void PointChargeEmbedding::accumulate_self_gradient(
    xt::xtensor<double, 2>& site_grad) const {
    const int n = static_cast<int>(q_.size());
    if (n < 2)
        return;
    if (static_cast<int>(site_grad.shape(0)) != n
        || static_cast<int>(site_grad.shape(1)) != 3)
        throw std::invalid_argument(
            "PointChargeEmbedding::accumulate_self_gradient: "
            "site_grad must be [n_charges, 3]");

    // E = Σ_{i<j} q_i q_j / r_ij
    // ∂E/∂R_i = −Σ_{j≠i} q_i q_j (R_i − R_j) / r³
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double dx = pos_bohr_[static_cast<std::size_t>(i)][0]
                              - pos_bohr_[static_cast<std::size_t>(j)][0];
            const double dy = pos_bohr_[static_cast<std::size_t>(i)][1]
                              - pos_bohr_[static_cast<std::size_t>(j)][1];
            const double dz = pos_bohr_[static_cast<std::size_t>(i)][2]
                              - pos_bohr_[static_cast<std::size_t>(j)][2];
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 == 0.0)
                throw std::invalid_argument(
                    "PointChargeEmbedding::accumulate_self_gradient: "
                    "two charges coincide");
            const double r3 = r2 * std::sqrt(r2);
            const double qq = q_[static_cast<std::size_t>(i)]
                              * q_[static_cast<std::size_t>(j)];
            const double gx = -qq * dx / r3;
            const double gy = -qq * dy / r3;
            const double gz = -qq * dz / r3;
            site_grad(i, 0) += gx;
            site_grad(i, 1) += gy;
            site_grad(i, 2) += gz;
            site_grad(j, 0) -= gx;
            site_grad(j, 1) -= gy;
            site_grad(j, 2) -= gz;
        }
    }
}

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
