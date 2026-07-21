/// @file cpu/scf/internal/point_charges.cpp
/// @brief PointChargeEmbedding: nuclear-attraction machinery with charge
///        scaling + classical QM–MM / MM–MM sums.

#include "atomiverse/cpu/scf/internal/point_charges.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/internal/shell_pair.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace internal {

namespace {

bool finite3(double x, double y, double z) {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
}

} // namespace

PointChargeEmbedding::PointChargeEmbedding(
    const std::vector<PointChargeSite>& sites) {
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
                            q_[c] * buf[static_cast<std::size_t>(ia * B.n_cart
                                                                  + ib)];
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

} // namespace internal
} // namespace scf
} // namespace cpu
} // namespace ATV
