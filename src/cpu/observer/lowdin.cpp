/// @file cpu/observer/lowdin.cpp
/// @brief Löwdin population analysis implementation. @see lowdin.h

#include "atomiverse/cpu/observer/lowdin.h"

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
using scf::native::ShellBlock;

/// Tolerance for matching a shell center to an atom position [Bohr]. Both
/// come from the same Frame → Bohr conversion, so agreement is bit-near.
constexpr double kCenterTol = 1e-8;

/// @brief Per-atom trace Σ_{μ∈A} (S^{1/2} P S^{1/2})_{μμ} for one density P.
///
/// Rebuilds S from the handle's Cartesian shells, sandwiches to the
/// determinant dimension for spherical packs, forms S^{1/2} by symmetric
/// eigendecomposition, and scatters the diagonal per atom using a
/// block-wise AO → atom map (each Cartesian block maps to one atom; in the
/// spherical dimension its 2l+1 columns map to the same atom — block order
/// is preserved by the block-diagonal c2s).
std::vector<double> lowdin_populations(const MeanFieldOutputs& outputs,
                                       const Frame& frame,
                                       const RealMatrix& P_in) {
    const auto& shells = *outputs.shells;
    const auto blocks = scf::blocks_from_shells(shells);
    const int n_blocks = static_cast<int>(blocks.size());
    const int n_cart = static_cast<int>(shells.size());

    // ── Overlap in the engine's Cartesian convention ──────────────────
    RealMatrix S_cart = xt::zeros<double>({n_cart, n_cart});
    std::vector<double> buf;
    for (int i = 0; i < n_blocks; ++i) {
        for (int j = 0; j <= i; ++j) {
            const ShellBlock& A = blocks[static_cast<std::size_t>(i)];
            const ShellBlock& B = blocks[static_cast<std::size_t>(j)];
            buf.assign(static_cast<std::size_t>(A.n_cart * B.n_cart), 0.0);
            scf::native::overlap_block(A, B, buf.data());
            for (int ia = 0; ia < A.n_cart; ++ia)
                for (int ib = 0; ib < B.n_cart; ++ib) {
                    const double v = buf[ia * B.n_cart + ib];
                    S_cart(A.ao_start + ia, B.ao_start + ib) = v;
                    S_cart(B.ao_start + ib, A.ao_start + ia) = v;
                }
        }
    }

    // ── Sandwich to the determinant's AO dimension (spherical packs) ──
    const bool sph = outputs.c2s && outputs.c2s->size() > 0;
    RealMatrix S = S_cart;
    if (sph) {
        const auto& C = *outputs.c2s;
        S = xt::linalg::dot(xt::transpose(C),
                            RealMatrix(xt::linalg::dot(S_cart, C)));
    }
    const int n_ao = static_cast<int>(S.shape()[0]);
    if (static_cast<int>(P_in.shape()[0]) != n_ao)
        throw std::runtime_error(
            "lowdin: density dimension " + std::to_string(P_in.shape()[0])
            + " != AO dimension " + std::to_string(n_ao));

    // ── Unit-normalize each AO (S_ii = 1) before orthogonalization ────
    // Löwdin populations are NOT invariant under per-AO rescaling (unlike
    // Mulliken/ESP), and Cartesian d/f engine conventions differ across
    // codes in exactly such diagonal factors. Analyzing in the normalized
    // AO basis (S̄ = D S D, P̄ = D⁻¹ P D⁻¹, D = diag(1/√S_ii)) makes the
    // result convention-independent; Tr(P̄S̄) = Tr(PS) is untouched.
    RealMatrix P_bar = P_in;
    for (int i = 0; i < n_ao; ++i) {
        if (S(i, i) <= 0.0)
            throw std::runtime_error(
                "lowdin: AO self-overlap S(" + std::to_string(i) + ","
                + std::to_string(i) + ") = " + std::to_string(S(i, i))
                + " <= 0");
        const double di = std::sqrt(S(i, i));
        for (int j = 0; j < n_ao; ++j) {
            const double dj = std::sqrt(S(j, j));
            P_bar(i, j) = P_in(i, j) * di * dj;
            if (j != i)
                S(i, j) /= di * dj;
        }
    }
    for (int i = 0; i < n_ao; ++i)
        S(i, i) = 1.0;
    const RealMatrix& P = P_bar;

    // ── S^{1/2} via symmetric eigendecomposition ──────────────────────
    auto [s_eig, U] = xt::linalg::eigh(S);
    RealMatrix Us = xt::zeros<double>({n_ao, n_ao});
    for (int j = 0; j < n_ao; ++j) {
        if (s_eig(j) <= 0.0)
            throw std::runtime_error(
                "lowdin: overlap eigenvalue " + std::to_string(s_eig(j))
                + " <= 0 — basis is numerically linearly dependent");
        const double sq = std::sqrt(s_eig(j));
        for (int i = 0; i < n_ao; ++i)
            Us(i, j) = U(i, j) * sq;
    }
    const RealMatrix S_half = xt::linalg::dot(Us, xt::transpose(U));

    // ── Block-wise AO → atom map, built in the target dimension ───────
    const int natm = frame.nlocal();
    const auto x = frame.span(kAtom_x);
    const auto y = frame.span(kAtom_y);
    const auto z = frame.span(kAtom_z);
    std::vector<int> ao_to_atom(static_cast<std::size_t>(n_ao), -1);
    int ao = 0;
    for (int b = 0; b < n_blocks; ++b) {
        const ShellBlock& blk = blocks[static_cast<std::size_t>(b)];
        int atom = -1;
        for (int A = 0; A < natm; ++A) {
            const double pos[3] = {x[A] * units::kAngstromInBohr,
                                   y[A] * units::kAngstromInBohr,
                                   z[A] * units::kAngstromInBohr};
            if (std::abs(blk.center[0] - pos[0]) < kCenterTol
                && std::abs(blk.center[1] - pos[1]) < kCenterTol
                && std::abs(blk.center[2] - pos[2]) < kCenterTol) {
                atom = A;
                break;
            }
        }
        if (atom < 0)
            throw std::runtime_error(
                "lowdin: shell block " + std::to_string(b)
                + " center matches no atom position in the Frame");
        const int n_fn = sph ? (2 * blk.l + 1) : blk.n_cart;
        for (int k = 0; k < n_fn; ++k)
            ao_to_atom[static_cast<std::size_t>(ao + k)] = atom;
        ao += n_fn;
    }
    if (ao != n_ao)
        throw std::runtime_error(
            "lowdin: AO map covers " + std::to_string(ao) + " of "
            + std::to_string(n_ao) + " functions");

    // ── diag(S^{1/2} P S^{1/2}) scattered per atom ────────────────────
    const RealMatrix M = xt::linalg::dot(
        S_half, RealMatrix(xt::linalg::dot(P, S_half)));
    std::vector<double> pop(static_cast<std::size_t>(natm), 0.0);
    for (int mu = 0; mu < n_ao; ++mu)
        pop[static_cast<std::size_t>(ao_to_atom[static_cast<std::size_t>(mu)])]
            += M(mu, mu);
    return pop;
}

/// @brief Common handle guard for the compute entries.
void check_handle(const MeanFieldOutputs& outputs) {
    if (!outputs.result || !outputs.shells || !outputs.ready)
        throw std::invalid_argument(
            "lowdin: null MeanFieldOutputs (did you forget "
            "method.result_handle()?)");
    if (!*outputs.ready)
        throw std::invalid_argument(
            "lowdin: MeanFieldOutputs not ready — run the method first");
}

} // namespace

std::vector<double> lowdin_charge(const scf::MeanFieldOutputs& outputs,
                                  const Frame& frame) {
    check_handle(outputs);
    const auto pop =
        lowdin_populations(outputs, frame, outputs.result->density_total());
    const auto type = frame.span(kAtom_type);
    std::vector<double> q(pop.size());
    for (std::size_t A = 0; A < pop.size(); ++A)
        q[A] = static_cast<double>(type[static_cast<int>(A)]) - pop[A];
    return q;
}

std::vector<double> lowdin_spin(const scf::MeanFieldOutputs& outputs,
                                const Frame& frame) {
    check_handle(outputs);
    using Spin = scf::SlaterDeterminant::Spin;
    const auto& det = *outputs.result;
    const RealMatrix P_diff = det.density(Spin::alpha) - det.density(Spin::beta);
    return lowdin_populations(outputs, frame, P_diff);
}

} // namespace cpu
} // namespace ATV
