/// @file cpu/scf/mp3.cpp
/// @brief Closed-shell canonical MP3 via spin-resolved energy contractions.

#include "atomiverse/cpu/scf/mp3.h"
#include "atomiverse/core/log.h"
#include "atomiverse/cpu/linalg/internal/blas.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/mp3_pack.h"
#include "atomiverse/cpu/scf/internal/spin_orbital_eri.h"

#include <algorithm>
#include <array>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <xtensor/containers/xtensor.hpp>

namespace ATV {
namespace cpu {
namespace scf {

namespace {

using OrbitalPair = std::array<int, 2>;

std::vector<OrbitalPair> orbital_pairs(int first, int last,
                                       bool antisymmetric) {
    std::vector<OrbitalPair> pairs;
    const auto count = static_cast<std::size_t>(last - first);
    pairs.reserve(antisymmetric ? count * (count - 1) / 2 : count * count);
    for (int i = first; i < last; ++i)
        for (int j = first; j < (antisymmetric ? i : last); ++j)
            pairs.push_back({i, j});
    return pairs;
}

// Contract a symmetric interaction with T T^T (or T^T T) without storing
// the interaction matrix. Only one triangle of the Gram matrix is formed.
template <typename Interaction>
double gram_energy(const RealMatrix& t, bool transpose, std::span<double> gram,
                   Interaction interaction) {
    const int rows = static_cast<int>(t.shape(0));
    const int cols = static_cast<int>(t.shape(1));
    const int n = transpose ? cols : rows;
    const int k = transpose ? rows : cols;
    const int panel =
        static_cast<int>(std::min<std::size_t>(n, gram.size() / n));
    double diagonal = 0.0;
    double off_diagonal = 0.0;
    for (int first = 0; first < n; first += panel) {
        const int height = n - first;
        const int width = std::min(panel, height);
        if (panel == n) {
            linalg::internal::syrk_lower(transpose ? 'T' : 'N', n, k, 1.0,
                                         t.data(), rows, 0.0, gram.data(), n);
        } else {
            // Form only the lower block row from this column panel onward.
            const double* start =
                t.data()
                + static_cast<std::size_t>(first) * (transpose ? rows : 1);
            linalg::internal::gemm(transpose ? 'T' : 'N', transpose ? 'N' : 'T',
                                   height, width, k, 1.0, start, rows, start,
                                   rows, 0.0, gram.data(), height);
        }
        for (int j = 0; j < width; ++j) {
            const double* column =
                gram.data() + static_cast<std::size_t>(j) * height;
            diagonal += column[j] * interaction(first + j, first + j);
            for (int i = j + 1; i < height; ++i)
                off_diagonal += column[i] * interaction(first + i, first + j);
        }
    }
    return diagonal + 2.0 * off_diagonal;
}

} // namespace

MP3Result mp3_energy(const internal::IntegralView& ints,
                     const SlaterDeterminant& det, double e_scf_ha,
                     int n_frozen) {
    using Spin = SlaterDeterminant::Spin;
    if (ints.ERI.size() == 0) {
        throw std::invalid_argument("mp3_energy: ERI is not materialized; call "
                                    "internal::IntegralWorkspace{..., "
                                    "compute_eri=true}.");
    }
    if (det.n_electrons(Spin::alpha) != det.n_electrons(Spin::beta)) {
        throw std::invalid_argument(
            "mp3_energy: closed-shell reference only (n_alpha must equal "
            "n_beta).");
    }
    if (det.generalized()) {
        throw std::invalid_argument(
            "mp3_energy: GHF (spinor) reference is not supported.");
    }

    const xt::xtensor<double, 2> C = det.coefficients(Spin::alpha);
    const int n_occ = det.n_electrons(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape(1));
    const int n_vir = n_mo - n_occ;

    if (n_frozen < 0 || n_frozen > n_occ) {
        throw std::invalid_argument(
            "mp3_energy: n_frozen must satisfy 0 <= n_frozen <= n_occ.");
    }
    if (n_vir == 0 || n_frozen == n_occ) {
        return {/*e_corr=*/0.0, /*e_tot=*/e_scf_ha, /*e_corr_mp2=*/0.0,
                /*e_corr_mp3=*/0.0, n_frozen};
    }

    const int n_active = n_occ - n_frozen;
    const int max_orbitals = std::max(n_active, n_vir);
    if (max_orbitals > std::numeric_limits<int>::max() / max_orbitals)
        throw std::overflow_error(
            "mp3_energy: pair dimensions exceed BLAS integer range");

    const auto start = LogClock::now();
    // Full MO ERI in chemist notation (pq|rs).
    const xt::xtensor<double, 4> g = ao2mo_full(ints.ERI, C, n_mo);
    const auto transformed = LogClock::now();
    const xt::xtensor<double, 1>& eps = det.orbital_energies(Spin::alpha);
    const auto occupied = orbital_pairs(n_frozen, n_occ, false);
    const auto virtuals = orbital_pairs(n_occ, n_mo, false);
    const auto gram_dim = std::max(occupied.size(), virtuals.size());
    // A highly unequal occupied/virtual split can make a full Gram as
    // large as the MO integral tensor. Bound its storage by using panels.
    constexpr std::size_t gram_budget = 32 * 1024 * 1024 / sizeof(double);
    auto gram = xt::xtensor<double, 1>::from_shape(
        {std::min(gram_dim * gram_dim, std::max(gram_dim, gram_budget))});
    const std::span<double> scratch{gram.data(), gram.size()};

    // Unique spin-orbital pairs remove the ordered-pair factors in the
    // ladders. The alpha-beta block contains all ordered spatial pairs.
    auto mixed = RealMatrix::from_shape({occupied.size(), virtuals.size()});
    double e2 = 0.0;
    for (std::size_t col = 0; col < virtuals.size(); ++col) {
        const auto [a, b] = virtuals[col];
        for (std::size_t row = 0; row < occupied.size(); ++row) {
            const auto [i, j] = occupied[row];
            const double eri = g(i, a, j, b);
            const double t =
                eri / mp_denominator(eps(i), eps(j), eps(a), eps(b));
            mixed(row, col) = t;
            e2 += t * eri;
        }
    }

    const auto ladders = [&](const RealMatrix& t, const auto& occ,
                             const auto& vir, bool same_spin) {
        const auto interaction = [&](const auto& pairs, int row, int col) {
            const auto [p, q] = pairs[row];
            const auto [r, s] = pairs[col];
            const double direct = g(p, r, q, s);
            return same_spin ? direct - g(p, s, q, r) : direct;
        };
        const double occ_energy =
            gram_energy(t, false, scratch, [&](int row, int col) {
                return interaction(occ, row, col);
            });
        return occ_energy
               + gram_energy(t, true, scratch, [&](int row, int col) {
                     return interaction(vir, row, col);
                 });
    };
    double e3 = ladders(mixed, occupied, virtuals, false);
    const auto amplitude = [&](int i, int j, int a, int b) {
        return mixed((i - n_frozen) * n_active + (j - n_frozen),
                     (a - n_occ) * n_vir + (b - n_occ));
    };
    if (n_active > 1 && n_vir > 1) {
        // Alpha-alpha and beta-beta have the same antisymmetric block.
        const auto occ = orbital_pairs(n_frozen, n_occ, true);
        const auto vir = orbital_pairs(n_occ, n_mo, true);
        auto same = RealMatrix::from_shape({occ.size(), vir.size()});
        double same_e2 = 0.0;
        for (std::size_t col = 0; col < vir.size(); ++col) {
            const auto [a, b] = vir[col];
            for (std::size_t row = 0; row < occ.size(); ++row) {
                const auto [i, j] = occ[row];
                const double t = amplitude(i, j, a, b) - amplitude(i, j, b, a);
                same(row, col) = t;
                same_e2 += t * (g(i, a, j, b) - g(i, b, j, a));
            }
        }
        e2 += 2.0 * same_e2;
        e3 += 2.0 * ladders(same, occ, vir, true);
    }

    // In the scalar ring energy the four external permutations contract
    // equally with antisymmetric doubles and cancel the spin-orbital 1/4.
    // In (ia,jb) order write D=t_ijab and X=t_ijba for alpha-beta amplitudes.
    // The spin-symmetric particle-hole sector uses 2D-X. The spin difference
    // and two spin-flip sectors give three identical exchange contractions.
    const auto nov = static_cast<std::size_t>(n_active) * n_vir;
    auto ring = RealMatrix::from_shape({nov, nov});
    const auto ring_energy = [&](bool spin_symmetric) {
        for (std::size_t col = 0; col < nov; ++col) {
            const int j = n_frozen + static_cast<int>(col / n_vir);
            const int b = n_occ + static_cast<int>(col % n_vir);
            for (std::size_t row = 0; row < nov; ++row) {
                const int i = n_frozen + static_cast<int>(row / n_vir);
                const int a = n_occ + static_cast<int>(row % n_vir);
                const double exchange = amplitude(i, j, b, a);
                ring(row, col) = spin_symmetric
                                     ? 2.0 * amplitude(i, j, a, b) - exchange
                                     : exchange;
            }
        }
        return gram_energy(ring, false, scratch, [&](int row, int col) {
            const int i = n_frozen + row / n_vir, a = n_occ + row % n_vir;
            const int k = n_frozen + col / n_vir, c = n_occ + col % n_vir;
            const double exchange = g(a, c, k, i);
            return spin_symmetric ? 2.0 * g(a, i, k, c) - exchange : exchange;
        });
    };
    e3 += ring_energy(true);
    e3 -= 3.0 * ring_energy(false);

    ATV_DEBUG("[mp3] ao2mo_ms={:.3f} contractions_ms={:.3f}",
              elapsed_ms(start, transformed), elapsed_ms(transformed));
    const double e_corr = e2 + e3;
    return {e_corr, e_scf_ha + e_corr, e2, e3, n_frozen};
}

} // namespace scf
} // namespace cpu
} // namespace ATV
