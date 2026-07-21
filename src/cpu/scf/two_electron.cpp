/// @file cpu/scf/two_electron.cpp
/// @brief Dense ERI tensor assembly (IntegralPack::eri_dense) — thin wrapper
///        over native::eri.

#include "atomiverse/cpu/scf/internal/integrals.h"

#include <array>
#include <vector>

#include "atomiverse/core/log.h"
#include "atomiverse/cpu/scf/native_integrals.h"

namespace ATV {
namespace cpu {
namespace scf {

xt::xtensor<double, 4> IntegralPack::eri_dense(const ShellList& shells) {
    const int n = shells.n_basis;
    xt::xtensor<double, 4> ERI = xt::zeros<double>({n, n, n, n});

    const int n_pairs = n * (n + 1) / 2;
    std::vector<std::array<int, 2>> shell_pairs;
    shell_pairs.reserve(static_cast<std::size_t>(n_pairs));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i; ++j)
            shell_pairs.push_back({i, j});

    const auto t_start = LogClock::now();
    for (int pair_ij = 0; pair_ij < n_pairs; ++pair_ij) {
        const int i = shell_pairs[pair_ij][0];
        const int j = shell_pairs[pair_ij][1];

        for (int pair_kl = 0; pair_kl <= pair_ij; ++pair_kl) {
            const int k = shell_pairs[pair_kl][0];
            const int l = shell_pairs[pair_kl][1];
            const double v = native::eri(shells.shells[i], shells.shells[j],
                                         shells.shells[k], shells.shells[l]);

            // 8-fold symmetry scatter.
            ERI(i, j, k, l) = v;
            ERI(j, i, k, l) = v;
            ERI(i, j, l, k) = v;
            ERI(j, i, l, k) = v;
            ERI(k, l, i, j) = v;
            ERI(l, k, i, j) = v;
            ERI(k, l, j, i) = v;
            ERI(l, k, j, i) = v;
        }
    }
    ATV_DEBUG("[eri_dense] basis={} ms={:.1f}", n, elapsed_ms(t_start));

    return ERI;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
