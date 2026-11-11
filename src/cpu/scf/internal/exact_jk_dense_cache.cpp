// Compile the dense cache worker separately to preserve inlining in the
// streamed and sparse cache workers.
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "exact_jk_contraction.h"
#include "exact_jk_symmetric_exchange.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV::cpu::scf::internal::exact_jk_detail {

bool try_contract_dense_cache(int n, int rank, int ranks,
                              const std::vector<double>& eri, int mj, int mk,
                              const double* pj, const double* pk,
                              const double* sk,
                              std::vector<std::vector<double>>& js,
                              std::vector<std::vector<double>>& ks) {
    if (mj != 1 || mk != 1)
        return false;

    for (int a = 0; a < n; ++a)
        for (int b = 0; b < a; ++b)
            if (pk[a + b * n] != pk[b + a * n])
                return false;

    const auto nn = static_cast<std::size_t>(n) * n;
    const auto np = static_cast<std::size_t>(n) * (n + 1) / 2;
    // Fold both orientations even when the Coulomb density is nonsymmetric.
    std::vector<double> p(np);
    std::size_t index = 0;
    for (int a = 0; a < n; ++a) {
        for (int b = 0; b < a; ++b, ++index)
            p[index] = pj[a + b * n] + pj[b + a * n];
        p[index++] = pj[a + a * n];
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        js[tid].assign(nn, 0.0);
        double* const j = js[tid].data();
        ks[tid].assign(nn, 0.0);
        double* const k = ks[tid].data();
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4) nowait
#endif
        for (std::int64_t ab = rank; ab < static_cast<std::int64_t>(np);
             ab += ranks) {
            const int a = static_cast<int>((std::sqrt(8.0 * ab + 1) - 1) / 2);
            const int b = ab - static_cast<std::int64_t>(a) * (a + 1) / 2;
            const double* const row = eri.data() + ab * (ab + 1) / 2;
            const double p_ab = p[ab];
            double j_ab = 0.0;
            std::size_t cd = 0;
            for (int c = 0; c <= a; ++c)
                for (int d = 0; d < (c == a ? b : c + 1); ++d, ++cd) {
                    const double v = row[cd];
                    if (v == 0.0)
                        continue;
                    j_ab += v * p[cd];
                    j[cd] += v * p_ab;
                    exchange_distinct(n, pk, k, v * sk[0], a, b, c, d);
                }
            const double diagonal = row[ab];
            if (diagonal != 0.0) {
                j_ab += diagonal * p_ab;
                exchange_same_pair(n, pk, k, diagonal * sk[0], a, b);
            }
            j[ab] += j_ab;
        }

        // Expand the packed J into the upper triangle from the end.
        // Each destination is at or above its packed source index.
        std::size_t ab = np;
        for (int a = n - 1; a >= 0; --a)
            for (int b = a; b >= 0; --b)
                j[b + a * n] = j[--ab];
        for (int a = 0; a < n; ++a)
            for (int b = 0; b < a; ++b) {
                j[a + b * n] = j[b + a * n];
                k[b + a * n] = k[a + b * n];
            }
    }
    return true;
}

} // namespace ATV::cpu::scf::internal::exact_jk_detail
