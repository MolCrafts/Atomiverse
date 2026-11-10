// Compile the fixed-lane cache worker separately to preserve inlining in
// the general cached and streamed contractions.
#include "exact_jk_contraction.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ATV::cpu::scf::internal::exact_jk_detail {
namespace {
template <int K>
void contract_screened_rows(int n, int rank, int ranks, const double* eri,
                            const std::uint8_t* active_rows, const double* pj,
                            double* jt, const double* pk, double* kt,
                            const double* sk) {
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 4)
#endif
    for (std::int64_t ab = rank;
         ab < static_cast<std::int64_t>(n) * (n + 1) / 2; ab += ranks) {
        if (!active_rows[ab])
            continue;
        const int a = static_cast<int>((std::sqrt(8.0 * ab + 1) - 1) / 2);
        const int b = ab - static_cast<std::int64_t>(a) * (a + 1) / 2;
        const double* row = eri + ab * (ab + 1) / 2;
        std::size_t cd = 0;
        for (int c = 0; c <= a; ++c)
            for (int d = 0; d <= (c == a ? b : c); ++d, ++cd) {
                const double v = row[cd];
                if (v == 0.0)
                    continue;
                contract_j_n<1>(n, 1, pj, jt, a, b, c, d, v);
                contract_k_n<K>(n, K, pk, kt, sk, a, b, c, d, v);
            }
    }
}

bool try_contract_screened_cache(const ShellBlockPairList& pairs, int n,
                                 int rank, int ranks,
                                 const std::vector<double>& eri, int mj, int mk,
                                 const double* pj, const double* pk,
                                 const double* sk,
                                 std::vector<std::vector<double>>& js,
                                 std::vector<std::vector<double>>& ks) {
    const auto blocks = pairs.blocks.size();
    if (mj != 1 || (mk != 1 && mk != 2 && mk != 4)
        || pairs.pairs.size() == blocks * (blocks + 1) / 2)
        return false;
    // A shell pair excluded during cache construction leaves its AO-pair
    // rows zero. Mark the rows that can contribute, retaining contiguous
    // traversal within each row. Dense-input caches have no shell pairs.
    std::vector<std::uint8_t> active_rows;
    const bool spherical = n != pairs.n_basis;
    std::vector<int> offsets(blocks + 1);
    for (std::size_t s = 0; s < blocks; ++s) {
        const auto& block = pairs.blocks[s];
        offsets[s + 1] =
            offsets[s] + (spherical ? 2 * block.l + 1 : block.n_cart);
    }
    const auto nao = static_cast<std::size_t>(n);
    active_rows.resize(nao * (nao + 1) / 2, 0);
    for (const auto& pair : pairs.pairs) {
        const int first = offsets[pair.sj];
        for (int a = offsets[pair.si]; a < offsets[pair.si + 1]; ++a) {
            const int end = pair.si == pair.sj ? a + 1 : offsets[pair.sj + 1];
            std::fill_n(active_rows.data() + pair_index(a, first), end - first,
                        std::uint8_t{1});
        }
    }
    // Prefer the screened worker when the omitted rows account for at least
    // half of the packed slots.
    std::size_t skipped = 0;
    for (std::size_t ab = 0; ab < active_rows.size(); ++ab)
        if (!active_rows[ab])
            skipped += ab + 1;
    if (skipped < eri.size() - skipped)
        return false;

    if (mk == 1
        && try_contract_screened_symmetric_cache(n, rank, ranks, eri, pj, pk,
                                                 sk, js, ks, active_rows))
        return true;

    using Worker = decltype(&contract_screened_rows<1>);
    const Worker worker = mk == 1   ? contract_screened_rows<1>
                          : mk == 2 ? contract_screened_rows<2>
                                    : contract_screened_rows<4>;
    const std::size_t nn = static_cast<std::size_t>(n) * n;
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        js[tid].assign(nn, 0.0);
        ks[tid].assign(nn * mk, 0.0);
        worker(n, rank, ranks, eri.data(), active_rows.data(), pj,
               js[tid].data(), pk, ks[tid].data(), sk);
    }
    return true;
}
} // namespace

bool try_contract_cached(const ShellBlockPairList& pairs, int n, int rank,
                         int ranks, const std::vector<double>& eri, int mj,
                         int mk, const double* pj, const double* pk,
                         const double* sk, std::vector<std::vector<double>>& js,
                         std::vector<std::vector<double>>& ks) {
    return try_contract_screened_cache(pairs, n, rank, ranks, eri, mj, mk, pj,
                                       pk, sk, js, ks)
           || try_contract_dense_cache(n, rank, ranks, eri, mj, mk, pj, pk, sk,
                                       js, ks);
}

} // namespace ATV::cpu::scf::internal::exact_jk_detail
