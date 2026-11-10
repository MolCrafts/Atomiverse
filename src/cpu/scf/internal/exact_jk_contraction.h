#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "atomiverse/cpu/scf/internal/shell_pair.h"

namespace ATV::cpu::scf::internal::exact_jk_detail {
static inline std::size_t pair_index(std::size_t a, std::size_t b) {
    if (a < b)
        std::swap(a, b);
    return a * (a + 1) / 2 + b;
}

// Distinct permutations only; no density symmetry assumption. The same
// contraction handles RHF, open-shell and all four real GHF spin blocks.
// Densities and outputs are interleaved, element-major ([i + j n][m] for m
// densities), so one integral updates contiguous memory for every density.
// J is symmetric even for nonsymmetric densities: combine the two density
// orientations before scattering.
template <int M>
static inline void contract_j_n(int n, int m, const double* p, double* j, int a,
                                int b, int c, int d, double v) {
    if constexpr (M > 0)
        m = M;
    const auto at = [n, m](int x, int y) {
        return static_cast<std::size_t>(x + y * n) * m;
    };
    const double* pcd = p + at(c, d);
    const double* pdc = p + at(d, c);
    const double* pab = p + at(a, b);
    const double* pba = p + at(b, a);
    double* jab = j + at(a, b);
    double* jba = j + at(b, a);
    double* jcd = j + at(c, d);
    double* jdc = j + at(d, c);
    const bool cd_off = c != d, ab_off = a != b, mirror = a != c || b != d;
    for (int t = 0; t < m; ++t) {
        const double x = v * (pcd[t] + (cd_off ? pdc[t] : 0));
        jab[t] += x;
        if (ab_off)
            jba[t] += x;
        if (mirror) {
            const double y = v * (pab[t] + (ab_off ? pba[t] : 0));
            jcd[t] += y;
            if (cd_off)
                jdc[t] += y;
        }
    }
}

// K uses all eight permutations; `scale[t]` is density t's exchange factor.
template <int M>
static inline void contract_k_n(int n, int m, const double* p, double* k,
                                const double* scale, int a, int b, int c, int d,
                                double v) {
    if constexpr (M > 0)
        m = M;
    const auto add = [&](int i, int q, int r, int s) {
        double* out = k + static_cast<std::size_t>(i + r * n) * m;
        const double* in = p + static_cast<std::size_t>(q + s * n) * m;
        for (int t = 0; t < m; ++t)
            out[t] += v * scale[t] * in[t];
    };
    add(a, b, c, d);
    if (a != b)
        add(b, a, c, d);
    if (c != d)
        add(a, b, d, c);
    if (a != b && c != d)
        add(b, a, d, c);
    if (a != c || b != d) {
        add(c, d, a, b);
        if (c != d)
            add(d, c, a, b);
        if (a != b)
            add(c, d, b, a);
        if (a != b && c != d)
            add(d, c, b, a);
    }
}

bool try_contract_cached(const ShellBlockPairList& pairs, int n, int rank,
                         int ranks, const std::vector<double>& eri, int mj,
                         int mk, const double* pj, const double* pk,
                         const double* sk, std::vector<std::vector<double>>& js,
                         std::vector<std::vector<double>>& ks);

bool try_contract_dense_cache(int n, int rank, int ranks,
                              const std::vector<double>& eri, int mj, int mk,
                              const double* pj, const double* pk,
                              const double* sk,
                              std::vector<std::vector<double>>& js,
                              std::vector<std::vector<double>>& ks);

bool try_contract_screened_symmetric_cache(
    int n, int rank, int ranks, const std::vector<double>& eri,
    const double* pj, const double* pk, const double* sk,
    std::vector<std::vector<double>>& js, std::vector<std::vector<double>>& ks,
    std::span<const std::uint8_t> active_rows);

} // namespace ATV::cpu::scf::internal::exact_jk_detail
