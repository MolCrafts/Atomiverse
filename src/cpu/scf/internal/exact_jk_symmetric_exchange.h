#pragma once

#include <algorithm>

namespace ATV::cpu::scf::internal::exact_jk_detail {
namespace {

void exchange_distinct(int n, const double* p, double* k, double w, int a,
                       int b, int c, int d) {
    // For distinct pair rows, pair exchange supplies each transpose.
    // Keep one triangle, counting both contributions to its diagonal.
    const auto add = [=](int i, int q, int r, int s) {
        auto& out = k[std::max(i, r) + std::min(i, r) * n];
        const double update = w * p[q + s * n];
        out += update;
        if (i == r)
            out += update;
    };
    add(a, b, c, d);
    if (a != b)
        add(b, a, c, d);
    if (c != d) {
        add(a, b, d, c);
        if (a != b)
            add(b, a, d, c);
    }
}

void exchange_same_pair(int n, const double* p, double* k, double w, int a,
                        int b) {
    k[a + a * n] += w * p[b + b * n];
    if (a != b) {
        k[b + b * n] += w * p[a + a * n];
        k[a + b * n] += w * p[b + a * n];
    }
}

} // namespace
} // namespace ATV::cpu::scf::internal::exact_jk_detail
