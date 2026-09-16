/// @file regressions/longrange/fmm_point_charges/reg_fmm_point_charges.cpp
/// @brief END-TO-END REGRESSION: open-boundary FMM vs direct Coulomb sum.
///
/// See REFERENCE.md for Greengard–Rokhlin / White–Head-Gordon DOIs, the
/// documented order p★, and the error-bound formula.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Exits non-zero on failure. Budget ≤ 60 s.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

#include "atomiverse/core/precision.h"
#include "atomiverse/core/vec3.h"
#include "atomiverse/cpu/longrange/fmm_tree.h"

using namespace ATV;
using ATV::longrange::FmmTree;

namespace {

AccReal direct_energy(const std::vector<Vec3>& pos,
                      const std::vector<AccReal>& q) {
    AccReal e = 0.0;
    const std::size_t N = pos.size();
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = i + 1; j < N; ++j) {
            const AccReal r = (pos[i] - pos[j]).norm();
            if (r > 0.0)
                e += q[i] * q[j] / r;
        }
    return e;
}

bool check_order_table(const char* label, const std::vector<Vec3>& pos,
                       const std::vector<AccReal>& q, int p_star, int leaf) {
    const AccReal e_ref = direct_energy(pos, q);
    std::printf("\n=== %s (N=%zu, E_direct=%.12e) ===\n", label, pos.size(),
                e_ref);
    std::printf("%4s  %14s  %12s  %10s\n", "p", "E_fmm", "rel_err", "ratio");

    std::vector<AccReal> errs;
    bool ok = true;
    AccReal prev_err = -1.0;
    for (int p = 2; p <= p_star; p += 2) {
        FmmTree tree(p, leaf);
        const AccReal e = tree.energy(pos, q);
        const AccReal rel =
            std::abs(e - e_ref) / std::max(std::abs(e_ref), 1e-30);
        AccReal ratio = 0.0;
        if (prev_err > 0.0)
            ratio = rel / prev_err;
        std::printf("%4d  %14.6e  %12.3e  %10.3f\n", p, e, rel, ratio);
        errs.push_back(rel);

        // Geometric envelope: every +2 in p, error must drop (while above
        // roundoff). Multipole truncations can oscillate order-to-order.
        if (prev_err > 1e-12 && rel > 1e-14) {
            if (!(ratio < 1.0)) {
                std::printf("FAIL: %s p=%d envelope ratio %.3f not < 1\n",
                            label, p, ratio);
                ok = false;
            }
        }
        if (p == p_star && !(rel <= 1e-10)) {
            std::printf("FAIL: %s p★=%d rel_err=%.3e > 1e-10\n", label, p, rel);
            ok = false;
        }
        prev_err = rel;
    }
    return ok;
}

} // namespace

int main() {
    // REFERENCE.md: p★ = 16 with MAC c≥2 reaches ≤1e-10 on these sets.
    const int p_star = 16;
    const int leaf = 8;
    bool ok = true;

    // ── Two well-separated random clusters (forces M2L; shows p-decay) ─
    // Cluster A near origin, cluster B near (20,0,0): MAC c≫2 so far field
    // is pure multipole, geometric error decay is visible.
    {
        const int n_per = 32;
        std::mt19937 rng(20260720);
        std::uniform_real_distribution<double> U(-0.5, 0.5);
        std::vector<Vec3> pos;
        std::vector<AccReal> q;
        pos.reserve(static_cast<std::size_t>(2 * n_per));
        q.reserve(static_cast<std::size_t>(2 * n_per));
        for (int i = 0; i < n_per; ++i) {
            pos.push_back(Vec3{U(rng), U(rng), U(rng)});
            q.push_back(U(rng));
        }
        for (int i = 0; i < n_per; ++i) {
            pos.push_back(Vec3{20.0 + U(rng), U(rng), U(rng)});
            q.push_back(U(rng));
        }
        if (!check_order_table("random_clusters", pos, q, p_star, leaf))
            ok = false;
    }

    // ── Two well-separated lattice blocks, alternating ±1 ───────────────
    {
        const int n_side = 3;
        const double spacing = 1.0;
        const double shift = 20.0;
        std::vector<Vec3> pos;
        std::vector<AccReal> q;
        for (int block = 0; block < 2; ++block) {
            const double x0 = block * shift;
            for (int ix = 0; ix < n_side; ++ix)
                for (int iy = 0; iy < n_side; ++iy)
                    for (int iz = 0; iz < n_side; ++iz) {
                        pos.push_back(Vec3{x0 + ix * spacing, iy * spacing,
                                           iz * spacing});
                        q.push_back(((ix + iy + iz + block) & 1) ? AccReal(-1)
                                                                 : AccReal(1));
                    }
        }
        if (!check_order_table("lattice_clusters", pos, q, p_star, leaf))
            ok = false;
    }

    // ── Scaling smoke: N vs 4N wall-clock ratio < 8 ─────────────────────
    // Larger N + larger leaf pushes work into the O(N) multipole regime and
    // reduces O(L²) leaf-pair overhead relative to particle work.
    {
        const int N = 1024;
        const int p = 4;
        const int leaf_s = 16;
        auto fill = [](int n, unsigned seed) {
            std::mt19937 rng(seed);
            std::uniform_real_distribution<double> U(-1.0, 1.0);
            std::vector<Vec3> pos(static_cast<std::size_t>(n));
            std::vector<AccReal> q(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) {
                pos[static_cast<std::size_t>(i)] = Vec3{U(rng), U(rng), U(rng)};
                q[static_cast<std::size_t>(i)] = U(rng);
            }
            return std::make_pair(std::move(pos), std::move(q));
        };
        auto run_ms = [&](int n, unsigned seed) {
            auto pq = fill(n, seed);
            FmmTree tree(p, leaf_s);
            (void)tree.energy(pq.first, pq.second);
            const auto t0 = std::chrono::steady_clock::now();
            AccReal e = 0.0;
            const int reps = 3;
            for (int r = 0; r < reps; ++r)
                e += tree.energy(pq.first, pq.second);
            const auto t1 = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count()
                / reps;
            (void)e;
            return ms;
        };
        const double t_n = run_ms(N, 1);
        const double t_4n = run_ms(4 * N, 2);
        const double ratio = t_4n / std::max(t_n, 1e-9);
        std::printf("\n=== scaling smoke N=%d vs 4N (p=%d, leaf=%d) ===\n", N,
                    p, leaf_s);
        std::printf("t(N)=%.3f ms  t(4N)=%.3f ms  ratio=%.3f (need < 8)\n", t_n,
                    t_4n, ratio);
        if (!(ratio < 8.0)) {
            std::printf("FAIL: scaling ratio %.3f >= 8 (O(N²)-like)\n", ratio);
            ok = false;
        }
    }

    if (!ok) {
        std::printf("\nREGRESSION FAILED\n");
        return 1;
    }
    std::printf("\nREGRESSION PASSED\n");
    return 0;
}
