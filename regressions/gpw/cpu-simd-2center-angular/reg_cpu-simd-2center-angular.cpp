/// @file regressions/gpw/cpu-simd-2center-angular/reg_cpu-simd-2center-angular.cpp
/// @brief END-TO-END fragment: portable xsimd mixed-angular 2-center S/T
///        Hermite on a slab multi-image lattice vs independent prim image sum.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "atomiverse/cpu/gpw/internal/collocate.h"
#include "atomiverse/cpu/gpw/internal/one_electron.h"

using ATV::cpu::gpw::contracted_ST_periodic;
using ATV::cpu::gpw::GridBasisFunction;
using ATV::cpu::gpw::GridPrimitive;
using ATV::cpu::gpw::kinetic;
using ATV::cpu::gpw::overlap;
using ATV::cpu::gpw::prim_kinetic;
using ATV::cpu::gpw::prim_overlap;

namespace {

GridPrimitive make_prim(double alpha, double cx, double cy, double cz, int lx,
                        int ly, int lz, double coeff) {
    GridPrimitive p;
    p.alpha = alpha;
    p.center[0] = cx;
    p.center[1] = cy;
    p.center[2] = cz;
    p.l[0] = lx;
    p.l[1] = ly;
    p.l[2] = lz;
    p.coeff = coeff;
    return p;
}

void prim_image_sum_ST(const GridBasisFunction& a, const GridBasisFunction& b,
                       const double L[3], const int m[3], double& s_out,
                       double& t_out) {
    double s = 0.0, t = 0.0;
    for (int tx = -m[0]; tx <= m[0]; ++tx)
        for (int ty = -m[1]; ty <= m[1]; ++ty)
            for (int tz = -m[2]; tz <= m[2]; ++tz) {
                for (const auto& pa : a) {
                    for (const auto& pb : b) {
                        GridPrimitive bT = pb;
                        bT.center[0] = pb.center[0] + tx * L[0];
                        bT.center[1] = pb.center[1] + ty * L[1];
                        bT.center[2] = pb.center[2] + tz * L[2];
                        s += prim_overlap(pa, bT);
                        t += prim_kinetic(pa, bT);
                    }
                }
            }
    s_out = s;
    t_out = t;
}

bool near(double got, double ref, double rel) {
    const double tol = rel * std::max(1.0, std::abs(ref));
    return std::abs(got - ref) <= tol;
}

} // namespace

int main() {
    GridBasisFunction a = {
        make_prim(0.55, 0, 0, 0, 1, 0, 0, 0.40),
        make_prim(0.55, 0, 0, 0, 0, 1, 0, 0.35),
        make_prim(0.55, 0, 0, 0, 0, 0, 1, 0.30),
        make_prim(1.40, 0, 0, 0, 1, 0, 0, 0.25),
    };
    GridBasisFunction b = {
        make_prim(0.70, 0.7, -0.15, 0.25, 1, 0, 0, 0.45),
        make_prim(0.70, 0.7, -0.15, 0.25, 0, 1, 0, 0.40),
        make_prim(0.70, 0.7, -0.15, 0.25, 0, 0, 1, 0.35),
        make_prim(1.90, 0.7, -0.15, 0.25, 0, 1, 1, 0.20),
    };
    const double L[3] = {6.0, 6.0, 4.0};
    const int m[3] = {2, 2, 4};
    const double reach = 40.0;

    double s_fused = 0.0, t_fused = 0.0;
    contracted_ST_periodic(a, b, L, m, reach, s_fused, t_fused);

    double s_ref = 0.0, t_ref = 0.0;
    prim_image_sum_ST(a, b, L, m, s_ref, t_ref);

    const double s_pub = overlap(a, b, L, m, reach);
    const double t_pub = kinetic(a, b, L, m, reach);

    constexpr double kTol = 1.0e-11;
    int fail = 0;
    if (!near(s_fused, s_ref, kTol)) {
        std::printf("FAIL: S fused=% .16e ref=% .16e |d|=%.3e\n", s_fused,
                    s_ref, std::abs(s_fused - s_ref));
        ++fail;
    }
    if (!near(t_fused, t_ref, kTol)) {
        std::printf("FAIL: T fused=% .16e ref=% .16e |d|=%.3e\n", t_fused,
                    t_ref, std::abs(t_fused - t_ref));
        ++fail;
    }
    if (!near(s_pub, s_fused, 1e-14) || !near(t_pub, t_fused, 1e-14)) {
        std::printf("FAIL: public overlap/kinetic disagree with fused\n");
        ++fail;
    }

    constexpr int kRep = 2000;
    const auto t0 = std::chrono::steady_clock::now();
    double sink_s = 0.0, sink_t = 0.0;
    for (int r = 0; r < kRep; ++r) {
        double ss = 0.0, tt = 0.0;
        contracted_ST_periodic(a, b, L, m, reach, ss, tt);
        sink_s += ss;
        sink_t += tt;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_ms =
        1e-6
        * std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    const auto c0 = std::chrono::steady_clock::now();
    double sink_ref = 0.0;
    for (int r = 0; r < kRep; ++r) {
        double ss = 0.0, tt = 0.0;
        prim_image_sum_ST(a, b, L, m, ss, tt);
        sink_ref += ss + tt;
    }
    const auto c1 = std::chrono::steady_clock::now();
    const double wall_ref_ms =
        1e-6
        * std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count();

    if (!std::isfinite(sink_s + sink_t + sink_ref)) {
        std::printf("FAIL: non-finite microbench sink\n");
        return 1;
    }

    const double speedup =
        wall_ref_ms > 0.0 ? (wall_ref_ms / std::max(wall_ms, 1e-12)) : 0.0;
    std::printf("cpu-simd-2center-angular PASS\n");
    std::printf("  S=%.12e T=%.12e |dS_ref|=%.3e |dT_ref|=%.3e\n", s_fused,
                t_fused, std::abs(s_fused - s_ref), std::abs(t_fused - t_ref));
    std::printf(
        "  microbench rep=%d fused=%.3f ms prim_sum=%.3f ms speedup=%.2f×\n",
        kRep, wall_ms, wall_ref_ms, speedup);
    std::printf(
        "  note: perf_exempt vs fair twins (slab-class multi-image only)\n");
    return fail == 0 ? 0 : 1;
}
