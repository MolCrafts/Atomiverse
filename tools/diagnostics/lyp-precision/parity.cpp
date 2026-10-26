#include "atomiverse/cpu/scf/xc_internal.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <xc.h>
int main() {
    int major, minor, micro;
    xc_version(&major, &minor, &micro);
    std::printf("libxc %d.%d.%d, actual xc_gga_exc_vxc XC_GGA_C_LYP id131\n",
                major, minor, micro);
    const double points[][2] = {
        {.5, .5}, {.8, .2}, {1, .05}, {1e-6, 1e-8}, {100, 50}};
    const double sigmas[] = {0, .01, .1, 1};
    const char* names[] = {"energy density", "vrho a",    "vrho b",
                           "vsigma aa",      "vsigma ab", "vsigma bb"};
    xc_func_type func;
    if (xc_func_init(&func, 131, XC_POLARIZED))
        return 2;
    int fail = 0;
    double max_rel = 0;
    for (const auto& rho : points)
        for (double s : sigmas) {
            double native[6], oracle[6], eps;
            double sigma[3] = {s, s, s};
            ATV::cpu::scf::lyp_correlation(rho[0], rho[1], s, s, s, native[0],
                                           native[1], native[2], native[3],
                                           native[4], native[5]);
            xc_gga_exc_vxc(&func, 1, rho, sigma, &eps, oracle + 1, oracle + 3);
            oracle[0] = eps * (rho[0] + rho[1]);
            if (rho[0] == 1e-6)
                std::printf("raw oracle rho=(%.17g,%.17g) sigma=%.17g f=%.17g "
                            "vrho=(%.17g,%.17g) vsigma=(%.17g,%.17g,%.17g)\n",
                            rho[0], rho[1], s, oracle[0], oracle[1], oracle[2],
                            oracle[3], oracle[4], oracle[5]);
            for (int c = 0; c < 6; ++c) {
                double diff = std::abs(native[c] - oracle[c]),
                       scale =
                           std::max(std::abs(native[c]), std::abs(oracle[c]));
                max_rel = std::max(max_rel, diff / std::max(scale, 1.));
                if (!std::isfinite(native[c])
                    || diff > std::max(1e-12, 1e-12 * scale)) {
                    std::printf(
                        "FAIL rho=(%.3g,%.3g) sigma=%.3g %s native=%.17g "
                        "oracle=%.17g diff=%.17g bound=%.17g\n",
                        rho[0], rho[1], s, names[c], native[c], oracle[c], diff,
                        std::max(1e-12, 1e-12 * scale));
                    ++fail;
                }
            }
        }
    xc_func_end(&func);
    std::printf("20 point/sigma pairs,120 channels, failures=%d "
                "max_scaled_error=%.17g\n",
                fail, max_rel);
    return fail ? 1 : 0;
}
