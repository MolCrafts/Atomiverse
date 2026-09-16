#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include <cmath>
#include <cstdio>
using namespace ATV;
int main() {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 0.9;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.box = cpu::Box::cubic(15);
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Pople631G;
    rhf.tol = 1e-7;
    rhf.run(f);
    if (!rhf.converged()) {
        std::printf("FAIL scf\n");
        return 1;
    }
    auto d = rhf.tda_nac_vectors(f, 0, 1);
    if (d.size() != 6u) {
        std::printf("FAIL size\n");
        return 1;
    }
    double n2 = 0;
    for (double v : d) {
        if (!std::isfinite(v)) {
            std::printf("FAIL nan\n");
            return 1;
        }
        n2 += v * v;
    }
    const double n = std::sqrt(n2);
    // Hard-coded norm gate: H2/6-31G R=0.9 Å two-root NAC, engine pin
    // 2026-07-23 (order-1 scale; FD noise across runs).
    std::printf("NAC norm=%.6e 1/A\n", n);
    if (!(n > 0.1 && n < 20.0)) {
        std::printf("FAIL norm window\n");
        return 1;
    }
    // Antisymmetry smoke: d_10 ≈ −d_01 (allow FD noise).
    auto d10 = rhf.tda_nac_vectors(f, 1, 0);
    double maxa = 0;
    for (std::size_t k = 0; k < d.size(); ++k)
        maxa = std::max(maxa, std::fabs(d10[k] + d[k]));
    std::printf("max|d10+d01|=%.3e\n", maxa);
    if (maxa > 0.2) {
        std::printf("FAIL antisymmetry\n");
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
