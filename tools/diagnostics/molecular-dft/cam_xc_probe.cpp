// Scalar diagnostic only: molecular qualification uses the public Python driver.
#include "atomiverse/cpu/xc/xc_functional.h"
#include "atomiverse/cpu/xc/xc_internal.h"
#include <array>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc != 6)
        return 2;
    using namespace ATV::cpu::xc;
    std::array<double, 5> x;
    for (int i = 0; i < 5; ++i)
        x[i] = std::stod(argv[i + 1]);
    const auto value =
        evaluate_xc(XCType::CAMB3LYP, x[0], x[1], x[2], x[3], x[4]);
    std::printf(
        "{\"output\":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g],\"short_range\":[",
        value.energy_density, value.vrho_alpha, value.vrho_beta,
        value.vsigma_aa, value.vsigma_ab, value.vsigma_bb);
    for (int spin = 0; spin < 2; ++spin) {
        double energy, vrho, vsigma;
        b88_short_range_exchange_spin(x[spin], x[spin == 0 ? 2 : 4], 0.33,
                                      energy, vrho, vsigma);
        std::printf("%s[%.17g,%.17g,%.17g]", spin ? "," : "", energy, vrho,
                    vsigma);
    }
    double f, va, vb, saa, sab, sbb;
    lyp_correlation(x[0], x[1], x[2], x[3], x[4], f, va, vb, saa, sab, sbb);
    std::printf("],\"lyp\":[%.17g,%.17g,%.17g,%.17g,%.17g,%.17g]", f, va, vb,
                saa, sab, sbb);
    double ea, eb, va0, vb0, sa0, sb0;
    b88_exchange_spin(x[0], x[2], ea, va, saa);
    b88_exchange_spin(x[1], x[4], eb, vb, sbb);
    std::printf(",\"b88\":[%.17g,%.17g,%.17g,%.17g,0,%.17g]", ea + eb, va, vb,
                saa, sbb);
    b88_exchange_spin(x[0], 0., ea, va0, sa0);
    b88_exchange_spin(x[1], 0., eb, vb0, sb0);
    const auto lda = evaluate_xc(XCType::LSDA, x[0], x[1]);
    std::printf(",\"vwn\":[%.17g,%.17g,%.17g,0,0,0]}\n",
                lda.energy_density - ea - eb, lda.vrho_alpha - va0,
                lda.vrho_beta - vb0);
}
