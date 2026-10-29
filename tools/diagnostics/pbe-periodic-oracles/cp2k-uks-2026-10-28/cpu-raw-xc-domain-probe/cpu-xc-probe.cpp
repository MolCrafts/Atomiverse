#include "atomiverse/cpu/xc/xc_functional.h"
#include <array>
#include <cmath>
#include <cstdio>
using namespace ATV::cpu::xc;
int main() {
    for (auto type :
         {XCType::PBE, XCType::PBEsol, XCType::RevPBE, XCType::RPBE}) {
        for (double rb : {-.01, 0., .1}) {
            const double sab = rb == 0 ? 0 : -.01, sbb = rb == 0 ? 0 : .015;
            auto get = [&](double b) {
                return evaluate_xc(type, .2, b, .02, sab, sbb);
            };
            auto o = get(rb);
            std::printf(
                "POINT xc=%d ra=.2 rb=%.17g saa=.02 sab=%.17g sbb=%.17g "
                "E=%.17g va=%.17g vb=%.17g vsaa=%.17g vsab=%.17g vsbb=%.17g\n",
                int(type), rb, sab, sbb, o.energy_density, o.vrho_alpha,
                o.vrho_beta, o.vsigma_aa, o.vsigma_ab, o.vsigma_bb);
            for (double h :
                 {1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12, 1e-14, 1e-16}) {
                double plus =
                    (get(rb + h).energy_density - o.energy_density) / h;
                double minus =
                    (o.energy_density - get(rb - h).energy_density) / h;
                std::printf("FD xc=%d rb=%.17g h=%.17g central=%.17g "
                            "positive=%.17g negative=%.17g vb=%.17g\n",
                            int(type), rb, h, .5 * (plus + minus), plus, minus,
                            o.vrho_beta);
            }
        }
    }
}
