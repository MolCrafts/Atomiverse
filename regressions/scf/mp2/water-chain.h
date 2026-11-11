#pragma once

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/mp2.h"

#include <cmath>
#include <cstdio>

namespace atv_mp2_water_chain {

inline ATV::cpu::Frame geometry() {
    using namespace ATV;
    cpu::Frame frame(24);
    constexpr double y[3] = {0, 0.7572, -0.7572};
    constexpr double z[3] = {0.1173, -0.4692, -0.4692};
    for (int w = 0; w < 8; ++w)
        for (int a = 0; a < 3; ++a) {
            const int i = 3 * w + a;
            frame.span(kAtom_atomic_number)[i] = a == 0 ? 8 : 1;
            frame.span(kAtom_x)[i] = 4.0 * w;
            frame.span(kAtom_y)[i] = y[a];
            frame.span(kAtom_z)[i] = z[a];
        }
    frame.box = cpu::Box::cubic(52.0);
    return frame;
}

inline int check(const ATV::cpu::scf::MP2Result& result,
                 const ATV::cpu::Frame& frame) {
    // Offline PySCF 2.11.0, Cartesian STO-3G, RHF conv_tol=1e-12,
    // Hcore guess, 8 water molecules spaced by 4 Angstrom (REFERENCE.md).
    const double got[] = {result.e_tot - result.e_corr, result.e_corr,
                          result.e_corr_os, result.e_corr_ss, result.e_tot};
    constexpr double want[] = {-599.6949297548834, -0.2851551624659191,
                               -0.2688466882176113, -0.016308474248307774,
                               -599.9800849173494};
    constexpr const char* names[] = {"HF", "correlation", "OS", "SS", "total"};
    bool ok = !result.ri && result.frozen == 0;
    for (int i = 0; i < 5; ++i) {
        std::printf("water8 %s got=%.14f want=%.14f tol=1e-6 Ha\n", names[i],
                    got[i], want[i]);
        ok &= std::isfinite(got[i]) && std::abs(got[i] - want[i]) < 1e-6;
    }
    ok &= std::abs(result.e_corr - result.e_corr_os - result.e_corr_ss) < 1e-12;
    ok &= std::abs(frame.meta.get(ATV::kEnergy)
                   - result.e_tot * ATV::units::kHartreeInEv)
          < 1e-8;
    return ok ? 0 : 1;
}

} // namespace atv_mp2_water_chain
