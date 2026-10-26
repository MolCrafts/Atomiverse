#include "../../../examples/xc-extension/pins.h"
#include "../../../examples/xc-extension/scaled_slater.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"
#include <cmath>
#include <cstdio>
#include <memory>
namespace atv_reg_custom_xc_h2plus {
int run() {
    using namespace ATV;
    using namespace cpu::scf;
    cpu::Frame frame(2);
    const double xyz[2][3] = {{0, 0, 0}, {0, 0, 1}};
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 1;
        frame.span(kAtom_x)[i] = xyz[i][0];
        frame.span(kAtom_y)[i] = xyz[i][1];
        frame.span(kAtom_z)[i] = xyz[i][2];
    }
    frame.box = cpu::Box::cubic(20);
    auto xc = std::make_shared<const xc_example::ScaledSlater>();
    UKS method(BasisSet::STO3G, 1, 0, xc, 1);
    method.grid_level = GridLevel::MatchPySCF;
    method.tol = 1e-12;
    method.max_iter = 150;
    // Energy-only geometry changes must not publish DIIS-extrapolated orbital
    // energies as the physical Pulay multipliers on the restored geometry.
    method.run(frame);
    frame.span(kAtom_z)[1] = 1.01;
    method.run(frame);
    frame.span(kAtom_z)[1] = 1.0;
    const auto result = method.forces(frame);
    const double e = result.energy / units::kHartreeInEv;
    int failures = 0;
    if (!method.converged() || !std::isfinite(e)
        || std::abs(e - xc_example::h2plus_energy) > xc_example::energy_tol) {
        std::fprintf(stderr, "FAIL E got=%.14f want=%.14f tol=%.1e Ha\n", e,
                     xc_example::h2plus_energy, xc_example::energy_tol);
        ++failures;
    }
    for (int i = 0; i < 2; ++i)
        for (int d = 0; d < 3; ++d) {
            const double g =
                -result.forces(i, d) / units::kHartreePerBohrInEvPerAngstrom;
            const double ref = xc_example::h2plus_gradient[i][d];
            if (!std::isfinite(g)
                || std::abs(g - ref) > xc_example::gradient_tol) {
                std::fprintf(stderr,
                             "FAIL gradient(%d,%d) got=%.14f want=%.14f "
                             "tol=%.1e Ha/Bohr\n",
                             i, d, g, ref, xc_example::gradient_tol);
                ++failures;
            }
        }
    std::printf(
        "%s custom_xc_h2plus energy %.14f Ha, all force components checked\n",
        failures ? "FAIL" : "PASS", e);
    return failures ? 1 : 0;
}
} // namespace atv_reg_custom_xc_h2plus
ATV_REGRESSION("scf/uks", "custom-xc-h2plus", atv_reg_custom_xc_h2plus::run);
