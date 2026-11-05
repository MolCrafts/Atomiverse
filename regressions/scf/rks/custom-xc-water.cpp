#include "../../../examples/extensions/xc/pins.h"
#include "../../../examples/extensions/xc/scaled_slater.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "harness/registry.h"
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
namespace atv_reg_custom_xc_water {
// Failure injection is test instrumentation, not provider-owned solver state.
class FailsOnce final : public ATV::electronic::XcFunctional {
  public:
    ATV::electronic::XcIngredients ingredients() const noexcept override {
        return value_.ingredients();
    }
    bool supports(ATV::electronic::XcSpin spin,
                  ATV::electronic::XcOrder order) const noexcept override {
        return value_.supports(spin, order);
    }

  protected:
    void evaluate_batch(const ATV::electronic::XcBatch& in,
                        ATV::electronic::XcOrder order,
                        ATV::electronic::XcOutput& out) const override {
        if (fail_) {
            fail_ = false;
            throw std::runtime_error("injected XC failure");
        }
        value_.evaluate(in, order, out);
    }

  private:
    xc_example::ScaledSlater value_;
    mutable bool fail_ = true;
};

int run() {
    using namespace ATV;
    using namespace cpu::scf;
    cpu::Frame frame(3);
    const double xyz[3][3] = {
        {0, 0, 0.1173}, {0, 0.7572, -0.4692}, {0, -0.7572, -0.4692}};
    for (int i = 0; i < 3; ++i) {
        frame.span(kAtom_type)[i] = i == 0 ? 8 : 1;
        frame.span(kAtom_x)[i] = xyz[i][0];
        frame.span(kAtom_y)[i] = xyz[i][1];
        frame.span(kAtom_z)[i] = xyz[i][2];
    }
    frame.box = cpu::Box::cubic(20);
    auto xc = std::make_shared<const FailsOnce>();
    RKS method(BasisSet::STO3G, xc);
    method.grid_level = GridLevel::MatchPySCF;
    method.tol = 1e-12;
    method.max_iter = 150;
    frame.meta.get(kEnergy) = 123;
    frame.span(kAtom_fx)[0] = 17;
    bool rejected = false;
    try {
        method.run(frame);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("injected XC failure")
                   != std::string::npos;
    }
    if (!rejected || frame.meta.get(kEnergy) != 123
        || frame.span(kAtom_fx)[0] != 17) {
        std::fprintf(
            stderr,
            "FAIL: XC failure changed caller outputs or was swallowed\n");
        return 1;
    }
    // Retry the same method/provider and pin the complete successful result.
    const auto result = method.forces(frame);
    const double e = result.energy / units::kHartreeInEv;
    int failures = 0;
    if (!method.converged() || !std::isfinite(e)
        || std::abs(e - xc_example::water_energy) > xc_example::energy_tol) {
        std::fprintf(stderr, "FAIL E got=%.14f want=%.14f tol=%.1e Ha\n", e,
                     xc_example::water_energy, xc_example::energy_tol);
        ++failures;
    }
    for (int i = 0; i < 3; ++i)
        for (int d = 0; d < 3; ++d) {
            const double g =
                -result.forces(i, d) / units::kHartreePerBohrInEvPerAngstrom;
            const double ref = xc_example::water_gradient[i][d];
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
        "%s custom_xc_water energy %.14f Ha, all force components checked\n",
        failures ? "FAIL" : "PASS", e);
    return failures ? 1 : 0;
}
} // namespace atv_reg_custom_xc_water
ATV_REGRESSION("scf/rks", "custom-xc-water", atv_reg_custom_xc_water::run);
