/// @file reg_component-plugin-01-probe.cpp
/// @brief Public-API gate: energy/force probes on kinds + HF + FixLangevin.
///
/// Hard-coded bool literals (oracle: harness). No MD/SCF loop.

#include <cstdio>
#include <string>
#include <type_traits>

#include "atomiverse/cpu/component/bias.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/component/gpw.h"
#include "atomiverse/cpu/component/hf.h"
#include "atomiverse/cpu/component/path_coupling.h"
#include "atomiverse/cpu/component/plane_wave.h"
#include "atomiverse/cpu/component/potential.h"
#include "atomiverse/cpu/component/semiempirical.h"

#ifdef ATV_HAS_MD
#include "atomiverse/cpu/dynamics/fix_langevin.h"
#endif

namespace {

template <typename Base>
class ProbeStub : public Base {
  public:
    std::string name() const override { return "probe_stub"; }
    void launch(const ATV::cpu::LaunchContext& ctx) override { (void)ctx; }
};

struct HfTag : ATV::cpu::HF {};

bool check(const char* label, bool energy, bool force, bool want_e,
           bool want_f) {
    std::printf("%s energy=%d force=%d\n", label, energy ? 1 : 0,
                force ? 1 : 0);
    if (energy != want_e || force != want_f) {
        std::printf("FAIL: %s want energy=%d force=%d\n", label, want_e ? 1 : 0,
                    want_f ? 1 : 0);
        return false;
    }
    return true;
}

} // namespace

int main() {
    static_assert(!std::is_base_of_v<ATV::cpu::Component, ATV::cpu::HF>,
                  "HF is a tag, not a Component");

    bool ok = true;
    {
        ProbeStub<ATV::cpu::Component> c;
        ok &= check("component", c.provides_energy(), c.provides_force(), false,
                    false);
    }
    {
        ProbeStub<ATV::cpu::Potential> c;
        ok &= check("potential", c.provides_energy(), c.provides_force(), true,
                    true);
    }
    {
        ProbeStub<ATV::cpu::Bias> c;
        ok &=
            check("bias", c.provides_energy(), c.provides_force(), true, true);
    }
    {
        ProbeStub<ATV::cpu::PathCoupling> c;
        ok &= check("path_coupling", c.provides_energy(), c.provides_force(),
                    true, true);
    }
    {
        ProbeStub<ATV::cpu::Semiempirical> c;
        ok &= check("semiempirical", c.provides_energy(), c.provides_force(),
                    true, true);
    }
    {
        ProbeStub<ATV::cpu::PlaneWave> c;
        ok &= check("plane_wave", c.provides_energy(), c.provides_force(), true,
                    true);
    }
    {
        ProbeStub<ATV::cpu::GPW> c;
        ok &= check("gpw", c.provides_energy(), c.provides_force(), true, true);
    }
    {
        HfTag hf;
        ok &=
            check("hf", hf.provides_energy(), hf.provides_force(), true, true);
    }
#ifdef ATV_HAS_MD
    {
        ATV::cpu::md::FixLangevin langevin{300.0, 300.0, 100.0, 1.0, 1};
        ok &= check("langevin", langevin.provides_energy(),
                    langevin.provides_force(), false, true);
    }
#endif
    return ok ? 0 : 1;
}
