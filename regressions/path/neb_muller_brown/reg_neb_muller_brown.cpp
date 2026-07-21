/// @file regressions/path/neb_muller_brown/reg_neb_muller_brown.cpp
/// @brief CI-NEB on the Müller-Brown surface: does the climbing image land on the saddle?
///
/// Out-of-library: public headers + link atomiverse, nothing else. The potential itself
/// is defined HERE, as a user-written component — which is also the point. If a two-line
/// analytic surface cannot be plugged into the Path driver from outside the library, then
/// the component API is not an API.
///
/// See REFERENCE.md for the literature values this checks against.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/driver/path_driver.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/path/neb.h"
#include "atomiverse/cpu/replica/replica_expansion.h"
#include "atomiverse/facade/stage.h"

using namespace ATV;

namespace {

// ── The Müller-Brown surface (Müller & Brown 1979, Theor. Chim. Acta 53, 75) ──────
//
//   V(x,y) = Σ_k A_k exp[ a_k (x-x0_k)² + b_k (x-x0_k)(y-y0_k) + c_k (y-y0_k)² ]
//
// Dimensionless — this is a model surface, not a molecule, so the engine's eV/Å carry
// no meaning here and the numbers are used as published.
constexpr double kA[4] = {-200.0, -100.0, -170.0, 15.0};
constexpr double kAa[4] = {-1.0, -1.0, -6.5, 0.7};
constexpr double kBb[4] = {0.0, 0.0, 11.0, 0.6};
constexpr double kCc[4] = {-10.0, -10.0, -6.5, 0.7};
constexpr double kX0[4] = {1.0, 0.0, -0.5, -1.0};
constexpr double kY0[4] = {0.0, 0.5, 1.5, 1.0};

/// A single particle on the surface. One atom; z is unused and its force is left at zero,
/// so the particle stays in the plane it started in.
class MullerBrown : public cpu::Component {
  public:
    std::string name() const override { return "muller_brown"; }
    /* role via Potential/kind base */
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::path::Phase::force, 0)};
    }

    void build(const cpu::BuildContext&) override {}

    void launch(const cpu::LaunchContext& ctx) override {
        auto& f = ctx.frame;
        const double x = f.span(ATV::kAtom_x)[0];
        const double y = f.span(ATV::kAtom_y)[0];

        double v = 0.0, dvdx = 0.0, dvdy = 0.0;
        for (int k = 0; k < 4; ++k) {
            const double dx = x - kX0[k];
            const double dy = y - kY0[k];
            const double e = kA[k]
                             * std::exp(kAa[k] * dx * dx + kBb[k] * dx * dy
                                        + kCc[k] * dy * dy);
            v += e;
            dvdx += e * (2.0 * kAa[k] * dx + kBb[k] * dy);
            dvdy += e * (kBb[k] * dx + 2.0 * kCc[k] * dy);
        }

        f.meta.get(ATV::kEnergy) = v;
        f.span(ATV::kAtom_fx)[0] = -dvdx; // force = -grad V
        f.span(ATV::kAtom_fy)[0] = -dvdy;
        f.span(ATV::kAtom_fz)[0] = 0.0;
    }
};

cpu::Frame particle_at(double x, double y) {
    cpu::Frame f(1);
    f.span(ATV::kAtom_x)[0] = x;
    f.span(ATV::kAtom_y)[0] = y;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_mass)[0] = 1.0;
    f.span(ATV::kAtom_type)[0] = 1;
    f.simbox = ATV::cpu::Box::cubic(
        100.0); // the surface is unbounded; the box never binds
    return f;
}

/// Published stationary points of the surface (see REFERENCE.md).
constexpr double kSaddleX = -0.822;
constexpr double kSaddleY = 0.624;
constexpr double kSaddleV = -40.665;
constexpr double kMinAV = -146.700;

int check(const char* what, double got, double want, double tol) {
    const double err = std::abs(got - want);
    const bool ok = err <= tol;
    std::printf(
        "  %-28s got %10.4f   want %10.4f   |err| %8.4f   tol %6.3f   %s\n",
        what, got, want, err, tol, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}

} // namespace

int main() {
    std::printf("regression [path/neb_muller_brown] — CI-NEB saddle point\n");

    // The two minima the path connects: the global minimum A and the minimum B.
    cpu::Frame initial = particle_at(-0.558, 1.442);    // minimum A
    cpu::Frame final_state = particle_at(0.623, 0.028); // minimum B

    // The band: 13 images, generated from ONE frame by a strategy. The product state is
    // the STRATEGY's configuration, not a second argument to run() — which is what keeps
    // the Path category from carrying an NEB-shaped signature.
    cpu::LinearInterpolation expand{final_state, 13};

    cpu::PathDriver neb;
    neb.on(ATV::path::Phase::force).add(MullerBrown{});
    neb.add(cpu::path::NebCoupling{/*spring_k=*/10.0, /*climbing=*/true,
                                   /*climb_after=*/20});
    neb.add(
        cpu::path::BandLBFGS{/*memory=*/10, /*max_step=*/0.01, /*ftol=*/0.5});
    neb.set_threads(
        1); // one atom per image — threads would cost more than the physics
    neb.enable_timing(false);
    neb.build();

    const int iters = neb.run(initial, expand, 1500);

    // Frame in, Frame out: `initial` now holds the collapsed result — for CI-NEB, the
    // climbing image, i.e. the saddle-point configuration.
    const double sx = initial.span(ATV::kAtom_x)[0];
    const double sy = initial.span(ATV::kAtom_y)[0];
    const double sv = initial.meta.get(ATV::kEnergy);
    const double barrier = sv - kMinAV;

    std::printf("  converged in %d iterations\n", iters);
    int fails = 0;
    fails += check("saddle x", sx, kSaddleX, 0.02);
    fails += check("saddle y", sy, kSaddleY, 0.02);
    fails += check("saddle energy", sv, kSaddleV, 0.5);
    fails += check("barrier from minimum A", barrier, kSaddleV - kMinAV, 0.5);

    if (fails) {
        std::printf("FAILED (%d checks)\n", fails);
        return 1;
    }
    std::printf("PASSED\n");
    return 0;
}
