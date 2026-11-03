#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

TEST(PwEnergy, SpinDoubleCountingUsesBothChannelsAtTheInputDensity) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    using Complex = std::complex<AccReal>;
    PWWorkspace ws;
    ws.n_g = 1;
    ws.nspin = 2;
    ws.rho_G = xt::xarray<Complex>{2.};
    ws.V_H_G = xt::xarray<Complex>{3.};
    ws.V_xc_G =
        xt::xarray<Complex>{17.}; // Last spin's scratch, not the spin sum.
    ws.V_loc_G = xt::xarray<Complex>{0.};
    ws.E_xc = 4.;
    ws.e_xc_int_spin = 5.;
    const auto spin = iteration_energy(ws, 100., 5., 2.);
    EXPECT_DOUBLE_EQ(spin.e_h, 6.);
    EXPECT_DOUBLE_EQ(spin.e_xc_int, 5.);
    EXPECT_DOUBLE_EQ(spin.e_one, 83.);
    EXPECT_DOUBLE_EQ(spin.total, 98.);
    ws.nspin = 1;
    const auto restricted = iteration_energy(ws, 100., 5., 2.);
    EXPECT_DOUBLE_EQ(restricted.e_xc_int, 68.);
    EXPECT_DOUBLE_EQ(restricted.total, 35.);
}

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/electronic/norm_conserving_potential.h"
#include <array>
#include <limits>

namespace {
class SampledNc final : public ATV::electronic::NormConservingPotential {
  public:
    int atomic_number() const noexcept override { return z; }
    int valence() const noexcept override { return val; }
    ATV::electronic::NcProjectorSetView projectors() const override {
        ++projector_calls;
        return {radius, jacobian, projector, coupling, {}, {}};
    }
    int z = 2, val = 2;
    std::vector<ATV::AccReal> radius{0, .5, 1, 1.5, 2};
    std::vector<ATV::AccReal> jacobian{.5, .5, .5, .5, .5};
    std::vector<ATV::AccReal> beta{0, .5, 1};
    std::array<ATV::electronic::NcProjectorView, 1> projector{{{0, beta}}};
    std::vector<ATV::AccReal> coupling{.4};
    mutable int local_calls = 0;
    mutable int projector_calls = 0;
    bool incomplete = false;

  protected:
    void evaluate_local(std::span<const ATV::AccReal> r,
                        std::span<ATV::AccReal> v,
                        std::span<ATV::AccReal> dv) const override {
        ++local_calls;
        if (incomplete)
            return;
        for (std::size_t i = 0; i < r.size(); ++i) {
            const auto q = 1 + r[i] * r[i];
            v[i] = -val / std::sqrt(q);
            dv[i] = val * r[i] / std::pow(q, 1.5);
        }
    }
};
} // namespace

TEST(NcAdmission, SnapshotOwnsRadialDataAndSamplesOneLocalBatch) {
    using namespace ATV::cpu::pw::internal;
    SampledNc provider;
    const NcPotentialData snapshot(provider);
    EXPECT_EQ(provider.local_calls, 1);
    EXPECT_EQ(snapshot.atomic_number, 2);
    EXPECT_EQ(snapshot.z_val, 2);
    EXPECT_EQ(snapshot.kb[0].radial_points, 3u);
    EXPECT_DOUBLE_EQ(snapshot.v_loc[0], -2);
    provider.beta[1] = 200;
    provider.radius[1] = .25;
    provider.coupling[0] = 40;
    EXPECT_DOUBLE_EQ(snapshot.kb[0].beta[1], .5);
    EXPECT_DOUBLE_EQ(snapshot.r[1], .5);
    EXPECT_DOUBLE_EQ(snapshot.d_ij[0], .4);
}

TEST(NcAdmission, InvalidPhysicalViewsRejectBeforeLocalCallback) {
    using namespace ATV::cpu::pw::internal;
    SampledNc provider;
    provider.val = 3;
    EXPECT_THROW(NcPotentialData{provider}, std::invalid_argument);
    EXPECT_EQ(provider.local_calls, 0);
    provider.val = 2;
    provider.projector[0].l = 3;
    EXPECT_THROW(NcPotentialData{provider}, std::invalid_argument);
    provider.projector[0].l = 0;
    provider.coupling[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(NcPotentialData{provider}, std::invalid_argument);
    EXPECT_EQ(provider.local_calls, 0);
    provider.coupling[0] = .4;
    provider.incomplete = true;
    EXPECT_THROW(NcPotentialData{provider}, std::runtime_error);
    EXPECT_EQ(provider.local_calls, 1);
}

TEST(NcAdmission, EveryFrameSpeciesMustMatchTheImmutableProvider) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    SampledNc provider;
    cpu::Frame frame(2);
    frame.span(kAtom_atomic_number)[0] = frame.span(kAtom_atomic_number)[1] = 2;
    EXPECT_NO_THROW(validate_potential_frame(frame, provider));
    const NcPotentialData snapshot(provider);
    EXPECT_NO_THROW(validate_potential_frame(frame, snapshot));
    frame.span(kAtom_atomic_number)[1] = 1;
    EXPECT_THROW(validate_potential_frame(frame, provider),
                 std::invalid_argument);
    EXPECT_THROW(validate_potential_frame(frame, snapshot),
                 std::invalid_argument);
    cpu::Frame empty;
    EXPECT_THROW(validate_potential_frame(empty, provider),
                 std::invalid_argument);
}

TEST(NcAdmission,
     StrainedGeometryAssemblyReusesAdmittedPhysicsWithoutCallbacks) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    SampledNc provider;
    cpu::Frame frame(1);
    frame.span(kAtom_atomic_number)[0] = 2;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0;
    frame.box = cpu::Box::cubic(4);
    PWProblem reference;
    setup_pw_problem(frame, provider, 5, 1, reference);
    ASSERT_EQ(provider.local_calls, 1);
    ASSERT_EQ(provider.projector_calls, 1);

    // Any accidental resampling would now throw. Geometry-only setup must
    // share the already admitted radial data and rebuild the changed volume.
    provider.incomplete = true;
    cpu::Frame strained = frame;
    strained.box = cpu::Box::cubic(4.1);
    PWProblem sample;
    setup_pw_problem(strained, reference.potential, 5, 1, sample);
    EXPECT_EQ(provider.local_calls, 1);
    EXPECT_EQ(provider.projector_calls, 1);
    EXPECT_EQ(sample.potential, reference.potential);
    EXPECT_EQ(sample.n_elec, reference.n_elec);
    EXPECT_NEAR(sample.ws.omega / reference.ws.omega, std::pow(4.1 / 4, 3),
                1e-13);

    strained.span(kAtom_atomic_number)[0] = 1;
    PWProblem rejected;
    rejected.n_elec = 77;
    EXPECT_THROW(
        setup_pw_problem(strained, reference.potential, 5, 1, rejected),
        std::invalid_argument);
    EXPECT_EQ(rejected.n_elec, 77);
    EXPECT_FALSE(rejected.potential);
}
