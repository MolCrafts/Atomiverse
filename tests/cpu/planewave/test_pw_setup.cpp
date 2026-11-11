#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include <array>
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
    const std::array<Complex, 1> local{};
    ws.V_loc_G = local;
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

TEST(NcAdmission, AssignsChargesByAtomicNumberInFrameOrder) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    const auto helium = std::make_shared<SampledNc>();
    const auto hydrogen = std::make_shared<SampledNc>();
    hydrogen->z = hydrogen->val = 1;
    const std::vector<
        std::shared_ptr<const electronic::NormConservingPotential>>
        providers{helium, hydrogen};
    cpu::Frame frame(3);
    auto z = frame.span(kAtom_atomic_number);
    z[0] = 1;
    z[1] = 2;
    z[2] = 1;
    const auto assigned = assign_nc_species(frame, providers);
    EXPECT_EQ(assigned.atom_species, (std::vector<std::size_t>{1, 0, 1}));
    EXPECT_EQ(assigned.ionic_charges, (std::vector<AccReal>{1, 2, 1}));
    EXPECT_EQ(assigned.n_elec, 4);
    EXPECT_THROW(assign_nc_species(frame, NcProviders(providers).first(1)),
                 std::invalid_argument);

    const std::vector<
        std::shared_ptr<const electronic::NormConservingPotential>>
        duplicates{helium, hydrogen, helium};
    EXPECT_THROW(assign_nc_species(frame, duplicates), std::invalid_argument);
    const cpu::Frame empty;
    EXPECT_THROW(assign_nc_species(empty, providers), std::invalid_argument);
    EXPECT_EQ(helium->local_calls, 0);
    EXPECT_EQ(hydrogen->local_calls, 0);
}

TEST(NcAdmission,
     StrainedGeometryAssemblyReusesAdmittedPhysicsWithoutCallbacks) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    const auto provider = std::make_shared<SampledNc>();
    const std::shared_ptr<const electronic::NormConservingPotential> nc =
        provider;
    cpu::Frame frame(1);
    frame.span(kAtom_atomic_number)[0] = 2;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0;
    frame.box = cpu::Box::cubic(4);
    PWSetup reference;
    PWWorkspace reference_ws;
    setup_pw_problem(frame, std::span{&nc, 1}, 5, 1, reference, reference_ws);
    ASSERT_EQ(provider->local_calls, 1);
    ASSERT_EQ(provider->projector_calls, 1);

    // Any accidental resampling would now throw. Geometry-only setup must
    // share the already admitted radial data and rebuild the changed volume.
    provider->incomplete = true;
    cpu::Frame strained = frame;
    strained.box = cpu::Box::cubic(4.1);
    PWSetup sample;
    PWWorkspace sample_ws;
    setup_pw_problem_fixed_basis(strained, reference.strain_reference(), 5, 1,
                                 sample, sample_ws);
    EXPECT_EQ(provider->local_calls, 1);
    EXPECT_EQ(provider->projector_calls, 1);
    EXPECT_EQ(sample.ions.species, reference.ions.species);
    EXPECT_EQ(sample.ions.n_elec, reference.ions.n_elec);
    EXPECT_NEAR(sample_ws.omega / reference_ws.omega, std::pow(4.1 / 4, 3),
                1e-13);

    strained.span(kAtom_atomic_number)[0] = 1;
    PWSetup rejected;
    PWWorkspace rejected_ws;
    rejected.ions.n_elec = 77;
    EXPECT_THROW(setup_pw_problem_fixed_basis(strained,
                                              reference.strain_reference(), 5,
                                              1, rejected, rejected_ws),
                 std::invalid_argument);
    EXPECT_EQ(rejected.ions.n_elec, 77);
    EXPECT_TRUE(rejected.ions.species.empty());
}

TEST(PwSetup, KPointSwitchesBorrowProjectorsFromEachMesh) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    const std::shared_ptr<const electronic::NormConservingPotential> nc =
        std::make_shared<SampledNc>();
    cpu::Frame frame(1);
    frame.span(kAtom_atomic_number)[0] = 2;
    frame.span(kAtom_x)[0] = frame.span(kAtom_y)[0] = frame.span(kAtom_z)[0] =
        0;
    frame.box = cpu::Box::cubic(4);

    PWSetup setup;
    PWWorkspace ws;
    setup_pw_problem(frame, std::span{&nc, 1}, 5, 2, setup, ws);
    const auto original = setup.mesh.point(0);
    ASSERT_FALSE(original.projectors[0].empty());
    const auto* beta = original.projectors[0][0].beta_G.data();

    const auto shifted = make_pw_k_mesh(setup.ions, frame.box, 5, {{.1, 0, 0}});
    for (const auto point : {original, setup.mesh.point(1), shifted.point(0)}) {
        ws.bind(point);
        EXPECT_EQ(ws.gvec, &point.modes);
        EXPECT_EQ(ws.kb_proj.data(), point.projectors.data());
        EXPECT_EQ(ws.kb_proj[0][0].beta_G.data(),
                  point.projectors[0][0].beta_G.data());
        EXPECT_EQ(ws.atom_wfc.data(), point.orbitals.data());
    }
    EXPECT_EQ(setup.mesh.point(0).projectors[0][0].beta_G.data(), beta);
}
