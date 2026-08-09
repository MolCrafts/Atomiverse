/// @file tests/cpu/semiempirical/test_am1_scaffold.cpp
/// @brief AM1 semiempirical API scaffold tests.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/role.h"
#include "atomiverse/cpu/evaluate/evaluate.h"
#include "atomiverse/cpu/semiempirical/am1_bcc_charge_assigner.h"
#include "atomiverse/cpu/semiempirical/am1_evaluator.h"
#include "atomiverse/cpu/semiempirical/am1_parameters.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/internal/nddo_system.h"
#include "atomiverse/cpu/semiempirical/nddo_density.h"
#include "atomiverse/cpu/semiempirical/nddo_fock.h"
#include "atomiverse/cpu/semiempirical/nddo_hcore.h"
#include "atomiverse/cpu/semiempirical/nddo_matrix.h"
#include "atomiverse/cpu/semiempirical/nddo_overlap.h"
#include "atomiverse/cpu/semiempirical/nddo_two_electron.h"
#include "atomiverse/molrs.hpp"

namespace {

struct AtomSpec {
    int atomic_number;
    double x;
    double y;
    double z;
};

ATV::cpu::Frame make_frame(std::initializer_list<AtomSpec> atoms) {
    ATV::cpu::Frame f(static_cast<int>(atoms.size()));
    int i = 0;
    for (const auto& atom : atoms) {
        f.span(ATV::kAtom_type)[i] = atom.atomic_number;
        f.span(ATV::kAtom_x)[i] = atom.x;
        f.span(ATV::kAtom_y)[i] = atom.y;
        f.span(ATV::kAtom_z)[i] = atom.z;
        ++i;
    }
    f.box = ATV::cpu::Box::cubic(20.0);
    return f;
}

ATV::cpu::Frame make_h_atom_frame() {
    ATV::cpu::Frame f(1);
    f.span(ATV::kAtom_type)[0] = 1;
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.box = ATV::cpu::Box::cubic(20.0);
    return f;
}

ATV::cpu::Frame make_methane_frame() {
    ATV::cpu::Frame f(5);
    f.span(ATV::kAtom_type)[0] = 6;
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    for (int i = 1; i < 5; ++i) {
        f.span(ATV::kAtom_type)[i] = 1;
        f.span(ATV::kAtom_x)[i] = static_cast<double>(i);
        f.span(ATV::kAtom_y)[i] = 0.0;
        f.span(ATV::kAtom_z)[i] = 0.0;
    }
    f.box = ATV::cpu::Box::cubic(20.0);
    return f;
}

ATV::cpu::Frame make_tetrahedral_methane_frame() {
    ATV::cpu::Frame f(5);
    f.span(ATV::kAtom_type)[0] = 6;
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;

    const double h = 0.629118;
    f.span(ATV::kAtom_type)[1] = 1;
    f.span(ATV::kAtom_x)[1] = h;
    f.span(ATV::kAtom_y)[1] = h;
    f.span(ATV::kAtom_z)[1] = h;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_x)[2] = -h;
    f.span(ATV::kAtom_y)[2] = -h;
    f.span(ATV::kAtom_z)[2] = h;
    f.span(ATV::kAtom_type)[3] = 1;
    f.span(ATV::kAtom_x)[3] = -h;
    f.span(ATV::kAtom_y)[3] = h;
    f.span(ATV::kAtom_z)[3] = -h;
    f.span(ATV::kAtom_type)[4] = 1;
    f.span(ATV::kAtom_x)[4] = h;
    f.span(ATV::kAtom_y)[4] = -h;
    f.span(ATV::kAtom_z)[4] = -h;
    f.box = ATV::cpu::Box::cubic(20.0);
    return f;
}

ATV::cpu::Frame make_methanol_frame() {
    ATV::cpu::Frame f(6);
    f.span(ATV::kAtom_type)[0] = 6;
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_type)[1] = 8;
    f.span(ATV::kAtom_x)[1] = 1.43;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_type)[2] = 1;
    f.span(ATV::kAtom_x)[2] = 1.76;
    f.span(ATV::kAtom_y)[2] = 0.9;
    f.span(ATV::kAtom_z)[2] = 0.0;
    f.span(ATV::kAtom_type)[3] = 1;
    f.span(ATV::kAtom_x)[3] = -0.36;
    f.span(ATV::kAtom_y)[3] = 1.02;
    f.span(ATV::kAtom_z)[3] = 0.0;
    f.span(ATV::kAtom_type)[4] = 1;
    f.span(ATV::kAtom_x)[4] = -0.36;
    f.span(ATV::kAtom_y)[4] = -0.51;
    f.span(ATV::kAtom_z)[4] = 0.883346;
    f.span(ATV::kAtom_type)[5] = 1;
    f.span(ATV::kAtom_x)[5] = -0.36;
    f.span(ATV::kAtom_y)[5] = -0.51;
    f.span(ATV::kAtom_z)[5] = -0.883346;
    f.box = ATV::cpu::Box::cubic(20.0);
    return f;
}

void expect_am1_sqm_single_point_charges(ATV::cpu::Frame frame,
                                         const std::vector<double>& expected) {
    ATV::cpu::semiempirical::AM1Solver solver;
    solver.build(frame);
    const auto& result = solver.run(frame);

    EXPECT_TRUE(result.converged);
    ASSERT_EQ(result.mulliken_charges.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result.mulliken_charges[i], expected[i], 2.5e-3)
            << "atom " << i;
    }
}

::rust::Box<molrs::FrameRef>
make_molrs_methane_graph_frame(bool include_atomic_numbers = true) {
    auto frame = molrs::frame_new();

    rust::Vec<rust::String> elements;
    elements.push_back(rust::String("C"));
    elements.push_back(rust::String("H"));
    elements.push_back(rust::String("H"));
    elements.push_back(rust::String("H"));
    elements.push_back(rust::String("H"));
    molrs::frame_set_column_str(
        *frame, "atoms", "element",
        ::rust::Slice<const rust::String>(elements.data(), elements.size()));
    if (include_atomic_numbers) {
        // molrs 0.13: Z lives in `atomic_number` (UInt); `type` is a String
        // force-field label and refuses an integer outright.
        const std::vector<std::uint32_t> z = {6, 1, 1, 1, 1};
        molrs::frame_set_column_u32(
            *frame, "atoms", "atomic_number",
            ::rust::Slice<const std::uint32_t>(z.data(), z.size()));
    }

    const std::vector<double> x = {0.0, 0.629118, -0.629118, -0.629118,
                                   0.629118};
    const std::vector<double> y = {0.0, 0.629118, -0.629118, 0.629118,
                                   -0.629118};
    const std::vector<double> z = {0.0, 0.629118, 0.629118, -0.629118,
                                   -0.629118};
    molrs::frame_set_column_f64(
        *frame, "atoms", "x", ::rust::Slice<const double>(x.data(), x.size()));
    molrs::frame_set_column_f64(
        *frame, "atoms", "y", ::rust::Slice<const double>(y.data(), y.size()));
    molrs::frame_set_column_f64(
        *frame, "atoms", "z", ::rust::Slice<const double>(z.data(), z.size()));

    const std::vector<std::uint32_t> atomi = {0, 0, 0, 0};
    const std::vector<std::uint32_t> atomj = {1, 2, 3, 4};
    const std::vector<double> order = {1.0, 1.0, 1.0, 1.0};
    molrs::frame_set_column_u32(
        *frame, "bonds", "atomi",
        ::rust::Slice<const std::uint32_t>(atomi.data(), atomi.size()));
    molrs::frame_set_column_u32(
        *frame, "bonds", "atomj",
        ::rust::Slice<const std::uint32_t>(atomj.data(), atomj.size()));
    molrs::frame_set_column_f64(
        *frame, "bonds", "order",
        ::rust::Slice<const double>(order.data(), order.size()));

    return frame;
}

} // namespace

TEST(AM1Parameters, CarriesReferenceAm1ElementTable) {
    const auto* c = ATV::cpu::semiempirical::am1_parameter(6);
    ASSERT_NE(c, nullptr);
    EXPECT_STREQ(c->symbol, "C");
    EXPECT_NEAR(c->uss, -52.0286580, 1.0e-12);
    EXPECT_NEAR(c->gss, 12.2300000, 1.0e-12);

    const int expected[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 31,
                            32, 33, 34, 35, 36, 37, 38, 42, 49, 50, 51,
                            52, 53, 54, 55, 56, 80, 81, 82, 83};
    for (const int z : expected) {
        EXPECT_TRUE(ATV::cpu::semiempirical::am1_has_parameter(z)) << z;
    }

    EXPECT_FALSE(ATV::cpu::semiempirical::am1_has_parameter(21));
    EXPECT_FALSE(ATV::cpu::semiempirical::am1_has_parameter(57));
    EXPECT_FALSE(ATV::cpu::semiempirical::am1_has_parameter(84));
    EXPECT_FALSE(ATV::cpu::semiempirical::am1_has_parameter(102));
    EXPECT_FALSE(ATV::cpu::semiempirical::am1_has_parameter(104));
    EXPECT_FALSE(ATV::cpu::semiempirical::am1_has_parameter(106));
}

TEST(AM1Parameters, CarriesReferenceOrbitalOccupancyMetadata) {
    const auto* br = ATV::cpu::semiempirical::am1_parameter(35);
    const auto* iodine = ATV::cpu::semiempirical::am1_parameter(53);
    const auto* zinc = ATV::cpu::semiempirical::am1_parameter(30);
    const auto* mo = ATV::cpu::semiempirical::am1_parameter(42);
    ASSERT_NE(br, nullptr);
    ASSERT_NE(iodine, nullptr);
    ASSERT_NE(zinc, nullptr);
    ASSERT_NE(mo, nullptr);

    EXPECT_EQ(br->ios, 2);
    EXPECT_EQ(br->iop, 5);
    EXPECT_EQ(br->iod, 0);
    EXPECT_EQ(br->n_orbitals, 4);
    EXPECT_DOUBLE_EQ(br->core_charge, 7.0);

    EXPECT_EQ(iodine->ios, 2);
    EXPECT_EQ(iodine->iop, 5);
    EXPECT_DOUBLE_EQ(iodine->core_charge, 7.0);

    EXPECT_EQ(zinc->ios, 2);
    EXPECT_EQ(zinc->iop, 0);
    EXPECT_EQ(zinc->iod, 0);
    EXPECT_DOUBLE_EQ(zinc->core_charge, 2.0);

    EXPECT_EQ(mo->ios, 1);
    EXPECT_EQ(mo->iop, 0);
    EXPECT_EQ(mo->iod, 5);
    EXPECT_EQ(mo->n_orbitals, 9);
    EXPECT_DOUBLE_EQ(mo->core_charge, 6.0);
}

TEST(AM1Parameters, CarriesReferenceCalparDerivedMetadata) {
    const auto* h = ATV::cpu::semiempirical::am1_parameter(1);
    const auto* c = ATV::cpu::semiempirical::am1_parameter(6);
    const auto* o = ATV::cpu::semiempirical::am1_parameter(8);
    const auto* br = ATV::cpu::semiempirical::am1_parameter(35);
    ASSERT_NE(h, nullptr);
    ASSERT_NE(c, nullptr);
    ASSERT_NE(o, nullptr);
    ASSERT_NE(br, nullptr);

    EXPECT_EQ(c->principal_s, 2);
    EXPECT_EQ(c->principal_p, 2);
    EXPECT_EQ(c->principal_d, 0);

    EXPECT_NEAR(h->am, 0.4721552913, 1.0e-10);
    EXPECT_NEAR(h->ad, 0.4721552913, 1.0e-10);
    EXPECT_NEAR(c->dd, 0.8236735591, 1.0e-10);
    EXPECT_NEAR(c->qq, 0.7268015207, 1.0e-10);
    EXPECT_NEAR(c->am, 0.4494442102, 1.0e-10);
    EXPECT_NEAR(c->ad, 0.6082805205, 1.0e-10);
    EXPECT_NEAR(c->aq, 0.6423386614, 1.0e-10);
    EXPECT_NEAR(c->eisol, -120.8157940000, 1.0e-10);
    EXPECT_NEAR(o->dd, 0.4988896404, 1.0e-10);
    EXPECT_NEAR(o->aq, 0.9065078297, 1.0e-10);
    EXPECT_NEAR(br->qq, 1.0407133396, 1.0e-10);
    EXPECT_NEAR(br->eisol, -352.3142090000, 1.0e-10);
}

TEST(AM1Evaluator, ExplicitBindContractAndGradientToggle) {
    ATV::cpu::semiempirical::AM1Evaluator eval;
    EXPECT_EQ(eval.role(), 0); // Semiempirical: main-loop required
    EXPECT_TRUE(eval.gradient_enabled());
    eval.with_gradient(false);
    EXPECT_FALSE(eval.gradient_enabled());
    eval.with_gradient(true);
    EXPECT_TRUE(eval.gradient_enabled());

    // Explicit-bind convention (like SCF/PWEvaluator/GPWEvaluator):
    // no default bindings — the evaluator must be attached with
    // .on(phase).add().
    EXPECT_TRUE(eval.default_bindings().empty());
}

TEST(AM1BCCChargeAssigner, AssignsChargesThroughMolrsTypifier) {
    auto frame = make_molrs_methane_graph_frame();
    ATV::cpu::semiempirical::AM1BCCChargeAssigner assigner;

    const auto charges = assigner.assign(*frame);

    ASSERT_EQ(charges.size(), 5u);
    const auto written = molrs::frame_column_f64(*frame, "atoms", "charge");
    ASSERT_EQ(written.size(), charges.size());
    for (std::size_t i = 0; i < charges.size(); ++i)
        EXPECT_DOUBLE_EQ(written[i], charges[i]) << "atom " << i;

    EXPECT_TRUE(assigner.last_am1_result().converged);
    EXPECT_NEAR(charges[0], -0.1047, 5.0e-3);
    for (std::size_t i = 1; i < charges.size(); ++i)
        EXPECT_NEAR(charges[i], 0.0262, 5.0e-3) << "atom " << i;
}

TEST(AM1BCCChargeAssigner, RejectsSymbolOnlyAtomicNumberFallback) {
    auto frame = make_molrs_methane_graph_frame(false);
    ATV::cpu::semiempirical::AM1BCCChargeAssigner assigner;
    EXPECT_THROW(
        {
            try {
                (void)assigner.assign(*frame);
            } catch (const std::runtime_error& error) {
                // Since molrs 0.13 the element identity column is
                // `atoms.atomic_number`; a symbol-only frame must still be
                // rejected rather than silently falling back.
                EXPECT_NE(
                    std::string(error.what()).find("atoms.atomic_number"),
                    std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

TEST(AM1Solver, BuildRejectsElementsWithoutPortedParameters) {
    ATV::cpu::semiempirical::AM1Solver solver;
    ATV::cpu::Frame frame(1);
    frame.span(ATV::kAtom_type)[0] = 99;
    EXPECT_THROW(solver.build(frame), std::runtime_error);
}

TEST(NDDOSystem, MethaneMatchesReferenceOrbitalAndElectronBookkeeping) {
    auto frame = make_methane_frame();
    const auto system = ATV::cpu::semiempirical::NDDOSystem{frame, 0, 1};

    ASSERT_EQ(system.atoms.size(), 5u);
    EXPECT_EQ(system.n_orbitals, 8);
    EXPECT_EQ(system.n_electrons, 8);
    EXPECT_EQ(system.n_alpha, 4);
    EXPECT_EQ(system.n_beta, 4);
    EXPECT_EQ(system.charge, 0);

    EXPECT_EQ(system.atoms[0].atomic_number, 6);
    EXPECT_EQ(system.atoms[0].first_ao, 0);
    EXPECT_EQ(system.atoms[0].last_ao, 3);
    EXPECT_EQ(system.atoms[0].n_orbitals, 4);
    EXPECT_DOUBLE_EQ(system.atoms[0].core_charge, 4.0);

    for (int i = 1; i < 5; ++i) {
        const auto& atom = system.atoms[static_cast<std::size_t>(i)];
        EXPECT_EQ(atom.atomic_number, 1);
        EXPECT_EQ(atom.first_ao, i + 3);
        EXPECT_EQ(atom.last_ao, i + 3);
        EXPECT_EQ(atom.n_orbitals, 1);
        EXPECT_DOUBLE_EQ(atom.core_charge, 1.0);
    }
}

TEST(NDDOSystem, DOrbitalBookkeepingComesFromReferenceButFockIsGuarded) {
    ATV::cpu::Frame frame(1);
    frame.span(ATV::kAtom_type)[0] = 42;
    frame.box = ATV::cpu::Box::cubic(20.0);

    const auto system = ATV::cpu::semiempirical::NDDOSystem{frame, 0, 1};
    ASSERT_EQ(system.atoms.size(), 1u);
    EXPECT_EQ(system.n_orbitals, 9);
    EXPECT_EQ(system.n_electrons, 6);
    EXPECT_DOUBLE_EQ(system.atoms[0].core_charge, 6.0);
    EXPECT_THROW(ATV::cpu::semiempirical::NDDOOneCenterIntegralBuilder{}.build(
                     system.atoms[0]),
                 std::runtime_error);
}

TEST(NDDOSystem, InitialDensityGuessFollowsReferencePdiagRule) {
    auto frame = make_methane_frame();
    const auto system = ATV::cpu::semiempirical::NDDOSystem{frame, 0, 1};
    const auto diag =
        ATV::cpu::semiempirical::initial_diagonal_density_guess(system);

    ASSERT_EQ(diag.size(), 8u);
    for (double value : diag) {
        EXPECT_DOUBLE_EQ(value, 1.0);
    }
}

TEST(NDDOSystem, PackedDensityPopulationsFollowReferenceChrgeRule) {
    auto frame = make_methane_frame();
    const auto system = ATV::cpu::semiempirical::NDDOSystem{frame, 0, 1};
    std::vector<double> density(
        static_cast<std::size_t>(system.n_orbitals)
            * static_cast<std::size_t>(system.n_orbitals + 1) / 2u,
        0.0);
    for (int ao = 0; ao < 4; ++ao) {
        density[ATV::cpu::semiempirical::packed_index(ao, ao)] = 1.1;
    }
    for (int ao = 4; ao < 8; ++ao) {
        density[ATV::cpu::semiempirical::packed_index(ao, ao)] = 0.9;
    }

    const auto populations =
        ATV::cpu::semiempirical::atomic_populations_from_packed_density(
            system, density);
    const auto charges =
        ATV::cpu::semiempirical::atomic_charges_from_packed_density(system,
                                                                    density);

    ASSERT_EQ(populations.size(), 5u);
    ASSERT_EQ(charges.size(), 5u);
    EXPECT_NEAR(populations[0], 4.4, 1.0e-12);
    EXPECT_NEAR(charges[0], -0.4, 1.0e-12);
    for (std::size_t i = 1; i < 5; ++i) {
        EXPECT_NEAR(populations[i], 0.9, 1.0e-12);
        EXPECT_NEAR(charges[i], 0.1, 1.0e-12);
    }
}

TEST(NDDODensityBuilder, ClosedShellIdentityMatchesReferenceDensit) {
    const std::vector<double> identity = {
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    ATV::cpu::semiempirical::NDDODensityInput input;
    input.mdim = 4;
    input.norbs = 4;
    input.n_occupied = 2;
    input.n_fractional = 2;
    input.full_occupancy = 2.0;
    input.fractional_occupancy = 0.0;

    const auto density =
        ATV::cpu::semiempirical::NDDODensityBuilder{}.build_packed(identity,
                                                                   input);
    EXPECT_DOUBLE_EQ(density[ATV::cpu::semiempirical::packed_index(0, 0)], 2.0);
    EXPECT_DOUBLE_EQ(density[ATV::cpu::semiempirical::packed_index(1, 1)], 2.0);
    EXPECT_DOUBLE_EQ(density[ATV::cpu::semiempirical::packed_index(2, 2)], 0.0);
    EXPECT_DOUBLE_EQ(density[ATV::cpu::semiempirical::packed_index(3, 3)], 0.0);
}

TEST(NDDODensityBuilder, FractionalOccupationMatchesReferenceDensit) {
    const std::vector<double> identity = {
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    ATV::cpu::semiempirical::NDDODensityInput input;
    input.mdim = 4;
    input.norbs = 4;
    input.n_occupied = 1;
    input.n_fractional = 2;
    input.full_occupancy = 2.0;
    input.fractional_occupancy = 0.5;

    const auto density =
        ATV::cpu::semiempirical::NDDODensityBuilder{}.build_packed(identity,
                                                                   input);
    EXPECT_DOUBLE_EQ(density[ATV::cpu::semiempirical::packed_index(0, 0)], 2.0);
    EXPECT_DOUBLE_EQ(density[ATV::cpu::semiempirical::packed_index(1, 1)], 0.5);
    EXPECT_DOUBLE_EQ(density[ATV::cpu::semiempirical::packed_index(2, 2)], 0.0);
}

TEST(NDDODensityBuilder, PositronEquivalentBranchMatchesReferenceDensit) {
    const std::vector<double> identity = {
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };
    ATV::cpu::semiempirical::NDDODensityInput input;
    input.mdim = 4;
    input.norbs = 4;
    input.n_occupied = 1;
    input.n_fractional = 3;
    input.full_occupancy = 2.0;
    input.fractional_occupancy = 0.5;
    input.mode = 1;

    const auto density =
        ATV::cpu::semiempirical::NDDODensityBuilder{}.build_packed(identity,
                                                                   input);
    EXPECT_DOUBLE_EQ(density[ATV::cpu::semiempirical::packed_index(0, 0)], 2.0);
    EXPECT_DOUBLE_EQ(density[ATV::cpu::semiempirical::packed_index(1, 1)], 0.5);
    EXPECT_DOUBLE_EQ(density[ATV::cpu::semiempirical::packed_index(2, 2)], 0.5);
    EXPECT_DOUBLE_EQ(density[ATV::cpu::semiempirical::packed_index(3, 3)], 0.0);
}

TEST(NDDODensityBuilder, RotatedOccupiedOrbitalProducesOffDiagonalDensity) {
    const double inv_sqrt2 = 0.70710678118654752440;
    const std::vector<double> coefficients = {
        inv_sqrt2,
        inv_sqrt2,
        inv_sqrt2,
        -inv_sqrt2,
    };
    ATV::cpu::semiempirical::NDDODensityInput input;
    input.mdim = 2;
    input.norbs = 2;
    input.n_occupied = 1;
    input.n_fractional = 1;
    input.full_occupancy = 2.0;
    input.fractional_occupancy = 0.0;

    const auto density =
        ATV::cpu::semiempirical::NDDODensityBuilder{}.build_packed(coefficients,
                                                                   input);
    EXPECT_NEAR(density[ATV::cpu::semiempirical::packed_index(0, 0)], 1.0,
                1.0e-14);
    EXPECT_NEAR(density[ATV::cpu::semiempirical::packed_index(1, 0)], 1.0,
                1.0e-14);
    EXPECT_NEAR(density[ATV::cpu::semiempirical::packed_index(1, 1)], 1.0,
                1.0e-14);
}

TEST(NDDOHCoreBuilder, OneCenterDiagonalMatchesReferenceUspdInitialization) {
    auto frame = make_methane_frame();
    const auto system = ATV::cpu::semiempirical::NDDOSystem{frame, 0, 1};
    const auto hcore =
        ATV::cpu::semiempirical::NDDOHCoreBuilder{}.build_one_center(system);

    const auto* c = ATV::cpu::semiempirical::am1_parameter(6);
    const auto* h = ATV::cpu::semiempirical::am1_parameter(1);
    ASSERT_NE(c, nullptr);
    ASSERT_NE(h, nullptr);
    ASSERT_EQ(hcore.packed_hcore.size(), 36u);
    EXPECT_FALSE(hcore.includes_two_center_terms);
    EXPECT_DOUBLE_EQ(hcore.nuclear_repulsion_ev, 0.0);

    EXPECT_DOUBLE_EQ(
        hcore.packed_hcore[ATV::cpu::semiempirical::packed_index(0, 0)],
        c->uss);
    for (int ao = 1; ao < 4; ++ao) {
        EXPECT_DOUBLE_EQ(
            hcore.packed_hcore[ATV::cpu::semiempirical::packed_index(ao, ao)],
            c->upp);
    }
    for (int ao = 4; ao < 8; ++ao) {
        EXPECT_DOUBLE_EQ(
            hcore.packed_hcore[ATV::cpu::semiempirical::packed_index(ao, ao)],
            h->uss);
    }
    EXPECT_DOUBLE_EQ(
        hcore.packed_hcore[ATV::cpu::semiempirical::packed_index(1, 0)], 0.0);
    EXPECT_DOUBLE_EQ(
        hcore.packed_hcore[ATV::cpu::semiempirical::packed_index(4, 0)], 0.0);
}

TEST(NDDOFockBuilder, OneCenterIntegralsMatchReferenceWstoreSpSubset) {
    auto frame = make_methane_frame();
    const auto system = ATV::cpu::semiempirical::NDDOSystem{frame, 0, 1};
    const auto* c = ATV::cpu::semiempirical::am1_parameter(6);
    ASSERT_NE(c, nullptr);

    const auto w =
        ATV::cpu::semiempirical::NDDOOneCenterIntegralBuilder{}.build(
            system.atoms[0]);
    ASSERT_EQ(w.dimension, 10);
    EXPECT_DOUBLE_EQ(w.at(0, 0), c->gss);
    EXPECT_DOUBLE_EQ(w.at(2, 0), c->gsp);
    EXPECT_DOUBLE_EQ(w.at(0, 2), c->gsp);
    EXPECT_DOUBLE_EQ(w.at(2, 2), c->gpp);
    EXPECT_DOUBLE_EQ(w.at(5, 2), c->gp2);
    EXPECT_DOUBLE_EQ(w.at(1, 1), c->hsp);
    EXPECT_DOUBLE_EQ(w.at(4, 4), 0.5 * (c->gpp - c->gp2));
}

TEST(NDDOFockBuilder, OneCenterFockUsesReferenceFock1Contraction) {
    auto frame = make_h_atom_frame();
    const auto system = ATV::cpu::semiempirical::NDDOSystem{frame, 0, 2};
    const auto* h = ATV::cpu::semiempirical::am1_parameter(1);
    ASSERT_NE(h, nullptr);

    std::vector<double> hcore = {0.0};
    std::vector<double> total_density = {2.0};
    std::vector<double> spin_density = {1.0};
    const auto fock = ATV::cpu::semiempirical::NDDOFockBuilder{}.add_one_center(
        system, hcore, total_density, spin_density);

    ASSERT_EQ(fock.size(), 1u);
    EXPECT_DOUBLE_EQ(fock[0], h->gss);
}

TEST(NDDOFockBuilder, ElectronicEnergyMatchesReferenceHelectPackedRules) {
    const std::vector<double> density = {1.0, 3.0, 2.0};
    const std::vector<double> hcore = {4.0, 10.0, 8.0};
    const std::vector<double> fock = {6.0, 20.0, 12.0};
    const double energy =
        ATV::cpu::semiempirical::electronic_energy_ev(2, density, hcore, fock);
    EXPECT_DOUBLE_EQ(energy, 115.0);
}

TEST(NDDOOverlap, H1ElecMatchesReferenceH2HcoreOffDiagonal) {
    const auto h1 = ATV::cpu::semiempirical::one_electron_two_center(
        1, 1, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.74});

    // AM1 reference 23.2.5, input: `AM1 1SCF HCORE XYZ` for H2 at 0.74 A.
    EXPECT_NEAR(h1.at(0, 0), -4.198332, 5.0e-6);
    EXPECT_DOUBLE_EQ(h1.at(0, 1), 0.0);
}

TEST(NDDOHCoreBuilder, SPHcoreMatchesReferenceH2HcoreMatrix) {
    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_type)[0] = 1;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.74;
    frame.box = ATV::cpu::Box::cubic(20.0);

    const auto system = ATV::cpu::semiempirical::NDDOSystem{frame, 0, 1};
    const auto hcore =
        ATV::cpu::semiempirical::NDDOHCoreBuilder{}.build(system, frame);

    ASSERT_EQ(hcore.packed_hcore.size(), 3u);
    EXPECT_TRUE(hcore.includes_two_center_terms);
    // AM1 reference 23.2.5, input: `AM1 1SCF HCORE XYZ` for H2 at 0.74 A.
    EXPECT_NEAR(hcore.packed_hcore[ATV::cpu::semiempirical::packed_index(0, 0)],
                -22.118203, 5.0e-6);
    EXPECT_NEAR(hcore.packed_hcore[ATV::cpu::semiempirical::packed_index(1, 0)],
                -4.198332, 5.0e-6);
    EXPECT_NEAR(hcore.packed_hcore[ATV::cpu::semiempirical::packed_index(1, 1)],
                -22.118203, 5.0e-6);
    EXPECT_GT(hcore.nuclear_repulsion_ev, 0.0);
}

TEST(NDDOTwoElectron, H2TwoCenterIntegralMatchesReferenceHcorePrint) {
    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_type)[0] = 1;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.74;
    frame.box = ATV::cpu::Box::cubic(20.0);

    const auto system = ATV::cpu::semiempirical::NDDOSystem{frame, 0, 1};
    const auto w = ATV::cpu::semiempirical::build_two_center_integral_block(
        system, frame.span(ATV::kAtom_x), frame.span(ATV::kAtom_y),
        frame.span(ATV::kAtom_z), 1, 0);

    ASSERT_EQ(w.n_pairs_i, 1);
    ASSERT_EQ(w.n_pairs_j, 1);
    ASSERT_EQ(w.values.size(), 1u);
    // AM1 reference 23.2.5 `TWO-ELECTRON MATRIX IN HCORE`, middle H-H term.
    EXPECT_NEAR(w.at(0, 0), 10.7218, 5.0e-5);
}

TEST(NDDOFockBuilder, H2FinalDensityFockMatchesReferenceEigenvalues) {
    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_type)[0] = 1;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.74;
    frame.box = ATV::cpu::Box::cubic(20.0);

    const auto system = ATV::cpu::semiempirical::NDDOSystem{frame, 0, 1};
    const auto hcore =
        ATV::cpu::semiempirical::NDDOHCoreBuilder{}.build(system, frame);
    const std::vector<double> density = {1.0, 1.0, 1.0};
    const std::vector<double> spin_density = {0.5, 0.5, 0.5};
    const auto fock = ATV::cpu::semiempirical::NDDOFockBuilder{}.build(
        system, frame, hcore.packed_hcore, density, spin_density);
    const auto orbitals =
        ATV::cpu::semiempirical::NDDODiagonalizer{}.diagonalize_packed(fock, 2);

    ASSERT_EQ(orbitals.eigenvalues.size(), 2u);
    // AM1 reference 23.2.5 final H2 eigenvalues for `AM1 1SCF HCORE XYZ`.
    EXPECT_NEAR(orbitals.eigenvalues[0], -14.531648, 5.0e-5);
    EXPECT_NEAR(orbitals.eigenvalues[1], 4.58679, 5.0e-5);
}

TEST(NDDOMatrixBuilder, ConvertsPackedLowerTriangleToDenseSymmetricMatrix) {
    const std::vector<double> packed = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const auto dense =
        ATV::cpu::semiempirical::NDDOMatrixBuilder{}.packed_to_dense(packed, 3);
    EXPECT_DOUBLE_EQ(dense(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(dense(1, 0), 2.0);
    EXPECT_DOUBLE_EQ(dense(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(dense(2, 1), 5.0);
    EXPECT_DOUBLE_EQ(dense(1, 2), 5.0);
    EXPECT_DOUBLE_EQ(dense(2, 2), 6.0);

    const auto roundtrip =
        ATV::cpu::semiempirical::NDDOMatrixBuilder{}.dense_to_packed(dense);
    EXPECT_EQ(roundtrip, packed);
}

TEST(NDDODiagonalizer, DiagonalizesPackedSymmetricFockMatrix) {
    const std::vector<double> packed = {2.0, 0.0, 1.0};
    const auto orbitals =
        ATV::cpu::semiempirical::NDDODiagonalizer{}.diagonalize_packed(packed,
                                                                       2);
    ASSERT_EQ(orbitals.eigenvalues.size(), 2u);
    ASSERT_EQ(orbitals.coefficients_row_major.size(), 4u);
    EXPECT_NEAR(orbitals.eigenvalues[0], 1.0, 1.0e-12);
    EXPECT_NEAR(orbitals.eigenvalues[1], 2.0, 1.0e-12);
}

TEST(AM1Solver, BuildCreatesReferenceAoIndexSnapshot) {
    ATV::cpu::semiempirical::AM1Solver solver;
    auto frame = make_methane_frame();
    solver.build(frame);

    const auto& system = solver.system();
    EXPECT_EQ(system.n_orbitals, 8);
    EXPECT_EQ(system.n_electrons, 8);
    EXPECT_EQ(system.atoms[0].first_ao, 0);
    EXPECT_EQ(system.atoms[4].last_ao, 7);
}

TEST(AM1Solver, BuildCreatesReferenceScfInitialState) {
    ATV::cpu::semiempirical::AM1Solver solver;
    auto frame = make_methane_frame();
    solver.build(frame);

    const auto& state = solver.state();
    ASSERT_EQ(state.initial_diagonal_density.size(), 8u);
    ASSERT_EQ(state.initial_packed_density.size(), 36u);
    ASSERT_EQ(state.initial_spin_density.size(), 36u);
    ASSERT_EQ(state.packed_hcore.size(), 36u);
    ASSERT_EQ(state.initial_one_center_fock.size(), 36u);
    EXPECT_GT(state.nuclear_repulsion_ev, 0.0);

    for (int ao = 0; ao < 8; ++ao) {
        const auto ii = ATV::cpu::semiempirical::packed_index(ao, ao);
        EXPECT_DOUBLE_EQ(
            state.initial_packed_density[ii],
            state.initial_diagonal_density[static_cast<std::size_t>(ao)]);
        EXPECT_DOUBLE_EQ(state.initial_spin_density[ii],
                         0.5 * state.initial_packed_density[ii]);
    }
    EXPECT_NE(
        state.initial_one_center_fock[ATV::cpu::semiempirical::packed_index(0,
                                                                            0)],
        state.packed_hcore[ATV::cpu::semiempirical::packed_index(0, 0)]);
}

TEST(AM1Solver, RunConvergesClosedShellH2AgainstReferenceChargeOracle) {
    ATV::cpu::semiempirical::AM1Solver solver;
    ATV::cpu::Frame frame(2);
    frame.span(ATV::kAtom_type)[0] = 1;
    frame.span(ATV::kAtom_type)[1] = 1;
    frame.span(ATV::kAtom_x)[0] = 0.0;
    frame.span(ATV::kAtom_y)[0] = 0.0;
    frame.span(ATV::kAtom_z)[0] = 0.0;
    frame.span(ATV::kAtom_x)[1] = 0.0;
    frame.span(ATV::kAtom_y)[1] = 0.0;
    frame.span(ATV::kAtom_z)[1] = 0.74;
    frame.box = ATV::cpu::Box::cubic(20.0);

    solver.build(frame);
    const auto& result = solver.run(frame);

    EXPECT_TRUE(result.converged);
    EXPECT_LE(result.iterations, 4);
    ASSERT_EQ(result.mulliken_charges.size(), 2u);
    // AM1 reference 23.2.5 H2 output: both NET ATOMIC CHARGES print as 0.000000.
    EXPECT_NEAR(result.mulliken_charges[0], 0.0, 1.0e-8);
    EXPECT_NEAR(result.mulliken_charges[1], 0.0, 1.0e-8);
}

TEST(AM1Solver, RunConvergesMethaneAgainstReferenceChargeOracle) {
    ATV::cpu::semiempirical::AM1Solver solver;
    auto frame = make_tetrahedral_methane_frame();

    solver.build(frame);
    const auto& result = solver.run(frame);

    EXPECT_TRUE(result.converged);
    ASSERT_EQ(result.mulliken_charges.size(), 5u);
    // AM1 reference 23.2.5, `AM1 1SCF XYZ`, tetrahedral CH4 at C-H 1.0897 A.
    EXPECT_NEAR(result.mulliken_charges[0], -0.262125, 5.0e-5);
    for (int i = 1; i < 5; ++i) {
        EXPECT_NEAR(result.mulliken_charges[static_cast<std::size_t>(i)],
                    0.065531, 5.0e-5);
    }
}

TEST(AM1Solver, RunConvergesMethanolAgainstReferenceChargeOracle) {
    ATV::cpu::semiempirical::AM1Solver solver;
    auto frame = make_methanol_frame();

    solver.build(frame);
    const auto& result = solver.run(frame);

    EXPECT_TRUE(result.converged);
    ASSERT_EQ(result.mulliken_charges.size(), 6u);
    // AM1 reference 23.2.5, `AM1 1SCF XYZ`, same coordinates as fixture.
    const double expected[] = {-0.056689, -0.341786, 0.200857,
                               0.049308,  0.074155,  0.074155};
    for (int i = 0; i < 6; ++i) {
        EXPECT_NEAR(result.mulliken_charges[static_cast<std::size_t>(i)],
                    expected[i], 5.0e-5);
    }
}

TEST(AM1Solver, RunMatchesreferenceSQMSinglePointOrganicChargeOracles) {
    // reference25 `sqm`, AM1 with `maxcyc=0`, `scfconv=1.d-10`, no
    // geometry optimization. These are the AM1 precharge values that feed
    // AM1-BCC when antechamber's SQM minimization is disabled.
    expect_am1_sqm_single_point_charges(make_frame({{6, 0.0, 0.0, 0.0},
                                                    {1, 0.63, 0.63, 0.63},
                                                    {1, -0.63, -0.63, 0.63},
                                                    {1, -0.63, 0.63, -0.63},
                                                    {1, 0.63, -0.63, -0.63}}),
                                        {-0.262, 0.066, 0.066, 0.066, 0.066});

    expect_am1_sqm_single_point_charges(
        make_frame({{6, 0.0, 0.0, 0.0},
                    {8, 1.43, 0.0, 0.0},
                    {1, -0.63, 0.63, 0.63},
                    {1, -0.63, -0.63, 0.63},
                    {1, -0.63, 0.0, -0.89},
                    {1, 1.76, 0.89, 0.0}}),
        {-0.130, -0.277, 0.046, 0.084, 0.076, 0.202});

    expect_am1_sqm_single_point_charges(make_frame({{6, 0.0, 0.0, 0.0},
                                                    {17, 1.78, 0.0, 0.0},
                                                    {1, -0.63, 0.63, 0.63},
                                                    {1, -0.63, -0.63, 0.63},
                                                    {1, -0.63, 0.0, -0.89}}),
                                        {-0.234, -0.043, 0.085, 0.085, 0.106});

    expect_am1_sqm_single_point_charges(
        make_frame({{7, 0.0, 0.0, 0.0},
                    {1, 0.0, 0.9377, 0.3816},
                    {1, 0.8121, -0.4688, 0.3816},
                    {1, -0.8121, -0.4688, 0.3816}}),
        {-0.362, 0.121, 0.121, 0.121});

    expect_am1_sqm_single_point_charges(
        make_frame({{8, 0.0, 0.0, 0.0},
                    {1, 0.9572, 0.0, 0.0},
                    {1, -0.239987, 0.926627, 0.0}}),
        {-0.385, 0.193, 0.193});

    expect_am1_sqm_single_point_charges(make_frame({{6, 0.0, 0.0, 0.0},
                                                    {9, 1.39, 0.0, 0.0},
                                                    {1, -0.63, 0.63, 0.63},
                                                    {1, -0.63, -0.63, 0.63},
                                                    {1, -0.63, 0.0, -0.89}}),
                                        {-0.105, -0.121, 0.066, 0.066, 0.094});
}

TEST(AM1Evaluator, EvaluateAcceptsChargeCapableEvaluatorButLaunchIsGuarded) {
    ATV::cpu::Evaluate eval;
    eval.on(ATV::evaluate::Phase::evaluate)
        .add(ATV::cpu::semiempirical::AM1Evaluator{});
    EXPECT_NO_THROW(eval.build());

    auto frame = make_h_atom_frame();
    EXPECT_THROW(eval.run(frame), std::runtime_error);
}

TEST(AM1Evaluator, LaunchWritesClosedShellChargesToFrame) {
    ATV::cpu::Context runtime;
    auto frame = make_tetrahedral_methane_frame();
    ATV::cpu::semiempirical::AM1Evaluator evaluator;
    evaluator.with_gradient(false);

    ATV::cpu::BuildContext build{runtime, frame};
    ATV::cpu::LaunchContext launch{
        runtime, frame, ATV::slot_of(ATV::evaluate::Phase::evaluate), 0};
    evaluator.build(build);
    evaluator.launch(launch);

    EXPECT_NEAR(frame.span(ATV::kAtom_charge)[0], -0.262125, 5.0e-5);
    for (int i = 1; i < 5; ++i) {
        EXPECT_NEAR(frame.span(ATV::kAtom_charge)[i], 0.065531, 5.0e-5);
    }
    EXPECT_LT(frame.meta.get(ATV::kEnergy), 0.0);
}

TEST(NDDOSolver, AnalyticForceMatchesRescfEnergyFiniteDifference) {
    // Slightly stretched water so residual forces are non-zero.
    auto frame = make_frame(
        {{8, 0.0, 0.0, 0.0}, {1, 1.05, 0.0, 0.0}, {1, -0.26, 1.0, 0.0}});

    ATV::cpu::semiempirical::AM1Solver solver;
    solver.build(frame);
    const auto& scf = solver.run(frame);
    ASSERT_TRUE(scf.converged);
    solver.compute_forces(frame, 1.0e-4);
    const auto& forces = solver.last_result().forces_ev_a;
    ASSERT_EQ(forces.size(), 9u);

    // Re-SCF central FD of total energy vs analytic (fixed-P) force.
    constexpr double h = 5.0e-4; // Å
    auto energy = [&](ATV::cpu::Frame f) {
        ATV::cpu::semiempirical::AM1Solver s;
        s.build(f);
        return s.run(f).total_energy_ev;
    };

    double max_abs_err = 0.0;
    double net_fx = 0.0;
    double net_fy = 0.0;
    double net_fz = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int d = 0; d < 3; ++d) {
            auto fp = frame;
            auto fm = frame;
            ATV::StorageReal* cp = d == 0 ? fp.span(ATV::kAtom_x)
                                          : (d == 1 ? fp.span(ATV::kAtom_y)
                                                    : fp.span(ATV::kAtom_z));
            ATV::StorageReal* cm = d == 0 ? fm.span(ATV::kAtom_x)
                                          : (d == 1 ? fm.span(ATV::kAtom_y)
                                                    : fm.span(ATV::kAtom_z));
            cp[i] =
                static_cast<ATV::StorageReal>(static_cast<double>(cp[i]) + h);
            cm[i] =
                static_cast<ATV::StorageReal>(static_cast<double>(cm[i]) - h);
            const double fd = -(energy(fp) - energy(fm)) / (2.0 * h);
            const double an = forces[static_cast<std::size_t>(3 * i + d)];
            max_abs_err = std::max(max_abs_err, std::abs(an - fd));
            EXPECT_NEAR(an, fd, 5.0e-3)
                << "atom " << i << " dim " << d << " analytic=" << an
                << " re-SCF FD=" << fd;
        }
        net_fx += forces[static_cast<std::size_t>(3 * i + 0)];
        net_fy += forces[static_cast<std::size_t>(3 * i + 1)];
        net_fz += forces[static_cast<std::size_t>(3 * i + 2)];
    }
    EXPECT_LT(max_abs_err, 5.0e-3);
    EXPECT_NEAR(net_fx, 0.0, 1.0e-6);
    EXPECT_NEAR(net_fy, 0.0, 1.0e-6);
    EXPECT_NEAR(net_fz, 0.0, 1.0e-6);
}

TEST(AM1Evaluator, LaunchWritesForcesWhenGradientEnabled) {
    ATV::cpu::Context runtime;
    auto frame = make_frame(
        {{8, 0.0, 0.0, 0.0}, {1, 1.05, 0.0, 0.0}, {1, -0.26, 1.0, 0.0}});
    // Zero forces so we can detect writes.
    for (int i = 0; i < 3; ++i) {
        frame.span(ATV::kAtom_fx)[i] = 0;
        frame.span(ATV::kAtom_fy)[i] = 0;
        frame.span(ATV::kAtom_fz)[i] = 0;
    }

    ATV::cpu::semiempirical::AM1Evaluator evaluator; // gradient on by default
    ATV::cpu::BuildContext build{runtime, frame};
    ATV::cpu::LaunchContext launch{
        runtime, frame, ATV::slot_of(ATV::evaluate::Phase::evaluate), 0};
    evaluator.build(build);
    evaluator.launch(launch);

    EXPECT_LT(frame.meta.get(ATV::kEnergy), 0.0);
    double f2 = 0.0;
    for (int i = 0; i < 3; ++i) {
        f2 += static_cast<double>(frame.span(ATV::kAtom_fx)[i])
              * static_cast<double>(frame.span(ATV::kAtom_fx)[i]);
        f2 += static_cast<double>(frame.span(ATV::kAtom_fy)[i])
              * static_cast<double>(frame.span(ATV::kAtom_fy)[i]);
        f2 += static_cast<double>(frame.span(ATV::kAtom_fz)[i])
              * static_cast<double>(frame.span(ATV::kAtom_fz)[i]);
    }
    EXPECT_GT(f2, 1.0e-8);

    // Energy-only launch leaves forces untouched when gradient disabled.
    for (int i = 0; i < 3; ++i) {
        frame.span(ATV::kAtom_fx)[i] = 42.0;
        frame.span(ATV::kAtom_fy)[i] = 42.0;
        frame.span(ATV::kAtom_fz)[i] = 42.0;
    }
    evaluator.with_gradient(false);
    evaluator.launch(launch);
    for (int i = 0; i < 3; ++i) {
        EXPECT_DOUBLE_EQ(static_cast<double>(frame.span(ATV::kAtom_fx)[i]),
                         42.0);
        EXPECT_DOUBLE_EQ(static_cast<double>(frame.span(ATV::kAtom_fy)[i]),
                         42.0);
        EXPECT_DOUBLE_EQ(static_cast<double>(frame.span(ATV::kAtom_fz)[i]),
                         42.0);
    }
}

TEST(AM1Solver, ExperimentalWaterGeometryConverges) {
    auto frame = make_frame({{8, 0.0, 0.0, 0.0},
                             {1, 0.9572, 0.0, 0.0},
                             {1, -0.239987, 0.926627, 0.0}});
    ATV::cpu::semiempirical::AM1Solver solver;
    solver.build(frame);
    const auto& r = solver.run(frame);
    EXPECT_TRUE(r.converged);
    EXPECT_LT(r.total_energy_ev, 0.0);
    solver.compute_forces(frame);
    EXPECT_EQ(solver.last_result().forces_ev_a.size(), 9u);
}

TEST(AM1Evaluator, MultiLaunchWarmStartOnWater) {
    ATV::cpu::Context runtime;
    auto frame = make_frame({{8, 0.0, 0.0, 0.0},
                             {1, 0.9572, 0.0, 0.0},
                             {1, -0.239987, 0.926627, 0.0}});
    ATV::cpu::semiempirical::AM1Evaluator eval;
    ATV::cpu::BuildContext build{runtime, frame};
    eval.build(build);
    ATV::cpu::LaunchContext launch{
        runtime, frame, ATV::slot_of(ATV::evaluate::Phase::evaluate), 0};
    for (int i = 0; i < 5; ++i) {
        // small random-ish displacement of H
        frame.span(ATV::kAtom_x)[1] =
            static_cast<ATV::StorageReal>(0.9572 + 0.001 * i);
        eval.launch(launch);
        EXPECT_LT(frame.meta.get(ATV::kEnergy), 0.0) << "step " << i;
    }
}

TEST(AM1Parameters, AtomicHeatsMatchMopacConvention) {
    EXPECT_NEAR(ATV::cpu::semiempirical::am1_atomic_heat_kcal_mol(1), 52.102,
                1e-12);
    EXPECT_NEAR(ATV::cpu::semiempirical::am1_atomic_heat_kcal_mol(6), 170.89,
                1e-12);
    EXPECT_NEAR(ATV::cpu::semiempirical::am1_atomic_heat_kcal_mol(8), 59.559,
                1e-12);
    EXPECT_THROW(ATV::cpu::semiempirical::am1_atomic_heat_kcal_mol(99),
                 std::runtime_error);
}

TEST(AM1Solver, WaterHeatOfFormationNearAm1Literature) {
    // Experimental-like geometry (Å). Dewar AM1 water ΔHf is typically near
    // −59 kcal/mol (expt. −57.8); pin within a few kcal/mol of the AM1 regime.
    auto frame = make_frame({{8, 0.0, 0.0, 0.0},
                             {1, 0.9572, 0.0, 0.0},
                             {1, -0.239987, 0.926627, 0.0}});
    ATV::cpu::semiempirical::AM1Solver solver;
    solver.build(frame);
    const auto& r = solver.run(frame);
    ASSERT_TRUE(r.converged);
    // Sanity: formula recovers roughly experimental water heat of formation.
    EXPECT_NEAR(r.heat_of_formation_kcal_mol, -59.0, 8.0)
        << "ΔHf=" << r.heat_of_formation_kcal_mol;
    // Cross-check explicit formula against result field.
    double sum_eisol = 0.0;
    double sum_at = 0.0;
    for (int z : {8, 1, 1}) {
        sum_eisol += ATV::cpu::semiempirical::am1_parameter(z)->eisol;
        sum_at += ATV::cpu::semiempirical::am1_atomic_heat_kcal_mol(z);
    }
    const double expect = (r.total_energy_ev - sum_eisol)
                              * ATV::cpu::semiempirical::kAm1EvToKcalMol
                          + sum_at;
    EXPECT_NEAR(r.heat_of_formation_kcal_mol, expect, 1e-9);
}
