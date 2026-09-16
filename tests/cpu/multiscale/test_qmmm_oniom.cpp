/// @file tests/cpu/multiscale/test_qmmm_oniom.cpp
/// @brief Unit: two-layer subtractive ONIOM energy identity.
#include <cmath>
#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/column.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"
#include "atomiverse/cpu/multiscale/qmmm_oniom.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

TEST(QmmmOniom, EnergyIdentityHighPlusLowRealMinusLowModel) {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_x)[2] = 5.0;
    f.span(kAtom_type)[0] = 1;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 18;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.span(kAtom_tag)[2] = 3;
    for (int i = 0; i < 3; ++i)
        f.span(kAtom_charge)[i] = 0;
    f.box = cpu::Box::cubic(30);
    f.blocks.at("atoms").insert(
        "qm_z",
        cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                        kDynamicFieldId});
    f.span(cpu::kAtomQmZ)[0] = 1;
    f.span(cpu::kAtomQmZ)[1] = 1;
    f.span(cpu::kAtomQmZ)[2] = 0;
    f.meta.get(kEnergy) = 0;
    f.meta.get(kEnergyElectronic) = 0;

    // H–H and Ar–Ar diagonals so E_low(model) and E_low(real) both nonzero.
    const double eps = 0.01, sigma = 3.4, cut = 12.0;
    cpu::QmmmOniom oniom{{{1, 1, eps, sigma}, {18, 18, eps, sigma}}, cut};
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = false;
    oniom.set_engine(std::move(dftb));

    cpu::Context ctx;
    oniom.build({ctx, f});
    cpu::LaunchContext lc{ctx, f, 0, 0};
    lc.forces_requested = false;
    oniom.launch(lc);

    const double E = f.meta.get(kEnergy);
    const double recon = oniom.last_high_energy() + oniom.last_low_real_energy()
                         - oniom.last_low_model_energy();
    EXPECT_TRUE(std::isfinite(E));
    // Associativity of kEnergy += high; += (real−model) vs high+real−model.
    EXPECT_NEAR(E, recon, 1e-9);
    EXPECT_NE(oniom.last_low_model_energy(), 0.0); // H–H low present
    // Real includes H–Ar (and Ar–Ar if any); must differ from model-only.
    EXPECT_GT(std::fabs(oniom.last_low_real_energy()
                         - oniom.last_low_model_energy()),
              1e-6);
}

TEST(QmmmOniom, ForcesFiniteWhenRequested) {
    cpu::Frame f(3);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_x)[1] = 0.74;
    f.span(kAtom_x)[2] = 4.0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_type)[2] = 18;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_charge)[i] = 0;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(30);
    f.blocks.at("atoms").insert(
        "qm_z",
        cpu::ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,
                        kDynamicFieldId});
    f.span(cpu::kAtomQmZ)[0] = 1;
    f.span(cpu::kAtomQmZ)[1] = 1;
    f.span(cpu::kAtomQmZ)[2] = 0;
    f.meta.get(kEnergy) = 0;

    cpu::QmmmOniom oniom{{{1, 1, 0.01, 3.4}, {18, 18, 0.01, 3.4}}, 12.0};
    cpu::semiempirical::DFTB2 dftb;
    dftb.with_forces = true;
    oniom.set_engine(std::move(dftb));
    cpu::Context ctx;
    oniom.build({ctx, f});
    cpu::LaunchContext lc{ctx, f, 0, 0};
    lc.forces_requested = true;
    oniom.launch(lc);
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(f.span(kAtom_fx)[i]));
        EXPECT_TRUE(std::isfinite(f.span(kAtom_fy)[i]));
        EXPECT_TRUE(std::isfinite(f.span(kAtom_fz)[i]));
    }
}
