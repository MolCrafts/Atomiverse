/// @file tests/cpu/vibration/test_phonon_gpw_wiring.cpp
/// @brief Unit: Phonon accepts GPW force kind without NeighborList.
///
/// Full FD spectrum is the e2e regression `reg_gpw-fd-phonon-si` (6N SCF).
/// This file only gates build-time wiring of the GPW force consumer residual.

#include <gtest/gtest.h>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/gpw/gaussian_basis.h"
#include "atomiverse/cpu/gpw/gpw_evaluator.h"
#include "atomiverse/cpu/gpw/gth_potential.h"
#include "atomiverse/cpu/vibration/phonon.h"

using namespace ATV;
using namespace ATV::cpu::vib;

TEST(PhononGpw, BuildAcceptsGpwWithoutNeighborList) {
    const cpu::gpw::Kind Si{
        molrs::Element::Si,
        cpu::gpw::GaussianBasis(molrs::Element::Si, "DZVP-MOLOPT-GTH"),
        cpu::gpw::GthPotential(molrs::Element::Si, "GTH-PBE-q4")};
    cpu::gpw::GPWEvaluator eval(
        /*kinds*/ {Si}, /*cutoff_ry*/ 150.0, /*max_iter*/ 20, /*tol*/ 1e-6,
        /*mix*/ 0.4, /*xc*/ cpu::gpw::XcKind::PADE, /*nonperiodic*/ false,
        /*compute_forces*/ true);

    Phonon phonon(1, 1, 1);
    // No set_neighbor — GPW is not a NeighborConsumer.
    phonon.on(ATV::vib::Phase::force).add_ref(eval);
    EXPECT_NO_THROW(phonon.build());
}
