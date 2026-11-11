#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>

#include <gtest/gtest.h>

#include "atomiverse/cpu/planewave/dfpt_phonon.h"

namespace {

using namespace ATV;
using namespace cpu::pw;

class DfptMpAdmission : public ::testing::TestWithParam<std::tuple<bool, int>> {
};

TEST_P(DfptMpAdmission, RejectsSignedWeightsBeforeScfOrSupercellWork) {
    const auto [spin, path] = GetParam();
    PW method{
        {std::make_shared<GthPseudopotential>(molrs::Element::H, "GTH-PBE-q1")},
        6.,
        1};
    method.spin_polarized = spin;
    method.multiplicity = spin ? 2 : 1;
    method.smearing = ATV::electronic::Smearing::methfessel_paxton;
    method.degauss = .04;
    DfptPhonon phonon;
    const std::array<std::array<double, 3>, 3> q{
        {{0., 0., 0.}, {.5, 0., 0.}, {.3141592653589793, 0., 0.}}};
    phonon.q_frac = q[static_cast<std::size_t>(path)];

    // An empty Frame has no SCF model. The specific admission message
    // proves rejection precedes frame validation and every q-path's work.
    cpu::Frame frame;
    try {
        (void)phonon.compute(method, frame);
        FAIL() << "MP response was admitted";
    } catch (const std::invalid_argument& error) {
        EXPECT_NE(
            std::string(error.what())
                .find("signed Methfessel-Paxton response is not implemented"),
            std::string::npos);
    }
}

INSTANTIATE_TEST_SUITE_P(RestrictedAndSpinAtAllQ, DfptMpAdmission,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Values(0, 1, 2)));

} // namespace
