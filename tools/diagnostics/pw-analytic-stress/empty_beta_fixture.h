#pragma once
#include "empty_beta_pins.h"
#include "multi_k_fd_fixture.h"
namespace empty_beta_fixture {
inline pw_analytic_diagnostic::Recipe recipe() {
    pw_analytic_diagnostic::Recipe r;
    r.smearing = ATV::electronic::Smearing::none;
    r.width_ry = 0;
    r.kmesh = 1;
    r.geometry = pw_analytic_diagnostic::Geometry::stretched_h2_triplet;
    return r;
}
inline ATV::cpu::Frame frame() {
    return pw_multi_k_fd::frame(empty_beta_qe::pin, recipe());
}
inline std::unique_ptr<ATV::cpu::pw::PW> method(ATV::cpu::xc::XCType xc) {
    auto p = empty_beta_qe::pin;
    p.model = xc;
    return pw_multi_k_fd::method(p, recipe());
}
} // namespace empty_beta_fixture
