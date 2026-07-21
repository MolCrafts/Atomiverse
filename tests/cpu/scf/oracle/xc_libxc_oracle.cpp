/// @file tests/cpu/scf/oracle/xc_libxc_oracle.cpp
/// @brief libxc-backed XC oracle implementation (test-only).
///
/// Each call performs a local xc_func_init / xc_func_end pair so the
/// oracle is thread-safe and stateless. This is test scaffolding, not
/// a hot path, so the per-call init cost is irrelevant.

#include "xc_libxc_oracle.h"

#include <xc.h>
#include <xc_funcs.h> // XC_LDA_X, XC_LDA_C_VWN, XC_GGA_X_PBE, XC_GGA_C_PBE,
                      // XC_GGA_X_B88, XC_GGA_C_LYP, XC_LDA_C_VWN_RPA

#include <cstring>
#include <initializer_list>

namespace ATV {
namespace test {
namespace xc_oracle {

namespace {

constexpr double kRhoMin = 1e-12; ///< Match native evaluate_xc cutoff.

/// @brief Run an XC functional pair (X+C) in spin-polarized LSDA mode,
///        sum into the output struct.
void run_lsda_pair(int func_x_id, int func_c_id, double rho_a, double rho_b,
                   ::ATV::cpu::scf::XCOutput& out) {
    const double rho[2] = {rho_a, rho_b};
    double exc[1] = {0.0};
    double vrho[2] = {0.0, 0.0};

    for (int fid : {func_x_id, func_c_id}) {
        xc_func_type func;
        xc_func_init(&func, fid, XC_POLARIZED);
        double exc_i[1] = {0.0};
        double vrho_i[2] = {0.0, 0.0};
        xc_lda_exc_vxc(&func, /*np=*/1, rho, exc_i, vrho_i);
        exc[0] += exc_i[0];
        vrho[0] += vrho_i[0];
        vrho[1] += vrho_i[1];
        xc_func_end(&func);
    }

    out.exc = exc[0];
    out.vrho_alpha = vrho[0];
    out.vrho_beta = vrho[1];
    out.vsigma_aa = 0.0;
    out.vsigma_ab = 0.0;
    out.vsigma_bb = 0.0;
}

/// @brief Run an XC functional pair (X+C) in spin-polarized GGA mode,
///        sum into the output struct.
void run_gga_pair(int func_x_id, int func_c_id, double rho_a, double rho_b,
                  double sigma_aa, double sigma_ab, double sigma_bb,
                  ::ATV::cpu::scf::XCOutput& out) {
    const double rho[2] = {rho_a, rho_b};
    const double sigma[3] = {sigma_aa, sigma_ab, sigma_bb};
    double exc[1] = {0.0};
    double vrho[2] = {0.0, 0.0};
    double vsigma[3] = {0.0, 0.0, 0.0};

    for (int fid : {func_x_id, func_c_id}) {
        xc_func_type func;
        xc_func_init(&func, fid, XC_POLARIZED);
        double exc_i[1] = {0.0};
        double vrho_i[2] = {0.0, 0.0};
        double vsigma_i[3] = {0.0, 0.0, 0.0};
        xc_gga_exc_vxc(&func, /*np=*/1, rho, sigma, exc_i, vrho_i, vsigma_i);
        exc[0] += exc_i[0];
        vrho[0] += vrho_i[0];
        vrho[1] += vrho_i[1];
        vsigma[0] += vsigma_i[0];
        vsigma[1] += vsigma_i[1];
        vsigma[2] += vsigma_i[2];
        xc_func_end(&func);
    }

    out.exc = exc[0];
    out.vrho_alpha = vrho[0];
    out.vrho_beta = vrho[1];
    out.vsigma_aa = vsigma[0];
    out.vsigma_ab = vsigma[1];
    out.vsigma_bb = vsigma[2];
}

::ATV::cpu::scf::XCOutput zero_output() {
    ::ATV::cpu::scf::XCOutput out{};
    std::memset(&out, 0, sizeof(out));
    return out;
}

/// @brief Run a single GGA functional in spin-polarized mode.
void run_gga_single(int func_id, double rho_a, double rho_b, double sigma_aa,
                    double sigma_ab, double sigma_bb,
                    ::ATV::cpu::scf::XCOutput& out) {
    const double rho[2] = {rho_a, rho_b};
    const double sigma[3] = {sigma_aa, sigma_ab, sigma_bb};
    double exc[1] = {0.0};
    double vrho[2] = {0.0, 0.0};
    double vsigma[3] = {0.0, 0.0, 0.0};

    xc_func_type func;
    xc_func_init(&func, func_id, XC_POLARIZED);
    xc_gga_exc_vxc(&func, /*np=*/1, rho, sigma, exc, vrho, vsigma);
    xc_func_end(&func);

    out.exc = exc[0];
    out.vrho_alpha = vrho[0];
    out.vrho_beta = vrho[1];
    out.vsigma_aa = vsigma[0];
    out.vsigma_ab = vsigma[1];
    out.vsigma_bb = vsigma[2];
}

/// @brief Run a single LDA functional in spin-polarized mode.
void run_lda_single(int func_id, double rho_a, double rho_b,
                    ::ATV::cpu::scf::XCOutput& out) {
    const double rho[2] = {rho_a, rho_b};
    double exc[1] = {0.0};
    double vrho[2] = {0.0, 0.0};

    xc_func_type func;
    xc_func_init(&func, func_id, XC_POLARIZED);
    xc_lda_exc_vxc(&func, /*np=*/1, rho, exc, vrho);
    xc_func_end(&func);

    out.exc = exc[0];
    out.vrho_alpha = vrho[0];
    out.vrho_beta = vrho[1];
    out.vsigma_aa = 0.0;
    out.vsigma_ab = 0.0;
    out.vsigma_bb = 0.0;
}

} // namespace

::ATV::cpu::scf::XCOutput evaluate_lsda(double rho_a, double rho_b) {
    if (rho_a + rho_b < kRhoMin)
        return zero_output();
    ::ATV::cpu::scf::XCOutput out = zero_output();
    run_lsda_pair(XC_LDA_X, XC_LDA_C_VWN, rho_a, rho_b, out);
    return out;
}

::ATV::cpu::scf::XCOutput evaluate_pbe(double rho_a, double rho_b,
                                       double sigma_aa, double sigma_ab,
                                       double sigma_bb) {
    if (rho_a + rho_b < kRhoMin)
        return zero_output();
    ::ATV::cpu::scf::XCOutput out = zero_output();
    run_gga_pair(XC_GGA_X_PBE, XC_GGA_C_PBE, rho_a, rho_b, sigma_aa, sigma_ab,
                 sigma_bb, out);
    return out;
}

::ATV::cpu::scf::XCOutput evaluate_gga_x_b88(double rho_a, double rho_b,
                                             double sigma_aa, double sigma_ab,
                                             double sigma_bb) {
    if (rho_a + rho_b < kRhoMin)
        return zero_output();
    ::ATV::cpu::scf::XCOutput out = zero_output();
    run_gga_single(XC_GGA_X_B88, rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                   out);
    return out;
}

::ATV::cpu::scf::XCOutput evaluate_gga_c_lyp(double rho_a, double rho_b,
                                             double sigma_aa, double sigma_ab,
                                             double sigma_bb) {
    if (rho_a + rho_b < kRhoMin)
        return zero_output();
    ::ATV::cpu::scf::XCOutput out = zero_output();
    run_gga_single(XC_GGA_C_LYP, rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb,
                   out);
    return out;
}

::ATV::cpu::scf::XCOutput evaluate_lda_c_vwn_rpa(double rho_a, double rho_b) {
    if (rho_a + rho_b < kRhoMin)
        return zero_output();
    ::ATV::cpu::scf::XCOutput out = zero_output();
    run_lda_single(XC_LDA_C_VWN_RPA, rho_a, rho_b, out);
    return out;
}

::ATV::cpu::scf::XCOutput evaluate_lda_x(double rho_a, double rho_b) {
    if (rho_a + rho_b < kRhoMin)
        return zero_output();
    const double rho[2] = {rho_a, rho_b};
    double exc[1] = {0.0};
    double vrho[2] = {0.0, 0.0};

    xc_func_type func;
    xc_func_init(&func, XC_LDA_X, XC_POLARIZED);
    xc_lda_exc_vxc(&func, 1, rho, exc, vrho);
    xc_func_end(&func);

    ::ATV::cpu::scf::XCOutput out = zero_output();
    out.exc = exc[0];
    out.vrho_alpha = vrho[0];
    out.vrho_beta = vrho[1];
    return out;
}

} // namespace xc_oracle
} // namespace test
} // namespace ATV
