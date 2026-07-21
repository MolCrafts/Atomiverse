/// @file tests/cpu/scf/oracle/xc_libxc_oracle.h
/// @brief libxc-backed XC oracle for test-only parity validation.
///
/// libxc is NOT an Atomiverse dependency. This oracle exists solely to
/// validate the native `evaluate_xc()` implementation in
/// `src/cpu/scf/xc_functional.cpp` against a reference library at the
/// primitive level (single-point exc/vrho/vsigma). After sub-spec 03's
/// primitive parity gate stays green, this directory is deletable
/// without touching libatomiverse.
///
/// Production code MUST NOT include this header.
#pragma once

#include "atomiverse/cpu/scf/xc_functional.h" // for ATV::cpu::scf::XCOutput

namespace ATV {
namespace test {
namespace xc_oracle {

/// @brief Evaluate LSDA (Slater + VWN5) via libxc.
/// @param rho_a Alpha density [electrons/Bohr^3].
/// @param rho_b Beta density [electrons/Bohr^3].
/// @return XCOutput populated using libxc's d(rho*exc)/d(rho_sigma) convention.
/// @note Functional IDs: XC_LDA_X + XC_LDA_C_VWN (id 7, VWN5 — not VWN3=5).
::ATV::cpu::scf::XCOutput evaluate_lsda(double rho_a, double rho_b);

/// @brief Evaluate PBE (PBE_X + PBE_C) via libxc.
/// @param rho_a Alpha density [electrons/Bohr^3].
/// @param rho_b Beta density [electrons/Bohr^3].
/// @param sigma_aa |grad rho_alpha|^2.
/// @param sigma_ab grad rho_alpha . grad rho_beta.
/// @param sigma_bb |grad rho_beta|^2.
/// @return XCOutput populated using libxc's d(rho*exc)/d(rho_sigma) convention.
::ATV::cpu::scf::XCOutput evaluate_pbe(double rho_a, double rho_b,
                                       double sigma_aa, double sigma_ab,
                                       double sigma_bb);

/// @brief Evaluate Slater LDA exchange ALONE via libxc XC_LDA_X.
/// @param rho_a Alpha density.
/// @param rho_b Beta density.
/// @return XCOutput with only exchange contribution (no correlation).
/// @note Diagnostic helper: textbook check |exc(0.5,0.5) − (−(3/4)(3/pi)^(1/3))| < 1e-10.
::ATV::cpu::scf::XCOutput evaluate_lda_x(double rho_a, double rho_b);

/// @brief Evaluate Becke-88 GGA exchange ALONE via libxc XC_GGA_X_B88 (id 106).
/// @param rho_a Alpha density [electrons/Bohr^3].
/// @param rho_b Beta density [electrons/Bohr^3].
/// @param sigma_aa |grad rho_alpha|^2 [electrons^2/Bohr^8].
/// @param sigma_ab grad rho_alpha . grad rho_beta [electrons^2/Bohr^8].
/// @param sigma_bb |grad rho_beta|^2 [electrons^2/Bohr^8].
/// @return XCOutput with B88 exchange exc/vrho/vsigma channels populated.
::ATV::cpu::scf::XCOutput evaluate_gga_x_b88(double rho_a, double rho_b,
                                             double sigma_aa, double sigma_ab,
                                             double sigma_bb);

/// @brief Evaluate Lee-Yang-Parr GGA correlation ALONE via libxc
///        XC_GGA_C_LYP (id 131).
/// @param rho_a Alpha density [electrons/Bohr^3].
/// @param rho_b Beta density [electrons/Bohr^3].
/// @param sigma_aa |grad rho_alpha|^2 [electrons^2/Bohr^8].
/// @param sigma_ab grad rho_alpha . grad rho_beta [electrons^2/Bohr^8].
/// @param sigma_bb |grad rho_beta|^2 [electrons^2/Bohr^8].
/// @return XCOutput with LYP correlation exc/vrho/vsigma channels populated.
::ATV::cpu::scf::XCOutput evaluate_gga_c_lyp(double rho_a, double rho_b,
                                             double sigma_aa, double sigma_ab,
                                             double sigma_bb);

/// @brief Evaluate the RPA-parametrized VWN LDA correlation ALONE via
///        libxc XC_LDA_C_VWN_RPA (id 8).
/// @param rho_a Alpha density [electrons/Bohr^3].
/// @param rho_b Beta density [electrons/Bohr^3].
/// @return XCOutput with VWN-RPA correlation exc/vrho channels populated.
/// @note libxc id 8 (VWN_RPA), the functional PySCF aliases as "VWN3";
///       distinct from libxc's own XC_LDA_C_VWN_3 (id 30).
::ATV::cpu::scf::XCOutput evaluate_lda_c_vwn_rpa(double rho_a, double rho_b);

} // namespace xc_oracle
} // namespace test
} // namespace ATV
