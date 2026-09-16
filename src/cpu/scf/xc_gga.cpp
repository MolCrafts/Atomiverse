/// @file cpu/scf/xc_gga.cpp
/// @brief Becke-88 GGA exchange and Lee-Yang-Parr GGA correlation.
///
/// These two semi-local functionals are the GGA building blocks of the
/// B3LYP / BLYP hybrid family. They are kept in this translation unit
/// (separate from `xc_functional.cpp`) to honour the 800-line source-file
/// cap; shared declarations live in `xc_internal.h`.
///
/// All formulas use atomic units (Hartree energy, Bohr length). Each
/// functional is spin-polarized and closed-shell-correct as the
/// `rho_alpha = rho_beta` limit of its spin-polarized expression.
///
/// References:
///   - B88 exchange: A. D. Becke, Phys. Rev. A 38, 3098 (1988).
///   - LYP correlation: C. Lee, W. Yang, R. G. Parr, Phys. Rev. B 37,
///     785 (1988); B. Miehlich, A. Savin, H. Stoll, H. Preuss,
///     Chem. Phys. Lett. 157, 200 (1989).
///
/// libxc cross-reference: `XC_GGA_X_B88` (id 106), `XC_GGA_C_LYP`
/// (id 131). The LYP potential kernel is a faithful port of libxc's
/// maple2c-generated `func_vxc_pol` for `gga_c_lyp`, guaranteeing
/// 1e-12 channel-wise parity; the B88 kernel uses the closed analytic
/// derivatives of the readable Becke enhancement factor.

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/xc_internal.h"

#include <cmath>

namespace ATV {
namespace cpu {
namespace scf {

// ── Constants ────────────────────────────────────────────────────────────

static constexpr double kRhoMin = 1e-12; // total density cutoff

// Dirac exchange constant in libxc's convention:
//   X_FACTOR_C = (3/8) (3/pi)^{1/3} 4^{2/3}
// This equals 2^{1/3} * (3/4)(3/pi)^{1/3}, i.e. the spin-scaled Slater
// constant used by slater_exchange / pbe_exchange_spin.
static const double kXFactorC =
    (3.0 / 8.0) * std::cbrt(3.0 / units::pi) * std::cbrt(16.0);

// libxc threshold below which a spin channel is treated as empty.
static constexpr double kZetaThreshold = 2.2204460492503131e-16; // DBL_EPSILON

// ── B88 exchange ─────────────────────────────────────────────────────────

// Becke-1988 parameters (libxc gga_x_b88 defaults).
static constexpr double kB88_beta = 0.0042;
static constexpr double kB88_gamma = 6.0;

/// @brief Becke-1988 GGA exchange for one spin channel (libxc
///        XC_GGA_X_B88, id 106).
///
/// Energy density per volume:
///   ex_s = -X_FACTOR_C * rho_s^{4/3} * F(x),
/// reduced gradient x = |grad rho_s| / rho_s^{4/3}, Becke enhancement
///   F(x) = 1 + (beta/X_FACTOR_C) x^2 / (1 + gamma*beta*x*asinh(x)),
/// with beta = 0.0042, gamma = 6. An empty spin channel returns exactly
/// zero on all three channels, matching libxc's dens_threshold guard.
///
/// @param rho_s    Spin density for this channel [electrons/Bohr^3].
/// @param sigma_ss |grad rho_sigma|^2 for this channel [e^2/Bohr^8].
/// @param[out] ex_s      Exchange energy density per volume for this
///                       spin channel [Hartree/Bohr^3].
/// @param[out] vrho_s    d(ex_s)/drho_s, exchange potential [Hartree].
/// @param[out] vsigma_ss d(ex_s)/dsigma_ss [Hartree*Bohr^2].
void b88_exchange_spin(double rho_s, double sigma_ss, double& ex_s,
                       double& vrho_s, double& vsigma_ss, double beta,
                       double gamma) {
    // Empty spin channel: exact zero on every channel, matching libxc's
    // dens_threshold early-return for B88 exchange.
    if (rho_s < kRhoMin * 0.5) {
        ex_s = 0.0;
        vrho_s = 0.0;
        vsigma_ss = 0.0;
        return;
    }

    const double C = kXFactorC;
    const double k = beta / C; // = beta / X_FACTOR_C

    // Reduced gradient x = |grad rho_s| / rho_s^{4/3}.
    const double r43 = std::pow(rho_s, 4.0 / 3.0);
    const double r13 = std::cbrt(rho_s);
    const double g = std::sqrt(sigma_ss > 0.0 ? sigma_ss : 0.0);
    const double x = g / r43;

    // Becke enhancement factor F(x) = 1 + k x^2 / D,
    //   D = 1 + gamma*beta*x*asinh(x).
    const double asinh_x = std::asinh(x);
    const double D = 1.0 + gamma * beta * x * asinh_x;
    const double F = 1.0 + k * x * x / D;

    // D'(x) = gamma*beta*(asinh(x) + x/sqrt(1+x^2)).
    const double dD = gamma * beta * (asinh_x + x / std::sqrt(1.0 + x * x));

    // F'(x)/x = k (2D - x D') / D^2  -- finite as x -> 0 (-> 2k).
    const double dF_over_x = k * (2.0 * D - x * dD) / (D * D);
    // x F'(x) = k x^2 (2D - x D') / D^2.
    const double x_dF = dF_over_x * x * x;

    // Energy density per volume for this spin channel:
    //   ex_s = -X_FACTOR_C * rho_s^{4/3} * F(x).
    ex_s = -C * r43 * F;

    // vrho_s = d(ex_s)/drho_s
    //        = -C rho_s^{1/3} [ (4/3) F - (4/3) x F'(x) ].
    vrho_s = -C * r13 * (4.0 / 3.0) * (F - x_dF);

    // vsigma_ss = d(ex_s)/dsigma_ss = -C (F'(x)/x) / (2 rho_s^{4/3}).
    vsigma_ss = -C * dF_over_x / (2.0 * r43);
    (void)kB88_beta;
    (void)kB88_gamma;
}

// ── LYP correlation ──────────────────────────────────────────────────────

// Lee-Yang-Parr empirical parameters (libxc gga_c_lyp defaults).
static constexpr double kLYP_a = 0.04918;
static constexpr double kLYP_b = 0.132;
static constexpr double kLYP_c = 0.2533;
static constexpr double kLYP_d = 0.349;

namespace {

// libxc maple2c my_piecewise3(cond, a, b) -> cond ? a : b
inline double pw3(bool cond, double a, double b) { return cond ? a : b; }

} // namespace

/// @brief Lee-Yang-Parr GGA correlation (libxc XC_GGA_C_LYP, id 131),
///        spin-polarized.
///
/// Faithful port of libxc's maple2c-generated `func_vxc_pol` kernel for
/// `gga_c_lyp`, in the Miehlich-Savin-Stoll-Preuss closed form with
/// empirical constants a = 0.04918, b = 0.132, c = 0.2533, d = 0.349.
/// The intermediate variables `t1..t377` mirror the libxc kernel so the
/// port is line-by-line auditable; this guarantees 1e-12 channel-wise
/// parity with libxc id 131.
///
/// @param rho_a    Alpha spin density [electrons/Bohr^3].
/// @param rho_b    Beta spin density [electrons/Bohr^3].
/// @param sigma_aa |grad rho_alpha|^2 [e^2/Bohr^8].
/// @param sigma_ab grad rho_alpha . grad rho_beta [e^2/Bohr^8].
/// @param sigma_bb |grad rho_beta|^2 [e^2/Bohr^8].
/// @param[out] ec        Correlation energy density per electron
///                       [Hartree].
/// @param[out] vc_a      d(rho*ec)/drho_alpha [Hartree].
/// @param[out] vc_b      d(rho*ec)/drho_beta [Hartree].
/// @param[out] vsigma_aa d(rho*ec)/dsigma_aa [Hartree*Bohr^2].
/// @param[out] vsigma_ab d(rho*ec)/dsigma_ab [Hartree*Bohr^2].
/// @param[out] vsigma_bb d(rho*ec)/dsigma_bb [Hartree*Bohr^2].
void lyp_correlation(double rho_a, double rho_b, double sigma_aa,
                     double sigma_ab, double sigma_bb, double& ec, double& vc_a,
                     double& vc_b, double& vsigma_aa, double& vsigma_ab,
                     double& vsigma_bb) {
    // Direct port of libxc's maple2c gga_c_lyp `func_vxc_pol` kernel.
    // Variable names t1..t377 mirror the generated code so the port is
    // line-by-line auditable against the libxc reference. `tzk0` is the
    // energy density per electron; `tvrho{0,1}` / `tvsigma{0,1,2}` are
    // d(rho*exc)/d{rho,sigma} in libxc's convention.
    const double a = kLYP_a, b = kLYP_b, c = kLYP_c, d = kLYP_d;
    const double zeta_threshold = kZetaThreshold;

    // Density cutoff: below it, libxc's work_gga returns zero.
    double rho0 = rho_a > 0.0 ? rho_a : 0.0;
    double rho1 = rho_b > 0.0 ? rho_b : 0.0;
    double sigma0 = sigma_aa;
    double sigma1 = sigma_ab;
    double sigma2 = sigma_bb;
    if (rho0 + rho1 < kRhoMin) {
        ec = 0.0;
        vc_a = 0.0;
        vc_b = 0.0;
        vsigma_aa = 0.0;
        vsigma_ab = 0.0;
        vsigma_bb = 0.0;
        return;
    }
    // Empty spin channel: the maple2c intermediates contain 1/rho_s^{8/3}
    // factors (t70/t77). With rho_s = 0 and sigma_ss = 0 that is 0·∞ → NaN
    // and poisons UKS (n_β=0). Floor the empty channel dens_threshold-style
    // and zero its sigma (libxc dens_threshold guard). Occupied-channel
    // numbers at finite density are unchanged (floor ≪ dens_threshold).
    constexpr double kSpinFloor = 1e-14;
    if (rho0 < kSpinFloor) {
        rho0 = kSpinFloor;
        sigma0 = 0.0;
        sigma1 = 0.0;
    }
    if (rho1 < kSpinFloor) {
        rho1 = kSpinFloor;
        sigma2 = 0.0;
        sigma1 = 0.0;
    }

    auto POW_1_3 = [](double v) { return std::cbrt(v); };
    const double M_CBRT3 = 1.442249570307408382321638310780109588392;
    const double M_CBRT2 = 1.259921049894873164767210607278228350570;

    // ── energy density tzk0 ──
    const double t1 = rho0 - rho1;
    const double t2 = t1 * t1;
    const double t3 = rho0 + rho1;
    const double t4 = t3 * t3;
    const double t5 = 1.0 / t4;
    const double t7 = -t2 * t5 + 1.0;
    const double t8 = POW_1_3(t3);
    const double t9 = 1.0 / t8;
    const double t11 = d * t9 + 1.0;
    const double t12 = 1.0 / t11;
    const double t15 = std::exp(-c * t9);
    const double t16 = b * t15;
    const double t18 = sigma0 + 2.0 * sigma1 + sigma2;
    const double t19 = t8 * t8;
    const double t21 = 1.0 / t19 / t4;
    const double t22 = t18 * t21;
    const double t24 = d * t12 + c;
    const double t25 = t24 * t9;
    const double t27 = 47.0 - 7.0 * t25;
    const double t30 = t7 * t27 / 72.0 - 2.0 / 3.0;
    const double t32 = M_CBRT3;
    const double t33 = t32 * t32;
    const double t34 = units::pi * units::pi;
    const double t35 = POW_1_3(t34);
    const double t36 = t35 * t35;
    const double t37 = t33 * t36;
    const double t38 = 1.0 / t3;
    const double t39 = t1 * t38;
    const double t40 = 1.0 + t39;
    const bool t41 = t40 <= zeta_threshold;
    const double t42 = zeta_threshold * zeta_threshold;
    const double t43 = POW_1_3(zeta_threshold);
    const double t44 = t43 * t43;
    const double t45 = t44 * t42;
    const double t46 = t40 * t40;
    const double t47 = POW_1_3(t40);
    const double t48 = t47 * t47;
    const double t49 = t48 * t46;
    const double t50 = pw3(t41, t45, t49);
    const double t51 = 1.0 - t39;
    const bool t52 = t51 <= zeta_threshold;
    const double t53 = t51 * t51;
    const double t54 = POW_1_3(t51);
    const double t55 = t54 * t54;
    const double t56 = t55 * t53;
    const double t57 = pw3(t52, t45, t56);
    const double t58 = t50 + t57;
    const double t62 = M_CBRT2;
    const double t63 = t62 * t7;
    const double t65 = 5.0 / 2.0 - t25 / 18.0;
    const double t66 = rho0 * rho0;
    const double t67 = POW_1_3(rho0);
    const double t68 = t67 * t67;
    const double t70 = 1.0 / t68 / t66;
    const double t71 = sigma0 * t70;
    const double t72 = t71 * t50;
    const double t73 = rho1 * rho1;
    const double t74 = POW_1_3(rho1);
    const double t75 = t74 * t74;
    const double t77 = 1.0 / t75 / t73;
    const double t78 = sigma2 * t77;
    const double t79 = t78 * t57;
    const double t80 = t72 + t79;
    const double t81 = t65 * t80;
    const double t84 = t25 - 11.0;
    const double t86 = t44 * t42 * zeta_threshold;
    const double t89 = pw3(t41, t86, t48 * t46 * t40);
    const double t93 = pw3(t52, t86, t55 * t53 * t51);
    const double t95 = t71 * t89 + t78 * t93;
    const double t96 = t84 * t95;
    const double t101 = pw3(t41, t42, t46);
    const double t102 = t101 * sigma2;
    const double t103 = t77 * t57;
    const double t106 = pw3(t52, t42, t53);
    const double t107 = t106 * sigma0;
    const double t108 = t70 * t50;
    const double t114 = -t22 * t30 - 3.0 / 20.0 * t37 * t7 * t58
                        + t63 * t81 / 32.0 + t63 * t96 / 576.0
                        - t62
                              * (2.0 / 3.0 * t72 + 2.0 / 3.0 * t79
                                 - t102 * t103 / 4.0 - t107 * t108 / 4.0)
                              / 8.0;
    const double tzk0 = a * (t16 * t12 * t114 - t7 * t12);

    // ── vrho[0] = tvrho0 ──
    const double t118 = t3 * a;
    const double t119 = t1 * t5;
    const double t120 = t4 * t3;
    const double t121 = 1.0 / t120;
    const double t122 = t2 * t121;
    const double t124 = -2.0 * t119 + 2.0 * t122;
    const double t126 = t11 * t11;
    const double t127 = 1.0 / t126;
    const double t128 = t7 * t127;
    const double t130 = 1.0 / t8 / t3;
    const double t131 = d * t130;
    const double t133 = t128 * t131 / 3.0;
    const double t134 = b * c;
    const double t135 = t134 * t130;
    const double t136 = t15 * t12;
    const double t137 = t136 * t114;
    const double t139 = t135 * t137 / 3.0;
    const double t140 = t16 * t127;
    const double t141 = t114 * d;
    const double t144 = t140 * t141 * t130 / 3.0;
    const double t146 = 1.0 / t19 / t120;
    const double t147 = t18 * t146;
    const double t149 = 8.0 / 3.0 * t147 * t30;
    const double t151 = d * d;
    const double t152 = t151 * t127;
    const double t154 = 1.0 / t19 / t3;
    const double t157 = t24 * t130 - t152 * t154;
    const double t158 = 7.0 / 3.0 * t157;
    const double t159 = t7 * t158;
    const double t161 = t124 * t27 / 72.0 + t159 / 72.0;
    const double t166 = t48 * t40;
    const double t167 = t38 - t119;
    const double t168 = t166 * t167;
    const double t170 = pw3(t41, 0.0, 8.0 / 3.0 * t168);
    const double t171 = t55 * t51;
    const double t172 = -t167;
    const double t173 = t171 * t172;
    const double t175 = pw3(t52, 0.0, 8.0 / 3.0 * t173);
    const double t176 = t170 + t175;
    const double t180 = t62 * t124;
    const double t183 = t157 / 54.0;
    const double t184 = t183 * t80;
    const double t186 = t63 * t184 / 32.0;
    const double t189 = 1.0 / t68 / t66 / rho0;
    const double t190 = sigma0 * t189;
    const double t191 = t190 * t50;
    const double t193 = t71 * t170;
    const double t194 = t78 * t175;
    const double t195 = -8.0 / 3.0 * t191 + t193 + t194;
    const double t196 = t65 * t195;
    const double t202 = -t157 / 3.0;
    const double t203 = t202 * t95;
    const double t205 = t63 * t203 / 576.0;
    const double t210 = pw3(t41, 0.0, 11.0 / 3.0 * t49 * t167);
    const double t214 = pw3(t52, 0.0, 11.0 / 3.0 * t56 * t172);
    const double t216 = -8.0 / 3.0 * t190 * t89 + t71 * t210 + t78 * t214;
    const double t217 = t84 * t216;
    const double t225 = pw3(t41, 0.0, 2.0 * t40 * t167);
    const double t226 = t225 * sigma2;
    const double t229 = t77 * t175;
    const double t234 = pw3(t52, 0.0, 2.0 * t51 * t172);
    const double t235 = t234 * sigma0;
    const double t238 = t189 * t50;
    const double t241 = t70 * t170;
    const double t247 =
        t149 - t22 * t161 - 3.0 / 20.0 * t37 * t124 * t58
        - 3.0 / 20.0 * t37 * t7 * t176 + t180 * t81 / 32.0 + t186
        + t63 * t196 / 32.0 + t180 * t96 / 576.0 + t205 + t63 * t217 / 576.0
        - t62
              * (-16.0 / 9.0 * t191 + 2.0 / 3.0 * t193 + 2.0 / 3.0 * t194
                 - t226 * t103 / 4.0 - t102 * t229 / 4.0 - t235 * t108 / 4.0
                 + 2.0 / 3.0 * t107 * t238 - t107 * t241 / 4.0)
              / 8.0;
    const double t250 = t16 * t12 * t247 - t124 * t12 - t133 + t139 + t144;
    const double tvrho0 = t118 * t250 + tzk0;

    // ── vrho[1] = tvrho1 ──
    const double t253 = 2.0 * t119 + 2.0 * t122;
    const double t257 = t253 * t27 / 72.0 + t159 / 72.0;
    const double t262 = -t38 - t119;
    const double t263 = t166 * t262;
    const double t265 = pw3(t41, 0.0, 8.0 / 3.0 * t263);
    const double t266 = -t262;
    const double t267 = t171 * t266;
    const double t269 = pw3(t52, 0.0, 8.0 / 3.0 * t267);
    const double t270 = t265 + t269;
    const double t274 = t62 * t253;
    const double t277 = t71 * t265;
    const double t280 = 1.0 / t75 / t73 / rho1;
    const double t281 = sigma2 * t280;
    const double t282 = t281 * t57;
    const double t284 = t78 * t269;
    const double t285 = t277 - 8.0 / 3.0 * t282 + t284;
    const double t286 = t65 * t285;
    const double t293 = pw3(t41, 0.0, 11.0 / 3.0 * t49 * t262);
    const double t299 = pw3(t52, 0.0, 11.0 / 3.0 * t56 * t266);
    const double t301 = t71 * t293 - 8.0 / 3.0 * t281 * t93 + t78 * t299;
    const double t302 = t84 * t301;
    const double t310 = pw3(t41, 0.0, 2.0 * t40 * t262);
    const double t311 = t310 * sigma2;
    const double t314 = t280 * t57;
    const double t317 = t77 * t269;
    const double t322 = pw3(t52, 0.0, 2.0 * t51 * t266);
    const double t323 = t322 * sigma0;
    const double t326 = t70 * t265;
    const double t332 =
        t149 - t22 * t257 - 3.0 / 20.0 * t37 * t253 * t58
        - 3.0 / 20.0 * t37 * t7 * t270 + t274 * t81 / 32.0 + t186
        + t63 * t286 / 32.0 + t274 * t96 / 576.0 + t205 + t63 * t302 / 576.0
        - t62
              * (2.0 / 3.0 * t277 - 16.0 / 9.0 * t282 + 2.0 / 3.0 * t284
                 - t311 * t103 / 4.0 + 2.0 / 3.0 * t102 * t314
                 - t102 * t317 / 4.0 - t323 * t108 / 4.0 - t107 * t326 / 4.0)
              / 8.0;
    const double t335 = t16 * t12 * t332 - t253 * t12 - t133 + t139 + t144;
    const double tvrho1 = t118 * t335 + tzk0;

    // ── vsigma[0] = tvsigma0 ──
    const double t337 = t118 * b;
    const double t338 = t21 * t30;
    const double t339 = t65 * t70;
    const double t340 = t339 * t50;
    const double t343 = t84 * t70;
    const double t344 = t343 * t89;
    const double t348 = t106 * t70;
    const double t354 = -t338 + t63 * t340 / 32.0 + t63 * t344 / 576.0
                        - t62 * (2.0 / 3.0 * t108 - t348 * t50 / 4.0) / 8.0;
    const double t355 = t136 * t354;
    const double tvsigma0 = t337 * t355;

    // ── vsigma[1] = tvsigma1 ──
    const double t356 = t154 * a;
    const double t357 = t356 * b;
    const double t358 = t136 * t30;
    const double tvsigma1 = -2.0 * t357 * t358;

    // ── vsigma[2] = tvsigma2 ──
    const double t361 = t65 * t77;
    const double t362 = t361 * t57;
    const double t365 = t84 * t77;
    const double t366 = t365 * t93;
    const double t370 = t101 * t77;
    const double t376 = -t338 + t63 * t362 / 32.0 + t63 * t366 / 576.0
                        - t62 * (2.0 / 3.0 * t103 - t370 * t57 / 4.0) / 8.0;
    const double t377 = t136 * t376;
    const double tvsigma2 = t337 * t377;

    // Empty spin channels are screened out exactly as libxc's work_gga
    // does: the maple2c pol kernel handles the (1-zeta)^{-n} terms of an
    // empty channel through its zeta-threshold piecewise guards (pw3),
    // so no extra per-spin density floor is needed beyond the total
    // density cutoff applied above.
    ec = tzk0;
    vc_a = tvrho0;
    vc_b = tvrho1;
    vsigma_aa = tvsigma0;
    vsigma_ab = tvsigma1;
    vsigma_bb = tvsigma2;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
