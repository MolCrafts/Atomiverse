/// @file cuda/planewave/internal/xc_pbe.cu
/// @brief Device port of the closed-shell PBE path used by
///        `cpu::pw::internal::XCBuilder::step(ws, XCType::PBE)` — PBE
///        exchange + PW92-based PBE correlation. Constants and formulas are
///        transcribed verbatim from src/cpu/scf/xc_functional.cpp, then
///        collapsed at ζ = 0 exactly (see below).
///
/// NUMERICAL PARITY CONTRACT (device vs CPU, per grid point):
///   * The algebra is identical term for term and in the same association.
///     Every ζ = 0 simplification drops only terms the CPU evaluates to an
///     EXACT IEEE zero (f(0) = f'(0) = 0, ζ⁴ = 0, dφ/dρ_σ = 0, φ = 1 from
///     pow(1, y) = 1), so no reordering error is introduced.
///   * `pow` / `exp` / `log` / `sqrt` come from libdevice here and libm on
///     the host; each is ~1 ulp but not bit-identical, and PBE chains more of
///     them than LSDA (≈ 10 per point). Agreement is ~1e-14 relative per
///     point. The registered unit test asserts 1e-12·max(1,|ref|), the same
///     bound the LSDA kernel carries.
///   * `nvcc` FMA contraction may fuse products the host keeps separate —
///     absorbed by the same bound (this file introduces no reduction; E_xc
///     reuses `xc_energy` from xc_lda.cu unchanged).
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/planewave/internal/xc_pbe.cuh"

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

namespace {

constexpr int kBlock = ::ATV::cuda::kDefaultBlock; // 256

// ── Constants (verbatim from xc_functional.cpp) ──────────────────────────
__device__ constexpr double kPi = 3.14159265358979323846;
__device__ constexpr double kRhoMin = 1e-12;      // total density cutoff
__device__ constexpr double kSpinDensMin = 1e-15; // per-spin density floor
// Slater exchange constant C_x = (3/4)(3/π)^{1/3}.
__device__ constexpr double kCx = 0.7385587663820224;
// PBE exchange: F(s) = 1 + κ − κ/(1 + μ s²/κ).
__device__ constexpr double kKappa = 0.804;
__device__ constexpr double kMu = 0.2195149727645171; // beta*pi^2/3
// PBE correlation: H = γ φ³ ln{1 + (β/γ) t² (1+At²)/(1+At²+A²t⁴)}.
__device__ constexpr double kBeta = 0.06672455060314922;
__device__ constexpr double kGamma = 0.031090690869654895034; // (1 − ln2)/π²
// PW92 paramagnetic parameter row (Table I, ζ = 0) — the only channel that
// survives at ζ = 0 (the ferromagnetic and spin-stiffness terms carry exact
// factors of f(0) = 0 or ζ⁴ = 0 on the CPU).
__device__ constexpr double kPwA = 0.0310907;
__device__ constexpr double kPwAlpha1 = 0.21370;
__device__ constexpr double kPwBeta1 = 7.5957;
__device__ constexpr double kPwBeta2 = 3.5876;
__device__ constexpr double kPwBeta3 = 1.6382;
__device__ constexpr double kPwBeta4 = 0.49294;
__device__ constexpr double kPwP = 1.0;

/// PW92 G(r_s) and dG/dr_s for the paramagnetic parameter set — verbatim
/// `pw92_G(rs, kPW92_P, …)`.
__device__ __forceinline__ void pw92_G_paramagnetic(double rs, double& G,
                                                    double& dG_drs) {
    const double sqrt_rs = sqrt(rs);
    const double rs_3_2 = rs * sqrt_rs;
    const double rs_p1 = pow(rs, kPwP + 1.0);
    const double rs_p = pow(rs, kPwP);

    const double q1 = 2.0 * kPwA
                      * (kPwBeta1 * sqrt_rs + kPwBeta2 * rs + kPwBeta3 * rs_3_2
                         + kPwBeta4 * rs_p1);
    const double dq1_drs =
        2.0 * kPwA
        * (0.5 * kPwBeta1 / sqrt_rs + kPwBeta2 + 1.5 * kPwBeta3 * sqrt_rs
           + (kPwP + 1.0) * kPwBeta4 * rs_p);

    const double q2 = log(1.0 + 1.0 / q1);
    const double dq2_drs = -dq1_drs / (q1 * (q1 + 1.0));

    const double prefactor = 1.0 + kPwAlpha1 * rs;
    G = -2.0 * kPwA * prefactor * q2;
    dG_drs = -2.0 * kPwA * (kPwAlpha1 * q2 + prefactor * dq2_drs);
}

/// PW92 correlation at ζ = 0, matching `pw92_correlation(rho/2, rho/2, …)`.
/// The CPU's spin terms multiply f(0) = 0 / ζ⁴ = 0 exactly, and its
/// `−(ζ ∓ 1)·dε_c/dζ` adds an exact ±0, so ε_c and v_c collapse to the
/// paramagnetic channel: ε_c = G_P(r_s), v_c = ε_c − (r_s/3) G_P'(r_s).
__device__ __forceinline__ void
pw92_correlation_zeta0(double rho_a, double rho_b, double& ec, double& vc_a) {
    // Per-spin density floor matching libxc work_lda clamping (CPU parity).
    rho_a = fmax(rho_a, kSpinDensMin);
    rho_b = fmax(rho_b, kSpinDensMin);

    const double rho = rho_a + rho_b;
    const double rs = pow(3.0 / (4.0 * kPi * rho), 1.0 / 3.0);

    double ecP, decP_drs;
    pw92_G_paramagnetic(rs, ecP, decP_drs);
    ec = ecP;
    vc_a = ec - (rs / 3.0) * decP_drs;
}

/// PBE exchange for one spin channel — verbatim `pbe_exchange_spin(rho_s,
/// sigma_ss, …)` with the PBE μ/κ and the rational (non-RPBE) form.
__device__ __forceinline__ void pbe_exchange_spin(double rho_s, double sigma_ss,
                                                  double& ex_s, double& vrho_s,
                                                  double& vsigma_ss) {
    // Transcribed empty-spin early-return. Behind the caller's ρ < 1e-12
    // floor it is unreachable (ρ/2 < kRhoMin·0.5 ⇔ ρ < kRhoMin), matching
    // the CPU where the outer XCBuilder floor fires first.
    if (rho_s < kRhoMin * 0.5) {
        ex_s = 0.0;
        vrho_s = 0.0;
        vsigma_ss = 0.0;
        return;
    }

    const double Cx_spin = pow(2.0, 1.0 / 3.0) * kCx;

    const double rs43 = pow(rho_s, 4.0 / 3.0);
    const double ex_lda = -Cx_spin * rs43; // per volume

    const double kf = pow(6.0 * kPi * kPi, 1.0 / 3.0) * pow(rho_s, 1.0 / 3.0);
    const double grad = sqrt(fmax(sigma_ss, 0.0));
    const double s = grad / (2.0 * kf * rho_s);
    const double s2 = s * s;

    const double denom = 1.0 + kMu * s2 / kKappa;
    const double Fx = 1.0 + kKappa - kKappa / denom;
    const double dFx_ds2 = kMu / (denom * denom);

    ex_s = ex_lda * Fx;

    const double ds2_drho = -(8.0 / 3.0) * s2 / rho_s;
    const double kf2_fac =
        4.0 * pow(6.0 * kPi * kPi, 2.0 / 3.0) * pow(rho_s, 8.0 / 3.0);
    const double ds2_dsigma = 1.0 / kf2_fac;

    const double dex_lda_drho = -(4.0 / 3.0) * Cx_spin * pow(rho_s, 1.0 / 3.0);
    vrho_s = dex_lda_drho * Fx + ex_lda * dFx_ds2 * ds2_drho;
    vsigma_ss = ex_lda * dFx_ds2 * ds2_dsigma;
}

/// PBE H gradient correction at ζ = 0 — `pbe_correlation(...)` with φ = 1
/// exactly (φ(0) = ½(pow(1,2/3)+pow(1,2/3)) = 1 in IEEE) and the dφ/dρ_σ
/// terms dropped because the CPU computes them as an exact 0 at ρ_α = ρ_β
/// ((1/3)/ρ − (1/3)/ρ). Every surviving expression keeps the CPU's
/// association; γφ³ appears as plain γ because γ·1.0 is exact.
__device__ __forceinline__ void
pbe_correlation_zeta0(double rho, double ec_lda, double vc_lda,
                      double sigma_total, double& ec_pbe, double& vc_a,
                      double& vsigma_aa, double& vsigma_ab) {
    const double kF = pow(3.0 * kPi * kPi * rho, 1.0 / 3.0);
    const double ks = sqrt(4.0 * kF / kPi);

    const double grad_rho = sqrt(fmax(sigma_total, 0.0));
    const double t = grad_rho / (2.0 * ks * rho); // 2.0·φ·ks·ρ with φ = 1
    const double t2 = t * t;

    const double exp_arg = -ec_lda / kGamma; // γφ³ = γ
    const double exp_val = exp(exp_arg);
    const double A = (kBeta / kGamma) / (exp_val - 1.0);

    const double At2 = A * t2;
    const double numer = 1.0 + At2;
    const double denom = 1.0 + At2 + A * A * t2 * t2;
    const double ratio = (kBeta / kGamma) * t2 * numer / denom;
    const double H = kGamma * log(1.0 + ratio); // γφ³ = γ

    ec_pbe = ec_lda + H;

    const double u = At2;
    const double one_u = 1.0 + u;
    const double one_u_u2 = 1.0 + u + u * u;
    const double d_frac_du =
        (1.0 * one_u_u2 - one_u * (1.0 + 2.0 * u)) / (one_u_u2 * one_u_u2);

    const double dratio_dt2 =
        (kBeta / kGamma) * (one_u / one_u_u2 + t2 * A * d_frac_du);
    const double dH_dt2 = kGamma / (1.0 + ratio) * dratio_dt2; // γφ³ = γ

    const double dt2_dsigma =
        1.0 / (4.0 * ks * ks * rho * rho); // 4φ²ks²ρ² with φ = 1
    const double dH_dsigma_total = dH_dt2 * dt2_dsigma;

    vsigma_aa = rho * dH_dsigma_total;
    vsigma_ab = 2.0 * rho * dH_dsigma_total;

    const double dt2_drho = -(7.0 / 3.0) * t2 / rho;
    const double dA_dfactor = A * A * exp_val / (kBeta / kGamma);
    const double dA_dec_lda = dA_dfactor / kGamma; // γφ³ = γ
    const double dec_lda_drho_a = (vc_lda - ec_lda) / rho;

    const double d_frac_dA = -t2 * u * (2.0 + u) / (one_u_u2 * one_u_u2);
    const double dratio_dA = (kBeta / kGamma) * t2 * d_frac_dA;
    const double dH_dA = kGamma / (1.0 + ratio) * dratio_dA; // γφ³ = γ

    // dH_dphi_total · dφ/dρ_σ is an exact CPU zero at ζ = 0 — dropped.
    const double dH_drho_a =
        dH_dt2 * dt2_drho + dH_dA * (dA_dec_lda * dec_lda_drho_a);

    vc_a = vc_lda + H + rho * dH_drho_a;
}

/// One grid point: local v_ρ, ε_xc, and the GGA flux from ρ and ∇ρ.
/// Mirrors the `evaluate_xc(PBE, ρ/2, ρ/2, σ/4, σ/4, σ/4)` call plus the
/// `vsig_eff = vσ_aa + ½ vσ_ab; flux_d = vsig_eff·∂_dρ` lines of
/// XCBuilder::step's GGA grid loop. Overwrites the gradient with the flux.
__global__ void xc_pbe_kernel(const double* __restrict__ rho_r,
                              double* __restrict__ gx_r,
                              double* __restrict__ gy_r,
                              double* __restrict__ gz_r,
                              double* __restrict__ vxc_r,
                              double* __restrict__ exc_r, int nr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr)
        return;
    const double rho = rho_r[i];
    if (rho < kRhoMin) {
        vxc_r[i] = 0.0;
        exc_r[i] = 0.0;
        gx_r[i] = 0.0;
        gy_r[i] = 0.0;
        gz_r[i] = 0.0;
        return;
    }

    const double gx = gx_r[i];
    const double gy = gy_r[i];
    const double gz = gz_r[i];
    // Spin-unpolarized: σ_aa = σ_ab = σ_bb = |∇ρ|²/4, as XCBuilder forms it.
    const double sigma = gx * gx + gy * gy + gz * gz;
    const double sig4 = 0.25 * sigma;

    const double ra = rho / 2.0; // exact; ρ_α = ρ_β bitwise ⇒ ζ = 0 exactly
    // The CPU's σ guards are identities here: sa = max(σ/4, 0) = σ/4 and the
    // sab clamp to ±½(sa+sb) = ±σ/4 leaves sab = σ/4 untouched.

    // PBE-family exchange, both spin channels identical.
    double ex_s, vrho_x, vsig_x;
    pbe_exchange_spin(ra, sig4, ex_s, vrho_x, vsig_x);
    const double ex_pbe = (ex_s + ex_s) / rho; // (ex_a + ex_b)/ρ, per electron

    // PW92 LSDA correlation base (NOT VWN5 — see xc_functional.cpp).
    double ec_lda, vc_lda;
    pw92_correlation_zeta0(ra, ra, ec_lda, vc_lda);

    // PBE H gradient correction. σ_total = σ_aa + 2σ_ab + σ_bb, in the CPU's
    // association.
    const double sigma_total = sig4 + 2.0 * sig4 + sig4;
    double ec_pbe, vc_pbe, vsig_c_aa, vsig_c_ab;
    pbe_correlation_zeta0(rho, ec_lda, vc_lda, sigma_total, ec_pbe, vc_pbe,
                          vsig_c_aa, vsig_c_ab);

    exc_r[i] = ex_pbe + ec_pbe;
    vxc_r[i] = vrho_x + vc_pbe;

    // Closed-shell flux coefficient (XCBuilder: vσ_aa + ½ vσ_ab).
    const double vsigma_aa = vsig_x + vsig_c_aa;
    const double vsig_eff = vsigma_aa + 0.5 * vsig_c_ab;
    gx_r[i] = vsig_eff * gx;
    gy_r[i] = vsig_eff * gy;
    gz_r[i] = vsig_eff * gz;
}

} // namespace

void xc_pbe_grid(const double* d_rho_r, double* d_gx_r, double* d_gy_r,
                 double* d_gz_r, double* d_vxc_r, double* d_exc_r, int nr,
                 cudaStream_t stream) {
    if (nr <= 0)
        return;
    xc_pbe_kernel<<<::ATV::cuda::grid_size(nr, kBlock), kBlock, 0, stream>>>(
        d_rho_r, d_gx_r, d_gy_r, d_gz_r, d_vxc_r, d_exc_r, nr);
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
