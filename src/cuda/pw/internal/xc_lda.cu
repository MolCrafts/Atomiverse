/// @file cuda/pw/internal/xc_lda.cu
/// @brief Device port of the LSDA XC path used by cpu::pw::internal::XCBuilder — Slater
///        exchange + PW92 correlation, the sigma=0 limit of evaluate_xc(PBE).
///        Constants and formulas are transcribed verbatim from
///        src/cpu/scf/xc_functional.cpp to match it to ~1e-12.
#include "atomiverse/cuda/pw/internal/xc_lda.cuh"

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

namespace {

// ── Constants (verbatim from xc_functional.cpp) ──────────────────────────
__device__ constexpr double kPi = 3.14159265358979323846;
__device__ constexpr double kRhoMin = 1e-12;      // total density cutoff
__device__ constexpr double kSpinDensMin = 1e-15; // per-spin density floor
// Slater exchange constant C_x = (3/4)(3/π)^{1/3}.
__device__ constexpr double kCx = 0.7385587663820224;

// f(ζ) = [(1+ζ)^{4/3} + (1-ζ)^{4/3} - 2] / [2(2^{1/3} - 1)]
__device__ __forceinline__ double spin_f(double zeta) {
    double denom = 2.0 * (pow(2.0, 1.0 / 3.0) - 1.0);
    return (pow(1.0 + zeta, 4.0 / 3.0) + pow(1.0 - zeta, 4.0 / 3.0) - 2.0)
           / denom;
}

// df/dζ
__device__ __forceinline__ double spin_df(double zeta) {
    double denom = 2.0 * (pow(2.0, 1.0 / 3.0) - 1.0);
    return (4.0 / 3.0)
           * (pow(1.0 + zeta, 1.0 / 3.0) - pow(1.0 - zeta, 1.0 / 3.0)) / denom;
}

// ── PW92 correlation G(r_s) for one parameter set ────────────────────────
__device__ __forceinline__ void pw92_G(double rs, double A, double alpha1,
                                       double beta1, double beta2, double beta3,
                                       double beta4, double pp, double& G,
                                       double& dG_drs) {
    double sqrt_rs = sqrt(rs);
    double rs_3_2 = rs * sqrt_rs;
    double rs_p1 = pow(rs, pp + 1.0);
    double rs_p = pow(rs, pp);

    double q1 =
        2.0 * A
        * (beta1 * sqrt_rs + beta2 * rs + beta3 * rs_3_2 + beta4 * rs_p1);
    double dq1_drs = 2.0 * A
                     * (0.5 * beta1 / sqrt_rs + beta2 + 1.5 * beta3 * sqrt_rs
                        + (pp + 1.0) * beta4 * rs_p);

    double q2 = log(1.0 + 1.0 / q1);
    double dq2_drs = -dq1_drs / (q1 * (q1 + 1.0));

    double prefactor = 1.0 + alpha1 * rs;
    G = -2.0 * A * prefactor * q2;
    dG_drs = -2.0 * A * (alpha1 * q2 + prefactor * dq2_drs);
}

// PW92 spin-polarized correlation (Perdew & Wang, PRB 45, 13244 (1992)).
__device__ __forceinline__ void pw92_correlation(double rho_a, double rho_b,
                                                 double& ec, double& vc_a,
                                                 double& vc_b) {
    rho_a = fmax(rho_a, kSpinDensMin);
    rho_b = fmax(rho_b, kSpinDensMin);
    double rho = rho_a + rho_b;
    double rs = pow(3.0 / (4.0 * kPi * rho), 1.0 / 3.0);
    double zeta = (rho_a - rho_b) / rho;

    double ecP, ecF, ac, decP_drs, decF_drs, dac_drs;
    // Paramagnetic (ζ=0), ferromagnetic (ζ=1), spin-stiffness (-α_c).
    pw92_G(rs, 0.0310907, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294, 1.0, ecP,
           decP_drs);
    pw92_G(rs, 0.01554535, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517, 1.0, ecF,
           decF_drs);
    pw92_G(rs, 0.0168869, 0.11125, 10.357, 3.6231, 0.88026, 0.49671, 1.0, ac,
           dac_drs);
    ac = -ac;
    dac_drs = -dac_drs;

    double kFppZero = 4.0 / (9.0 * (pow(2.0, 1.0 / 3.0) - 1.0));
    double fz = spin_f(zeta);
    double z4 = zeta * zeta * zeta * zeta;

    ec = ecP + ac * fz / kFppZero * (1.0 - z4) + (ecF - ecP) * fz * z4;

    double dec_drs = decP_drs + dac_drs * fz / kFppZero * (1.0 - z4)
                     + (decF_drs - decP_drs) * fz * z4;

    double dfz = spin_df(zeta);
    double dz4 = 4.0 * zeta * zeta * zeta;
    double dec_dz = ac / kFppZero * (dfz * (1.0 - z4) - fz * dz4)
                    + (ecF - ecP) * (dfz * z4 + fz * dz4);

    double rs3_dec = (rs / 3.0) * dec_drs;
    vc_a = ec - rs3_dec - (zeta - 1.0) * dec_dz;
    vc_b = ec - rs3_dec - (zeta + 1.0) * dec_dz;
}

// One grid point: V_xc and ε_xc from total density, matching internal::XCBuilder.
__global__ void xc_lda_kernel(const double* __restrict__ rho_r,
                              double* __restrict__ vxc_r,
                              double* __restrict__ exc_r, int nr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr)
        return;
    double rho = rho_r[i];
    if (rho < kRhoMin) {
        vxc_r[i] = 0.0;
        exc_r[i] = 0.0;
        return;
    }
    double ra = rho * 0.5;
    double rb = rho * 0.5;

    // Exchange: PBE at sigma=0 → Slater (per-spin, as the CPU dispatch forms it).
    double Cx_spin = pow(2.0, 1.0 / 3.0) * kCx;
    double ex_a = -Cx_spin * pow(ra, 4.0 / 3.0);
    double ex_b = -Cx_spin * pow(rb, 4.0 / 3.0);
    double ex = (ex_a + ex_b) / rho;
    double vx_a = -(4.0 / 3.0) * Cx_spin * pow(ra, 1.0 / 3.0);

    // Correlation: PW92 LSDA base (PBE gradient term vanishes at sigma=0).
    double ec, vc_a, vc_b;
    pw92_correlation(ra, rb, ec, vc_a, vc_b);

    exc_r[i] = ex + ec;
    vxc_r[i] = vx_a + vc_a;
}

// Block-reduce Σ ρ·exc, atomic-accumulate into *acc.
__global__ void xc_energy_kernel(const double* __restrict__ rho_r,
                                 const double* __restrict__ exc_r, int nr,
                                 double* __restrict__ acc) {
    __shared__ double s[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = (i < nr) ? rho_r[i] * exc_r[i] : 0.0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(acc, s[0]);
}

__global__ void scale_scalar_kernel(double* x, double factor) { *x *= factor; }

} // namespace

void xc_lda_grid(const double* d_rho_r, double* d_vxc_r, double* d_exc_r,
                 int nr, cudaStream_t stream) {
    if (nr <= 0)
        return;
    constexpr int kBlock = 256;
    const int grid = (nr + kBlock - 1) / kBlock;
    xc_lda_kernel<<<grid, kBlock, 0, stream>>>(d_rho_r, d_vxc_r, d_exc_r, nr);
}

void xc_energy(const double* d_rho_r, const double* d_exc_r, int nr, double dV,
               double* d_Exc, cudaStream_t stream) {
    cudaMemsetAsync(d_Exc, 0, sizeof(double), stream);
    if (nr <= 0)
        return;
    constexpr int kBlock = 256;
    const int grid = (nr + kBlock - 1) / kBlock;
    xc_energy_kernel<<<grid, kBlock, 0, stream>>>(d_rho_r, d_exc_r, nr, d_Exc);
    scale_scalar_kernel<<<1, 1, 0, stream>>>(d_Exc, dV);
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
