/// @file cuda/planewave/internal/xc_lda.cu
/// @brief Device port of the LSDA XC path used by
///        `cpu::pw::internal::XCBuilder::step(ws, XCType::LSDA)` — Slater
///        exchange + VWN5 correlation. Constants and formulas are transcribed
///        verbatim from src/cpu/scf/xc_functional.cpp.
///
/// NUMERICAL PARITY CONTRACT (device vs CPU, per grid point):
///   * The algebra is identical term for term and in the same association,
///     so no reordering error is introduced.
///   * `pow` / `log` / `atan2` / `sqrt` come from libdevice here and from
///     libm on the host. Both are accurate to ~1 ulp but are NOT bit-identical
///     implementations, so agreement is ~1e-15 relative per point, never
///     exact. The registered unit test asserts 1e-12·max(1,|ref|).
///   * The E_xc reduction is a fixed-geometry tree here and a serial loop on
///     the host: reproducible run-to-run on each side, ~1e-16 relative apart
///     from each other, and (being a tree) slightly MORE accurate than the
///     host's serial accumulation.
#include "atomiverse/cuda/device/launch.cuh"
#include "atomiverse/cuda/planewave/internal/xc_lda.cuh"

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {

namespace {

constexpr int kBlock = ::ATV::cuda::kDefaultBlock; // 256
/// Fixed stage-1 grid: the summation partition (and hence the reduction
/// tree) must not depend on `nr`, or E_xc stops being reproducible.
constexpr int kReduceBlocks = kXcEnergyPartials;
static_assert(kReduceBlocks == kBlock,
              "stage 2 reduces kReduceBlocks partials with kBlock threads");

// ── Constants (verbatim from xc_functional.cpp) ──────────────────────────
__device__ constexpr double kPi = 3.14159265358979323846;
__device__ constexpr double kRhoMin = 1e-12;      // total density cutoff
__device__ constexpr double kSpinDensMin = 1e-15; // per-spin density floor
// Slater exchange constant C_x = (3/4)(3/π)^{1/3}.
__device__ constexpr double kCx = 0.7385587663820224;

// VWN5 channels {A, x0, b, c}, Hartree convention (A_Ha = A_Ry/2). Held as
// scalars rather than a struct so no aggregate device global is needed.
// Paramagnetic (ζ = 0), ferromagnetic (ζ = 1), spin-stiffness α_c.
__device__ constexpr double kVwnP_A = 0.0310907;
__device__ constexpr double kVwnP_X0 = -0.10498;
__device__ constexpr double kVwnP_B = 3.72744;
__device__ constexpr double kVwnP_C = 12.9352;
__device__ constexpr double kVwnF_A = 0.01554535;
__device__ constexpr double kVwnF_X0 = -0.32500;
__device__ constexpr double kVwnF_B = 7.06042;
__device__ constexpr double kVwnF_C = 18.0578;
__device__ constexpr double kVwnA_A = -1.0 / (6.0 * kPi * kPi);
__device__ constexpr double kVwnA_X0 = -0.0047584;
__device__ constexpr double kVwnA_B = 1.13107;
__device__ constexpr double kVwnA_C = 13.0045;
// f''(0) = 4 / (9 (2^{1/3} − 1)) — spin-interpolation second derivative.
// Numeric value matches CPU `4.0 / (9.0 * (pow(2,1/3) - 1))` to double.
__device__ constexpr double kFppZero = 1.7099209341613654;

/// F(x) = A { ln(x²/X) + 2b/Q atan(Q/(2x+b))
///          − b x0/X(x0) [ ln((x−x0)²/X) + 2(b+2x0)/Q atan(Q/(2x+b)) ] }
/// with X(x) = x² + bx + c, Q = √(4c − b²).
__device__ __forceinline__ double vwn5_F(double x, double A, double x0,
                                         double b, double c) {
    const double X = x * x + b * x + c;
    const double X0 = x0 * x0 + b * x0 + c;
    const double Q = sqrt(4.0 * c - b * b);
    const double atanQ = atan2(Q, 2.0 * x + b);

    const double term1 = log(x * x / X);
    const double term2 = 2.0 * b / Q * atanQ;
    const double term3 =
        -b * x0 / X0
        * (log((x - x0) * (x - x0) / X) + 2.0 * (b + 2.0 * x0) / Q * atanQ);

    return A * (term1 + term2 + term3);
}

/// dF/dx. The identity (2x+b)² + Q² = 4X collapses both arctan derivatives,
/// so Q never appears — the same simplification the CPU function uses.
__device__ __forceinline__ double vwn5_dFdx(double x, double A, double x0,
                                            double b, double c) {
    const double X = x * x + b * x + c;
    const double X0 = x0 * x0 + b * x0 + c;

    const double dterm1 = 2.0 / x - (2.0 * x + b) / X;
    const double dterm2 = -4.0 * b / (4.0 * X);
    const double fac = -b * x0 / X0;
    const double dlog = 2.0 / (x - x0) - (2.0 * x + b) / X;
    const double datan = -4.0 * (b + 2.0 * x0) / (4.0 * X);
    const double dterm3 = fac * (dlog + datan);

    return A * (dterm1 + dterm2 + dterm3);
}

/// f(ζ) = [(1+ζ)^{4/3} + (1−ζ)^{4/3} − 2] / [2(2^{1/3} − 1)].
__device__ __forceinline__ double spin_f(double zeta) {
    const double denom = 2.0 * (pow(2.0, 1.0 / 3.0) - 1.0);
    return (pow(1.0 + zeta, 4.0 / 3.0) + pow(1.0 - zeta, 4.0 / 3.0) - 2.0)
           / denom;
}

__device__ __forceinline__ double spin_df(double zeta) {
    const double denom = 2.0 * (pow(2.0, 1.0 / 3.0) - 1.0);
    return (4.0 / 3.0)
           * (pow(1.0 + zeta, 1.0 / 3.0) - pow(1.0 - zeta, 1.0 / 3.0)) / denom;
}

/// VWN5 correlation at ζ = 0 (spin-unpolarized), matching
/// `vwn5_correlation(rho/2, rho/2, ...)` in xc_functional.cpp.
///
/// At ζ = 0 the CPU evaluates f(0) = 0, f'(0) = 0 and ζ⁴ = 0 **exactly** in
/// IEEE arithmetic, so `ac·f/f''(0)·(1−ζ⁴)`, `(ecF−ecP)·f·ζ⁴` and ∂ε_c/∂ζ are
/// all exactly zero: the ferromagnetic and spin-stiffness channels cancel out
/// of both ε_c and v_c and only the paramagnetic Padé set survives.
__device__ __forceinline__ void
vwn5_correlation_zeta0(double rho_a, double rho_b, double& ec, double& vc_a) {
    // Per-spin density floor matching libxc work_lda clamping (CPU parity).
    rho_a = fmax(rho_a, kSpinDensMin);
    rho_b = fmax(rho_b, kSpinDensMin);

    const double rho = rho_a + rho_b;
    const double rs = pow(3.0 / (4.0 * kPi * rho), 1.0 / 3.0);
    const double x = sqrt(rs);

    ec = vwn5_F(x, kVwnP_A, kVwnP_X0, kVwnP_B, kVwnP_C);
    // (r_s/3)·∂ε_c/∂r_s = (x/6)·∂ε_c/∂x because x = √r_s.
    vc_a = ec - x / 6.0 * vwn5_dFdx(x, kVwnP_A, kVwnP_X0, kVwnP_B, kVwnP_C);
}

/// Full spin-polarized VWN5 correlation — `vwn5_correlation` twin.
__device__ __forceinline__ void vwn5_correlation_spin(double rho_a,
                                                      double rho_b, double& ec,
                                                      double& vc_a,
                                                      double& vc_b) {
    rho_a = fmax(rho_a, kSpinDensMin);
    rho_b = fmax(rho_b, kSpinDensMin);

    const double rho = rho_a + rho_b;
    const double rs = pow(3.0 / (4.0 * kPi * rho), 1.0 / 3.0);
    const double x = sqrt(rs);
    const double zeta = (rho_a - rho_b) / rho;

    const double ecP = vwn5_F(x, kVwnP_A, kVwnP_X0, kVwnP_B, kVwnP_C);
    const double ecF = vwn5_F(x, kVwnF_A, kVwnF_X0, kVwnF_B, kVwnF_C);
    const double ac = vwn5_F(x, kVwnA_A, kVwnA_X0, kVwnA_B, kVwnA_C);

    const double fz = spin_f(zeta);
    const double z4 = zeta * zeta * zeta * zeta;
    ec = ecP + ac * fz / kFppZero * (1.0 - z4) + (ecF - ecP) * fz * z4;

    const double decP_dx = vwn5_dFdx(x, kVwnP_A, kVwnP_X0, kVwnP_B, kVwnP_C);
    const double decF_dx = vwn5_dFdx(x, kVwnF_A, kVwnF_X0, kVwnF_B, kVwnF_C);
    const double dac_dx = vwn5_dFdx(x, kVwnA_A, kVwnA_X0, kVwnA_B, kVwnA_C);
    const double dec_dx = decP_dx + dac_dx * fz / kFppZero * (1.0 - z4)
                          + (decF_dx - decP_dx) * fz * z4;
    const double rs3_dec = x / 6.0 * dec_dx;

    const double dfz = spin_df(zeta);
    const double dz4 = 4.0 * zeta * zeta * zeta;
    const double dec_dz = ac / kFppZero * (dfz * (1.0 - z4) - fz * dz4)
                          + (ecF - ecP) * (dfz * z4 + fz * dz4);

    vc_a = ec - rs3_dec - (zeta - 1.0) * dec_dz;
    vc_b = ec - rs3_dec - (zeta + 1.0) * dec_dz;
}

/// One grid point: V_xc and ε_xc from the total density (restricted, ζ = 0).
__global__ void xc_lsda_kernel(const double* __restrict__ rho_r,
                               double* __restrict__ vxc_r,
                               double* __restrict__ exc_r, int nr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr)
        return;
    const double rho = rho_r[i];
    if (rho < kRhoMin) {
        vxc_r[i] = 0.0;
        exc_r[i] = 0.0;
        return;
    }
    const double ra = rho * 0.5;
    const double rb = rho * 0.5;

    // Slater exchange — the per-spin form `slater_exchange` builds. The CPU
    // passes the UNFLOORED rho/2 here; only correlation floors its inputs.
    const double Cx_spin = pow(2.0, 1.0 / 3.0) * kCx;
    const double ex_a = -Cx_spin * pow(ra, 4.0 / 3.0);
    const double ex_b = -Cx_spin * pow(rb, 4.0 / 3.0);
    const double ex = (ex_a + ex_b) / rho;
    const double vx_a = -(4.0 / 3.0) * Cx_spin * pow(ra, 1.0 / 3.0);

    // VWN5 correlation (ζ = 0 branch).
    double ec, vc_a;
    vwn5_correlation_zeta0(ra, rb, ec, vc_a);

    exc_r[i] = ex + ec;
    vxc_r[i] = vx_a + vc_a;
}

/// One grid point: collinear LSDA from (ρ_α, ρ_β). Floors negatives like
/// `XCBuilder::step_spin` before evaluating.
__global__ void xc_lsda_spin_kernel(const double* __restrict__ rho_a,
                                    const double* __restrict__ rho_b,
                                    double* __restrict__ vxc_a,
                                    double* __restrict__ vxc_b,
                                    double* __restrict__ exc_r, int nr) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nr)
        return;
    double ra = rho_a[i];
    double rb = rho_b[i];
    if (ra < 0.0)
        ra = 0.0;
    if (rb < 0.0)
        rb = 0.0;
    const double rho = ra + rb;
    if (rho < kRhoMin) {
        vxc_a[i] = 0.0;
        vxc_b[i] = 0.0;
        exc_r[i] = 0.0;
        return;
    }

    const double Cx_spin = pow(2.0, 1.0 / 3.0) * kCx;
    const double ex_a = -Cx_spin * pow(ra, 4.0 / 3.0);
    const double ex_b = -Cx_spin * pow(rb, 4.0 / 3.0);
    const double ex = (ex_a + ex_b) / rho;
    const double vx_a = -(4.0 / 3.0) * Cx_spin * pow(ra, 1.0 / 3.0);
    const double vx_b = -(4.0 / 3.0) * Cx_spin * pow(rb, 1.0 / 3.0);

    double ec, vc_a, vc_b;
    vwn5_correlation_spin(ra, rb, ec, vc_a, vc_b);

    exc_r[i] = ex + ec;
    vxc_a[i] = vx_a + vc_a;
    vxc_b[i] = vx_b + vc_b;
}

/// Stage 1: fixed grid of `kReduceBlocks` blocks, grid-stride over nr, one
/// partial per block. Threads past `nr` contribute the additive identity 0.
__global__ void xc_energy_stage1(const double* __restrict__ rho_r,
                                 const double* __restrict__ exc_r, int nr,
                                 double* __restrict__ partials) {
    __shared__ double s[kBlock];
    double acc = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nr;
         i += gridDim.x * blockDim.x)
        acc += rho_r[i] * exc_r[i];

    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partials[blockIdx.x] = s[0];
}

/// Stage 1 collinear: Σ (ρ_α + ρ_β) ε_xc  (floors negatives like step_spin).
__global__ void xc_energy_spin_stage1(const double* __restrict__ rho_a,
                                      const double* __restrict__ rho_b,
                                      const double* __restrict__ exc_r, int nr,
                                      double* __restrict__ partials) {
    __shared__ double s[kBlock];
    double acc = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nr;
         i += gridDim.x * blockDim.x) {
        double ra = rho_a[i];
        double rb = rho_b[i];
        if (ra < 0.0)
            ra = 0.0;
        if (rb < 0.0)
            rb = 0.0;
        acc += (ra + rb) * exc_r[i];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partials[blockIdx.x] = s[0];
}

/// Stage 1: Σ (ρ_α V_xc^α + ρ_β V_xc^β) for e_xc_int_spin.
__global__ void xc_int_spin_stage1(const double* __restrict__ rho_a,
                                   const double* __restrict__ rho_b,
                                   const double* __restrict__ vxc_a,
                                   const double* __restrict__ vxc_b, int nr,
                                   double* __restrict__ partials) {
    __shared__ double s[kBlock];
    double acc = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nr;
         i += gridDim.x * blockDim.x) {
        double ra = rho_a[i];
        double rb = rho_b[i];
        if (ra < 0.0)
            ra = 0.0;
        if (rb < 0.0)
            rb = 0.0;
        acc += ra * vxc_a[i] + rb * vxc_b[i];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partials[blockIdx.x] = s[0];
}

/// Stage 2: one block reduces exactly `kReduceBlocks` partials, applies dV.
__global__ void xc_energy_stage2(const double* __restrict__ partials, double dV,
                                 double* __restrict__ out) {
    __shared__ double s[kReduceBlocks];
    const int t = threadIdx.x;
    s[t] = partials[t];
    __syncthreads();
    for (int stride = kReduceBlocks / 2; stride > 0; stride >>= 1) {
        if (t < stride)
            s[t] += s[t + stride];
        __syncthreads();
    }
    if (t == 0)
        *out = s[0] * dV;
}

} // namespace

void xc_lsda_grid(const double* d_rho_r, double* d_vxc_r, double* d_exc_r,
                  int nr, cudaStream_t stream) {
    if (nr <= 0)
        return;
    xc_lsda_kernel<<<::ATV::cuda::grid_size(nr, kBlock), kBlock, 0, stream>>>(
        d_rho_r, d_vxc_r, d_exc_r, nr);
}

void xc_lsda_spin_grid(const double* d_rho_a, const double* d_rho_b,
                       double* d_vxc_a, double* d_vxc_b, double* d_exc_r,
                       int nr, cudaStream_t stream) {
    if (nr <= 0)
        return;
    xc_lsda_spin_kernel<<<::ATV::cuda::grid_size(nr, kBlock), kBlock, 0,
                          stream>>>(d_rho_a, d_rho_b, d_vxc_a, d_vxc_b, d_exc_r,
                                    nr);
}

void xc_energy(const double* d_rho_r, const double* d_exc_r, int nr, double dV,
               double* d_Exc, double* d_scratch, cudaStream_t stream) {
    cudaMemsetAsync(d_Exc, 0, sizeof(double), stream);
    if (nr <= 0)
        return;
    xc_energy_stage1<<<kReduceBlocks, kBlock, 0, stream>>>(d_rho_r, d_exc_r, nr,
                                                           d_scratch);
    xc_energy_stage2<<<1, kReduceBlocks, 0, stream>>>(d_scratch, dV, d_Exc);
}

void xc_energy_spin(const double* d_rho_a, const double* d_rho_b,
                    const double* d_exc_r, int nr, double dV, double* d_Exc,
                    double* d_scratch, cudaStream_t stream) {
    cudaMemsetAsync(d_Exc, 0, sizeof(double), stream);
    if (nr <= 0)
        return;
    xc_energy_spin_stage1<<<kReduceBlocks, kBlock, 0, stream>>>(
        d_rho_a, d_rho_b, d_exc_r, nr, d_scratch);
    xc_energy_stage2<<<1, kReduceBlocks, 0, stream>>>(d_scratch, dV, d_Exc);
}

void xc_int_spin(const double* d_rho_a, const double* d_rho_b,
                 const double* d_vxc_a, const double* d_vxc_b, int nr,
                 double dV, double* d_e_xc_int, double* d_scratch,
                 cudaStream_t stream) {
    cudaMemsetAsync(d_e_xc_int, 0, sizeof(double), stream);
    if (nr <= 0)
        return;
    xc_int_spin_stage1<<<kReduceBlocks, kBlock, 0, stream>>>(
        d_rho_a, d_rho_b, d_vxc_a, d_vxc_b, nr, d_scratch);
    xc_energy_stage2<<<1, kReduceBlocks, 0, stream>>>(d_scratch, dV,
                                                      d_e_xc_int);
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
