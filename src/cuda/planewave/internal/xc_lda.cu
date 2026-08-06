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

// VWN5 paramagnetic channel {A, x0, b, c}, Hartree convention (A_Ha = A_Ry/2,
// A_Ry = 0.0621814). Held as scalars rather than a struct so no aggregate
// device global is needed.
__device__ constexpr double kVwnA = 0.0310907;
__device__ constexpr double kVwnX0 = -0.10498;
__device__ constexpr double kVwnB = 3.72744;
__device__ constexpr double kVwnC = 12.9352;

/// F(x) = A { ln(x²/X) + 2b/Q atan(Q/(2x+b))
///          − b x0/X(x0) [ ln((x−x0)²/X) + 2(b+2x0)/Q atan(Q/(2x+b)) ] }
/// with X(x) = x² + bx + c, Q = √(4c − b²).
__device__ __forceinline__ double vwn5_F(double x) {
    const double X = x * x + kVwnB * x + kVwnC;
    const double X0 = kVwnX0 * kVwnX0 + kVwnB * kVwnX0 + kVwnC;
    const double Q = sqrt(4.0 * kVwnC - kVwnB * kVwnB);
    const double atanQ = atan2(Q, 2.0 * x + kVwnB);

    const double term1 = log(x * x / X);
    const double term2 = 2.0 * kVwnB / Q * atanQ;
    const double term3 = -kVwnB * kVwnX0 / X0
                         * (log((x - kVwnX0) * (x - kVwnX0) / X)
                            + 2.0 * (kVwnB + 2.0 * kVwnX0) / Q * atanQ);

    return kVwnA * (term1 + term2 + term3);
}

/// dF/dx. The identity (2x+b)² + Q² = 4X collapses both arctan derivatives,
/// so Q never appears — the same simplification the CPU function uses.
__device__ __forceinline__ double vwn5_dFdx(double x) {
    const double X = x * x + kVwnB * x + kVwnC;
    const double X0 = kVwnX0 * kVwnX0 + kVwnB * kVwnX0 + kVwnC;

    const double dterm1 = 2.0 / x - (2.0 * x + kVwnB) / X;
    const double dterm2 = -4.0 * kVwnB / (4.0 * X);
    const double fac = -kVwnB * kVwnX0 / X0;
    const double dlog = 2.0 / (x - kVwnX0) - (2.0 * x + kVwnB) / X;
    const double datan = -4.0 * (kVwnB + 2.0 * kVwnX0) / (4.0 * X);
    const double dterm3 = fac * (dlog + datan);

    return kVwnA * (dterm1 + dterm2 + dterm3);
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

    ec = vwn5_F(x);
    // (r_s/3)·∂ε_c/∂r_s = (x/6)·∂ε_c/∂x because x = √r_s.
    vc_a = ec - x / 6.0 * vwn5_dFdx(x);
}

/// One grid point: V_xc and ε_xc from the total density.
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

void xc_energy(const double* d_rho_r, const double* d_exc_r, int nr, double dV,
               double* d_Exc, double* d_scratch, cudaStream_t stream) {
    cudaMemsetAsync(d_Exc, 0, sizeof(double), stream);
    if (nr <= 0)
        return;
    xc_energy_stage1<<<kReduceBlocks, kBlock, 0, stream>>>(d_rho_r, d_exc_r, nr,
                                                           d_scratch);
    xc_energy_stage2<<<1, kReduceBlocks, 0, stream>>>(d_scratch, dV, d_Exc);
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
