/// @file cpu/gpw/internal/vdw_df.cpp
/// @brief RPS evaluation of vdW-DF (DRSLL / DF1) and vdW-DF2 (LMKLL / DF2).
///
/// Energy + self-consistent potential follow Soler et al. and CP2K
/// `qs_dispersion_nonloc.F` (`get_q0_on_grid_vdw`, `vdW_energy`, `get_potential`).

#include "atomiverse/cpu/gpw/internal/vdw_df.h"

#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include <fftw3.h>

#include "atomiverse/cpu/device/chunked_range.h"
#include "atomiverse/cpu/gpw/internal/fft_cache.h"
#include "atomiverse/cpu/gpw/internal/grid_fft.h"
#include "atomiverse/cpu/gpw/internal/rvv10.h"
#include "atomiverse/cpu/gpw/internal/vdw_df_kernel_data.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace {

// Perdew–Wang LDA constants used inside q₀ (libvdwxc / Dion).
constexpr double kPi = 3.14159265358979323846;
constexpr double kKfPref = 3.0936677262801355; // (3 π²)^{1/3}
constexpr double kRsPref = 0.6203504908994001; // (3/(4π))^{1/3}
constexpr double kLdaA = 0.031091;
constexpr double kLdaA1 = 0.2137;
constexpr double kLdaB1 = 7.5957;
constexpr double kLdaB2 = 3.5876;
constexpr double kLdaB3 = 1.6382;
constexpr double kLdaB4 = 0.49294;
constexpr double kRhoCut = 1e-12;
constexpr double kQMin = 1.0e-4; // soft floor like CP2K q_min

void interpolate_kernel(double k, double* kernel_aa) {
    using namespace vdw_df_data;
    const float* tab = kernel_kaa();
    const int asqr = kNAlpha * kNAlpha;
    const double index = k / kDeltaK;
    int istart = static_cast<int>(std::floor(index));
    if (k < 0.0 || istart > kNKpoints - 2) {
        for (int i = 0; i < asqr; ++i)
            kernel_aa[i] = 0.0;
        return;
    }
    const double t = index - static_cast<double>(istart);
    const double t2 = 1.0 - t;
    const float* left = tab + static_cast<std::size_t>(istart) * asqr;
    const float* right = tab + static_cast<std::size_t>(istart + 1) * asqr;
    for (int i = 0; i < asqr; ++i)
        kernel_aa[i] = static_cast<double>(left[i]) * t2
                       + static_cast<double>(right[i]) * t;
}

/// @brief Evaluate p_α(q) and optionally dp_α/dq (cubic splines; prefactor=1).
void palpha_and_deriv(double q, double* p_out, double* dp_out) {
    using namespace vdw_df_data;
    const double* qm = qmesh();
    const float* spl = splines_arc();
    int alpha = 0;
    if (q > qm[kNAlpha - 1]) {
        q = qm[kNAlpha - 1];
        alpha = kNAlpha - 2;
    } else if (q <= qm[0]) {
        q = qm[0];
        alpha = 0;
    } else {
        for (alpha = 1; alpha < kNAlpha; ++alpha) {
            if (qm[alpha] >= q) {
                --alpha;
                break;
            }
        }
    }
    const double dq = q - qm[alpha];
    for (int beta = 0; beta < kNAlpha; ++beta) {
        const float* poly = spl + (beta * kNRegions + alpha) * kNPCoef;
        const double c0 = static_cast<double>(poly[0]);
        const double c1 = static_cast<double>(poly[1]);
        const double c2 = static_cast<double>(poly[2]);
        const double c3 = static_cast<double>(poly[3]);
        p_out[beta] = c0 + dq * (c1 + dq * (c2 + dq * c3));
        if (dp_out)
            dp_out[beta] = c1 + dq * (2.0 * c2 + dq * 3.0 * c3);
    }
}

/// Soft saturation q0 = qcut (1 - exp(-Σ (q/qcut)^m / m)), m=1..12.
/// Returns {q0, dq0/dq}.
void soft_saturate(double q, double qcut, double& q0, double& dq0_dq) {
    double exponent = 0.0;
    double d_exp_d_qs = 0.0; // d(exponent)/d(q/qcut)
    double qscaled0 = q / qcut;
    double qscaled = 1.0;
    for (int m = 1; m <= 12; ++m) {
        qscaled *= qscaled0;
        exponent += qscaled / static_cast<double>(m);
        // d/d(qscaled0) of (qscaled0^m / m) = qscaled0^{m-1}
        d_exp_d_qs += std::pow(qscaled0, m - 1);
    }
    const double em = std::exp(-exponent);
    q0 = qcut * (1.0 - em);
    // dq0/dq = qcut * em * d_exp_d_qs * (1/qcut) = em * d_exp_d_qs
    dq0_dq = em * d_exp_d_qs;
}

/// @brief Shared RPS energy (+ optional SC potential) with explicit Z_ab.
VdwDfResult vdw_df_rps(const UniformGrid& grid, const std::vector<double>& rho,
                       const std::array<std::vector<double>, 3>& grad_rho,
                       double zab, bool want_potential) {
    using namespace vdw_df_data;
    const bool prof = std::getenv("ATV_GPW_PROF") != nullptr;
    const auto profile_start = std::chrono::steady_clock::now();
    const int n0 = grid.n[0];
    const int n1 = grid.n[1];
    const int n2 = grid.n[2];
    const long ntot = static_cast<long>(n0) * n1 * n2;
    if (static_cast<long>(rho.size()) != ntot
        || static_cast<long>(grad_rho[0].size()) != ntot
        || static_cast<long>(grad_rho[1].size()) != ntot
        || static_cast<long>(grad_rho[2].size()) != ntot)
        throw std::runtime_error("vdW-DF: density size mismatch vs grid");

    const int n2c = n2 / 2 + 1;
    const long ncplx = static_cast<long>(n0) * n1 * n2c;
    const int nalpha = kNAlpha;
    const double qcut = kVdwDfQCut;
    const double dV = grid.weight();

    // q0 + Soler-style derivatives (ρ ∂q0/∂ρ and ρ-weighted grad term).
    std::vector<double> q0(static_cast<std::size_t>(ntot));
    std::vector<double> dq0_drho(
        want_potential ? static_cast<std::size_t>(ntot) : 0, 0.0);
    std::vector<double> dq0_dgradrho(
        want_potential ? static_cast<std::size_t>(ntot) : 0, 0.0);
#pragma omp parallel for schedule(static)
    for (long g = 0; g < ntot; ++g) {
        const double mrho = rho[static_cast<std::size_t>(g)];
        const double gx = grad_rho[0][static_cast<std::size_t>(g)];
        const double gy = grad_rho[1][static_cast<std::size_t>(g)];
        const double gz = grad_rho[2][static_cast<std::size_t>(g)];
        const double msigma = gx * gx + gy * gy + gz * gz;
        if (mrho < kRhoCut) {
            q0[static_cast<std::size_t>(g)] = qcut;
            continue;
        }
        const double rho1_root3 = std::cbrt(mrho);
        const double kF = kKfPref * rho1_root3;
        const double rs = kRsPref / rho1_root3;
        const double rs_sqr = rs * rs;
        const double rs_root2 = std::sqrt(rs);
        const double rho_sqr = mrho * mrho;
        const double gradient_correction =
            -zab / (36.0 * kF * rho_sqr) * msigma;
        const double LDA_1 = 8.0 * kPi / 3.0 * (kLdaA * (1.0 + kLdaA1 * rs));
        const double LDA_2 = 2.0 * kLdaA
                             * (kLdaB1 * rs_root2 + kLdaB2 * rs
                                + kLdaB3 * rs * rs_root2 + kLdaB4 * rs_sqr);
        const double q =
            kF + LDA_1 * std::log(1.0 + 1.0 / LDA_2) + gradient_correction;
        double q0g = 0.0, dq0_dq = 0.0;
        soft_saturate(q, qcut, q0g, dq0_dq);
        if (q0g < kQMin)
            q0g = kQMin;
        q0[static_cast<std::size_t>(g)] = q0g;
        if (!want_potential)
            continue;
        // CP2K: dq0_drho stores (ρ ∂q0/∂ρ) factors used as
        //   ∂θ/∂ρ = P + dP/dq * dq0_drho  with θ = ρ P.
        dq0_drho[static_cast<std::size_t>(g)] =
            dq0_dq
            * (kF / 3.0 - 7.0 / 3.0 * gradient_correction
               - 8.0 * kPi / 9.0 * kLdaA * kLdaA1 * rs
                     * std::log(1.0 + 1.0 / LDA_2)
               + LDA_1 / (LDA_2 * (1.0 + LDA_2))
                     * (2.0 * kLdaA
                        * (kLdaB1 / 6.0 * rs_root2 + kLdaB2 / 3.0 * rs
                           + kLdaB3 / 2.0 * rs * rs_root2
                           + 2.0 * kLdaB4 / 3.0 * rs_sqr)));
        // ρ * ∂q0/∂σ * 2  packaged for flux · ∇ρ (σ = |∇ρ|²).
        dq0_dgradrho[static_cast<std::size_t>(g)] =
            mrho * dq0_dq * 2.0 * (-zab) / (36.0 * kF * rho_sqr);
    }

    FftwBuffer<double> real_storage(static_cast<std::size_t>(ntot) * nalpha);
    FftwBuffer<fftw_complex> complex_storage(static_cast<std::size_t>(ncplx)
                                             * nalpha);
    double* rbuf = real_storage.data();
    fftw_complex* cbuf = complex_storage.data();

    // θ_α = ρ p_α(q0)
    // Inflate immutable spline data on the caller thread before workers.
    (void)splines_arc();
#pragma omp parallel
    {
        std::array<double, kNAlpha> ptmp;
#pragma omp for schedule(static)
        for (long g = 0; g < ntot; ++g) {
            const double rg = rho[static_cast<std::size_t>(g)];
            palpha_and_deriv(q0[static_cast<std::size_t>(g)], ptmp.data(),
                             nullptr);
            for (int a = 0; a < nalpha; ++a)
                rbuf[a * ntot + g] = rg * ptmp[static_cast<std::size_t>(a)];
        }
    }

    const auto theta_end = std::chrono::steady_clock::now();
    R2cFft& fft = R2cFft::for_shape(n0, n1, n2);
    for (int a = 0; a < nalpha; ++a) {
        double* rin = rbuf + a * ntot;
        fftw_complex* cout = cbuf + a * ncplx;
        fft.execute_r2c(rin, cout);
    }

    const auto forward_end = std::chrono::steady_clock::now();
    const double two_pi = 2.0 * kPi;
    const double gx_unit = two_pi / grid.L[0];
    const double gy_unit = two_pi / grid.L[1];
    const double gz_unit = two_pi / grid.L[2];

    // The table cardinality is the compile-time kNAlpha for this functional.
    // Load immutable data before entering workers (inflate errors propagate).
    (void)kernel_kaa();
    const device::ChunkedRange reciprocal_chunks(ncplx, 64);
    std::vector<double> partial_energy(
        static_cast<std::size_t>(reciprocal_chunks.chunks()), 0.0);
#pragma omp parallel for schedule(dynamic, 1)
    for (long chunk = 0; chunk < reciprocal_chunks.chunks(); ++chunk) {
        std::array<double, kNAlpha * kNAlpha> kernel_aa;
        std::array<std::complex<double>, kNAlpha> theta;
        double energy_chunk = 0.0;
        for (long kindex = reciprocal_chunks.begin(chunk);
             kindex < reciprocal_chunks.end(chunk); ++kindex) {
            const int i2 = static_cast<int>(kindex % n2c);
            const int i1 = static_cast<int>((kindex / n2c) % n1);
            const int i0 =
                static_cast<int>(kindex / (static_cast<long>(n1) * n2c));
            const int m0 = (i0 <= n0 / 2) ? i0 : i0 - n0;
            const int m1 = (i1 <= n1 / 2) ? i1 : i1 - n1;
            const double gx = m0 * gx_unit, gy = m1 * gy_unit,
                         gz = i2 * gz_unit;
            const int weight =
                (i2 == 0 || (n2 % 2 == 0 && i2 == n2c - 1)) ? 1 : 2;
            const double k = std::sqrt(gx * gx + gy * gy + gz * gz);
            interpolate_kernel(k, kernel_aa.data());
            for (int a = 0; a < nalpha; ++a) {
                const fftw_complex& th = cbuf[a * ncplx + kindex];
                theta[static_cast<std::size_t>(a)] = {th[0], th[1]};
            }
            for (int a1 = 0; a1 < nalpha; ++a1) {
                std::complex<double> F(0.0, 0.0);
                for (int a2 = 0; a2 < nalpha; ++a2) {
                    const double phi =
                        kernel_aa[static_cast<std::size_t>(a1 * nalpha + a2)];
                    F += theta[static_cast<std::size_t>(a2)] * phi;
                }
                energy_chunk +=
                    weight
                    * (theta[static_cast<std::size_t>(a1)].real() * F.real()
                       + theta[static_cast<std::size_t>(a1)].imag() * F.imag());
                if (want_potential) {
                    fftw_complex& u = cbuf[a1 * ncplx + kindex];
                    u[0] = F.real();
                    u[1] = F.imag();
                }
            }
        }
        partial_energy[static_cast<std::size_t>(chunk)] = energy_chunk;
    }
    double energy = 0.0;
    for (const double e : partial_energy)
        energy += e;

    const auto convolution_end = std::chrono::steady_clock::now();
    auto profile = [&]() {
        if (!prof)
            return;
        const auto now = std::chrono::steady_clock::now();
        std::fprintf(
            stderr,
            "[gpw nonlocal] vdW-DF ngrid=%ld potential=%d theta=%.0f "
            "forward=%.0f convolution=%.0f response=%.0f ms\n",
            ntot, static_cast<int>(want_potential),
            std::chrono::duration<double, std::milli>(theta_end - profile_start)
                .count(),
            std::chrono::duration<double, std::milli>(forward_end - theta_end)
                .count(),
            std::chrono::duration<double, std::milli>(convolution_end
                                                      - forward_end)
                .count(),
            std::chrono::duration<double, std::milli>(now - convolution_end)
                .count());
    };
    energy *= 0.5 * dV / static_cast<double>(ntot);

    VdwDfResult res;
    res.energy = energy;

    if (!want_potential) {
        profile();
        return res;
    }

    // IFFT u_α(k) → u_α(r); FFTW c2r multiplies by N → divide by N.
    for (int a = 0; a < nalpha; ++a) {
        fftw_complex* cin = cbuf + a * ncplx;
        double* rout = rbuf + a * ntot;
        fft.execute_c2r(cin, rout);
        const double invN = 1.0 / static_cast<double>(ntot);
        for (long g = 0; g < ntot; ++g)
            rout[g] *= invN;
    }

    // Reuse the input spectral ∇ρ for the divergence flux.
    res.v_r.assign(static_cast<std::size_t>(ntot), 0.0);
    std::array<std::vector<double>, 3> flux;
    for (int d = 0; d < 3; ++d)
        flux[d].assign(static_cast<std::size_t>(ntot), 0.0);

#pragma omp parallel
    {
        std::array<double, kNAlpha> p_a, dp_a;
#pragma omp for schedule(static)
        for (long g = 0; g < ntot; ++g) {
            const double qg = q0[static_cast<std::size_t>(g)];
            palpha_and_deriv(qg, p_a.data(), dp_a.data());
            double pot = 0.0;
            double hpref = 0.0;
            for (int a = 0; a < nalpha; ++a) {
                const double u = rbuf[a * ntot + g];
                // Soler eq. 10 / CP2K get_potential (DRSLL/LMKLL):
                //   v += u * (P + dP/dq * dq0_drho)
                //   h += u * dP/dq * dq0_dgradrho
                pot += u
                       * (p_a[static_cast<std::size_t>(a)]
                          + dp_a[static_cast<std::size_t>(a)]
                                * dq0_drho[static_cast<std::size_t>(g)]);
                if (qg != qcut) // saturated q0 ⇒ zero gradient response
                    hpref += u * dp_a[static_cast<std::size_t>(a)]
                             * dq0_dgradrho[static_cast<std::size_t>(g)];
            }
            res.v_r[static_cast<std::size_t>(g)] = pot;
            flux[0][static_cast<std::size_t>(g)] =
                hpref * grad_rho[0][static_cast<std::size_t>(g)];
            flux[1][static_cast<std::size_t>(g)] =
                hpref * grad_rho[1][static_cast<std::size_t>(g)];
            flux[2][static_cast<std::size_t>(g)] =
                hpref * grad_rho[2][static_cast<std::size_t>(g)];
        }
    }

    const std::vector<double> divflux = divergence(grid, flux);
    for (long g = 0; g < ntot; ++g)
        res.v_r[static_cast<std::size_t>(g)] -=
            divflux[static_cast<std::size_t>(g)];

    profile();
    return res;
}

} // namespace

void vdw_df_q0(const std::vector<double>& rho, const std::vector<double>& sigma,
               std::vector<double>& q0) {
    vdw_df_q0(rho, sigma, q0, kVdwDf1Zab);
}

void vdw_df_q0(const std::vector<double>& rho, const std::vector<double>& sigma,
               std::vector<double>& q0, double zab) {
    const std::size_t n = rho.size();
    if (sigma.size() != n)
        throw std::runtime_error("vdW-DF q0: rho/sigma size mismatch");
    q0.resize(n);
    const double qcut = kVdwDfQCut;
    for (std::size_t i = 0; i < n; ++i) {
        double mrho = rho[i];
        double msigma = sigma[i];
        if (mrho < kRhoCut) {
            q0[i] = qcut;
            continue;
        }
        const double rho1_root3 = std::cbrt(mrho);
        const double kF = kKfPref * rho1_root3;
        const double rs = kRsPref / rho1_root3;
        const double rs_sqr = rs * rs;
        const double rs_root2 = std::sqrt(rs);
        const double rho_sqr = mrho * mrho;
        const double gradient_correction =
            -zab / (36.0 * kF * rho_sqr) * msigma;
        const double LDA_1 = 8.0 * kPi / 3.0 * (kLdaA * (1.0 + kLdaA1 * rs));
        const double LDA_2 = 2.0 * kLdaA
                             * (kLdaB1 * rs_root2 + kLdaB2 * rs
                                + kLdaB3 * rs * rs_root2 + kLdaB4 * rs_sqr);
        const double q =
            kF + LDA_1 * std::log(1.0 + 1.0 / LDA_2) + gradient_correction;
        double q0g = 0.0, dq0_dq = 0.0;
        soft_saturate(q, qcut, q0g, dq0_dq);
        if (q0g < kQMin)
            q0g = kQMin;
        q0[i] = q0g;
    }
}

VdwDfResult vdw_df_drsll(const UniformGrid& grid,
                         const std::vector<double>& rho,
                         const std::array<std::vector<double>, 3>& grad_rho,
                         bool potential) {
    return vdw_df_rps(grid, rho, grad_rho, kVdwDf1Zab, potential);
}

VdwDfResult vdw_df_lmkll(const UniformGrid& grid,
                         const std::vector<double>& rho,
                         const std::array<std::vector<double>, 3>& grad_rho,
                         bool potential) {
    return vdw_df_rps(grid, rho, grad_rho, kVdwDf2Zab, potential);
}

NonlocalVdwOnGrid nonlocal_vdw_on_grid(const UniformGrid& grid,
                                       const std::vector<double>& rho,
                                       bool vdw_df, bool vdw_df2, bool rvv10,
                                       bool potential) {
    NonlocalVdwOnGrid out;
    if (!(vdw_df || vdw_df2 || rvv10))
        return out;
    const auto gxyz = gradient(grid, rho);
    if (rvv10) {
        Rvv10Result r =
            rvv10_energy(grid, rho, gxyz, kRvv10B, kRvv10C, potential);
        out.energy = r.energy;
        out.v_r = std::move(r.v_r);
    } else if (vdw_df2) {
        VdwDfResult r = vdw_df_lmkll(grid, rho, gxyz, potential);
        out.energy = r.energy;
        out.v_r = std::move(r.v_r);
    } else {
        VdwDfResult r = vdw_df_drsll(grid, rho, gxyz, potential);
        out.energy = r.energy;
        out.v_r = std::move(r.v_r);
    }
    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
