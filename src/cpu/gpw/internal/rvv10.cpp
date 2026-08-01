/// @file cpu/gpw/internal/rvv10.cpp
/// @brief RPS evaluation of rVV10 nonlocal correlation (CP2K TYPE RVV10).
///
/// Energy + self-consistent potential follow CP2K `qs_dispersion_nonloc.F`
/// (`get_q0_on_grid_rvv10`, `vdW_energy`, `get_potential`).

#include "atomiverse/cpu/gpw/internal/rvv10.h"

#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

#include <fftw3.h>

#include "atomiverse/cpu/gpw/internal/fft_cache.h"
#include "atomiverse/cpu/gpw/internal/grid_fft.h"
#include "atomiverse/cpu/gpw/internal/rvv10_kernel_data.h"

namespace ATV {
namespace cpu {
namespace gpw {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kRhoCut = 1e-10;

/// @brief Natural cubic spline second derivatives of Kronecker-δ basis on q-mesh
///        (CP2K initialize_spline_interpolation).
void init_p_spline_d2(const double* x, int n, std::vector<double>& d2) {
    d2.assign(static_cast<std::size_t>(n * n), 0.0);
    std::vector<double> y(static_cast<std::size_t>(n));
    std::vector<double> temp(static_cast<std::size_t>(n));
    for (int P_i = 0; P_i < n; ++P_i) {
        for (int j = 0; j < n; ++j)
            y[static_cast<std::size_t>(j)] = 0.0;
        y[static_cast<std::size_t>(P_i)] = 1.0;
        d2[static_cast<std::size_t>(P_i * n + 0)] = 0.0;
        temp[0] = 0.0;
        for (int index = 1; index < n - 1; ++index) {
            const double temp1 =
                (x[index] - x[index - 1]) / (x[index + 1] - x[index - 1]);
            const double temp2 =
                temp1 * d2[static_cast<std::size_t>(P_i * n + index - 1)]
                + 2.0;
            d2[static_cast<std::size_t>(P_i * n + index)] =
                (temp1 - 1.0) / temp2;
            temp[static_cast<std::size_t>(index)] =
                (y[static_cast<std::size_t>(index + 1)]
                 - y[static_cast<std::size_t>(index)])
                    / (x[index + 1] - x[index])
                - (y[static_cast<std::size_t>(index)]
                   - y[static_cast<std::size_t>(index - 1)])
                      / (x[index] - x[index - 1]);
            temp[static_cast<std::size_t>(index)] =
                (6.0 * temp[static_cast<std::size_t>(index)]
                     / (x[index + 1] - x[index - 1])
                 - temp1 * temp[static_cast<std::size_t>(index - 1)])
                / temp2;
        }
        d2[static_cast<std::size_t>(P_i * n + (n - 1))] = 0.0;
        for (int index = n - 2; index >= 0; --index) {
            d2[static_cast<std::size_t>(P_i * n + index)] =
                d2[static_cast<std::size_t>(P_i * n + index)]
                    * d2[static_cast<std::size_t>(P_i * n + index + 1)]
                + temp[static_cast<std::size_t>(index)];
        }
    }
}

/// @brief Evaluate all p_α(q) and dp_α/dq via cubic spline on Kronecker basis.
void palpha_and_deriv_at_q(const double* x, const std::vector<double>& d2, int n,
                           double q, double* p_out, double* dp_out) {
    int lower = 0;
    int upper = n - 1;
    if (q <= x[0]) {
        for (int a = 0; a < n; ++a) {
            p_out[a] = (a == 0) ? 1.0 : 0.0;
            if (dp_out)
                dp_out[a] = 0.0;
        }
        return;
    }
    if (q >= x[n - 1]) {
        for (int a = 0; a < n; ++a) {
            p_out[a] = (a == n - 1) ? 1.0 : 0.0;
            if (dp_out)
                dp_out[a] = 0.0;
        }
        return;
    }
    while (upper - lower > 1) {
        const int mid = (upper + lower) / 2;
        if (q > x[mid])
            lower = mid;
        else
            upper = mid;
    }
    const double dx = x[upper] - x[lower];
    const double const_ = dx * dx / 6.0;
    const double a = (x[upper] - q) / dx;
    const double b = (q - x[lower]) / dx;
    const double c = (a * a * a - a) * const_;
    const double d = (b * b * b - b) * const_;
    // da/dq = -1/dx, db/dq = 1/dx
    // dc/dq = (3a²-1) * da/dq * const_ ; same for d with b
    const double e = (3.0 * a * a - 1.0) * (dx / 6.0); // used like CP2K
    const double f = (3.0 * b * b - 1.0) * (dx / 6.0);
    for (int P_i = 0; P_i < n; ++P_i) {
        const double yl = (P_i == lower) ? 1.0 : 0.0;
        const double yu = (P_i == upper) ? 1.0 : 0.0;
        const double d2l = d2[static_cast<std::size_t>(P_i * n + lower)];
        const double d2u = d2[static_cast<std::size_t>(P_i * n + upper)];
        p_out[P_i] = a * yl + b * yu + c * d2l + d * d2u;
        if (dp_out) {
            // CP2K: dP_dq0 = (y_hi - y_low)/dq - e*d2_low + f*d2_hi
            dp_out[P_i] = (yu - yl) / dx - e * d2l + f * d2u;
        }
    }
}

void interpolate_kernel_linear(double k, double* kernel_aa) {
    using namespace rvv10_data;
    const float* tab = kernel_kaa();
    const int asqr = kNAlpha * kNAlpha;
    const double index = k / kDeltaK;
    int istart = static_cast<int>(std::floor(index));
    if (k < 0.0 || istart > kNKpoints - 2) {
        for (int i = 0; i < asqr; ++i)
            kernel_aa[i] = 0.0;
        return;
    }
    if (istart < 0)
        istart = 0;
    const double t = index - static_cast<double>(istart);
    const double t2 = 1.0 - t;
    const float* left = tab + static_cast<std::size_t>(istart) * asqr;
    const float* right = tab + static_cast<std::size_t>(istart + 1) * asqr;
    for (int i = 0; i < asqr; ++i)
        kernel_aa[i] = static_cast<double>(left[i]) * t2
                       + static_cast<double>(right[i]) * t;
}

void soft_saturate(double q, double qcut, double& q0, double& dq0_dq) {
    double exponent = 0.0;
    double d_exp_d_qs = 0.0;
    double qscaled0 = q / qcut;
    double qscaled = 1.0;
    for (int m = 1; m <= 12; ++m) {
        qscaled *= qscaled0;
        exponent += qscaled / static_cast<double>(m);
        d_exp_d_qs += std::pow(qscaled0, m - 1);
    }
    const double em = std::exp(-exponent);
    q0 = qcut * (1.0 - em);
    dq0_dq = em * d_exp_d_qs;
}

} // namespace

void rvv10_q0(const std::vector<double>& rho, const std::vector<double>& sigma,
              std::vector<double>& q0, double b, double C) {
    const std::size_t n = rho.size();
    if (sigma.size() != n)
        throw std::runtime_error("rVV10 q0: rho/sigma size mismatch");
    q0.resize(n);
    const double qcut = kRvv10QCut;
    const double qmin = kRvv10QMin;
    for (std::size_t i = 0; i < n; ++i) {
        const double mrho = rho[i];
        if (mrho < kRhoCut) {
            q0[i] = qcut;
            continue;
        }
        const double gmod2 = sigma[i];
        const double wp2 = 16.0 * kPi * mrho;
        const double rho2 = mrho * mrho;
        const double rho4 = rho2 * rho2;
        const double wg2 = 4.0 * C * (gmod2 * gmod2) / rho4;
        const double k =
            b * 3.0 * kPi * std::pow(mrho / (9.0 * kPi), 1.0 / 6.0);
        const double w0 = std::sqrt(wg2 + wp2 / 3.0);
        double q = w0 / k;
        double qsat = 0.0, dq0_dq = 0.0;
        soft_saturate(q, qcut, qsat, dq0_dq);
        if (qsat < qmin)
            qsat = qmin;
        q0[i] = qsat;
    }
}

Rvv10Result rvv10_energy(const UniformGrid& grid, const std::vector<double>& rho,
                         const std::vector<double>& sigma, double b, double C,
                         bool want_potential) {
    using namespace rvv10_data;
    const int n0 = grid.n[0];
    const int n1 = grid.n[1];
    const int n2 = grid.n[2];
    const long ntot = static_cast<long>(n0) * n1 * n2;
    if (static_cast<long>(rho.size()) != ntot
        || static_cast<long>(sigma.size()) != ntot)
        throw std::runtime_error("rVV10: density size mismatch vs grid");

    const double qcut = kRvv10QCut;
    const double qmin = kRvv10QMin;
    const double dV = grid.weight();

    // q0 + derivatives (CP2K get_q0_on_grid_rvv10)
    std::vector<double> q0(static_cast<std::size_t>(ntot));
    std::vector<double> dq0_drho(static_cast<std::size_t>(ntot), 0.0);
    std::vector<double> dq0_dgradrho(static_cast<std::size_t>(ntot), 0.0);
    for (long g = 0; g < ntot; ++g) {
        const double mrho = rho[static_cast<std::size_t>(g)];
        const double gmod2 = sigma[static_cast<std::size_t>(g)];
        if (mrho < kRhoCut) {
            q0[static_cast<std::size_t>(g)] = qcut;
            continue;
        }
        const double rho2 = mrho * mrho;
        const double rho4 = rho2 * rho2;
        const double wp2 = 16.0 * kPi * mrho;
        const double wg2 = 4.0 * C * (gmod2 * gmod2) / rho4;
        const double k =
            b * 3.0 * kPi * std::pow(mrho / (9.0 * kPi), 1.0 / 6.0);
        const double w0 = std::sqrt(wg2 + wp2 / 3.0);
        const double q = w0 / k;
        double qsat = 0.0, dq0_dq = 0.0;
        soft_saturate(q, qcut, qsat, dq0_dq);
        if (qsat < qmin)
            qsat = qmin;
        q0[static_cast<std::size_t>(g)] = qsat;

        const double dw0_dn =
            1.0 / (2.0 * w0) * (16.0 / 3.0 * kPi - 4.0 * wg2 / mrho);
        const double dk_dn = k / (6.0 * mrho);
        dq0_drho[static_cast<std::size_t>(g)] =
            dq0_dq * (1.0 / (k * k)) * (dw0_dn * k - dk_dn * w0);
        if (gmod2 > kRhoCut)
            dq0_dgradrho[static_cast<std::size_t>(g)] =
                dq0_dq * (1.0 / (2.0 * k * w0)) * 4.0 * wg2 / gmod2;
    }

    // θ_α = const * ρ^{3/4} * p_α(q0)
    const double rootpi = std::sqrt(kPi);
    const double const_theta =
        1.0 / (3.0 * rootpi * std::pow(b, 1.5) * std::pow(kPi, 0.75));
    // β = (1/32) (3 / b²)^{3/4}
    const double beta = 0.03125 * std::pow(3.0 / (b * b), 0.75);

    const double* qm = qmesh();
    std::vector<double> d2y;
    init_p_spline_d2(qm, kNAlpha, d2y);

    const int n2c = n2 / 2 + 1;
    const long ncplx = static_cast<long>(n0) * n1 * n2c;
    const int nalpha = kNAlpha;

    double* rbuf = static_cast<double*>(
        fftw_malloc(sizeof(double) * static_cast<std::size_t>(ntot * nalpha)));
    fftw_complex* cbuf = static_cast<fftw_complex*>(fftw_malloc(
        sizeof(fftw_complex) * static_cast<std::size_t>(ncplx * nalpha)));
    if (!rbuf || !cbuf)
        throw std::runtime_error("rVV10: FFTW alloc failed");

    std::vector<double> ptmp(static_cast<std::size_t>(nalpha));
    double nelec = 0.0;
    for (long g = 0; g < ntot; ++g) {
        const double rg = rho[static_cast<std::size_t>(g)];
        nelec += rg;
        if (rg < kRhoCut) {
            for (int a = 0; a < nalpha; ++a)
                rbuf[a * ntot + g] = 0.0;
            continue;
        }
        palpha_and_deriv_at_q(qm, d2y, nalpha, q0[static_cast<std::size_t>(g)],
                              ptmp.data(), nullptr);
        const double pref = const_theta * std::pow(rg, 0.75);
        for (int a = 0; a < nalpha; ++a)
            rbuf[a * ntot + g] = pref * ptmp[static_cast<std::size_t>(a)];
    }
    nelec *= dV;

    R2cFft& fft = R2cFft::for_shape(n0, n1, n2);
    for (int a = 0; a < nalpha; ++a) {
        double* rin = rbuf + a * ntot;
        fftw_complex* cout = cbuf + a * ncplx;
        fft.execute_r2c(rin, cout);
    }

    const double two_pi = 2.0 * kPi;
    const double gx_unit = two_pi / grid.L[0];
    const double gy_unit = two_pi / grid.L[1];
    const double gz_unit = two_pi / grid.L[2];

    std::vector<double> kernel_aa(static_cast<std::size_t>(nalpha * nalpha));
    double energy = 0.0;

    for (int i0 = 0; i0 < n0; ++i0) {
        const int m0 = (i0 <= n0 / 2) ? i0 : i0 - n0;
        const double gx = m0 * gx_unit;
        for (int i1 = 0; i1 < n1; ++i1) {
            const int m1 = (i1 <= n1 / 2) ? i1 : i1 - n1;
            const double gy = m1 * gy_unit;
            for (int i2 = 0; i2 < n2c; ++i2) {
                const double gz = i2 * gz_unit;
                const long kindex =
                    (static_cast<long>(i0) * n1 + i1) * n2c + i2;
                const int weight =
                    (i2 == 0 || (n2 % 2 == 0 && i2 == n2c - 1)) ? 1 : 2;
                const double k = std::sqrt(gx * gx + gy * gy + gz * gz);
                interpolate_kernel_linear(k, kernel_aa.data());

                std::vector<std::complex<double>> theta(
                    static_cast<std::size_t>(nalpha));
                for (int a = 0; a < nalpha; ++a) {
                    const fftw_complex& th = cbuf[a * ncplx + kindex];
                    theta[static_cast<std::size_t>(a)] = {th[0], th[1]};
                }
                for (int a1 = 0; a1 < nalpha; ++a1) {
                    std::complex<double> F(0.0, 0.0);
                    for (int a2 = 0; a2 < nalpha; ++a2) {
                        const double phi = kernel_aa[static_cast<std::size_t>(
                            a1 * nalpha + a2)];
                        F += theta[static_cast<std::size_t>(a2)] * phi;
                    }
                    energy += weight
                              * (theta[static_cast<std::size_t>(a1)].real()
                                     * F.real()
                                 + theta[static_cast<std::size_t>(a1)].imag()
                                       * F.imag());
                    if (want_potential) {
                        fftw_complex& u = cbuf[a1 * ncplx + kindex];
                        u[0] = F.real();
                        u[1] = F.imag();
                    }
                }
            }
        }
    }

    // Same FFT convention as DRSLL; rVV10 applies an extra ½ (CP2K).
    energy *= 0.5 * dV / static_cast<double>(ntot);
    const double e_kernel = 0.5 * energy;
    const double e_beta = beta * nelec;

    Rvv10Result res;
    res.e_kernel = e_kernel;
    res.e_beta = e_beta;
    res.energy = e_kernel + e_beta;

    if (!want_potential) {
        fftw_free(rbuf);
        fftw_free(cbuf);
        return res;
    }

    // IFFT u_α → real space (divide by N).
    for (int a = 0; a < nalpha; ++a) {
        fftw_complex* cin = cbuf + a * ncplx;
        double* rout = rbuf + a * ntot;
        fft.execute_c2r(cin, rout);
        const double invN = 1.0 / static_cast<double>(ntot);
        for (long g = 0; g < ntot; ++g)
            rout[g] *= invN;
    }

    const auto gxyz = gradient(grid, rho);
    res.v_r.assign(static_cast<std::size_t>(ntot), 0.0);
    std::array<std::vector<double>, 3> flux;
    for (int d = 0; d < 3; ++d)
        flux[d].assign(static_cast<std::size_t>(ntot), 0.0);

    std::vector<double> p_a(static_cast<std::size_t>(nalpha));
    std::vector<double> dp_a(static_cast<std::size_t>(nalpha));
    for (long g = 0; g < ntot; ++g) {
        const double rg = rho[static_cast<std::size_t>(g)];
        const double qg = q0[static_cast<std::size_t>(g)];
        palpha_and_deriv_at_q(qm, d2y, nalpha, qg, p_a.data(), dp_a.data());
        double pot = 0.0;
        double hpref = 0.0;
        if (rg > kRhoCut) {
            const double tmp_1_2 = std::sqrt(rg);
            const double tmp_1_4 = std::sqrt(tmp_1_2); // ρ^{1/4}
            const double tmp_3_4 = tmp_1_4 * tmp_1_4 * tmp_1_4; // ρ^{3/4}
            for (int a = 0; a < nalpha; ++a) {
                const double u = rbuf[a * ntot + g];
                // CP2K get_potential RVV10 (before ½ scaling):
                pot += u
                       * (const_theta * 0.75 / tmp_1_4
                              * p_a[static_cast<std::size_t>(a)]
                          + const_theta * tmp_3_4
                                * dp_a[static_cast<std::size_t>(a)]
                                * dq0_drho[static_cast<std::size_t>(g)]);
                if (qg != qcut)
                    hpref += u * const_theta * tmp_3_4
                             * dp_a[static_cast<std::size_t>(a)]
                             * dq0_dgradrho[static_cast<std::size_t>(g)];
            }
        }
        // CP2K: potential = 0.5 * pot + beta; hpot = 0.5 * hpref
        res.v_r[static_cast<std::size_t>(g)] = 0.5 * pot + beta;
        hpref *= 0.5;
        flux[0][static_cast<std::size_t>(g)] =
            hpref * gxyz[0][static_cast<std::size_t>(g)];
        flux[1][static_cast<std::size_t>(g)] =
            hpref * gxyz[1][static_cast<std::size_t>(g)];
        flux[2][static_cast<std::size_t>(g)] =
            hpref * gxyz[2][static_cast<std::size_t>(g)];
    }

    const std::vector<double> divflux = divergence(grid, flux);
    for (long g = 0; g < ntot; ++g)
        res.v_r[static_cast<std::size_t>(g)] -=
            divflux[static_cast<std::size_t>(g)];

    fftw_free(rbuf);
    fftw_free(cbuf);
    return res;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
