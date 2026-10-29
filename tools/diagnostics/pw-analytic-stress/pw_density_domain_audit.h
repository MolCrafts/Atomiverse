#pragma once
// Observation only: reconstruct the actual two-channel spectral density and
// gradients; report clipped/floor/endpoint branches without projecting input.
#include "atomiverse/cpu/planewave/internal/fft.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include <algorithm>
#include <array>
#include <complex>
#include <cstdio>
#include <limits>
#include <vector>
namespace pw_density_domain_audit {
template <class Channels>
void inspect(const char* label, const ATV::cpu::pw::internal::PWProblem& p,
             const Channels& rho) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    const auto& g = p.gv0;
    int nx = 3 * p.ws.fft_nx, ny = 3 * p.ws.fft_ny, nz = 3 * p.ws.fft_nz,
        nxc = nx / 2 + 1;
    FFTPlan3D fft(nx, ny, nz);
    int N = fft.nr(), nc = fft.nc();
    FFTWBuffer<fftw_complex> half(nc);
    std::vector<double> x[2][4];
    for (int s = 0; s < 2; ++s)
        for (int d = 0; d < 4; ++d) {
            x[s][d].resize(N);
            for (int j = 0; j < nc; ++j)
                half.data()[j][0] = half.data()[j][1] = 0;
            for (int i = 0; i < g.count(); ++i) {
                int a = (g.h[i] % nx + nx) % nx, b = (g.k[i] % ny + ny) % ny,
                    c = (g.l[i] % nz + nz) % nz;
                bool conjugate = a > nx / 2;
                if (conjugate) {
                    a = nx - a;
                    b = (ny - b) % ny;
                    c = (nz - c) % nz;
                }
                std::complex<double> v = rho[s][i];
                if (d)
                    v *= std::complex<double>(0, g.g[i][d - 1]);
                if (conjugate)
                    v = std::conj(v);
                int k = a + nxc * (b + ny * c);
                half.data()[k][0] = v.real();
                half.data()[k][1] = v.imag();
            }
            fft.c2r(half.data(), x[s][d].data());
        }
    double mina = 1, minb = 1, minabs_a = 1, minabs_b = 1, minaa = 1, minbb = 1,
           minab = 1, minsum = 1, maxab = -1;
    std::size_t clipa = 0, clipb = 0, floora = 0, floorb = 0, vacuum = 0,
                outside = 0, endpoint = 0, floorcross = 0, cutcross = 0;
    for (int i = 0; i < N; ++i) {
        double a = x[0][0][i], b = x[1][0][i], aa = 0, ab = 0, bb = 0;
        for (int d = 1; d <= 3; ++d) {
            aa += x[0][d][i] * x[0][d][i];
            ab += x[0][d][i] * x[1][d][i];
            bb += x[1][d][i] * x[1][d][i];
        }
        mina = std::min(mina, a);
        minb = std::min(minb, b);
        minabs_a = std::min(minabs_a, std::abs(a));
        minabs_b = std::min(minabs_b, std::abs(b));
        minaa = std::min(minaa, aa);
        minbb = std::min(minbb, bb);
        minab = std::min(minab, ab);
        maxab = std::max(maxab, ab);
        minsum = std::min(minsum, aa + 2 * ab + bb);
        clipa += a <= 0;
        clipb += b <= 0;
        floora += a > 0 && a < 1e-15;
        floorb += b > 0 && b < 1e-15;
        double n = std::max(a, 0.) + std::max(b, 0.);
        vacuum += n < 1e-12;
        outside += a + b > 1e-12 && std::abs((a - b) / (a + b)) > 1;
        endpoint += n >= 1e-12 && (a <= 0 || b <= 0);
        for (double r : {a, b})
            floorcross += std::abs(r - 1e-15) < 3.1e-5 * 1e-15;
        cutcross += std::abs(n - 1e-12) < 3.1e-5 * 1e-12;
    }
    std::printf(
        "PW_DENSITY_DOMAIN %s points=%d quad=%d,%d,%d min_alpha=%.17g "
        "min_beta=%.17g min_abs_alpha=%.17g min_abs_beta=%.17g "
        "clipped_alpha=%zu clipped_beta=%zu positive_spin_floor_alpha=%zu "
        "positive_spin_floor_beta=%zu vacuum=%zu raw_abs_zeta_gt1=%zu "
        "clipped_zeta_endpoint=%zu sigma_aa_min=%.17g sigma_bb_min=%.17g "
        "sigma_ab_min=%.17g sigma_ab_max=%.17g sigma_total_min=%.17g "
        "frozen_h3e5_floor_cross_candidates=%zu "
        "frozen_h3e5_cutoff_cross_candidates=%zu audit_SCF_calls=0\n",
        label, N, nx, ny, nz, mina, minb, minabs_a, minabs_b, clipa, clipb,
        floora, floorb, vacuum, outside, endpoint, minaa, minbb, minab, maxab,
        minsum, floorcross, cutcross);
}
} // namespace pw_density_domain_audit
