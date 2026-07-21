/// @file cpu/pw/internal/xc_builder.cpp
/// @brief XC potential: compute V_xc(r) from ρ(r), FFT to G-space.
#include "atomiverse/cpu/pw/internal/xc_builder.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include <cmath>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

void XCBuilder::prepare(const PWWorkspace& ws) {
    if (ws.nr <= 0)
        return;
    rho_g_half_.resize(static_cast<std::size_t>(ws.nc));
    rho_r_.resize(static_cast<std::size_t>(ws.nr));
    vxc_g_half_.resize(static_cast<std::size_t>(ws.nc));
}

void XCBuilder::step(PWWorkspace& ws) {
    int ng = ws.n_g;

    if (!ws.fft || ws.nr <= 0) {
        for (int i = 0; i < ng; ++i)
            ws.V_xc_G[i] = std::complex<AccReal>(0, 0);
        return;
    }

    int nr = ws.nr;
    int nc = ws.nc;
    int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
    int nxc = nx / 2 + 1;
    auto& gv = *ws.gvec;

    // Step 1: FFT ρ(G) → ρ(r)
    // Pad ρ(G) to half-complex FFT grid, then c2r
    auto* rho_g_half = rho_g_half_.data();
    auto* rho_r = rho_r_.data();

    for (int i = 0; i < nc; ++i)
        rho_g_half[i][0] = rho_g_half[i][1] = 0;

    for (int ig = 0; ig < ng; ++ig) {
        int h = gv.h[ig], k = gv.k[ig], l = gv.l[ig];
        int kx = ((h % nx) + nx) % nx;
        int ky = ((k % ny) + ny) % ny;
        int kz = ((l % nz) + nz) % nz;
        bool conj = false;
        if (kx > nx / 2) {
            kx = nx - kx;
            ky = (ny - ky) % ny;
            kz = (nz - kz) % nz;
            conj = true;
        }
        int idx = kx + nxc * (ky + ny * kz);
        if (idx >= 0 && idx < nc) {
            rho_g_half[idx][0] = static_cast<double>(ws.rho_G[ig].real());
            rho_g_half[idx][1] = static_cast<double>(
                conj ? -ws.rho_G[ig].imag() : ws.rho_G[ig].imag());
        }
    }

    // FFTW does NOT normalize: c2r output = Σ_G input[G] exp(iGr)
    // ρ(r) = Σ_G ρ(G) exp(iGr), so no scaling needed.
    ws.fft->c2r(rho_g_half, rho_r);

    // Step 2: Compute V_xc(r) and E_xc from ρ(r)
    if (ws.V_xc_r.size() != static_cast<std::size_t>(nr))
        ws.V_xc_r.resize({static_cast<std::size_t>(nr)});
    if (ws.exc_r.size() != static_cast<std::size_t>(nr))
        ws.exc_r.resize({static_cast<std::size_t>(nr)});

    AccReal dV = ws.omega / static_cast<AccReal>(nr); // volume element
    AccReal E_xc = 0;

    for (int i = 0; i < nr; ++i) {
        AccReal rho = rho_r[i];
        if (rho < AccReal(1e-12)) {
            ws.V_xc_r[i] = 0;
            ws.exc_r[i] = 0;
        } else {
            auto out = cpu::scf::evaluate_xc(
                cpu::scf::XCType::PBE, rho / AccReal(2), rho / AccReal(2));
            ws.V_xc_r[i] = static_cast<AccReal>(out.vrho_alpha);
            ws.exc_r[i] = static_cast<AccReal>(out.exc);
            // E_xc = ∫ ρ(r) ε_xc(r) d³r = Σ ρ_i * exc_i * dV
            E_xc += rho * static_cast<AccReal>(out.exc) * dV;
        }
    }
    ws.E_xc = E_xc;

    // Step 3: FFT V_xc(r) → V_xc(G)
    // r2c: out[G] = Σ_r in[r] exp(-iGr)  (unnormalized)
    // V_xc(G) = (1/N) Σ_r V_xc(r) exp(-iGr) = (1/N) * out[G]
    auto* vxc_g_half = vxc_g_half_.data();
    ws.fft->r2c(ws.V_xc_r.data(), vxc_g_half);

    AccReal inv_nr = AccReal(1) / static_cast<AccReal>(nr);
    for (int ig = 0; ig < ng; ++ig) {
        int h = gv.h[ig], k = gv.k[ig], l = gv.l[ig];
        int kx = ((h % nx) + nx) % nx;
        int ky = ((k % ny) + ny) % ny;
        int kz = ((l % nz) + nz) % nz;
        bool conj = false;
        if (kx > nx / 2) {
            kx = nx - kx;
            ky = (ny - ky) % ny;
            kz = (nz - kz) % nz;
            conj = true;
        }
        int idx = kx + nxc * (ky + ny * kz);
        if (idx >= 0 && idx < nc) {
            ws.V_xc_G[ig] = std::complex<AccReal>(
                static_cast<AccReal>(vxc_g_half[idx][0]) * inv_nr,
                static_cast<AccReal>(conj ? -vxc_g_half[idx][1]
                                          : vxc_g_half[idx][1])
                    * inv_nr);
        } else {
            ws.V_xc_G[ig] = std::complex<AccReal>(0, 0);
        }
    }

    // Ensure V_xc(G=0) is real
    ws.V_xc_G[0] = std::complex<AccReal>(ws.V_xc_G[0].real(), AccReal(0));
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
