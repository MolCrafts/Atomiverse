/// @file cpu/pw/internal/veff_builder.cpp
/// @brief Assemble V_eff(G) = V_loc + V_H + V_xc, then FFT to real-space V_eff(r).
#include "atomiverse/cpu/pw/internal/veff_builder.h"

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

void VeffBuilder::prepare(const PWWorkspace& ws) {
    if (ws.nr <= 0)
        return;
    vg_.resize(static_cast<std::size_t>(ws.nc));
}

void VeffBuilder::step(PWWorkspace& ws) {
    int ng = ws.n_g;

    // Assemble in G-space
    for (int i = 0; i < ng; ++i)
        ws.V_eff_G[i] = ws.V_loc_G[i] + ws.V_H_G[i] + ws.V_xc_G[i];

    // FFT V_eff(G) → V_eff(r) for use in FFT-based H·ψ
    if (ws.fft && ws.nr > 0) {
        int nr = ws.nr, nc = ws.nc;
        int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
        int nxc = nx / 2 + 1;
        auto& gv = *ws.gvec;

        // Pad V_eff(G) to half-complex FFT grid
        auto* vg = vg_.data();
        for (int i = 0; i < nc; ++i)
            vg[i][0] = vg[i][1] = 0;

        for (int ig = 0; ig < ng; ++ig) {
            int h = gv.h[ig], k = gv.k[ig], l = gv.l[ig];
            int kx = ((h % nx) + nx) % nx, ky = ((k % ny) + ny) % ny,
                kz = ((l % nz) + nz) % nz;
            bool conj = false;
            if (kx > nx / 2) {
                kx = nx - kx;
                ky = (ny - ky) % ny;
                kz = (nz - kz) % nz;
                conj = true;
            }
            int idx = kx + nxc * (ky + ny * kz);
            if (idx >= 0 && idx < nc) {
                vg[idx][0] = static_cast<double>(ws.V_eff_G[ig].real());
                vg[idx][1] = static_cast<double>(conj ? -ws.V_eff_G[ig].imag()
                                                      : ws.V_eff_G[ig].imag());
            }
        }

        // Resize V_eff_r if needed
        if (ws.V_eff_r.size() != static_cast<std::size_t>(nr))
            ws.V_eff_r.resize({static_cast<std::size_t>(nr)});

        // c2r: V_eff(G) → V_eff(r). FFTW does NOT normalize.
        // V_eff(r) = Σ_G V_eff(G) exp(iGr)
        ws.fft->c2r(vg, ws.V_eff_r.data());
    }
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
