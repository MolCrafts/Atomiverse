/// @file cpu/pw/internal/density_builder.cpp
/// @brief Build charge density from wavefunctions using FFT + Pulay mixing.
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

void DensityBuilder::prepare(const PWWorkspace& ws) {
    if (ws.nr <= 0)
        return;
    rho_r_accum_.resize(static_cast<std::size_t>(ws.nr));
    psi_cc_.resize(static_cast<std::size_t>(ws.nr));
    rho_g_full_.resize(static_cast<std::size_t>(ws.nc));
}

void DensityBuilder::reset(PWWorkspace& ws) {
    ws.pulay_res.clear();
    ws.pulay_rho.clear();
}

void DensityBuilder::begin(const PWWorkspace& ws) {
    auto* acc = rho_r_accum_.data();
    for (int i = 0; i < ws.nr; ++i)
        acc[i] = AccReal(0);
}

void DensityBuilder::accumulate_kpoint(const PWWorkspace& ws, AccReal weight,
                                       int n_bands, AccReal spin_degen) {
    if (ws.nr <= 0 || !ws.fft)
        return;

    int nb = std::min(n_bands, static_cast<int>(ws.eigenvalues.size()));
    int ng_k = ws.n_g;
    int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
    auto& gvk = *ws.gvec;
    auto* psi_cc = psi_cc_.data();
    auto* acc = rho_r_accum_.data();

    for (int b = 0; b < nb; ++b) {
        for (int i = 0; i < ws.nr; ++i)
            psi_cc[i][0] = psi_cc[i][1] = 0;
        for (int ig = 0; ig < ng_k; ++ig) {
            int ix = ((gvk.h[ig] % nx) + nx) % nx;
            int iy = ((gvk.k[ig] % ny) + ny) % ny;
            int iz = ((gvk.l[ig] % nz) + nz) % nz;
            int idx = ix + nx * (iy + ny * iz);
            if (idx >= 0 && idx < ws.nr) {
                auto val = ws.wfc(b, ig);
                psi_cc[idx][0] = static_cast<double>(val.real());
                psi_cc[idx][1] = static_cast<double>(val.imag());
            }
        }
        ws.fft->c2c_bwd(psi_cc);
        for (int i = 0; i < ws.nr; ++i) {
            AccReal psi_sq = static_cast<AccReal>(
                psi_cc[i][0] * psi_cc[i][0] + psi_cc[i][1] * psi_cc[i][1]);
            // ψ_phys(r) = ψ_fft(r)/√Ω, so |ψ|²_phys = |ψ_fft|²/Ω
            acc[i] += weight * spin_degen * psi_sq / ws.omega;
        }
    }
}

void DensityBuilder::accumulate_kpoint(const PWWorkspace& ws, AccReal weight,
                                       int n_bands, const AccReal* occ) {
    if (ws.nr <= 0 || !ws.fft)
        return;

    int nb = std::min(n_bands, static_cast<int>(ws.eigenvalues.size()));
    if (ws.wfc.shape().size() >= 1)
        nb = std::min(nb, static_cast<int>(ws.wfc.shape(0)));
    int ng_k = ws.n_g;
    int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
    auto& gvk = *ws.gvec;
    auto* psi_cc = psi_cc_.data();
    auto* acc = rho_r_accum_.data();

    for (int b = 0; b < nb; ++b) {
        const AccReal fb = occ ? occ[b] : AccReal(2);
        if (fb == AccReal(0))
            continue;
        for (int i = 0; i < ws.nr; ++i)
            psi_cc[i][0] = psi_cc[i][1] = 0;
        for (int ig = 0; ig < ng_k; ++ig) {
            int ix = ((gvk.h[ig] % nx) + nx) % nx;
            int iy = ((gvk.k[ig] % ny) + ny) % ny;
            int iz = ((gvk.l[ig] % nz) + nz) % nz;
            int idx = ix + nx * (iy + ny * iz);
            if (idx >= 0 && idx < ws.nr) {
                auto val = ws.wfc(b, ig);
                psi_cc[idx][0] = static_cast<double>(val.real());
                psi_cc[idx][1] = static_cast<double>(val.imag());
            }
        }
        ws.fft->c2c_bwd(psi_cc);
        for (int i = 0; i < ws.nr; ++i) {
            AccReal psi_sq = static_cast<AccReal>(
                psi_cc[i][0] * psi_cc[i][0] + psi_cc[i][1] * psi_cc[i][1]);
            // ψ_phys(r) = ψ_fft(r)/√Ω, so |ψ|²_phys = |ψ_fft|²/Ω
            acc[i] += weight * fb * psi_sq / ws.omega;
        }
    }
}

void DensityBuilder::finish(PWWorkspace& ws) {
    if (ws.nr <= 0 || !ws.fft)
        return;

    int nc = ws.nc;
    auto* rho_g_full = rho_g_full_.data();
    ws.fft->r2c(rho_r_accum_.data(), rho_g_full);

    for (int i = 0; i < ws.n_g; ++i)
        ws.rho_G_old[i] = ws.rho_G[i];

    // Extract ρ(G) at the density G-sphere from the half-complex grid.
    int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz, nxc = nx / 2 + 1;
    auto& gv = *ws.gvec;
    AccReal inv_nr = AccReal(1) / AccReal(ws.nr);
    for (int ig = 0; ig < ws.n_g; ++ig) {
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
            ws.rho_G[ig] = std::complex<AccReal>(
                static_cast<AccReal>(rho_g_full[idx][0]) * inv_nr,
                static_cast<AccReal>(conj ? -rho_g_full[idx][1]
                                          : rho_g_full[idx][1])
                    * inv_nr);
        } else
            ws.rho_G[ig] = std::complex<AccReal>(0, 0);
    }

    // ── Pulay/DIIS mixing — the shared CPU/CUDA implementation in
    // internal::pulay_mix (pw_setup.cpp), driven by the LIVE workspace
    // beta / history depth. This builder must not write them: it is
    // constructed once per method object while `mix_beta` / `pulay_depth`
    // are public members a caller may change between run()s, so a
    // constructor snapshot here silently defeated PW::run's assignment.
    internal::pulay_mix(ws);
}

void DensityBuilder::extract(const PWWorkspace& ws,
                             xt::xarray<std::complex<AccReal>>& rho_out) {
    if (ws.nr <= 0 || !ws.fft)
        return;

    int nc = ws.nc;
    auto* rho_g_full = rho_g_full_.data();
    ws.fft->r2c(rho_r_accum_.data(), rho_g_full);

    if (static_cast<int>(rho_out.size()) != ws.n_g)
        rho_out.resize({static_cast<std::size_t>(ws.n_g)});

    int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz, nxc = nx / 2 + 1;
    auto& gv = *ws.gvec;
    AccReal inv_nr = AccReal(1) / AccReal(ws.nr);
    for (int ig = 0; ig < ws.n_g; ++ig) {
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
            rho_out[ig] = std::complex<AccReal>(
                static_cast<AccReal>(rho_g_full[idx][0]) * inv_nr,
                static_cast<AccReal>(conj ? -rho_g_full[idx][1]
                                          : rho_g_full[idx][1])
                    * inv_nr);
        } else
            rho_out[ig] = std::complex<AccReal>(0, 0);
    }
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
