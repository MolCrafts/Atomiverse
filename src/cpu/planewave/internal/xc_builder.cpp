/// @file cpu/planewave/internal/xc_builder.cpp
/// @brief XC potential: LDA and true GGA (spectral σ) on the PW FFT grid.
///
/// Closed-shell GGA potential (same collapse as GPW `xc_grid_pbe`):
///   v_xc = vrho_a - div[ (vsigma_aa + 1/2 vsigma_ab) grad rho ]
/// with rho_a = rho_b = rho/2, sigma_aa = sigma_ab = sigma_bb = |grad rho|^2 / 4.
///
/// Collinear spin-GGA (same as GPW `xc_grid_gga_spin`):
///   v_σ = vrho_σ − div[ 2 vσ_σσ ∇ρ_σ + vσ_αβ ∇ρ_σ̄ ].
#include "atomiverse/cpu/planewave/internal/xc_builder.h"

#include <cmath>
#include <complex>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {

namespace {

/// @brief Map a G-sphere mode into the r2c half-complex FFT layout.
/// @return flat index in [0, nc); sets `conj` when the sphere G was folded.
inline int pack_half_index(int h, int k, int l, int nx, int ny, int nz, int nxc,
                           bool& conj) {
    int kx = ((h % nx) + nx) % nx;
    int ky = ((k % ny) + ny) % ny;
    int kz = ((l % nz) + nz) % nz;
    conj = false;
    if (kx > nx / 2) {
        kx = nx - kx;
        ky = (ny - ky) % ny;
        kz = (nz - kz) % nz;
        conj = true;
    }
    return kx + nxc * (ky + ny * kz);
}

inline void zero_half(fftw_complex* buf, int nc) {
    for (int i = 0; i < nc; ++i)
        buf[i][0] = buf[i][1] = 0.0;
}

/// @brief Pack sphere values into half-complex layout (Hermitian fold).
template <typename F>
void pack_sphere_to_half(const GVectorList& gv, int ng, int nx, int ny, int nz,
                         int nxc, int nc, fftw_complex* half, F&& value_at_ig) {
    zero_half(half, nc);
    for (int ig = 0; ig < ng; ++ig) {
        bool conj = false;
        const int idx = pack_half_index(gv.h[ig], gv.k[ig], gv.l[ig], nx, ny,
                                        nz, nxc, conj);
        if (idx < 0 || idx >= nc)
            continue;
        const std::complex<double> v = value_at_ig(ig);
        half[idx][0] = v.real();
        half[idx][1] = conj ? -v.imag() : v.imag();
    }
}

inline void unpack_half_to_sphere(const GVectorList& gv, int ng, int nx, int ny,
                                  int nz, int nxc, int nc,
                                  const fftw_complex* half, AccReal inv_nr,
                                  xt::xarray<std::complex<AccReal>>& out) {
    for (int ig = 0; ig < ng; ++ig) {
        bool conj = false;
        const int idx = pack_half_index(gv.h[ig], gv.k[ig], gv.l[ig], nx, ny,
                                        nz, nxc, conj);
        if (idx >= 0 && idx < nc) {
            out[ig] = std::complex<AccReal>(
                static_cast<AccReal>(half[idx][0]) * inv_nr,
                static_cast<AccReal>(conj ? -half[idx][1] : half[idx][1])
                    * inv_nr);
        } else {
            out[ig] = std::complex<AccReal>(0, 0);
        }
    }
}

} // namespace

void XCBuilder::prepare(const PWWorkspace& ws) {
    if (ws.nr <= 0)
        return;
    const auto nr = static_cast<std::size_t>(ws.nr);
    const auto nc = static_cast<std::size_t>(ws.nc);
    rho_g_half_.resize(nc);
    rho_r_.resize(nr);
    vxc_g_half_.resize(nc);
    grad_x_.resize(nr);
    grad_y_.resize(nr);
    grad_z_.resize(nr);
    flux_x_.resize(nr);
    flux_y_.resize(nr);
    flux_z_.resize(nr);
    div_r_.resize(nr);
    work_g_half_.resize(nc);
    if (ws.nspin == 2) {
        rho_up_r_.resize(nr);
        rho_dn_r_.resize(nr);
        vxc_up_r_.resize(nr);
        vxc_dn_r_.resize(nr);
        // Spin-GGA channel-β gradients + fluxes (α reuses grad_*/flux_*).
        grad_dn_x_.resize(nr);
        grad_dn_y_.resize(nr);
        grad_dn_z_.resize(nr);
        flux_dn_x_.resize(nr);
        flux_dn_y_.resize(nr);
        flux_dn_z_.resize(nr);
        div_dn_r_.resize(nr);
    }
}

void XCBuilder::step(PWWorkspace& ws, cpu::scf::XCType xc) {
    const int ng = ws.n_g;

    if (!ws.fft || ws.nr <= 0) {
        for (int i = 0; i < ng; ++i)
            ws.V_xc_G[i] = std::complex<AccReal>(0, 0);
        return;
    }

    const int nr = ws.nr;
    const int nc = ws.nc;
    const int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
    const int nxc = nx / 2 + 1;
    auto& gv = *ws.gvec;
    const bool gga = cpu::scf::is_gga(xc);

    // ── 1. FFT ρ(G) → ρ(r) ──────────────────────────────────────────
    auto* rho_g_half = rho_g_half_.data();
    auto* rho_r = rho_r_.data();

    pack_sphere_to_half(gv, ng, nx, ny, nz, nxc, nc, rho_g_half, [&](int ig) {
        return std::complex<double>(static_cast<double>(ws.rho_G[ig].real()),
                                    static_cast<double>(ws.rho_G[ig].imag()));
    });
    // FFTW does NOT normalize: c2r output = Σ_G input[G] exp(iGr)
    ws.fft->c2r(rho_g_half, rho_r);

    if (ws.V_xc_r.size() != static_cast<std::size_t>(nr))
        ws.V_xc_r.resize({static_cast<std::size_t>(nr)});
    if (ws.exc_r.size() != static_cast<std::size_t>(nr))
        ws.exc_r.resize({static_cast<std::size_t>(nr)});

    // ── 1b. GGA: spectral ∇ρ from i G ρ(G) ──────────────────────────
    if (gga) {
        auto* work = work_g_half_.data();
        AccReal* grad_comp[3] = {grad_x_.data(), grad_y_.data(),
                                 grad_z_.data()};
        for (int d = 0; d < 3; ++d) {
            pack_sphere_to_half(gv, ng, nx, ny, nz, nxc, nc, work, [&](int ig) {
                const double gd = static_cast<double>(gv.g[ig][d]);
                const std::complex<double> rho(
                    static_cast<double>(ws.rho_G[ig].real()),
                    static_cast<double>(ws.rho_G[ig].imag()));
                return std::complex<double>(0.0, gd) * rho;
            });
            ws.fft->c2r(work, grad_comp[d]);
        }
    }

    // ── 2. Pointwise XC (+ flux for GGA) ─────────────────────────────
    const AccReal dV = ws.omega / static_cast<AccReal>(nr);
    AccReal E_xc = 0;
    auto* fx = flux_x_.data();
    auto* fy = flux_y_.data();
    auto* fz = flux_z_.data();

    for (int i = 0; i < nr; ++i) {
        const AccReal rho = rho_r[i];
        if (rho < AccReal(1e-12)) {
            ws.V_xc_r[i] = 0;
            ws.exc_r[i] = 0;
            if (gga)
                fx[i] = fy[i] = fz[i] = AccReal(0);
            continue;
        }

        double sigma_aa = 0, sigma_ab = 0, sigma_bb = 0;
        double gx = 0, gy = 0, gz = 0;
        if (gga) {
            gx = static_cast<double>(grad_x_.data()[i]);
            gy = static_cast<double>(grad_y_.data()[i]);
            gz = static_cast<double>(grad_z_.data()[i]);
            const double sigma = gx * gx + gy * gy + gz * gz;
            // Spin-unpolarized: σ_aa = σ_ab = σ_bb = |∇ρ|²/4
            sigma_aa = sigma_ab = sigma_bb = 0.25 * sigma;
        }

        const auto out = cpu::scf::evaluate_xc(
            xc, static_cast<double>(rho) / 2.0, static_cast<double>(rho) / 2.0,
            sigma_aa, sigma_ab, sigma_bb);
        ws.V_xc_r[i] = static_cast<AccReal>(out.vrho_alpha);
        ws.exc_r[i] = static_cast<AccReal>(out.exc);
        E_xc += rho * static_cast<AccReal>(out.exc) * dV;

        if (gga) {
            const double vsig_eff = out.vsigma_aa + 0.5 * out.vsigma_ab;
            fx[i] = static_cast<AccReal>(vsig_eff * gx);
            fy[i] = static_cast<AccReal>(vsig_eff * gy);
            fz[i] = static_cast<AccReal>(vsig_eff * gz);
        }
    }
    ws.E_xc = E_xc;

    // ── 2b. GGA: v_xc -= div(flux) ───────────────────────────────────
    // div flux = Σ_d IFFT( i G_d · FFT(flux_d) ) with FFT normalized 1/N.
    if (gga) {
        auto* work = work_g_half_.data();
        const AccReal inv_nr = AccReal(1) / static_cast<AccReal>(nr);
        AccReal* flux_comp[3] = {fx, fy, fz};

        // Build div(G) on the density sphere (reuse V_xc_G as scratch).
        for (int ig = 0; ig < ng; ++ig)
            ws.V_xc_G[ig] = std::complex<AccReal>(0, 0);

        for (int d = 0; d < 3; ++d) {
            ws.fft->r2c(flux_comp[d], work);
            for (int ig = 0; ig < ng; ++ig) {
                bool conj = false;
                const int idx = pack_half_index(gv.h[ig], gv.k[ig], gv.l[ig],
                                                nx, ny, nz, nxc, conj);
                if (idx < 0 || idx >= nc)
                    continue;
                const AccReal gd = gv.g[ig][d];
                const AccReal re = static_cast<AccReal>(work[idx][0]) * inv_nr;
                const AccReal im =
                    static_cast<AccReal>(conj ? -work[idx][1] : work[idx][1])
                    * inv_nr;
                // i * G_d * flux_d(G) = (-G_d im, G_d re)
                ws.V_xc_G[ig] += std::complex<AccReal>(-gd * im, gd * re);
            }
        }

        pack_sphere_to_half(gv, ng, nx, ny, nz, nxc, nc, work, [&](int ig) {
            return std::complex<double>(
                static_cast<double>(ws.V_xc_G[ig].real()),
                static_cast<double>(ws.V_xc_G[ig].imag()));
        });
        ws.fft->c2r(work, div_r_.data());
        for (int i = 0; i < nr; ++i)
            ws.V_xc_r[i] -= div_r_.data()[i];
    }

    // ── 3. FFT V_xc(r) → V_xc(G) ─────────────────────────────────────
    auto* vxc_g_half = vxc_g_half_.data();
    ws.fft->r2c(ws.V_xc_r.data(), vxc_g_half);

    const AccReal inv_nr = AccReal(1) / static_cast<AccReal>(nr);
    unpack_half_to_sphere(gv, ng, nx, ny, nz, nxc, nc, vxc_g_half, inv_nr,
                          ws.V_xc_G);

    ws.V_xc_G[0] = std::complex<AccReal>(ws.V_xc_G[0].real(), AccReal(0));
}

void fft_rho_g_to_r(const PWWorkspace& ws,
                    const xt::xarray<std::complex<AccReal>>& rho_G,
                    fftw_complex* rho_g_half, AccReal* rho_r) {
    int ng = ws.n_g;
    int nc = ws.nc;
    int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
    int nxc = nx / 2 + 1;
    auto& gv = *ws.gvec;
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
            rho_g_half[idx][0] = static_cast<double>(rho_G[ig].real());
            rho_g_half[idx][1] = static_cast<double>(conj ? -rho_G[ig].imag()
                                                          : rho_G[ig].imag());
        }
    }
    ws.fft->c2r(rho_g_half, rho_r);
}

void fft_vxc_r_to_g(const PWWorkspace& ws, AccReal* vxc_r,
                    fftw_complex* vxc_g_half,
                    xt::xarray<std::complex<AccReal>>& V_xc_G) {
    int ng = ws.n_g;
    int nc = ws.nc;
    int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
    int nxc = nx / 2 + 1;
    auto& gv = *ws.gvec;
    ws.fft->r2c(vxc_r, vxc_g_half);
    AccReal inv_nr = AccReal(1) / static_cast<AccReal>(ws.nr);
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
            V_xc_G[ig] = std::complex<AccReal>(
                static_cast<AccReal>(vxc_g_half[idx][0]) * inv_nr,
                static_cast<AccReal>(conj ? -vxc_g_half[idx][1]
                                          : vxc_g_half[idx][1])
                    * inv_nr);
        } else {
            V_xc_G[ig] = std::complex<AccReal>(0, 0);
        }
    }
    V_xc_G[0] = std::complex<AccReal>(V_xc_G[0].real(), AccReal(0));
}

void XCBuilder::step_spin(PWWorkspace& ws, cpu::scf::XCType xc) {
    int ng = ws.n_g;
    if (!ws.fft || ws.nr <= 0 || ws.nspin != 2 || ws.rho_spin_G.size() != 2) {
        for (int i = 0; i < ng; ++i)
            ws.V_xc_G[i] = std::complex<AccReal>(0, 0);
        ws.E_xc = 0;
        ws.e_xc_int_spin = 0;
        return;
    }

    const int nr = ws.nr;
    const int nc = ws.nc;
    const int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
    const int nxc = nx / 2 + 1;
    auto& gv = *ws.gvec;
    const bool gga = cpu::scf::is_gga(xc);

    auto* rho_g_half = rho_g_half_.data();
    auto* rho_up = rho_up_r_.data();
    auto* rho_dn = rho_dn_r_.data();
    auto* vxc_up = vxc_up_r_.data();
    auto* vxc_dn = vxc_dn_r_.data();

    fft_rho_g_to_r(ws, ws.rho_spin_G[0], rho_g_half, rho_up);
    fft_rho_g_to_r(ws, ws.rho_spin_G[1], rho_g_half, rho_dn);

    if (ws.V_xc_spin_G.size() != 2) {
        ws.V_xc_spin_G.resize(2);
        ws.V_xc_spin_G[0].resize({static_cast<std::size_t>(ng)});
        ws.V_xc_spin_G[1].resize({static_cast<std::size_t>(ng)});
    }
    for (int s = 0; s < 2; ++s) {
        if (ws.V_xc_spin_r[s].size() != static_cast<std::size_t>(nr))
            ws.V_xc_spin_r[s].resize({static_cast<std::size_t>(nr)});
    }
    if (ws.exc_r.size() != static_cast<std::size_t>(nr))
        ws.exc_r.resize({static_cast<std::size_t>(nr)});

    // ── Spectral ∇ρ_σ from i G ρ_σ(G) (true spin-GGA) ────────────────
    // α → grad_*; β → grad_dn_*.
    if (gga) {
        auto* work = work_g_half_.data();
        AccReal* g_up[3] = {grad_x_.data(), grad_y_.data(), grad_z_.data()};
        AccReal* g_dn[3] = {grad_dn_x_.data(), grad_dn_y_.data(),
                            grad_dn_z_.data()};
        for (int s = 0; s < 2; ++s) {
            AccReal** gcomp = (s == 0) ? g_up : g_dn;
            const auto& rhoG = ws.rho_spin_G[s];
            for (int d = 0; d < 3; ++d) {
                pack_sphere_to_half(
                    gv, ng, nx, ny, nz, nxc, nc, work, [&](int ig) {
                        const double gd = static_cast<double>(gv.g[ig][d]);
                        const std::complex<double> rho(
                            static_cast<double>(rhoG[ig].real()),
                            static_cast<double>(rhoG[ig].imag()));
                        return std::complex<double>(0.0, gd) * rho;
                    });
                ws.fft->c2r(work, gcomp[d]);
            }
        }
    }

    AccReal dV = ws.omega / static_cast<AccReal>(nr);
    AccReal E_xc = 0;
    auto* fx_up = flux_x_.data();
    auto* fy_up = flux_y_.data();
    auto* fz_up = flux_z_.data();
    auto* fx_dn = flux_dn_x_.data();
    auto* fy_dn = flux_dn_y_.data();
    auto* fz_dn = flux_dn_z_.data();

    for (int i = 0; i < nr; ++i) {
        AccReal ra = rho_up[i];
        AccReal rb = rho_dn[i];
        // Floor negatives from FFT ringing so LSDA/GGA stay defined.
        if (ra < AccReal(0))
            ra = AccReal(0);
        if (rb < AccReal(0))
            rb = AccReal(0);
        AccReal rtot = ra + rb;
        if (rtot < AccReal(1e-12)) {
            vxc_up[i] = 0;
            vxc_dn[i] = 0;
            ws.V_xc_spin_r[0][i] = 0;
            ws.V_xc_spin_r[1][i] = 0;
            ws.exc_r[i] = 0;
            if (gga) {
                fx_up[i] = fy_up[i] = fz_up[i] = AccReal(0);
                fx_dn[i] = fy_dn[i] = fz_dn[i] = AccReal(0);
            }
            continue;
        }

        double saa = 0, sab = 0, sbb = 0;
        double gax = 0, gay = 0, gaz = 0;
        double gbx = 0, gby = 0, gbz = 0;
        if (gga) {
            gax = static_cast<double>(grad_x_.data()[i]);
            gay = static_cast<double>(grad_y_.data()[i]);
            gaz = static_cast<double>(grad_z_.data()[i]);
            gbx = static_cast<double>(grad_dn_x_.data()[i]);
            gby = static_cast<double>(grad_dn_y_.data()[i]);
            gbz = static_cast<double>(grad_dn_z_.data()[i]);
            saa = gax * gax + gay * gay + gaz * gaz;
            sbb = gbx * gbx + gby * gby + gbz * gbz;
            sab = gax * gbx + gay * gby + gaz * gbz;
        }

        // True spin-GGA when gga; σ=0 for pure LDA.
        const auto out =
            cpu::scf::evaluate_xc(xc, static_cast<double>(ra),
                                  static_cast<double>(rb), saa, sab, sbb);
        vxc_up[i] = static_cast<AccReal>(out.vrho_alpha);
        vxc_dn[i] = static_cast<AccReal>(out.vrho_beta);
        ws.exc_r[i] = static_cast<AccReal>(out.exc);
        E_xc += rtot * static_cast<AccReal>(out.exc) * dV;

        if (gga) {
            // flux_α = 2 vσ_αα ∇ρ_α + vσ_αβ ∇ρ_β
            // flux_β = 2 vσ_ββ ∇ρ_β + vσ_αβ ∇ρ_α
            const double caa = 2.0 * out.vsigma_aa;
            const double cbb = 2.0 * out.vsigma_bb;
            const double cab = out.vsigma_ab;
            fx_up[i] = static_cast<AccReal>(caa * gax + cab * gbx);
            fy_up[i] = static_cast<AccReal>(caa * gay + cab * gby);
            fz_up[i] = static_cast<AccReal>(caa * gaz + cab * gbz);
            fx_dn[i] = static_cast<AccReal>(cbb * gbx + cab * gax);
            fy_dn[i] = static_cast<AccReal>(cbb * gby + cab * gay);
            fz_dn[i] = static_cast<AccReal>(cbb * gbz + cab * gaz);
        }
    }
    ws.E_xc = E_xc;

    // ── GGA: v_σ -= div(flux_σ) ───────────────────────────────────────
    if (gga) {
        auto* work = work_g_half_.data();
        const AccReal inv_nr = AccReal(1) / static_cast<AccReal>(nr);
        AccReal* flux_up[3] = {fx_up, fy_up, fz_up};
        AccReal* flux_dn[3] = {fx_dn, fy_dn, fz_dn};
        AccReal* vxc[2] = {vxc_up, vxc_dn};
        AccReal* div_buf[2] = {div_r_.data(), div_dn_r_.data()};

        for (int s = 0; s < 2; ++s) {
            AccReal** flux = (s == 0) ? flux_up : flux_dn;
            // Accumulate div(G) into V_xc_spin_G[s] scratch then IFFT.
            for (int ig = 0; ig < ng; ++ig)
                ws.V_xc_spin_G[s][ig] = std::complex<AccReal>(0, 0);

            for (int d = 0; d < 3; ++d) {
                ws.fft->r2c(flux[d], work);
                for (int ig = 0; ig < ng; ++ig) {
                    bool conj = false;
                    const int idx = pack_half_index(
                        gv.h[ig], gv.k[ig], gv.l[ig], nx, ny, nz, nxc, conj);
                    if (idx < 0 || idx >= nc)
                        continue;
                    const AccReal gd = gv.g[ig][d];
                    const AccReal re =
                        static_cast<AccReal>(work[idx][0]) * inv_nr;
                    const AccReal im = static_cast<AccReal>(conj ? -work[idx][1]
                                                                 : work[idx][1])
                                       * inv_nr;
                    // i * G_d * flux_d(G) = (-G_d im, G_d re)
                    ws.V_xc_spin_G[s][ig] +=
                        std::complex<AccReal>(-gd * im, gd * re);
                }
            }

            pack_sphere_to_half(gv, ng, nx, ny, nz, nxc, nc, work, [&](int ig) {
                return std::complex<double>(
                    static_cast<double>(ws.V_xc_spin_G[s][ig].real()),
                    static_cast<double>(ws.V_xc_spin_G[s][ig].imag()));
            });
            ws.fft->c2r(work, div_buf[s]);
            for (int i = 0; i < nr; ++i)
                vxc[s][i] -= div_buf[s][i];
        }
    }

    // Double-counting ∫(ρ↑ Vxc↑ + ρ↓ Vxc↓) uses the full functional derivative
    // (local + div term for GGA), matching restricted G-space e_xc_int.
    AccReal e_xc_int = 0;
    for (int i = 0; i < nr; ++i) {
        AccReal ra = rho_up[i];
        AccReal rb = rho_dn[i];
        if (ra < AccReal(0))
            ra = AccReal(0);
        if (rb < AccReal(0))
            rb = AccReal(0);
        ws.V_xc_spin_r[0][i] = vxc_up[i];
        ws.V_xc_spin_r[1][i] = vxc_dn[i];
        e_xc_int += (ra * vxc_up[i] + rb * vxc_dn[i]) * dV;
    }
    ws.e_xc_int_spin = e_xc_int;

    auto* vxc_g_half = vxc_g_half_.data();
    fft_vxc_r_to_g(ws, vxc_up, vxc_g_half, ws.V_xc_spin_G[0]);
    fft_vxc_r_to_g(ws, vxc_dn, vxc_g_half, ws.V_xc_spin_G[1]);
    // Restricted-compatible total Vxc (not used by spin Davidson).
    for (int ig = 0; ig < ng; ++ig)
        ws.V_xc_G[ig] =
            AccReal(0.5) * (ws.V_xc_spin_G[0][ig] + ws.V_xc_spin_G[1][ig]);
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
