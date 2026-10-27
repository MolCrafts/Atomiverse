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
#include "atomiverse/cpu/planewave/internal/xc_quadrature.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

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

// One point prescription for ordinary XC and its finite-difference response.
// Gradients retain their spectral definition when a density sample is clipped.
struct XcPointInput {
    double rho[2]{};
    double gradient[2][3]{};
};
struct XcPointOutput {
    AccReal energy = 0;
    AccReal potential[2]{};
    AccReal flux[2][3]{};
};
XcPointOutput evaluate_point(cpu::xc::XCType xc, bool gga,
                             const XcPointInput& input) {
    XcPointOutput result;
    const double ra = std::max(input.rho[0], 0.0);
    const double rb = std::max(input.rho[1], 0.0);
    if (ra + rb < 1e-12)
        return result;
    double aa = 0, ab = 0, bb = 0;
    if (gga)
        for (int d = 0; d < 3; ++d) {
            aa += input.gradient[0][d] * input.gradient[0][d];
            ab += input.gradient[0][d] * input.gradient[1][d];
            bb += input.gradient[1][d] * input.gradient[1][d];
        }
    const auto point = cpu::xc::evaluate_xc(xc, ra, rb, aa, ab, bb);
    result.energy = static_cast<AccReal>(point.energy_density);
    result.potential[0] =
        input.rho[0] > 0 ? static_cast<AccReal>(point.vrho_alpha) : AccReal(0);
    result.potential[1] =
        input.rho[1] > 0 ? static_cast<AccReal>(point.vrho_beta) : AccReal(0);
    if (gga)
        for (int d = 0; d < 3; ++d) {
            result.flux[0][d] =
                static_cast<AccReal>(2 * point.vsigma_aa * input.gradient[0][d]
                                     + point.vsigma_ab * input.gradient[1][d]);
            result.flux[1][d] =
                static_cast<AccReal>(2 * point.vsigma_bb * input.gradient[1][d]
                                     + point.vsigma_ab * input.gradient[0][d]);
        }
    return result;
}

// The physical density sphere must be strictly below every even-grid
// Nyquist plane. This is the PW setup contract; folding two distinct edge
// modes into one real-grid sample would not be an adjoint projection.
void validate_density_modes(const PWWorkspace& ws) {
    if (!ws.gvec || ws.n_g < 1 || ws.nr != ws.fft_nx * ws.fft_ny * ws.fft_nz
        || !(ws.omega > 0) || !std::isfinite(ws.omega))
        throw std::invalid_argument("PW XC: invalid grid, G-sphere or volume");
    const auto& gv = *ws.gvec;
    const auto ng = static_cast<std::size_t>(ws.n_g);
    if (gv.g.size() != ng || gv.h.size() != ng || gv.k.size() != ng
        || gv.l.size() != ng)
        throw std::invalid_argument("PW XC: inconsistent G-sphere sizes");
    for (int ig = 0; ig < ws.n_g; ++ig)
        if (2 * std::abs(gv.h[ig]) >= ws.fft_nx
            || 2 * std::abs(gv.k[ig]) >= ws.fft_ny
            || 2 * std::abs(gv.l[ig]) >= ws.fft_nz)
            throw std::invalid_argument("PW XC: density modes must lie below "
                                        "the base-grid Nyquist plane");
}

} // namespace

void XCBuilder::prepare(const PWWorkspace& ws) {
    if (ws.nr <= 0)
        return;
    // Fractional powers and low-density behaviour make both LDA and GGA
    // non-polynomial. Use fixed numerical oversampling on both paths, with
    // more points for the GGA gradient dependence. Keep these grids fixed
    // throughout energy/potential evaluation and response differences.
    const int lx = xc_quadrature_dimension(ws.fft_nx, false),
              ly = xc_quadrature_dimension(ws.fft_ny, false),
              lz = xc_quadrature_dimension(ws.fft_nz, false);
    if (!local_quadrature_fft_ || local_quadrature_fft_->nx() != lx
        || local_quadrature_fft_->ny() != ly
        || local_quadrature_fft_->nz() != lz)
        local_quadrature_fft_ = std::make_unique<FFTPlan3D>(lx, ly, lz);
    const int gx = xc_quadrature_dimension(ws.fft_nx, true),
              gy = xc_quadrature_dimension(ws.fft_ny, true),
              gz = xc_quadrature_dimension(ws.fft_nz, true);
    if (!gradient_quadrature_fft_ || gradient_quadrature_fft_->nx() != gx
        || gradient_quadrature_fft_->ny() != gy
        || gradient_quadrature_fft_->nz() != gz)
        gradient_quadrature_fft_ = std::make_unique<FFTPlan3D>(gx, gy, gz);
    // One scratch allocation serves both plans; the GGA grid is larger.
    const auto nr = static_cast<std::size_t>(gradient_quadrature_fft_->nr());
    const auto nc = static_cast<std::size_t>(gradient_quadrature_fft_->nc());
    restricted_vxc_r_.resize(nr);
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

void XCBuilder::check_prepared(const PWWorkspace& ws) const {
    if (!local_quadrature_fft_ || !gradient_quadrature_fft_
        || local_quadrature_fft_->nx()
               != xc_quadrature_dimension(ws.fft_nx, false)
        || local_quadrature_fft_->ny()
               != xc_quadrature_dimension(ws.fft_ny, false)
        || local_quadrature_fft_->nz()
               != xc_quadrature_dimension(ws.fft_nz, false)
        || gradient_quadrature_fft_->nx()
               != xc_quadrature_dimension(ws.fft_nx, true)
        || gradient_quadrature_fft_->ny()
               != xc_quadrature_dimension(ws.fft_ny, true)
        || gradient_quadrature_fft_->nz()
               != xc_quadrature_dimension(ws.fft_nz, true)
        || (ws.nspin == 2
            && rho_up_r_.size()
                   < static_cast<std::size_t>(gradient_quadrature_fft_->nr())))
        throw std::logic_error(
            "PW XC: call prepare after changing the grid or spin layout");
}

void XCBuilder::project_to_base(
    const PWWorkspace& ws, const xt::xarray<std::complex<AccReal>>& potential,
    std::span<AccReal> output) {
    auto* work = work_g_half_.data();
    pack_sphere_to_half(*ws.gvec, ws.n_g, ws.fft_nx, ws.fft_ny, ws.fft_nz,
                        ws.fft_nx / 2 + 1, ws.nc, work,
                        [&](int ig) { return potential[ig]; });
    ws.fft->c2r(work, output.data());
}

void XCBuilder::step(PWWorkspace& ws, cpu::xc::XCType xc) {
    const int ng = ws.n_g;

    if (!ws.fft || ws.nr <= 0) {
        for (int i = 0; i < ng; ++i)
            ws.V_xc_G[i] = std::complex<AccReal>(0, 0);
        return;
    }

    validate_density_modes(ws);
    const bool gga = cpu::xc::is_gga(xc);
    check_prepared(ws);
    if (ws.rho_G.size() != static_cast<std::size_t>(ng))
        throw std::invalid_argument("PW XC: density does not match G-sphere");
    FFTPlan3D& fft = gga ? *gradient_quadrature_fft_ : *local_quadrature_fft_;
    const int nr = fft.nr(), nc = fft.nc();
    const int nx = fft.nx(), ny = fft.ny(), nz = fft.nz();
    const int nxc = nx / 2 + 1;
    auto& gv = *ws.gvec;
    if (ws.V_xc_G.size() != static_cast<std::size_t>(ng))
        ws.V_xc_G.resize({static_cast<std::size_t>(ng)});

    // ── 1. FFT ρ(G) → ρ(r) ──────────────────────────────────────────
    auto* rho_g_half = rho_g_half_.data();
    auto* rho_r = rho_r_.data();

    pack_sphere_to_half(gv, ng, nx, ny, nz, nxc, nc, rho_g_half, [&](int ig) {
        return std::complex<double>(static_cast<double>(ws.rho_G[ig].real()),
                                    static_cast<double>(ws.rho_G[ig].imag()));
    });
    // FFTW does NOT normalize: c2r output = Σ_G input[G] exp(iGr)
    fft.c2r(rho_g_half, rho_r);

    auto* vxc_r = restricted_vxc_r_.data();

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
            fft.c2r(work, grad_comp[d]);
        }
    }

    // ── 2. Pointwise XC (+ flux for GGA) ─────────────────────────────
    const AccReal dV = ws.omega / static_cast<AccReal>(nr);
    AccReal E_xc = 0;
    auto* fx = flux_x_.data();
    auto* fy = flux_y_.data();
    auto* fz = flux_z_.data();

    const AccReal* gradients[3] = {grad_x_.data(), grad_y_.data(),
                                   grad_z_.data()};
    AccReal* fluxes[3] = {fx, fy, fz};
    for (int i = 0; i < nr; ++i) {
        XcPointInput input;
        input.rho[0] = input.rho[1] = static_cast<double>(rho_r[i]) / 2;
        if (gga)
            for (int d = 0; d < 3; ++d)
                input.gradient[0][d] = input.gradient[1][d] =
                    static_cast<double>(gradients[d][i]) / 2;
        const auto point = evaluate_point(xc, gga, input);
        vxc_r[i] = point.potential[0];
        E_xc += point.energy * dV;
        if (gga)
            for (int d = 0; d < 3; ++d)
                fluxes[d][i] = point.flux[0][d];
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
            fft.r2c(flux_comp[d], work);
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
        fft.c2r(work, div_r_.data());
        for (int i = 0; i < nr; ++i)
            vxc_r[i] -= div_r_.data()[i];
    }

    // ── 3. FFT V_xc(r) → V_xc(G) ─────────────────────────────────────
    auto* vxc_g_half = vxc_g_half_.data();
    fft.r2c(vxc_r, vxc_g_half);

    const AccReal inv_nr = AccReal(1) / static_cast<AccReal>(nr);
    unpack_half_to_sphere(gv, ng, nx, ny, nz, nxc, nc, vxc_g_half, inv_nr,
                          ws.V_xc_G);

    ws.V_xc_r.resize({static_cast<std::size_t>(ws.nr)});
    project_to_base(ws, ws.V_xc_G,
                    {ws.V_xc_r.data(), static_cast<std::size_t>(ws.nr)});
}

void XCBuilder::step_spin(PWWorkspace& ws, cpu::xc::XCType xc) {
    int ng = ws.n_g;
    if (!ws.fft || ws.nr <= 0 || ws.nspin != 2 || ws.rho_spin_G.size() != 2) {
        for (int i = 0; i < ng; ++i)
            ws.V_xc_G[i] = std::complex<AccReal>(0, 0);
        ws.E_xc = 0;
        ws.e_xc_int_spin = 0;
        return;
    }

    validate_density_modes(ws);
    const bool gga = cpu::xc::is_gga(xc);
    check_prepared(ws);
    for (const auto& channel : ws.rho_spin_G)
        if (channel.size() != static_cast<std::size_t>(ng))
            throw std::invalid_argument(
                "PW XC: spin density does not match G-sphere");
    FFTPlan3D& fft = gga ? *gradient_quadrature_fft_ : *local_quadrature_fft_;
    const int nr = fft.nr(), nc = fft.nc();
    const int nx = fft.nx(), ny = fft.ny(), nz = fft.nz();
    const int nxc = nx / 2 + 1;
    auto& gv = *ws.gvec;
    if (ws.V_xc_G.size() != static_cast<std::size_t>(ng))
        ws.V_xc_G.resize({static_cast<std::size_t>(ng)});

    auto* rho_g_half = rho_g_half_.data();
    auto* rho_up = rho_up_r_.data();
    auto* rho_dn = rho_dn_r_.data();
    auto* vxc_up = vxc_up_r_.data();
    auto* vxc_dn = vxc_dn_r_.data();

    for (int spin = 0; spin < 2; ++spin) {
        pack_sphere_to_half(gv, ng, nx, ny, nz, nxc, nc, rho_g_half,
                            [&](int ig) { return ws.rho_spin_G[spin][ig]; });
        fft.c2r(rho_g_half, spin == 0 ? rho_up : rho_dn);
    }

    if (ws.V_xc_spin_G.size() != 2) {
        ws.V_xc_spin_G.resize(2);
        ws.V_xc_spin_G[0].resize({static_cast<std::size_t>(ng)});
        ws.V_xc_spin_G[1].resize({static_cast<std::size_t>(ng)});
    }
    for (int s = 0; s < 2; ++s) {
        if (ws.V_xc_spin_G[s].size() != static_cast<std::size_t>(ng))
            ws.V_xc_spin_G[s].resize({static_cast<std::size_t>(ng)});
    }

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
                fft.c2r(work, gcomp[d]);
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

    const AccReal* gradients[2][3] = {
        {grad_x_.data(), grad_y_.data(), grad_z_.data()},
        {grad_dn_x_.data(), grad_dn_y_.data(), grad_dn_z_.data()}};
    AccReal* fluxes[2][3] = {{fx_up, fy_up, fz_up}, {fx_dn, fy_dn, fz_dn}};
    for (int i = 0; i < nr; ++i) {
        XcPointInput input;
        input.rho[0] = static_cast<double>(rho_up[i]);
        input.rho[1] = static_cast<double>(rho_dn[i]);
        if (gga)
            for (int spin = 0; spin < 2; ++spin)
                for (int d = 0; d < 3; ++d)
                    input.gradient[spin][d] =
                        static_cast<double>(gradients[spin][d][i]);
        const auto point = evaluate_point(xc, gga, input);
        vxc_up[i] = point.potential[0];
        vxc_dn[i] = point.potential[1];
        E_xc += point.energy * dV;
        if (gga)
            for (int spin = 0; spin < 2; ++spin)
                for (int d = 0; d < 3; ++d)
                    fluxes[spin][d][i] = point.flux[spin][d];
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
                fft.r2c(flux[d], work);
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
            fft.c2r(work, div_buf[s]);
            for (int i = 0; i < nr; ++i)
                vxc[s][i] -= div_buf[s][i];
        }
    }

    auto* vxc_g_half = vxc_g_half_.data();
    AccReal* potentials[2] = {vxc_up, vxc_dn};
    for (int spin = 0; spin < 2; ++spin) {
        fft.r2c(potentials[spin], vxc_g_half);
        unpack_half_to_sphere(gv, ng, nx, ny, nz, nxc, nc, vxc_g_half,
                              AccReal(1) / nr, ws.V_xc_spin_G[spin]);
        ws.V_xc_spin_r[spin].resize({static_cast<std::size_t>(ws.nr)});
        project_to_base(
            ws, ws.V_xc_spin_G[spin],
            {ws.V_xc_spin_r[spin].data(), static_cast<std::size_t>(ws.nr)});
    }
    // Double counting uses the physical spectral density and the same
    // projected derivative as the Hamiltonian, including negative ringing.
    // Parseval is the exact weighted adjoint of the fine-grid interpolation.
    AccReal e_xc_int = 0;
    for (int spin = 0; spin < 2; ++spin)
        for (int ig = 0; ig < ng; ++ig)
            e_xc_int += ws.omega
                        * (std::conj(ws.rho_spin_G[spin][ig])
                           * ws.V_xc_spin_G[spin][ig])
                              .real();
    ws.e_xc_int_spin = e_xc_int;
    // Restricted-compatible total Vxc (not used by spin Davidson).
    for (int ig = 0; ig < ng; ++ig)
        ws.V_xc_G[ig] =
            AccReal(0.5) * (ws.V_xc_spin_G[0][ig] + ws.V_xc_spin_G[1][ig]);
}

void XCBuilder::cache_response_reference(
    const PWWorkspace& ws, cpu::xc::XCType xc,
    std::span<const ResponseChannel> channels) {
    FFTPlan3D& fft = cpu::xc::is_gga(xc) ? *gradient_quadrature_fft_
                                         : *local_quadrature_fft_;
    const std::array<int, 3> shape{fft.nx(), fft.ny(), fft.nz()};
    const auto& gv = *ws.gvec;
    bool same = response_channels_ == static_cast<int>(channels.size())
                && response_shape_ == shape && response_h_ == gv.h
                && response_k_ == gv.k && response_l_ == gv.l
                && response_metric_.size() == gv.g.size();
    if (same)
        for (std::size_t g = 0; g < gv.g.size(); ++g)
            for (int d = 0; d < 3; ++d)
                if (response_metric_[g][d] != gv.g[g][d])
                    same = false;
    if (same)
        for (std::size_t spin = 0; spin < channels.size(); ++spin)
            if (response_reference_[spin].size()
                    != channels[spin].reference.size()
                || !std::equal(response_reference_[spin].begin(),
                               response_reference_[spin].end(),
                               channels[spin].reference.begin()))
                same = false;
    if (same)
        return;
    // Commit the cache key only after every reference transform succeeds.
    response_channels_ = 0;
    const auto nr = static_cast<std::size_t>(fft.nr());
    auto* work = work_g_half_.data();
    for (std::size_t spin = 0; spin < channels.size(); ++spin) {
        response_density_[spin].resize(nr);
        pack_sphere_to_half(gv, ws.n_g, fft.nx(), fft.ny(), fft.nz(),
                            fft.nx() / 2 + 1, fft.nc(), work,
                            [&](int g) { return channels[spin].reference[g]; });
        fft.c2r(work, response_density_[spin].data());
        if (cpu::xc::is_gga(xc))
            for (int d = 0; d < 3; ++d) {
                response_gradient_[spin][d].resize(nr);
                pack_sphere_to_half(
                    gv, ws.n_g, fft.nx(), fft.ny(), fft.nz(), fft.nx() / 2 + 1,
                    fft.nc(), work, [&](int g) {
                        return std::complex<AccReal>(0, gv.g[g][d])
                               * channels[spin].reference[g];
                    });
                fft.c2r(work, response_gradient_[spin][d].data());
            }
        response_reference_[spin].assign(channels[spin].reference.begin(),
                                         channels[spin].reference.end());
    }
    response_shape_ = shape;
    response_h_ = gv.h;
    response_k_ = gv.k;
    response_l_ = gv.l;
    response_metric_ = gv.g;
    response_channels_ = static_cast<int>(channels.size());
}

void XCBuilder::response(const PWWorkspace& ws, cpu::xc::XCType xc,
                         std::span<const ResponseChannel> channels,
                         AccReal epsilon) {
    validate_density_modes(ws);
    check_prepared(ws);
    if ((channels.size() != 1 && channels.size() != 2)
        || channels.size() != static_cast<std::size_t>(ws.nspin)
        || !std::isfinite(epsilon) || epsilon <= 0)
        throw std::invalid_argument(
            "PW XC response: invalid channels or epsilon");
    for (const auto& channel : channels) {
        if (channel.reference.size() != static_cast<std::size_t>(ws.n_g)
            || channel.variation.size() != static_cast<std::size_t>(ws.n_g)
            || channel.potential.size() != static_cast<std::size_t>(ws.nr))
            throw std::invalid_argument(
                "PW XC response: channel shape mismatch");
        for (std::size_t g = 0; g < channel.reference.size(); ++g)
            if (!std::isfinite(channel.reference[g].real())
                || !std::isfinite(channel.reference[g].imag())
                || !std::isfinite(channel.variation[g].real())
                || !std::isfinite(channel.variation[g].imag()))
                throw std::invalid_argument(
                    "PW XC response: nonfinite Fourier coefficient");
    }
    cache_response_reference(ws, xc, channels);
    const bool gga = cpu::xc::is_gga(xc);
    FFTPlan3D& fft = gga ? *gradient_quadrature_fft_ : *local_quadrature_fft_;
    const int nr = fft.nr(), nc = fft.nc(), ng = ws.n_g;
    const int nx = fft.nx(), ny = fft.ny(), nz = fft.nz(), nxc = nx / 2 + 1;
    const auto& gv = *ws.gvec;
    const bool spin = channels.size() == 2;
    AccReal* density_delta[2] = {spin ? rho_up_r_.data() : rho_r_.data(),
                                 rho_dn_r_.data()};
    AccReal* gradient_delta[2][3] = {
        {grad_x_.data(), grad_y_.data(), grad_z_.data()},
        {grad_dn_x_.data(), grad_dn_y_.data(), grad_dn_z_.data()}};
    AccReal* potential_delta[2] = {
        spin ? vxc_up_r_.data() : restricted_vxc_r_.data(), vxc_dn_r_.data()};
    AccReal* flux_delta[2][3] = {
        {flux_x_.data(), flux_y_.data(), flux_z_.data()},
        {flux_dn_x_.data(), flux_dn_y_.data(), flux_dn_z_.data()}};
    auto* work = work_g_half_.data();
    for (std::size_t s = 0; s < channels.size(); ++s) {
        pack_sphere_to_half(gv, ng, nx, ny, nz, nxc, nc, work,
                            [&](int g) { return channels[s].variation[g]; });
        fft.c2r(work, density_delta[s]);
        if (gga)
            for (int d = 0; d < 3; ++d) {
                pack_sphere_to_half(
                    gv, ng, nx, ny, nz, nxc, nc, work, [&](int g) {
                        return std::complex<AccReal>(0, gv.g[g][d])
                               * channels[s].variation[g];
                    });
                fft.c2r(work, gradient_delta[s][d]);
            }
    }
    const AccReal inverse = AccReal(1) / (2 * epsilon);
    for (int i = 0; i < nr; ++i) {
        XcPointInput plus, minus;
        for (int s = 0; s < 2; ++s) {
            const int source = spin ? s : 0;
            const double weight = spin ? 1.0 : .5;
            const double r0 = response_density_[source].data()[i];
            const double dr = epsilon * density_delta[source][i];
            plus.rho[s] = weight * (r0 + dr);
            minus.rho[s] = weight * (r0 - dr);
            if (gga)
                for (int d = 0; d < 3; ++d) {
                    const double g0 = response_gradient_[source][d].data()[i];
                    const double dg = epsilon * gradient_delta[source][d][i];
                    plus.gradient[s][d] = weight * (g0 + dg);
                    minus.gradient[s][d] = weight * (g0 - dg);
                }
        }
        const auto positive = evaluate_point(xc, gga, plus);
        const auto negative = evaluate_point(xc, gga, minus);
        for (std::size_t s = 0; s < channels.size(); ++s) {
            potential_delta[s][i] =
                (positive.potential[s] - negative.potential[s]) * inverse;
            if (gga)
                for (int d = 0; d < 3; ++d)
                    flux_delta[s][d][i] =
                        (positive.flux[s][d] - negative.flux[s][d]) * inverse;
        }
    }
    const AccReal inv_nr = AccReal(1) / nr;
    for (std::size_t s = 0; s < channels.size(); ++s) {
        auto& potential = response_potential_G_[s];
        potential.resize({static_cast<std::size_t>(ng)});
        fft.r2c(potential_delta[s], work);
        unpack_half_to_sphere(gv, ng, nx, ny, nz, nxc, nc, work, inv_nr,
                              potential);
        if (gga)
            for (int d = 0; d < 3; ++d) {
                fft.r2c(flux_delta[s][d], work);
                for (int g = 0; g < ng; ++g) {
                    bool conjugate = false;
                    const int index = pack_half_index(
                        gv.h[g], gv.k[g], gv.l[g], nx, ny, nz, nxc, conjugate);
                    const AccReal re =
                        static_cast<AccReal>(work[index][0]) * inv_nr;
                    const AccReal im =
                        static_cast<AccReal>(conjugate ? -work[index][1]
                                                       : work[index][1])
                        * inv_nr;
                    const AccReal gd = gv.g[g][d];
                    potential[g] -= std::complex<AccReal>(-gd * im, gd * re);
                }
            }
        project_to_base(ws, potential, channels[s].potential);
    }
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
