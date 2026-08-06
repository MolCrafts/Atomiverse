/// @file cpu/pw/internal/pw_stress.cpp
/// @brief Plane-wave cell stress: Nielsen–Martin one-SCF analytic path +
///        multi-SCF strain-FD oracle.
///
/// Analytic path (production `pw_stress`):
///   1. One full SCF at the reference geometry.
///   2. Hellmann–Feynman cell stress by central finite differences of the
///      *non-self-consistent* total energy under symmetric strain: density is
///      affinely remapped (fixed fractional coordinates, ρ → ρ·Ω₀/Ω), V_H /
///      V_xc / V_loc / Ewald rebuilt, one Davidson diagonalization per
///      strained cell, then iteration_energy. This is the numerical form of
///      the Nielsen–Martin HF theorem (Nielsen & Martin, PRB 32, 3780 (1985))
///      and matches multi-SCF strain-FD when the residual is tight, at
///      ~1 SCF + 12 diagonalizations instead of 12 SCFs.
///
/// Oracle path (`pw_stress_fd`): full multi-SCF strain-FD of E_tot (12 SCFs).
///
/// Sign convention (productization 2026-07-25): **mechanical / CP2K**
/// \(\sigma = -(1/\Omega)\,\partial E/\partial\varepsilon\), matching
/// GPW `gpw_stress` and `AnisotropicCellRelax` / `IsotropicCellRelax`
/// (positive diagonal stress when the cell is compressed → expand).

#include "atomiverse/cpu/planewave/internal/pw_stress.h"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/fft.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/pw_workspace.h"
#include "atomiverse/cpu/planewave/internal/scf_loop.h"
#include "atomiverse/cpu/planewave/internal/veff_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cpu/scf/xc_functional.h"

#ifdef ATV_HAS_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace ATV {
namespace cpu {
namespace pw {
namespace internal {
namespace {

#ifdef ATV_HAS_MPI
MPI_Datatype mpi_real_type() {
    return sizeof(AccReal) == sizeof(double) ? MPI_DOUBLE : MPI_FLOAT;
}
#endif

struct MpiLayout {
    int rank = 0;
    int nprocs = 1;
    bool active = false;
};

MpiLayout mpi_layout() {
    MpiLayout layout;
#ifdef ATV_HAS_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &layout.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &layout.nprocs);
        layout.active = (layout.nprocs > 1);
    }
#endif
    return layout;
}

PW copy_pw(const PW& m) {
    PW p;
    p.upf_path = m.upf_path;
    p.ecut_ry = m.ecut_ry;
    p.k_mesh = m.k_mesh;
    p.tol = m.tol;
    p.max_iter = m.max_iter;
    p.mix_beta = m.mix_beta;
    p.pulay_depth = m.pulay_depth;
    p.xc = m.xc;
    p.spin_polarized = m.spin_polarized;
    p.multiplicity = m.multiplicity;
    p.smearing = m.smearing;
    p.degauss = m.degauss;
    p.mp_order = m.mp_order;
    p.extra_bands = m.extra_bands;
    return p;
}

void voigt_pair(int v, int& a, int& b) {
    static const int A[6] = {0, 1, 2, 0, 0, 1};
    static const int B[6] = {0, 1, 2, 1, 2, 2};
    a = A[v];
    b = B[v];
}

void apply_strain(cpu::Frame& frame, const Mat3& strain) {
    Mat3 F{};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            F(a, b) = (a == b ? AccReal(1) : AccReal(0)) + strain(a, b);

    Mat3 cell = frame.box.cell();
    Mat3 cell_new;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            AccReal s = 0;
            for (int k = 0; k < 3; ++k)
                s += F(i, k) * cell(k, j);
            cell_new(i, j) = s;
        }
    frame.box.set_cell(cell_new);

    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    const int n = frame.nlocal();
    for (int a = 0; a < n; ++a) {
        AccReal rx = x[a], ry = y[a], rz = z[a];
        x[a] = F(0, 0) * rx + F(0, 1) * ry + F(0, 2) * rz;
        y[a] = F(1, 0) * rx + F(1, 1) * ry + F(1, 2) * rz;
        z[a] = F(2, 0) * rx + F(2, 1) * ry + F(2, 2) * rz;
    }
}

double scf_energy_ry(const cpu::Frame& frame, const PW& method) {
    PW solver = copy_pw(method);
    solver.build(frame);
    return solver.run(frame).energy;
}

/// Pack Γ-sphere ρ_G into a dense half-complex buffer then to real space
/// on the *reference* grid; returns real-space density [Bohr⁻³].
std::vector<AccReal> rho_g_to_real(const PWWorkspace& ws) {
    std::vector<AccReal> rho_r(static_cast<std::size_t>(ws.nr), AccReal(0));
    if (!ws.fft || ws.nr <= 0 || !ws.gvec)
        return rho_r;

    const int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
    const int nxc = nx / 2 + 1;
    const int nc = ws.nc;
    // fftw_complex is double[2] — not a valid std::vector element type on
    // libc++. Stage as interleaved doubles and view as fftw_complex*.
    std::vector<double> half_reim(static_cast<std::size_t>(2 * nc), 0.0);
    auto* half = reinterpret_cast<fftw_complex*>(half_reim.data());

    auto& gv = *ws.gvec;
    for (int ig = 0; ig < ws.n_g; ++ig) {
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
            half[static_cast<std::size_t>(idx)][0] =
                static_cast<double>(ws.rho_G[ig].real());
            half[static_cast<std::size_t>(idx)][1] = static_cast<double>(
                conj ? -ws.rho_G[ig].imag() : ws.rho_G[ig].imag());
        }
    }
    ws.fft->c2r(half, rho_r.data());
    return rho_r;
}

/// Load a real-space density onto a (possibly different) PW workspace's
/// ρ_G via r2c, scaled by `scale` (volume remapping Ω₀/Ω).
void load_rho_real_scaled(PWWorkspace& ws, const std::vector<AccReal>& rho_r0,
                          AccReal scale) {
    if (!ws.fft || ws.nr <= 0 || !ws.gvec)
        return;
    // If grid sizes match, scale in place; else interpolate by nearest
    // (tutorial cells keep the same FFT dims under small strain at fixed
    // ecut — G-sphere cardinality may change but real grid is set by ecut).
    std::vector<AccReal> rho_r(static_cast<std::size_t>(ws.nr), AccReal(0));
    if (static_cast<int>(rho_r0.size()) == ws.nr) {
        for (int i = 0; i < ws.nr; ++i)
            rho_r[static_cast<std::size_t>(i)] =
                rho_r0[static_cast<std::size_t>(i)] * scale;
    } else {
        // Fallback: uniform scale of mean density (rare for small h).
        AccReal mean = 0;
        for (AccReal v : rho_r0)
            mean += v;
        mean /= AccReal(std::max<std::size_t>(1, rho_r0.size()));
        for (int i = 0; i < ws.nr; ++i)
            rho_r[static_cast<std::size_t>(i)] = mean * scale;
    }

    const int nx = ws.fft_nx, ny = ws.fft_ny, nz = ws.fft_nz;
    const int nxc = nx / 2 + 1;
    const int nc = ws.nc;
    std::vector<double> half_reim(static_cast<std::size_t>(2 * nc), 0.0);
    auto* half = reinterpret_cast<fftw_complex*>(half_reim.data());
    ws.fft->r2c(rho_r.data(), half);
    AccReal inv_nr = AccReal(1) / AccReal(ws.nr);
    auto& gv = *ws.gvec;
    for (int ig = 0; ig < ws.n_g; ++ig) {
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
            AccReal re =
                static_cast<AccReal>(half[static_cast<std::size_t>(idx)][0])
                * inv_nr;
            AccReal im =
                static_cast<AccReal>(half[static_cast<std::size_t>(idx)][1])
                * inv_nr;
            if (conj)
                im = -im;
            ws.rho_G[ig] = std::complex<AccReal>(re, im);
        } else {
            ws.rho_G[ig] = std::complex<AccReal>(0, 0);
        }
    }
}

/// Non-self-consistent total energy [Ry] at `frame` with remapped density
/// from the reference SCF (`rho_r0` at volume `omega0_bohr`).
double nscf_energy_ry(const cpu::Frame& frame, const PW& method,
                      const std::vector<AccReal>& rho_r0, AccReal omega0_bohr,
                      PoissonSolver& poisson, XCBuilder& xc, VeffBuilder& veff,
                      DavidsonSolver& davidson, int n_bands) {
    PWProblem prob;
    setup_pw_problem(frame, method.upf_path, method.ecut_ry, method.k_mesh,
                     prob);
    PWWorkspace& ws = prob.ws;
    ws.mix_beta = AccReal(method.mix_beta);
    ws.pulay_depth = method.pulay_depth;

    xc.prepare(ws);
    veff.prepare(ws);
    davidson.prepare(ws);

    // Remap density: ρ'(r) = ρ₀(r) · (Ω₀ / Ω) so ∫ρ is conserved under
    // volume change (fixed fractional ionic coordinates).
    AccReal scale = omega0_bohr / ws.omega;
    load_rho_real_scaled(ws, rho_r0, scale);

    poisson.step(ws);
    xc.step(ws, method.xc);
    veff.step(ws);

    const auto layout = mpi_layout();
    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    AccReal e_band = 0;
    auto* saved_gvec = ws.gvec;
    auto& gv0 = prob.gv0;

    for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
        auto& gvk = prob.gv_k[static_cast<std::size_t>(ik)];
        ws.gvec = &gvk;
        ws.n_g = gvk.count();
        ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
        for (int ig = 0; ig < ws.n_g; ++ig)
            ws.T_G[ig] = gvk.ekin[ig];

        if (!prob.kb_proj_k.empty())
            ws.kb_proj = prob.kb_proj_k[static_cast<std::size_t>(ik)];
        if (!prob.atom_wfc_k.empty())
            ws.atom_wfc = prob.atom_wfc_k[static_cast<std::size_t>(ik)];

        davidson.step(ws);

        const AccReal weight = prob.kg.weights[static_cast<std::size_t>(ik)];
        for (int b = 0; b < n_bands; ++b)
            e_band += weight * ws.eigenvalues[b] * AccReal(2);

        ws.gvec = saved_gvec;
        ws.n_g = gv0.count();
    }

#ifdef ATV_HAS_MPI
    if (layout.active) {
        MPI_Allreduce(MPI_IN_PLACE, &e_band, 1, mpi_real_type(), MPI_SUM,
                      MPI_COMM_WORLD);
    }
#endif

    // Restore Γ sphere for energy Parseval (ρ, V_H, V_xc, V_loc live there).
    ws.gvec = &gv0;
    ws.n_g = gv0.count();
    const auto terms =
        iteration_energy(ws, e_band, prob.ew_total, prob.vol_bohr);
    return static_cast<double>(terms.total) / qe_units::rydberg_to_hartree;
}

} // namespace

std::array<double, 6> pw_stress_fd_impl(const cpu::Frame& frame,
                                        const PW& method, double h) {
    if (frame.nlocal() <= 0)
        throw std::invalid_argument("pw_stress_fd: frame has no atoms");
    if (frame.box.volume() <= 0)
        throw std::invalid_argument("pw_stress_fd: non-positive cell volume");
    if (!(h > 0.0))
        throw std::invalid_argument("pw_stress_fd: h must be > 0");

    const double vol = static_cast<double>(frame.box.volume());
    const double ry_to_ev = qe_units::rydberg_to_ev;

    std::array<double, 6> stress{};
    for (int v = 0; v < 6; ++v) {
        int a = 0, b = 0;
        voigt_pair(v, a, b);
        Mat3 ep = Mat3{};
        Mat3 em = Mat3{};
        if (a == b) {
            ep(a, a) = h;
            em(a, a) = -h;
        } else {
            ep(a, b) = ep(b, a) = h;
            em(a, b) = em(b, a) = -h;
        }
        cpu::Frame fp = frame;
        cpu::Frame fm = frame;
        apply_strain(fp, ep);
        apply_strain(fm, em);
        const double ep_ry = scf_energy_ry(fp, method);
        const double em_ry = scf_energy_ry(fm, method);
        const double dE_dev = (ep_ry - em_ry) * ry_to_ev / (2.0 * h);
        // Mechanical / CP2K: σ = −(1/Ω) ∂E/∂ε (matches GPW cell relax).
        stress[static_cast<std::size_t>(v)] = -dE_dev / vol;
    }
    return stress;
}

std::array<double, 6> pw_stress_analytic_impl(const cpu::Frame& frame,
                                              const PW& method,
                                              double* energy_ry_out) {
    if (frame.nlocal() <= 0)
        throw std::invalid_argument("pw_stress: frame has no atoms");
    if (frame.box.volume() <= 0)
        throw std::invalid_argument("pw_stress: non-positive cell volume");

    const int n = frame.nlocal();
    int n_elec = 0;
    {
        const int* types = frame.span(kAtom_type);
        for (int i = 0; i < n; ++i)
            n_elec += valence_from_z(types[i]);
    }
    const int n_bands = std::max(n_elec / 2, 1);

    PoissonSolver poisson;
    XCBuilder xc;
    VeffBuilder veff;
    DensityBuilder density;
    DavidsonSolver davidson{n_bands, AccReal(method.tol) * AccReal(0.1), 40};

    PWProblem scf_prob;
    const PWResult res = run_scf_fixed_occ(frame, method, scf_prob, poisson, xc,
                                           veff, density, davidson, "stress");
    if (energy_ry_out)
        *energy_ry_out = res.energy;
    if (!res.converged) {
        ATV_WARN("[pw] pw_stress_analytic: SCF not converged after {} iters; "
                 "using last density for HF stress",
                 res.iterations);
    }

    PWWorkspace& ws = scf_prob.ws;
    ws.gvec = &scf_prob.gv0;
    ws.n_g = scf_prob.gv0.count();
    const AccReal omega0 = ws.omega;
    const std::vector<AccReal> rho_r0 = rho_g_to_real(ws);

    // Hellmann–Feynman strain-FD of the NSCF energy (one SCF total).
    const double h = 2e-3;
    const double vol = static_cast<double>(frame.box.volume());
    const double ry_to_ev = qe_units::rydberg_to_ev;

    std::array<double, 6> stress{};
    for (int v = 0; v < 6; ++v) {
        int a = 0, b = 0;
        voigt_pair(v, a, b);
        Mat3 ep = Mat3{};
        Mat3 em = Mat3{};
        if (a == b) {
            ep(a, a) = h;
            em(a, a) = -h;
        } else {
            ep(a, b) = ep(b, a) = h;
            em(a, b) = em(b, a) = -h;
        }
        cpu::Frame fp = frame;
        cpu::Frame fm = frame;
        apply_strain(fp, ep);
        apply_strain(fm, em);

        const double ep_ry = nscf_energy_ry(fp, method, rho_r0, omega0, poisson,
                                            xc, veff, davidson, n_bands);
        const double em_ry = nscf_energy_ry(fm, method, rho_r0, omega0, poisson,
                                            xc, veff, davidson, n_bands);
        const double dE_dev = (ep_ry - em_ry) * ry_to_ev / (2.0 * h);
        // Mechanical / CP2K: σ = −(1/Ω) ∂E/∂ε (matches GPW cell relax).
        stress[static_cast<std::size_t>(v)] = -dE_dev / vol;
    }
    return stress;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
