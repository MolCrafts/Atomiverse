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
#include "atomiverse/cpu/planewave/internal/restricted_scf.h"
#include "atomiverse/cpu/planewave/internal/scf_loop.h"
#include "atomiverse/cpu/planewave/internal/smearing.h"
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
    PW solver = copy_pw_params(method);
    solver.build(frame);
    return solver.run(frame).energy;
}

/// Pack a Γ-sphere density channel into real space [Bohr⁻³].
///
/// Used for total ρ (restricted) and for each collinear spin channel
/// ρ_σ before Nielsen–Martin strain remapping.
std::vector<AccReal> rho_channel_g_to_real(
    const PWWorkspace& ws,
    const xt::xarray<std::complex<AccReal>>& rho_G) {
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
    const int n_g = std::min(ws.n_g, static_cast<int>(rho_G.size()));
    for (int ig = 0; ig < n_g; ++ig) {
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
                static_cast<double>(rho_G[ig].real());
            half[static_cast<std::size_t>(idx)][1] = static_cast<double>(
                conj ? -rho_G[ig].imag() : rho_G[ig].imag());
        }
    }
    ws.fft->c2r(half, rho_r.data());
    return rho_r;
}

/// Pack Γ-sphere ρ_G into a dense half-complex buffer then to real space
/// on the *reference* grid; returns real-space density [Bohr⁻³].
std::vector<AccReal> rho_g_to_real(const PWWorkspace& ws) {
    return rho_channel_g_to_real(ws, ws.rho_G);
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

/// Load two collinear spin real-space densities onto `ws` (ρ_σ and total ρ),
/// scaled by `scale` (volume remap Ω₀/Ω). Allocates spin G slots if needed.
void load_rho_spin_real_scaled(PWWorkspace& ws,
                               const std::vector<AccReal>& rho_a0,
                               const std::vector<AccReal>& rho_b0,
                               AccReal scale) {
    if (!ws.fft || ws.nr <= 0 || !ws.gvec)
        return;
    const int ng0 = ws.gvec->count();
    const std::size_t ngs = static_cast<std::size_t>(ng0);
    if (ws.rho_spin_G.size() != 2) {
        ws.rho_spin_G.resize(2);
        ws.rho_spin_G_old.resize(2);
        ws.V_xc_spin_G.resize(2);
    }
    for (int s = 0; s < 2; ++s) {
        if (ws.rho_spin_G[s].size() != ngs)
            ws.rho_spin_G[s].resize({ngs});
        if (ws.rho_spin_G_old[s].size() != ngs)
            ws.rho_spin_G_old[s].resize({ngs});
        if (ws.V_xc_spin_G[s].size() != ngs)
            ws.V_xc_spin_G[s].resize({ngs});
    }
    ws.nspin = 2;
    if (static_cast<int>(ws.rho_G.size()) < ng0)
        ws.rho_G.resize({static_cast<std::size_t>(ng0)});

    // Stage each channel through the restricted loader (writes rho_G), then
    // copy into rho_spin_G[s] and rebuild total.
    load_rho_real_scaled(ws, rho_a0, scale);
    for (int ig = 0; ig < ng0; ++ig)
        ws.rho_spin_G[0][ig] = ws.rho_G[ig];
    load_rho_real_scaled(ws, rho_b0, scale);
    for (int ig = 0; ig < ng0; ++ig)
        ws.rho_spin_G[1][ig] = ws.rho_G[ig];
    for (int ig = 0; ig < ng0; ++ig)
        ws.rho_G[ig] = ws.rho_spin_G[0][ig] + ws.rho_spin_G[1][ig];
}

/// Non-self-consistent collinear spin total energy [Ry] under remapped ρ_σ.
double nscf_energy_spin_ry(const cpu::Frame& frame, const PW& method,
                           const std::vector<AccReal>& rho_a0,
                           const std::vector<AccReal>& rho_b0,
                           AccReal omega0_bohr, PoissonSolver& poisson,
                           XCBuilder& xc, VeffBuilder& veff,
                           DavidsonSolver& davidson, int n_bands, int n_alpha,
                           int n_beta) {
    PWProblem prob;
    setup_pw_problem(frame, method.upf_path, method.gth_name, method.ecut_ry,
                     method.k_mesh, prob);
    PWWorkspace& ws = prob.ws;
    ws.mix_beta = AccReal(method.mix_beta);
    ws.pulay_depth = method.pulay_depth;

    xc.prepare(ws);
    veff.prepare(ws);
    davidson.prepare(ws);

    AccReal scale = omega0_bohr / ws.omega;
    load_rho_spin_real_scaled(ws, rho_a0, rho_b0, scale);

    poisson.step(ws);
    xc.step_spin(ws, method.xc);

    const auto layout = mpi_layout();
    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    const int n_occ[2] = {n_alpha, n_beta};
    AccReal e_band = 0;
    auto* saved_gvec = ws.gvec;
    auto& gv0 = prob.gv0;
    const int ng0 = gv0.count();

    for (int s = 0; s < 2; ++s) {
        if (n_occ[s] <= 0)
            continue;
        for (int ig = 0; ig < ng0; ++ig)
            ws.V_xc_G[ig] = ws.V_xc_spin_G[s][ig];
        veff.step(ws);

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

            const AccReal weight =
                prob.kg.weights[static_cast<std::size_t>(ik)];
            for (int b = 0; b < n_occ[s]; ++b)
                e_band += weight * ws.eigenvalues[b];

            ws.gvec = saved_gvec;
            ws.n_g = ng0;
        }
    }

#ifdef ATV_HAS_MPI
    if (layout.active) {
        MPI_Allreduce(MPI_IN_PLACE, &e_band, 1, mpi_real_type(), MPI_SUM,
                      MPI_COMM_WORLD);
    }
#endif

    ws.gvec = &gv0;
    ws.n_g = ng0;
    auto terms = iteration_energy(ws, e_band, prob.ew_total, prob.vol_bohr);
    // Collinear: replace Parseval ∫ρ V_xc (last channel) with e_xc_int_spin.
    terms.e_one = e_band - AccReal(2.0) * terms.e_h - ws.e_xc_int_spin;
    terms.total =
        terms.e_one + terms.e_h + ws.E_xc + prob.ew_total + ws.E_aug;
    return static_cast<double>(terms.total) / qe_units::rydberg_to_hartree;
}

/// Non-self-consistent free / total energy [Ry] at `frame` with remapped
/// density from the reference SCF (`rho_r0` at volume `omega0_bohr`).
///
/// Fixed occupations: band sum with f=2. Restricted metallic smear: shared
/// μ over the spectrum, fractional f_b∈[0,2], Mermin free energy
/// F = E_KS − σ·S (σ = degauss in Ha).
double nscf_energy_ry(const cpu::Frame& frame, const PW& method,
                      const std::vector<AccReal>& rho_r0, AccReal omega0_bohr,
                      PoissonSolver& poisson, XCBuilder& xc, VeffBuilder& veff,
                      DavidsonSolver& davidson, int n_bands) {
    PWProblem prob;
    setup_pw_problem(frame, method.upf_path, method.gth_name,
                     method.ecut_ry, method.k_mesh,
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
    const bool use_smear = (method.smearing != Smearing::none);
    AccReal e_band = 0;
    AccReal entropy_sum = AccReal(0);
    auto* saved_gvec = ws.gvec;
    auto& gv0 = prob.gv0;

    if (use_smear) {
        SmearKind smear_kind = SmearKind::fermi_dirac;
        if (method.smearing == Smearing::methfessel_paxton)
            smear_kind = SmearKind::methfessel_paxton;
        else if (method.smearing == Smearing::gaussian)
            smear_kind = SmearKind::gaussian;
        const AccReal degauss_ha =
            static_cast<AccReal>(method.degauss) * AccReal(0.5);
        const int mp_order = method.mp_order;

        std::vector<std::vector<AccReal>> evals_k(
            static_cast<std::size_t>(n_kpts));
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

            evals_k[static_cast<std::size_t>(ik)].assign(
                ws.eigenvalues.begin(),
                ws.eigenvalues.begin()
                    + std::min(n_bands,
                               static_cast<int>(ws.eigenvalues.size())));
            if (static_cast<int>(evals_k[static_cast<std::size_t>(ik)].size())
                > n_bands)
                evals_k[static_cast<std::size_t>(ik)].resize(
                    static_cast<std::size_t>(n_bands));

            ws.gvec = saved_gvec;
            ws.n_g = gv0.count();
        }

#ifdef ATV_HAS_MPI
        if (layout.active) {
            std::vector<AccReal> pack(
                static_cast<std::size_t>(n_kpts * n_bands), AccReal(0));
            for (int ik = layout.rank; ik < n_kpts; ik += layout.nprocs) {
                const auto& ev = evals_k[static_cast<std::size_t>(ik)];
                for (int b = 0;
                     b < n_bands && b < static_cast<int>(ev.size()); ++b)
                    pack[static_cast<std::size_t>(ik * n_bands + b)] =
                        ev[static_cast<std::size_t>(b)];
            }
            MPI_Allreduce(MPI_IN_PLACE, pack.data(), n_kpts * n_bands,
                          mpi_real_type(), MPI_SUM, MPI_COMM_WORLD);
            for (int ik = 0; ik < n_kpts; ++ik) {
                evals_k[static_cast<std::size_t>(ik)].resize(
                    static_cast<std::size_t>(n_bands));
                for (int b = 0; b < n_bands; ++b)
                    evals_k[static_cast<std::size_t>(ik)][b] =
                        pack[static_cast<std::size_t>(ik * n_bands + b)];
            }
        }
#endif

        std::vector<AccReal> all_evals;
        std::vector<AccReal> all_weights;
        all_evals.reserve(static_cast<std::size_t>(n_kpts * n_bands));
        all_weights.reserve(static_cast<std::size_t>(n_kpts * n_bands));
        for (int ik = 0; ik < n_kpts; ++ik) {
            const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
            const auto& ev = evals_k[static_cast<std::size_t>(ik)];
            for (int b = 0; b < n_bands && b < static_cast<int>(ev.size());
                 ++b) {
                all_evals.push_back(ev[static_cast<std::size_t>(b)]);
                all_weights.push_back(w);
            }
        }
        const AccReal mu_ha =
            find_fermi_level(all_evals, all_weights, AccReal(prob.n_elec),
                             degauss_ha, smear_kind, mp_order);

        for (int ik = 0; ik < n_kpts; ++ik) {
            const AccReal w = prob.kg.weights[static_cast<std::size_t>(ik)];
            const auto& ev = evals_k[static_cast<std::size_t>(ik)];
            std::vector<AccReal> w_one(ev.size(), w);
            auto fill = fill_occupations(ev, w_one, mu_ha, degauss_ha,
                                         smear_kind, mp_order);
            entropy_sum += fill.entropy;
            for (std::size_t b = 0; b < ev.size(); ++b) {
                const AccReal f_b =
                    smear_f_band(ev[b], mu_ha, degauss_ha, smear_kind, mp_order);
                e_band += w * ev[b] * f_b;
            }
        }
    } else {
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
    }

    // Restore Γ sphere for energy Parseval (ρ, V_H, V_xc, V_loc live there).
    ws.gvec = &gv0;
    ws.n_g = gv0.count();
    const auto terms =
        iteration_energy(ws, e_band, prob.ew_total, prob.vol_bohr);
    AccReal e_ha = terms.total;
    if (use_smear) {
        const AccReal degauss_ha =
            static_cast<AccReal>(method.degauss) * AccReal(0.5);
        e_ha += -degauss_ha * entropy_sum;
    }
    return static_cast<double>(e_ha) / qe_units::rydberg_to_hartree;
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

    const bool use_smear = (method.smearing != Smearing::none);
    if (use_smear && !(method.degauss > 0.0))
        throw std::invalid_argument(
            "pw_stress: degauss must be > 0 [Ry] when smearing is enabled");
    // Collinear fixed-occ stress is landed (LSDA/PBE); spin∧smear is a deeper
    // product row (cuda-pw-spin-smear-stress / joint free-energy stress).
    if (method.spin_polarized && use_smear)
        throw std::invalid_argument(
            "pw_stress: spin_polarized with metallic smearing has no analytic "
            "stress path yet — use fixed-occupation collinear stress, or "
            "pw_stress_fd for multi-SCF FD of F");

    const int n = frame.nlocal();
    int n_elec = 0;
    {
        const int* types = frame.span(kAtom_type);
        for (int i = 0; i < n; ++i)
            n_elec += valence_from_z(types[i]);
    }

    int n_alpha = 0;
    int n_beta = 0;
    int n_bands = std::max(n_elec / 2, 1);
    if (method.spin_polarized) {
        if (method.multiplicity < 1)
            throw std::invalid_argument(
                "pw_stress: multiplicity must be >= 1 when spin_polarized");
        const int mag = method.multiplicity - 1;
        if ((n_elec + mag) % 2 != 0 || mag > n_elec)
            throw std::invalid_argument(
                "pw_stress: n_elec and multiplicity have incompatible parity");
        n_alpha = (n_elec + mag) / 2;
        n_beta = (n_elec - mag) / 2;
        n_bands = std::max(std::max(n_alpha, n_beta), 1);
    } else if (use_smear) {
        const int n_occ_ceil = std::max((n_elec + 1) / 2, 1);
        n_bands = n_occ_ceil + std::max(method.extra_bands, 0);
    } else if (n_elec % 2 != 0) {
        throw std::invalid_argument(
            "pw_stress: " + std::to_string(n_elec)
            + " valence electrons is not closed-shell — use spin_polarized "
              "for open shell, or set smearing for metallic occupations");
    }

    PoissonSolver poisson;
    XCBuilder xc;
    VeffBuilder veff;
    DensityBuilder density;
    DavidsonSolver davidson{n_bands, AccReal(method.tol) * AccReal(0.1), 40};

    PWProblem scf_prob;
    PWResult res;
    if (method.spin_polarized) {
        res = run_collinear_scf(frame, method, scf_prob, n_alpha, n_beta);
        // Local builders inside run_collinear_scf are gone; prepare ours for
        // NSCF strain samples (same pattern as pw_forces_analytic spin path).
        xc.prepare(scf_prob.ws);
        veff.prepare(scf_prob.ws);
        davidson.prepare(scf_prob.ws);
    } else if (use_smear) {
        res = run_scf_smear_restricted(frame, method, scf_prob, poisson, xc,
                                       veff, density, davidson, n_bands,
                                       "stress");
    } else {
        res = run_scf_fixed_occ(frame, method, scf_prob, poisson, xc, veff,
                                density, davidson, "stress");
    }
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

    std::vector<AccReal> rho_r0;
    std::vector<AccReal> rho_a0;
    std::vector<AccReal> rho_b0;
    if (method.spin_polarized) {
        rho_a0 = rho_channel_g_to_real(ws, ws.rho_spin_G[0]);
        rho_b0 = rho_channel_g_to_real(ws, ws.rho_spin_G[1]);
        // Recompute SCF energy with e_xc_int_spin (run_collinear_scf leaves a
        // diagnostic restricted formula in res.energy).
        if (energy_ry_out) {
            *energy_ry_out = nscf_energy_spin_ry(
                frame, method, rho_a0, rho_b0, omega0, poisson, xc, veff,
                davidson, n_bands, n_alpha, n_beta);
        }
    } else {
        rho_r0 = rho_g_to_real(ws);
    }

    // Hellmann–Feynman strain-FD of the NSCF free / total energy (one SCF).
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

        double ep_ry = 0.0;
        double em_ry = 0.0;
        if (method.spin_polarized) {
            ep_ry = nscf_energy_spin_ry(fp, method, rho_a0, rho_b0, omega0,
                                        poisson, xc, veff, davidson, n_bands,
                                        n_alpha, n_beta);
            em_ry = nscf_energy_spin_ry(fm, method, rho_a0, rho_b0, omega0,
                                        poisson, xc, veff, davidson, n_bands,
                                        n_alpha, n_beta);
        } else {
            ep_ry = nscf_energy_ry(fp, method, rho_r0, omega0, poisson, xc, veff,
                                   davidson, n_bands);
            em_ry = nscf_energy_ry(fm, method, rho_r0, omega0, poisson, xc, veff,
                                   davidson, n_bands);
        }
        const double dE_dev = (ep_ry - em_ry) * ry_to_ev / (2.0 * h);
        // Mechanical / CP2K: σ = −(1/Ω) ∂E/∂ε (matches GPW cell relax).
        // Under smear E is the Mermin free energy F.
        stress[static_cast<std::size_t>(v)] = -dE_dev / vol;
    }
    return stress;
}

} // namespace internal
} // namespace pw
} // namespace cpu
} // namespace ATV
