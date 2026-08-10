/// @file cuda/planewave/internal/pw_stress.cu
/// @brief Device Nielsen–Martin analytic cell stress (one SCF + NSCF strain-FD).

#include "atomiverse/cuda/planewave/internal/pw_stress.cuh"

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/planewave/internal/convergence.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/qe_units.h"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"

#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {
namespace internal {
namespace {

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

/// Device SCF at `frame`; fills `energy_ry_out` and returns host ρ(r) + Ω.
struct RefScf {
    std::vector<double> rho_r;
    double omega_bohr = 0;
    double energy_ry = 0;
    int nr = 0;
    int iterations = 0;
    bool converged = false;
};

RefScf run_device_scf(const cpu::Frame& frame, const std::string& upf_path,
                     const std::string& gth_name, double ecut_ry, int k_mesh,
                     double tol, int max_iter, double mix_beta, int pulay_depth,
                     cpu::scf::XCType xc) {
    cpu::pw::internal::PWProblem prob;
    cpu::pw::internal::setup_pw_problem(frame, upf_path, gth_name, ecut_ry,
                                        k_mesh, prob);
    cpu::pw::internal::PWWorkspace& ws = prob.ws;
    if (prob.n_elec % 2 != 0)
        throw std::invalid_argument(
            "cuda::pw::pw_stress: "
            + std::to_string(prob.n_elec)
            + " valence electrons is not closed-shell — use cpu::pw for "
              "open-shell / metallic paths");

    ws.mix_beta = AccReal(mix_beta);
    ws.pulay_depth = pulay_depth;

    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    const int n_bands = prob.n_bands;
    const AccReal vol_bohr = prob.vol_bohr;
    const AccReal omega_bohr = ws.omega;

    DeviceDensity den(prob, mix_beta, pulay_depth, xc);
    DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz, n_bands,
                            den.stream());
    solver.bind_grids(den.density_accumulator(), den.effective_potential());
    solver.build_kpoints(prob);

    cpu::pw::internal::ConvergenceCheck conv{AccReal(tol), 2};
    AccReal last_energy_ha = AccReal(0);
    int last_iter = 0;
    bool converged = false;
    for (int iter = 0; iter < max_iter; ++iter) {
        den.hartree();
        den.exchange_correlation();
        den.assemble_veff();
        den.zero_accumulator();

        AccReal e_band = 0;
        for (int ik = 0; ik < n_kpts; ++ik) {
            solver.bind_kpoint(ik);
            solver.solve(ws, 40);
            const AccReal weight = prob.kg.weights[ik];
            for (int b = 0; b < n_bands; ++b)
                e_band += weight * ws.eigenvalues[b] * AccReal(2);
            solver.accumulate_density(weight, omega_bohr);
        }

        den.collect();
        den.mix(iter);
        const auto s = den.reduce(static_cast<double>(vol_bohr));
        ws.E_xc = AccReal(s.e_xc);
        const auto terms = cpu::pw::internal::combine_energy_terms(
            e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
            ws.E_xc, prob.ew_total, ws.E_aug);
        last_energy_ha = terms.total;
        last_iter = iter + 1;
        if (conv.step(AccReal(s.resid))) {
            converged = true;
            break;
        }
    }

    RefScf out;
    out.nr = den.nr();
    out.omega_bohr = den.omega();
    out.rho_r.resize(static_cast<std::size_t>(out.nr));
    den.fetch_rho_r(out.rho_r.data());
    out.energy_ry = static_cast<double>(last_energy_ha)
                    / cpu::pw::qe_units::rydberg_to_hartree;
    out.iterations = last_iter;
    out.converged = converged;
    return out;
}

/// Non-self-consistent total energy [Ry] with remapped density from ref SCF.
double nscf_energy_ry(const cpu::Frame& frame, const std::string& upf_path,
                      const std::string& gth_name, double ecut_ry, int k_mesh,
                      double mix_beta, int pulay_depth, cpu::scf::XCType xc,
                      const std::vector<double>& rho_r0, double omega0_bohr) {
    cpu::pw::internal::PWProblem prob;
    cpu::pw::internal::setup_pw_problem(frame, upf_path, gth_name, ecut_ry,
                                        k_mesh, prob);
    cpu::pw::internal::PWWorkspace& ws = prob.ws;
    ws.mix_beta = AccReal(mix_beta);
    ws.pulay_depth = pulay_depth;

    const int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    const int n_bands = prob.n_bands;
    const AccReal vol_bohr = prob.vol_bohr;
    const double scale = omega0_bohr / static_cast<double>(ws.omega);

    DeviceDensity den(prob, mix_beta, pulay_depth, xc);
    DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz, n_bands,
                            den.stream());
    solver.bind_grids(den.density_accumulator(), den.effective_potential());
    solver.build_kpoints(prob);

    den.load_rho_r_scaled(rho_r0.data(), static_cast<int>(rho_r0.size()),
                          scale);
    den.hartree();
    den.exchange_correlation();
    den.assemble_veff();

    AccReal e_band = 0;
    for (int ik = 0; ik < n_kpts; ++ik) {
        solver.bind_kpoint(ik);
        solver.solve(ws, 40);
        const AccReal weight = prob.kg.weights[ik];
        for (int b = 0; b < n_bands; ++b)
            e_band += weight * ws.eigenvalues[b] * AccReal(2);
    }

    const auto s = den.reduce(static_cast<double>(vol_bohr));
    const auto terms = cpu::pw::internal::combine_energy_terms(
        e_band, AccReal(s.e_h), AccReal(s.e_xc_int), AccReal(s.e_loc),
        AccReal(s.e_xc), prob.ew_total, ws.E_aug);
    return static_cast<double>(terms.total)
           / cpu::pw::qe_units::rydberg_to_hartree;
}

} // namespace

std::array<double, 6> pw_stress_analytic_impl(
    const cpu::Frame& frame, const std::string& upf_path,
    const std::string& gth_name, double ecut_ry, int k_mesh, double tol,
    int max_iter, double mix_beta, int pulay_depth, cpu::scf::XCType xc,
    double* energy_ry_out) {
    if (frame.nlocal() <= 0)
        throw std::invalid_argument("cuda::pw::pw_stress: frame has no atoms");
    if (frame.box.volume() <= 0)
        throw std::invalid_argument(
            "cuda::pw::pw_stress: non-positive cell volume");

    const RefScf ref =
        run_device_scf(frame, upf_path, gth_name, ecut_ry, k_mesh, tol,
                      max_iter, mix_beta, pulay_depth, xc);
    if (energy_ry_out)
        *energy_ry_out = ref.energy_ry;
    if (!ref.converged) {
        // No core/log.h from .cu — printf matches forces-scf breadcrumb.
        std::printf("[pw] cuda pw_stress: SCF not converged after %d iters; "
                    "using last density for HF stress (E=%.8f Ry)\n",
                    ref.iterations, ref.energy_ry);
    } else {
        std::printf("[pw] cuda stress-scf done iter=%d energy_ry=%.8f\n",
                    ref.iterations, ref.energy_ry);
    }

    // Hellmann–Feynman strain-FD of the NSCF energy (one SCF total).
    // Same step as CPU `pw_stress_analytic_impl`.
    constexpr double h = 2e-3;
    const double vol = static_cast<double>(frame.box.volume());
    const double ry_to_ev = cpu::pw::qe_units::rydberg_to_ev;

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

        const double ep_ry =
            nscf_energy_ry(fp, upf_path, gth_name, ecut_ry, k_mesh, mix_beta,
                           pulay_depth, xc, ref.rho_r, ref.omega_bohr);
        const double em_ry =
            nscf_energy_ry(fm, upf_path, gth_name, ecut_ry, k_mesh, mix_beta,
                           pulay_depth, xc, ref.rho_r, ref.omega_bohr);
        const double dE_dev = (ep_ry - em_ry) * ry_to_ev / (2.0 * h);
        // Mechanical / CP2K: σ = −(1/Ω) ∂E/∂ε.
        stress[static_cast<std::size_t>(v)] = -dE_dev / vol;
    }
    return stress;
}

} // namespace internal
} // namespace pw
} // namespace cuda
} // namespace ATV
