/// @file cuda/pw/pw_scf.cu
/// @brief cuda::pw::pw_scf() — GPU plane-wave SCF orchestrator.
///
/// Mirrors the CPU SCF loop in src/cpu/pw/pw_solver.cpp. Shared host steps
/// (problem assembly, Poisson, LDA/XC, V_eff, Pulay mixing, convergence,
/// energy report) are the exact same code as the CPU backend; only the
/// band solve + density accumulation are replaced by the device solver.

#include "atomiverse/cpu/pw/internal/convergence.h"
#include "atomiverse/cpu/pw/internal/g_vector.h"
#include "atomiverse/cpu/pw/internal/poisson.h"
#include "atomiverse/cpu/pw/internal/pw_setup.h"
#include "atomiverse/cpu/pw/internal/pw_workspace.h"
#include "atomiverse/cpu/pw/internal/veff_builder.h"
#include "atomiverse/cpu/pw/internal/xc_builder.h"
#include "atomiverse/cuda/pw/internal/device_solver.cuh"
#include "atomiverse/cuda/pw/pw_scf.cuh"
#include <algorithm>
#include <complex>
#include <iostream>
#include <vector>

namespace ATV {
namespace cuda {
namespace pw {

PWResult pw_scf(const cpu::Frame& frame, const std::string& upf_path,
                double ecut_ry, int k_mesh, double tol, int max_iter) {
    using cpu::pw::PWWorkspace;
    namespace cpw = cpu::pw;

    PWResult result;

    cpw::internal::PWProblem prob;
    cpw::internal::setup_pw_problem(frame, upf_path, ecut_ry, k_mesh, prob);
    PWWorkspace& ws = prob.ws;
    auto& gv0 = prob.gv0;
    auto& gv_k = prob.gv_k;
    int n_kpts = static_cast<int>(prob.kg.kpoints.size());
    int n_bands = prob.n_bands;
    Real vol_bohr = prob.vol_bohr;

    std::cout << "  [cuda::pw] device band solver on " << ws.fft_nx << "x"
              << ws.fft_ny << "x" << ws.fft_nz << " grid" << std::endl;

    internal::DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz, n_bands);

    // === SCF loop ===
    // The reused CPU host steps are the facade-private step helpers
    // (prepared once, stepped per iteration — lifecycle rule); the
    // ConvergenceCheck warmup counter persists across iterations.
    cpw::internal::PoissonSolver po;
    cpw::internal::XCBuilder lda;
    lda.prepare(ws);
    cpw::internal::VeffBuilder v;
    v.prepare(ws);
    cpw::internal::ConvergenceCheck conv{Real(tol), 2};

    bool converged = false;
    for (int iter = 0; iter < max_iter; ++iter) {
        po.step(ws);
        lda.step(ws);
        v.step(ws);

        solver.set_veff(ws.V_eff_r.data());
        solver.zero_density();

        Real e_band = 0;
        for (int ik = 0; ik < n_kpts; ++ik) {
            int n_g_k = gv_k[ik].count();
            auto* saved_gvec = ws.gvec;
            ws.gvec = &gv_k[ik];
            ws.n_g = n_g_k;
            // Nonlocal/wfc tables live on q = k+G: bind this k's tables.
            if (!prob.kb_proj_k.empty())
                ws.kb_proj = prob.kb_proj_k[ik];
            if (!prob.atom_wfc_k.empty())
                ws.atom_wfc = prob.atom_wfc_k[ik];
            // Kinetic on q = k+G, sized to this k-sphere (matches the CPU loop).
            ws.T_G.resize({static_cast<std::size_t>(n_g_k)});
            for (int ig = 0; ig < n_g_k; ++ig)
                ws.T_G[ig] = gv_k[ik].ekin[ig];

            solver.set_kpoint(ws);
            solver.solve(ws, Real(tol) * Real(0.1), 40);

            Real weight = prob.kg.weights[ik];
            for (int b = 0; b < n_bands; ++b)
                e_band += weight * ws.eigenvalues[b] * Real(2);

            solver.accumulate_density(weight, ws.omega);

            ws.gvec = saved_gvec;
            ws.n_g = gv0.count();
        }

        // New density: device FFT + gather on the Γ-sphere, then the
        // shared host Pulay mixing (same history policy as CPU).
        for (int i = 0; i < ws.n_g; ++i)
            ws.rho_G_old[i] = ws.rho_G[i];
        solver.fetch_density_g(gv0, ws.rho_G.data());
        cpw::internal::pulay_mix(ws);

        converged = conv.step(ws);

        result.iterations = iter + 1;

        const auto terms = cpw::internal::iteration_energy(
            ws, e_band, prob.ew_total, vol_bohr);
        cpw::internal::log_iteration_energy(terms, ws, prob.ew_total, iter);
        result.energy = static_cast<double>(terms.total) * 2.0;

        if (converged) {
            result.converged = true;
            std::cout << "\n>>> Converged in " << (iter + 1) << " iters"
                      << std::endl;
            break;
        }
    }

    if (!result.converged)
        std::cout << "\n>>> NOT converged after " << max_iter << " iters"
                  << std::endl;

    return result;
}

} // namespace pw
} // namespace cuda
} // namespace ATV
