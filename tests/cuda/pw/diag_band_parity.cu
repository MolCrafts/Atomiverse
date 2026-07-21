/// @file tests/cuda/pw/diag_band_parity.cu
/// @brief Diagnostic (not a registered test): fixed-V_eff band-solver
/// parity, CPU internal::DavidsonSolver vs cuda DeviceBandSolver, per
/// k-point.
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/pw/internal/davidson.h"
#include "atomiverse/cpu/pw/internal/poisson.h"
#include "atomiverse/cpu/pw/internal/pw_setup.h"
#include "atomiverse/cpu/pw/internal/veff_builder.h"
#include "atomiverse/cpu/pw/internal/xc_builder.h"
#include "atomiverse/cuda/pw/internal/device_solver.cuh"
#include <cstdio>
#include <cstdlib>

using namespace ATV;
using namespace ATV::cpu::pw;

int main(int argc, char** argv) {
    const char* upf = (argc > 1) ? argv[1] : "";
    double ecut = (argc > 2) ? atof(argv[2]) : 12.0;
    cpu::Frame f(2);
    f.atoms.type[0] = f.atoms.type[1] = 14;
    f.atoms.mass[0] = f.atoms.mass[1] = 28.0855;
    f.atoms.x[0] = f.atoms.y[0] = f.atoms.z[0] = 0;
    double a = 5.43, h = a / 2;
    f.atoms.x[1] = f.atoms.y[1] = f.atoms.z[1] = a / 4;
    Mat3 cell;
    cell(0, 0) = 0;
    cell(0, 1) = h;
    cell(0, 2) = h;
    cell(1, 0) = h;
    cell(1, 1) = 0;
    cell(1, 2) = h;
    cell(2, 0) = h;
    cell(2, 1) = h;
    cell(2, 2) = 0;
    f.box = cell;

    internal::PWProblem prob;
    internal::setup_pw_problem(f, upf, ecut, 2, prob);
    auto& ws = prob.ws;
    internal::PoissonSolver po;
    po.step(ws);
    internal::XCBuilder lda;
    lda.prepare(ws);
    lda.step(ws);
    internal::VeffBuilder v;
    v.prepare(ws);
    v.step(ws);

    int nb = prob.n_bands;
    bool fresh = (argc > 3); // any 3rd arg: new solver per k-point
    cuda::pw::internal::DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz,
                                                nb);
    solver.set_veff(ws.V_eff_r.data());

    double max_diff = 0;
    for (size_t ik = 0; ik < prob.gv_k.size(); ++ik) {
        ws.gvec = &prob.gv_k[ik];
        ws.n_g = prob.gv_k[ik].count();
        if (!prob.kb_proj_k.empty())
            ws.kb_proj = prob.kb_proj_k[ik];
        if (!prob.atom_wfc_k.empty())
            ws.atom_wfc = prob.atom_wfc_k[ik];

        // Kinetic on q = k+G, sized to this k-sphere.
        ws.T_G.resize({static_cast<std::size_t>(ws.n_g)});
        for (int ig = 0; ig < ws.n_g; ++ig)
            ws.T_G[ig] = ws.gvec->ekin[ig];
        internal::DavidsonSolver d{nb, 1e-9, 40};
        d.prepare(ws);
        d.step(ws);
        std::printf("k%zu CPU :", ik);
        std::vector<Real> e_cpu(nb);
        for (int b = 0; b < nb; ++b) {
            e_cpu[b] = ws.eigenvalues[b];
            std::printf(" %.6f", e_cpu[b]);
        }
        std::printf("\n");
        std::vector<std::complex<Real>> wfc_cpu_b0(ws.n_g);
        for (int ig = 0; ig < ws.n_g; ++ig)
            wfc_cpu_b0[ig] = ws.wfc(0, ig);

        if (fresh) {
            cuda::pw::internal::DeviceBandSolver s2(ws.fft_nx, ws.fft_ny,
                                                    ws.fft_nz, nb);
            s2.set_veff(ws.V_eff_r.data());
            s2.set_kpoint(ws);
            s2.solve(ws, Real(1e-9), 40);
        } else {
            solver.set_kpoint(ws);
            solver.solve(ws, Real(1e-9), 40);
        }
        std::printf("k%zu CUDA:", ik);
        for (int b = 0; b < nb; ++b) {
            std::printf(" %.6f", ws.eigenvalues[b]);
            double diff = std::abs(ws.eigenvalues[b] - e_cpu[b]);
            if (diff > max_diff)
                max_diff = diff;
        }
        std::printf("\n");

        // Ground-truth arbiter: residual norm of the CPU band-0 eigenpair
        // computed with the device H (validated identical to CPU H in the
        // round-by-round trace).
        {
            int ngk = ws.n_g;
            std::vector<std::complex<Real>> psi(ngk), hpsi(ngk);
            for (int ig = 0; ig < ngk; ++ig)
                psi[ig] = wfc_cpu_b0[ig];
            cuda::pw::internal::DeviceBandSolver s3(ws.fft_nx, ws.fft_ny,
                                                    ws.fft_nz, nb);
            s3.set_veff(ws.V_eff_r.data());
            s3.set_kpoint(ws);
            s3.apply_h_host(psi.data(), hpsi.data());
            double r2 = 0, n2 = 0;
            for (int ig = 0; ig < ngk; ++ig) {
                auto r = hpsi[ig] - std::complex<Real>(e_cpu[0]) * psi[ig];
                r2 += std::norm(r);
                n2 += std::norm(psi[ig]);
            }
            std::printf("k%zu CPU-eig0 residual=%.3e (|psi|=%.3f)\n", ik,
                        std::sqrt(r2), std::sqrt(n2));
        }
    }
    std::printf("MAX_EIG_DIFF=%.3e Ha\n", max_diff);
    return max_diff < 1e-5 ? 0 : 1;
}
