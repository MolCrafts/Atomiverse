/// @file tests/cuda/planewave/diag_band_parity.cu
/// @brief Diagnostic (not a registered test): fixed-V_eff band-solver
/// parity, CPU internal::DavidsonSolver vs cuda DeviceBandSolver, per
/// k-point.
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/internal/davidson.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/veff_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {
/// The band solver no longer owns V_eff(r) or the density accumulator — they
/// belong to `cuda::pw::internal::DeviceDensity` in production. This
/// diagnostic stands in for it with two plain device buffers so the band
/// comparison stays a band comparison.
struct BorrowedGrids {
    double* d_veff = nullptr;
    double* d_rho = nullptr;

    explicit BorrowedGrids(int nr, const AccReal* host_veff) {
        cudaMalloc(&d_veff, static_cast<size_t>(nr) * sizeof(double));
        cudaMalloc(&d_rho, static_cast<size_t>(nr) * sizeof(double));
        cudaMemcpy(d_veff, host_veff, static_cast<size_t>(nr) * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemset(d_rho, 0, static_cast<size_t>(nr) * sizeof(double));
    }
    ~BorrowedGrids() {
        cudaFree(d_veff);
        cudaFree(d_rho);
    }
    BorrowedGrids(const BorrowedGrids&) = delete;
    BorrowedGrids& operator=(const BorrowedGrids&) = delete;
};

/// @brief Bind `ws` to k-sphere @p ik for the CPU reference solver.
///
/// The device solver gets its k-point from `bind_kpoint`; the host
/// `DavidsonSolver` this diagnostic compares against still reads the
/// workspace, so the rebinding lives on here.
void rebind_k(const internal::PWProblem& prob, internal::PWWorkspace& ws,
              size_t ik) {
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
}
} // namespace

int main(int argc, char** argv) {
    const char* upf = (argc > 1) ? argv[1] : "";
    double ecut = (argc > 2) ? atof(argv[2]) : 12.0;
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 14;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 28.0855;
    f.span(kAtom_x)[0] = f.span(kAtom_y)[0] = f.span(kAtom_z)[0] = 0;
    double a = 5.43, h = a / 2;
    f.span(kAtom_x)[1] = f.span(kAtom_y)[1] = f.span(kAtom_z)[1] = a / 4;
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
    f.box.set_cell(cell);

    internal::PWProblem prob;
    // Signature: (frame, upf_path, gth_name, ecut_ry, k_mesh, out). Empty
    // gth_name selects the UPF path (exactly one PP source required).
    internal::setup_pw_problem(f, upf, /*gth_name=*/"", ecut, /*k_mesh=*/2,
                               prob);
    auto& ws = prob.ws;
    internal::PoissonSolver po;
    po.step(ws);
    internal::XCBuilder lda;
    lda.prepare(ws);
    // LSDA explicitly: XCBuilder::step defaults to PBE. The functional is
    // irrelevant to this band-solver harness — both legs diagonalize the
    // SAME fixed V_eff — so keep the cheaper local one.
    lda.step(ws, cpu::scf::XCType::LSDA);
    internal::VeffBuilder v;
    v.prepare(ws);
    v.step(ws);

    int nb = prob.n_bands;
    bool fresh = (argc > 3); // any 3rd arg: new solver per k-point
    cuda::pw::internal::DeviceBandSolver solver(ws.fft_nx, ws.fft_ny, ws.fft_nz,
                                                nb);
    BorrowedGrids grids(ws.nr, ws.V_eff_r.data());
    solver.bind_grids(grids.d_rho, grids.d_veff);
    // Per-k device tables are geometry: built once, then bound by index.
    // (Leaves ws bound to the density sphere; the loop below rebinds it for
    // the CPU reference solver anyway.)
    solver.build_kpoints(prob);

    double max_diff = 0;
    for (size_t ik = 0; ik < prob.gv_k.size(); ++ik) {
        rebind_k(prob, ws, ik);
        internal::DavidsonSolver d{nb, 1e-9, 40};
        d.prepare(ws);
        d.step(ws);
        std::printf("k%zu CPU :", ik);
        std::vector<AccReal> e_cpu(nb);
        for (int b = 0; b < nb; ++b) {
            e_cpu[b] = ws.eigenvalues[b];
            std::printf(" %.6f", e_cpu[b]);
        }
        std::printf("\n");
        std::vector<std::complex<AccReal>> wfc_cpu_b0(ws.n_g);
        for (int ig = 0; ig < ws.n_g; ++ig)
            wfc_cpu_b0[ig] = ws.wfc(0, ig);

        if (fresh) {
            cuda::pw::internal::DeviceBandSolver s2(ws.fft_nx, ws.fft_ny,
                                                    ws.fft_nz, nb);
            s2.bind_grids(grids.d_rho, grids.d_veff);
            s2.build_kpoints(prob);
            s2.bind_kpoint(static_cast<int>(ik));
            s2.solve(ws, 40);
            // build_kpoints rebinds ws while it walks the k-spheres; put it
            // back on this k for the CPU-side reads that follow.
            rebind_k(prob, ws, ik);
        } else {
            solver.bind_kpoint(static_cast<int>(ik));
            solver.solve(ws, 40);
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
            std::vector<std::complex<AccReal>> psi(ngk), hpsi(ngk);
            for (int ig = 0; ig < ngk; ++ig)
                psi[ig] = wfc_cpu_b0[ig];
            cuda::pw::internal::DeviceBandSolver s3(ws.fft_nx, ws.fft_ny,
                                                    ws.fft_nz, nb);
            s3.bind_grids(grids.d_rho, grids.d_veff);
            s3.build_kpoints(prob);
            s3.bind_kpoint(static_cast<int>(ik));
            s3.apply_h_host(psi.data(), hpsi.data());
            rebind_k(prob, ws, ik);
            double r2 = 0, n2 = 0;
            for (int ig = 0; ig < ngk; ++ig) {
                auto r = hpsi[ig] - std::complex<AccReal>(e_cpu[0]) * psi[ig];
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
