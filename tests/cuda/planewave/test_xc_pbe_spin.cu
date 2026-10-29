/// @file tests/cuda/planewave/test_xc_pbe_spin.cu
/// @brief Verifies cuda::xc::xc_pbe_spin_grid against
///        cpu::xc::evaluate_xc(PBE, ρ_α, ρ_β, σ_αα, σ_αβ, σ_ββ) plus the
///        spin flux assembly of XCBuilder::step_spin.
#include "atomiverse/cpu/xc/xc_functional.h"
#include "atomiverse/cuda/xc/xc_pbe.cuh"
#include <array>
#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using ATV::cpu::xc::evaluate_xc;
using ATV::cpu::xc::XCType;
using ATV::cuda::xc::xc_pbe_spin_grid;

namespace {

void* dmalloc(size_t bytes) {
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    return p;
}

void host_ref(XCType type, double ra, double rb, double gax, double gay,
              double gaz, double gbx, double gby, double gbz, double& vxa,
              double& vxb, double& exc, double& fax, double& fay, double& faz,
              double& fbx, double& fby, double& fbz) {
    const bool active_a = ra > 0.0;
    const bool active_b = rb > 0.0;
    if (ra < 0.0)
        ra = 0.0;
    if (rb < 0.0)
        rb = 0.0;
    const double rtot = ra + rb;
    if (rtot < 1e-12) {
        vxa = vxb = exc = 0.0;
        fax = fay = faz = fbx = fby = fbz = 0.0;
        return;
    }
    double saa = gax * gax + gay * gay + gaz * gaz;
    double sbb = gbx * gbx + gby * gby + gbz * gbz;
    double sab = gax * gbx + gay * gby + gaz * gbz;
    const auto out = evaluate_xc(type, ra, rb, saa, sab, sbb);
    vxa = active_a ? out.vrho_alpha : 0.0;
    vxb = active_b ? out.vrho_beta : 0.0;
    exc = out.energy_density / rtot;
    const double caa = 2.0 * out.vsigma_aa;
    const double cbb = 2.0 * out.vsigma_bb;
    const double cab = out.vsigma_ab;
    fax = caa * gax + cab * gbx;
    fay = caa * gay + cab * gby;
    faz = caa * gaz + cab * gbz;
    fbx = cbb * gbx + cab * gax;
    fby = cbb * gby + cab * gay;
    fbz = cbb * gbz + cab * gaz;
}

void make_sweep(std::vector<double>& ra, std::vector<double>& rb,
                std::vector<double>& gax, std::vector<double>& gay,
                std::vector<double>& gaz, std::vector<double>& gbx,
                std::vector<double>& gby, std::vector<double>& gbz) {
    const double rho_a_vals[] = {0.0, 1e-14, 1e-8, 0.01, 0.2, 1.0, 3.0};
    const double rho_b_vals[] = {0.0, 1e-14, 5e-9, 0.005, 0.1, 0.8, 2.5};
    const double g_vals[] = {0.0, 0.01, 0.5, 2.0};
    for (const double a : rho_a_vals)
        for (const double b : rho_b_vals)
            for (const double g : g_vals) {
                ra.push_back(a);
                rb.push_back(b);
                // Anisotropic spin gradients so all flux lanes fire.
                gax.push_back(0.6 * g);
                gay.push_back(-0.4 * g);
                gaz.push_back(0.3 * g);
                gbx.push_back(-0.2 * g);
                gby.push_back(0.7 * g);
                gbz.push_back(-0.5 * g);
            }
}

} // namespace

class CudaPWXcPbeSpin : public ::testing::TestWithParam<XCType> {};

INSTANTIATE_TEST_SUITE_P(PbeFamily, CudaPWXcPbeSpin,
                         ::testing::Values(XCType::PBE, XCType::PBEsol,
                                           XCType::RevPBE, XCType::RPBE));

TEST_P(CudaPWXcPbeSpin, MatchesCpuEvaluateXc) {
    std::vector<double> ra, rb, gax, gay, gaz, gbx, gby, gbz;
    make_sweep(ra, rb, gax, gay, gaz, gbx, gby, gbz);
    const int nr = static_cast<int>(ra.size());

    auto* d_ra = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_rb = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gax = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gay = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gaz = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gbx = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gby = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_gbz = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vxa = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_vxb = static_cast<double*>(dmalloc(nr * sizeof(double)));
    auto* d_exc = static_cast<double*>(dmalloc(nr * sizeof(double)));

    ASSERT_EQ(cudaMemcpy(d_ra, ra.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_rb, rb.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gax, gax.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gay, gay.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gaz, gaz.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gbx, gbx.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gby, gby.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_gbz, gbz.data(), nr * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);

    xc_pbe_spin_grid(GetParam(), d_ra, d_rb, d_gax, d_gay, d_gaz, d_gbx, d_gby,
                     d_gbz, d_vxa, d_vxb, d_exc, nr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> vxa(nr), vxb(nr), exc(nr);
    std::vector<double> fax(nr), fay(nr), faz(nr), fbx(nr), fby(nr), fbz(nr);
    ASSERT_EQ(cudaMemcpy(vxa.data(), d_vxa, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(vxb.data(), d_vxb, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(exc.data(), d_exc, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(fax.data(), d_gax, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(fay.data(), d_gay, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(faz.data(), d_gaz, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(fbx.data(), d_gbx, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(fby.data(), d_gby, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(fbz.data(), d_gbz, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);

    // Closed-shell PBE uses 1e-12·max(1,|ref|). Full spin-PBE chains three
    // PW92 channels + φ(ζ) derivatives (~2–3× more pow/exp/log), so
    // libdevice vs libm spreads to a few 1e-12 on small v_ρ; 1e-11 absorbs it
    // without masking real algebra bugs (still ≪ energy dual-gate).
    const auto tol = [](double ref) {
        return 1e-11 * std::fmax(1.0, std::abs(ref));
    };
    for (int i = 0; i < nr; ++i) {
        double rvxa, rvxb, re, rfax, rfay, rfaz, rfbx, rfby, rfbz;
        host_ref(GetParam(), ra[i], rb[i], gax[i], gay[i], gaz[i], gbx[i],
                 gby[i], gbz[i], rvxa, rvxb, re, rfax, rfay, rfaz, rfbx, rfby,
                 rfbz);
        EXPECT_NEAR(vxa[i], rvxa, tol(rvxa)) << "i=" << i << " vxa";
        EXPECT_NEAR(vxb[i], rvxb, tol(rvxb)) << "i=" << i << " vxb";
        EXPECT_NEAR(exc[i], re, tol(re)) << "i=" << i << " exc";
        EXPECT_NEAR(fax[i], rfax, tol(rfax)) << "i=" << i << " fax";
        EXPECT_NEAR(fay[i], rfay, tol(rfay)) << "i=" << i << " fay";
        EXPECT_NEAR(faz[i], rfaz, tol(rfaz)) << "i=" << i << " faz";
        EXPECT_NEAR(fbx[i], rfbx, tol(rfbx)) << "i=" << i << " fbx";
        EXPECT_NEAR(fby[i], rfby, tol(rfby)) << "i=" << i << " fby";
        EXPECT_NEAR(fbz[i], rfbz, tol(rfbz)) << "i=" << i << " fbz";
    }

    cudaFree(d_ra);
    cudaFree(d_rb);
    cudaFree(d_gax);
    cudaFree(d_gay);
    cudaFree(d_gaz);
    cudaFree(d_gbx);
    cudaFree(d_gby);
    cudaFree(d_gbz);
    cudaFree(d_vxa);
    cudaFree(d_vxb);
    cudaFree(d_exc);
}

// Differentiate the actual device energy, independently of CPU derivatives:
// d(rho*epsilon)/d rho_sigma = local v_sigma and
// d(rho*epsilon)/d grad_d rho_sigma = flux_sigma,d.
TEST_P(CudaPWXcPbeSpin, DeviceEnergyDerivativesMatchPotentialAndFlux) {
    constexpr int nc = 8;
    constexpr int nr = 1 + 2 * nc;
    constexpr double h = 1e-6;
    const std::array<double, nc> base = {.2, .07, .1, -.2, .08, -.04, .06, .03};
    std::array<std::array<double, nr>, nc> inputs{}, outputs{};
    std::array<double, nr> epsilon{};
    for (int c = 0; c < nc; ++c) {
        inputs[c].fill(base[c]);
        inputs[c][1 + 2 * c] += h;
        inputs[c][2 + 2 * c] -= h;
    }
    std::array<double*, nc> device{};
    std::array<double*, 2> potential{};
    double* d_epsilon = static_cast<double*>(dmalloc(nr * sizeof(double)));
    for (int c = 0; c < nc; ++c) {
        device[c] = static_cast<double*>(dmalloc(nr * sizeof(double)));
        ASSERT_EQ(cudaMemcpy(device[c], inputs[c].data(), nr * sizeof(double),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
    }
    for (auto& p : potential)
        p = static_cast<double*>(dmalloc(nr * sizeof(double)));
    xc_pbe_spin_grid(GetParam(), device[0], device[1], device[2], device[3],
                     device[4], device[5], device[6], device[7], potential[0],
                     potential[1], d_epsilon, nr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(epsilon.data(), d_epsilon, nr * sizeof(double),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    for (int c = 0; c < nc; ++c) {
        const auto* source = c < 2 ? potential[c] : device[c];
        ASSERT_EQ(cudaMemcpy(outputs[c].data(), source, nr * sizeof(double),
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
        const int plus = 1 + 2 * c, minus = 2 + 2 * c;
        const double f_plus =
            (inputs[0][plus] + inputs[1][plus]) * epsilon[plus];
        const double f_minus =
            (inputs[0][minus] + inputs[1][minus]) * epsilon[minus];
        EXPECT_NEAR(outputs[c][0], (f_plus - f_minus) / (2 * h), 5e-9)
            << "density/gradient channel=" << c;
    }
    for (auto* p : device)
        cudaFree(p);
    for (auto* p : potential)
        cudaFree(p);
    cudaFree(d_epsilon);
}

TEST(CudaPWXcPbeSpinAdmission, RejectsUnsupportedBeforeDeviceAccess) {
    for (const auto type :
         {XCType::LSDA, XCType::PADE, XCType::PBE0, XCType::BLYP,
          XCType::Custom, static_cast<XCType>(-1)}) {
        EXPECT_THROW(xc_pbe_spin_grid(type, nullptr, nullptr, nullptr, nullptr,
                                      nullptr, nullptr, nullptr, nullptr,
                                      nullptr, nullptr, nullptr, 0),
                     std::invalid_argument);
    }
}
