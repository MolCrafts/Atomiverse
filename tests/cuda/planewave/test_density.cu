/// @file tests/cuda/planewave/test_density.cu
/// @brief Unit tests for `cuda::pw::internal::DeviceDensity` — each device
///        stage against the CPU stage helper it mirrors.
///
/// UNIT SCOPE: no SCF loop, no Davidson, no convergence loop. Every test
/// drives ONE device method (or a fixed 5-call mixing sequence with synthetic
/// input) and compares it to the actual CPU stage object on the same
/// `PWWorkspace`. The end-to-end CPU/CUDA PW parity solve lives in
/// regressions/pw/pw-cuda-lda-si, never under ctest.
///
/// TOLERANCES ARE NOT ZERO BY DESIGN. cuFFT and FFTW use different butterfly
/// orders, the device reductions are fixed-geometry trees while the CPU sums
/// serially, and libdevice vs libm transcendentals differ by ≲ 1 ulp. Each
/// assertion below states which of those it is absorbing.
#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/internal/density_builder.h"
#include "atomiverse/cpu/planewave/internal/poisson.h"
#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include "atomiverse/cpu/planewave/internal/veff_builder.h"
#include "atomiverse/cpu/planewave/internal/xc_builder.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"

#include <cmath>
#include <complex>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using ATV::AccReal;
using ATV::cuda::pw::internal::DeviceDensity;
using ATV::cuda::pw::internal::GField;
namespace cpupw = ATV::cpu::pw::internal;

namespace {

constexpr double kMixBeta = 0.7;
constexpr int kDepth = 3;

/// 2-atom fcc-primitive Si, a = 5.43 Å — the same cell the PW diagnostics and
/// the pw-lda-si regression use, small enough that `setup_pw_problem` at
/// ecut = 6 Ry / k_mesh = 1 runs in milliseconds.
ATV::cpu::Frame make_si_fcc() {
    ATV::cpu::Frame f(2);
    f.span(ATV::kAtom_type)[0] = f.span(ATV::kAtom_type)[1] = 14;
    f.span(ATV::kAtom_mass)[0] = f.span(ATV::kAtom_mass)[1] = 28.0855;
    f.span(ATV::kAtom_x)[0] = 0;
    f.span(ATV::kAtom_y)[0] = 0;
    f.span(ATV::kAtom_z)[0] = 0;
    const double a = 5.43, h = a / 2;
    f.span(ATV::kAtom_x)[1] = f.span(ATV::kAtom_y)[1] =
        f.span(ATV::kAtom_z)[1] = a / 4;
    ATV::Mat3 cell;
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
    return f;
}

void build_problem(cpupw::PWProblem& prob) {
    ATV::cpu::Frame f = make_si_fcc();
    // Analytic GTH: a real published potential with no file dependency. The
    // density side under test does not touch the pseudopotential beyond
    // V_loc(G), but it must be a physical V_loc(G).
    cpupw::setup_pw_problem(f, "", "GTH-PBE-q4", /*ecut_ry=*/6.0,
                            /*k_mesh=*/1, prob);
}

/// Synthetic real-space density accumulator for pseudo-iteration `t`.
/// Deliberately varied per iteration so successive Pulay residuals are well
/// separated and the augmented DIIS system stays well conditioned.
std::vector<double> synth_accumulation(int nr, int t) {
    std::vector<double> v(static_cast<size_t>(nr));
    for (int i = 0; i < nr; ++i)
        v[static_cast<size_t>(i)] =
            0.030 + 0.004 * t
            + 0.020 * std::sin(0.011 * static_cast<double>(i)
                               + 0.7 * static_cast<double>(t))
            + 0.006 * std::cos(0.003 * static_cast<double>(i * (t + 1)));
    return v;
}

double max_abs(const std::vector<std::complex<double>>& v) {
    double m = 0;
    for (const auto& z : v)
        m = std::max(m, std::abs(z));
    return m;
}

} // namespace

// ── Stage parity: Poisson, LSDA XC, V_eff ────────────────────────────────
TEST(CudaPWDensity, HartreeXcVeffMatchCpu) {
    cpupw::PWProblem prob;
    build_problem(prob);
    auto& ws = prob.ws;
    const int ng = prob.gv0.count();
    const int nr = ws.nr;
    ASSERT_GT(ng, 0);
    ASSERT_GT(nr, 0);

    // Device stages on the atomic seed density uploaded by the constructor.
    DeviceDensity den(prob, kMixBeta, kDepth);
    den.hartree();
    den.exchange_correlation();
    den.assemble_veff();

    // Host stages on the same workspace (they only READ ws.rho_G).
    cpupw::PoissonSolver po;
    po.step(ws);
    cpupw::XCBuilder xcb;
    xcb.prepare(ws);
    xcb.step(ws, ATV::cpu::scf::XCType::LSDA);
    cpupw::VeffBuilder vb;
    vb.prepare(ws);
    vb.step(ws);

    std::vector<std::complex<double>> dev(static_cast<size_t>(ng));

    // V_H(G) = 4π ρ(G)/|G|². Identical inputs; the CPU forms
    // (4π·ρ)/g² and the device (4π/g²)·ρ, so they differ by ≤ 1 ulp.
    den.fetch_g(GField::v_hartree, dev.data());
    double ref_max = 0;
    for (int i = 0; i < ng; ++i)
        ref_max = std::max(ref_max, std::abs(ws.V_H_G[i]));
    for (int i = 0; i < ng; ++i)
        EXPECT_NEAR(std::abs(dev[static_cast<size_t>(i)] - ws.V_H_G[i]), 0.0,
                    1e-13 * std::max(1.0, ref_max))
            << "V_H(G) mismatch at ig=" << i;

    // V_xc(G): two FFTs deep (c2r → pointwise LSDA → r2c) plus the
    // libdevice/libm transcendental spread — absolute tolerance.
    den.fetch_g(GField::v_xc, dev.data());
    for (int i = 0; i < ng; ++i)
        EXPECT_NEAR(std::abs(dev[static_cast<size_t>(i)] - ws.V_xc_G[i]), 0.0,
                    1e-10)
            << "V_xc(G) mismatch at ig=" << i;
    // Im[V_xc(G=0)] must be exactly zero on both backends.
    EXPECT_EQ(dev[0].imag(), 0.0);

    // V_eff(G) = V_loc + V_H + V_xc. This is the assertion that would have
    // caught the real-only V_loc signature: Im[V_loc(G)] is nonzero for the
    // 2-atom fcc basis (structure factor Σ_a e^{-iG·R_a}).
    den.fetch_g(GField::v_eff, dev.data());
    for (int i = 0; i < ng; ++i)
        EXPECT_NEAR(std::abs(dev[static_cast<size_t>(i)] - ws.V_eff_G[i]), 0.0,
                    1e-10)
            << "V_eff(G) mismatch at ig=" << i;

    // Guard the guard: the test cell must actually have a complex V_loc, or
    // the check above proves nothing.
    std::vector<std::complex<double>> vloc(static_cast<size_t>(ng));
    den.fetch_g(GField::v_loc, vloc.data());
    double max_im_vloc = 0;
    for (const auto& z : vloc)
        max_im_vloc = std::max(max_im_vloc, std::abs(z.imag()));
    EXPECT_GT(max_im_vloc, 1e-6)
        << "test cell has a real V_loc(G): it cannot detect a dropped "
           "imaginary part";

    // V_eff(r): the c2r is UNNORMALIZED, so only a relative bound is
    // meaningful.
    std::vector<double> veff_r(static_cast<size_t>(nr));
    den.fetch_veff_r(veff_r.data());
    double veff_max = 0;
    for (int i = 0; i < nr; ++i)
        veff_max = std::max(veff_max, std::abs(static_cast<double>(
                                          ws.V_eff_r[i])));
    ASSERT_GT(veff_max, 0.0);
    for (int i = 0; i < nr; ++i)
        EXPECT_NEAR(veff_r[static_cast<size_t>(i)],
                    static_cast<double>(ws.V_eff_r[i]), 1e-9 * veff_max)
            << "V_eff(r) mismatch at ir=" << i;

    // E_xc from the same fused reduction the SCF loop reads.
    const auto s = den.reduce(static_cast<double>(prob.vol_bohr));
    EXPECT_NEAR(s.e_xc, static_cast<double>(ws.E_xc),
                1e-12 * std::max(1.0, std::abs(static_cast<double>(ws.E_xc))));
}

// ── Pulay mixing parity, including the ring wrap ─────────────────────────
TEST(CudaPWDensity, MixingMatchesCpuPulay) {
    cpupw::PWProblem prob;
    build_problem(prob);
    auto& ws = prob.ws;
    const int ng = prob.gv0.count();
    const int nr = ws.nr;

    DeviceDensity den(prob, kMixBeta, kDepth);

    // The CPU builder keeps no ctor snapshot: finish() honours the LIVE
    // workspace mixing parameters, so pin them to match DeviceDensity.
    ws.mix_beta = AccReal(kMixBeta);
    ws.pulay_depth = kDepth;
    cpupw::DensityBuilder db;
    db.prepare(ws);
    db.reset(ws);

    std::vector<std::complex<double>> dev(static_cast<size_t>(ng));
    // depth = 3 with 5 pushes exercises the linear branch (t = 0), the
    // partially-filled Pulay branch (t = 1, 2) and the wrapped ring
    // (t = 3, 4), which is where the slot ↔ history-index mapping can go
    // wrong without ever tripping a shorter test.
    for (int t = 0; t < 5; ++t) {
        const std::vector<double> synth = synth_accumulation(nr, t);

        db.begin(ws);
        std::memcpy(db.accumulation(), synth.data(),
                    static_cast<size_t>(nr) * sizeof(AccReal));
        db.finish(ws); // r2c → ρ(G), save ρ_old, pulay_mix

        ASSERT_EQ(cudaMemcpy(den.density_accumulator(), synth.data(),
                             static_cast<size_t>(nr) * sizeof(double),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        den.collect();
        den.mix(t);
        den.fetch_g(GField::rho, dev.data());

        double ref_max = 0;
        for (int i = 0; i < ng; ++i)
            ref_max = std::max(ref_max, std::abs(ws.rho_G[i]));
        ASSERT_GT(ref_max, 0.0);
        // Bound rationale: the Gram matrix is one cublasZgemm here and a
        // serial host loop there (~1e-14 relative apart), and the augmented
        // DIIS solve amplifies that by its condition number. This is a
        // parity bound, not a bit-exactness claim.
        for (int i = 0; i < ng; ++i)
            EXPECT_NEAR(std::abs(dev[static_cast<size_t>(i)] - ws.rho_G[i]),
                        0.0, 1e-9 * ref_max)
                << "mixed rho(G) mismatch at t=" << t << " ig=" << i;
    }

    EXPECT_GT(max_abs(dev), 0.0);
}

// ── Fused reductions vs iteration_energy / ConvergenceCheck ──────────────
TEST(CudaPWDensity, ReductionsMatchIterationEnergy) {
    cpupw::PWProblem prob;
    build_problem(prob);
    auto& ws = prob.ws;
    const int ng = prob.gv0.count();
    const int nr = ws.nr;

    DeviceDensity den(prob, kMixBeta, kDepth);

    // One pseudo-iteration, host and device in lockstep.
    den.hartree();
    den.exchange_correlation();
    den.assemble_veff();

    cpupw::PoissonSolver po;
    po.step(ws);
    cpupw::XCBuilder xcb;
    xcb.prepare(ws);
    xcb.step(ws, ATV::cpu::scf::XCType::LSDA);
    cpupw::VeffBuilder vb;
    vb.prepare(ws);
    vb.step(ws);

    const std::vector<double> synth = synth_accumulation(nr, 0);
    // The CPU builder keeps no ctor snapshot: finish() honours the LIVE
    // workspace mixing parameters, so pin them to match DeviceDensity.
    ws.mix_beta = AccReal(kMixBeta);
    ws.pulay_depth = kDepth;
    cpupw::DensityBuilder db;
    db.prepare(ws);
    db.reset(ws);
    db.begin(ws);
    std::memcpy(db.accumulation(), synth.data(),
                static_cast<size_t>(nr) * sizeof(AccReal));
    db.finish(ws);

    ASSERT_EQ(cudaMemcpy(den.density_accumulator(), synth.data(),
                         static_cast<size_t>(nr) * sizeof(double),
                         cudaMemcpyHostToDevice),
              cudaSuccess);
    den.collect();
    den.mix(0);

    const double vol = static_cast<double>(prob.vol_bohr);
    const auto s = den.reduce(vol);

    const AccReal e_band_synth = AccReal(-1.234); // arbitrary; host-only term
    const auto ref = cpupw::iteration_energy(ws, e_band_synth, prob.ew_total,
                                             prob.vol_bohr);

    auto rel = [](double ref_v) {
        return 1e-10 * std::max(1.0, std::abs(ref_v));
    };
    // Tree (device) vs serial (host) summation of the same Parseval sums.
    EXPECT_NEAR(s.e_h, static_cast<double>(ref.e_h), rel(ref.e_h));
    EXPECT_NEAR(s.e_xc_int, static_cast<double>(ref.e_xc_int),
                rel(ref.e_xc_int));
    EXPECT_NEAR(s.e_loc, static_cast<double>(ref.e_loc), rel(ref.e_loc));

    // Residual: ConvergenceCheck sums |ρ − ρ_old|² via std::abs (hypot) and
    // the device sums (Δre² + Δim²); the two differ at rounding level.
    double resid2 = 0;
    for (int i = 0; i < ng; ++i) {
        const double d = std::abs(ws.rho_G[i] - ws.rho_G_old[i]);
        resid2 += d * d;
    }
    const double resid_ref = std::sqrt(resid2);
    ASSERT_GT(resid_ref, 0.0);
    EXPECT_NEAR(s.resid, resid_ref, 1e-10 * resid_ref);
}

// ── Reductions are reproducible run to run (fixed launch geometry) ───────
TEST(CudaPWDensity, ReductionsAreBitwiseReproducible) {
    cpupw::PWProblem prob;
    build_problem(prob);

    DeviceDensity den(prob, kMixBeta, kDepth);
    const double vol = static_cast<double>(prob.vol_bohr);

    // Re-run the WHOLE XC pass each time so the E_xc reduction is exercised
    // too — that is the one that used to be an atomicAdd.
    den.hartree();
    den.exchange_correlation();
    const auto a = den.reduce(vol);
    den.hartree();
    den.exchange_correlation();
    const auto b = den.reduce(vol);
    // Same inputs, same fixed grid ⇒ identical summation tree ⇒ identical
    // bits. This is what replacing the atomicAdd in xc_energy bought.
    EXPECT_EQ(a.e_xc, b.e_xc);
    EXPECT_EQ(a.e_h, b.e_h);
    EXPECT_EQ(a.e_xc_int, b.e_xc_int);
    EXPECT_EQ(a.e_loc, b.e_loc);
    EXPECT_EQ(a.resid, b.resid);
}
