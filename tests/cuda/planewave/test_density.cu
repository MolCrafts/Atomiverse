/// @file tests/cuda/planewave/test_density.cu
/// @brief Unit tests for `cuda::pw::internal::DeviceDensity` — each device
///        stage against the CPU stage helper it mirrors.
///
/// UNIT SCOPE: no SCF loop, no Davidson, no convergence loop. Every test
/// drives ONE device method (or a fixed 5-call mixing sequence with synthetic
/// input) and compares it to the actual CPU stage object on the same
/// `PWWorkspace`. The end-to-end CPU/CUDA PW parity solve lives in
/// regressions/pw/rks, never under ctest.
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
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/xc/xc_functional.h"
#include "atomiverse/cuda/planewave/internal/device_density.cuh"
#include "atomiverse/cuda/planewave/internal/device_solver.cuh"

#include <cmath>
#include <complex>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

using ATV::AccReal;
using ATV::cpu::xc::XCType;
using ATV::cuda::pw::internal::DeviceBandSolver;
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
    f.span(ATV::kAtom_atomic_number)[0] = f.span(ATV::kAtom_atomic_number)[1] =
        14;
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
    const ATV::cpu::pw::GthPseudopotential potential{molrs::Element::Si,
                                                     "GTH-PBE-q4"};
    cpupw::setup_pw_problem(f, potential, /*ecut_ry=*/6.0, /*k_mesh=*/1, prob);
}

/// Synthetic real-space density accumulator for pseudo-iteration `t`.
/// Deliberately varied per iteration so successive Pulay residuals are well
/// separated and the augmented DIIS system stays well conditioned.
std::vector<double> synth_accumulation(int nr, int t) {
    std::vector<double> v(static_cast<size_t>(nr));
    for (int i = 0; i < nr; ++i)
        v[static_cast<size_t>(i)] =
            0.030 + 0.004 * t
            + 0.020
                  * std::sin(0.011 * static_cast<double>(i)
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

// ── Stage parity: Poisson, XC (LSDA or PBE), V_eff ───────────────────────
// Shared body: device stages vs CPU stage helpers for one functional. The
// tolerances are the LDA-parity bounds; the PBE run reuses them unchanged —
// its extra FFT depth (3 gradient c2r + 3 flux r2c + 1 divergence c2r on
// BOTH backends) stays well inside the same envelope.
static void expect_stage_parity(XCType xc, bool high_frequency = false) {
    cpupw::PWProblem prob;
    build_problem(prob);
    auto& ws = prob.ws;
    const int ng = prob.gv0.count();
    const int nr = ws.nr;
    ASSERT_GT(ng, 0);
    ASSERT_GT(nr, 0);

    if (high_frequency) {
        // A positive density with a near-cutoff Fourier mode. The smooth
        // atomic seed alone cannot reveal aliasing of non-polynomial XC.
        int mode = 1;
        for (int i = 1; i < ng; ++i)
            if (prob.gv0.g[i].norm_squared() > prob.gv0.g[mode].norm_squared())
                mode = i;
        int mirror = -1;
        for (int i = 0; i < ng; ++i)
            if (prob.gv0.h[i] == -prob.gv0.h[mode]
                && prob.gv0.k[i] == -prob.gv0.k[mode]
                && prob.gv0.l[i] == -prob.gv0.l[mode])
                mirror = i;
        ASSERT_GE(mirror, 0);
        ws.rho_G.fill(std::complex<AccReal>(0, 0));
        ws.rho_G[0] = 0.05;
        ws.rho_G[mode] = ws.rho_G[mirror] = 0.024;
    }

    // Device stages on the atomic seed density uploaded by the constructor.
    DeviceDensity den(prob, kMixBeta, kDepth, xc);
    den.hartree();
    den.exchange_correlation();
    den.assemble_veff();

    // Host stages on the same workspace (they only READ ws.rho_G).
    cpupw::PoissonSolver po;
    po.step(ws);
    cpupw::XCBuilder xcb;
    xcb.prepare(ws);
    xcb.step(ws, xc);
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
        veff_max =
            std::max(veff_max, std::abs(static_cast<double>(ws.V_eff_r[i])));
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

TEST(CudaPWDensity, HartreeXcVeffMatchCpu) {
    expect_stage_parity(XCType::LSDA);
}

// The PBE leg additionally exercises the spectral ∇ρ, the pointwise PBE
// kernel, and the v_xc −= ∇·flux divergence pass against
// XCBuilder::step(ws, PBE) — the CPU twin runs its true-GGA branch on the
// same atomic seed density.
TEST(CudaPWDensity, HartreeXcVeffMatchCpuPbeFamily) {
    for (const auto xc :
         {XCType::PBE, XCType::PBEsol, XCType::RevPBE, XCType::RPBE}) {
        SCOPED_TRACE(static_cast<int>(xc));
        expect_stage_parity(xc);
    }
}

// ── Pulay mixing parity, including the ring wrap ─────────────────────────
TEST(CudaPWDensity, MixingMatchesCpuPulay) {
    cpupw::PWProblem prob;
    build_problem(prob);
    auto& ws = prob.ws;
    const int ng = prob.gv0.count();
    const int nr = ws.nr;

    DeviceDensity den(prob, kMixBeta, kDepth, XCType::LSDA);

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

    DeviceDensity den(prob, kMixBeta, kDepth, XCType::LSDA);

    // One pseudo-iteration, host and device in lockstep.
    den.hartree();
    den.exchange_correlation();
    den.assemble_veff();

    cpupw::PoissonSolver po;
    po.step(ws);
    cpupw::XCBuilder xcb;
    xcb.prepare(ws);
    xcb.step(ws, ATV::cpu::xc::XCType::LSDA);
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
    const auto ref =
        cpupw::iteration_energy(ws, e_band_synth, prob.ew_total, prob.vol_bohr);

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

    DeviceDensity den(prob, kMixBeta, kDepth, XCType::LSDA);
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

// ── Stream contract (cuda-pw-nondefault-stream) ──────────────────────────
// Density owns a non-default stream; the band solver can borrow it so both
// SCF halves share one serial chain. Physics-free: only the handle identity.
TEST(CudaPWDensity, NonDefaultStreamOwnedAndShareable) {
    cpupw::PWProblem prob;
    build_problem(prob);
    DeviceDensity den(prob, kMixBeta, kDepth, XCType::LSDA);
    ASSERT_NE(den.stream(), static_cast<cudaStream_t>(nullptr));

    DeviceBandSolver solver(prob.ws.fft_nx, prob.ws.fft_ny, prob.ws.fft_nz,
                            prob.n_bands, den.stream());
    EXPECT_EQ(solver.stream(), den.stream());

    // Stand-alone solver (diagnostics) still refuses the default stream.
    DeviceBandSolver own(prob.ws.fft_nx, prob.ws.fft_ny, prob.ws.fft_nz,
                         prob.n_bands);
    ASSERT_NE(own.stream(), static_cast<cudaStream_t>(nullptr));
    EXPECT_NE(own.stream(), den.stream());
}

TEST(CudaPWDensity, BorrowedStreamSurvivesDensityDestruction) {
    cpupw::PWProblem prob;
    build_problem(prob);
    cudaStream_t parent = nullptr;
    ASSERT_EQ(cudaStreamCreate(&parent), cudaSuccess);
    struct Release {
        cudaStream_t stream;
        ~Release() { cudaStreamDestroy(stream); }
    } release{parent};
    // Run one actual density stage on the borrowed stream, then destroy all
    // density buffers and plans. The parent must retain a usable stream.
    {
        DeviceDensity den(prob, kMixBeta, kDepth, XCType::LSDA, false, 0, 0, 0,
                          parent);
        ASSERT_EQ(den.stream(), parent);
        den.hartree();
    }
    cudaEvent_t event = nullptr;
    ASSERT_EQ(cudaEventCreate(&event), cudaSuccess);
    struct EventRelease {
        cudaEvent_t event;
        ~EventRelease() { cudaEventDestroy(event); }
    } event_release{event};
    ASSERT_EQ(cudaEventRecord(event, parent), cudaSuccess);
    EXPECT_EQ(cudaEventSynchronize(event), cudaSuccess);
    // A second independent density stage can acquire that same execution
    // stream after the first owner of scientific scratch has gone away.
    {
        DeviceDensity den(prob, kMixBeta, kDepth, XCType::LSDA, false, 0, 0, 0,
                          parent);
        den.hartree();
    }
    ASSERT_EQ(cudaEventRecord(event, parent), cudaSuccess);
    EXPECT_EQ(cudaEventSynchronize(event), cudaSuccess);
}

// Each spin channel must use the same padded quadrature and adjoint
// projection as the restricted path. One XC stage, no SCF solve.
TEST(CudaPWDensity, PaddedSpinXcMatchesCpu) {
    for (const auto xc : {XCType::LSDA, XCType::PBE, XCType::PBEsol,
                          XCType::RevPBE, XCType::RPBE}) {
        cpupw::PWProblem prob;
        build_problem(prob);
        auto& ws = prob.ws;
        const int ng = prob.gv0.count();
        DeviceDensity den(prob, kMixBeta, kDepth, xc, true, 5, 3, 8);
        den.hartree();
        den.exchange_correlation_spin();
        ws.nspin = 2;
        ws.rho_spin_G.resize(2);
        ws.V_xc_spin_G.resize(2);
        for (int spin = 0; spin < 2; ++spin) {
            ws.rho_spin_G[spin] =
                ws.rho_G * (spin == 0 ? 5.0 / 8.0 : 3.0 / 8.0);
            ws.V_xc_spin_G[spin].resize({static_cast<std::size_t>(ng)});
        }
        cpupw::XCBuilder xc_builder;
        xc_builder.prepare(ws);
        xc_builder.step_spin(ws, xc);
        std::vector<std::complex<double>> got(static_cast<std::size_t>(ng));
        for (int spin = 0; spin < 2; ++spin) {
            den.assemble_veff_spin(spin);
            den.fetch_g(GField::v_xc, got.data());
            for (int i = 0; i < ng; ++i)
                EXPECT_NEAR(std::abs(got[static_cast<std::size_t>(i)]
                                     - ws.V_xc_spin_G[spin][i]),
                            0.0, 1e-10)
                    << "xc=" << static_cast<int>(xc) << " spin=" << spin
                    << " ig=" << i;
        }
        const auto scalars =
            den.reduce_spin(static_cast<double>(prob.vol_bohr));
        EXPECT_NEAR(scalars.e_xc, static_cast<double>(ws.E_xc), 1e-10);
        EXPECT_NEAR(scalars.e_xc_int, static_cast<double>(ws.e_xc_int_spin),
                    1e-10);
    }
}

// The atomic seed stays positive. A near-cutoff cosine that rings below zero
// is what makes the clamp chain rule and the raw-density double counting
// observable: a potential that ignores max(ρ, 0) diverges from XCBuilder.
TEST(CudaPWDensity, NegativeSpinRingingMatchesCpuClamp) {
    for (const auto xc : {XCType::LSDA, XCType::PBE, XCType::PBEsol,
                          XCType::RevPBE, XCType::RPBE}) {
        cpupw::PWProblem prob;
        build_problem(prob);
        auto& ws = prob.ws;
        const int ng = prob.gv0.count();
        int mode = 1;
        for (int i = 1; i < ng; ++i)
            if (prob.gv0.g[i].norm_squared() > prob.gv0.g[mode].norm_squared())
                mode = i;
        int mirror = -1;
        for (int i = 0; i < ng; ++i)
            if (prob.gv0.h[i] == -prob.gv0.h[mode]
                && prob.gv0.k[i] == -prob.gv0.k[mode]
                && prob.gv0.l[i] == -prob.gv0.l[mode])
                mirror = i;
        ASSERT_GE(mirror, 0);
        ASSERT_NE(mirror, mode);
        ws.rho_G.fill(std::complex<AccReal>(0, 0));
        ws.rho_G[0] = 0.01;
        ws.rho_G[mode] = ws.rho_G[mirror] = 0.04;

        DeviceDensity den(prob, kMixBeta, kDepth, xc, true, 5, 3, 8);
        std::vector<double> channel(static_cast<std::size_t>(ws.nr));
        den.fetch_rho_spin_r(0, channel.data());
        double lo = channel[0];
        double hi = channel[0];
        for (double value : channel) {
            lo = std::min(lo, value);
            hi = std::max(hi, value);
        }
        ASSERT_LT(lo, 0.0) << "xc=" << static_cast<int>(xc);
        ASSERT_GT(hi, 0.0) << "xc=" << static_cast<int>(xc);

        den.exchange_correlation_spin();
        ws.nspin = 2;
        ws.rho_spin_G.resize(2);
        ws.V_xc_spin_G.resize(2);
        for (int spin = 0; spin < 2; ++spin) {
            ws.rho_spin_G[spin] =
                ws.rho_G * (spin == 0 ? 5.0 / 8.0 : 3.0 / 8.0);
            ws.V_xc_spin_G[spin].resize({static_cast<std::size_t>(ng)});
        }
        cpupw::XCBuilder xc_builder;
        xc_builder.prepare(ws);
        xc_builder.step_spin(ws, xc);

        std::vector<std::complex<double>> got(static_cast<std::size_t>(ng));
        for (int spin = 0; spin < 2; ++spin) {
            den.assemble_veff_spin(spin);
            den.fetch_g(GField::v_xc, got.data());
            for (int i = 0; i < ng; ++i)
                EXPECT_NEAR(std::abs(got[static_cast<std::size_t>(i)]
                                     - ws.V_xc_spin_G[spin][i]),
                            0.0, 1e-8)
                    << "xc=" << static_cast<int>(xc) << " spin=" << spin
                    << " ig=" << i;
        }
        const auto scalars =
            den.reduce_spin(static_cast<double>(prob.vol_bohr));
        EXPECT_NEAR(scalars.e_xc, static_cast<double>(ws.E_xc), 1e-8);
        EXPECT_NEAR(scalars.e_xc_int, static_cast<double>(ws.e_xc_int_spin),
                    1e-8);
    }
}

TEST(CudaPWDensity, PaddedXcResolvesHighFrequencyDensity) {
    for (const auto xc : {XCType::LSDA, XCType::PBE, XCType::PBEsol,
                          XCType::RevPBE, XCType::RPBE}) {
        SCOPED_TRACE(static_cast<int>(xc));
        expect_stage_parity(xc, true);
    }
}

// A pure magnetization rearrangement has zero total-charge residual. It must
// still prevent convergence, and its norm must not depend on the mixing beta.
TEST(CudaPWDensity, OppositeSpinUpdatesHaveJointUnmixedResidual) {
    for (const double beta : {0.2, 0.7}) {
        cpupw::PWProblem prob;
        build_problem(prob);
        const int nr = prob.ws.nr, ng = prob.gv0.count();
        DeviceDensity den(prob, beta, kDepth, XCType::LSDA, true, 4, 4, 8);
        den.hartree();
        den.exchange_correlation_spin();
        constexpr double delta = 0.001;
        std::vector<double> channel(static_cast<std::size_t>(nr));
        for (int spin = 0; spin < 2; ++spin) {
            den.fetch_rho_spin_r(spin, channel.data());
            for (auto& value : channel)
                value += spin == 0 ? delta : -delta;
            ASSERT_EQ(cudaMemcpy(den.density_accumulator(), channel.data(),
                                 channel.size() * sizeof(double),
                                 cudaMemcpyHostToDevice),
                      cudaSuccess);
            den.collect_spin_out(spin);
        }
        den.mix_spin(0);
        const auto values = den.reduce_spin(static_cast<double>(prob.vol_bohr));
        // A constant real-space update changes only G=0, so the independent
        // analytic joint norm is sqrt(delta²+(-delta)²), before mixing.
        EXPECT_NEAR(values.resid, std::sqrt(2.0) * delta, 1e-12)
            << "mixing beta=" << beta;
        std::vector<std::complex<double>> total(static_cast<std::size_t>(ng));
        den.fetch_g(GField::rho, total.data());
        for (int ig = 0; ig < ng; ++ig)
            EXPECT_NEAR(std::abs(total[static_cast<std::size_t>(ig)]
                                 - prob.ws.rho_G[ig]),
                        0.0, 1e-12);
    }
}

TEST(CudaPWDensity, FrozenSpinDensityRetainsEveryModeAndVolumeScale) {
    cpupw::PWProblem prob;
    build_problem(prob);
    DeviceDensity den(prob, kMixBeta, kDepth, XCType::LSDA, true, 5, 3, 8);
    std::vector<double> alpha(static_cast<std::size_t>(prob.ws.nr)),
        beta(alpha.size());
    den.fetch_rho_spin_r(0, alpha.data());
    den.fetch_rho_spin_r(1, beta.data());
    constexpr double scale = 0.83;
    den.load_rho_spin_r_scaled(alpha.data(), beta.data(), prob.ws.nr, scale);
    std::vector<std::complex<double>> total(
        static_cast<std::size_t>(prob.gv0.count()));
    den.fetch_g(GField::rho, total.data());
    for (int ig = 0; ig < prob.gv0.count(); ++ig)
        EXPECT_NEAR(std::abs(total[static_cast<std::size_t>(ig)]
                             - scale * prob.ws.rho_G[ig]),
                    0.0, 1e-12);
    EXPECT_THROW(den.load_rho_spin_r_scaled(alpha.data(), beta.data(),
                                            prob.ws.nr - 1, scale),
                 std::invalid_argument);
    EXPECT_THROW(den.load_rho_r_scaled(alpha.data(), prob.ws.nr - 1, scale),
                 std::invalid_argument);
}

// Joint history must retain both channels, including wrap and depth-one fallback.
TEST(CudaPWDensity, JointSpinPulayMatchesCpuThroughHistoryWrap) {
    for (const int depth : {1, 3})
        for (const double beta : {0.2, 0.7}) {
            SCOPED_TRACE(depth);
            SCOPED_TRACE(beta);
            cpupw::PWProblem prob;
            build_problem(prob);
            auto& ws = prob.ws;
            const int ng = prob.gv0.count(), nr = ws.nr;
            DeviceDensity den(prob, beta, depth, XCType::LSDA, true, 5, 3, 8);
            ws.nspin = 2;
            ws.mix_beta = beta;
            ws.pulay_depth = depth;
            ws.rho_spin_G.resize(2);
            ws.rho_spin_G_old.resize(2);
            for (int spin = 0; spin < 2; ++spin) {
                ws.rho_spin_G[spin] =
                    ws.rho_G * (spin == 0 ? 5.0 / 8 : 3.0 / 8);
                ws.rho_spin_G_old[spin] = ws.rho_spin_G[spin];
            }
            cpupw::DensityBuilder db;
            db.prepare(ws);
            db.reset(ws);
            xt::xarray<std::complex<AccReal>> out[2];
            std::vector<double> fetched(nr);
            for (int t = 0; t < 7; ++t) {
                for (int spin = 0; spin < 2; ++spin) {
                    auto real = synth_accumulation(nr, 2 * t + spin);
                    std::memcpy(db.accumulation(), real.data(),
                                nr * sizeof(double));
                    db.extract(ws, out[spin]);
                    ASSERT_EQ(cudaMemcpyAsync(den.density_accumulator(),
                                              real.data(), nr * sizeof(double),
                                              cudaMemcpyHostToDevice,
                                              den.stream()),
                              cudaSuccess);
                    den.collect_spin_out(spin);
                    ASSERT_EQ(cudaStreamSynchronize(den.stream()), cudaSuccess);
                }
                const double expected_residual =
                    db.mix_spin(ws, out[0], out[1]);
                den.mix_spin(t);
                EXPECT_NEAR(den.reduce_spin(prob.vol_bohr).resid,
                            expected_residual, 1e-11);
                for (int spin = 0; spin < 2; ++spin) {
                    den.fetch_rho_spin_r(spin, fetched.data());
                    std::memcpy(db.accumulation(), fetched.data(),
                                nr * sizeof(double));
                    xt::xarray<std::complex<AccReal>> got;
                    db.extract(ws, got);
                    double scale = 0;
                    for (const auto value : ws.rho_spin_G[spin])
                        scale = std::max(scale, std::abs(value));
                    ASSERT_GT(scale, 0);
                    for (int g = 0; g < ng; ++g)
                        EXPECT_LE(std::abs(got[g] - ws.rho_spin_G[spin][g]),
                                  1e-9 * scale)
                            << "step=" << t << " spin=" << spin << " G=" << g;
                }
            }
        }
}
