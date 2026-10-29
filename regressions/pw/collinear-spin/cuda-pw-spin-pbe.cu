/// @file regressions/pw/cuda-pw-spin-pbe/reg_cuda-pw-spin-pbe.cu
/// @brief END-TO-END: collinear spin-GGA PBE (O2 triplet) on cuda::pw::PW —
///        dual-gate CPU↔CUDA + magnetization + GGA activation.
///
/// Twin of `pw-spin-gga` on the device surface. Public API only.
/// Requires ATV_BUILD_PW + ATV_BUILD_CUDA. See REFERENCE.md.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/xc/xc_functional.h"
#include "atomiverse/cuda/planewave/pw_solver.cuh"

#ifndef ATV_REG_DATA_DIR
#error "ATV_REG_DATA_DIR must be set to tests/data"
#endif
#include "harness/registry.h"

namespace atv_reg_pw_cuda_pw_spin_pbe {

using namespace ATV;

namespace {

constexpr double kWantMag = 2.0;
constexpr double kTolMag = 1e-9;
// Same-settings CPU↔CUDA energy parity [Ha]. Non-bit-exact (cuFFT vs FFTW,
// tree vs serial reductions, libdevice vs libm, Davidson path). Matches the
// collinear LSDA dual-gate envelope.
constexpr double kParityTolHa = 2.0e-3;
// GGA activation floor vs spin-LSDA twin [Ha].
constexpr double kDeFloorHa = 1.0e-4;
constexpr double kEcutRy = 12.0;
constexpr double kBoxAng = 8.0;
constexpr double kBondAng = 1.2075;

enum class Device { Cpu, Cuda };

const char* device_name(Device d) { return d == Device::Cuda ? "cuda" : "cpu"; }

std::string parse_device_arg(int argc, char** argv) {
    std::string want = "both";
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--device=", 9) == 0) {
            want = a + 9;
        } else if (std::strcmp(a, "--device") == 0 && i + 1 < argc) {
            want = argv[++i];
        } else if (std::strcmp(a, "cpu") == 0 || std::strcmp(a, "cuda") == 0
                   || std::strcmp(a, "both") == 0) {
            want = a;
        }
    }
    return want;
}

cpu::Frame make_o2() {
    cpu::Frame frame(2);
    for (int i = 0; i < 2; ++i) {
        frame.span(kAtom_type)[i] = 8;
        frame.span(kAtom_mass)[i] = 15.999;
        frame.span(kAtom_x)[i] = 0.0;
        frame.span(kAtom_y)[i] = 0.0;
    }
    frame.span(kAtom_z)[0] = 0.0;
    frame.span(kAtom_z)[1] = kBondAng;
    frame.box = cpu::Box::cubic(kBoxAng);
    return frame;
}

struct RunOut {
    double energy_ha = 0;
    double magnetization = 0;
    bool converged = false;
    int iterations = 0;
};

RunOut run_one(Device dev, const cpu::Frame& frame, const char* upf,
               cpu::xc::XCType xc) {
    RunOut o;
    if (dev == Device::Cpu) {
        cpu::pw::PW pw{std::make_shared<cpu::pw::UpfPseudopotential>(upf),
                       kEcutRy, 1, xc};
        pw.tol = 5e-4;
        pw.max_iter = 60;
        pw.mix_beta = 0.3;
        pw.pulay_depth = 5;
        pw.spin_polarized = true;
        pw.multiplicity = 3;
        pw.build(frame);
        const auto& r = pw.run(frame);
        o.energy_ha = r.energy_ha;
        o.magnetization = r.magnetization;
        o.converged = r.converged;
        o.iterations = r.iterations;
    } else {
        cuda::pw::PW pw{std::make_shared<cpu::pw::UpfPseudopotential>(upf),
                        kEcutRy, 1, xc};
        pw.tol = 5e-4;
        pw.max_iter = 60;
        pw.mix_beta = 0.3;
        pw.pulay_depth = 5;
        pw.spin_polarized = true;
        pw.multiplicity = 3;
        pw.build(frame);
        const auto& r = pw.run(frame);
        o.energy_ha = r.energy_ha;
        o.magnetization = r.magnetization;
        o.converged = r.converged;
        o.iterations = r.iterations;
    }
    return o;
}

bool check_one(const RunOut& r, Device dev, const char* tag) {
    bool ok = true;
    std::printf("[%s %s] E=%.10f Ha  mag=%.6f  conv=%d iters=%d\n",
                device_name(dev), tag, r.energy_ha, r.magnetization,
                (int)r.converged, r.iterations);
    if (!std::isfinite(r.energy_ha)) {
        std::printf("FAIL [%s %s]: non-finite energy\n", device_name(dev), tag);
        ok = false;
    }
    if (!r.converged) {
        std::printf("FAIL [%s %s]: SCF not converged\n", device_name(dev), tag);
        ok = false;
    }
    if (std::fabs(r.magnetization - kWantMag) > kTolMag) {
        std::printf("FAIL [%s %s]: mag got=%.12f want=%.12f\n",
                    device_name(dev), tag, r.magnetization, kWantMag);
        ok = false;
    }
    return ok;
}

} // namespace

int run(int argc, char** argv) {
    const char* upf = ATV_REG_DATA_DIR "/pw/O_ONCV_PBE-1.2.upf";
    if (std::FILE* f = std::fopen(upf, "r")) {
        std::fclose(f);
    } else {
        std::printf("FAIL: cannot open %s\n", upf);
        return 1;
    }

    const std::string want = parse_device_arg(argc, argv);
    auto frame = make_o2();

    RunOut cpu_pbe{}, cuda_pbe{}, cpu_lsda{};
    bool ok = true;

    // Always run CPU PBE (oracle) when cpu or both; CUDA PBE when cuda/both.
    if (want == "cpu" || want == "both") {
        cpu_pbe = run_one(Device::Cpu, frame, upf, cpu::xc::XCType::PBE);
        ok = check_one(cpu_pbe, Device::Cpu, "PBE") && ok;
        cpu_lsda = run_one(Device::Cpu, frame, upf, cpu::xc::XCType::LSDA);
        ok = check_one(cpu_lsda, Device::Cpu, "LSDA") && ok;
        if (ok) {
            const double de = std::fabs(cpu_pbe.energy_ha - cpu_lsda.energy_ha);
            std::printf("|E_PBE - E_LSDA|_cpu = %.6e Ha (floor %.1e)\n", de,
                        kDeFloorHa);
            if (de <= kDeFloorHa) {
                std::printf(
                    "FAIL: GGA activation |ΔE| too small on CPU twin\n");
                ok = false;
            }
        }
    }
    if (want == "cuda" || want == "both") {
        cuda_pbe = run_one(Device::Cuda, frame, upf, cpu::xc::XCType::PBE);
        ok = check_one(cuda_pbe, Device::Cuda, "PBE") && ok;
    }
    if (want == "both") {
        const double de = std::fabs(cuda_pbe.energy_ha - cpu_pbe.energy_ha);
        std::printf("parity |E_cuda - E_cpu| = %.6e Ha (tol %.1e)\n", de,
                    kParityTolHa);
        if (de > kParityTolHa) {
            std::printf("FAIL: CPU↔CUDA spin-PBE energy parity\n");
            ok = false;
        }
    }

    if (!ok)
        return 1;

    // Force + stress dual-gate on a stretched bond (non-vacuous |F|, |σ|).
    // Same XC / UPF / ecut as the energy gate. Spin∧smear stays rejected.
    auto frame_f = make_o2();
    frame_f.span(kAtom_z)[1] = 1.30;
    constexpr double kForceTolEvA = 5.0e-3;
    constexpr double kStressTolEvA3 = 5.0e-2;

    const auto pbe_pp = std::make_shared<cpu::pw::UpfPseudopotential>(upf);
    auto fill_pbe = [](auto& pw) {
        pw.tol = 5e-4;
        pw.max_iter = 60;
        pw.mix_beta = 0.3;
        pw.pulay_depth = 5;
        pw.spin_polarized = true;
        pw.multiplicity = 3;
    };

    // CPU oracle: `PW::forces` writes the Frame's force columns + kEnergy, so
    // it runs on a copy (frame_f is reused by the CUDA leg) and the triples
    // [eV/Å] are read back from that copy.
    auto cpu_forces = [](const cpu::Frame& f, cpu::pw::PW& pw) {
        cpu::Frame frame_copy = f;
        pw.forces(frame_copy);
        const auto fx = frame_copy.span(kAtom_fx);
        const auto fy = frame_copy.span(kAtom_fy);
        const auto fz = frame_copy.span(kAtom_fz);
        std::vector<std::array<double, 3>> F(
            static_cast<std::size_t>(frame_copy.nlocal()));
        for (std::size_t i = 0; i < F.size(); ++i)
            F[i] = {static_cast<double>(fx[i]), static_cast<double>(fy[i]),
                    static_cast<double>(fz[i])};
        return F;
    };

    std::vector<std::array<double, 3>> F_cpu, F_cuda;
    std::array<double, 6> S_cpu{}, S_cuda{};
    if (want == "cpu" || want == "both") {
        cpu::pw::PW pw{pbe_pp, kEcutRy, 1, cpu::xc::XCType::PBE};
        fill_pbe(pw);
        pw.build(frame_f);
        F_cpu = cpu_forces(frame_f, pw);
        {
            cpu::Frame frame_copy = frame_f; // stress() writes kStress/kEnergy
            S_cpu = pw.stress(frame_copy);   // σ [eV/Å³]
        }
    }
    if (want == "cuda" || want == "both") {
        cuda::pw::PW pw{pbe_pp, kEcutRy, 1, cpu::xc::XCType::PBE};
        fill_pbe(pw);
        pw.build(frame_f);
        F_cuda = ([&]() {
            ATV::cpu::Frame result_frame(frame_f);
            pw.forces(result_frame);
            std::vector<std::array<double, 3>> result(result_frame.nlocal());
            for (int i = 0; i < result_frame.nlocal(); ++i)
                result[i] = {result_frame.span(ATV::kAtom_fx)[i],
                             result_frame.span(ATV::kAtom_fy)[i],
                             result_frame.span(ATV::kAtom_fz)[i]};
            return result;
        }());
        S_cuda = ([&]() {
            ATV::cpu::Frame result_frame(frame_f);
            const auto result = pw.stress(result_frame);
            return result;
        }());
    }
    if (want == "both") {
        double max_df = 0.0;
        double max_f = 0.0;
        for (int ia = 0; ia < 2; ++ia) {
            for (int k = 0; k < 3; ++k) {
                const double fc = F_cpu[static_cast<std::size_t>(ia)]
                                       [static_cast<std::size_t>(k)];
                const double fg = F_cuda[static_cast<std::size_t>(ia)]
                                        [static_cast<std::size_t>(k)];
                if (!std::isfinite(fc) || !std::isfinite(fg)) {
                    std::printf("FAIL: non-finite spin-PBE force\n");
                    return 1;
                }
                max_df = std::max(max_df, std::abs(fc - fg));
                max_f = std::max(max_f, std::abs(fc));
            }
        }
        double max_ds = 0.0;
        double max_s = 0.0;
        for (int k = 0; k < 6; ++k) {
            if (!std::isfinite(S_cpu[static_cast<std::size_t>(k)])
                || !std::isfinite(S_cuda[static_cast<std::size_t>(k)])) {
                std::printf("FAIL: non-finite spin-PBE stress\n");
                return 1;
            }
            max_ds = std::max(max_ds,
                              std::abs(S_cpu[static_cast<std::size_t>(k)]
                                       - S_cuda[static_cast<std::size_t>(k)]));
            max_s =
                std::max(max_s, std::abs(S_cpu[static_cast<std::size_t>(k)]));
        }
        std::printf("spin-PBE F  max|Δ|=%.6e eV/A  (tol %.1e)  max|F|=%.6f\n",
                    max_df, kForceTolEvA, max_f);
        std::printf("spin-PBE σ  max|Δ|=%.6e eV/A3 (tol %.1e)  max|σ|=%.6f\n",
                    max_ds, kStressTolEvA3, max_s);
        if (!(max_f > 0.05)) {
            std::printf("FAIL: spin-PBE force vacuous\n");
            return 1;
        }
        if (max_df > kForceTolEvA) {
            std::printf("FAIL: spin-PBE force parity\n");
            return 1;
        }
        if (max_ds > kStressTolEvA3) {
            std::printf("FAIL: spin-PBE stress parity\n");
            return 1;
        }
    }

    // Collinear spin × FD smear (energy + forces). Same O2 / UPF / ecut.
    {
        cpu::pw::PW cpu_s{pbe_pp, kEcutRy, 1, cpu::xc::XCType::PBE};
        cuda::pw::PW cuda_s{pbe_pp, kEcutRy, 1, cpu::xc::XCType::PBE};
        fill_pbe(cpu_s);
        fill_pbe(cuda_s);
        cpu_s.smearing = ATV::electronic::Smearing::fermi_dirac;
        cuda_s.smearing = cuda::ATV::electronic::Smearing::fermi_dirac;
        cpu_s.degauss = 0.02;
        cuda_s.degauss = 0.02;
        cpu_s.extra_bands = 4;
        cuda_s.extra_bands = 4;
        cpu_s.build(frame_f);
        cuda_s.build(frame_f);
        const auto rc = cpu_s.run(frame_f);
        const auto rg = cuda_s.run(frame_f);
        const double de = std::fabs(rg.energy_ha - rc.energy_ha);
        std::printf("spin×smear E  |Δ|=%.6e Ha  mag_cpu=%.4f mag_cuda=%.4f\n",
                    de, rc.magnetization, rg.magnetization);
        if (!rc.converged || !rg.converged || de > kParityTolHa) {
            std::printf("FAIL: spin×smear energy\n");
            return 1;
        }
        auto Fc = cpu_forces(frame_f, cpu_s);
        auto Fg = ([&]() {
            ATV::cpu::Frame result_frame(frame_f);
            cuda_s.forces(result_frame);
            std::vector<std::array<double, 3>> result(result_frame.nlocal());
            for (int i = 0; i < result_frame.nlocal(); ++i)
                result[i] = {result_frame.span(ATV::kAtom_fx)[i],
                             result_frame.span(ATV::kAtom_fy)[i],
                             result_frame.span(ATV::kAtom_fz)[i]};
            return result;
        }());
        double max_df = 0.0;
        double max_f = 0.0;
        for (int ia = 0; ia < 2; ++ia) {
            for (int k = 0; k < 3; ++k) {
                const double fc = Fc[static_cast<std::size_t>(ia)]
                                    [static_cast<std::size_t>(k)];
                const double fg = Fg[static_cast<std::size_t>(ia)]
                                    [static_cast<std::size_t>(k)];
                if (!std::isfinite(fc) || !std::isfinite(fg)) {
                    std::printf("FAIL: non-finite spin×smear force\n");
                    return 1;
                }
                max_df = std::max(max_df, std::abs(fc - fg));
                max_f = std::max(max_f, std::abs(fc));
            }
        }
        std::printf("spin×smear F  max|Δ|=%.6e eV/A  (tol %.1e)  max|F|=%.6f\n",
                    max_df, kForceTolEvA, max_f);
        if (!(max_f > 0.05)) {
            std::printf("FAIL: spin×smear force vacuous\n");
            return 1;
        }
        if (max_df > kForceTolEvA) {
            std::printf("FAIL: spin×smear force parity\n");
            return 1;
        }
    }

    std::printf("PASS: cuda-pw-spin-pbe (mag + spin-GGA + F/σ + spin×smear)\n");
    return 0;
}

} // namespace atv_reg_pw_cuda_pw_spin_pbe

ATV_REGRESSION("pw/collinear-spin", "cuda-pw-spin-pbe",
               atv_reg_pw_cuda_pw_spin_pbe::run);
