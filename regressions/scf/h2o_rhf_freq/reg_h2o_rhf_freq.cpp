/// @file regressions/scf/h2o_rhf_freq/reg_h2o_rhf_freq.cpp
/// @brief H₂O HF/6-31G* harmonic frequencies from the analytic CPHF
///        Hessian (`RHF::hessian`) via `cpu::vib::Hessian::run_from_cartesian`,
///        gated against in-house FD (`Hessian::run`) and pinned PySCF 2.14.0.
///        See REFERENCE.md.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Frame is the only I/O token. Exit non-zero on any miss. ≤ 60 s.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/vibration/hessian.h"

int main() {
    using namespace ATV;

    // ── The repository's standard water geometry: R(OH) = 0.95 Å,
    //    ∠HOH = 104.5° (identical to regressions/scf/h2o_rhf_sto3g).
    constexpr double kPi = 3.14159265358979323846;
    const double r_oh = 0.95;                      // Å
    const double half = 0.5 * 104.5 * kPi / 180.0; // half bond angle [rad]
    const double hx = r_oh * std::sin(half);
    const double hz = r_oh * std::cos(half);

    const auto make_frame = [&]() {
        cpu::Frame frame(3);
        frame.span(kAtom_type)[0] = 8; // O
        frame.span(kAtom_x)[0] = 0.0;
        frame.span(kAtom_y)[0] = 0.0;
        frame.span(kAtom_z)[0] = 0.0;
        frame.span(kAtom_type)[1] = 1; // H
        frame.span(kAtom_x)[1] = static_cast<StorageReal>(hx);
        frame.span(kAtom_y)[1] = 0.0;
        frame.span(kAtom_z)[1] = static_cast<StorageReal>(hz);
        frame.span(kAtom_type)[2] = 1; // H
        frame.span(kAtom_x)[2] = static_cast<StorageReal>(-hx);
        frame.span(kAtom_y)[2] = 0.0;
        frame.span(kAtom_z)[2] = static_cast<StorageReal>(hz);
        frame.box = cpu::Box::cubic(20.0);
        return frame;
    };

    // ── Analytic: RHF::hessian (CPHF) → Hessian::run_from_cartesian.
    cpu::Frame frame = make_frame();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Pople631Gs;
    rhf.tol = 1e-12;
    rhf.conv_tol_grad = 1e-10;
    const cpu::vib::HessianResult analytic_H = rhf.hessian(frame);

    cpu::vib::Hessian vib;
    const cpu::vib::HessianResult analytic =
        vib.run_from_cartesian(frame, analytic_H.hessian);
    if (analytic.frequencies.size() != 3) {
        std::fprintf(stderr, "FAIL: expected 3 vibrational modes, got %zu\n",
                     static_cast<std::size_t>(analytic.frequencies.size()));
        return 1;
    }

    // ── FD production line: full vib::Hessian on the same RHF.
    cpu::Frame frame_fd = make_frame();
    cpu::scf::RHF rhf_fd;
    rhf_fd.basis = cpu::scf::BasisSet::Pople631Gs;
    rhf_fd.tol = 1e-12;
    rhf_fd.conv_tol_grad = 1e-10;
    cpu::vib::Hessian fd_driver;
    fd_driver.on(ATV::vib::Phase::force).add(std::move(rhf_fd));
    fd_driver.build();
    const cpu::vib::HessianResult fd = fd_driver.run(frame_fd);
    if (fd.frequencies.size() != 3) {
        std::fprintf(stderr, "FAIL: FD pipeline expected 3 modes, got %zu\n",
                     static_cast<std::size_t>(fd.frequencies.size()));
        return 1;
    }

    const double want_pyscf[3] = {1843.7084, 4036.1737, 4146.3782}; // cm⁻¹
    const double tol = 1.0;                                         // cm⁻¹

    int status = 0;
    double max_g1 = 0.0;
    double max_g2 = 0.0;
    for (int k = 0; k < 3; ++k) {
        const double nu_an = analytic.frequencies(k);
        const double nu_fd = fd.frequencies(k);
        const double d1 = std::abs(nu_an - nu_fd);
        const double d2 = std::abs(nu_an - want_pyscf[k]);
        if (d1 > max_g1)
            max_g1 = d1;
        if (d2 > max_g2)
            max_g2 = d2;
        if (!(d1 <= tol)) {
            std::fprintf(stderr,
                         "FAIL gate1 (analytic vs in-house FD): mode %d "
                         "analytic=%.4f fd=%.4f tol=%.1f cm^-1\n",
                         k, nu_an, nu_fd, tol);
            status = 1;
        }
        if (!(d2 <= tol)) {
            std::fprintf(stderr,
                         "FAIL gate2 (vs pinned PySCF hessian.RHF): mode %d "
                         "got=%.4f want=%.4f tol=%.1f cm^-1\n",
                         k, nu_an, want_pyscf[k], tol);
            status = 1;
        }
    }
    if (status != 0)
        return status;

    std::printf("PASS: H2O HF/6-31G* harmonic frequencies "
                "[%.4f, %.4f, %.4f] cm^-1\n"
                "      gate1 analytic-vs-FD max dev %.4f cm^-1; "
                "gate2 vs PySCF 2.14.0 [%.4f, %.4f, %.4f] max dev %.4f "
                "cm^-1 (tol 1.0)\n",
                analytic.frequencies(0), analytic.frequencies(1),
                analytic.frequencies(2), max_g1, want_pyscf[0], want_pyscf[1],
                want_pyscf[2], max_g2);
    return 0;
}
