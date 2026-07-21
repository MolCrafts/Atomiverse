/// @file regressions/scf/h2o_rhf_freq/reg_h2o_rhf_freq.cpp
/// @brief H₂O HF/6-31G* harmonic frequencies from the analytic CPHF
///        Hessian (`RHF::hessian`) through the existing vib pipeline —
///        gated first against the in-house finite-difference production
///        line (`cpu::vib::Hessian` + the same RHF), then against the
///        pinned PySCF 2.14.0 `hessian.RHF` values. See REFERENCE.md.
///
/// Out-of-library consumer: public headers + link atomiverse only.
/// Two producers (analytic CPHF vs central FD of analytic forces) meet
/// at the same consumer chain `vib::MassWeighting` →
/// `vib::HessianDiagonalizer` on a `vib::HessianWorkspace`. Gate 1:
/// mode-by-mode ≤ 1 cm⁻¹ analytic vs FD (both in-house). Gate 2:
/// mode-by-mode ≤ 1 cm⁻¹ vs the pinned external PySCF frequencies.
/// Exit non-zero on any miss, printing got / want / tol. ≤ 60 s.

#include <cmath>
#include <cstdio>

#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/vib/hessian.h"
#include "atomiverse/cpu/vib/hessian_diagonalizer.h"
#include "atomiverse/cpu/vib/hessian_workspace.h"
#include "atomiverse/cpu/vib/mass_weighting.h"

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
        frame.simbox = cpu::Box::cubic(20.0);
        return frame;
    };

    // ── Analytic producer: RHF::hessian (CPHF) → vib mass-weighting +
    //    diagonalization on a caller-owned workspace.
    cpu::Frame frame = make_frame();
    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Pople631Gs;
    rhf.tol = 1e-12;
    rhf.conv_tol_grad = 1e-10;
    const cpu::vib::HessianResult analytic = rhf.hessian(frame);

    cpu::vib::HessianWorkspace ws;
    ws.allocate(analytic.n_atoms);
    ws.hessian = analytic.hessian; // [3N,3N] eV/Å²
    // Equilibrium geometry cache (Å, atom-major) — the diagonalizer builds
    // its translation/rotation projector from ws.r0.
    for (int i = 0; i < analytic.n_atoms; ++i) {
        ws.r0[3 * i + 0] = static_cast<double>(frame.span(kAtom_x)[i]);
        ws.r0[3 * i + 1] = static_cast<double>(frame.span(kAtom_y)[i]);
        ws.r0[3 * i + 2] = static_cast<double>(frame.span(kAtom_z)[i]);
    }
    ws.r0_ready = true;
    cpu::vib::MassWeighting mw{&ws};
    cpu::vib::HessianDiagonalizer diag{&ws};
    cpu::Context rt;
    // The two components activate on the finalize pass (step ≥ 6N).
    const cpu::LaunchContext fin{rt, frame, /*stage=*/0,
                                 /*step=*/6 * analytic.n_atoms};
    mw.launch(fin);
    diag.launch(fin);
    if (ws.frequencies.size() != 3) {
        std::fprintf(stderr,
                     "FAIL: expected 3 vibrational modes, got %zu\n",
                     static_cast<std::size_t>(ws.frequencies.size()));
        return 1;
    }

    // ── FD production line: the vib::Hessian driver displacing the SAME
    //    concrete RHF (central differences of the analytic forces).
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
        std::fprintf(stderr,
                     "FAIL: FD pipeline expected 3 modes, got %zu\n",
                     static_cast<std::size_t>(fd.frequencies.size()));
        return 1;
    }

    // ── Gate 1: analytic vs in-house FD, mode by mode ≤ 1 cm⁻¹.
    // ── Gate 2: analytic vs pinned PySCF 2.14.0 hessian.RHF (cart=True,
    //    conv_tol 1e-12, this exact geometry) ≤ 1 cm⁻¹.
    const double want_pyscf[3] = {1843.7084, 4036.1737, 4146.3782}; // cm⁻¹
    const double tol = 1.0;                                         // cm⁻¹

    int status = 0;
    for (int k = 0; k < 3; ++k) {
        const double nu_an = ws.frequencies(k);
        const double nu_fd = fd.frequencies(k);
        if (!(std::abs(nu_an - nu_fd) <= tol)) {
            std::fprintf(stderr,
                         "FAIL gate1 (analytic vs in-house FD): mode %d "
                         "analytic=%.4f fd=%.4f tol=%.1f cm^-1\n",
                         k, nu_an, nu_fd, tol);
            status = 1;
        }
        if (!(std::abs(nu_an - want_pyscf[k]) <= tol)) {
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
                "cm^-1 (tol 1.0)\n"
                "      [Pople-Krishnan-Schlegel-Binkley 1979, "
                "DOI 10.1002/qua.560160825]\n",
                ws.frequencies(0), ws.frequencies(1), ws.frequencies(2),
                std::max({std::abs(ws.frequencies(0) - fd.frequencies(0)),
                          std::abs(ws.frequencies(1) - fd.frequencies(1)),
                          std::abs(ws.frequencies(2) - fd.frequencies(2))}),
                want_pyscf[0], want_pyscf[1], want_pyscf[2],
                std::max({std::abs(ws.frequencies(0) - want_pyscf[0]),
                          std::abs(ws.frequencies(1) - want_pyscf[1]),
                          std::abs(ws.frequencies(2) - want_pyscf[2])}));
    return 0;
}
