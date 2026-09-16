/// @file regressions/pw/pw-pbe-gga-si/reg_pw-pbe-gga-si.cpp
/// @brief END-TO-END: plane-wave true PBE GGA on Si diamond (symmetry forces
///        + GGA vs LSDA energy gap). Public API only. See REFERENCE.md.
///
/// Requires ATV_BUILD_PW. No live CP2K/QE.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "atomiverse/core/mat3.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/planewave/pw_solver.h"
#include "atomiverse/cpu/scf/xc_functional.h"

using namespace ATV;
using namespace ATV::cpu::pw;

namespace {

cpu::Frame make_si_diamond(double a_ang) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = 14;
    f.span(kAtom_type)[1] = 14;
    f.span(kAtom_mass)[0] = 28.0855;
    f.span(kAtom_mass)[1] = 28.0855;
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = a_ang / 4.0;
    f.span(kAtom_y)[1] = a_ang / 4.0;
    f.span(kAtom_z)[1] = a_ang / 4.0;
    const double h = a_ang / 2.0;
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
    return f;
}

} // namespace

int main() {
    const double force_tol = 0.5;    // eV/Å — symmetry residual
    const double de_floor_ry = 1e-4; // |E_PBE − E_LSDA| floor [Ry]

    // Compiled-in absolute path (ATV_REG_DATA_DIR): resolves identically from
    // the build tree, the source tree, or a bare ./reg_<case>.
    const char* upf = ATV_REG_DATA_DIR "/pw/Si_lda_nc.upf";
    if (std::FILE* f = std::fopen(upf, "r")) {
        std::fclose(f);
    } else {
        std::printf("FAIL: cannot open %s\n", upf);
        return 1;
    }

    auto frame = make_si_diamond(5.43);

    PW pbe;
    pbe.upf_path = upf;
    pbe.ecut_ry = 15.0;
    pbe.k_mesh = 2;
    pbe.tol = 1e-5;
    pbe.max_iter = 40;
    pbe.mix_beta = 0.5;
    pbe.xc = cpu::scf::XCType::PBE;

    pbe.build(frame);
    const auto& r_pbe = pbe.run(frame);
    if (!std::isfinite(r_pbe.energy)) {
        std::printf("FAIL: non-finite PBE energy\n");
        return 1;
    }

    const auto F = pw_forces_analytic(frame, pbe);
    if (F.size() != 2u) {
        std::printf("FAIL: expected 2 forces, got %zu\n", F.size());
        return 1;
    }
    for (std::size_t i = 0; i < F.size(); ++i) {
        const double mag = std::sqrt(F[i][0] * F[i][0] + F[i][1] * F[i][1]
                                     + F[i][2] * F[i][2]);
        if (!(mag < force_tol)) {
            std::printf("FAIL: |F[%zu]|=%.6g eV/Å  tol=%g (diamond symmetry)\n",
                        i, mag, force_tol);
            return 1;
        }
    }

    PW lda;
    lda.upf_path = upf;
    lda.ecut_ry = 15.0;
    lda.k_mesh = 2;
    lda.tol = 1e-5;
    lda.max_iter = 40;
    lda.mix_beta = 0.5;
    lda.xc = cpu::scf::XCType::LSDA;
    lda.build(frame);
    const auto& r_lda = lda.run(frame);
    if (!std::isfinite(r_lda.energy)) {
        std::printf("FAIL: non-finite LSDA energy\n");
        return 1;
    }

    const double de = std::abs(r_pbe.energy - r_lda.energy);
    if (!(de > de_floor_ry)) {
        std::printf("FAIL: |E_PBE−E_LSDA|=%.6g Ry  floor=%g (GGA inactive?)\n",
                    de, de_floor_ry);
        return 1;
    }

    std::printf("ok: E_PBE=%.8f Ry  E_LSDA=%.8f Ry  |Δ|=%.6g Ry  "
                "iters_pbe=%d converged=%d\n",
                r_pbe.energy, r_lda.energy, de, r_pbe.iterations,
                int(r_pbe.converged));
    std::printf(
        "ok: |F0|=%.4g  |F1|=%.4g eV/Å  upf=%s\n",
        std::sqrt(F[0][0] * F[0][0] + F[0][1] * F[0][1] + F[0][2] * F[0][2]),
        std::sqrt(F[1][0] * F[1][0] + F[1][1] * F[1][1] + F[1][2] * F[1][2]),
        upf);
    std::printf("PASS: pw-pbe-gga-si (true GGA PBE + symmetry forces)\n");
    return 0;
}
