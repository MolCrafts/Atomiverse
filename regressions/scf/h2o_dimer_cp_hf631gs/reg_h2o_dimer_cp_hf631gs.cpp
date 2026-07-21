/// @file regressions/scf/h2o_dimer_cp_hf631gs/reg_h2o_dimer_cp_hf631gs.cpp
/// @brief Water-dimer HF/6-31G* Boys–Bernardi counterpoise vs PySCF ghost pin.
///
/// Out-of-library consumer: public headers only (`cpu/scf/methods.h` brings
/// `Counterpoise` + `RHF`; `cpu/frame/frame.h`; `core/units.h`). Links
/// `atomiverse`. Five-SCF CP interaction energy checked to 1e-6 Ha against
/// an externally fixed PySCF 2.14.0 `ghost:` prefix result. See REFERENCE.md.
///
/// Exit non-zero on mismatch. Target runtime ≤ 60 s, single CPU.

#include <cmath>
#include <cstdio>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/scf/methods.h"

int main() {
    using namespace ATV;

    // Fixed water-dimer geometry [Å] — same as the PySCF pin in REFERENCE.md.
    // Monomer A: atoms 0..2 (O H H); monomer B: atoms 3..5 (O H H).
    const double coords[6][3] = {
        {-1.551007, -0.114520, 0.000000}, // O_A
        {-1.934259, 0.762503, 0.000000},  // H_A
        {-0.599677, 0.040712, 0.000000},  // H_A
        {1.350625, 0.111469, 0.000000},   // O_B
        {1.680398, -0.373741, -0.758561}, // H_B
        {1.680398, -0.373741, 0.758561},  // H_B
    };
    const int Z[6] = {8, 1, 1, 8, 1, 1};

    cpu::Frame dimer(6);
    for (int i = 0; i < 6; ++i) {
        dimer.span(kAtom_type)[i] = Z[i];
        dimer.span(kAtom_x)[i] = coords[i][0];
        dimer.span(kAtom_y)[i] = coords[i][1];
        dimer.span(kAtom_z)[i] = coords[i][2];
    }
    dimer.simbox = cpu::Box::cubic(30.0);

    cpu::scf::RHF rhf;
    rhf.basis = cpu::scf::BasisSet::Pople631Gs; // 6-31G*
    rhf.tol = 1e-10;
    rhf.max_iter = 100;

    cpu::scf::Counterpoise cp({0, 1, 2}, {3, 4, 5});
    const auto r = cp.run(dimer, rhf);

    // PySCF 2.14.0 HF/6-31G* cart=True + ghost: prefix — REFERENCE.md.
    // (Atomiverse Pople bases are Cartesian; PySCF default is spherical.)
    constexpr double kEIntCpRef = -0.007542441215193207; // Ha
    constexpr double kBsseRef = -0.001483681026428485;   // Ha
    constexpr double kTol = 1.0e-6;                      // Ha

    if (!std::isfinite(r.e_int_cp) || !std::isfinite(r.bsse)) {
        std::fprintf(stderr, "FAIL: non-finite CP result\n");
        return 1;
    }
    if (std::abs(r.e_int_cp - kEIntCpRef) > kTol) {
        std::fprintf(stderr,
                     "FAIL: e_int_cp got=%.12f want=%.12f tol=%.1e Ha\n",
                     r.e_int_cp, kEIntCpRef, kTol);
        return 1;
    }
    if (std::abs(r.bsse - kBsseRef) > kTol) {
        std::fprintf(stderr, "FAIL: bsse got=%.12f want=%.12f tol=%.1e Ha\n",
                     r.bsse, kBsseRef, kTol);
        return 1;
    }
    // Sign convention lock: bsse ≤ 0; magnitude in the published mHa class.
    if (r.bsse > 0.0) {
        std::fprintf(stderr, "FAIL: bsse=%.12f > 0 (convention lock)\n",
                     r.bsse);
        return 1;
    }
    const double abs_bsse_mHa = std::abs(r.bsse) * 1.0e3;
    if (abs_bsse_mHa < 0.1 || abs_bsse_mHa > 50.0) {
        std::fprintf(stderr,
                     "FAIL: |bsse|=%.3f mHa outside published mHa class "
                     "[0.1, 50]\n",
                     abs_bsse_mHa);
        return 1;
    }

    std::printf("PASS: water-dimer HF/6-31G* CP  e_int_cp=%.12f Ha "
                "bsse=%.12f Ha  (|bsse|=%.3f mHa)  "
                "[PySCF 2.14.0 ghost pin; Boys & Bernardi 1970]\n",
                r.e_int_cp, r.bsse, abs_bsse_mHa);
    return 0;
}
