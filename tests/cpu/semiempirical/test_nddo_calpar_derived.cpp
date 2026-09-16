/// @file tests/cpu/semiempirical/test_nddo_calpar_derived.cpp
/// @brief Unit gate: the derived columns of the AM1 / RM1 tables must be the
///        MOPAC `calpar.F90` functions of that same row's own parameters.
///
/// `eisol` (free-atom NDDO energy), `dd`/`qq` (multipole charge separations)
/// and `am`/`ad`/`aq` (multipole additive terms) are **not** published
/// parameters. MOPAC derives all six in `src/models/calpar.F90` from the active
/// parameter set. A method that rewrites the two-electron parameters therefore
/// cannot inherit any of them from another method — which is exactly how RM1
/// shipped a free-atom energy carrying AM1's `gss/gsp/gpp/gp2/hsp`
/// (ΔHf(H2O) was 19 kcal/mol too positive). This gate makes that unwritable.

#include <algorithm>
#include <cmath>

#include <gtest/gtest.h>

#include "atomiverse/cpu/semiempirical/am1_parameters.h"
#include "atomiverse/cpu/semiempirical/rm1_parameters.h"

using namespace ATV::cpu::semiempirical;

namespace {

/// @brief One electron-volt in atomic units, MOPAC `conref_C.F90` fpcref(1,4).
constexpr double kMopacEv = 27.211386245988;

/// @brief The six values MOPAC `calpar.F90` derives for one element.
struct CalparDerived {
    double eisol; ///< free-atom NDDO energy [eV]
    double dd;    ///< s-p dipole charge separation [Bohr]
    double qq;    ///< p-p quadrupole charge separation [Bohr]
    double am;    ///< monopole-monopole additive [au]
    double ad;    ///< dipole additive [au]
    double aq;    ///< quadrupole additive [au]
};

/// @brief Re-derive MOPAC's `calpar.F90` block for one parameter row.
/// @param p Table row (primary parameters only are read).
/// @return The six derived values in MOPAC's units.
CalparDerived calpar(const AM1Parameter& p) {
    const int k = p.iop;
    const int l = std::min(k, 6 - k);
    const double gssc = std::max(p.ios - 1, 0);
    const double gspc = static_cast<double>(p.ios) * k;
    const double gp2c =
        static_cast<double>((k * (k - 1)) / 2) + 0.25 * l * (l - 1);
    const double gppc = -0.25 * l * (l - 1);
    const double hspc = -0.5 * k * p.ios;
    const double hsp = std::max(1.0e-7, p.hsp);
    const double hpp = std::max(0.1, 0.5 * (p.gpp - p.gp2));

    CalparDerived d{};
    d.eisol = p.uss * p.ios + p.upp * p.iop + p.udd * p.iod + p.gss * gssc +
              p.gpp * gppc + p.gsp * gspc + p.gp2 * gp2c + hsp * hspc;
    d.am = p.gss / kMopacEv;

    if (p.zp < 1.0e-8) { // s-only valence shell (H): MOPAC's post-loop branch
        d.dd = 0.0;
        d.qq = 0.0;
        d.ad = d.am;
        d.aq = d.am;
        d.eisol = p.uss;
        return d;
    }

    // Principal quantum number of the valence shell (MOPAC `nspqn`).
    const double qn = (p.iop > 0) ? p.principal_p : p.principal_s;
    d.dd = (2.0 * qn + 1.0) * std::pow(4.0 * p.zs * p.zp, qn + 0.5) /
           std::pow(p.zs + p.zp, 2.0 * qn + 2.0) / std::sqrt(3.0);
    d.qq = std::sqrt((4.0 * qn * qn + 6.0 * qn + 2.0) / 20.0) / p.zp;

    // AD solves  hsp/ev = 0.5 x - 0.5 / sqrt(4 dd^2 + 1/x^2), 5 secant steps
    // from x = (hsp / (ev dd^2))^(1/3).
    const double dd2 = d.dd * d.dd;
    double x1 = std::cbrt(hsp / (kMopacEv * dd2));
    double x2 = x1 + 0.04;
    for (int j = 0; j < 5; ++j) {
        const double dx = x2 - x1;
        const double f1 =
            0.5 * x1 - 0.5 / std::sqrt(4.0 * dd2 + 1.0 / (x1 * x1));
        const double f2 =
            0.5 * x2 - 0.5 / std::sqrt(4.0 * dd2 + 1.0 / (x2 * x2));
        if (std::abs(f2 - f1) < 1.0e-25)
            break;
        const double x3 = x1 + dx * (hsp / kMopacEv - f1) / (f2 - f1);
        x1 = x2;
        x2 = x3;
    }
    d.ad = x2;

    // AQ solves  hpp/ev = 0.25 x - 0.5 / sqrt(4 qq^2 + 1/x^2)
    //                          + 0.25 / sqrt(8 qq^2 + 1/x^2), 5 secant steps
    // from x = (16 hpp / (48 ev qq^4))^0.2.
    const double qq2 = d.qq * d.qq;
    double y1 = std::pow(16.0 * hpp / (kMopacEv * 48.0 * qq2 * qq2), 0.2);
    double y2 = y1 + 0.04;
    for (int j = 0; j < 5; ++j) {
        const double dy = y2 - y1;
        const double f1 = 0.25 * y1
                          - 0.5 / std::sqrt(4.0 * qq2 + 1.0 / (y1 * y1))
                          + 0.25 / std::sqrt(8.0 * qq2 + 1.0 / (y1 * y1));
        const double f2 = 0.25 * y2
                          - 0.5 / std::sqrt(4.0 * qq2 + 1.0 / (y2 * y2))
                          + 0.25 / std::sqrt(8.0 * qq2 + 1.0 / (y2 * y2));
        if (std::abs(f2 - f1) < 1.0e-25)
            break;
        const double y3 = y1 + dy * (hpp / kMopacEv - f1) / (f2 - f1);
        y1 = y2;
        y2 = y3;
    }
    d.aq = y2;
    return d;
}

void expect_row_matches_calpar(const AM1Parameter& p) {
    const CalparDerived d = calpar(p);
    SCOPED_TRACE(testing::Message()
                 << "Z=" << p.atomic_number << " " << p.symbol);
    EXPECT_NEAR(p.eisol, d.eisol, 1.0e-7) << "eisol [eV]";
    EXPECT_NEAR(p.dd, d.dd, 1.0e-8) << "dd [Bohr]";
    EXPECT_NEAR(p.qq, d.qq, 1.0e-8) << "qq [Bohr]";
    EXPECT_NEAR(p.am, d.am, 1.0e-8) << "am [au]";
    EXPECT_NEAR(p.ad, d.ad, 1.0e-8) << "ad [au]";
    EXPECT_NEAR(p.aq, d.aq, 1.0e-8) << "aq [au]";
}

} // namespace

TEST(NddoCalparDerived, Am1TableIsSelfConsistent) {
    int rows = 0;
    for (int z = 1; z <= 100; ++z) {
        const auto* p = am1_parameter(z);
        if (p == nullptr)
            continue;
        ++rows;
        expect_row_matches_calpar(*p);
    }
    EXPECT_GT(rows, 30) << "AM1 table unexpectedly small";
}

TEST(NddoCalparDerived, Rm1TableIsSelfConsistent) {
    int rows = 0;
    for (int z = 1; z <= 100; ++z) {
        const auto* p = rm1_parameter(z);
        if (p == nullptr)
            continue;
        ++rows;
        expect_row_matches_calpar(*p);
    }
    EXPECT_EQ(rows, 6) << "RM1 v1 coverage is H, C, N, O, P, S";
}

TEST(NddoCalparDerived, Rm1OxygenEisolIsNotTheAm1Shift) {
    const auto* o_rm1 = rm1_parameter(8);
    const auto* o_am1 = am1_parameter(8);
    ASSERT_NE(o_rm1, nullptr);
    ASSERT_NE(o_am1, nullptr);

    // The fossil: AM1's eisol shifted by the one-electron change only.
    const double am1_shift = o_am1->eisol +
                             o_rm1->ios * (o_rm1->uss - o_am1->uss) +
                             o_rm1->iop * (o_rm1->upp - o_am1->upp);
    EXPECT_NEAR(am1_shift, -312.8526806, 1.0e-6) << "fossil reconstructed";
    EXPECT_GT(std::abs(o_rm1->eisol - am1_shift), 0.5)
        << "RM1 eisol must not be AM1's shifted by uss/upp alone";
    EXPECT_NEAR(o_rm1->eisol, -312.0403541, 1.0e-6)
        << "MOPAC calpar value [eV]";
}
