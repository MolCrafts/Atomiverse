/// @file tests/cpu/semiempirical/test_dftb_published_constants.cpp
/// @brief Unit tests for published DFTB U^d / W tables, HOAO collapse, and
///        unset-parameter refusal.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/log.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_3ob_hubbard_derivs.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_3ob_tables.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_spin_constants.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"
#include "atomiverse/cpu/semiempirical/dftb3.h"
#include "atomiverse/cpu/semiempirical/internal/dftb3_scc.h"
#include "atomiverse/cpu/semiempirical/internal/dftb_spin.h"
#include "atomiverse/cpu/semiempirical/td_dftb2.h"

using namespace ATV;
using namespace ATV::cpu::semiempirical;

template <typename T, typename = void>
struct HasHubbardDu : std::false_type {};
template <typename T>
struct HasHubbardDu<T, std::void_t<decltype(T::hubbard_du)>> : std::true_type {
};

template <typename T, typename = void>
struct HasSpinW : std::false_type {};
template <typename T>
struct HasSpinW<T, std::void_t<decltype(T::spin_w)>> : std::true_type {};

namespace {

cpu::Frame make_h2(double r = 0.74) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 1;
    f.span(kAtom_mass)[0] = f.span(kAtom_mass)[1] = 1.008;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_tag)[1] = 2;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    for (int i = 0; i < 2; ++i)
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    return f;
}

cpu::Frame make_water() {
    cpu::Frame f(3);
    f.span(kAtom_type)[0] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_type)[1] = 1;
    f.span(kAtom_x)[1] = 0.96;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    f.span(kAtom_type)[2] = 1;
    f.span(kAtom_x)[2] = -0.24;
    f.span(kAtom_y)[2] = 0.93;
    f.span(kAtom_z)[2] = 0.0;
    for (int i = 0; i < 3; ++i) {
        f.span(kAtom_mass)[i] = (i == 0) ? 15.999 : 1.008;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

cpu::Frame make_o2(double r = 1.21) {
    cpu::Frame f(2);
    f.span(kAtom_type)[0] = f.span(kAtom_type)[1] = 8;
    f.span(kAtom_x)[0] = 0.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_x)[1] = r;
    f.span(kAtom_y)[1] = 0.0;
    f.span(kAtom_z)[1] = 0.0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 15.999;
        f.span(kAtom_tag)[i] = i + 1;
        f.span(kAtom_fx)[i] = f.span(kAtom_fy)[i] = f.span(kAtom_fz)[i] = 0;
    }
    f.box = cpu::Box::cubic(30.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

cpu::Frame make_fluorine() {
    cpu::Frame f(1);
    f.span(kAtom_type)[0] = 9;
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_mass)[0] = 18.998;
    f.span(kAtom_tag)[0] = 1;
    f.span(kAtom_fx)[0] = f.span(kAtom_fy)[0] = f.span(kAtom_fz)[0] = 0;
    f.box = cpu::Box::cubic(20.0);
    f.meta.get(kEnergy) = 0;
    return f;
}

const basis_data::DftbSpinConstant*
find_spin(const std::vector<basis_data::DftbSpinConstant>& table, int z) {
    for (const auto& row : table) {
        if (row.z == z)
            return &row;
    }
    return nullptr;
}

int count_substr(const std::string& hay, const std::string& needle) {
    int n = 0;
    for (std::size_t p = 0; (p = hay.find(needle, p)) != std::string::npos;
         p += needle.size())
        ++n;
    return n;
}

template <typename Ex, typename Fn>
void expect_throw_has(Fn&& fn, std::initializer_list<const char*> needles) {
    try {
        fn();
        FAIL() << "expected an exception that was not thrown";
    } catch (const Ex& e) {
        const std::string msg(e.what());
        for (const char* n : needles)
            EXPECT_NE(msg.find(n), std::string::npos) << msg;
    }
}

} // namespace

// ── Happy path: published tables ───────────────────────────────────────

TEST(DftbPublishedConstants, HubbardDerivsHaExactHCNO) {
    using namespace basis_data;
    EXPECT_EQ(k3obHubbardDerivsHa.size(), 96u);
    EXPECT_EQ(k3obHubbardDerivsHa[1], -0.1857);
    EXPECT_EQ(k3obHubbardDerivsHa[6], -0.1492);
    EXPECT_EQ(k3obHubbardDerivsHa[7], -0.1535);
    EXPECT_EQ(k3obHubbardDerivsHa[8], -0.1575);
}

TEST(DftbPublishedConstants, HubbardDerivsHaZeroOutsideHCNO) {
    using namespace basis_data;
    for (int z = 0; z <= 95; ++z) {
        if (z == 1 || z == 6 || z == 7 || z == 8)
            continue;
        EXPECT_EQ(k3obHubbardDerivsHa[static_cast<std::size_t>(z)], 0.0)
            << "Z=" << z;
    }
}

TEST(DftbPublishedConstants, SpinConstantsPbeOxygenRow) {
    const auto* o = find_spin(basis_data::kDftbSpinConstantsPbeHa, 8);
    ASSERT_NE(o, nullptr);
    EXPECT_EQ(o->n_shell, 2);
    EXPECT_EQ(o->w_ha[1][1], -0.028);
    EXPECT_EQ(o->w_ha[0][1], -0.030);
    EXPECT_EQ(o->w_ha[1][0], -0.030);
    EXPECT_EQ(o->w_ha[0][0], -0.035);
}

TEST(DftbPublishedConstants, SpinConstantsPbeHydrogenRow) {
    const auto* h = find_spin(basis_data::kDftbSpinConstantsPbeHa, 1);
    ASSERT_NE(h, nullptr);
    EXPECT_EQ(h->n_shell, 1);
    EXPECT_EQ(h->w_ha[0][0], -0.072);
}

TEST(DftbPublishedConstants, SpinConstantsLdaHydrogenWss) {
    const auto* h = find_spin(basis_data::kDftbSpinConstantsLdaHa, 1);
    ASSERT_NE(h, nullptr);
    EXPECT_EQ(h->n_shell, 1);
    EXPECT_EQ(h->w_ha[0][0], -0.064);
}

// ── Happy path: HOAO collapse + Ha→eV ──────────────────────────────────

TEST(DftbPublishedConstants, CollapseSpinWEvHoao) {
    using basis_data::kDftbSpinConstantsPbeHa;
    const auto w8 = internal::collapse_spin_w_ev(kDftbSpinConstantsPbeHa, {8});
    const auto w1 = internal::collapse_spin_w_ev(kDftbSpinConstantsPbeHa, {1});
    const auto w6 = internal::collapse_spin_w_ev(kDftbSpinConstantsPbeHa, {6});
    const auto w7 = internal::collapse_spin_w_ev(kDftbSpinConstantsPbeHa, {7});
    ASSERT_EQ(w8.size(), 1u);
    ASSERT_EQ(w1.size(), 1u);
    ASSERT_EQ(w6.size(), 1u);
    ASSERT_EQ(w7.size(), 1u);
    EXPECT_NEAR(w8[0], -0.761918814888, 1e-12);
    EXPECT_NEAR(w1[0], -1.959219809711, 1e-12);
    EXPECT_NEAR(w6[0], -0.625861883658, 1e-12);
    EXPECT_NEAR(w7[0], -0.707496042396, 1e-12);
}

// ── Edge: unset W / U^d refuses the taken path ─────────────────────────

TEST(DftbPublishedConstants, Dftb2LaunchThrowsWhenSpinWHaEmptyOpenShell) {
    auto f = make_o2();
    cpu::Context ctx;
    DFTB2 d;
    d.boundary = DftbBoundary::cluster;
    d.multiplicity = 3;
    d.with_forces = false;
    d.build({ctx, f});
    expect_throw_has<std::invalid_argument>(
        [&] { d.launch({ctx, f, 0, 0}); },
        {"spin_w_ha", "d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;"});
}

TEST(DftbPublishedConstants, Dftb2TdCasidaThrowsWhenSpinWHaEmptyTriplet) {
    auto f = make_h2();
    DFTB2 d;
    d.boundary = DftbBoundary::cluster;
    expect_throw_has<std::invalid_argument>(
        [&] { d.td_casida(f, 1, TdDftbChannel::triplet); },
        {"spin_w_ha", "d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;"});
}

TEST(DftbPublishedConstants, Dftb3LaunchThrowsWhenHubbardDuHaUnset) {
    auto f = make_water();
    cpu::Context ctx;
    DFTB3 d3;
    d3.sk_set = DftbSkSet::MioSketch;
    d3.with_forces = false;
    d3.build({ctx, f});
    expect_throw_has<std::invalid_argument>(
        [&] { d3.launch({ctx, f, 0, 0}); },
        {"hubbard_du_ha",
         "d3.hubbard_du_ha = basis_data::k3obHubbardDerivsHa;"});
}

TEST(DftbPublishedConstants, Dftb3LaunchThrowsWhenHubbardDuHaMissingZ) {
    auto f = make_fluorine();
    cpu::Context ctx;
    DFTB3 d3;
    d3.sk_set = DftbSkSet::MioSketch;
    d3.hubbard_du_ha = basis_data::k3obHubbardDerivsHa;
    d3.with_forces = false;
    d3.build({ctx, f});
    expect_throw_has<std::invalid_argument>([&] { d3.launch({ctx, f, 0, 0}); },
                                            {"hubbard_du_ha"});
}

// ── Edge: DFTB3 SK selector (Unset / ThreeOb / MioSketch-warn) ─────────

TEST(DftbPublishedConstants, Dftb3LaunchThrowsWhenSkSetUnset) {
    auto f = make_h2();
    cpu::Context ctx;
    DFTB3 d3;
    EXPECT_EQ(d3.sk_set, DftbSkSet::Unset);
    d3.hubbard_du_ha = basis_data::k3obHubbardDerivsHa;
    d3.with_forces = false;
    d3.build({ctx, f});
    expect_throw_has<std::invalid_argument>(
        [&] { d3.launch({ctx, f, 0, 0}); },
        {"sk_set", "d3.sk_set = cpu::semiempirical::DftbSkSet::MioSketch;"});
}

TEST(DftbPublishedConstants, Dftb3LaunchChecksSkSetBeforeHubbardDu) {
    auto f = make_h2();
    cpu::Context ctx;
    DFTB3 d3;
    EXPECT_EQ(d3.sk_set, DftbSkSet::Unset);
    d3.with_forces = false;
    d3.build({ctx, f});
    expect_throw_has<std::invalid_argument>(
        [&] { d3.launch({ctx, f, 0, 0}); },
        {"sk_set", "d3.sk_set = cpu::semiempirical::DftbSkSet::MioSketch;"});
}

TEST(DftbPublishedConstants, Dftb3LaunchThrowsWhenSkSetThreeOb) {
    auto f = make_h2();
    cpu::Context ctx;
    DFTB3 d3;
    d3.sk_set = DftbSkSet::ThreeOb;
    d3.hubbard_du_ha = basis_data::k3obHubbardDerivsHa;
    d3.with_forces = false;
    d3.build({ctx, f});
    expect_throw_has<std::runtime_error>([&] { d3.launch({ctx, f, 0, 0}); },
                                         {"dftb-gamma-h-damping"});
}

TEST(DftbPublishedConstants, Dftb3MioSketchLaunchWarnsOnceAndReturnsEnergy) {
    auto f = make_h2();
    cpu::Context ctx;
    DFTB3 d3;
    d3.sk_set = DftbSkSet::MioSketch;
    d3.hubbard_du_ha = basis_data::k3obHubbardDerivsHa;
    d3.with_forces = false;

    const std::string path = "atv_test_dftb3_sk_set.log";
    std::remove(path.c_str());
    LogConfig lc;
    lc.console = false;
    lc.logfile = path;
    log_init(lc);

    d3.build({ctx, f});
    d3.launch({ctx, f, 0, 0});
    const double e1 = d3.last_energy();
    EXPECT_TRUE(std::isfinite(e1));
    d3.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(d3.last_energy()));
    logger().flush();

    std::ifstream in(path);
    ASSERT_TRUE(in.good());
    const std::string content((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
    EXPECT_EQ(count_substr(content, "supersedes the MIO parameter set"), 1);
    std::remove(path.c_str());
}

// ── Edge: shell_resolved_spin ──────────────────────────────────────────

TEST(DftbPublishedConstants, Dftb2LaunchThrowsWhenShellResolvedSpinTrue) {
    auto f = make_o2();
    cpu::Context ctx;
    DFTB2 d;
    d.boundary = DftbBoundary::cluster;
    d.multiplicity = 3;
    d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;
    d.shell_resolved_spin = true;
    d.with_forces = false;
    d.build({ctx, f});
    expect_throw_has<std::runtime_error>([&] { d.launch({ctx, f, 0, 0}); },
                                         {"dftb-shell-resolved-spin"});
}

TEST(DftbPublishedConstants, Dftb2TdCasidaThrowsWhenShellResolvedSpinTrue) {
    auto f = make_h2();
    DFTB2 d;
    d.boundary = DftbBoundary::cluster;
    d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;
    d.shell_resolved_spin = true;
    expect_throw_has<std::runtime_error>(
        [&] { d.td_casida(f, 1, TdDftbChannel::triplet); },
        {"dftb-shell-resolved-spin"});
}

TEST(DftbPublishedConstants,
     Dftb2OpenShellLaunchDoesNotThrowWhenShellResolvedSpinFalse) {
    auto f = make_o2();
    cpu::Context ctx;
    DFTB2 d;
    d.boundary = DftbBoundary::cluster;
    d.multiplicity = 3;
    d.spin_w_ha = basis_data::kDftbSpinConstantsPbeHa;
    EXPECT_FALSE(d.shell_resolved_spin);
    d.with_forces = false;
    d.build({ctx, f});
    EXPECT_NO_THROW(d.launch({ctx, f, 0, 0}));
}

// ── Edge: DFTB2 Unset sentinel ─────────────────────────────────────────

TEST(DftbPublishedConstants, Dftb2DefaultSkSetIsMioSketch) {
    DFTB2 d;
    EXPECT_EQ(d.sk_set, DftbSkSet::MioSketch);
}

TEST(DftbPublishedConstants, Dftb2DefaultLaunchOnH2Succeeds) {
    auto f = make_h2();
    cpu::Context ctx;
    DFTB2 d;
    d.with_forces = false;
    d.build({ctx, f});
    d.launch({ctx, f, 0, 0});
    EXPECT_TRUE(std::isfinite(d.last_energy()));
}

TEST(DftbPublishedConstants, Dftb2LaunchThrowsWhenSkSetUnset) {
    auto f = make_h2();
    cpu::Context ctx;
    DFTB2 d;
    d.sk_set = DftbSkSet::Unset;
    d.with_forces = false;
    d.build({ctx, f});
    expect_throw_has<std::exception>([&] { d.launch({ctx, f, 0, 0}); },
                                     {"sk_set"});
}

// ── Edge: closed-shell lazy validation ─────────────────────────────────

TEST(DftbPublishedConstants, Dftb2ClosedShellLaunchDoesNotRequireSpinWHa) {
    auto f = make_h2();
    cpu::Context ctx;
    DFTB2 d;
    EXPECT_TRUE(d.spin_w_ha.empty());
    d.multiplicity = 1;
    d.with_forces = false;
    d.build({ctx, f});
    EXPECT_NO_THROW(d.launch({ctx, f, 0, 0}));
}

TEST(DftbPublishedConstants,
     Dftb2ClosedShellDoesNotThrowWhenShellResolvedSpinTrue) {
    auto f = make_h2();
    cpu::Context ctx;
    DFTB2 d;
    d.multiplicity = 1;
    d.shell_resolved_spin = true;
    d.with_forces = false;
    d.build({ctx, f});
    EXPECT_NO_THROW(d.launch({ctx, f, 0, 0}));
}

// ── Edge: Dftb3Scc ctor gate ───────────────────────────────────────────

TEST(DftbPublishedConstants, Dftb3SccCtorThrowsOnAllZeroUd) {
    const std::vector<double> u{10.0, 10.0};
    const std::vector<double> ud{0.0, 0.0};
    const std::vector<double> r{0.0, 1.0, 1.0, 0.0};
    EXPECT_THROW(internal::Dftb3Scc(u, ud, r), std::invalid_argument);
}

TEST(DftbPublishedConstants, Dftb3SccCtorThrowsOnSizeMismatch) {
    const std::vector<double> u{10.0, 10.0};
    const std::vector<double> ud{-1.0};
    const std::vector<double> r{0.0, 1.0, 1.0, 0.0};
    EXPECT_THROW(internal::Dftb3Scc(u, ud, r), std::invalid_argument);
}

// ── Domain: 3ob full-precision storage ─────────────────────────────────

TEST(DftbPublishedConstants, Dftb3obSkPointIsEightDoubles) {
    EXPECT_EQ(sizeof(basis_data::Dftb3obSkPoint), 8 * sizeof(double));
    EXPECT_EQ(sizeof(basis_data::Dftb3obSkPoint::h_ss), sizeof(double));
    // 17-significant-digit SKF literal from tools/gen_dftb_3ob_tables.py
    // (dftb_3ob_tables.h kDftb3obSk_HH[0].h_ss).
    EXPECT_EQ(basis_data::kDftb3obSk_HH[0].h_ss, -5.8561907684890002e-01);
}

// ── Domain: no fabricated residue on DftbElementRow ────────────────────

TEST(DftbPublishedConstants, DftbElementRowHasNoFabricatedHubbardDuOrSpinW) {
    static_assert(!HasHubbardDu<basis_data::DftbElementRow>::value,
                  "DftbElementRow must not carry fabricated hubbard_du");
    static_assert(!HasSpinW<basis_data::DftbElementRow>::value,
                  "DftbElementRow must not carry fabricated spin_w");
    EXPECT_FALSE(HasHubbardDu<basis_data::DftbElementRow>::value);
    EXPECT_FALSE(HasSpinW<basis_data::DftbElementRow>::value);
}
