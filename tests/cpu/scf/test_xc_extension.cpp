#include "../../../examples/xc-extension/scaled_slater.h"
#include "atomiverse/cpu/scf/rks.h"
#include "atomiverse/cpu/scf/tdhf.h"
#include "atomiverse/cpu/scf/xc_functional.h"
#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <memory>

namespace {
using namespace ATV::electronic;
using namespace ATV::cpu::scf;

TEST(XcExtension, BuiltinBatchPreservesEnergyDensityAndSpinDerivatives) {
    std::array<double, 4> a{0.3, 0.7, 1.0, 0.0}, b{0.2, 0.1, 0.0, 0.0};
    std::array<double, 4> aa{0.1, 0.2, 0.3, 0.0}, ab{0.02, 0.01, 0, 0},
        bb{0.04, 0.03, 0, 0};
    for (auto type : {XCType::LSDA, XCType::PBE, XCType::BLYP, XCType::B3LYP}) {
        BuiltinXc functional(type);
        XcBatch in;
        in.rho_alpha = a;
        in.rho_beta = b;
        in.sigma_aa = aa;
        in.sigma_ab = ab;
        in.sigma_bb = bb;
        std::array<double, 4> e{}, va{}, vb{}, sa{}, sab{}, sb{};
        XcOutput out{e, va, vb, sa, sab, sb};
        functional.evaluate(in, XcOrder::First, out);
        std::array<double, 4> energy_only{};
        XcOutput energy_output;
        energy_output.energy_density = energy_only;
        functional.evaluate(in, XcOrder::Energy, energy_output);
        for (std::size_t i = 0; i < a.size(); ++i) {
            const auto ref = evaluate_xc(type, a[i], b[i], aa[i], ab[i], bb[i]);
            EXPECT_NEAR(e[i], ref.energy_density, 1e-12);
            EXPECT_DOUBLE_EQ(e[i], energy_only[i]);
            EXPECT_NEAR(va[i], ref.vrho_alpha, 1e-12);
            EXPECT_NEAR(vb[i], ref.vrho_beta, 1e-12);
            if (is_gga(type)) {
                EXPECT_NEAR(sa[i], ref.vsigma_aa, 1e-12);
                EXPECT_NEAR(sab[i], ref.vsigma_ab, 1e-12);
                EXPECT_NEAR(sb[i], ref.vsigma_bb, 1e-12);
            }
        }
    }
}

TEST(XcExtension, ExternalSlaterBatchMatchesAnalyticSpinScaling) {
    std::array<double, 2> a{0.125, 0}, b{0.027, 0}, e{}, va{}, vb{};
    XcBatch in;
    in.rho_alpha = a;
    in.rho_beta = b;
    XcOutput out{e, va, vb};
    xc_example::ScaledSlater f;
    f.evaluate(in, XcOrder::First, out);
    const double c = 0.8 * 0.75 * std::cbrt(6.0 / std::numbers::pi);
    EXPECT_NEAR(e[0], -c * (0.0625 + 0.0081), 1e-15);
    EXPECT_NEAR(va[0], -(4.0 / 3.0) * c * 0.5, 1e-15);
    EXPECT_NEAR(vb[0], -(4.0 / 3.0) * c * 0.3, 1e-15);
    EXPECT_EQ(e[1], 0);
    EXPECT_EQ(va[1], 0);
    EXPECT_EQ(vb[1], 0);
}

class Incomplete final : public XcFunctional {
  public:
    XcIngredients ingredients() const noexcept override {
        return XcIngredients::Rho;
    }
    bool supports(XcSpin, XcOrder order) const noexcept override {
        return order != XcOrder::Second;
    }

  protected:
    void evaluate_batch(const XcBatch&, XcOrder, XcOutput& out) const override {
        std::fill(out.energy_density.begin(), out.energy_density.end(), 0.0);
        // Deliberately omit the derivatives; they must never silently become zero.
    }
};

TEST(XcExtension, MissingDerivativesAndUnsupportedOrdersReject) {
    std::array<double, 1> a{0.5}, e{}, va{}, vb{};
    XcBatch in;
    in.rho_alpha = a;
    in.rho_beta = a;
    XcOutput out{e, va, vb};
    Incomplete f;
    EXPECT_THROW(f.evaluate(in, XcOrder::First, out), std::runtime_error);
    EXPECT_THROW(f.evaluate(in, XcOrder::Second, out), std::invalid_argument);
    EXPECT_NO_THROW(f.evaluate(in, XcOrder::Energy, out));
    in.spin = static_cast<XcSpin>(-1);
    EXPECT_THROW(f.evaluate(in, XcOrder::Energy, out), std::invalid_argument);
}

TEST(XcExtension, InvalidBatchRejectsBeforeOutputMutation) {
    std::array<double, 1> a{0.5}, e{42.0};
    XcBatch in;
    in.rho_alpha = a;
    XcOutput out;
    out.energy_density = e;
    EXPECT_THROW(BuiltinXc(XCType::LSDA).evaluate(in, XcOrder::Energy, out),
                 std::invalid_argument);
    EXPECT_EQ(e[0], 42.0);
    in.rho_beta = a;
    // Input is now valid; an incomplete output shape must still reject before
    // poisoning the valid energy span with NaNs.
    EXPECT_THROW(BuiltinXc(XCType::LSDA).evaluate(in, XcOrder::First, out),
                 std::invalid_argument);
    EXPECT_EQ(e[0], 42.0);
    a[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(BuiltinXc(XCType::LSDA).evaluate(in, XcOrder::Energy, out),
                 std::invalid_argument);
    EXPECT_EQ(e[0], 42.0);
    EXPECT_THROW((BuiltinXc(XCType::Custom)), std::invalid_argument);
}

class TauFunctional final : public XcFunctional {
  public:
    XcIngredients ingredients() const noexcept override {
        return XcIngredients::Rho | XcIngredients::Tau;
    }
    bool supports(XcSpin, XcOrder) const noexcept override { return true; }

  protected:
    void evaluate_batch(const XcBatch&, XcOrder, XcOutput&) const override {}
};

TEST(XcExtension, MolecularAdmissionRejectsMissingChannelsAndNullProviders) {
    EXPECT_THROW((RKS(BasisSet::STO3G, std::make_shared<TauFunctional>())),
                 std::invalid_argument);
    EXPECT_THROW((RKS(BasisSet::STO3G, std::shared_ptr<const XcFunctional>{})),
                 std::invalid_argument);
    RKS method(BasisSet::STO3G, std::make_shared<xc_example::ScaledSlater>());
    EXPECT_THROW((TDA<RKS>{method}.run(1)), std::invalid_argument);
    EXPECT_THROW((TDDFT<RKS>{method}.run(1)), std::invalid_argument);
}
} // namespace
