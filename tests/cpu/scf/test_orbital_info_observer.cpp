#include "atomiverse/cpu/scf/methods.h"
/// @file tests/cpu/scf/test_orbital_info_observer.cpp
/// @brief Tests for cpu::scf::OrbitalInfoObserver.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/component/context.h"
#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/minimizer/lbfgs.h"
#include "atomiverse/cpu/scf/internal/scf_types.h"
#include "atomiverse/cpu/scf/orbital_info_observer.h"

using namespace ATV;

namespace {

cpu::Frame make_h2(double bond_length) {
    cpu::Frame f(2);
    f.span(ATV::kAtom_x)[0] = 0.0;
    f.span(ATV::kAtom_x)[1] = bond_length;
    f.span(ATV::kAtom_y)[0] = 0.0;
    f.span(ATV::kAtom_y)[1] = 0.0;
    f.span(ATV::kAtom_z)[0] = 0.0;
    f.span(ATV::kAtom_z)[1] = 0.0;
    f.span(ATV::kAtom_mass)[0] = 1.008;
    f.span(ATV::kAtom_mass)[1] = 1.008;
    f.span(ATV::kAtom_type)[0] = 1;
    f.span(ATV::kAtom_type)[1] = 1;
    f.simbox = ATV::cpu::Box::cubic(100.0);
    return f;
}

} // namespace

TEST(OrbitalInfoObserverTest, WritesHeaderAndRowsForH2) {
    auto frame = make_h2(0.74);
    auto tmp = std::filesystem::temp_directory_path() / "atv_obs_h2.tsv";
    std::error_code ec;
    std::filesystem::remove(tmp, ec);

    cpu::scf::RHF scf{};
    auto handle = scf.result_handle();

    cpu::minimizer::Minimize opt;
    opt.add(cpu::minimizer::LBFGS{5, 0.01, 10});
    opt.add_criterion(cpu::ForceConverged{/*ftol=*/0.5});
    opt.on(minimizer::Phase::force).add(std::move(scf));
    opt.on(minimizer::Phase::end_of_step)
        .add(cpu::scf::OrbitalInfoObserver{handle, tmp.string()});
    opt.build();
    opt.run(frame, 2);

    std::ifstream is(tmp);
    ASSERT_TRUE(is.good());
    std::string header;
    std::getline(is, header);
    EXPECT_NE(header.find("homo_eV"), std::string::npos);
    EXPECT_NE(header.find("gap_eV"), std::string::npos);

    int rows = 0;
    double last_homo = 0, last_lumo = 0;
    std::string line;
    while (std::getline(is, line)) {
        if (line.empty())
            continue;
        ++rows;
        int step;
        double homo, lumo, gap;
        std::sscanf(line.c_str(), "%d\t%lf\t%lf\t%lf", &step, &homo, &lumo,
                    &gap);
        last_homo = homo;
        last_lumo = lumo;
    }
    EXPECT_EQ(rows, 2);
    // H2 / STO-3G HOMO sits near -11 eV and LUMO above 0 eV (gap ~ 20-25 eV).
    EXPECT_LT(last_homo, last_lumo);
    EXPECT_LT(last_homo, 0.0);
    std::filesystem::remove(tmp, ec);
}

TEST(OrbitalInfoObserverTest, NullHandleThrows) {
    cpu::scf::MeanFieldOutputs empty{};
    EXPECT_THROW(cpu::scf::OrbitalInfoObserver(empty, ""),
                 std::invalid_argument);
}

// ── Koopmans IP extension (hf-capability-gaps-04-charges) ────────────────
// Hand-built SlaterDeterminant — no SCF run: ip_koopmans_eV() must equal
// −homo_eV(), and the TSV gains an ip_koopmans column [eV].

TEST(OrbitalInfoObserverTest, IpKoopmansFromHandBuiltDeterminant) {
    using cpu::scf::SlaterDeterminant;

    // Two-orbital restricted determinant: eps = (−0.5, 0.2) Ha, HOMO = −0.5.
    cpu::linalg::RealMatrix C = xt::eye<double>(2);
    cpu::linalg::RealVector eps = {-0.5, 0.2};
    cpu::linalg::RealVector occ = {1.0, 0.0};
    auto det = std::make_shared<SlaterDeterminant>(C, eps, occ, 1);
    auto shells =
        std::make_shared<std::vector<cpu::scf::native::Shell>>();
    auto ready = std::make_shared<bool>(true);
    cpu::scf::MeanFieldOutputs handle{
        det, shells, ready, std::make_shared<const xt::xtensor<double, 2>>()};

    auto tmp = std::filesystem::temp_directory_path() / "atv_obs_ip.tsv";
    std::error_code ec;
    std::filesystem::remove(tmp, ec);

    cpu::scf::OrbitalInfoObserver obs{handle, tmp.string()};
    auto frame = make_h2(0.74);
    cpu::Context ctx;
    cpu::BuildContext bc{ctx, frame};
    cpu::LaunchContext lc{ctx, frame, 0, 0};
    obs.build(bc);
    obs.launch(lc);

    EXPECT_NEAR(obs.homo_eV(), -0.5 * units::kHartreeInEv, 1e-12);
    EXPECT_NEAR(obs.ip_koopmans_eV(), -obs.homo_eV(), 1e-15);
    EXPECT_NEAR(obs.ip_koopmans_eV(), 0.5 * units::kHartreeInEv, 1e-12);

    std::ifstream is(tmp);
    ASSERT_TRUE(is.good());
    std::string header;
    std::getline(is, header);
    EXPECT_NE(header.find("ip_koopmans"), std::string::npos);
    std::string row;
    std::getline(is, row);
    // Row carries 5 tab-separated fields; the last is ip_koopmans [eV].
    int step;
    double homo, lumo, gap, ip;
    ASSERT_EQ(std::sscanf(row.c_str(), "%d\t%lf\t%lf\t%lf\t%lf", &step, &homo,
                          &lumo, &gap, &ip),
              5);
    EXPECT_NEAR(ip, -homo, 1e-9);
    std::filesystem::remove(tmp, ec);

    cpu::DestroyContext dc{ctx, frame};
    obs.destroy(dc);
}
