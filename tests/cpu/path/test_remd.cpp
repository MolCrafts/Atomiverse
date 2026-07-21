/// @file tests/cpu/path/test_remd.cpp
#include <gtest/gtest.h>

#include "atomiverse/cpu/frame/frame.h"
#include "atomiverse/cpu/md/md.h"
#include "atomiverse/cpu/neighbor/neighbor_list.h"
#include "atomiverse/cpu/path/remd.h"
#include "atomiverse/cpu/potentials/pair_lj_cut.h"

using namespace ATV;

namespace {
void build_lj(cpu::md::MD& md, double /*T*/, void*) {
    md.add_potential(cpu::PairLJCut{0.0104, 3.4, 8.5});
    md.set_neighbor(cpu::md::NeighborList<>{8.5, 0.3});
}
} // namespace

TEST(Remd, BuildsAndRunsExchangeRounds) {
    cpu::Frame f(2);
    f.span(kAtom_x)[0] = 0;
    f.span(kAtom_y)[0] = 0;
    f.span(kAtom_z)[0] = 0;
    f.span(kAtom_x)[1] = 4.0;
    f.span(kAtom_y)[1] = 0;
    f.span(kAtom_z)[1] = 0;
    for (int i = 0; i < 2; ++i) {
        f.span(kAtom_mass)[i] = 39.948;
        f.span(kAtom_type)[i] = 0;
        f.span(kAtom_tag)[i] = i + 1;
    }
    f.simbox = cpu::Box::cubic(20.0);

    cpu::path::Remd remd({{100.0, 200.0, 300.0, 400.0}, 1.0, 5, 42});
    remd.set_program_builder(&build_lj, nullptr);
    remd.run(f, /*n_exchanges=*/4);
    EXPECT_GT(remd.exchange_attempts(), 0);
    // Accepts may be 0 for short run; still a valid outcome.
    EXPECT_GE(remd.exchange_accepts(), 0);
    EXPECT_LE(remd.exchange_accepts(), remd.exchange_attempts());
    EXPECT_EQ(f.nlocal(), 2);
}

TEST(Remd, RejectsNonMonotoneTemperatures) {
    EXPECT_THROW(cpu::path::Remd({{300.0, 200.0}, 1.0, 1, 1}),
                 std::invalid_argument);
}
