#include "atomiverse/cpu/planewave/internal/pw_setup.h"
#include <complex>
#include <gtest/gtest.h>

TEST(PwEnergy, SpinDoubleCountingUsesBothChannelsAtTheInputDensity) {
    using namespace ATV;
    using namespace cpu::pw::internal;
    using Complex = std::complex<AccReal>;
    PWWorkspace ws;
    ws.n_g = 1;
    ws.nspin = 2;
    ws.rho_G = xt::xarray<Complex>{2.};
    ws.V_H_G = xt::xarray<Complex>{3.};
    ws.V_xc_G =
        xt::xarray<Complex>{17.}; // Last spin's scratch, not the spin sum.
    ws.V_loc_G = xt::xarray<Complex>{0.};
    ws.E_xc = 4.;
    ws.e_xc_int_spin = 5.;
    const auto spin = iteration_energy(ws, 100., 5., 2.);
    EXPECT_DOUBLE_EQ(spin.e_h, 6.);
    EXPECT_DOUBLE_EQ(spin.e_xc_int, 5.);
    EXPECT_DOUBLE_EQ(spin.e_one, 83.);
    EXPECT_DOUBLE_EQ(spin.total, 98.);
    ws.nspin = 1;
    const auto restricted = iteration_energy(ws, 100., 5., 2.);
    EXPECT_DOUBLE_EQ(restricted.e_xc_int, 68.);
    EXPECT_DOUBLE_EQ(restricted.total, 35.);
}
