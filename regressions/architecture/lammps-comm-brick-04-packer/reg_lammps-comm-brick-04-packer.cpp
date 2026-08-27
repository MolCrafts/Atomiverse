/// @file reg_lammps-comm-brick-04-packer.cpp
/// @brief Hard-coded packer goldens: XSHAKE 19.0 Å, VP 0.4 Å/fs, q 0.25 e.
///
/// Public headers + link atomiverse. No third-party. Literature: Andersen
/// JCP 52 (1983); Ahrens-Iwers JCP 157 (2022); LAMMPS 78a7ddf
/// src/RIGID/fix_rattle.cpp:652-678.

#include <cstdio>
#include <cstdlib>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/comm.h"
#include "atomiverse/cpu/component/component.h"
#include "atomiverse/cpu/dynamics/charge_packer.h"
#include "atomiverse/cpu/dynamics/fix_rattle.h"
#include "atomiverse/cpu/frame/frame.h"

using namespace ATV;

namespace {

cpu::Frame one_atom() {
    cpu::Frame f(1);
    f.span(kAtom_x)[0] = 9.0;
    f.span(kAtom_y)[0] = 0.0;
    f.span(kAtom_z)[0] = 0.0;
    f.span(kAtom_vx)[0] = 0.4;
    f.span(kAtom_vy)[0] = 0.0;
    f.span(kAtom_vz)[0] = 0.0;
    f.span(kAtom_charge)[0] = 0.25;
    f.span(kAtom_mass)[0] = 1.0;
    f.box = cpu::Box::cubic(10.0);
    return f;
}

} // namespace

int main() {
    cpu::Frame frame = one_atom();
    cpu::Context ctx;
    cpu::md::FixRattle rattle{1.0};
    cpu::LaunchContext lc{ctx, frame,
                          ATV::slot_of(ATV::md::Phase::post_integrate), 0};
    rattle.launch(lc);

    int list[] = {0};
    int pbc[] = {1, 0, 0};
    double xshake[3] = {};
    rattle.set_comm_mode(cpu::md::FixRattle::CommMode::XSHAKE);
    rattle.pack_forward_comm(1, list, xshake, 1, pbc);
    std::printf("xshake %.8f Å\n", xshake[0]);

    double vp[3] = {};
    rattle.set_comm_mode(cpu::md::FixRattle::CommMode::VP);
    rattle.pack_forward_comm(1, list, vp, 1, pbc);
    std::printf("vp %.8f Å/fs\n", vp[0]);

    cpu::ChargePacker charge{frame};
    double q[1] = {};
    charge.pack_forward_comm(1, list, q, 1, pbc);
    std::printf("charge %.8f e\n", q[0]);

    if (xshake[0] != 19.0) {
        std::fprintf(stderr, "FAIL: XSHAKE=%.8f Å want 19.0 Å\n", xshake[0]);
        return EXIT_FAILURE;
    }
    if (vp[0] != 0.4) {
        std::fprintf(stderr, "FAIL: VP=%.8f Å/fs want 0.4 Å/fs\n", vp[0]);
        return EXIT_FAILURE;
    }
    if (q[0] != 0.25) {
        std::fprintf(stderr, "FAIL: charge=%.8f e want 0.25 e\n", q[0]);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
