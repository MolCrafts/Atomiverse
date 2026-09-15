/// @file cpu/vibration/phonon.cpp
/// @brief cpu::vib::Phonon — supercell FD phonon driver run loop.

#include "atomiverse/cpu/vibration/phonon.h"

namespace ATV {
namespace cpu {
namespace vib {

PhononResult Phonon::run(cpu::Frame& unit) {
    if (na < 1 || nb < 1 || nc < 1)
        throw std::invalid_argument(
            "Phonon::run: supercell factors na,nb,nc must be ≥ 1");

    const int n_unit = unit.nlocal();
    if (n_unit <= 0)
        throw std::invalid_argument("Phonon::run: unit cell has no atoms");

    // Work frame: expand when supercell > 1×1×1; else run in place on unit.
    const bool expanded = !(na == 1 && nb == 1 && nc == 1);
    cpu::Frame owned_sc(0);
    cpu::Frame* work = &unit;
    if (expanded) {
        owned_sc = expand_supercell(unit, na, nb, nc);
        work = &owned_sc;
    }

    const int n_atoms = work->nlocal();
    workspace_->allocate(n_atoms);

    const int n_disp = 6 * n_atoms;
    setup_pass(*work);
    for (int d = 0; d < n_disp; ++d)
        step_once(d);
    step_once(n_disp); // finalize: symmetrize, mass-weight, diagonalize
    teardown_pass();

    last_result_.frequencies = workspace_->frequencies;
    last_result_.hessian = workspace_->hessian;
    last_result_.normal_modes = workspace_->normal_modes;
    last_result_.n_atoms = n_atoms;
    last_result_.n_unit = n_unit;
    last_result_.na = na;
    last_result_.nb = nb;
    last_result_.nc = nc;
    last_result_.frequency_units = "cm^-1";
    last_result_.hessian_units = "eV/Angstrom^2";
    return last_result_;
}

} // namespace vib
} // namespace cpu
} // namespace ATV
