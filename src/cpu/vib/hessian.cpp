/// @file src/cpu/vib/hessian.cpp
/// @brief cpu::vib::Hessian — constructor wiring and run() implementation.

#include "atomiverse/cpu/vib/hessian.h"

#include "atomiverse/cpu/vib/frequency_report.h"
#include "atomiverse/cpu/vib/hessian_diagonalizer.h"
#include "atomiverse/cpu/vib/mass_weighting.h"
#include "atomiverse/cpu/vib/normal_mode_writer.h"
#include "atomiverse/cpu/vib/thermochem.h"

namespace ATV {
namespace cpu {
namespace vib {

namespace {

/// @brief Register the full vibrational-analysis component chain.
///
/// Auto-registers the four numerical-Hessian structural components, the
/// two frequency components, and the three thermochemistry-link
/// components, all bound to the driver-owned shared workspace. On
/// `vib::Phase::diagonalize` the ordering is Symmetrize (-1) ->
/// MassWeighting (0) -> HessianDiagonalizer (+1); on `vib::Phase::thermochem`
/// the Thermochem Observer runs; on `vib::Phase::end_of_step` the
/// FrequencyReport (0) and NormalModeWriter (+1) Recorders run.
///
/// @param driver Driver to register the components into.
/// @param ws     Driver-owned shared workspace.
/// @param tc     Thermochemistry inputs for the Thermochem component.
/// @param out    Output paths for the FrequencyReport / NormalModeWriter.
template <class Driver>
void wire_full_chain(Driver& driver, HessianWorkspace* ws,
                     const Thermochem::Config& tc,
                     const Hessian::OutputConfig& out) {
    driver.add(ReferenceGeometry{ws});
    driver.add(DisplaceStep{ws});
    driver.add(HessianAccumulator{ws});
    driver.add(Symmetrize{ws});
    driver.add(MassWeighting{ws});
    driver.add(HessianDiagonalizer{ws});
    driver.add(Thermochem{ws, tc});
    driver.add(FrequencyReport{ws, out.frequency_tsv});
    driver.add(NormalModeWriter{ws, out.mode_xyz_prefix});
}

} // namespace

Hessian::Hessian() {
    // Default thermochemistry: standard conditions, closed-shell singlet.
    wire_full_chain(*this, workspace_.get(), Thermochem::Config{},
                    OutputConfig{});
}

Hessian::Hessian(const Thermochem::Config& tc) {
    wire_full_chain(*this, workspace_.get(), tc, OutputConfig{});
}

Hessian::Hessian(const Thermochem::Config& tc, const OutputConfig& out) {
    wire_full_chain(*this, workspace_.get(), tc, out);
}

Hessian::Hessian(WithoutAccumulator /*tag*/) {
    // Test-only configuration: omit HessianAccumulator so the build-time
    // validation throws for the missing Capability::Hessian component.
    add(ReferenceGeometry{workspace_.get()});
    add(DisplaceStep{workspace_.get()});
    add(Symmetrize{workspace_.get()});
}

HessianResult Hessian::run(cpu::Frame& frame) {
    // Size the shared workspace from the frame before the component loop
    // runs — no allocation happens inside any component launch().
    const int n_atoms = frame.nlocal();
    workspace_->allocate(n_atoms);

    // The 6N-displacement loop: two displacements per Cartesian coordinate. A fixed
    // budget, not a policy — every displacement must complete for the Hessian to be
    // defined, so there is nothing for a criterion to decide and none is polled.
    const int n_disp = 6 * n_atoms;
    setup_pass(frame);
    for (int d = 0; d < n_disp; ++d)
        step_once(d);
    // The finalize pass: Symmetrize, MassWeighting and the diagonalizer run here;
    // DisplaceStep restores the equilibrium geometry.
    step_once(n_disp);
    teardown_pass();

    // Publish the assembled, symmetrized Hessian, the vibrational
    // frequencies (MassWeighting + HessianDiagonalizer), and the harmonic
    // thermochemistry (Thermochem).
    last_result_.hessian = workspace_->hessian;
    last_result_.n_atoms = n_atoms;
    last_result_.units = "eV/Angstrom^2";
    last_result_.frequencies = workspace_->frequencies;
    last_result_.normal_modes = workspace_->normal_modes;
    last_result_.n_imaginary = workspace_->n_imaginary;
    last_result_.is_linear = workspace_->is_linear;
    last_result_.zpe = workspace_->zpe;
    last_result_.u_tot = workspace_->u_tot;
    last_result_.h_tot = workspace_->h_tot;
    last_result_.s_tot = workspace_->s_tot;
    last_result_.s_elec = workspace_->s_elec;
    last_result_.s_trans = workspace_->s_trans;
    last_result_.s_rot = workspace_->s_rot;
    last_result_.s_vib = workspace_->s_vib;
    last_result_.g_tot = workspace_->g_tot;
    last_result_.g_thermal = workspace_->g_thermal;
    return last_result_;
}

} // namespace vib
} // namespace cpu
} // namespace ATV
