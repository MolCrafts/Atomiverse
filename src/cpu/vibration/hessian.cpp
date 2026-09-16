/// @file cpu/vibration/hessian.cpp
/// @brief cpu::vib::Hessian — constructor wiring and run() implementation.

#include "atomiverse/cpu/vibration/hessian.h"

#include <stdexcept>
#include <string>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/zero.h"
#include "atomiverse/cpu/driver/driver.h"
#include "atomiverse/cpu/vibration/frequency_report.h"
#include "atomiverse/cpu/vibration/hessian_diagonalizer.h"
#include "atomiverse/cpu/vibration/mass_weighting.h"
#include "atomiverse/cpu/vibration/normal_mode_writer.h"
#include "atomiverse/cpu/vibration/thermochem.h"

namespace ATV {
namespace cpu {
namespace vib {

namespace {

/// @brief Bind the driver-owned accumulator zeroing for the force phase.
///
/// Zeroing the force/energy accumulators is the hosting driver's job, never a
/// force source's: a source that zeroed would erase a co-bound source. Without
/// it an accumulating (`+=`) classical Potential integrates its forces across
/// the `6N` displacement passes and the central difference returns `k/2`
/// instead of the curvature `k`. Orders `-1000`/`-999` mirror `md::MD` and
/// `vib::Phonon`, so every driver's force slot has the same head.
///
/// Called from **every** construction path (`wire_full_chain` and the
/// `WithoutAccumulator` ctor) exactly once, so the two wiring sites cannot
/// drift apart.
///
/// @param driver Driver to register the two Zero components into.
template <class Driver>
void wire_zeroing(Driver& driver) {
    driver.on(ATV::vib::Phase::force)
        .add(cpu::ZeroForces{}, -1000)
        .add(cpu::ZeroEnergy{}, -999);
}

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
} // namespace

void Hessian::wire_full_chain_(HessianWorkspace* ws, double temperature,
                               double pressure, int multiplicity,
                               int symmetry_number,
                               const std::string& frequency_tsv,
                               const std::string& mode_xyz_prefix) {
    wire_zeroing(*this);
    // default_bindings → protected bind_defaults (no public Driver::add)
    this->bind_defaults(ReferenceGeometry{ws});
    this->bind_defaults(DisplaceStep{ws});
    this->bind_defaults(HessianAccumulator{ws});
    this->bind_defaults(Symmetrize{ws});
    this->bind_defaults(MassWeighting{ws});
    this->bind_defaults(HessianDiagonalizer{ws});
    this->bind_defaults(
        Thermochem{ws, temperature, pressure, multiplicity, symmetry_number});
    this->bind_defaults(FrequencyReport{ws, frequency_tsv});
    this->bind_defaults(NormalModeWriter{ws, mode_xyz_prefix});
}

Hessian::Hessian() {
    // Default thermochemistry: standard conditions, closed-shell singlet.
    wire_full_chain_(workspace_.get(), 298.15, 101325.0, 1, 1,
                     "frequencies.tsv", "mode");
}

Hessian::Hessian(double temperature, double pressure, int multiplicity,
                 int symmetry_number) {
    wire_full_chain_(workspace_.get(), temperature, pressure, multiplicity,
                     symmetry_number, "frequencies.tsv", "mode");
}

Hessian::Hessian(double temperature, double pressure, int multiplicity,
                 int symmetry_number, std::string frequency_tsv,
                 std::string mode_xyz_prefix) {
    wire_full_chain_(workspace_.get(), temperature, pressure, multiplicity,
                     symmetry_number, frequency_tsv, mode_xyz_prefix);
}

Hessian::Hessian(WithoutAccumulator /*tag*/) {
    // Test-only configuration: omit HessianAccumulator so the build-time
    // validation throws for the missing Capability::Hessian component.
    wire_zeroing(*this);
    this->bind_defaults(ReferenceGeometry{workspace_.get()});
    this->bind_defaults(DisplaceStep{workspace_.get()});
    this->bind_defaults(Symmetrize{workspace_.get()});
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

namespace {

/// Hosts MassWeighting + HessianDiagonalizer for one finalize pass when the
/// Cartesian Hessian is already known (no FD loop). Library-internal only.
class CartesianFinalize : public Driver<ATV::vib::Phase> {
  public:
    explicit CartesianFinalize(HessianWorkspace* ws) : mw_(ws), diag_(ws) {
        this->on(ATV::vib::Phase::diagonalize).add_ref(mw_, /*order=*/0);
        this->on(ATV::vib::Phase::diagonalize).add_ref(diag_, /*order=*/+1);
    }

    void run_finalize(cpu::Frame& frame, int n_atoms) {
        this->build();
        setup_pass(frame);
        step_once(6 * n_atoms); // components gate on step >= 6N
        teardown_pass();
    }

  protected:
    void validate_components() const override {}

  private:
    MassWeighting mw_;
    HessianDiagonalizer diag_;
};

} // namespace

HessianResult
Hessian::run_from_cartesian(cpu::Frame& frame,
                            xt::xtensor<double, 2> hessian_ev_ang2) {
    const int n = static_cast<int>(frame.nlocal());
    if (n < 1)
        throw std::invalid_argument("Hessian::run_from_cartesian: empty frame");
    const int dim = 3 * n;
    if (static_cast<int>(hessian_ev_ang2.shape(0)) != dim
        || static_cast<int>(hessian_ev_ang2.shape(1)) != dim) {
        throw std::invalid_argument(
            "Hessian::run_from_cartesian: hessian shape must be [3N,3N] with N="
            + std::to_string(n));
    }

    // Dedicated workspace so a prior / later FD run() is not disturbed.
    HessianWorkspace ws;
    ws.allocate(n);
    ws.hessian = std::move(hessian_ev_ang2);
    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    for (int i = 0; i < n; ++i) {
        ws.r0[static_cast<std::size_t>(3 * i + 0)] = static_cast<double>(x[i]);
        ws.r0[static_cast<std::size_t>(3 * i + 1)] = static_cast<double>(y[i]);
        ws.r0[static_cast<std::size_t>(3 * i + 2)] = static_cast<double>(z[i]);
    }
    ws.r0_ready = true;

    CartesianFinalize host{&ws};
    host.run_finalize(frame, n);

    last_result_.n_atoms = n;
    last_result_.units = "eV/Angstrom^2";
    last_result_.hessian = ws.hessian;
    last_result_.frequencies = ws.frequencies;
    last_result_.normal_modes = ws.normal_modes;
    last_result_.n_imaginary = ws.n_imaginary;
    last_result_.is_linear = ws.is_linear;
    // Thermo not run on this path.
    last_result_.zpe = 0.0;
    last_result_.u_tot = 0.0;
    last_result_.h_tot = 0.0;
    last_result_.s_tot = 0.0;
    last_result_.s_elec = 0.0;
    last_result_.s_trans = 0.0;
    last_result_.s_rot = 0.0;
    last_result_.s_vib = 0.0;
    last_result_.g_tot = 0.0;
    last_result_.g_thermal = 0.0;
    return last_result_;
}

} // namespace vib
} // namespace cpu
} // namespace ATV
