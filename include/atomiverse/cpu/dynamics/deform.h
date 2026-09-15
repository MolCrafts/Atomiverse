/// @file cpu/dynamics/deform.h
/// @brief CPU affine box-deformation driver (single-axis uniaxial strain).
#pragma once

#include <string>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/integrator.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Uniaxial affine deformation -- aligned with LAMMPS `fix deform`.
///
/// `Deform` reproduces the per-axis strain control of LAMMPS
/// `lammps/src/fix_deform.cpp` for the two uniaxial-tension modes
/// `erate` and `trate`. At each `initial_integrate` call it sets the
/// chosen box edge to a target length and affinely remaps every owned
/// atom's coordinate along that axis (= LAMMPS `remap x`, the default).
/// Multi-rank: every rank applies the same deterministic scaling, so the
/// box stays consistent across ranks without any explicit collective.
///
/// **Reference length `L0` (load-bearing).** LAMMPS `fix deform`
/// snapshots the deformed box edge length at the time the fix is *issued*
/// -- `set[i].lo_start` / `set[i].hi_start` are captured in
/// `FixDeform::init()` from `domain->boxlo/boxhi`, NOT from the
/// simulation-initial box. `Deform` mirrors this: it snapshots `L0` for
/// the deformed axis on the *first* `launch()` (Deform activation) and
/// records that step as the time origin. Strain is therefore referenced
/// to the box at Deform activation, even if earlier MD stages (e.g. an
/// NPT compression) changed the box edge first.
///
/// ### LAMMPS `fix deform` correspondence (audit verdict table)
///
/// Audited against `lammps/src/fix_deform.cpp` (fetched verbatim from
/// the LAMMPS `stable` branch). Functions cited: `FixDeform::init()`
/// (the `lo_start`/`hi_start` snapshot and ERATE/TRATE `lo_stop`/
/// `hi_stop` setup) and `FixDeform::apply_strain()` (the per-step
/// `lo_target`/`hi_target` update).
///
/// | Aspect              | LAMMPS `fix deform`                              | `cpu::md::Deform`                         | Verdict  |
/// |---------------------|--------------------------------------------------|-------------------------------------------|----------|
/// | Strain mode         | `erate` = engineering, `trate` = true strain     | `Mode::erate` / `Mode::trate` enum        | aligned  |
/// | `erate` formula     | `L(t) = L0*(1 + R*t_elapsed)` (engineering)      | `L = L0*(1 + R*t_elapsed)`                | aligned  |
/// | `trate` formula     | `L(t) = L0*exp(R*t_elapsed)` (true / log strain) | `L = L0*exp(R*t_elapsed)`                 | aligned  |
/// | Reference length L0 | snapshotted at fix issue (`init()`)              | snapshotted at first `launch()`           | aligned  |
/// | Time origin         | elapsed from fix-issue step                      | elapsed from activation step              | aligned  |
/// | Sign convention     | `+R` = stretch, `-R` = compress                  | `+rate` = stretch, `-rate` = compress     | aligned  |
/// | Axis selection      | per-axis `x`/`y`/`z` keywords                    | `int axis` (0 = x, 1 = y, 2 = z)          | aligned  |
/// | Remap               | `remap x` (default) -- affine atom remap         | always affine remap (= `remap x`)         | aligned  |
/// | `remap none` / `v`  | optional non-affine remap modes                  | not supported (deferred, out of scope)    | gap      |
/// | `vel`/`final`/`delta` modes | absolute-velocity / target-length modes  | not supported (deferred, out of scope)    | gap      |
///
/// `erate` and `trate` coincide only to first order in `R*t`; `trate`
/// holds a constant *relative* rate `L_dot/L = R`.
///
/// Bound at `initial_integrate @ +200`, after VelocityVerlet (+0) and
/// NoseHoover (+100).
class Deform : public cpu::Integrator {
  public:
    /// @brief Per-axis strain mode -- mirrors LAMMPS `fix deform` keywords.
    enum class Mode {
        /// Engineering strain: target edge `L = L0*(1 + R*t_elapsed)`,
        /// referenced to the activation-time length `L0`. LAMMPS `erate`.
        erate,
        /// True (logarithmic) strain: target edge `L = L0*exp(R*t_elapsed)`,
        /// a constant relative rate `L_dot/L = R`. LAMMPS `trate`.
        trate,
    };

    /// @param mode  Strain mode: `Mode::erate` (engineering) or
    ///        `Mode::trate` (true strain). Named at the call site so
    ///        the two semantics cannot be silently confused.
    /// @param rate  Strain rate `R` along the chosen axis [1 / fs].
    ///        Positive = stretch, negative = compress. `0` disables.
    /// @param dt    MD step size [fs] -- converts elapsed steps to time.
    /// @param axis  Box axis to deform: 0 = x, 1 = y, 2 = z.
    Deform(Mode mode, double rate, double dt, int axis = 0);

    std::string name() const override { return "deform"; }
    /* role via Integrator base */
    /// @brief Write-set: affine remap of coordinates as the box strains.
    std::vector<DefaultBinding> default_bindings() const override {
        return {ATV::make_binding(ATV::md::Phase::initial_integrate, 200)};
    }

    /// @brief Reset the `L0` snapshot / activation origin.
    ///
    /// The deformed-axis reference length is re-snapshotted on the next
    /// `launch()`, so a rebuilt driver re-references strain to the box
    /// edge at the new activation point.
    /// @param ctx Build context (unused).
    void build(const BuildContext& ctx) override;

    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Retune the strain rate between `Driver::run()` segments.
    ///
    /// Setting `rate` to 0 disables deformation while keeping the
    /// component registered (no driver rebuild required). The `Mode`
    /// and `L0` snapshot are unchanged.
    /// @param rate New strain rate `R` [1/fs].
    void set_rate(double rate) noexcept { rate_ = rate; }

    /// @return Current strain rate `R` [1/fs].
    double rate() const noexcept { return rate_; }

    /// @return Current strain mode.
    Mode mode() const noexcept { return mode_; }

  private:
    Mode mode_;      ///< erate (engineering) or trate (true strain)
    double rate_;    ///< strain rate R [1/fs]
    double dt_;      ///< MD step size [fs]
    int axis_;       ///< deformed axis: 0=x, 1=y, 2=z
    double L0_{0.0}; ///< deformed-axis edge at activation [Angstrom]
    int origin_step_{
        -1}; ///< step index at activation; -1 = not yet snapshotted
};

} // namespace md
} // namespace cpu
} // namespace ATV
