/// @file cpu/dynamics/piston_electrode.h
/// @brief Rigid electrode piston for constant normal pressure \(P_z\) (E6).
///
/// MetalWalls-class: treat one electrode type as a rigid body free only along
/// \(z\), with external force \(\pm P_{\mathrm{target}} A\). Component on
/// `cpu::md::MD` — not a Driver category, not inside `KSpace*`.
#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "atomiverse/core/binding.h"
#include "atomiverse/core/precision.h"
#include "atomiverse/core/stage.h"
#include "atomiverse/cpu/component/component.h"

namespace ATV {
namespace cpu {
namespace md {

/// @brief Rigid \(z\)-piston electrode at target pressure \(P_z\).
///
/// ## MetalWalls correspondence
///
/// MetalWalls JOSS (Marin-Laflèche et al., 2020, 10.21105/joss.02373): constant
/// \(P_z\) by treating electrodes as **rigid pistons**. Map slice **E6** —
/// electrode mechanics; *rigid piston ≠ generic barostat*.
///
/// | Aspect | MetalWalls / literature | `PistonElectrode` | Verdict |
/// |---|---|---|---|
/// | Selection | electrode group | single atom `type` | ALIGNED (textbook) |
/// | Rigid body | wall translates as one | mass-weighted \(F_z\) + equal \(v_z\) | ALIGNED |
/// | External force | \(P_z A\) on piston | \(\pm P_{\mathrm{target}} A\), sign by `Side` | ALIGNED |
/// | Free axes | \(z\) only | \(F_x=F_y=0\), restore \(xy\) offsets | ALIGNED |
/// | Charge solve | orthogonal (CG when movable) | does **not** solve \(Q\) | out of scope |
///
/// Instantaneous diagnostic pressure (positive when fluid compresses the wall
/// against the external load):
///
/// - `Side::Upper`: \(P_z = F_z^{\mathrm{atoms}} / A\)
/// - `Side::Lower`: \(P_z = -F_z^{\mathrm{atoms}} / A\)
///
/// with \(A = L_x L_y\). Equilibrium: \(\langle P_z\rangle \to P_{\mathrm{target}}\).
///
/// **Do not** register `FixFreeze` on the same type — this component owns the
/// piston dynamics for that type. Freeze the *other* wall separately.
///
/// Parameters are constructor arguments and public members — no Config bag.
///
/// @code
/// cpu::md::PistonElectrode piston(/*type=*/3, /*P [eV/Å³]=*/1e-4,
///                                 cpu::md::PistonElectrode::Side::Upper);
/// piston.tau_damp = 50.0; // optional COM friction [fs]
/// sim.set_observer(std::move(piston));
/// @endcode
class PistonElectrode : public cpu::Component {
  public:
    /// @brief Which plate: sets the sign of \(F_{\mathrm{ext}} = \pm P A\).
    enum class Side {
        /// Wall below the fluid; external force \(+P A\) (pushes \(+z\)).
        Lower = 0,
        /// Wall above the fluid; external force \(-P A\) (pushes \(-z\)).
        Upper = 1,
    };

    /// @param type_id Atom type of this electrode piston (matches `kAtom_type`).
    /// @param P_target Target normal pressure [eV/Å³]. 1 bar ≈ 6.2415e-7 eV/Å³.
    /// @param side Upper or lower wall (external-force sign).
    PistonElectrode(int type_id, double P_target, Side side);

    /// @return Frozen/piston atom type id.
    int type_id() const noexcept { return type_id_; }
    /// @return Wall side (external-force convention).
    Side side() const noexcept { return side_; }

    // ── Public parameters (mutable after construction) ───────────────
    double P_target = 0.0; ///< Target \(P_z\) [eV/Å³].
    /// Optional COM friction time constant [fs]. If `>0`, adds
    /// \(F_{\mathrm{damp}} = -M v_{\mathrm{com},z}/\tau\) into the net force
    /// (textbook equilibration aid; default off).
    double tau_damp = 0.0;

    std::string name() const override { return "piston_electrode"; }

    std::vector<DefaultBinding> default_bindings() const override {
        // After physical forces / biases; before FixFreeze (+1000) so freeze
        // on a *different* type still works. Own type must not be frozen.
        return {ATV::make_binding(ATV::md::Phase::post_force, +900),
                ATV::make_binding(ATV::md::Phase::post_integrate, 0),
                ATV::make_binding(ATV::md::Phase::final_integrate, +200)};
    }

    void build(const cpu::BuildContext& ctx) override;
    void launch(const cpu::LaunchContext& ctx) override;

    /// @brief Instantaneous piston \(P_z\) [eV/Å³] from last `post_force`.
    double last_P_z() const noexcept { return last_P_z_; }
    /// @brief \(\sum F_z\) on piston atoms before redistribution [eV/Å].
    double last_F_z_atoms() const noexcept { return last_F_z_atoms_; }
    /// @brief Net force including \(F_{\mathrm{ext}}\) (+ optional damp) [eV/Å].
    double last_F_net() const noexcept { return last_F_net_; }
    /// @brief Area \(A=L_x L_y\) used last step [Å²].
    double last_area() const noexcept { return last_area_; }
    /// @brief COM \(z\) of piston atoms last step [Å].
    double last_z_com() const noexcept { return last_z_com_; }
    /// @brief Total piston mass last step [amu].
    double last_mass() const noexcept { return last_mass_; }

  private:
    void resolve_rows_(const cpu::Frame& f, int n_owned);
    void snapshot_xy_offsets_(const cpu::Frame& f);
    void apply_forces_(cpu::Frame& f);
    void enforce_rigid_(cpu::Frame& f, bool update_positions);

    int type_id_;
    Side side_;

    std::vector<int> rows_;
    /// Per-tag \((x,y)\) offsets from COM at build [Å] (rigid in plane).
    std::unordered_map<TagInt, std::array<StorageReal, 2>> xy_off_by_tag_;
    /// Per-tag \(z\) offset from COM at build [Å].
    std::unordered_map<TagInt, StorageReal> z_off_by_tag_;

    double last_P_z_ = 0.0;
    double last_F_z_atoms_ = 0.0;
    double last_F_net_ = 0.0;
    double last_area_ = 0.0;
    double last_z_com_ = 0.0;
    double last_mass_ = 0.0;
    double last_vz_com_ = 0.0;
};

} // namespace md
} // namespace cpu
} // namespace ATV
