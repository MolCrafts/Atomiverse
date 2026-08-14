/// @file core/system_scalars.h
/// @brief SystemScalars -- the cross-backend bundle of per-frame system
///        scalars (energy, kinetic, conserved, stress, kinetic stress).
///
/// Extracted from `cpu::Frame` so the same shape is reused verbatim by the
/// CUDA frame (rust-frame-lowering chain step -04). The struct
/// is a plain aggregate of POD fields with no behaviour -- it carries only
/// fundamental result quantities, never derived caches.
#pragma once

namespace ATV {

/// @brief Per-frame system scalar results, shared by every backend Frame.
///
/// All values are device-resident on the owning Frame's device. Units are
/// the MD-internal set (eV, Angstrom, fs, amu, K) -- see `core/units.h`.
struct SystemScalars {
    double energy =
        0; ///< Potential energy [eV] — sum of all evaluator contributions.

    /// @brief Pair (non-bonded) potential energy [eV].
    ///
    /// Accumulated by `cpu::PairLJCut` (and any future pair evaluator)
    /// alongside `energy`; cleared by `ZeroEnergy` each step. Mirrors
    /// LAMMPS `thermo_style epair` (`thermo.cpp`, keyword `epair`).
    double energy_pair = 0;

    /// @brief Bond (2-body bonded) potential energy [eV].
    ///
    /// Accumulated by `cpu::BondHarmonic` (and any future bond evaluator)
    /// alongside `energy`. Mirrors LAMMPS `thermo_style ebond`.
    double energy_bond = 0;

    /// @brief Angle (3-body bonded) potential energy [eV].
    ///
    /// Accumulated by `cpu::AngleHarmonic` (and any future angle evaluator)
    /// alongside `energy`. Mirrors LAMMPS `thermo_style eangle`.
    double energy_angle = 0;

    /// @brief Dihedral (4-body bonded) potential energy [eV].
    ///
    /// Accumulated by `cpu::DihedralMultiHarmonic` (and any future dihedral
    /// evaluator) alongside `energy`. Mirrors LAMMPS `thermo_style edihed`.
    ///
    /// Invariant: `energy ≈ energy_pair + energy_bond + energy_angle +
    /// energy_dihedral + energy_improper` (plus any future per-term fields)
    /// to round-off precision, when every contributing evaluator writes both
    /// `energy` and its term-specific field on the same `+=` line.
    double energy_dihedral = 0;

    /// @brief Improper (out-of-plane / harmonic) potential energy [eV].
    ///
    /// Accumulated by `cpu::ImproperHarmonic` alongside `energy`. Mirrors
    /// LAMMPS `thermo_style eimp`.
    double energy_improper = 0;

    double kinetic = 0; ///< Kinetic energy [eV].

    /// @brief Integrator's conserved quantity [eV].
    ///
    /// For an extended-Lagrangian integrator this is the conserved
    /// *extended* Hamiltonian — e.g. `NoseHooverNPT` writes the full
    /// MTK Hamiltonian (atom KE+PE + barostat KE + Nosé-Hoover chain
    /// KE/PE + p·V). For a plain symplectic integrator it is just
    /// `KE+PE`. The integrator writes it once per step; it stays `0`
    /// when no integrator populates it. A flat `conserved` trace is the
    /// canonical correctness check that an integrator is symplectic;
    /// `Thermo` logs it as a column.
    double conserved = 0;

    /// @brief Pair / bonded virial tensor Σ r·F [eV], Voigt order xx yy zz xy xz yz.
    ///
    /// Written by pair potentials (LJ, Coulomb, ...) and bonded
    /// evaluators at `post_force`. **Does not** include the kinetic
    /// contribution Σ m·v·v -- that lives in `stress_kinetic` so each
    /// piece is consistent with its own time slice (virial is fixed
    /// once forces are computed, kinetic varies as velocities are
    /// updated through the Trotter step). The full pressure-tensor
    /// sum is `stress[i] + stress_kinetic[i]`, and dividing the trace
    /// by `3·V` gives the internal pressure under the LAMMPS sign
    /// convention (positive ⇔ compressive).
    double stress[6] = {};

    /// @brief Kinetic stress tensor Σ m·v·v [eV], Voigt order xx yy zz xy xz yz.
    ///
    /// Written by `Virial` at `post_force` using the velocities
    /// at that point. Time-dependent: integrators like
    /// `NoseHooverNPT` that update velocities between `post_force`
    /// and pressure evaluation recompute the kinetic stress fresh
    /// from the post-update velocities rather than reusing this
    /// stored value, so this field is *informational* for any
    /// downstream observer/recorder that wants the post-force
    /// pressure snapshot.
    double stress_kinetic[6] = {};
};

} // namespace ATV
