/// @file cpu/task/task_spec.h
/// @brief Closed, versioned, typed sum of launcher-dispatchable CPU tasks.
///
/// TaskSpec is an in-memory/wire boundary, not a universal component IR. Each
/// task alternative owns its own phase enum, component allow-list, and budget:
/// MD cannot carry a minimizer phase and Minimize cannot carry an MD phase.
/// Historic wire layouts are rejected; there is no compatibility decoder.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "atomiverse/core/stage.h"

namespace ATV::cpu::task {

/// @brief Which concrete CPU driver pipeline a TaskSpec materializes.
enum class TaskKind : std::uint8_t {
    MD = 0,
    SinglePoint = 1,
    Minimize = 2,
};

inline const char* task_kind_name(TaskKind kind) noexcept {
    switch (kind) {
    case TaskKind::MD:
        return "MD";
    case TaskKind::SinglePoint:
        return "SinglePoint";
    case TaskKind::Minimize:
        return "Minimize";
    }
    return "unknown";
}

// ── Constructor parameter mirrors ─────────────────────────────────────────

struct PairLJCutParams {
    double epsilon;
    double sigma;
    double cutoff;
};
inline bool operator==(const PairLJCutParams& a, const PairLJCutParams& b) {
    return a.epsilon == b.epsilon && a.sigma == b.sigma && a.cutoff == b.cutoff;
}

struct NeighborListParams {
    double cutoff;
    double skin;
};
inline bool operator==(const NeighborListParams& a,
                       const NeighborListParams& b) {
    return a.cutoff == b.cutoff && a.skin == b.skin;
}

struct BondHarmonicParams {
    std::vector<double> K;
    std::vector<double> r0;
};
inline bool operator==(const BondHarmonicParams& a,
                       const BondHarmonicParams& b) {
    return a.K == b.K && a.r0 == b.r0;
}

struct AngleHarmonicParams {
    std::vector<double> K;
    std::vector<double> theta0;
};
inline bool operator==(const AngleHarmonicParams& a,
                       const AngleHarmonicParams& b) {
    return a.K == b.K && a.theta0 == b.theta0;
}

struct DihedralMultiHarmonicParams {
    std::vector<std::vector<double>> Cn;
};
inline bool operator==(const DihedralMultiHarmonicParams& a,
                       const DihedralMultiHarmonicParams& b) {
    return a.Cn == b.Cn;
}

struct VelocityVerletParams {
    double dt;
};
inline bool operator==(const VelocityVerletParams& a,
                       const VelocityVerletParams& b) {
    return a.dt == b.dt;
}

struct InitVelocityParams {
    double temperature;
    unsigned seed;
};
inline bool operator==(const InitVelocityParams& a,
                       const InitVelocityParams& b) {
    return a.temperature == b.temperature && a.seed == b.seed;
}

struct NoseHooverParams {
    double T_start;
    double tau_T;
    double dt;
};
inline bool operator==(const NoseHooverParams& a, const NoseHooverParams& b) {
    return a.T_start == b.T_start && a.tau_T == b.tau_T && a.dt == b.dt;
}

struct ThermoParams {
    int freq;
};
inline bool operator==(const ThermoParams& a, const ThermoParams& b) {
    return a.freq == b.freq;
}

/// @brief Wire mirror of `cpu::XyzRecorder(filename, freq)` — the MD
///        trajectory sink.
///
/// The recorder itself needs `ATV_BUILD_IO` (its writer goes through molrs),
/// but this mirror does not: the wire layer is always compiled, so a payload
/// naming the recorder stays decodable on a build without the writer and is
/// refused with a how-to when `TaskMaterializer` tries to construct it —
/// exactly the `KSpacePmeParams` arrangement.
///
/// This is the first payload carrying a **text** field. Variable-length
/// numeric fields already existed (`BondHarmonicParams` and friends write
/// length-prefixed `double` vectors), so the codec reuses that shape for
/// `filename`: a `u64` byte count followed by the raw bytes.
struct XyzRecorderParams {
    /// Output XYZ path. Truncated on the recorder's first write, appended to
    /// afterwards. This mirror is a plain aggregate and validates nothing;
    /// an empty path is rejected later, by `cpu::XyzRecorder`'s constructor.
    std::string filename;
    /// Write cadence [steps; >= 1] — one frame whenever `step % freq == 0`.
    /// Also checked by `cpu::XyzRecorder`, not here.
    int freq;
};
inline bool operator==(const XyzRecorderParams& a, const XyzRecorderParams& b) {
    return a.filename == b.filename && a.freq == b.freq;
}

/// @brief Mirrors `cpu::md::FixLangevin(T_start, T_stop, damp, dt, seed)`.
struct FixLangevinParams {
    double T_start;
    double T_stop;
    double damp;
    double dt;
    std::uint32_t seed;
};
inline bool operator==(const FixLangevinParams& a, const FixLangevinParams& b) {
    return a.T_start == b.T_start && a.T_stop == b.T_stop && a.damp == b.damp
           && a.dt == b.dt && a.seed == b.seed;
}

/// @brief Mirrors `cpu::md::FixCSVR(T_start, tau, dt, seed)`.
struct FixCsvrParams {
    double T_start;
    double tau;
    double dt;
    std::uint32_t seed;
};
inline bool operator==(const FixCsvrParams& a, const FixCsvrParams& b) {
    return a.T_start == b.T_start && a.tau == b.tau && a.dt == b.dt
           && a.seed == b.seed;
}

/// @brief Wire mirror of the ten-argument `cpu::md::NoseHooverNPT`
///        constructor, argument for argument.
///
/// `NoseHooverNPT` is the Martyna-Tobias-Klein (MTK) integrator for the NPT
/// ensemble — constant particle number N, pressure P and temperature T — so
/// it propagates the simulation cell as well as the atoms. Field order here
/// is exactly the constructor order (`T_target, tau_T, P_target, tau_P, dt,
/// drag, mtchain, mpchain, couple_axis, T_stop`), and `task_codec` writes
/// the fields in that same order — so this declaration doubles as the byte
/// layout. Changing either side is a `kWireVersion` bump.
///
/// Not mirrored — a launcher-dispatched NPT run gets the C++ defaults for
/// `NoseHooverNPT::set_couple_xy` (lockstep x/y cell edges) and
/// `set_surface_tension` (constant-surface-tension target). The ramp span
/// `set_run_span` is absent for a different reason: it is derived, not
/// chosen, and `TaskMaterializer` installs `[0, n_steps]` from the owning
/// `MDTask`.
struct NoseHooverNptParams {
    /// Thermostat setpoint, and the start value of the optional ramp [K].
    double T_target;
    /// Thermostat coupling time [fs] (LAMMPS `Tdamp`) — the timescale on
    /// which the thermostat pulls the kinetic temperature to `T_target`.
    double tau_T;
    /// Barostat setpoint [eV/Å³], LAMMPS sign (positive ⇔ compressive).
    /// 1 bar ≈ 6.2415e-7 eV/Å³.
    double P_target;
    /// Barostat coupling time [fs] (LAMMPS `Pdamp`).
    double tau_P;
    /// Timestep [fs].
    double dt;
    /// Empirical extra damping of the Nose-Hoover chain momenta (LAMMPS
    /// `drag`); dimensionless, 0 disables it.
    double drag;
    /// Length of the atom thermostat's Nose-Hoover chain — a chain of
    /// thermostat variables, each one thermostatting the previous
    /// [count; LAMMPS `mtchain`].
    int mtchain;
    /// Length of the barostat's own Nose-Hoover chain
    /// [count; LAMMPS `mpchain`].
    int mpchain;
    /// Per-axis barostat enable `{x, y, z}` (LAMMPS `p_flag`). A `false`
    /// axis keeps its cell edge rigid, i.e. that direction stays NVT.
    std::array<bool, 3> couple_axis;
    /// End temperature of the linear setpoint ramp [K]. A negative value
    /// folds to `T_target` in the constructor, i.e. no ramp.
    double T_stop;
};
inline bool operator==(const NoseHooverNptParams& a,
                       const NoseHooverNptParams& b) {
    return a.T_target == b.T_target && a.tau_T == b.tau_T
           && a.P_target == b.P_target && a.tau_P == b.tau_P && a.dt == b.dt
           && a.drag == b.drag && a.mtchain == b.mtchain
           && a.mpchain == b.mpchain && a.couple_axis == b.couple_axis
           && a.T_stop == b.T_stop;
}

/// @brief Wire mirror of the five leading `cpu::md::ParrinelloRahmanNPT`
///        constructor arguments.
///
/// `ParrinelloRahmanNPT` is the same MTK NPT scheme as `NoseHooverNPT` but
/// with a full cell matrix: all six independent strain-rate components
/// (xx, yy, zz, xy, xz, yz) evolve, so the cell can shear, not just breathe.
///
/// Watch the argument order — both setpoints first, then both coupling times
/// (`T_target, P_target, tau_T, tau_P, dt`) — where `NoseHooverNptParams`
/// pairs each setpoint with its own coupling time. The codec writes the
/// fields in this declaration order.
///
/// Not mirrored (C++ defaults apply on the launcher path): the optional
/// sixth constructor argument `W_cell` (barostat mass [eV·fs²]) and the
/// setters `set_drag` / `set_mtchain` / `set_mpchain`.
struct ParrinelloRahmanNptParams {
    /// Thermostat setpoint [K].
    double T_target;
    /// Target hydrostatic pressure [eV/Å³], LAMMPS sign (positive ⇔
    /// compressive). 1 bar ≈ 6.2415e-7 eV/Å³.
    double P_target;
    /// Thermostat coupling time [fs] (LAMMPS `Tdamp`).
    double tau_T;
    /// Barostat coupling time [fs] (LAMMPS `Pdamp`).
    double tau_P;
    /// Timestep [fs].
    double dt;
};
inline bool operator==(const ParrinelloRahmanNptParams& a,
                       const ParrinelloRahmanNptParams& b) {
    return a.T_target == b.T_target && a.P_target == b.P_target
           && a.tau_T == b.tau_T && a.tau_P == b.tau_P && a.dt == b.dt;
}

/// @brief Wire mirror of `cpu::md::Barostat(P_target, tau_P, dt,
///        bulk_modulus)` — the Berendsen barostat.
///
/// A barostat drives the cell volume towards a target pressure; the
/// Berendsen variant does it by rescaling the box and every atom position by
/// one scalar factor per step. Unlike the two NPT integrators above, it
/// rides *alongside* an integrator and a thermostat instead of replacing
/// them, so it is materialized through `MD::set_barostat`.
///
/// No field is defaulted here: every wire payload states `bulk_modulus`
/// explicitly. The C++ default of 1.0 is supplied by whoever builds this
/// struct (e.g. the Python binding), not by the wire layer.
struct BerendsenBarostatParams {
    /// Target pressure [eV/Å³], LAMMPS sign (positive ⇔ compressive).
    /// 1 bar ≈ 6.2415e-7 eV/Å³.
    double P_target;
    /// Barostat coupling time [fs] (LAMMPS `Pdamp`).
    double tau_P;
    /// Timestep [fs].
    double dt;
    /// Tunable modulus β [eV/Å³] (LAMMPS `modulus`). It divides the
    /// per-step rescale, so a larger β means a gentler box change and
    /// slower pressure relaxation. The C++ default is 1.0 — deliberately
    /// not LAMMPS's 10.0; the reasoning is in `cpu/dynamics/barostat.h`.
    double bulk_modulus;
};
inline bool operator==(const BerendsenBarostatParams& a,
                       const BerendsenBarostatParams& b) {
    return a.P_target == b.P_target && a.tau_P == b.tau_P && a.dt == b.dt
           && a.bulk_modulus == b.bulk_modulus;
}

/// @brief Wire mirror of `cpu::PairCoulCut(cutoff)` — truncated Coulomb.
///
/// LAMMPS `pair_style coul/cut`: the bare `q_i q_j / r_ij` sum over the
/// neighbor list, hard-cut at `cutoff`. Self-contained — unlike
/// `CoulLongParams` it takes no reciprocal-space partner, so it is the whole
/// electrostatic model of the task that names it.
///
/// Charges are Frame data, not a parameter: the style reads the per-atom
/// `charge` column [e] of the Frame it is launched on.
///
/// Not mirrored: `set_special_bonds` (the 1-2 / 1-3 / 1-4 weights — the
/// per-pair scale factors LAMMPS applies to atoms one, two or three bonds
/// apart). A launcher-dispatched run gets the C++ defaults, i.e. every
/// weight 1.
struct PairCoulCutParams {
    /// Real-space cutoff [Å]. The neighbor list must be built with a cutoff
    /// ≥ this value.
    double cutoff;
};
inline bool operator==(const PairCoulCutParams& a, const PairCoulCutParams& b) {
    return a.cutoff == b.cutoff;
}

/// @brief Wire mirror of `cpu::CoulLong(alpha, cutoff)` — the real-space
///        half of an Ewald/PME split.
///
/// An *Ewald split* rewrites the conditionally convergent periodic Coulomb
/// sum as two absolutely convergent ones: a screened short-range part that
/// decays fast enough to truncate at a cutoff, plus a smooth long-range
/// part that converges quickly in reciprocal (Fourier) space. The splitting
/// parameter α [1/Å] sets where the boundary between the two falls.
///
/// This mirror is the short-range half: `q_i q_j · erfc(α r_ij) / r_ij`
/// summed over the neighbor list (LAMMPS `pair_style coul/long`), where
/// `erfc` is the complementary error function — 1 at r = 0, decaying to 0
/// as α r grows. The complementary smooth part is summed by the matching
/// `KSpacePmeParams` entry of the same task. α is therefore not this
/// component's private knob: it must equal the reciprocal partner's α, and
/// the driver enforces the matched real/reciprocal pair at `build()`
/// (`Driver::validate_long_range_contract_`). A payload naming only this
/// half fails loudly there instead of running half an electrostatic model.
///
/// Charges are Frame data: the per-atom `charge` column [e] of the Frame.
///
/// Not mirrored: `set_special_bonds` (real-space 1-2 / 1-3 / 1-4 weights);
/// the C++ defaults apply on the launcher path.
struct CoulLongParams {
    /// Ewald splitting parameter α [1/Å]. Larger α screens the real-space
    /// term harder, moving work out of the neighbor-list sum into the
    /// reciprocal one. The LAMMPS rule of thumb is α ≈ 3.5 / cutoff, at
    /// which the screened pair term has fallen to erfc(3.5) ≈ 7e-7 of the
    /// bare one at the cutoff (`regressions/md/npt_pme_nacl/REFERENCE.md`).
    double alpha;
    /// Real-space cutoff [Å]. The neighbor list must be built with a cutoff
    /// ≥ this value.
    double cutoff;
};
inline bool operator==(const CoulLongParams& a, const CoulLongParams& b) {
    return a.alpha == b.alpha && a.cutoff == b.cutoff;
}

/// @brief Wire mirror of `cpu::KSpacePME(alpha, grid_spacing, order)` — the
///        reciprocal-space half of the same Ewald/PME split.
///
/// Smooth Particle-Mesh Ewald (Essmann et al., J. Chem. Phys. 103, 8577
/// (1995)) evaluates the long-range half on a grid instead of by an O(N²)
/// double sum: each point charge is smeared onto nearby mesh points with
/// B-spline weights (piecewise-polynomial interpolation weights that sum to
/// one), the mesh is transformed by a fast Fourier transform (FFT),
/// multiplied point-by-point by the *influence function* — the
/// reciprocal-space kernel `exp(−π²m²/α²)/m²` of the smooth Ewald term,
/// times a B-spline de-convolution factor, so it depends only on the cell,
/// α and the spline order — and transformed back. That is O(N log N)
/// instead of O(N²).
///
/// The solver needs `ATV_BUILD_KSPACE` (FFTW3), but this mirror does not:
/// the wire layer is always compiled, so a payload naming PME stays
/// decodable on a build without the solver and is rejected with a how-to
/// when `TaskMaterializer` tries to construct it.
///
/// No field is defaulted here: every wire payload states `grid_spacing` and
/// `order` explicitly. The C++ defaults (1.0 Å, order 6) are supplied by
/// whoever builds this struct (e.g. the Python binding), not by the wire
/// layer. `cpu::KSpacePME`'s constructor throws on `alpha <= 0`,
/// `grid_spacing <= 0` or `order < 3`, so an out-of-range payload fails at
/// materialization rather than producing a wrong reciprocal sum.
struct KSpacePmeParams {
    /// Ewald splitting parameter α [1/Å] — must equal the `CoulLongParams`
    /// α of the same task, or the driver refuses to build.
    double alpha;
    /// Target real-space grid spacing [Å]. The mesh itself is derived from
    /// the box and rounded to FFT-friendly sizes, so a smaller spacing buys
    /// a finer mesh and a more accurate — costlier — reciprocal sum.
    double grid_spacing;
    /// B-spline interpolation order [count; ≥ 3] — the width, in mesh
    /// points per dimension, of the stencil each charge is spread over.
    int order;
};
inline bool operator==(const KSpacePmeParams& a, const KSpacePmeParams& b) {
    return a.alpha == b.alpha && a.grid_spacing == b.grid_spacing
           && a.order == b.order;
}

/// @brief Mirrors the `cpu::semiempirical::AM1Evaluator` constructor exactly.
///
/// One params struct with two legal homes. As a `SinglePointTask` payload it
/// is the whole task; as an `MDComponentParams` alternative it makes the
/// semiempirical method an MD force source — Born-Oppenheimer molecular
/// dynamics — either through the categorical evaluator slot (entry binding
/// `std::nullopt`) or with an explicit `MDBinding`, since `task_materializer`
/// replays both paths through the same gradient-enabled builder. Both homes
/// decode from wire ID 9 with the byte layout written here, so they can never
/// disagree about what an AM1 payload contains.
struct AM1EvaluatorParams {
    /// @brief Total molecular charge [e].
    int charge = 0;
    /// @brief Spin multiplicity 2S+1 (dimensionless).
    int multiplicity = 1;
    /// @brief Cap on SCF iterations; overrunning it throws (dimensionless count).
    int max_iter = 200;
    /// @brief Convergence threshold on the largest absolute change of any
    ///        density-matrix element between two SCF iterations
    ///        (dimensionless).
    double density_tol = 1e-8;
    /// @brief Convergence threshold on |ΔE| between two SCF iterations [eV].
    ///        Meeting *either* tolerance ends the loop.
    double energy_tol = 1e-10;
};
inline bool operator==(const AM1EvaluatorParams& a,
                       const AM1EvaluatorParams& b) {
    return a.charge == b.charge && a.multiplicity == b.multiplicity
           && a.max_iter == b.max_iter && a.density_tol == b.density_tol
           && a.energy_tol == b.energy_tol;
}

/// @brief Wire mirror of the fifteen public members of `cpu::pw::PW` — the
///        plane-wave DFT method as an MD force source (plane-wave BOMD).
///
/// Like `AM1EvaluatorParams` it rides the categorical evaluator slot, so a
/// launcher payload can drive Born-Oppenheimer dynamics on a plane-wave
/// Hamiltonian — the electrons re-converged at every geometry the nuclei
/// reach. Budget two SCF solves per force-producing step, not one: the
/// evaluator runs the energy SCF and then the analytic force path, which
/// converges its own density (`cpu/planewave/pw_evaluator.cpp`).
///
/// The solver itself needs `ATV_BUILD_PW` (FFTW3 + the plane-wave sources),
/// but this mirror does not: the wire layer is always compiled, so a payload
/// naming PW stays decodable on a build without the solver and is refused
/// with a how-to when `TaskMaterializer` tries to construct it — exactly the
/// `KSpacePmeParams` / `XyzRecorderParams` arrangement.
///
/// Two members cross the wire as plain `int` rather than as their C++ enums
/// (`cpu::scf::XCType`, `cpu::pw::Smearing`): those enums live in headers
/// behind the build flag, and this struct is compiled unconditionally. The
/// materializer casts them back on the way in, so the enumerator VALUES are
/// part of the wire contract.
///
/// Field order is the declaration order below, which is what `task_codec`
/// writes and `task_materializer` replays; this declaration therefore doubles
/// as the byte layout. Defaults are `cpu::pw::PW`'s own, member for member.
struct PwEvaluatorParams {
    /// @brief Norm-conserving UPF (Unified Pseudopotential Format) file — a
    ///        pseudopotential replaces the nucleus plus core electrons by an
    ///        effective potential, which is what makes the plane-wave
    ///        expansion affordable. Mutually exclusive with `gth_name`;
    ///        exactly one must be set (checked by `cpu::pw::PW::build`, not
    ///        here — a plane-wave run has no default pseudopotential).
    std::string upf_path;
    /// @brief In-tree analytic Goedecker-Teter-Hutter parameterization, e.g.
    ///        "GTH-PBE-q4". Needs no external file. Mutually exclusive with
    ///        `upf_path`. One potential serves the whole cell, so a frame
    ///        mixing atomic numbers is refused at run time.
    std::string gth_name;
    /// @brief Wavefunction cutoff [Ry] — plane waves with kinetic energy
    ///        above it are dropped, so this sets basis size and cost.
    double ecut_ry = 30.0;
    /// @brief Monkhorst-Pack mesh size per dimension (dimensionless); the
    ///        Brillouin zone is sampled on `k_mesh`³ equally weighted points,
    ///        1 meaning the Gamma point alone.
    int k_mesh = 4;
    /// @brief SCF tolerance on the density-change L2 norm (dimensionless).
    double tol = 1e-6;
    /// @brief Maximum SCF iterations (dimensionless count).
    int max_iter = 50;
    /// @brief Density mixing parameter (QE `mixing_beta`, dimensionless).
    double mix_beta = 0.7;
    /// @brief Pulay mixing history window (dimensionless count).
    int pulay_depth = 5;
    /// @brief Exchange-correlation functional as the integer value of
    ///        `cpu::scf::XCType` (0 = LSDA, 1 = PBE = the C++ default, …).
    int xc = 1;
    /// @brief Collinear spin-polarized KS (nspin = 2); false = restricted.
    bool spin_polarized = false;
    /// @brief Spin multiplicity 2S+1, read only when `spin_polarized`.
    int multiplicity = 1;
    /// @brief Metallic smearing kind as the integer value of
    ///        `cpu::pw::Smearing` (0 = none = fixed occupations, 1 =
    ///        Fermi-Dirac, 2 = Methfessel-Paxton, 3 = Gaussian).
    int smearing = 0;
    /// @brief Smearing width [Ry] (QE `degauss`); must be > 0 when smearing
    ///        is on.
    double degauss = 0.02;
    /// @brief Methfessel-Paxton order N, read only by that smearing arm
    ///        (dimensionless).
    int mp_order = 1;
    /// @brief Extra empty bands beyond ceil(N_e/2) when smearing is on
    ///        (dimensionless count, per spin channel when collinear).
    int extra_bands = 4;
};
inline bool operator==(const PwEvaluatorParams& a, const PwEvaluatorParams& b) {
    return a.upf_path == b.upf_path && a.gth_name == b.gth_name
           && a.ecut_ry == b.ecut_ry && a.k_mesh == b.k_mesh && a.tol == b.tol
           && a.max_iter == b.max_iter && a.mix_beta == b.mix_beta
           && a.pulay_depth == b.pulay_depth && a.xc == b.xc
           && a.spin_polarized == b.spin_polarized
           && a.multiplicity == b.multiplicity && a.smearing == b.smearing
           && a.degauss == b.degauss && a.mp_order == b.mp_order
           && a.extra_bands == b.extra_bands;
}

/// @brief Mirrors the `cpu::minimizer::LBFGS` constructor exactly.
struct LBFGSParams {
    int memory = 5;
    double alpha0 = 0.01;
    int max_ls = 10;
};
inline bool operator==(const LBFGSParams& a, const LBFGSParams& b) {
    return a.memory == b.memory && a.alpha0 == b.alpha0 && a.max_ls == b.max_ls;
}

/// @brief Mirrors all five `cpu::ForceConverged` constructor tolerances.
struct ForceConvergedParams {
    double ftol = 0.05;
    double rms_ftol = 0.03;
    double max_dtol = 0.002;
    double rms_dtol = 0.001;
    double etol = 1e-5;
};
inline bool operator==(const ForceConvergedParams& a,
                       const ForceConvergedParams& b) {
    return a.ftol == b.ftol && a.rms_ftol == b.rms_ftol
           && a.max_dtol == b.max_dtol && a.rms_dtol == b.rms_dtol
           && a.etol == b.etol;
}

/// @brief Stable wire IDs, independent of variant ordering.
///
/// The value, not the position, is what crosses the wire, so this list is
/// append-only: an ID is never reused or renumbered. `task_codec` decodes it
/// with an exhaustive switch and throws on an unknown value.
enum class ComponentId : std::uint16_t {
    PairLJCut = 0,
    NeighborList = 1,
    BondHarmonic = 2,
    AngleHarmonic = 3,
    DihedralMultiHarmonic = 4,
    VelocityVerlet = 5,
    InitVelocity = 6,
    NoseHoover = 7,
    Thermo = 8,
    AM1Evaluator = 9,
    LBFGS = 10,
    FixLangevin = 11,
    FixCSVR = 12,
    NoseHooverNPT = 13,
    ParrinelloRahmanNPT = 14,
    BerendsenBarostat = 15,
    // Electrostatics. `PairCoulCut` is a complete model on its own;
    // `CoulLong` (real space) and `KSpacePME` (reciprocal space) are the two
    // halves of one Ewald split and are only ever legal together, at a
    // shared α — see `CoulLongParams`.
    PairCoulCut = 16, ///< Truncated Coulomb — `cpu::PairCoulCut`.
    CoulLong = 17,    ///< Ewald/PME real-space half — `cpu::CoulLong`.
    KSpacePME = 18,   ///< Ewald/PME reciprocal half — `cpu::KSpacePME`.
    /// XYZ trajectory sink — `cpu::XyzRecorder`. Decodable everywhere;
    /// materializable only where `ATV_BUILD_IO` compiled the writer in.
    XyzRecorder = 19,
    /// Plane-wave DFT force source — `cpu::pw::PWEvaluator`. Decodable
    /// everywhere; materializable only where `ATV_BUILD_PW` compiled the
    /// plane-wave solver in.
    PWEvaluator = 20,
};

inline const char* component_name(ComponentId id) noexcept {
    switch (id) {
    case ComponentId::PairLJCut:
        return "PairLJCut";
    case ComponentId::NeighborList:
        return "NeighborList";
    case ComponentId::BondHarmonic:
        return "BondHarmonic";
    case ComponentId::AngleHarmonic:
        return "AngleHarmonic";
    case ComponentId::DihedralMultiHarmonic:
        return "DihedralMultiHarmonic";
    case ComponentId::VelocityVerlet:
        return "VelocityVerlet";
    case ComponentId::InitVelocity:
        return "InitVelocity";
    case ComponentId::NoseHoover:
        return "NoseHoover";
    case ComponentId::Thermo:
        return "Thermo";
    case ComponentId::AM1Evaluator:
        return "AM1Evaluator";
    case ComponentId::LBFGS:
        return "LBFGS";
    case ComponentId::FixLangevin:
        return "FixLangevin";
    case ComponentId::FixCSVR:
        return "FixCSVR";
    case ComponentId::NoseHooverNPT:
        return "NoseHooverNPT";
    case ComponentId::ParrinelloRahmanNPT:
        return "ParrinelloRahmanNPT";
    case ComponentId::BerendsenBarostat:
        return "BerendsenBarostat";
    case ComponentId::PairCoulCut:
        return "PairCoulCut";
    case ComponentId::CoulLong:
        return "CoulLong";
    case ComponentId::KSpacePME:
        return "KSpacePME";
    case ComponentId::XyzRecorder:
        return "XyzRecorder";
    case ComponentId::PWEvaluator:
        return "PWEvaluator";
    }
    return "unknown";
}

using MDComponentParams =
    std::variant<PairLJCutParams, NeighborListParams, BondHarmonicParams,
                 AngleHarmonicParams, DihedralMultiHarmonicParams,
                 VelocityVerletParams, InitVelocityParams, NoseHooverParams,
                 ThermoParams, AM1EvaluatorParams, FixLangevinParams,
                 FixCsvrParams, NoseHooverNptParams, ParrinelloRahmanNptParams,
                 BerendsenBarostatParams, PairCoulCutParams, CoulLongParams,
                 KSpacePmeParams, XyzRecorderParams, PwEvaluatorParams>;

using MinimizeComponentParams =
    std::variant<PairLJCutParams, NeighborListParams, LBFGSParams>;

namespace detail {
template <typename>
inline constexpr bool unsupported_component_v = false;
} // namespace detail

inline ComponentId component_id(const MDComponentParams& params) noexcept {
    return std::visit(
        [](const auto& p) -> ComponentId {
            using P = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<P, PairLJCutParams>)
                return ComponentId::PairLJCut;
            else if constexpr (std::is_same_v<P, NeighborListParams>)
                return ComponentId::NeighborList;
            else if constexpr (std::is_same_v<P, BondHarmonicParams>)
                return ComponentId::BondHarmonic;
            else if constexpr (std::is_same_v<P, AngleHarmonicParams>)
                return ComponentId::AngleHarmonic;
            else if constexpr (std::is_same_v<P, DihedralMultiHarmonicParams>)
                return ComponentId::DihedralMultiHarmonic;
            else if constexpr (std::is_same_v<P, VelocityVerletParams>)
                return ComponentId::VelocityVerlet;
            else if constexpr (std::is_same_v<P, InitVelocityParams>)
                return ComponentId::InitVelocity;
            else if constexpr (std::is_same_v<P, NoseHooverParams>)
                return ComponentId::NoseHoover;
            else if constexpr (std::is_same_v<P, ThermoParams>)
                return ComponentId::Thermo;
            else if constexpr (std::is_same_v<P, AM1EvaluatorParams>)
                return ComponentId::AM1Evaluator;
            else if constexpr (std::is_same_v<P, FixLangevinParams>)
                return ComponentId::FixLangevin;
            else if constexpr (std::is_same_v<P, FixCsvrParams>)
                return ComponentId::FixCSVR;
            else if constexpr (std::is_same_v<P, NoseHooverNptParams>)
                return ComponentId::NoseHooverNPT;
            else if constexpr (std::is_same_v<P, ParrinelloRahmanNptParams>)
                return ComponentId::ParrinelloRahmanNPT;
            else if constexpr (std::is_same_v<P, BerendsenBarostatParams>)
                return ComponentId::BerendsenBarostat;
            else if constexpr (std::is_same_v<P, PairCoulCutParams>)
                return ComponentId::PairCoulCut;
            else if constexpr (std::is_same_v<P, CoulLongParams>)
                return ComponentId::CoulLong;
            else if constexpr (std::is_same_v<P, KSpacePmeParams>)
                return ComponentId::KSpacePME;
            else if constexpr (std::is_same_v<P, XyzRecorderParams>)
                return ComponentId::XyzRecorder;
            else if constexpr (std::is_same_v<P, PwEvaluatorParams>)
                return ComponentId::PWEvaluator;
            else
                static_assert(detail::unsupported_component_v<P>);
        },
        params);
}

inline ComponentId
component_id(const MinimizeComponentParams& params) noexcept {
    return std::visit(
        [](const auto& p) -> ComponentId {
            using P = std::decay_t<decltype(p)>;
            if constexpr (std::is_same_v<P, PairLJCutParams>)
                return ComponentId::PairLJCut;
            else if constexpr (std::is_same_v<P, NeighborListParams>)
                return ComponentId::NeighborList;
            else if constexpr (std::is_same_v<P, LBFGSParams>)
                return ComponentId::LBFGS;
            else
                static_assert(detail::unsupported_component_v<P>);
        },
        params);
}

struct MDBinding {
    ATV::md::Phase phase;
    int order;
};
inline bool operator==(const MDBinding& a, const MDBinding& b) {
    return a.phase == b.phase && a.order == b.order;
}

struct MinimizeBinding {
    ATV::minimizer::Phase phase;
    int order;
};
inline bool operator==(const MinimizeBinding& a, const MinimizeBinding& b) {
    return a.phase == b.phase && a.order == b.order;
}

struct MDComponentEntry {
    MDComponentParams params;
    std::optional<MDBinding> binding;
};
inline bool operator==(const MDComponentEntry& a, const MDComponentEntry& b) {
    return a.params == b.params && a.binding == b.binding;
}

struct MinimizeComponentEntry {
    MinimizeComponentParams params;
    std::optional<MinimizeBinding> binding;
};
inline bool operator==(const MinimizeComponentEntry& a,
                       const MinimizeComponentEntry& b) {
    return a.params == b.params && a.binding == b.binding;
}

struct MDTask {
    std::vector<MDComponentEntry> entries;
    std::int64_t n_steps;
};
inline bool operator==(const MDTask& a, const MDTask& b) {
    return a.entries == b.entries && a.n_steps == b.n_steps;
}

struct SinglePointTask {
    AM1EvaluatorParams evaluator;
};
inline bool operator==(const SinglePointTask& a, const SinglePointTask& b) {
    return a.evaluator == b.evaluator;
}

struct MinimizeTask {
    std::vector<MinimizeComponentEntry> entries;
    std::int64_t max_iter;
    ForceConvergedParams convergence;
};
inline bool operator==(const MinimizeTask& a, const MinimizeTask& b) {
    return a.entries == b.entries && a.max_iter == b.max_iter
           && a.convergence == b.convergence;
}

using TaskPayload = std::variant<MDTask, SinglePointTask, MinimizeTask>;

/// @brief Control-plane result; simulation state remains exclusively in Frame.
struct TaskResult {
    std::int64_t iterations = 0;
    bool converged = false;
    std::string stop_reason;
};
inline bool operator==(const TaskResult& a, const TaskResult& b) {
    return a.iterations == b.iterations && a.converged == b.converged
           && a.stop_reason == b.stop_reason;
}

/// @brief One launcher-dispatchable task: a `TaskKind` tag plus the typed
///        payload that materializes it.
class TaskSpec {
  public:
    /// @brief Version tag written at the head of every encoded TaskSpec;
    ///        `task_codec` rejects a payload whose tag differs.
    ///
    /// v3 was a hard cut to task-specific typed payloads; v4 is a hard cut to
    /// the grown `NoseHooverNptParams` byte layout (the chain-length,
    /// per-axis and ramp fields). Adding the `ParrinelloRahmanNPT` and
    /// `BerendsenBarostat` component IDs did not by itself force a bump — a
    /// new ID only ever appears in payloads that use it — but both land
    /// under v4, and so do the electrostatic IDs (`PairCoulCut`, `CoulLong`,
    /// `KSpacePME`) and `XyzRecorder` for the same reason: appended IDs with
    /// fresh payloads leave every existing byte layout untouched. No v2 or v3
    /// decoder exists: an older payload is an error, never a best-effort read.
    ///
    /// Admitting `AM1EvaluatorParams` into the MD payload likewise stays at
    /// v4: the widening is append-only in the *alternative set*, not in the
    /// bytes. ID 9 already existed and its five fields are written by the one
    /// shared `write`/`read_am1` pair, so every previously encodable MD
    /// payload encodes to the identical bytes and a v4 decoder written before
    /// this change reads them unchanged.
    static constexpr std::uint32_t kWireVersion = 4;

    explicit TaskSpec(MDTask task);
    explicit TaskSpec(SinglePointTask task);
    explicit TaskSpec(MinimizeTask task);

    TaskKind kind() const noexcept;
    const TaskPayload& payload() const noexcept { return payload_; }
    const MDTask& md() const { return std::get<MDTask>(payload_); }
    const SinglePointTask& single_point() const {
        return std::get<SinglePointTask>(payload_);
    }
    const MinimizeTask& minimize() const {
        return std::get<MinimizeTask>(payload_);
    }

  private:
    TaskPayload payload_;
};

} // namespace ATV::cpu::task
