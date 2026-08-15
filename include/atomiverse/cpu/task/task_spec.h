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

/// @brief Wire mirror of `cpu::PairMorse(D, alpha, r0, cutoff)`.
///
/// Morse \(V(r)=D\,(1-e^{-\alpha(r-r_0)})^2\), shifted so \(V(r_c)=0\).
///
/// The Morse form is the textbook model of a chemical bond: a minimum at
/// \(r_0\), harmonic for small displacements, and — unlike a spring —
/// flattening out to the finite *dissociation energy* \(D\) as \(r\) grows, so
/// the bond can break. \(\alpha\) sets how fast that flattening happens; a
/// large \(\alpha\) is a narrow, stiff well.
///
/// The style walks all pairs O(N²) and withdraws its `NeighborConsumer`
/// (`cpu/potentials/pair_morse.h`), so a task naming it needs no
/// `NeighborListParams` entry — and must not be pointed at a bulk system.
/// It is the tree's "a new pair style costs two files" demonstration; the
/// mirror carries that caveat rather than hiding it.
struct PairMorseParams {
    /// @brief Well depth D [eV].
    double D = 0.0;
    /// @brief Width parameter α [Å⁻¹].
    double alpha = 0.0;
    /// @brief Equilibrium separation r₀ [Å].
    double r0 = 0.0;
    /// @brief Pair cutoff [Å] (> 0); the potential is shifted to zero there.
    double cutoff = 0.0;
};
inline bool operator==(const PairMorseParams& a, const PairMorseParams& b) {
    return a.D == b.D && a.alpha == b.alpha && a.r0 == b.r0
           && a.cutoff == b.cutoff;
}

/// @brief Wire mirror of `cpu::PairBuck(A, rho, C, cutoff)`.
///
/// Buckingham \(U(r)=A e^{-r/\rho}-C/r^{6}\), shifted at the cutoff (LAMMPS
/// `pair_style buck` with `pair_modify shift yes`).
///
/// The two terms are the two physical effects of a closed-shell pair. The
/// *Born-Mayer* exponential is Pauli repulsion — the energy cost of forcing two
/// filled electron clouds to overlap, which falls off over the softness length
/// ρ. The \(-C/r^{6}\) term is *London dispersion*, the attraction between the
/// instantaneous dipoles two polarizable atoms induce in each other.
///
/// Same arity as `PairMorseParams`, so the two are told apart on the wire by
/// their IDs alone. O(N²) all-pairs, like every style of this group.
struct PairBuckParams {
    /// @brief Born-Mayer prefactor A [eV] (≥ 0).
    double A = 0.0;
    /// @brief Softness length ρ [Å] (> 0).
    double rho = 0.0;
    /// @brief Dispersion coefficient C [eV·Å⁶] (≥ 0).
    double C = 0.0;
    /// @brief Pair cutoff [Å] (> 0).
    double cutoff = 0.0;
};
inline bool operator==(const PairBuckParams& a, const PairBuckParams& b) {
    return a.A == b.A && a.rho == b.rho && a.C == b.C && a.cutoff == b.cutoff;
}

/// @brief Wire mirror of the MULTI-TYPE `cpu::PairBorn(coeffs, cutoff)`
///        constructor — Born-Mayer-Huggins / Fumi-Tosi.
///
/// \(U(r)=A e^{(\sigma-r)/\rho}-C/r^{6}+D/r^{8}\), shifted at the cutoff.
///
/// The C++ constructor takes a `std::vector<cpu::BornTypeCoeff>`, i.e. a
/// vector of STRUCTS, which the codec's variable-length primitive cannot
/// express: its only such primitive is a length-prefixed run of one scalar
/// type. So the table crosses as PARALLEL VECTORS — the `BondHarmonicParams`
/// shape, one row per index across all seven vectors — and the materializer
/// rebuilds the coefficient structs from them.
///
/// The SINGLE-TYPE C++ constructor `PairBorn(A, rho, sigma, C, D, cutoff)` is
/// deliberately not mirrored separately: it is expressible here as one-row
/// vectors keyed on the frame's own type value, which is what the
/// materializer parity test uses. That equivalence is ENERGETIC, not
/// algorithmic — a one-row mirror still materializes the multi-type arm of
/// `cpu::PairBorn`, which walks all pairs O(N²) instead of taking the
/// single-type arm's `ctx.md.pair_list` fast path.
///
/// Type keys are frame `kAtom_type` values as-is. In this tree that column
/// carries the ATOMIC NUMBER (`cpu::from_molrs` fills it from molrs
/// `atomic_number`), so a Na-Cl row is `{11, 17}` and an argon dimer is
/// `{18, 18}` — not a 0-based dense index. Every interacting pair must be
/// listed; unlisted pairs do not interact (Fumi-Tosi parameters are not
/// Lorentz-Berthelot, so there is no silent mixing).
struct PairBornParams {
    /// @brief First type of each row (frame `kAtom_type`, i.e. Z).
    std::vector<int> type_i;
    /// @brief Second type of each row; the table is symmetric in (i, j).
    std::vector<int> type_j;
    /// @brief Born-Mayer prefactor A per row [eV].
    std::vector<double> A;
    /// @brief Softness length ρ per row [Å] (> 0).
    std::vector<double> rho;
    /// @brief Ionic-pair radius parameter σ per row [Å].
    std::vector<double> sigma;
    /// @brief Dispersion coefficient C per row [eV·Å⁶].
    std::vector<double> C;
    /// @brief \(r^{-8}\) coefficient D per row [eV·Å⁸].
    std::vector<double> D;
    /// @brief Interaction cutoff [Å] (> 0), shared by every row.
    double cutoff = 0.0;
};
inline bool operator==(const PairBornParams& a, const PairBornParams& b) {
    return a.type_i == b.type_i && a.type_j == b.type_j && a.A == b.A
           && a.rho == b.rho && a.sigma == b.sigma && a.C == b.C && a.D == b.D
           && a.cutoff == b.cutoff;
}

/// @brief Wire mirror of `cpu::PairYukawa(A, kappa, cutoff)`.
///
/// Screened Coulomb \(U(r)=A e^{-\kappa r}/r\), shifted at the cutoff.
///
/// *Screening* is what a cloud of mobile opposite charges (counter-ions around
/// a colloid, electrons in a plasma) does to a bare charge: it neutralizes it
/// beyond a distance \(1/\kappa\), turning the \(1/r\) Coulomb tail into an
/// exponentially cut one. \(\kappa = 0\) recovers unscreened \(A/r\).
///
/// Charges are NOT read: the amplitude is the whole interaction strength,
/// which is what separates this style from `PairCoulCutParams`.
struct PairYukawaParams {
    /// @brief Amplitude A [eV·Å]; positive is repulsive.
    double A = 0.0;
    /// @brief Screening wavenumber κ [Å⁻¹] (≥ 0).
    double kappa = 0.0;
    /// @brief Pair cutoff [Å] (> 0).
    double cutoff = 0.0;
};
inline bool operator==(const PairYukawaParams& a, const PairYukawaParams& b) {
    return a.A == b.A && a.kappa == b.kappa && a.cutoff == b.cutoff;
}

/// @brief Wire mirror of `cpu::PairWCA(epsilon, sigma)`.
///
/// Weeks-Chandler-Andersen: Lennard-Jones 12-6 cut and shifted at
/// \(r_c = 2^{1/6}\sigma\), so \(U(r_c)=F(r_c)=0\) and the style is purely
/// repulsive.
///
/// There is NO cutoff field, deliberately: the cutoff is derived from σ, not
/// chosen, which is the whole point of the style. This is therefore the only
/// two-double pair payload on the wire.
struct PairWcaParams {
    /// @brief Well depth of the parent LJ, ε [eV].
    double epsilon = 0.0;
    /// @brief Length scale σ [Å].
    double sigma = 0.0;
};
inline bool operator==(const PairWcaParams& a, const PairWcaParams& b) {
    return a.epsilon == b.epsilon && a.sigma == b.sigma;
}

/// @brief Wire mirror of the SINGLE-TYPE
///        `cpu::PairDpd(a, gamma, temperature, cutoff, dt, seed)`.
///
/// Dissipative particle dynamics (Groot-Warren 1997): a soft conservative
/// force \(F^C = a\,w(r)\hat r\), \(w = 1 - r/r_c\), plus a pairwise
/// dissipative/random thermostat with \(\sigma^2 = 2\gamma k_B T\).
///
/// The seed is a `std::uint64_t` — `cpu::PairDpd`'s own type — where
/// `FixLangevinParams` and `FixCsvrParams` carry `std::uint32_t` ones. It is
/// written through the `u64` codec primitive, so the layout is eight bytes
/// and not four.
///
/// Not mirrored: the MULTI-COMPONENT constructor taking a square
/// `a_matrix[t_i][t_j]`. A vector-of-vectors has no codec primitive (the same
/// reason `PairBornParams` is parallel vectors), and unlike the Born table it
/// is a DENSE square matrix rather than a list of keyed rows, so it is not
/// expressible as one flat run either. Its indices are frame `kAtom_type`
/// values as-is — atomic numbers in this tree — so a faithful mirror would
/// also have to carry max(Z) + 1 rows of mostly-unused amplitudes.
///
/// Two behaviours a caller has to know, both documented on the C++ class:
/// `kStress` carries the CONSERVATIVE virial only (the thermostat terms have
/// zero mean and \(F^R\propto1/\sqrt{dt}\)), and the random force is redrawn
/// every loop step from the `{seed, step, i, j}` stream, `i`/`j` being atom
/// INDICES — so a run is reproducible for a given seed, step budget and atom
/// ordering, and a single point is not reproducible across different steps
/// unless γ = 0.
struct PairDpdParams {
    /// @brief Soft repulsion amplitude a [eV/Å].
    double a = 0.0;
    /// @brief Dissipative friction γ [eV·fs/Å²]; 0 switches the dissipative
    ///        AND random terms off (σ = √(2γk_BT)).
    double gamma = 0.0;
    /// @brief Thermostat temperature T [K] (> 0).
    double temperature = 0.0;
    /// @brief Pair cutoff r_c [Å] (> 0).
    double cutoff = 0.0;
    /// @brief Timestep [fs] (> 0) — the random force scales as 1/√dt, so it
    ///        must be the timestep the integrator actually uses.
    double dt = 0.0;
    /// @brief Random-stream seed; 64-bit, unlike the thermostat mirrors'.
    std::uint64_t seed = 1;
};
inline bool operator==(const PairDpdParams& a, const PairDpdParams& b) {
    return a.a == b.a && a.gamma == b.gamma && a.temperature == b.temperature
           && a.cutoff == b.cutoff && a.dt == b.dt && a.seed == b.seed;
}

/// @brief Wire mirror of `cpu::BondFene(K, R0)` — the Kremer-Grest bond.
///
/// \(U = -\tfrac12 K R_0^2 \ln(1-(r/R_0)^2)\), *finitely extensible*: the
/// energy diverges as the bond approaches the maximum extension \(R_0\), so
/// unlike a harmonic spring the bond cannot be stretched arbitrarily far. The
/// component THROWS at \(r \ge R_0\) rather than reporting an infinite energy,
/// so a payload whose R₀ is shorter than the frame's bonds fails loudly at the
/// first launch.
///
/// The standard coarse-grained polymer bond, and the reason it is standard:
/// paired with a repulsive WCA pair style, the hard ceiling on bond length
/// plus the hard floor on non-bonded approach makes it geometrically
/// impossible for two chains to pass through each other.
///
/// Same two-parallel-vector shape as `BondHarmonicParams`, indexed by bond
/// type — hence its own ID: a swapped arm would read a maximum extension as
/// an equilibrium length.
struct BondFeneParams {
    /// @brief Spring constants per bond type [eV/Å²].
    std::vector<double> K;
    /// @brief Maximum extensions per bond type [Å].
    std::vector<double> R0;
};
inline bool operator==(const BondFeneParams& a, const BondFeneParams& b) {
    return a.K == b.K && a.R0 == b.R0;
}

/// @brief Wire mirror of `cpu::DihedralCharmm(K, n, delta)`.
///
/// Periodic Fourier torsion \(V = K\,(1 + \cos(n\varphi - \delta))\)
/// (LAMMPS `dihedral_style charmm` without the 1-4 pair weights, which stay
/// on the pair styles via `special_bonds`).
///
/// The FIRST payload on the wire carrying a variable-length INTEGER vector:
/// the multiplicity `n` is an `int` in the C++ constructor and is checked
/// against `n >= 1` there, so pushing it through the f64 `vec` primitive
/// would roundtrip the values while changing the byte layout and the type.
/// The codec gives it its own length-prefixed i32 run.
///
/// Not mirrored: `DihedralCharmm::set_compute_stress(bool)`. It is a
/// post-construction performance selector (skip the per-torsion virial when
/// no barostat reads it), not a constructor argument, so a launcher- or
/// Python-dispatched run gets the C++ default `true` — the
/// documented-omission arrangement `PairCoulCutParams` uses for
/// `set_special_bonds`.
struct DihedralCharmmParams {
    /// @brief Force constants per dihedral type [eV].
    std::vector<double> K;
    /// @brief Multiplicities per type — the number of energy minima the
    ///        torsion has per full 360° turn (dimensionless, ≥ 1). n = 3 is
    ///        the threefold barrier of an sp³-sp³ bond.
    std::vector<int> n;
    /// @brief Phase shifts per type δ [rad] — where those minima sit.
    std::vector<double> delta;
};
inline bool operator==(const DihedralCharmmParams& a,
                       const DihedralCharmmParams& b) {
    return a.K == b.K && a.n == b.n && a.delta == b.delta;
}

/// @brief Wire mirror of `cpu::ImproperHarmonic(K, chi0)`.
///
/// \(V = K(\chi - \chi_0)^2\) over the improper 4-tuples of the frame
/// (LAMMPS `improper_style harmonic`), χ being the angle between the two
/// planes of the tuple.
///
/// An *improper* is a 4-tuple that is not a chain, unlike a proper dihedral:
/// it measures how far a centre has buckled out of the plane of its three
/// neighbours, so \(\chi_0 = 0\) with a stiff K is the restraint that keeps an
/// sp² centre or an aromatic ring flat. It travels in the frame's own
/// `impropers` block, separate from `dihedrals`.
///
/// Two parallel vectors indexed by improper type — the same shape as
/// `BondFeneParams`, hence its own ID.
struct ImproperHarmonicParams {
    /// @brief Force constants per improper type [eV/rad²].
    std::vector<double> K;
    /// @brief Equilibrium improper angles per type χ₀ [rad].
    std::vector<double> chi0;
};
inline bool operator==(const ImproperHarmonicParams& a,
                       const ImproperHarmonicParams& b) {
    return a.K == b.K && a.chi0 == b.chi0;
}

/// @brief Mirrors the `cpu::minimizer::LBFGS` constructor exactly.
///
/// L-BFGS (limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton
/// optimizer: instead of building the full second-derivative matrix it
/// reconstructs its action on the gradient from the last `memory`
/// displacement/gradient-change pairs, then backtracks along that direction
/// until the energy has dropped enough (the Armijo condition).
///
/// Convergence tolerances are absent for the same reason as in `CGParams`:
/// they belong to the criterion, not the algorithm.
struct LBFGSParams {
    /// @brief Number of stored correction pairs [count]. More memory buys a
    ///        better curvature estimate at a linear cost in storage and work.
    int memory = 5;
    /// @brief Initial step size α₀ [Å²/eV] — the trial displacement per unit
    ///        force, before the line search backtracks.
    double alpha0 = 0.01;
    /// @brief Cap on backtracking attempts within one step [count].
    int max_ls = 10;
};
inline bool operator==(const LBFGSParams& a, const LBFGSParams& b) {
    return a.memory == b.memory && a.alpha0 == b.alpha0 && a.max_ls == b.max_ls;
}

/// @brief Wire mirror of the one `cpu::minimizer::CG` constructor argument.
///
/// `CG` is the Polak-Ribière⁺ conjugate-gradient Optimizer: the search
/// direction is steepest descent plus a β-weighted memory of the previous
/// direction, with β clamped at zero so an ill-conditioned region restarts
/// into plain steepest descent.
///
/// Convergence tolerances are deliberately absent — they belong to the
/// criterion (`ForceConvergedParams` and friends), which is exactly what lets
/// one stop policy judge `LBFGS`, `CG` and `FIRE` alike.
///
/// Not mirrored: `CG::line_search(mode, c1, c2, max_step, min_alpha)`. It is a
/// post-construction selector rather than a constructor argument, so a
/// launcher-dispatched CG run gets the C++ defaults (StrongWolfe, c1 = 1e-4,
/// c2 = 0.1, max_step = 0.2 Å, min_alpha = 1e-8) — the documented-omission
/// arrangement `PairCoulCutParams` already uses for `set_special_bonds`.
struct CGParams {
    /// @brief Step size α₀ [Å²/eV] — the trial step along the conjugate
    ///        direction, before the line search refines it.
    double alpha0 = 0.01;
};
inline bool operator==(const CGParams& a, const CGParams& b) {
    return a.alpha0 == b.alpha0;
}

/// @brief Wire mirror of the eight public members of `cpu::minimizer::FIRE`.
///
/// The Fast Inertial Relaxation Engine (Bitzek et al., Phys. Rev. Lett. 97,
/// 170201 (2006)) is damped dynamics rather than a line-search method: it
/// integrates a fictitious unit-mass trajectory, mixes the velocity towards
/// the force direction by α, grows the pseudo-timestep while the power `F·v`
/// stays positive, and zeroes the velocity outright when it turns negative.
///
/// `FIRE` has NO constructor arguments — every knob is a public member with an
/// ASE-aligned FIRE-1.0 default — so the field order here is the member
/// DECLARATION order of `cpu/minimize/fire.h`, which is also what `task_codec`
/// writes and what `task_materializer` assigns.
struct FIREParams {
    /// @brief Initial pseudo-timestep [fs].
    double dt0 = 0.1;
    /// @brief Ceiling on the pseudo-timestep [fs].
    double dt_max = 1.0;
    /// @brief Accelerating steps required before Δt grows (Bitzek `N_min`)
    ///        [count].
    int n_min = 5;
    /// @brief Δt growth factor while accelerating (dimensionless).
    double f_inc = 1.1;
    /// @brief Δt shrink factor when the power `F·v` turns non-positive
    ///        (dimensionless).
    double f_dec = 0.5;
    /// @brief Initial velocity-mixing parameter α_start (dimensionless).
    double alpha0 = 0.1;
    /// @brief Decay factor applied to α while accelerating (dimensionless).
    double f_alpha = 0.99;
    /// @brief Cap on the per-atom displacement of one step [Å].
    double max_step = 0.2;
};
inline bool operator==(const FIREParams& a, const FIREParams& b) {
    return a.dt0 == b.dt0 && a.dt_max == b.dt_max && a.n_min == b.n_min
           && a.f_inc == b.f_inc && a.f_dec == b.f_dec && a.alpha0 == b.alpha0
           && a.f_alpha == b.f_alpha && a.max_step == b.max_step;
}

/// @brief Wire mirror of the four public members of
///        `cpu::minimizer::IsotropicCellRelax`.
///
/// A *variable-cell* optimizer: it reads the Voigt virial from `kStress`,
/// forms the isotropic pressure `tr(σ)/(3V)`, and scales the whole cell — the
/// atom coordinates riding along affinely — towards `P_target`. It never moves
/// an atom relative to the cell, so a run that must also relax the geometry
/// pairs it with a force optimizer or uses `JointIsotropicRelaxParams`.
///
/// A variable-cell run does not terminate on forces: name
/// `PressureConvergedParams` (the extensive-virial convention every classical
/// evaluator writes) or `StressConvergedParams` as the task's criterion.
///
/// Two of the four fields are constructor arguments (`P_target`, `alpha0`) and
/// two are member-only knobs, so the field order here is the member
/// DECLARATION order — what `task_codec` writes and what `task_materializer`
/// replays as construct-then-assign.
struct IsotropicCellRelaxParams {
    /// @brief Target pressure [eV/Å³]; 0 is a relaxed (zero-stress) cell.
    double P_target = 0.0;
    /// @brief Relative scale-step gain (dimensionless) — the proportionality
    ///        between the pressure residual and the requested `|s−1|`.
    double alpha0 = 0.05;
    /// @brief Cap on `|s−1|` per step (dimensionless): how far one iteration
    ///        may breathe the cell.
    double max_scale = 0.05;
    /// @brief Soft bulk modulus [eV/Å³] dividing the step; larger means a
    ///        gentler cell change per iteration.
    double bulk_mod = 1.0;
};
inline bool operator==(const IsotropicCellRelaxParams& a,
                       const IsotropicCellRelaxParams& b) {
    return a.P_target == b.P_target && a.alpha0 == b.alpha0
           && a.max_scale == b.max_scale && a.bulk_mod == b.bulk_mod;
}

/// @brief Wire mirror of the seven public members of
///        `cpu::minimizer::AnisotropicCellRelax`.
///
/// The orthorhombic variant of `IsotropicCellRelaxParams`: the three box edges
/// scale independently from their own diagonal stress components, so the cell
/// changes shape as well as volume. The tilt factors stay put — shear is
/// `TriclinicCellRelaxParams`.
///
/// The per-axis mask is the CP2K `CELL_OPT` / `CONSTRAINT` twin: a `false`
/// axis is never scaled and never enters the residual, so a constrained
/// direction's reaction stress cannot block convergence.
struct AnisotropicCellRelaxParams {
    /// @brief Target pressure [eV/Å³], the same on every relaxed axis.
    double P_target = 0.0;
    /// @brief Per-axis relative scale gain (dimensionless).
    double alpha0 = 0.05;
    /// @brief Cap on `|s−1|` per axis per step (dimensionless).
    double max_scale = 0.05;
    /// @brief Soft bulk modulus [eV/Å³] dividing the step.
    double bulk_mod = 1.0;
    /// @brief Relax the x box edge; `false` freezes it.
    bool relax_x = true;
    /// @brief Relax the y box edge; `false` freezes it.
    bool relax_y = true;
    /// @brief Relax the z box edge; `false` freezes it.
    bool relax_z = true;
};
inline bool operator==(const AnisotropicCellRelaxParams& a,
                       const AnisotropicCellRelaxParams& b) {
    return a.P_target == b.P_target && a.alpha0 == b.alpha0
           && a.max_scale == b.max_scale && a.bulk_mod == b.bulk_mod
           && a.relax_x == b.relax_x && a.relax_y == b.relax_y
           && a.relax_z == b.relax_z;
}

/// @brief Wire mirror of the three public members of
///        `cpu::minimizer::TriclinicCellRelax`.
///
/// The full-cell variant: a Voigt strain tensor ε ∝ σ is formed from the whole
/// stress, off-diagonals included, and the cell matrix is updated as
/// `h ← (I + ε) h`, so the cell may shear. Hence `max_strain` rather than the
/// cubic relaxers' `max_scale` — the capped quantity is a strain component,
/// not a length ratio.
///
/// There is no `P_target`: this relaxer drives the cell towards ZERO stress
/// (the 0 K geometry), which is why its matching criterion is
/// `StressConvergedParams` rather than `PressureConvergedParams`.
struct TriclinicCellRelaxParams {
    /// @brief Strain gain (dimensionless) — the proportionality between the
    ///        stress residual and the requested strain.
    double alpha0 = 0.02;
    /// @brief Cap on `|ε_αβ|` per step (dimensionless).
    double max_strain = 0.05;
    /// @brief Soft modulus [eV/Å³] dividing the step.
    double bulk_mod = 1.0;
};
inline bool operator==(const TriclinicCellRelaxParams& a,
                       const TriclinicCellRelaxParams& b) {
    return a.alpha0 == b.alpha0 && a.max_strain == b.max_strain
           && a.bulk_mod == b.bulk_mod;
}

/// @brief Wire mirror of the five public members of
///        `cpu::minimizer::JointIsotropicRelax`.
///
/// *Joint* means each iteration does both halves of a variable-cell
/// optimization: a steepest-descent atom step along the unit force, then an
/// isotropic cell scale from the virial pressure. `atom_step` is the field
/// that makes it joint; the other four are `IsotropicCellRelaxParams`'.
///
/// Terminating it honestly therefore wants both a force and a pressure test.
/// `MinimizeTask::convergence` carries one criterion, so a launcher payload
/// names the binding one and leaves the other to the iteration budget.
struct JointIsotropicRelaxParams {
    /// @brief Target pressure [eV/Å³].
    double P_target = 0.0;
    /// @brief Atom displacement along the unit force per iteration [Å].
    double atom_step = 0.01;
    /// @brief Relative cell-scale gain (dimensionless).
    double alpha0 = 0.05;
    /// @brief Cap on `|s−1|` per step (dimensionless).
    double max_scale = 0.05;
    /// @brief Soft bulk modulus [eV/Å³] dividing the cell step.
    double bulk_mod = 1.0;
};
inline bool operator==(const JointIsotropicRelaxParams& a,
                       const JointIsotropicRelaxParams& b) {
    return a.P_target == b.P_target && a.atom_step == b.atom_step
           && a.alpha0 == b.alpha0 && a.max_scale == b.max_scale
           && a.bulk_mod == b.bulk_mod;
}

/// @brief Mirrors all five `cpu::ForceConverged` constructor tolerances.
///
/// The fixed-cell stop policy, and the one a `MinimizeTask` carries unless it
/// says otherwise. **All five** tests must hold at once for the criterion to
/// report convergence: a geometry is converged when the forces are small, the
/// last step moved almost nothing, and the energy stopped changing. The
/// displacement pair is skipped on an iteration where no previous geometry
/// exists to difference against.
struct ForceConvergedParams {
    /// @brief Largest per-atom force magnitude allowed [eV/Å].
    double ftol = 0.05;
    /// @brief Root-mean-square force over all atoms allowed [eV/Å].
    double rms_ftol = 0.03;
    /// @brief Largest per-atom displacement of the last step allowed [Å].
    double max_dtol = 0.002;
    /// @brief Root-mean-square displacement of the last step allowed [Å].
    double rms_dtol = 0.001;
    /// @brief Energy change between the last two iterations allowed [eV].
    double etol = 1e-5;
};
inline bool operator==(const ForceConvergedParams& a,
                       const ForceConvergedParams& b) {
    return a.ftol == b.ftol && a.rms_ftol == b.rms_ftol
           && a.max_dtol == b.max_dtol && a.rms_dtol == b.rms_dtol
           && a.etol == b.etol;
}

/// @brief Mirrors both `cpu::PressureConverged` constructor arguments.
///
/// The termination test of an isotropic variable-cell run: it reads the
/// extensive Voigt virial `kStress` [eV] plus the box volume, forms
/// `P = tr(W)/(3V)` [eV/Å³], and declares convergence once `|P − P_target|`
/// falls below `ptol`. Every producer writes that same extensive convention
/// (`core/frame_schema.h`); `StressConvergedParams` reads the identical key
/// and divides by the identical volume, and differs only in judging each
/// diagonal component separately instead of their mean.
///
/// Only the TOLERANCE carries an invariant: `cpu::PressureConverged` throws on
/// `ptol <= 0` and on nothing else, so a negative target (a tensile setpoint)
/// is a perfectly legal payload while a non-positive tolerance is refused at
/// `TaskSpec` construction rather than at materialization.
struct PressureConvergedParams {
    /// @brief Target pressure [eV/Å³]; 0 is a relaxed cell.
    double P_target = 0.0;
    /// @brief Absolute pressure tolerance [eV/Å³]; must be > 0.
    double ptol = 1e-5;
};
inline bool operator==(const PressureConvergedParams& a,
                       const PressureConvergedParams& b) {
    return a.P_target == b.P_target && a.ptol == b.ptol;
}

/// @brief Mirrors all five `cpu::StressConverged` constructor arguments.
///
/// The per-axis termination test: it reads the same extensive Voigt virial
/// `kStress` [eV] as `PressureConvergedParams` and judges the intensive
/// diagonal stress `σ_dd = W_dd/V` [eV/Å³] of every UNMASKED axis against
/// `sigma_target`, which is what pairs it with
/// `AnisotropicCellRelaxParams::relax_*` — mask the same
/// axes on both and a constrained direction's reaction stress cannot block
/// convergence (the CP2K `CELL_OPT` `CONSTRAINT` twin).
///
/// Two invariants, both `cpu::StressConverged`'s own and both mirrored at
/// `TaskSpec` construction: `stol > 0`, and at least one axis judged — an
/// all-masked criterion would test nothing and declare victory.
struct StressConvergedParams {
    /// @brief Target diagonal stress [eV/Å³]; 0 is a relaxed cell.
    double sigma_target = 0.0;
    /// @brief Absolute stress tolerance [eV/Å³]; must be > 0.
    double stol = 1e-4;
    /// @brief Judge σ_xx.
    bool check_x = true;
    /// @brief Judge σ_yy.
    bool check_y = true;
    /// @brief Judge σ_zz.
    bool check_z = true;
};
inline bool operator==(const StressConvergedParams& a,
                       const StressConvergedParams& b) {
    return a.sigma_target == b.sigma_target && a.stol == b.stol
           && a.check_x == b.check_x && a.check_y == b.check_y
           && a.check_z == b.check_z;
}

/// @brief The stop policy a `MinimizeTask` carries — one criterion, typed.
///
/// A fixed-cell run terminates on forces; a variable-cell run terminates on
/// pressure or on stress. `MinimizeTask::convergence` was a bare
/// `ForceConvergedParams` blob until the cell relaxers landed, which made a
/// pressure-terminated payload inexpressible. The default alternative stays
/// `ForceConvergedParams`, so a default-constructed `MinimizeTask` still means
/// what it always did.
using MinimizeCriterionParams =
    std::variant<ForceConvergedParams, PressureConvergedParams,
                 StressConvergedParams>;

/// @brief Stable wire IDs for the criterion alternatives.
///
/// A separate space from `ComponentId`: a criterion is not a plan entry, it is
/// the singular tail field of a `MinimizeTask`, and its id is read by
/// `read_criterion` alone. Append-only, exactly like `ComponentId` — the VALUE
/// is what crosses the wire.
enum class MinimizeCriterionId : std::uint16_t {
    ForceConverged = 0,    ///< `cpu::ForceConverged` — the fixed-cell default.
    PressureConverged = 1, ///< `cpu::PressureConverged` — isotropic cell.
    StressConverged = 2,   ///< `cpu::StressConverged` — per-axis stress.
};

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
    // The rest of the CPU minimize domain. `CG` and `FIRE` are fixed-cell
    // alternatives to `LBFGS`; the four relaxers below them move the BOX and
    // terminate on `PressureConverged` / `StressConverged` rather than on
    // forces. All six are Minimize-only.
    CG = 21,   ///< Polak-Ribière⁺ conjugate gradient — `cpu::minimizer::CG`.
    FIRE = 22, ///< Bitzek FIRE 1.0 — `cpu::minimizer::FIRE`.
    /// Isotropic variable-cell relax — `cpu::minimizer::IsotropicCellRelax`.
    IsotropicCellRelax = 23,
    /// Orthorhombic per-axis relax — `cpu::minimizer::AnisotropicCellRelax`.
    AnisotropicCellRelax = 24,
    /// Full-cell (shearable) strain relax — `cpu::minimizer::TriclinicCellRelax`.
    TriclinicCellRelax = 25,
    /// Joint atom + isotropic cell relax — `cpu::minimizer::JointIsotropicRelax`.
    JointIsotropicRelax = 26,
    // The widened real-space force-field catalogue. Six pair styles and three
    // bonded ones, all MD-legal force sources on the plural evaluator slot —
    // they JOIN `PairLJCut` and `BondHarmonic` rather than replacing them. All
    // six pair styles walk pairs O(N²) and withdraw their `NeighborConsumer`,
    // so none of them requires a `NeighborList` entry beside it.
    PairMorse = 27,  ///< Morse pair — `cpu::PairMorse`.
    PairBuck = 28,   ///< Buckingham pair — `cpu::PairBuck`.
    PairBorn = 29,   ///< Born-Mayer-Huggins pair table — `cpu::PairBorn`.
    PairYukawa = 30, ///< Screened Coulomb pair — `cpu::PairYukawa`.
    PairWCA = 31,    ///< Purely repulsive WCA pair — `cpu::PairWCA`.
    PairDpd = 32,    ///< Groot-Warren DPD pair — `cpu::PairDpd`.
    BondFene = 33,   ///< Kremer-Grest FENE bond — `cpu::BondFene`.
    /// CHARMM periodic torsion — `cpu::DihedralCharmm`.
    DihedralCharmm = 34,
    /// Harmonic improper torsion — `cpu::ImproperHarmonic`.
    ImproperHarmonic = 35,
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
    case ComponentId::CG:
        return "CG";
    case ComponentId::FIRE:
        return "FIRE";
    case ComponentId::IsotropicCellRelax:
        return "IsotropicCellRelax";
    case ComponentId::AnisotropicCellRelax:
        return "AnisotropicCellRelax";
    case ComponentId::TriclinicCellRelax:
        return "TriclinicCellRelax";
    case ComponentId::JointIsotropicRelax:
        return "JointIsotropicRelax";
    case ComponentId::PairMorse:
        return "PairMorse";
    case ComponentId::PairBuck:
        return "PairBuck";
    case ComponentId::PairBorn:
        return "PairBorn";
    case ComponentId::PairYukawa:
        return "PairYukawa";
    case ComponentId::PairWCA:
        return "PairWCA";
    case ComponentId::PairDpd:
        return "PairDpd";
    case ComponentId::BondFene:
        return "BondFene";
    case ComponentId::DihedralCharmm:
        return "DihedralCharmm";
    case ComponentId::ImproperHarmonic:
        return "ImproperHarmonic";
    }
    return "unknown";
}

/// @brief Map a criterion alternative to its stable wire id.
/// @param criterion The typed stop policy of a `MinimizeTask`.
/// @return The `MinimizeCriterionId` written ahead of its payload.
inline MinimizeCriterionId
criterion_id(const MinimizeCriterionParams& criterion) noexcept {
    return std::visit(
        [](const auto& c) -> MinimizeCriterionId {
            using C = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<C, ForceConvergedParams>)
                return MinimizeCriterionId::ForceConverged;
            else if constexpr (std::is_same_v<C, PressureConvergedParams>)
                return MinimizeCriterionId::PressureConverged;
            else
                return MinimizeCriterionId::StressConverged;
        },
        criterion);
}

/// @brief Human-readable name of a criterion wire id, for diagnostics.
/// @param id A decoded criterion id.
/// @return The C++ class name, or `"unknown"`.
inline const char* criterion_name(MinimizeCriterionId id) noexcept {
    switch (id) {
    case MinimizeCriterionId::ForceConverged:
        return "ForceConverged";
    case MinimizeCriterionId::PressureConverged:
        return "PressureConverged";
    case MinimizeCriterionId::StressConverged:
        return "StressConverged";
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
                 KSpacePmeParams, XyzRecorderParams, PwEvaluatorParams,
                 PairMorseParams, PairBuckParams, PairBornParams,
                 PairYukawaParams, PairWcaParams, PairDpdParams, BondFeneParams,
                 DihedralCharmmParams, ImproperHarmonicParams>;

using MinimizeComponentParams =
    std::variant<PairLJCutParams, NeighborListParams, LBFGSParams, CGParams,
                 FIREParams, IsotropicCellRelaxParams,
                 AnisotropicCellRelaxParams, TriclinicCellRelaxParams,
                 JointIsotropicRelaxParams>;

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
            else if constexpr (std::is_same_v<P, PairMorseParams>)
                return ComponentId::PairMorse;
            else if constexpr (std::is_same_v<P, PairBuckParams>)
                return ComponentId::PairBuck;
            else if constexpr (std::is_same_v<P, PairBornParams>)
                return ComponentId::PairBorn;
            else if constexpr (std::is_same_v<P, PairYukawaParams>)
                return ComponentId::PairYukawa;
            else if constexpr (std::is_same_v<P, PairWcaParams>)
                return ComponentId::PairWCA;
            else if constexpr (std::is_same_v<P, PairDpdParams>)
                return ComponentId::PairDpd;
            else if constexpr (std::is_same_v<P, BondFeneParams>)
                return ComponentId::BondFene;
            else if constexpr (std::is_same_v<P, DihedralCharmmParams>)
                return ComponentId::DihedralCharmm;
            else if constexpr (std::is_same_v<P, ImproperHarmonicParams>)
                return ComponentId::ImproperHarmonic;
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
            else if constexpr (std::is_same_v<P, CGParams>)
                return ComponentId::CG;
            else if constexpr (std::is_same_v<P, FIREParams>)
                return ComponentId::FIRE;
            else if constexpr (std::is_same_v<P, IsotropicCellRelaxParams>)
                return ComponentId::IsotropicCellRelax;
            else if constexpr (std::is_same_v<P, AnisotropicCellRelaxParams>)
                return ComponentId::AnisotropicCellRelax;
            else if constexpr (std::is_same_v<P, TriclinicCellRelaxParams>)
                return ComponentId::TriclinicCellRelax;
            else if constexpr (std::is_same_v<P, JointIsotropicRelaxParams>)
                return ComponentId::JointIsotropicRelax;
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

/// @brief One launcher-dispatchable geometry optimization: what to build,
///        how long to iterate, and when to stop.
struct MinimizeTask {
    /// Every component of the run — the optimizer, the neighbor list and the
    /// force sources — each with an optional explicit phase binding.
    std::vector<MinimizeComponentEntry> entries;
    /// Iteration budget [count]; the loop stops here even if no criterion has
    /// fired.
    std::int64_t max_iter;
    /// The single stop policy of this run. A variant since v5: a fixed-cell
    /// run judges forces, a variable-cell run judges pressure or stress.
    MinimizeCriterionParams convergence;
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
    ///
    /// v5 is a LAYOUT change, which is why it is a bump where the six appended
    /// minimize component IDs (21-26) would not have been: `MinimizeTask`'s
    /// tail stopped being five bare `ForceConverged` doubles and became a
    /// criterion variant — a `u16` id followed by that criterion's own fields.
    /// A v4 payload's doubles would decode as an id followed by garbage, so
    /// the exact-match version gate in `unpack` is what makes the break loud.
    /// Same reasoning as the v4 `NoseHooverNptParams` bump.
    ///
    /// The nine real-space catalogue IDs (27-35) land under v5 without a bump,
    /// for the reason IDs 21-26 would not have forced one: an appended ID with
    /// a fresh payload leaves every existing byte layout untouched, so a v5
    /// decoder written before them reads every previously encodable payload
    /// unchanged. Only a LAYOUT change is a bump.
    static constexpr std::uint32_t kWireVersion = 5;

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
