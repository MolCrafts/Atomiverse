/// @file src/cpu/vib/thermochem.cpp
/// @brief cpu::vib::Thermochem — verbatim port of PySCF
///        `pyscf/hessian/thermo.py::thermo()`.
///
/// Reproduces the electronic / translational / rotational / vibrational
/// component functions, the `rotation_const` and `_get_rotor_type`
/// helpers, and the `_sum` total aggregation. The reference-port
/// fidelity rule applies: the numerics mirror PySCF, they are not
/// paraphrased. All quantities are in Hartree (Eh) / Eh/K, exactly as
/// PySCF `thermo()` reports them.

#include "atomiverse/cpu/vib/thermochem.h"

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/vib/atomic_masses.h"
#include "atomiverse/cpu/vib/nist_constants.h"

namespace ATV {
namespace cpu {
namespace vib {

namespace {

/// @brief 3x3 matrix as a flat row-major array.
using Mat3x3 = std::array<double, 9>;

/// @brief Ascending eigenvalues of a symmetric 3x3 matrix.
///
/// Mirrors `numpy.linalg.eigvalsh` for the moment-of-inertia tensor.
/// @param m Symmetric 3x3 matrix, row-major.
/// @return The three eigenvalues, ascending.
std::array<double, 3> eigvalsh3(const Mat3x3& m) {
    xt::xtensor<double, 2> M = xt::zeros<double>({3, 3});
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M(i, j) = m[3 * i + j];
    auto eig = xt::linalg::eigvalsh(M);
    return {eig(0), eig(1), eig(2)};
}

/// @brief Principal-axis rotational constants [GHz].
///
/// Verbatim port of `pyscf/hessian/thermo.py::rotation_const` with
/// `unit='GHz'`. `coords` are mass-centered atomic coordinates in Bohr.
/// @param mass   Per-atom masses [amu], length N.
/// @param coords Mass-centered coordinates [Bohr], shape [N,3] row-major.
/// @return The three rotational constants [GHz], ascending; infinite for
///         a vanishing principal moment of inertia.
std::array<double, 3> rotation_const(const std::vector<double>& mass,
                                     const std::vector<double>& coords) {
    const int n = static_cast<int>(mass.size());
    // im = einsum('z,zr,zs->rs', mass, r, r)
    Mat3x3 im{};
    for (int z = 0; z < n; ++z)
        for (int r = 0; r < 3; ++r)
            for (int s = 0; s < 3; ++s)
                im[3 * r + s] +=
                    mass[z] * coords[3 * z + r] * coords[3 * z + s];
    // im = eye(3) * im.trace() - im
    const double tr = im[0] + im[4] + im[8];
    for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s)
            im[3 * r + s] = (r == s ? tr : 0.0) - im[3 * r + s];

    const std::array<double, 3> e = eigvalsh3(im);

    // unit_im = ATOMIC_MASS * BOHR_SI^2 ; unit_hz = HBAR / (4*pi*unit_im)
    const double unit_im = nist::ATOMIC_MASS * nist::BOHR_SI * nist::BOHR_SI;
    const double unit_hz = nist::HBAR / (4.0 * nist::PI * unit_im);
    std::array<double, 3> rc{};
    for (int i = 0; i < 3; ++i) {
        if (e[i] != 0.0)
            rc[i] = unit_hz / e[i] * 1e-9; // GHz
        else
            rc[i] = std::numeric_limits<double>::infinity();
    }
    return rc;
}

/// @brief Rotor-type classification — verbatim `_get_rotor_type`.
///
/// @param rc Rotational constants [GHz], ascending.
/// @return 0 = ATOM, 1 = LINEAR, 2 = REGULAR.
int get_rotor_type(const std::array<double, 3>& rc) {
    const bool all_huge = rc[0] > 1e8 && rc[1] > 1e8 && rc[2] > 1e8;
    if (all_huge)
        return 0; // ATOM
    if (rc[0] > 1e8 && (rc[1] - rc[2] < 1e-3))
        return 1; // LINEAR
    return 2;     // REGULAR
}

} // namespace

void Thermochem::launch(const cpu::LaunchContext& ctx) {
    if (ws_ == nullptr)
        return; // test-only role-check ctor.
    const int n = ws_->n_atoms;
    const int dim = 3 * n;
    if (ctx.step < 2 * dim)
        return; // not the finalize pass yet.

    // ── Constants (PySCF `thermo()` literal forms) ──────────────────────
    const double kB = nist::BOLTZMANN; // J/K
    const double h = nist::PLANCK;     // J*s
    const double T = cfg_.temperature; // K
    const double P = cfg_.pressure;    // Pa
    const int sigma = cfg_.symmetry_number;
    // R_Eh = kB * N_A / (HARTREE2J * N_A) = kB / HARTREE2J  [Eh/K].
    const double R_Eh =
        kB * nist::AVOGADRO / (nist::HARTREE2J * nist::AVOGADRO);

    const std::vector<double>& mass = ws_->mass_amu;

    // Mass-centered atomic coordinates in Bohr (PySCF works in Bohr).
    // r0 is the equilibrium geometry in Angstrom (link 1).
    double mass_sum = 0.0; // amu
    for (int i = 0; i < n; ++i)
        mass_sum += mass[i];
    double center[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d)
            center[d] += mass[i] * ws_->r0[3 * i + d];
    for (int d = 0; d < 3; ++d)
        center[d] /= mass_sum;

    std::vector<double> coords(dim); // mass-centered, Bohr.
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d)
            coords[3 * i + d] =
                (ws_->r0[3 * i + d] - center[d]) * units::kAngstromInBohr;

    // ── Electronic part ─────────────────────────────────────────────────
    // E0 (the SCF total energy) cancels in the thermal Gibbs correction
    // G_tot - E0, so it is carried as zero — `g_thermal` is exactly that
    // correction. S_elec = R_Eh * ln(multiplicity); 0 for a singlet.
    const double E0 = 0.0;
    const double S_elec =
        R_Eh * std::log(static_cast<double>(cfg_.multiplicity));
    const double E_elec = E0;
    const double H_elec = E0;

    // ── Translational part (Sackur-Tetrode) ─────────────────────────────
    const double mass_tot = mass_sum * nist::ATOMIC_MASS; // kg
    const double q_trans =
        std::pow(2.0 * nist::PI * mass_tot * kB * T / (h * h), 1.5) * kB * T
        / P;
    const double S_trans = R_Eh * (2.5 + std::log(q_trans));
    const double E_trans = 1.5 * R_Eh * T;
    const double H_trans = 2.5 * R_Eh * T; // includes the +kB*T term.

    // ── Rotational part ─────────────────────────────────────────────────
    const auto rot_const = rotation_const(mass, coords); // GHz
    const int rotor = get_rotor_type(rot_const);
    double S_rot = 0.0, E_rot = 0.0, H_rot = 0.0;
    if (rotor == 0) { // ATOM — no rotational DOF.
        S_rot = E_rot = H_rot = 0.0;
    } else if (rotor == 1) {                 // LINEAR — 2 rotational DOF.
        const double B = rot_const[1] * 1e9; // Hz
        const double q_rot = kB * T / (sigma * h * B);
        S_rot = R_Eh * (1.0 + std::log(q_rot));
        E_rot = H_rot = R_Eh * T;
    } else { // REGULAR — 3 rotational DOF.
        const double ABC = (rot_const[0] * 1e9) * (rot_const[1] * 1e9)
                           * (rot_const[2] * 1e9); // Hz^3
        const double q_rot = std::pow(kB * T / h, 1.5) * std::sqrt(nist::PI)
                             / (sigma * std::sqrt(ABC));
        S_rot = R_Eh * (1.5 + std::log(q_rot));
        E_rot = H_rot = 1.5 * R_Eh * T;
    }
    ws_->is_linear = (rotor == 1);

    // ── Vibrational part (harmonic oscillator) ──────────────────────────
    // The workspace `frequencies` are wavenumbers [cm^-1] with negative
    // entries for imaginary modes. PySCF works in atomic-unit angular
    // frequencies: freq_au = freq_cm1 / au2wavenumber(). Only real
    // positive modes enter the vibrational sums (`idx = freq.real > 0`).
    const double au2cm = nist::au2wavenumber();
    const double au2hz = nist::au2hz();
    const int n_modes = static_cast<int>(ws_->frequencies.size());

    double zpe = 0.0;         // R_Eh * 0.5 * sum theta_i
    double s_vib_acc = 0.0;   // R_Eh * sum (rt*e/(1-e) - ln(1-e))
    double e_vib_extra = 0.0; // R_Eh * T * sum rt*e/(1-e)
    int n_imag = 0;
    for (int i = 0; i < n_modes; ++i) {
        const double freq_cm1 = ws_->frequencies(i);
        if (freq_cm1 <= 0.0) {
            ++n_imag;
            continue;
        } // drop imaginary.
        const double freq_au = freq_cm1 / au2cm;
        // theta = freq_au * au2hz * h / kB  (vibrational temperature).
        const double theta = freq_au * au2hz * h / kB;
        const double rt = theta / std::max(1e-14, T);
        const double e = std::exp(-rt);
        zpe += theta;
        s_vib_acc += rt * e / (1.0 - e) - std::log(1.0 - e);
        e_vib_extra += rt * e / (1.0 - e);
    }
    const double ZPE = R_Eh * 0.5 * zpe;
    const double S_vib = R_Eh * s_vib_acc;
    const double E_vib = ZPE + R_Eh * T * e_vib_extra; // == H_vib.
    const double H_vib = E_vib;

    // ── Totals (PySCF `_sum` over elec/trans/rot/vib) ───────────────────
    const double S_tot = S_elec + S_trans + S_rot + S_vib;
    const double E_tot = E_elec + E_vib + E_trans + E_rot;
    const double H_tot = H_elec + H_vib + H_trans + H_rot;
    const double G_tot = H_tot - T * S_tot;

    // ── Publish into the shared workspace ───────────────────────────────
    ws_->zpe = ZPE;
    ws_->u_tot = E_tot;
    ws_->h_tot = H_tot;
    ws_->s_tot = S_tot;
    ws_->s_elec = S_elec;
    ws_->s_trans = S_trans;
    ws_->s_rot = S_rot;
    ws_->s_vib = S_vib;
    ws_->g_tot = G_tot;
    ws_->g_thermal = G_tot - E0; // E0 carried as zero.
    ws_->n_imaginary = n_imag;
    ws_->thermo_ready = true;
}

} // namespace vib
} // namespace cpu
} // namespace ATV
