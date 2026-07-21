/// @file src/cpu/vib/hessian_diagonalizer.cpp
/// @brief cpu::vib::HessianDiagonalizer — verbatim port of PySCF
///        `pyscf/hessian/thermo.py::harmonic_analysis`.
///
/// Reproduces `_get_TR`, `_get_rotor_type`, `rotation_const`, the QR
/// translation/rotation projector, the projected diagonalization, and
/// the `freq_au -> cm^-1` conversion. The reference-port-fidelity rule
/// applies: the numerics mirror PySCF, they are not paraphrased.

#include "atomiverse/cpu/vib/hessian_diagonalizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/vib/nist_constants.h"

namespace ATV {
namespace cpu {
namespace vib {

namespace {

/// @brief PySCF `harmonic_analysis` linear-dependence threshold.
constexpr double kLindepThreshold = 1e-7;

/// @brief 3x3 matrix as a flat row-major array.
using Mat3x3 = std::array<double, 9>;

/// @brief Symmetric 3x3 eigendecomposition, ascending eigenvalues.
///
/// Mirrors `numpy.linalg.eigh` for the moment-of-inertia tensor.
/// @param m  Symmetric 3x3 matrix, row-major.
/// @param w  [out] Eigenvalues, ascending, length 3.
/// @param v  [out] Eigenvectors as columns: column k is eigenvector k,
///           stored row-major 3x3.
void eigh3(const Mat3x3& m, std::array<double, 3>& w, Mat3x3& v) {
    xt::xtensor<double, 2> M = xt::zeros<double>({3, 3});
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M(i, j) = m[3 * i + j];
    auto [eig, vec] = xt::linalg::eigh(M);
    for (int i = 0; i < 3; ++i) {
        w[i] = eig(i);
        for (int j = 0; j < 3; ++j)
            v[3 * j + i] = vec(j, i);
    }
}

/// @brief Principal-axis rotational constants [GHz].
///
/// Verbatim port of `pyscf/hessian/thermo.py::rotation_const` with
/// `unit='GHz'`. `coords` are mass-centered atomic coordinates in Bohr.
/// @param mass   Per-atom masses [amu], length N.
/// @param coords Mass-centered coordinates [Bohr], shape [N,3] row-major.
/// @return The three rotational constants [GHz], ascending.
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

    std::array<double, 3> e{};
    Mat3x3 dummy{};
    eigh3(im, e, dummy); // eigvalsh -> ascending eigenvalues.

    // unit_im = ATOMIC_MASS * BOHR_SI^2 ; unit_hz = HBAR / (4*pi*unit_im)
    const double unit_im = nist::ATOMIC_MASS * nist::BOHR_SI * nist::BOHR_SI;
    const double unit_hz = nist::HBAR / (4.0 * nist::PI * unit_im);
    for (int i = 0; i < 3; ++i) {
        if (e[i] != 0.0)
            e[i] = unit_hz / e[i] * 1e-9; // GHz
        else
            e[i] = std::numeric_limits<double>::infinity();
    }
    return e;
}

/// @brief Rotor-type classification — verbatim `_get_rotor_type`.
///
/// @param rot_const Rotational constants [GHz].
/// @return 0 = ATOM, 1 = LINEAR, 2 = REGULAR.
int get_rotor_type(const std::array<double, 3>& rot_const) {
    const bool all_huge =
        rot_const[0] > 1e8 && rot_const[1] > 1e8 && rot_const[2] > 1e8;
    if (all_huge)
        return 0; // ATOM
    if (rot_const[0] > 1e8 && (rot_const[1] - rot_const[2] < 1e-3))
        return 1; // LINEAR
    return 2;     // REGULAR
}

/// @brief Build the 6 mass-weighted translation/rotation vectors.
///
/// Verbatim port of `pyscf/hessian/thermo.py::_get_TR`.
/// @param mass   Per-atom masses [amu], length N.
/// @param coords Mass-centered coordinates [Bohr], shape [N,3] row-major.
/// @return 6 vectors of length 3N: Tx,Ty,Tz,Rx,Ry,Rz.
std::array<std::vector<double>, 6> get_TR(const std::vector<double>& mass,
                                          const std::vector<double>& coords) {
    const int n = static_cast<int>(mass.size());
    const int dim = 3 * n;
    std::vector<double> massp(n);
    for (int i = 0; i < n; ++i)
        massp[i] = std::sqrt(mass[i]);

    std::array<std::vector<double>, 6> TR;
    for (auto& v : TR)
        v.assign(dim, 0.0);

    // Translational modes: Tx,Ty,Tz = massp ⊗ unit axis.
    for (int i = 0; i < n; ++i) {
        TR[0][3 * i + 0] = massp[i];
        TR[1][3 * i + 1] = massp[i];
        TR[2][3 * i + 2] = massp[i];
    }

    // im = einsum('m,mx,my->xy', mass, coords, coords)
    Mat3x3 im{};
    for (int m = 0; m < n; ++m)
        for (int x = 0; x < 3; ++x)
            for (int y = 0; y < 3; ++y)
                im[3 * x + y] +=
                    mass[m] * coords[3 * m + x] * coords[3 * m + y];
    const double tr = im[0] + im[4] + im[8];
    for (int x = 0; x < 3; ++x)
        for (int y = 0; y < 3; ++y)
            im[3 * x + y] = (x == y ? tr : 0.0) - im[3 * x + y];

    // w, paxes = eigh(im); then reverse so z-axis has smallest inertia.
    std::array<double, 3> w{};
    Mat3x3 paxes{}; // columns are eigenvectors.
    eigh3(im, w, paxes);
    // Reverse column order: paxes[:, ::-1].
    Mat3x3 paxes_rev{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            paxes_rev[3 * r + c] = paxes[3 * r + (2 - c)];

    // ex,ey,ez = paxes.T  (rows of paxes^T == columns of paxes_rev).
    const double ex[3] = {paxes_rev[0], paxes_rev[3], paxes_rev[6]};
    const double ey[3] = {paxes_rev[1], paxes_rev[4], paxes_rev[7]};
    const double ez[3] = {paxes_rev[2], paxes_rev[5], paxes_rev[8]};

    // coords_in_rot_frame = coords.dot(paxes)
    // cx,cy,cz are the columns of that product.
    for (int m = 0; m < n; ++m) {
        double c[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k)
            for (int r = 0; r < 3; ++r)
                c[k] += coords[3 * m + r] * paxes_rev[3 * r + k];
        const double cx = c[0], cy = c[1], cz = c[2];
        // Rx = massp * (cy*ez - cz*ey), etc.
        for (int d = 0; d < 3; ++d) {
            TR[3][3 * m + d] = massp[m] * (cy * ez[d] - cz * ey[d]);
            TR[4][3 * m + d] = massp[m] * (cz * ex[d] - cx * ez[d]);
            TR[5][3 * m + d] = massp[m] * (cx * ey[d] - cy * ex[d]);
        }
    }
    return TR;
}

} // namespace

void HessianDiagonalizer::launch(const cpu::LaunchContext& ctx) {
    const int n = ws_->n_atoms;
    const int dim = 3 * n;
    if (ctx.step < 2 * dim)
        return; // not the finalize pass yet.
    if (!ws_->mass_weighted)
        return; // MassWeighting must run first.

    if (n < 2) {
        // No genuine vibrational modes for a single atom.
        ws_->frequencies = xt::zeros<double>({0});
        ws_->normal_modes = xt::zeros<double>({dim, 0});
        ws_->n_imaginary = 0;
        ws_->is_linear = false;
        return;
    }

    const std::vector<double>& mass = ws_->mass_amu;

    // Mass-centered atomic coordinates in Bohr (PySCF works in Bohr).
    // r0 is the equilibrium geometry in Angstrom (link 1).
    double mass_sum = 0.0;
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

    // ── TR subspace ────────────────────────────────────────────────────
    auto TR = get_TR(mass, coords);
    const auto rot_const = rotation_const(mass, coords);
    const int rotor = get_rotor_type(rot_const);
    ws_->is_linear = (rotor == 1);

    // Assemble the TR-space rows: always 3 translations; rotations depend
    // on rotor type (ATOM: none, LINEAR: 2, REGULAR: 3).
    std::vector<int> tr_rows = {0, 1, 2};
    if (rotor == 1) { // LINEAR -> TR[3:5]
        tr_rows.push_back(3);
        tr_rows.push_back(4);
    } else if (rotor == 2) { // REGULAR -> TR[3:]
        tr_rows.push_back(3);
        tr_rows.push_back(4);
        tr_rows.push_back(5);
    }
    const int n_tr = static_cast<int>(tr_rows.size());

    // ── Mass-weighted Hessian (produced by MassWeighting) ──────────────
    xt::xtensor<double, 2> h = xt::zeros<double>({dim, dim});
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j)
            h(i, j) = ws_->hessian_mw(i, j);

    xt::xtensor<double, 1> force_const_au;
    xt::xtensor<double, 2> mode; // columns are eigenvectors in the full basis.

    if (n_tr > 0) {
        // TRspace.T has shape [3N, n_tr]; QR-orthonormalize it.
        xt::xtensor<double, 2> TRspaceT = xt::zeros<double>({dim, n_tr});
        for (int c = 0; c < n_tr; ++c) {
            const std::vector<double>& v = TR[tr_rows[c]];
            for (int r = 0; r < dim; ++r)
                TRspaceT(r, c) = v[r];
        }
        auto qr = xt::linalg::qr(TRspaceT);
        const xt::xtensor<double, 2>& q = std::get<0>(qr); // [3N, n_tr]

        // P = I - q q^T
        xt::xtensor<double, 2> P = xt::zeros<double>({dim, dim});
        for (int i = 0; i < dim; ++i)
            P(i, i) = 1.0;
        xt::xtensor<double, 2> qqt = xt::linalg::dot(q, xt::transpose(q));
        P = P - qqt;

        // w, v = eigh(P); bvec = v[:, w > LINDEP_THRESHOLD]
        auto [w, v] = xt::linalg::eigh(P);
        std::vector<int> keep;
        for (int i = 0; i < dim; ++i)
            if (w(i) > kLindepThreshold)
                keep.push_back(i);
        const int nb = static_cast<int>(keep.size());

        xt::xtensor<double, 2> bvec = xt::zeros<double>({dim, nb});
        for (int c = 0; c < nb; ++c)
            for (int r = 0; r < dim; ++r)
                bvec(r, c) = v(r, keep[c]);

        // h_proj = bvec^T h bvec  -> [nb, nb]
        xt::xtensor<double, 2> h_proj =
            xt::linalg::dot(xt::transpose(bvec), xt::linalg::dot(h, bvec));

        auto [fc, m_small] = xt::linalg::eigh(h_proj); // [nb], [nb,nb]
        force_const_au = fc;
        // mode = bvec.dot(m_small) -> lift back to [3N, nb]
        mode = xt::linalg::dot(bvec, m_small);
    } else {
        // ATOM: no TR projection, diagonalize the full mass-weighted H.
        auto [fc, m_full] = xt::linalg::eigh(h);
        force_const_au = fc;
        mode = m_full;
    }

    // ── Force constants -> wavenumbers ─────────────────────────────────
    // freq_au = sqrt(force_const_au); imaginary modes (lambda<0) reported
    // as negative wavenumbers (imaginary_freq=False convention).
    const int n_modes = static_cast<int>(force_const_au.size());
    const double au2cm = nist::au2wavenumber();
    ws_->frequencies = xt::zeros<double>({n_modes});
    int n_imag = 0;
    for (int i = 0; i < n_modes; ++i) {
        const double lambda = force_const_au(i);
        if (lambda < 0.0) {
            ws_->frequencies(i) = -std::sqrt(-lambda) * au2cm;
            ++n_imag;
        } else {
            ws_->frequencies(i) = std::sqrt(lambda) * au2cm;
        }
    }
    ws_->n_imaginary = n_imag;

    // Normal modes lifted to the full 3N Cartesian basis.
    ws_->normal_modes = xt::zeros<double>({dim, n_modes});
    for (int i = 0; i < dim; ++i)
        for (int k = 0; k < n_modes; ++k)
            ws_->normal_modes(i, k) = mode(i, k);
}

} // namespace vib
} // namespace cpu
} // namespace ATV
