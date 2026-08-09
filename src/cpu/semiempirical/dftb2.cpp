/// @file src/cpu/semiempirical/dftb2.cpp
/// SCC-DFTB2 multi-element: mio-1-1 H–H pin + sketch (spin + analytic F) + 3ob.
/// Closed-shell analytic nuclear forces on BOTH SK sets; MioSketch runs closed
/// shell AND collinear open shell through ONE spin-resolved closed-form
/// gradient + pair-virial cell stress (`analytic_forces_stress`).
/// Γ-point PBC (MioSketch): lattice-summed H0/S, screened γ, and repulsive.
/// Multi-k (MioSketch restricted): Bloch-phased complex H0(k)/S(k) on an MP
/// mesh, plus the Bloch-sum analytic gradient and pair virial (real
/// image-resolved P̃/W̃ from P(k)/W(k)).
/// Optional additive Grimme D3(BJ) dispersion (molecular sum, analytic forces).
#include "atomiverse/cpu/semiempirical/dftb2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/core/mat3.h"
#include "atomiverse/core/units.h"
#include "atomiverse/cpu/gpw/internal/dftd3.h"
#include "atomiverse/cpu/gpw/internal/dftd3_params.h"
#include "atomiverse/cpu/linalg/eigensolve.h"
#include "atomiverse/cpu/linalg/matrix.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_3ob_tables.h"
#include "atomiverse/cpu/semiempirical/basis_data/dftb_mio_tables.h"
#include "atomiverse/cpu/semiempirical/internal/dftb_gamma.h"
#include "atomiverse/cpu/semiempirical/internal/mio_sk_block.h"
#include "atomiverse/cpu/semiempirical/internal/mio_sk_table.h"
#include "atomiverse/cpu/semiempirical/internal/scc_mixer.h"

namespace ATV {
namespace cpu {
namespace semiempirical {
namespace {

using basis_data::Dftb3obElement;
using basis_data::Dftb3obPair;
using basis_data::Dftb3obSkPoint;
using basis_data::DftbElementRow;
using basis_data::DftbPairRepRow;
using basis_data::DftbPairSkRow;
using basis_data::kDftb3obElements;
using basis_data::kDftb3obNElements;
using basis_data::kDftb3obNPairs;
using basis_data::kDftb3obPairs;
using basis_data::kDftbElements;
using basis_data::kDftbNElements;
using basis_data::kDftbNPairRep;
using basis_data::kDftbNPairSk;
using basis_data::kDftbPairRep;
using basis_data::kDftbPairSk;

// Hartree → eV; Bohr → Å (CODATA-aligned with common DFTB pins).
constexpr double kHa2eV = 27.211386245988;
constexpr double kBohr2A = 0.529177210903;
constexpr double kA2Bohr = 1.0 / kBohr2A;

const DftbElementRow& element_row(int z) {
    for (std::size_t i = 0; i < kDftbNElements; ++i)
        if (kDftbElements[i].z == z)
            return kDftbElements[i];
    throw std::runtime_error("DFTB2: unsupported element Z=" + std::to_string(z)
                             + " (static mio-style tables cover H/C/N/O only)");
}

const DftbPairSkRow& pair_sk(int za, int zb) {
    for (std::size_t i = 0; i < kDftbNPairSk; ++i)
        if (kDftbPairSk[i].z_a == za && kDftbPairSk[i].z_b == zb)
            return kDftbPairSk[i];
    throw std::runtime_error("DFTB2: no SK table for pair Z="
                             + std::to_string(za) + "-" + std::to_string(zb));
}

const DftbPairRepRow& pair_rep(int za, int zb) {
    for (std::size_t i = 0; i < kDftbNPairRep; ++i)
        if (kDftbPairRep[i].z_a == za && kDftbPairRep[i].z_b == zb)
            return kDftbPairRep[i];
    throw std::runtime_error("DFTB2: no repulsive table for pair Z="
                             + std::to_string(za) + "-" + std::to_string(zb));
}

double h_radial(double v0, double r, double r0, double d) {
    return v0 * std::exp(-(r - r0) / d);
}
double s_radial(double s0, double r, double r0, double w) {
    const double t = (r - r0) / w;
    return s0 * std::exp(-0.5 * t * t);
}

/// Retained iterates in the Anderson charge mixer (dimensionless).
constexpr int kSccMixDepth = 6;

/// Σ_μν A_μν B_μν — Tr(AB) for the symmetric density/Hamiltonian pair.
double trace_product(const linalg::RealMatrix& a, const linalg::RealMatrix& b) {
    const int n = static_cast<int>(a.shape(0));
    double t = 0;
    for (int mu = 0; mu < n; ++mu)
        for (int nu = 0; nu < n; ++nu)
            t += a(mu, nu) * b(mu, nu);
    return t;
}

/// A non-converged SCC is an error, never a silently returned iterate: the
/// undamped Picard loop is a limit cycle, so "last iterate" is selected by the
/// parity of max_scc rather than by physics.
void require_scc_converged(bool converged, double residual, double scc_tol,
                           int max_scc) {
    if (converged)
        return;
    throw std::runtime_error(
        "DFTB2: SCC did not converge in " + std::to_string(max_scc)
        + " iterations (residual " + std::to_string(residual) + " e > scc_tol "
        + std::to_string(scc_tol) + "); raise max_scc or lower mix_beta");
}

/// DFTB+ Elstner γ_AB [eV] with R [Å], U [eV] (see internal/dftb_gamma.h).
///
/// Was Ohno–Klopman; that over-screened poorly and left H₂O E² ≈ 0.65 eV vs
/// DFTB+ 0.496 eV at the same Δq. Family oracle is DFTB+ expGamma form.
double gamma_ab(double r, double ua, double ub) {
    return internal::dftb_gamma::gamma_eV(r, ua, ub);
}

/// mio-1-1 pair cutoff [Å] (SK grid + distFudge).
double pair_rcut_A(const DftbElementRow& ea, const DftbElementRow& eb, int za,
                   int zb) {
    (void)ea;
    (void)eb;
    return internal::MioSkTable{za, zb}.cutoff();
}

double pair_repulsive_eV(int za, int zb, double r_A) {
    return internal::MioSkTable{za, zb}.repulsive(r_A);
}

double pair_repulsive_dE_dr(int za, int zb, double r_A) {
    return internal::MioSkTable{za, zb}.d_repulsive_dr(r_A);
}

struct LatVec {
    double tx, ty, tz;
    int n0 = 0, n1 = 0, n2 = 0; ///< integer lattice indices (for Bloch phase)
};

/// @brief Enumerate lattice translations with |T| ≲ max_cut + cell edge.
/// Columns of `cell` are lattice vectors (matches Box). Includes T = 0.
std::vector<LatVec> enumerate_lattice(const Mat3& cell, double max_cut) {
    double col_len[3];
    for (int d = 0; d < 3; ++d) {
        const double cx = static_cast<double>(cell(0, d));
        const double cy = static_cast<double>(cell(1, d));
        const double cz = static_cast<double>(cell(2, d));
        col_len[d] = std::sqrt(cx * cx + cy * cy + cz * cz);
        if (col_len[d] < 1e-9)
            throw std::runtime_error("DFTB2: degenerate cell edge");
    }
    int nmax[3];
    for (int d = 0; d < 3; ++d)
        nmax[d] =
            std::max(1, static_cast<int>(std::ceil(max_cut / col_len[d])) + 1);

    std::vector<LatVec> out;
    out.reserve(static_cast<std::size_t>((2 * nmax[0] + 1) * (2 * nmax[1] + 1)
                                         * (2 * nmax[2] + 1)));
    for (int n0 = -nmax[0]; n0 <= nmax[0]; ++n0) {
        for (int n1 = -nmax[1]; n1 <= nmax[1]; ++n1) {
            for (int n2 = -nmax[2]; n2 <= nmax[2]; ++n2) {
                const double tx = n0 * static_cast<double>(cell(0, 0))
                                  + n1 * static_cast<double>(cell(0, 1))
                                  + n2 * static_cast<double>(cell(0, 2));
                const double ty = n0 * static_cast<double>(cell(1, 0))
                                  + n1 * static_cast<double>(cell(1, 1))
                                  + n2 * static_cast<double>(cell(1, 2));
                const double tz = n0 * static_cast<double>(cell(2, 0))
                                  + n1 * static_cast<double>(cell(2, 1))
                                  + n2 * static_cast<double>(cell(2, 2));
                out.push_back({tx, ty, tz, n0, n1, n2});
            }
        }
    }
    return out;
}

std::vector<LatVec>
lattice_images_for(const cpu::Box& box,
                   const std::vector<const DftbElementRow*>& el) {
    double max_cut = 0;
    for (const auto* e : el)
        max_cut = std::max(max_cut, e->r_cut);
    // mio-1-1 pair cutoffs (SK grid + distFudge) may outrange element r_cut.
    for (std::size_t i = 0; i < basis_data::kMioSkNPairs; ++i) {
        const auto& p = basis_data::kMioSkPairs[i];
        max_cut = std::max(max_cut, internal::MioSkTable{p.za, p.zb}.cutoff());
    }
    if (max_cut < 1e-12)
        max_cut = 1.0;
    return enumerate_lattice(box.cell(), max_cut);
}

/// dγ/dR [eV/Å] for the DFTB+ Elstner form.
double dgamma_dr(double r, double ua, double ub) {
    return internal::dftb_gamma::dgamma_dr_eV(r, ua, ub);
}

using Cplx = std::complex<double>;
using CplxMatrix = linalg::CplxMatrix;

/// One Monkhorst–Pack sampling point: fractional κ (dimensionless) + weight.
struct KPoint {
    double frac[3] = {0, 0, 0};
    double weight = 1.0;
};

struct Occupancy {
    int n_alpha = 0;
    int n_beta = 0;
    bool unrestricted = false;
};

Occupancy resolve_occupancy(double n_elec, int multiplicity) {
    if (multiplicity < 1)
        throw std::runtime_error("DFTB2: multiplicity must be >= 1");
    if (std::fabs(n_elec - std::round(n_elec)) > 1e-9)
        throw std::runtime_error("DFTB2: non-integer valence electron count");
    const int n_e = static_cast<int>(std::lround(n_elec));
    if (n_e < 0)
        throw std::runtime_error("DFTB2: negative electron count after charge");
    const int n_unpaired = multiplicity - 1;
    if ((n_e - n_unpaired) % 2 != 0 || n_e < n_unpaired)
        throw std::runtime_error("DFTB2: charge/multiplicity inconsistent with "
                                 "valence electron count");
    Occupancy occ;
    occ.n_alpha = (n_e + n_unpaired) / 2;
    occ.n_beta = (n_e - n_unpaired) / 2;
    occ.unrestricted = (multiplicity > 1) || (n_e % 2 != 0);
    return occ;
}

struct SccResult {
    double energy = 0;
    double residual = 0;
    int iterations = 0;
    std::vector<double> dq;
    std::vector<double> spin; // m_A = Nα − Nβ
    // Restricted-path orbitals for closed-form forces.
    linalg::RealMatrix C;
    linalg::RealVector eps;
    int n_occ = 0;
    // Unrestricted-path orbitals (α/β) for collinear open-shell analytic F.
    linalg::RealMatrix Ca;
    linalg::RealMatrix Cb;
    linalg::RealVector eps_a;
    linalg::RealVector eps_b;
    int n_alpha = 0;
    int n_beta = 0;
    bool unrestricted = false;
    // Multi-k path: converged Bloch densities for the analytic gradient.
    // Restricted: P(k) = 2 Σ_o C C†, W(k) = 2 Σ_o ε C C†.
    // Open shell: Pak/Pbk (α/β, occupation 1) and totals Pk=Pa+Pb, Wk=Wa+Wb.
    std::vector<KPoint> kpts;
    std::vector<CplxMatrix> Pk;
    std::vector<CplxMatrix> Wk;
    std::vector<CplxMatrix> Pak;
    std::vector<CplxMatrix> Pbk;
    std::vector<CplxMatrix> Wak;
    std::vector<CplxMatrix> Wbk;
};

/// Build lattice-summed H0/S at the Γ-point (molecular limit when L ≫ r_cut).
void build_h0_s(int n_at, int n_ao,
                const std::vector<const DftbElementRow*>& el,
                const std::vector<int>& ao0, const int* type,
                const StorageReal* x, const StorageReal* y,
                const StorageReal* z, const cpu::Box& box,
                linalg::RealMatrix& H0, linalg::RealMatrix& S) {
    H0 = xt::zeros<double>({n_ao, n_ao});
    S = xt::zeros<double>({n_ao, n_ao});
    const auto images = lattice_images_for(box, el);

    // Onsite (T = 0, a = b): diagonal e_s / e_p and S = 1.
    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        H0(ia, ia) = ea.e_s;
        S(ia, ia) = 1.0;
        if (ea.n_ao == 4) {
            for (int p = 1; p < 4; ++p) {
                H0(ia + p, ia + p) = ea.e_p;
                S(ia + p, ia + p) = 1.0;
            }
        }
    }

    // Lattice sum over ordered pairs (a,b) and images T of B.
    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        const double xa = static_cast<double>(x[a]);
        const double ya = static_cast<double>(y[a]);
        const double za = static_cast<double>(z[a]);
        for (int b = 0; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            const int ib = ao0[static_cast<std::size_t>(b)];
            const double xb = static_cast<double>(x[b]);
            const double yb = static_cast<double>(y[b]);
            const double zb = static_cast<double>(z[b]);
            const double rcut = pair_rcut_A(ea, eb, type[a], type[b]);
            const auto& sk = pair_sk(type[a], type[b]);
            for (const auto& T : images) {
                const double dx = xb + T.tx - xa;
                const double dy = yb + T.ty - ya;
                const double dz = zb + T.tz - za;
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-12 || r > rcut)
                    continue;
                internal::MioSketchPair{ea, eb, sk}.add_block(dx, dy, dz, r, H0,
                                                              S, ia, ib);
            }
        }
    }
}

double repulsive_energy(int n_at, const int* type, const StorageReal* x,
                        const StorageReal* y, const StorageReal* z,
                        const cpu::Box& box,
                        const std::vector<const DftbElementRow*>& el) {
    const auto images = lattice_images_for(box, el);
    double e_rep = 0;
    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const double xa = static_cast<double>(x[a]);
        const double ya = static_cast<double>(y[a]);
        const double za = static_cast<double>(z[a]);
        for (int b = 0; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            const double xb = static_cast<double>(x[b]);
            const double yb = static_cast<double>(y[b]);
            const double zb = static_cast<double>(z[b]);
            const double rcut = pair_rcut_A(ea, eb, type[a], type[b]);
            for (const auto& T : images) {
                const double dx = xb + T.tx - xa;
                const double dy = yb + T.ty - ya;
                const double dz = zb + T.tz - za;
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-12 || r > rcut)
                    continue;
                // 1/2 for double-counting over ordered (a,b,T).
                e_rep += 0.5 * pair_repulsive_eV(type[a], type[b], r);
            }
        }
    }
    return e_rep;
}

void apply_shift(int n_at, const std::vector<const DftbElementRow*>& el,
                 const std::vector<int>& ao0, const linalg::RealMatrix& H0,
                 const linalg::RealMatrix& S, const std::vector<double>& eps_at,
                 linalg::RealMatrix& H) {
    H = H0;
    for (int a = 0; a < n_at; ++a) {
        const int ia = ao0[static_cast<std::size_t>(a)];
        const int na = el[static_cast<std::size_t>(a)]->n_ao;
        for (int b = 0; b < n_at; ++b) {
            const int ib = ao0[static_cast<std::size_t>(b)];
            const int nb = el[static_cast<std::size_t>(b)]->n_ao;
            const double half = 0.5
                                * (eps_at[static_cast<std::size_t>(a)]
                                   + eps_at[static_cast<std::size_t>(b)]);
            for (int i = 0; i < na; ++i)
                for (int j = 0; j < nb; ++j)
                    H(ia + i, ib + j) += half * S(ia + i, ib + j);
        }
    }
}

double gamma_pair(int a, int b, const std::vector<const DftbElementRow*>& el,
                  const StorageReal* x, const StorageReal* y,
                  const StorageReal* z, const cpu::Box& box) {
    const double ua = el[static_cast<std::size_t>(a)]->hubbard_u;
    const double ub = el[static_cast<std::size_t>(b)]->hubbard_u;
    const double rcut = pair_rcut_A(
        *el[static_cast<std::size_t>(a)], *el[static_cast<std::size_t>(b)],
        el[static_cast<std::size_t>(a)]->z, el[static_cast<std::size_t>(b)]->z);
    const auto images = lattice_images_for(box, el);
    const double xa = static_cast<double>(x[a]);
    const double ya = static_cast<double>(y[a]);
    const double za = static_cast<double>(z[a]);
    const double xb = static_cast<double>(x[b]);
    const double yb = static_cast<double>(y[b]);
    const double zb = static_cast<double>(z[b]);
    double g = 0;
    for (const auto& T : images) {
        const double dx = xb + T.tx - xa;
        const double dy = yb + T.ty - ya;
        const double dz = zb + T.tz - za;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (r < 1e-12) {
            if (a == b)
                g += ua; // onsite Hubbard
            continue;
        }
        if (r > rcut)
            continue;
        g += gamma_ab(r, ua, ub);
    }
    return g;
}

SccResult scc_energy_restricted(int n_at, int n_ao, int n_occ,
                                const std::vector<const DftbElementRow*>& el,
                                const std::vector<int>& ao0, const int* type,
                                const StorageReal* x, const StorageReal* y,
                                const StorageReal* z, const cpu::Box& box,
                                double scc_tol, int max_scc, double mix_beta) {
    linalg::RealMatrix H0, S;
    build_h0_s(n_at, n_ao, el, ao0, type, x, y, z, box, H0, S);

    const std::size_t n = static_cast<std::size_t>(n_at);
    std::vector<double> dq(n, 0.0);
    std::vector<double> dq_out(n, 0.0);
    linalg::RealMatrix P = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix C = xt::zeros<double>({n_ao, n_ao});
    linalg::RealVector eps = xt::zeros<double>({n_ao});
    double residual = 0;
    int iterations = 0;
    bool converged = false;
    internal::ChargeMixer mixer(mix_beta, kSccMixDepth);

    for (int it = 0; it < max_scc; ++it) {
        iterations = it + 1;
        std::vector<double> dV(n, 0.0);
        for (int a = 0; a < n_at; ++a) {
            double shift = 0;
            for (int b = 0; b < n_at; ++b)
                shift += gamma_pair(a, b, el, x, y, z, box)
                         * dq[static_cast<std::size_t>(b)];
            dV[static_cast<std::size_t>(a)] = shift;
        }
        linalg::RealMatrix H;
        apply_shift(n_at, el, ao0, H0, S, dV, H);
        auto sol = linalg::diagonalize_fock_generalized(H, S);
        eps = sol.first;
        C = sol.second;

        P = xt::zeros<double>({n_ao, n_ao});
        for (int o = 0; o < n_occ; ++o)
            for (int mu = 0; mu < n_ao; ++mu)
                for (int nu = 0; nu < n_ao; ++nu)
                    P(mu, nu) += 2.0 * C(mu, o) * C(nu, o);

        linalg::RealMatrix PS = xt::linalg::dot(P, S);
        residual = 0;
        for (int a = 0; a < n_at; ++a) {
            double qa = 0;
            const int ia = ao0[static_cast<std::size_t>(a)];
            const int na = el[static_cast<std::size_t>(a)]->n_ao;
            for (int i = 0; i < na; ++i)
                qa += PS(ia + i, ia + i);
            dq_out[static_cast<std::size_t>(a)] =
                qa - el[static_cast<std::size_t>(a)]->n_val;
            residual =
                std::max(residual, std::abs(dq_out[static_cast<std::size_t>(a)]
                                            - dq[static_cast<std::size_t>(a)]));
        }

        if (residual < scc_tol) {
            dq = dq_out;
            converged = true;
            break;
        }
        dq = mixer.next(dq, dq_out);
    }

    require_scc_converged(converged, residual, scc_tol, max_scc);

    double e_scc = 0;
    for (int a = 0; a < n_at; ++a)
        for (int b = 0; b < n_at; ++b)
            e_scc += 0.5 * gamma_pair(a, b, el, x, y, z, box)
                     * dq[static_cast<std::size_t>(a)]
                     * dq[static_cast<std::size_t>(b)];

    SccResult out;
    // E = Tr(ρH⁰) + E² + E_rep. NOT E_band − E²: the band energy is
    // Tr(ρH⁰) + Σ_A V_A q_A with the GROSS Mulliken population, so that route
    // leaves a spurious Σ_A V_A q⁰_A which is linear in Δq.
    out.energy = trace_product(P, H0) + e_scc
                 + repulsive_energy(n_at, type, x, y, z, box, el);
    out.residual = residual;
    out.iterations = iterations;
    out.dq = std::move(dq);
    out.spin.assign(static_cast<std::size_t>(n_at), 0.0);
    out.C = std::move(C);
    out.eps = std::move(eps);
    out.n_occ = n_occ;
    return out;
}

// ── Multi-k (Bloch) restricted SCC ──────────────────────────────────────

/// CP2K / DFTB+ Monkhorst–Pack: κ_d = (2r − N − 1)/(2N), r = 1..N.
std::vector<KPoint> monkhorst_pack_mesh(const int mesh[3], bool time_reversal) {
    for (int d = 0; d < 3; ++d)
        if (mesh[d] < 1)
            throw std::invalid_argument(
                "DFTB2: k_mesh subdivision must be >= 1 (axis "
                + std::to_string(d) + " got " + std::to_string(mesh[d]) + ")");

    std::vector<KPoint> full;
    const double w0 = 1.0 / (static_cast<double>(mesh[0]) * mesh[1] * mesh[2]);
    for (int r0 = 1; r0 <= mesh[0]; ++r0)
        for (int r1 = 1; r1 <= mesh[1]; ++r1)
            for (int r2 = 1; r2 <= mesh[2]; ++r2) {
                KPoint kp;
                const int r[3] = {r0, r1, r2};
                for (int d = 0; d < 3; ++d) {
                    double f = (2.0 * r[d] - mesh[d] - 1.0)
                               / (2.0 * static_cast<double>(mesh[d]));
                    f -= std::floor(f + 0.5);
                    if (f <= -0.5)
                        f += 1.0;
                    kp.frac[d] = f;
                }
                kp.weight = w0;
                full.push_back(kp);
            }

    if (!time_reversal)
        return full;

    std::vector<KPoint> folded;
    std::vector<bool> used(full.size(), false);
    constexpr double tol = 1e-12;
    auto same = [&](const double a[3], const double b[3]) {
        for (int d = 0; d < 3; ++d) {
            double diff = a[d] - b[d];
            diff -= std::nearbyint(diff);
            if (std::abs(diff) > tol)
                return false;
        }
        return true;
    };
    for (std::size_t i = 0; i < full.size(); ++i) {
        if (used[i])
            continue;
        used[i] = true;
        KPoint kp = full[i];
        const double neg[3] = {-kp.frac[0], -kp.frac[1], -kp.frac[2]};
        for (std::size_t j = i + 1; j < full.size(); ++j) {
            if (used[j])
                continue;
            if (same(full[j].frac, neg)) {
                used[j] = true;
                kp.weight += full[j].weight;
                break;
            }
        }
        folded.push_back(kp);
    }
    return folded;
}

Cplx bloch_phase(const double kappa[3], int n0, int n1, int n2) {
    constexpr double kTwoPi = 6.28318530717958647692;
    const double ang = kTwoPi * (kappa[0] * n0 + kappa[1] * n1 + kappa[2] * n2);
    return Cplx(std::cos(ang), std::sin(ang));
}

/// Row-major complex product C = A B.
CplxMatrix cplx_mul(const CplxMatrix& A, const CplxMatrix& B) {
    const int n = static_cast<int>(A.shape(0));
    const int m = static_cast<int>(A.shape(1));
    const int p = static_cast<int>(B.shape(1));
    CplxMatrix C = xt::zeros<Cplx>({n, p});
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < m; ++k) {
            const Cplx a = A(i, k);
            if (a == Cplx(0, 0))
                continue;
            for (int j = 0; j < p; ++j)
                C(i, j) += a * B(k, j);
        }
    return C;
}

/// Write directed real SK block scaled by complex phase into H0/S.
void fill_sk_block_phased(const DftbElementRow& ea, const DftbElementRow& eb,
                          const DftbPairSkRow& sk, double dx, double dy,
                          double dz, double r, const Cplx& phase,
                          CplxMatrix& H0, CplxMatrix& S, int ia0, int ib0) {
    // Stage into a tiny real scratch via the shared MioSketchPair kernel
    // (transcribed mio-1-1), then phase-multiply into complex H0/S.
    if (r < 1e-12)
        return;
    const int na = ea.n_ao;
    const int nb = eb.n_ao;
    linalg::RealMatrix h_scratch = xt::zeros<double>({na, nb});
    linalg::RealMatrix s_scratch = xt::zeros<double>({na, nb});
    // add_block indexes (ia0+i, ib0+j); stage at origin of the scratch.
    internal::MioSketchPair{ea, eb, sk}.add_block(dx, dy, dz, r, h_scratch,
                                                   s_scratch, 0, 0);
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j) {
            H0(ia0 + i, ib0 + j) += phase * h_scratch(i, j);
            S(ia0 + i, ib0 + j) += phase * s_scratch(i, j);
        }
}

void hermitize_in_place(CplxMatrix& M) {
    const int n = static_cast<int>(M.shape(0));
    for (int i = 0; i < n; ++i) {
        M(i, i) = Cplx(M(i, i).real(), 0.0);
        for (int j = i + 1; j < n; ++j) {
            const Cplx avg = 0.5 * (M(i, j) + std::conj(M(j, i)));
            M(i, j) = avg;
            M(j, i) = std::conj(avg);
        }
    }
}

/// Build lattice-summed complex H0(k)/S(k) for one fractional k-point.
void build_h0_s_k(int n_at, int n_ao,
                  const std::vector<const DftbElementRow*>& el,
                  const std::vector<int>& ao0, const int* type,
                  const StorageReal* x, const StorageReal* y,
                  const StorageReal* z, const cpu::Box& box,
                  const double kappa[3], CplxMatrix& H0, CplxMatrix& S) {
    H0 = xt::zeros<Cplx>({n_ao, n_ao});
    S = xt::zeros<Cplx>({n_ao, n_ao});
    const auto images = lattice_images_for(box, el);

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        H0(ia, ia) = Cplx(ea.e_s, 0.0);
        S(ia, ia) = Cplx(1.0, 0.0);
        if (ea.n_ao == 4) {
            for (int p = 1; p < 4; ++p) {
                H0(ia + p, ia + p) = Cplx(ea.e_p, 0.0);
                S(ia + p, ia + p) = Cplx(1.0, 0.0);
            }
        }
    }

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        const double xa = static_cast<double>(x[a]);
        const double ya = static_cast<double>(y[a]);
        const double za = static_cast<double>(z[a]);
        for (int b = 0; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            const int ib = ao0[static_cast<std::size_t>(b)];
            const double xb = static_cast<double>(x[b]);
            const double yb = static_cast<double>(y[b]);
            const double zb = static_cast<double>(z[b]);
            const double rcut = pair_rcut_A(ea, eb, type[a], type[b]);
            const auto& sk = pair_sk(type[a], type[b]);
            for (const auto& T : images) {
                const double dx = xb + T.tx - xa;
                const double dy = yb + T.ty - ya;
                const double dz = zb + T.tz - za;
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-12 || r > rcut)
                    continue;
                const Cplx phase = bloch_phase(kappa, T.n0, T.n1, T.n2);
                fill_sk_block_phased(ea, eb, sk, dx, dy, dz, r, phase, H0, S,
                                     ia, ib);
            }
        }
    }
    hermitize_in_place(H0);
    hermitize_in_place(S);
}

void apply_shift_k(int n_at, const std::vector<const DftbElementRow*>& el,
                   const std::vector<int>& ao0, const CplxMatrix& H0,
                   const CplxMatrix& S, const std::vector<double>& eps_at,
                   CplxMatrix& H) {
    H = H0;
    for (int a = 0; a < n_at; ++a) {
        const int ia = ao0[static_cast<std::size_t>(a)];
        const int na = el[static_cast<std::size_t>(a)]->n_ao;
        for (int b = 0; b < n_at; ++b) {
            const int ib = ao0[static_cast<std::size_t>(b)];
            const int nb = el[static_cast<std::size_t>(b)]->n_ao;
            const double half = 0.5
                                * (eps_at[static_cast<std::size_t>(a)]
                                   + eps_at[static_cast<std::size_t>(b)]);
            for (int i = 0; i < na; ++i)
                for (int j = 0; j < nb; ++j)
                    H(ia + i, ib + j) += half * S(ia + i, ib + j);
        }
    }
}

/// Generalized Hermitian eigen: H C = S C ε via Löwdin S^{-1/2} + zheevd.
std::pair<std::vector<double>, CplxMatrix>
diagonalize_fock_generalized_k(const CplxMatrix& H, const CplxMatrix& S) {
    const int n = static_cast<int>(H.shape(0));
    std::vector<double> s_eig;
    CplxMatrix Us;
    linalg::eigh_hermitian(S, s_eig, Us);
    // X = U s^{-1/2}
    CplxMatrix X = xt::zeros<Cplx>({n, n});
    for (int j = 0; j < n; ++j) {
        if (s_eig[static_cast<std::size_t>(j)] <= 1e-12)
            throw std::runtime_error(
                "DFTB2 multi-k: non-positive overlap eigenvalue at k");
        const double inv_sqrt =
            1.0 / std::sqrt(s_eig[static_cast<std::size_t>(j)]);
        for (int i = 0; i < n; ++i)
            X(i, j) = Us(i, j) * inv_sqrt;
    }
    // H' = X† H X
    CplxMatrix HX = cplx_mul(H, X);
    CplxMatrix Hp = xt::zeros<Cplx>({n, n});
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            Cplx s(0, 0);
            for (int k = 0; k < n; ++k)
                s += std::conj(X(k, i)) * HX(k, j);
            Hp(i, j) = s;
        }
    hermitize_in_place(Hp);
    std::vector<double> eps;
    CplxMatrix Cp;
    linalg::eigh_hermitian(Hp, eps, Cp);
    return {std::move(eps), cplx_mul(X, Cp)};
}

double re_trace_product_k(const CplxMatrix& P, const CplxMatrix& H0) {
    const int n = static_cast<int>(P.shape(0));
    double t = 0;
    for (int mu = 0; mu < n; ++mu)
        for (int nu = 0; nu < n; ++nu)
            t += (P(mu, nu) * H0(nu, mu)).real();
    return t;
}

SccResult scc_energy_restricted_kmesh(
    int n_at, int n_ao, int n_occ, const std::vector<const DftbElementRow*>& el,
    const std::vector<int>& ao0, const int* type, const StorageReal* x,
    const StorageReal* y, const StorageReal* z, const cpu::Box& box,
    double scc_tol, int max_scc, double mix_beta, const int mesh[3],
    bool time_reversal) {
    if (n_occ > n_ao)
        throw std::runtime_error(
            "DFTB2 multi-k: n_occ exceeds basis size (no fractional fill)");

    const std::vector<KPoint> kpts = monkhorst_pack_mesh(mesh, time_reversal);
    const std::size_t nk = kpts.size();
    std::vector<CplxMatrix> H0k(nk), Sk(nk);
    for (std::size_t ik = 0; ik < nk; ++ik)
        build_h0_s_k(n_at, n_ao, el, ao0, type, x, y, z, box, kpts[ik].frac,
                     H0k[ik], Sk[ik]);

    const std::size_t n = static_cast<std::size_t>(n_at);
    std::vector<double> dq(n, 0.0);
    std::vector<double> dq_out(n, 0.0);
    double residual = 0;
    int iterations = 0;
    bool converged = false;
    internal::ChargeMixer mixer(mix_beta, kSccMixDepth);

    // Last-iterate densities for the energy assembly after SCC; Wk is the
    // energy-weighted twin the Pulay (−∂S) force term contracts against.
    std::vector<CplxMatrix> Pk(nk);
    std::vector<CplxMatrix> Wk(nk);

    for (int it = 0; it < max_scc; ++it) {
        iterations = it + 1;
        std::vector<double> dV(n, 0.0);
        for (int a = 0; a < n_at; ++a) {
            double shift = 0;
            for (int b = 0; b < n_at; ++b)
                shift += gamma_pair(a, b, el, x, y, z, box)
                         * dq[static_cast<std::size_t>(b)];
            dV[static_cast<std::size_t>(a)] = shift;
        }

        // Weighted Mulliken populations across the mesh.
        std::vector<double> q_ao(static_cast<std::size_t>(n_ao), 0.0);
        for (std::size_t ik = 0; ik < nk; ++ik) {
            CplxMatrix Hk;
            apply_shift_k(n_at, el, ao0, H0k[ik], Sk[ik], dV, Hk);
            auto sol = diagonalize_fock_generalized_k(Hk, Sk[ik]);
            const auto& eps = sol.first;
            const CplxMatrix& C = sol.second;
            (void)eps;

            CplxMatrix P = xt::zeros<Cplx>({n_ao, n_ao});
            for (int o = 0; o < n_occ; ++o)
                for (int mu = 0; mu < n_ao; ++mu)
                    for (int nu = 0; nu < n_ao; ++nu)
                        P(mu, nu) += 2.0 * C(mu, o) * std::conj(C(nu, o));
            Pk[ik] = P;

            // PS = P S; Mulliken AO pop = Re((PS)_μμ)
            const CplxMatrix PS = cplx_mul(P, Sk[ik]);
            const double w = kpts[ik].weight;
            for (int mu = 0; mu < n_ao; ++mu)
                q_ao[static_cast<std::size_t>(mu)] += w * PS(mu, mu).real();
        }

        residual = 0;
        for (int a = 0; a < n_at; ++a) {
            double qa = 0;
            const int ia = ao0[static_cast<std::size_t>(a)];
            const int na = el[static_cast<std::size_t>(a)]->n_ao;
            for (int i = 0; i < na; ++i)
                qa += q_ao[static_cast<std::size_t>(ia + i)];
            dq_out[static_cast<std::size_t>(a)] =
                qa - el[static_cast<std::size_t>(a)]->n_val;
            residual =
                std::max(residual, std::abs(dq_out[static_cast<std::size_t>(a)]
                                            - dq[static_cast<std::size_t>(a)]));
        }

        if (residual < scc_tol) {
            dq = dq_out;
            converged = true;
            break;
        }
        dq = mixer.next(dq, dq_out);
    }

    require_scc_converged(converged, residual, scc_tol, max_scc);

    // Rebuild final densities from the converged dq (last Pk is from the
    // residual check with dq_out; re-evaluate once for energy consistency).
    {
        std::vector<double> dV(n, 0.0);
        for (int a = 0; a < n_at; ++a) {
            double shift = 0;
            for (int b = 0; b < n_at; ++b)
                shift += gamma_pair(a, b, el, x, y, z, box)
                         * dq[static_cast<std::size_t>(b)];
            dV[static_cast<std::size_t>(a)] = shift;
        }
        for (std::size_t ik = 0; ik < nk; ++ik) {
            CplxMatrix Hk;
            apply_shift_k(n_at, el, ao0, H0k[ik], Sk[ik], dV, Hk);
            auto sol = diagonalize_fock_generalized_k(Hk, Sk[ik]);
            const std::vector<double>& eps_k = sol.first;
            const CplxMatrix& C = sol.second;
            CplxMatrix P = xt::zeros<Cplx>({n_ao, n_ao});
            CplxMatrix W = xt::zeros<Cplx>({n_ao, n_ao});
            for (int o = 0; o < n_occ; ++o) {
                const double e_o = eps_k[static_cast<std::size_t>(o)];
                for (int mu = 0; mu < n_ao; ++mu)
                    for (int nu = 0; nu < n_ao; ++nu) {
                        const Cplx cc = C(mu, o) * std::conj(C(nu, o));
                        P(mu, nu) += 2.0 * cc;
                        W(mu, nu) += 2.0 * e_o * cc;
                    }
            }
            Pk[ik] = std::move(P);
            Wk[ik] = std::move(W);
        }
    }

    double e_band = 0;
    for (std::size_t ik = 0; ik < nk; ++ik)
        e_band += kpts[ik].weight * re_trace_product_k(Pk[ik], H0k[ik]);

    double e_scc = 0;
    for (int a = 0; a < n_at; ++a)
        for (int b = 0; b < n_at; ++b)
            e_scc += 0.5 * gamma_pair(a, b, el, x, y, z, box)
                     * dq[static_cast<std::size_t>(a)]
                     * dq[static_cast<std::size_t>(b)];

    SccResult out;
    out.energy =
        e_band + e_scc + repulsive_energy(n_at, type, x, y, z, box, el);
    out.residual = residual;
    out.iterations = iterations;
    out.dq = std::move(dq);
    out.spin.assign(static_cast<std::size_t>(n_at), 0.0);
    out.n_occ = n_occ;
    // No real Γ orbitals here — the multi-k analytic gradient consumes the
    // Bloch pair (P(k), W(k)) below instead.
    out.kpts = kpts;
    out.Pk = std::move(Pk);
    out.Wk = std::move(Wk);
    return out;
}

SccResult scc_energy_unrestricted(int n_at, int n_ao, int n_alpha, int n_beta,
                                  const std::vector<const DftbElementRow*>& el,
                                  const std::vector<int>& ao0, const int* type,
                                  const StorageReal* x, const StorageReal* y,
                                  const StorageReal* z, const cpu::Box& box,
                                  double scc_tol, int max_scc,
                                  double mix_beta) {
    if (n_alpha > n_ao || n_beta > n_ao)
        throw std::runtime_error("DFTB2: n_alpha/n_beta exceeds basis size");

    linalg::RealMatrix H0, S;
    build_h0_s(n_at, n_ao, el, ao0, type, x, y, z, box, H0, S);

    const std::size_t n = static_cast<std::size_t>(n_at);
    std::vector<double> dq(n, 0.0);
    std::vector<double> mspin(n, 0.0);
    std::vector<double> dq_out(n, 0.0);
    std::vector<double> mspin_out(n, 0.0);
    // Seed high-spin magnetization so the first iteration is not spin-null.
    if (n_alpha != n_beta) {
        const double m_tot = static_cast<double>(n_alpha - n_beta);
        for (int a = 0; a < n_at; ++a)
            mspin[static_cast<std::size_t>(a)] =
                m_tot / static_cast<double>(n_at);
    }

    double e_scc = 0, e_spin = 0, residual = 0;
    int iterations = 0;
    bool converged = false;
    // dq and m are coupled by the same density: mix them as one vector.
    internal::ChargeMixer mixer(mix_beta, kSccMixDepth);
    linalg::RealMatrix Pa = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix Pb = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix CA = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix CB = xt::zeros<double>({n_ao, n_ao});
    linalg::RealVector epsA = xt::zeros<double>({n_ao});
    linalg::RealVector epsB = xt::zeros<double>({n_ao});

    for (int it = 0; it < max_scc; ++it) {
        iterations = it + 1;
        std::vector<double> dVc(static_cast<std::size_t>(n_at), 0.0);
        std::vector<double> dVs(static_cast<std::size_t>(n_at), 0.0);
        for (int a = 0; a < n_at; ++a) {
            double shift = 0;
            for (int b = 0; b < n_at; ++b)
                shift += gamma_pair(a, b, el, x, y, z, box)
                         * dq[static_cast<std::size_t>(b)];
            dVc[static_cast<std::size_t>(a)] = shift;
            dVs[static_cast<std::size_t>(a)] =
                el[static_cast<std::size_t>(a)]->spin_w
                * mspin[static_cast<std::size_t>(a)];
        }

        // ε^σ_A = dVc + σ dVs  (σ = +1 α, −1 β)
        std::vector<double> eps_shift_a(static_cast<std::size_t>(n_at));
        std::vector<double> eps_shift_b(static_cast<std::size_t>(n_at));
        for (int a = 0; a < n_at; ++a) {
            eps_shift_a[static_cast<std::size_t>(a)] =
                dVc[static_cast<std::size_t>(a)]
                + dVs[static_cast<std::size_t>(a)];
            eps_shift_b[static_cast<std::size_t>(a)] =
                dVc[static_cast<std::size_t>(a)]
                - dVs[static_cast<std::size_t>(a)];
        }

        linalg::RealMatrix Ha, Hb;
        apply_shift(n_at, el, ao0, H0, S, eps_shift_a, Ha);
        apply_shift(n_at, el, ao0, H0, S, eps_shift_b, Hb);

        auto sol_a = linalg::diagonalize_fock_generalized(Ha, S);
        auto sol_b = linalg::diagonalize_fock_generalized(Hb, S);
        epsA = sol_a.first;
        CA = sol_a.second;
        epsB = sol_b.first;
        CB = sol_b.second;

        Pa = xt::zeros<double>({n_ao, n_ao});
        Pb = xt::zeros<double>({n_ao, n_ao});
        for (int o = 0; o < n_alpha; ++o)
            for (int mu = 0; mu < n_ao; ++mu)
                for (int nu = 0; nu < n_ao; ++nu)
                    Pa(mu, nu) += CA(mu, o) * CA(nu, o);
        for (int o = 0; o < n_beta; ++o)
            for (int mu = 0; mu < n_ao; ++mu)
                for (int nu = 0; nu < n_ao; ++nu)
                    Pb(mu, nu) += CB(mu, o) * CB(nu, o);

        linalg::RealMatrix PSa = xt::linalg::dot(Pa, S);
        linalg::RealMatrix PSb = xt::linalg::dot(Pb, S);

        residual = 0;
        for (int a = 0; a < n_at; ++a) {
            double qa = 0, qb = 0;
            const int ia = ao0[static_cast<std::size_t>(a)];
            const int na = el[static_cast<std::size_t>(a)]->n_ao;
            for (int i = 0; i < na; ++i) {
                qa += PSa(ia + i, ia + i);
                qb += PSb(ia + i, ia + i);
            }
            dq_out[static_cast<std::size_t>(a)] =
                (qa + qb) - el[static_cast<std::size_t>(a)]->n_val;
            mspin_out[static_cast<std::size_t>(a)] = qa - qb;
            residual =
                std::max(residual, std::abs(dq_out[static_cast<std::size_t>(a)]
                                            - dq[static_cast<std::size_t>(a)]));
            residual = std::max(residual,
                                std::abs(mspin_out[static_cast<std::size_t>(a)]
                                         - mspin[static_cast<std::size_t>(a)]));
        }

        if (residual < scc_tol) {
            dq = dq_out;
            mspin = mspin_out;
            converged = true;
            break;
        }
        std::vector<double> in(dq);
        in.insert(in.end(), mspin.begin(), mspin.end());
        std::vector<double> out_vec(dq_out);
        out_vec.insert(out_vec.end(), mspin_out.begin(), mspin_out.end());
        const std::vector<double> mixed = mixer.next(in, out_vec);
        for (std::size_t a = 0; a < n; ++a) {
            dq[a] = mixed[a];
            mspin[a] = mixed[n + a];
        }
    }

    require_scc_converged(converged, residual, scc_tol, max_scc);

    e_scc = 0;
    for (int a = 0; a < n_at; ++a)
        for (int b = 0; b < n_at; ++b)
            e_scc += 0.5 * gamma_pair(a, b, el, x, y, z, box)
                     * dq[static_cast<std::size_t>(a)]
                     * dq[static_cast<std::size_t>(b)];
    e_spin = 0;
    for (int a = 0; a < n_at; ++a)
        e_spin += 0.5 * el[static_cast<std::size_t>(a)]->spin_w
                  * mspin[static_cast<std::size_t>(a)]
                  * mspin[static_cast<std::size_t>(a)];

    SccResult out;
    // E = Tr(ρH⁰) + E² + E_spin + E_rep (same convention as the restricted
    // path): Tr(ρ^σ H^σ) carries Σ_A V_A q_A on the GROSS population.
    out.energy = trace_product(Pa, H0) + trace_product(Pb, H0) + e_scc + e_spin
                 + repulsive_energy(n_at, type, x, y, z, box, el);
    out.residual = residual;
    out.iterations = iterations;
    out.dq = std::move(dq);
    out.spin = std::move(mspin);
    out.Ca = std::move(CA);
    out.Cb = std::move(CB);
    out.eps_a = std::move(epsA);
    out.eps_b = std::move(epsB);
    out.n_alpha = n_alpha;
    out.n_beta = n_beta;
    out.unrestricted = true;
    return out;
}


/// Closed-form nuclear gradient (+ optional intensive cell stress) for
/// **collinear spin-resolved** SCC-DFTB2 on the MioSketch tables.
///
/// ONE assembly serves both spin cases: the restricted closed shell is the
/// degenerate limit \(C^\alpha=C^\beta\), \(\varepsilon^\alpha=\varepsilon^\beta\),
/// \(n_\alpha=n_\beta=n_{\mathrm{occ}}\), \(m\equiv 0\) — i.e.
/// \(P^\alpha=P^\beta=P/2\), \(W^\alpha=W^\beta=W/2\). There is deliberately no
/// second copy of this contraction: the wrong-sign \(\gamma\) gradient of
/// 2026-07-28 survived precisely because it was written out three times.
///
/// Forces: \(F=-\nabla E\) [eV/Å]. Stress: \(\sigma=-(1/\Omega)\partial
/// E/\partial\varepsilon\) [eV/Å³] Voigt (xx,yy,zz,xy,xz,yz) — the pair virial
/// of the very same bond terms, since affine strain maps every bond
/// \(R\to(1+\varepsilon)R\).
///
/// Per ordered pair \(a<b\) and lattice image \(T\), bond \(R=R_b+T-R_a\):
///
/// | term | contribution to \(\partial E/\partial R\) |
/// | ---- | ----------------------------------------- |
/// | Hellmann–Feynman | \(\sum_\sigma P^\sigma\,\partial H^0/\partial R\) |
/// | Pulay | \(\sum_\sigma[P^\sigma\tfrac12(\varepsilon^\sigma_a+\varepsilon^\sigma_b)-W^\sigma]\,\partial S/\partial R\) |
/// | SCC \(\gamma\) | \(\Delta q_a\Delta q_b\,\partial\gamma_{ab}/\partial R\) (**total** Mulliken \(\Delta q\)) |
/// | repulsive | \(\partial E_{\mathrm{rep}}/\partial R\) |
///
/// with \(\varepsilon^\sigma_A=\sum_B\gamma_{AB}\Delta q_B+\sigma W_A m_A\) and
/// \(W^\sigma_{\mu\nu}=\sum_{o\in\mathrm{occ}\,\sigma}\varepsilon^\sigma_o
/// C^\sigma_{\mu o}C^\sigma_{\nu o}\).
///
/// **The spin polarization sits in the Pulay slot — it does not vanish.**
/// \(E_{\mathrm{spin}}=\tfrac12\sum_A W_A m_A^2\) carries no *explicit*
/// geometry dependence (\(W_A\) is a constant atomic parameter and \(m_A\) is
/// held at its SCC value), but \(m_A\) depends on geometry through \(S\).
/// Stationarity of the SCC functional (\(\partial E/\partial P^\sigma=H^\sigma\),
/// \(C^{\sigma\mathsf T}SC^\sigma=I\)) converts that response into exactly the
/// \(\pm W_A m_A\) half of \(\varepsilon^\sigma_A\) multiplying
/// \(\partial S/\partial R\) in the table. Replacing \(\varepsilon^\sigma\) by
/// the charge-only shift breaks both \(F=-\nabla E\) and the virial for every
/// system whose spin shift varies across atoms. (It is *uniform* for
/// homonuclear O₂ — a uniform \(cS\) shift leaves the generalized eigenvectors
/// invariant — which is why that case cannot detect the omission.)
///
/// **Γ-PBC:** walks the same lattice images as `build_h0_s` / `gamma_pair` /
/// `repulsive_energy`, not min-image. Self-images (\(a=a\), \(T\neq0\)) exert
/// zero nuclear force and are skipped; in a dense periodic cell they do carry
/// a virial — an open gap shared with the shipped restricted path, recorded in
/// `regressions/semiempirical/dftb2-analytic-stress/REFERENCE.md`.
void analytic_forces_stress(
    int n_at, int n_ao, int n_alpha, int n_beta,
    const std::vector<const DftbElementRow*>& el, const std::vector<int>& ao0,
    const int* type, const StorageReal* x, const StorageReal* y,
    const StorageReal* z, const cpu::Box& box, const std::vector<double>& dq,
    const std::vector<double>& mspin, const linalg::RealMatrix& Ca,
    const linalg::RealMatrix& Cb, const linalg::RealVector& eps_a_orb,
    const linalg::RealVector& eps_b_orb, AccReal* fx, AccReal* fy, AccReal* fz,
    bool want_forces, double* stress_voigt /* nullable, length 6 */) {
    // ε^σ_A = Σ_B γ_AB Δq_B ± W_A m_A (γ_AB already lattice-summed).
    std::vector<double> eps_at_a(static_cast<std::size_t>(n_at), 0.0);
    std::vector<double> eps_at_b(static_cast<std::size_t>(n_at), 0.0);
    for (int a = 0; a < n_at; ++a) {
        double shift = 0;
        for (int b = 0; b < n_at; ++b)
            shift += gamma_pair(a, b, el, x, y, z, box)
                     * dq[static_cast<std::size_t>(b)];
        const double vs = el[static_cast<std::size_t>(a)]->spin_w
                          * mspin[static_cast<std::size_t>(a)];
        eps_at_a[static_cast<std::size_t>(a)] = shift + vs;
        eps_at_b[static_cast<std::size_t>(a)] = shift - vs;
    }

    linalg::RealMatrix Pa = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix Pb = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix Wa = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix Wb = xt::zeros<double>({n_ao, n_ao});
    for (int o = 0; o < n_alpha; ++o)
        for (int mu = 0; mu < n_ao; ++mu)
            for (int nu = 0; nu < n_ao; ++nu) {
                Pa(mu, nu) += Ca(mu, o) * Ca(nu, o);
                Wa(mu, nu) += eps_a_orb(o) * Ca(mu, o) * Ca(nu, o);
            }
    for (int o = 0; o < n_beta; ++o)
        for (int mu = 0; mu < n_ao; ++mu)
            for (int nu = 0; nu < n_ao; ++nu) {
                Pb(mu, nu) += Cb(mu, o) * Cb(nu, o);
                Wb(mu, nu) += eps_b_orb(o) * Cb(mu, o) * Cb(nu, o);
            }

    linalg::RealMatrix dH0[3] = {xt::zeros<double>({n_ao, n_ao}),
                                 xt::zeros<double>({n_ao, n_ao}),
                                 xt::zeros<double>({n_ao, n_ao})};
    linalg::RealMatrix dS[3] = {xt::zeros<double>({n_ao, n_ao}),
                                xt::zeros<double>({n_ao, n_ao}),
                                xt::zeros<double>({n_ao, n_ao})};

    // Accumulate ∂E/∂ε (before −1/Ω) in Voigt order.
    double dE_de[6] = {0, 0, 0, 0, 0, 0};

    const auto images = lattice_images_for(box, el);

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        const double xa = static_cast<double>(x[a]);
        const double ya = static_cast<double>(y[a]);
        const double za = static_cast<double>(z[a]);
        for (int b = a + 1; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            const int ib = ao0[static_cast<std::size_t>(b)];
            const double xb = static_cast<double>(x[b]);
            const double yb = static_cast<double>(y[b]);
            const double zb = static_cast<double>(z[b]);
            const double rcut = pair_rcut_A(ea, eb, type[a], type[b]);
            const auto& sk = pair_sk(type[a], type[b]);
            const double dqa = dq[static_cast<std::size_t>(a)];
            const double dqb = dq[static_cast<std::size_t>(b)];
            const double shift_a = 0.5
                                   * (eps_at_a[static_cast<std::size_t>(a)]
                                      + eps_at_a[static_cast<std::size_t>(b)]);
            const double shift_b = 0.5
                                   * (eps_at_b[static_cast<std::size_t>(a)]
                                      + eps_at_b[static_cast<std::size_t>(b)]);
            const int na = ea.n_ao;
            const int nb = eb.n_ao;

            for (const auto& T : images) {
                const double dx = xb + T.tx - xa;
                const double dy = yb + T.ty - ya;
                const double dz = zb + T.tz - za;
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-12)
                    continue;
                const double dir[3] = {dx / r, dy / r, dz / r};
                // ∂E/∂bond for this image = the force on atom a (F_b = −F_a).
                double F_a[3] = {0, 0, 0};

                if (r <= rcut) {
                    for (int k = 0; k < 3; ++k) {
                        dH0[k] = xt::zeros<double>({n_ao, n_ao});
                        dS[k] = xt::zeros<double>({n_ao, n_ao});
                    }
                    internal::MioSketchPair{ea, eb, sk}.set_block_derivative(
                        dx, dy, dz, r, dH0, dS, ia, ib);
                    for (int k = 0; k < 3; ++k) {
                        double acc = 0;
                        for (int i = 0; i < na; ++i)
                            for (int j = 0; j < nb; ++j) {
                                const double dh = dH0[k](ia + i, ib + j);
                                const double ds = dS[k](ia + i, ib + j);
                                acc += Pa(ia + i, ib + j) * (dh + shift_a * ds)
                                       - Wa(ia + i, ib + j) * ds;
                                acc += Pb(ia + i, ib + j) * (dh + shift_b * ds)
                                       - Wb(ia + i, ib + j) * ds;
                            }
                        // Factor 2: only the A–B upper block is contracted.
                        F_a[k] += 2.0 * acc;
                    }

                    // Lattice γ: each image of γ(|R+T|) contributes
                    // independently. E² = ½Σ_AB γ_AB Δq_A Δq_B with
                    // γ_AB = Σ_T γ(|R+T|); the a<b pair term is
                    // γ_AB Δq_A Δq_B, so ∂E²/∂bond_T = Δq_A Δq_B dγ/dr.
                    const double dgdR =
                        dgamma_dr(r, ea.hubbard_u, eb.hubbard_u);
                    for (int k = 0; k < 3; ++k)
                        F_a[k] += dqa * dqb * dgdR * dir[k];

                    const double dErep_dr =
                        pair_repulsive_dE_dr(type[a], type[b], r);
                    for (int k = 0; k < 3; ++k)
                        F_a[k] += dErep_dr * dir[k];
                }

                if (want_forces && fx && fy && fz) {
                    fx[a] += static_cast<AccReal>(F_a[0]);
                    fy[a] += static_cast<AccReal>(F_a[1]);
                    fz[a] += static_cast<AccReal>(F_a[2]);
                    fx[b] -= static_cast<AccReal>(F_a[0]);
                    fy[b] -= static_cast<AccReal>(F_a[1]);
                    fz[b] -= static_cast<AccReal>(F_a[2]);
                }

                if (stress_voigt) {
                    const double R[3] = {dx, dy, dz};
                    dE_de[0] += F_a[0] * R[0];
                    dE_de[1] += F_a[1] * R[1];
                    dE_de[2] += F_a[2] * R[2];
                    dE_de[3] += F_a[0] * R[1] + F_a[1] * R[0]; // xy
                    dE_de[4] += F_a[0] * R[2] + F_a[2] * R[0]; // xz
                    dE_de[5] += F_a[1] * R[2] + F_a[2] * R[1]; // yz
                }
            }
        }
    }

    // Self-images (A, A+T): the bond is T alone, independent of every nuclear
    // coordinate, so they carry **no force** — but |T| scales with the cell, so
    // the virial must carry them. Omitting them is invisible for a molecule in
    // a large box (every image is beyond r_cut) and wrong for any cell whose
    // own images sit inside it.
    if (stress_voigt) {
        for (int a = 0; a < n_at; ++a) {
            const auto& ea = *el[static_cast<std::size_t>(a)];
            const int ia = ao0[static_cast<std::size_t>(a)];
            const int na = ea.n_ao;
            const double rcut = pair_rcut_A(ea, ea, type[a], type[a]);
            const auto& sk = pair_sk(type[a], type[a]);
            const double dqa = dq[static_cast<std::size_t>(a)];
            // ½(V_A + V_A) — the same Mulliken split as the a<b blocks.
            // ½(ε_A + ε_A) = ε_A — the same Mulliken split as the a<b
            // blocks, spin-resolved like them.
            const double shift_a = eps_at_a[static_cast<std::size_t>(a)];
            const double shift_b = eps_at_b[static_cast<std::size_t>(a)];
            for (const auto& T : images) {
                const double r =
                    std::sqrt(T.tx * T.tx + T.ty * T.ty + T.tz * T.tz);
                if (r < 1e-12 || r > rcut)
                    continue;
                const double dir[3] = {T.tx / r, T.ty / r, T.tz / r};
                for (int k = 0; k < 3; ++k) {
                    dH0[k] = xt::zeros<double>({n_ao, n_ao});
                    dS[k] = xt::zeros<double>({n_ao, n_ao});
                }
                internal::MioSketchPair{ea, ea, sk}.set_block_derivative(
                    T.tx, T.ty, T.tz, r, dH0, dS, ia, ia);
                // Factor 1: the image walk already visits both T and −T, and
                // the (a,a) block is contracted whole, not upper-triangle only.
                // γ_AA and E_rep are each half-weighted per ordered image.
                double dE_dT[3] = {0, 0, 0};
                for (int k = 0; k < 3; ++k) {
                    double acc = 0;
                    for (int i = 0; i < na; ++i)
                        for (int j = 0; j < na; ++j) {
                            const double dh = dH0[k](ia + i, ia + j);
                            const double ds = dS[k](ia + i, ia + j);
                            acc += Pa(ia + i, ia + j) * (dh + shift_a * ds)
                                   - Wa(ia + i, ia + j) * ds;
                            acc += Pb(ia + i, ia + j) * (dh + shift_b * ds)
                                   - Wb(ia + i, ia + j) * ds;
                        }
                    dE_dT[k] = acc;
                }
                const double half_pair =
                    0.5 * dqa * dqa * dgamma_dr(r, ea.hubbard_u, ea.hubbard_u)
                    + 0.5 * pair_repulsive_dE_dr(type[a], type[a], r);
                for (int k = 0; k < 3; ++k)
                    dE_dT[k] += half_pair * dir[k];

                const double R[3] = {T.tx, T.ty, T.tz};
                dE_de[0] += dE_dT[0] * R[0];
                dE_de[1] += dE_dT[1] * R[1];
                dE_de[2] += dE_dT[2] * R[2];
                dE_de[3] += dE_dT[0] * R[1] + dE_dT[1] * R[0]; // xy
                dE_de[4] += dE_dT[0] * R[2] + dE_dT[2] * R[0]; // xz
                dE_de[5] += dE_dT[1] * R[2] + dE_dT[2] * R[1]; // yz
            }
        }
    }

    if (stress_voigt) {
        const double vol = box.volume();
        if (vol <= 0.0)
            throw std::runtime_error(
                "DFTB2: non-positive box volume for stress");
        for (int v = 0; v < 6; ++v)
            stress_voigt[v] = -dE_de[v] / vol;
    }
}

/// Closed-form nuclear gradient for **restricted multi-k** SCC-DFTB2 at fixed
/// Δq and fixed Bloch densities. F = −∇E [eV/Å].
///
/// Elstner et al., Phys. Rev. B **58**, 7260 (1998) Eq. (15) carried to k ≠ Γ
/// (DFTB+ periodic force module, Hourahine et al., J. Chem. Phys. **152**,
/// 124101 (2020)). Substituting the Bloch sum
/// H⁰(k)_{μa,νb} = Σ_T e^{2πi κ·n_T} h_{μν}(R_b + T − R_a) into
/// E_band = Σ_k w_k Re Tr[P(k)H⁰(k)] and using P(k)† = P(k) collapses the band
/// energy onto **real** image-resolved blocks
///
///   P̃_ij(a,b,T) = Σ_k w_k Re[ P(k)_{i_a,j_b} e^{−2πi κ·n_T} ]
///
/// (and W̃ from W(k)), so each (a<b, T) pair contributes the *same* pair form
/// as the Γ path with 2P → P̃ and 2W → W̃. At Γ (one k, w = 1, κ = 0) this is
/// an identity, not an approximation.
///
/// **Reality of the k-sum.** P(−k) = conj P(k) for a real real-space
/// Hamiltonian and e^{−iθ(−k)} = conj e^{−iθ(k)}, so a ±k pair contributes
/// w·Re(·) exactly. With `time_reversal` folding, each non-self-paired entry
/// already stands for its conjugate partner — nothing is left to cancel. The
/// two cases where the imaginary part must vanish on its own are checked and
/// throw: an unfolded (full) mesh, and self-paired k (κ_d ∈ {0, ±½}).
///
/// Self-images (a = a + T, T ≠ 0) have bond T, independent of every nuclear
/// coordinate, hence zero force — but |T| scales with the cell, so they are
/// walked separately for the virial.
///
/// The k-mesh is defined in **fractional** coordinates, so an affine strain
/// leaves every κ (and every Bloch phase) untouched and enters only through
/// the image bonds: σ = −(1/Ω) Σ (∂E/∂d_α) d_β over the same terms.
///
/// @param dq            Converged k-integrated Mulliken Δq_A [e].
/// @param kpts          Monkhorst–Pack points + weights (weights sum to 1).
/// @param Pk            P(k) per k-point (AO×AO, Hermitian) [e].
/// @param Wk            W(k) = 2Σ_o ε_o C C† per k-point [eV].
/// @param time_reversal True when `kpts` is the ±k-folded mesh.
/// @param fx,fy,fz      Per-atom force accumulators [eV/Å]; F is **added**.
/// @param want_forces   Skip the force write (stress-only callers).
/// @param stress_voigt  Nullable 6-vector; intensive σ [eV/Å³] Voigt
///                      (xx,yy,zz,xy,xz,yz), **assigned** not accumulated.
void analytic_forces_restricted_kmesh(
    int n_at, int n_ao, const std::vector<const DftbElementRow*>& el,
    const std::vector<int>& ao0, const int* type, const StorageReal* x,
    const StorageReal* y, const StorageReal* z, const cpu::Box& box,
    const std::vector<double>& dq, const std::vector<KPoint>& kpts,
    const std::vector<CplxMatrix>& Pk, const std::vector<CplxMatrix>& Wk,
    bool time_reversal, AccReal* fx, AccReal* fy, AccReal* fz,
    bool want_forces, double* stress_voigt /* nullable, length 6 */,
    // Optional collinear open-shell dual channel (empty ⇒ restricted).
    const std::vector<double>* spin = nullptr,
    const std::vector<CplxMatrix>* Pak = nullptr,
    const std::vector<CplxMatrix>* Pbk = nullptr,
    const std::vector<CplxMatrix>* Wak = nullptr,
    const std::vector<CplxMatrix>* Wbk = nullptr) {
    const std::size_t nk = kpts.size();
    if (nk == 0 || Pk.size() != nk || Wk.size() != nk)
        throw std::runtime_error(
            "DFTB2 multi-k: missing converged Bloch densities for the gradient");
    const bool open_shell =
        spin != nullptr && Pak != nullptr && Pbk != nullptr && Wak != nullptr
        && Wbk != nullptr && !Pak->empty();
    if (open_shell
        && (Pak->size() != nk || Pbk->size() != nk || Wak->size() != nk
            || Wbk->size() != nk || spin->size() != static_cast<std::size_t>(n_at)))
        throw std::runtime_error(
            "DFTB2 multi-k open shell: dual Bloch densities size mismatch");

    // ε_A = Σ_B γ_AB Δq_B — k-independent (real-space lattice-summed γ).
    std::vector<double> dV(static_cast<std::size_t>(n_at), 0.0);
    std::vector<double> dVa(static_cast<std::size_t>(n_at), 0.0);
    std::vector<double> dVb(static_cast<std::size_t>(n_at), 0.0);
    for (int a = 0; a < n_at; ++a) {
        double shift = 0;
        for (int b = 0; b < n_at; ++b)
            shift += gamma_pair(a, b, el, x, y, z, box)
                     * dq[static_cast<std::size_t>(b)];
        dV[static_cast<std::size_t>(a)] = shift;
        if (open_shell) {
            const double vs = el[static_cast<std::size_t>(a)]->spin_w
                              * (*spin)[static_cast<std::size_t>(a)];
            dVa[static_cast<std::size_t>(a)] = shift + vs;
            dVb[static_cast<std::size_t>(a)] = shift - vs;
        }
    }

    // κ ≡ −κ (mod 1) ⇔ every 2κ_d is an integer: those points have no
    // time-reversal partner, so their own imaginary part must vanish.
    std::vector<char> self_paired(nk, 0);
    for (std::size_t ik = 0; ik < nk; ++ik) {
        bool sp = true;
        for (int d = 0; d < 3; ++d) {
            const double two_k = 2.0 * kpts[ik].frac[d];
            sp = sp && std::abs(two_k - std::nearbyint(two_k)) < 1e-12;
        }
        self_paired[ik] = sp ? 1 : 0;
    }

    linalg::RealMatrix dH0[3] = {xt::zeros<double>({n_ao, n_ao}),
                                 xt::zeros<double>({n_ao, n_ao}),
                                 xt::zeros<double>({n_ao, n_ao})};
    linalg::RealMatrix dS[3] = {xt::zeros<double>({n_ao, n_ao}),
                                xt::zeros<double>({n_ao, n_ao}),
                                xt::zeros<double>({n_ao, n_ao})};

    const auto images = lattice_images_for(box, el);
    std::vector<Cplx> conj_phase(nk);
    double max_imag = 0;
    // ∂E/∂ε before the −1/Ω, in Voigt order.
    double dE_de[6] = {0, 0, 0, 0, 0, 0};

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        const double xa = static_cast<double>(x[a]);
        const double ya = static_cast<double>(y[a]);
        const double za = static_cast<double>(z[a]);
        for (int b = a + 1; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            const int ib = ao0[static_cast<std::size_t>(b)];
            const double xb = static_cast<double>(x[b]);
            const double yb = static_cast<double>(y[b]);
            const double zb = static_cast<double>(z[b]);
            const double rcut = pair_rcut_A(ea, eb, type[a], type[b]);
            const auto& sk = pair_sk(type[a], type[b]);
            const double dqa = dq[static_cast<std::size_t>(a)];
            const double dqb = dq[static_cast<std::size_t>(b)];
            const double shift = 0.5 * (dV[static_cast<std::size_t>(a)]
                                        + dV[static_cast<std::size_t>(b)]);
            const double shift_a =
                open_shell ? 0.5
                                 * (dVa[static_cast<std::size_t>(a)]
                                    + dVa[static_cast<std::size_t>(b)])
                           : shift;
            const double shift_b =
                open_shell ? 0.5
                                 * (dVb[static_cast<std::size_t>(a)]
                                    + dVb[static_cast<std::size_t>(b)])
                           : shift;
            const int na = ea.n_ao;
            const int nb = eb.n_ao;

            for (const auto& T : images) {
                const double dx = xb + T.tx - xa;
                const double dy = yb + T.ty - ya;
                const double dz = zb + T.tz - za;
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-12 || r > rcut)
                    continue;
                const double dir[3] = {dx / r, dy / r, dz / r};
                double F_a[3] = {0, 0, 0};

                for (std::size_t ik = 0; ik < nk; ++ik)
                    conj_phase[ik] = std::conj(
                        bloch_phase(kpts[ik].frac, T.n0, T.n1, T.n2));

                for (int k = 0; k < 3; ++k) {
                    dH0[k] = xt::zeros<double>({n_ao, n_ao});
                    dS[k] = xt::zeros<double>({n_ao, n_ao});
                }
                internal::MioSketchPair{ea, eb, sk}.set_block_derivative(
                    dx, dy, dz, r, dH0, dS, ia, ib);

                double acc[3] = {0, 0, 0};
                for (int i = 0; i < na; ++i) {
                    for (int j = 0; j < nb; ++j) {
                        double p_re = 0, w_re = 0, p_im = 0, w_im = 0;
                        double pa_re = 0, pb_re = 0, wa_re = 0, wb_re = 0;
                        for (std::size_t ik = 0; ik < nk; ++ik) {
                            const double wk = kpts[ik].weight;
                            if (open_shell) {
                                const Cplx zpa =
                                    (*Pak)[ik](ia + i, ib + j) * conj_phase[ik];
                                const Cplx zpb =
                                    (*Pbk)[ik](ia + i, ib + j) * conj_phase[ik];
                                const Cplx zwa =
                                    (*Wak)[ik](ia + i, ib + j) * conj_phase[ik];
                                const Cplx zwb =
                                    (*Wbk)[ik](ia + i, ib + j) * conj_phase[ik];
                                pa_re += wk * zpa.real();
                                pb_re += wk * zpb.real();
                                wa_re += wk * zwa.real();
                                wb_re += wk * zwb.real();
                                if (!time_reversal || self_paired[ik]) {
                                    p_im += wk * (zpa.imag() + zpb.imag());
                                    w_im += wk * (zwa.imag() + zwb.imag());
                                }
                            } else {
                                const Cplx zp = Pk[ik](ia + i, ib + j)
                                                * conj_phase[ik];
                                const Cplx zw = Wk[ik](ia + i, ib + j)
                                                * conj_phase[ik];
                                p_re += wk * zp.real();
                                w_re += wk * zw.real();
                                if (!time_reversal || self_paired[ik]) {
                                    p_im += wk * zp.imag();
                                    w_im += wk * zw.imag();
                                }
                            }
                        }
                        max_imag = std::max(
                            max_imag, std::max(std::abs(p_im), std::abs(w_im)));
                        for (int k = 0; k < 3; ++k) {
                            const double dh = dH0[k](ia + i, ib + j);
                            const double ds = dS[k](ia + i, ib + j);
                            if (open_shell)
                                acc[k] += pa_re * (dh + shift_a * ds)
                                          - wa_re * ds
                                          + pb_re * (dh + shift_b * ds)
                                          - wb_re * ds;
                            else
                                acc[k] +=
                                    p_re * (dh + shift * ds) - w_re * ds;
                        }
                    }
                }
                // Factor 2: the (b,a,−T) ordered twin of this block.
                for (int k = 0; k < 3; ++k)
                    F_a[k] += 2.0 * acc[k];

                const double dgdR = dgamma_dr(r, ea.hubbard_u, eb.hubbard_u);
                const double dErep_dr =
                    pair_repulsive_dE_dr(type[a], type[b], r);
                for (int k = 0; k < 3; ++k)
                    F_a[k] += (dqa * dqb * dgdR + dErep_dr) * dir[k];

                if (want_forces && fx && fy && fz) {
                    fx[a] += static_cast<AccReal>(F_a[0]);
                    fy[a] += static_cast<AccReal>(F_a[1]);
                    fz[a] += static_cast<AccReal>(F_a[2]);
                    fx[b] -= static_cast<AccReal>(F_a[0]);
                    fy[b] -= static_cast<AccReal>(F_a[1]);
                    fz[b] -= static_cast<AccReal>(F_a[2]);
                }

                if (stress_voigt) {
                    const double R[3] = {dx, dy, dz};
                    dE_de[0] += F_a[0] * R[0];
                    dE_de[1] += F_a[1] * R[1];
                    dE_de[2] += F_a[2] * R[2];
                    dE_de[3] += F_a[0] * R[1] + F_a[1] * R[0]; // xy
                    dE_de[4] += F_a[0] * R[2] + F_a[2] * R[0]; // xz
                    dE_de[5] += F_a[1] * R[2] + F_a[2] * R[1]; // yz
                }
            }
        }
    }

    // Self-images (A, A+T): zero nuclear force, nonzero strain response.
    if (stress_voigt) {
        for (int a = 0; a < n_at; ++a) {
            const auto& ea = *el[static_cast<std::size_t>(a)];
            const int ia = ao0[static_cast<std::size_t>(a)];
            const int na = ea.n_ao;
            const double rcut = pair_rcut_A(ea, ea, type[a], type[a]);
            const auto& sk = pair_sk(type[a], type[a]);
            const double dqa = dq[static_cast<std::size_t>(a)];
            const double shift = dV[static_cast<std::size_t>(a)];
            for (const auto& T : images) {
                const double r =
                    std::sqrt(T.tx * T.tx + T.ty * T.ty + T.tz * T.tz);
                if (r < 1e-12 || r > rcut)
                    continue;
                const double dir[3] = {T.tx / r, T.ty / r, T.tz / r};
                for (std::size_t ik = 0; ik < nk; ++ik)
                    conj_phase[ik] = std::conj(
                        bloch_phase(kpts[ik].frac, T.n0, T.n1, T.n2));
                for (int k = 0; k < 3; ++k) {
                    dH0[k] = xt::zeros<double>({n_ao, n_ao});
                    dS[k] = xt::zeros<double>({n_ao, n_ao});
                }
                internal::MioSketchPair{ea, ea, sk}.set_block_derivative(
                    T.tx, T.ty, T.tz, r, dH0, dS, ia, ia);

                // Factor 1: the image walk already visits both T and −T, and
                // the (a,a) block is contracted whole. γ_AA and E_rep are each
                // half-weighted per ordered image.
                double dE_dT[3] = {0, 0, 0};
                for (int i = 0; i < na; ++i) {
                    for (int j = 0; j < na; ++j) {
                        double p_re = 0, w_re = 0;
                        for (std::size_t ik = 0; ik < nk; ++ik) {
                            const double wk = kpts[ik].weight;
                            p_re += wk * (Pk[ik](ia + i, ia + j)
                                          * conj_phase[ik])
                                             .real();
                            w_re += wk * (Wk[ik](ia + i, ia + j)
                                          * conj_phase[ik])
                                             .real();
                        }
                        for (int k = 0; k < 3; ++k) {
                            const double ds = dS[k](ia + i, ia + j);
                            dE_dT[k] += p_re
                                            * (dH0[k](ia + i, ia + j)
                                               + shift * ds)
                                        - w_re * ds;
                        }
                    }
                }
                const double half_pair =
                    0.5 * dqa * dqa * dgamma_dr(r, ea.hubbard_u, ea.hubbard_u)
                    + 0.5 * pair_repulsive_dE_dr(type[a], type[a], r);
                for (int k = 0; k < 3; ++k)
                    dE_dT[k] += half_pair * dir[k];

                const double R[3] = {T.tx, T.ty, T.tz};
                dE_de[0] += dE_dT[0] * R[0];
                dE_de[1] += dE_dT[1] * R[1];
                dE_de[2] += dE_dT[2] * R[2];
                dE_de[3] += dE_dT[0] * R[1] + dE_dT[1] * R[0]; // xy
                dE_de[4] += dE_dT[0] * R[2] + dE_dT[2] * R[0]; // xz
                dE_de[5] += dE_dT[1] * R[2] + dE_dT[2] * R[1]; // yz
            }
        }
    }

    if (stress_voigt) {
        const double vol = box.volume();
        if (vol <= 0.0)
            throw std::runtime_error(
                "DFTB2 multi-k: non-positive box volume for stress");
        for (int v = 0; v < 6; ++v)
            stress_voigt[v] = -dE_de[v] / vol;
    }

    constexpr double kImagTol = 1e-9;
    if (max_imag > kImagTol)
        throw std::runtime_error(
            "DFTB2 multi-k: Bloch density k-sum did not reduce to a real force "
            "(max |Im| " + std::to_string(max_imag) + " > "
            + std::to_string(kImagTol)
            + "); the ±k mesh is not time-reversal symmetric or P(k) is not "
              "Hermitian");
}


/// Collinear open-shell multi-k SCC: dual α/β H(k) with V^σ = γ·Δq ± W m.
SccResult scc_energy_unrestricted_kmesh(
    int n_at, int n_ao, int n_alpha, int n_beta,
    const std::vector<const DftbElementRow*>& el, const std::vector<int>& ao0,
    const int* type, const StorageReal* x, const StorageReal* y,
    const StorageReal* z, const cpu::Box& box, double scc_tol, int max_scc,
    double mix_beta, const int mesh[3], bool time_reversal) {
    if (n_alpha > n_ao || n_beta > n_ao)
        throw std::runtime_error(
            "DFTB2 multi-k open shell: n_alpha/n_beta exceeds basis size");

    const std::vector<KPoint> kpts = monkhorst_pack_mesh(mesh, time_reversal);
    const std::size_t nk = kpts.size();
    std::vector<CplxMatrix> H0k(nk), Sk(nk);
    for (std::size_t ik = 0; ik < nk; ++ik)
        build_h0_s_k(n_at, n_ao, el, ao0, type, x, y, z, box, kpts[ik].frac,
                     H0k[ik], Sk[ik]);

    const std::size_t n = static_cast<std::size_t>(n_at);
    std::vector<double> dq(n, 0.0), mspin(n, 0.0);
    std::vector<double> dq_out(n, 0.0), mspin_out(n, 0.0);
    if (n_alpha != n_beta) {
        const double m_tot = static_cast<double>(n_alpha - n_beta);
        for (int a = 0; a < n_at; ++a)
            mspin[static_cast<std::size_t>(a)] =
                m_tot / static_cast<double>(n_at);
    }

    double residual = 0;
    int iterations = 0;
    bool converged = false;
    internal::ChargeMixer mixer(mix_beta, kSccMixDepth);
    std::vector<CplxMatrix> Pak(nk), Pbk(nk), Wak(nk), Wbk(nk);

    for (int it = 0; it < max_scc; ++it) {
        iterations = it + 1;
        std::vector<double> dVc(n, 0.0), dVs(n, 0.0);
        for (int a = 0; a < n_at; ++a) {
            double shift = 0;
            for (int b = 0; b < n_at; ++b)
                shift += gamma_pair(a, b, el, x, y, z, box)
                         * dq[static_cast<std::size_t>(b)];
            dVc[static_cast<std::size_t>(a)] = shift;
            dVs[static_cast<std::size_t>(a)] =
                el[static_cast<std::size_t>(a)]->spin_w
                * mspin[static_cast<std::size_t>(a)];
        }
        std::vector<double> eps_a(n), eps_b(n);
        for (int a = 0; a < n_at; ++a) {
            eps_a[static_cast<std::size_t>(a)] =
                dVc[static_cast<std::size_t>(a)]
                + dVs[static_cast<std::size_t>(a)];
            eps_b[static_cast<std::size_t>(a)] =
                dVc[static_cast<std::size_t>(a)]
                - dVs[static_cast<std::size_t>(a)];
        }

        std::vector<double> qa_ao(static_cast<std::size_t>(n_ao), 0.0);
        std::vector<double> qb_ao(static_cast<std::size_t>(n_ao), 0.0);
        for (std::size_t ik = 0; ik < nk; ++ik) {
            CplxMatrix Ha, Hb;
            apply_shift_k(n_at, el, ao0, H0k[ik], Sk[ik], eps_a, Ha);
            apply_shift_k(n_at, el, ao0, H0k[ik], Sk[ik], eps_b, Hb);
            auto sol_a = diagonalize_fock_generalized_k(Ha, Sk[ik]);
            auto sol_b = diagonalize_fock_generalized_k(Hb, Sk[ik]);
            const CplxMatrix& Ca = sol_a.second;
            const CplxMatrix& Cb = sol_b.second;

            CplxMatrix Pa = xt::zeros<Cplx>({n_ao, n_ao});
            CplxMatrix Pb = xt::zeros<Cplx>({n_ao, n_ao});
            for (int o = 0; o < n_alpha; ++o)
                for (int mu = 0; mu < n_ao; ++mu)
                    for (int nu = 0; nu < n_ao; ++nu)
                        Pa(mu, nu) += Ca(mu, o) * std::conj(Ca(nu, o));
            for (int o = 0; o < n_beta; ++o)
                for (int mu = 0; mu < n_ao; ++mu)
                    for (int nu = 0; nu < n_ao; ++nu)
                        Pb(mu, nu) += Cb(mu, o) * std::conj(Cb(nu, o));
            Pak[ik] = Pa;
            Pbk[ik] = Pb;

            const CplxMatrix PSa = cplx_mul(Pa, Sk[ik]);
            const CplxMatrix PSb = cplx_mul(Pb, Sk[ik]);
            const double w = kpts[ik].weight;
            for (int mu = 0; mu < n_ao; ++mu) {
                qa_ao[static_cast<std::size_t>(mu)] += w * PSa(mu, mu).real();
                qb_ao[static_cast<std::size_t>(mu)] += w * PSb(mu, mu).real();
            }
        }

        residual = 0;
        for (int a = 0; a < n_at; ++a) {
            double qa = 0, qb = 0;
            const int ia = ao0[static_cast<std::size_t>(a)];
            const int na = el[static_cast<std::size_t>(a)]->n_ao;
            for (int i = 0; i < na; ++i) {
                qa += qa_ao[static_cast<std::size_t>(ia + i)];
                qb += qb_ao[static_cast<std::size_t>(ia + i)];
            }
            dq_out[static_cast<std::size_t>(a)] =
                (qa + qb) - el[static_cast<std::size_t>(a)]->n_val;
            mspin_out[static_cast<std::size_t>(a)] = qa - qb;
            residual =
                std::max(residual, std::abs(dq_out[static_cast<std::size_t>(a)]
                                            - dq[static_cast<std::size_t>(a)]));
            residual = std::max(residual,
                                std::abs(mspin_out[static_cast<std::size_t>(a)]
                                         - mspin[static_cast<std::size_t>(a)]));
        }
        if (residual < scc_tol) {
            dq = dq_out;
            mspin = mspin_out;
            converged = true;
            break;
        }
        std::vector<double> in(dq);
        in.insert(in.end(), mspin.begin(), mspin.end());
        std::vector<double> out_vec(dq_out);
        out_vec.insert(out_vec.end(), mspin_out.begin(), mspin_out.end());
        const std::vector<double> mixed = mixer.next(in, out_vec);
        for (std::size_t a = 0; a < n; ++a) {
            dq[a] = mixed[a];
            mspin[a] = mixed[n + a];
        }
    }
    require_scc_converged(converged, residual, scc_tol, max_scc);

    // Final densities + energy-weighted W(k) at converged (dq, m).
    {
        std::vector<double> dVc(n, 0.0), dVs(n, 0.0);
        for (int a = 0; a < n_at; ++a) {
            double shift = 0;
            for (int b = 0; b < n_at; ++b)
                shift += gamma_pair(a, b, el, x, y, z, box)
                         * dq[static_cast<std::size_t>(b)];
            dVc[static_cast<std::size_t>(a)] = shift;
            dVs[static_cast<std::size_t>(a)] =
                el[static_cast<std::size_t>(a)]->spin_w
                * mspin[static_cast<std::size_t>(a)];
        }
        std::vector<double> eps_a(n), eps_b(n);
        for (int a = 0; a < n_at; ++a) {
            eps_a[static_cast<std::size_t>(a)] =
                dVc[static_cast<std::size_t>(a)]
                + dVs[static_cast<std::size_t>(a)];
            eps_b[static_cast<std::size_t>(a)] =
                dVc[static_cast<std::size_t>(a)]
                - dVs[static_cast<std::size_t>(a)];
        }
        for (std::size_t ik = 0; ik < nk; ++ik) {
            CplxMatrix Ha, Hb;
            apply_shift_k(n_at, el, ao0, H0k[ik], Sk[ik], eps_a, Ha);
            apply_shift_k(n_at, el, ao0, H0k[ik], Sk[ik], eps_b, Hb);
            auto sol_a = diagonalize_fock_generalized_k(Ha, Sk[ik]);
            auto sol_b = diagonalize_fock_generalized_k(Hb, Sk[ik]);
            const auto& ea = sol_a.first;
            const auto& eb = sol_b.first;
            const CplxMatrix& Ca = sol_a.second;
            const CplxMatrix& Cb = sol_b.second;
            CplxMatrix Pa = xt::zeros<Cplx>({n_ao, n_ao});
            CplxMatrix Pb = xt::zeros<Cplx>({n_ao, n_ao});
            CplxMatrix Wa = xt::zeros<Cplx>({n_ao, n_ao});
            CplxMatrix Wb = xt::zeros<Cplx>({n_ao, n_ao});
            for (int o = 0; o < n_alpha; ++o) {
                const double e_o = ea[static_cast<std::size_t>(o)];
                for (int mu = 0; mu < n_ao; ++mu)
                    for (int nu = 0; nu < n_ao; ++nu) {
                        const Cplx cc = Ca(mu, o) * std::conj(Ca(nu, o));
                        Pa(mu, nu) += cc;
                        Wa(mu, nu) += e_o * cc;
                    }
            }
            for (int o = 0; o < n_beta; ++o) {
                const double e_o = eb[static_cast<std::size_t>(o)];
                for (int mu = 0; mu < n_ao; ++mu)
                    for (int nu = 0; nu < n_ao; ++nu) {
                        const Cplx cc = Cb(mu, o) * std::conj(Cb(nu, o));
                        Pb(mu, nu) += cc;
                        Wb(mu, nu) += e_o * cc;
                    }
            }
            Pak[ik] = Pa;
            Pbk[ik] = Pb;
            Wak[ik] = Wa;
            Wbk[ik] = Wb;
        }
    }

    double e_band = 0;
    for (std::size_t ik = 0; ik < nk; ++ik) {
        e_band += kpts[ik].weight
                  * (re_trace_product_k(Pak[ik], H0k[ik])
                     + re_trace_product_k(Pbk[ik], H0k[ik]));
    }
    double e_scc = 0;
    for (int a = 0; a < n_at; ++a)
        for (int b = 0; b < n_at; ++b)
            e_scc += 0.5 * gamma_pair(a, b, el, x, y, z, box)
                     * dq[static_cast<std::size_t>(a)]
                     * dq[static_cast<std::size_t>(b)];
    double e_spin = 0;
    for (int a = 0; a < n_at; ++a)
        e_spin += 0.5 * el[static_cast<std::size_t>(a)]->spin_w
                  * mspin[static_cast<std::size_t>(a)]
                  * mspin[static_cast<std::size_t>(a)];

    // Totals for charge-only callers; dual channels for the open-shell force.
    std::vector<CplxMatrix> Pk(nk), Wk(nk);
    for (std::size_t ik = 0; ik < nk; ++ik) {
        Pk[ik] = Pak[ik];
        Wk[ik] = Wak[ik];
        for (int mu = 0; mu < n_ao; ++mu)
            for (int nu = 0; nu < n_ao; ++nu) {
                Pk[ik](mu, nu) += Pbk[ik](mu, nu);
                Wk[ik](mu, nu) += Wbk[ik](mu, nu);
            }
    }

    SccResult out;
    out.energy = e_band + e_scc + e_spin
                 + repulsive_energy(n_at, type, x, y, z, box, el);
    out.residual = residual;
    out.iterations = iterations;
    out.dq = std::move(dq);
    out.spin = std::move(mspin);
    out.n_alpha = n_alpha;
    out.n_beta = n_beta;
    out.unrestricted = true;
    out.kpts = kpts;
    out.Pk = std::move(Pk);
    out.Wk = std::move(Wk);
    out.Pak = std::move(Pak);
    out.Pbk = std::move(Pbk);
    out.Wak = std::move(Wak);
    out.Wbk = std::move(Wbk);
    return out;
}

/// @param mesh  Monkhorst–Pack subdivisions; {1,1,1} uses the real Γ path.
/// @param time_reversal Fold ±k when multi-k.
SccResult scc_total(int n_at, int n_ao, const Occupancy& occ,
                    const std::vector<const DftbElementRow*>& el,
                    const std::vector<int>& ao0, const int* type,
                    const StorageReal* x, const StorageReal* y,
                    const StorageReal* z, const cpu::Box& box, double scc_tol,
                    int max_scc, double mix_beta, const int mesh[3],
                    bool time_reversal) {
    const bool multi_k = mesh[0] > 1 || mesh[1] > 1 || mesh[2] > 1;
    if (multi_k) {
        if (occ.unrestricted)
            return scc_energy_unrestricted_kmesh(
                n_at, n_ao, occ.n_alpha, occ.n_beta, el, ao0, type, x, y, z,
                box, scc_tol, max_scc, mix_beta, mesh, time_reversal);
        return scc_energy_restricted_kmesh(
            n_at, n_ao, occ.n_alpha, el, ao0, type, x, y, z, box, scc_tol,
            max_scc, mix_beta, mesh, time_reversal);
    }
    if (!occ.unrestricted)
        return scc_energy_restricted(n_at, n_ao, occ.n_alpha, el, ao0, type, x,
                                     y, z, box, scc_tol, max_scc, mix_beta);
    return scc_energy_unrestricted(n_at, n_ao, occ.n_alpha, occ.n_beta, el, ao0,
                                   type, x, y, z, box, scc_tol, max_scc,
                                   mix_beta);
}

/// Affine strain of box + coordinates (mechanical ε; matches PW/GPW stress FD).
void apply_affine_strain(cpu::Frame& frame, int voigt, double h) {
    // Voigt: 0=xx,1=yy,2=zz,3=xy,4=xz,5=yz. Off-diag: ε_αβ = ε_βα = h.
    Mat3 F{};
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            F(a, b) = (a == b) ? 1.0 : 0.0;
    if (voigt == 0)
        F(0, 0) += h;
    else if (voigt == 1)
        F(1, 1) += h;
    else if (voigt == 2)
        F(2, 2) += h;
    else if (voigt == 3) {
        F(0, 1) += h;
        F(1, 0) += h;
    } else if (voigt == 4) {
        F(0, 2) += h;
        F(2, 0) += h;
    } else if (voigt == 5) {
        F(1, 2) += h;
        F(2, 1) += h;
    }

    Mat3 cell = frame.box.cell();
    Mat3 cell_new;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double s = 0;
            for (int k = 0; k < 3; ++k)
                s += F(i, k) * static_cast<double>(cell(k, j));
            cell_new(i, j) = static_cast<AccReal>(s);
        }
    frame.box.set_cell(cell_new);

    auto x = frame.span(kAtom_x);
    auto y = frame.span(kAtom_y);
    auto z = frame.span(kAtom_z);
    const int n = frame.nlocal();
    for (int a = 0; a < n; ++a) {
        const double rx = static_cast<double>(x[a]);
        const double ry = static_cast<double>(y[a]);
        const double rz = static_cast<double>(z[a]);
        x[a] = static_cast<StorageReal>(F(0, 0) * rx + F(0, 1) * ry
                                        + F(0, 2) * rz);
        y[a] = static_cast<StorageReal>(F(1, 0) * rx + F(1, 1) * ry
                                        + F(1, 2) * rz);
        z[a] = static_cast<StorageReal>(F(2, 0) * rx + F(2, 1) * ry
                                        + F(2, 2) * rz);
    }
}

double launch_mio(cpu::Frame& f, double scc_tol, int max_scc, double mix_beta,
                  bool with_forces, bool with_stress, int charge,
                  int multiplicity, const int k_mesh[3], bool k_time_reversal,
                  std::vector<double>& out_dq, std::vector<double>& out_spin,
                  std::array<double, 6>& out_stress, double& out_residual,
                  int& out_iterations) {
    const int n_at = f.nlocal();
    const int* type = f.span(kAtom_type).data();
    StorageReal* x = f.span(kAtom_x);
    StorageReal* y = f.span(kAtom_y);
    StorageReal* z = f.span(kAtom_z);
    AccReal* fx = f.span(kAtom_fx);
    AccReal* fy = f.span(kAtom_fy);
    AccReal* fz = f.span(kAtom_fz);
    double& energy_meta = f.meta.get(kEnergy);
    out_stress.fill(0.0);

    const bool multi_k =
        k_mesh[0] > 1 || k_mesh[1] > 1 || k_mesh[2] > 1;

    std::vector<const DftbElementRow*> el(static_cast<std::size_t>(n_at));
    std::vector<int> ao0(static_cast<std::size_t>(n_at) + 1, 0);
    double n_elec = -static_cast<double>(charge);
    for (int a = 0; a < n_at; ++a) {
        el[static_cast<std::size_t>(a)] = &element_row(type[a]);
        n_elec += el[static_cast<std::size_t>(a)]->n_val;
        ao0[static_cast<std::size_t>(a) + 1] =
            ao0[static_cast<std::size_t>(a)]
            + el[static_cast<std::size_t>(a)]->n_ao;
    }
    const int n_ao = ao0[static_cast<std::size_t>(n_at)];
    if (n_ao < 1)
        throw std::runtime_error("DFTB2: no basis functions");

    const Occupancy occ = resolve_occupancy(n_elec, multiplicity);
    auto res = scc_total(n_at, n_ao, occ, el, ao0, type, x, y, z, f.box,
                         scc_tol, max_scc, mix_beta, k_mesh, k_time_reversal);
    out_dq = res.dq;
    out_spin = res.spin;
    out_residual = res.residual;
    out_iterations = res.iterations;
    const double energy = res.energy;
    energy_meta += energy;
    // The electronic per-term share of the total, same operator, same value
    // (esmm-contract-05): DFTB2 already accumulated into kEnergy.
    f.meta.get(kEnergyElectronic) += energy;

    // Multi-k: the analytic gradient (and pair virial) contracts the Bloch
    // pair (P(k), W(k)) instead of the real Γ orbitals. Open shell passes
    // the dual α/β densities for spin-resolved Pulay.
    if (multi_k) {
        if (with_forces || with_stress) {
            double stress_buf[6] = {0, 0, 0, 0, 0, 0};
            if (res.unrestricted)
                analytic_forces_restricted_kmesh(
                    n_at, n_ao, el, ao0, type, x, y, z, f.box, res.dq,
                    res.kpts, res.Pk, res.Wk, k_time_reversal, fx, fy, fz,
                    with_forces, with_stress ? stress_buf : nullptr, &res.spin,
                    &res.Pak, &res.Pbk, &res.Wak, &res.Wbk);
            else
                analytic_forces_restricted_kmesh(
                    n_at, n_ao, el, ao0, type, x, y, z, f.box, res.dq,
                    res.kpts, res.Pk, res.Wk, k_time_reversal, fx, fy, fz,
                    with_forces, with_stress ? stress_buf : nullptr);
            if (with_stress) {
                // kStress is the extensive Voigt virial [eV]; `stress_buf` and
                // `out_stress` (last_stress()) stay intensive [eV/Å³].
                const double vol = f.box.volume();
                auto& st = f.meta.get(kStress);
                for (int v = 0; v < 6; ++v) {
                    out_stress[static_cast<std::size_t>(v)] = stress_buf[v];
                    st[static_cast<std::size_t>(v)] += stress_buf[v] * vol;
                }
            }
        }
        return energy;
    }

    // Closed shell and collinear open shell take the SAME closed-form gradient
    // + pair virial: the restricted case is the α/β-degenerate limit. The
    // open-shell multi-SCC affine strain-FD stress fallback is gone.
    const bool have_orbitals =
        occ.unrestricted ? (res.Ca.shape(0) > 0) : (res.C.shape(0) > 0);
    if (!((with_forces || with_stress) && have_orbitals))
        return energy;

    double stress_buf[6] = {0, 0, 0, 0, 0, 0};
    if (occ.unrestricted)
        analytic_forces_stress(n_at, n_ao, res.n_alpha, res.n_beta, el, ao0,
                               type, x, y, z, f.box, res.dq, res.spin, res.Ca,
                               res.Cb, res.eps_a, res.eps_b, fx, fy, fz,
                               with_forces,
                               with_stress ? stress_buf : nullptr);
    else
        // Restricted: C^α = C^β = C, ε^α = ε^β = ε, n_α = n_β = n_occ,
        // m ≡ 0 ⇒ P^α = P^β = P/2, W^α = W^β = W/2.
        analytic_forces_stress(n_at, n_ao, res.n_occ, res.n_occ, el, ao0, type,
                               x, y, z, f.box, res.dq, res.spin, res.C, res.C,
                               res.eps, res.eps, fx, fy, fz, with_forces,
                               with_stress ? stress_buf : nullptr);

    if (with_stress) {
        // kStress is the extensive Voigt virial [eV]; `stress_buf` and
        // `out_stress` (last_stress()) stay intensive [eV/Å³].
        const double vol = f.box.volume();
        auto& st = f.meta.get(kStress);
        for (int v = 0; v < 6; ++v) {
            out_stress[static_cast<std::size_t>(v)] = stress_buf[v];
            st[static_cast<std::size_t>(v)] += stress_buf[v] * vol;
        }
    }
    return energy;
}

// ── 3ob-3-1 path (internal Ha / Bohr; public eV / Å) ───────────────────

const Dftb3obElement& element_3ob(int z) {
    for (std::size_t i = 0; i < kDftb3obNElements; ++i)
        if (kDftb3obElements[i].z == z)
            return kDftb3obElements[i];
    throw std::runtime_error("DFTB2(ThreeOb): unsupported element Z="
                             + std::to_string(z) + " (H/C/N/O only)");
}

const Dftb3obPair& pair_3ob(int za, int zb) {
    for (std::size_t i = 0; i < kDftb3obNPairs; ++i)
        if (kDftb3obPairs[i].z_a == za && kDftb3obPairs[i].z_b == zb)
            return kDftb3obPairs[i];
    throw std::runtime_error("DFTB2(ThreeOb): no SK table for pair Z="
                             + std::to_string(za) + "-" + std::to_string(zb));
}

/// DFTB+ `polyInterU1` Neville interpolant on a uniform abscissa (same as
/// `MioSkTable` / mio-1-1). Used for 3ob radial channels so CH4 matches DFTB+.
double poly_interp_u1(const double* xp, const double* yp, int nn, double xx) {
    std::vector<double> X(static_cast<std::size_t>(nn + 1));
    std::vector<double> C(static_cast<std::size_t>(nn + 1));
    std::vector<double> D(static_cast<std::size_t>(nn + 1));
    for (int i = 1; i <= nn; ++i) {
        X[static_cast<std::size_t>(i)] = xp[i - 1];
        C[static_cast<std::size_t>(i)] = yp[i - 1];
        D[static_cast<std::size_t>(i)] = yp[i - 1];
    }
    int iCl = static_cast<int>(
        std::ceil((xx - X[1]) / std::abs(X[2] - X[1])));
    if (iCl < 1)
        iCl = 1;
    if (iCl > nn)
        iCl = nn;
    double yy = yp[iCl - 1];
    iCl = iCl - 1;
    for (int mm = 1; mm <= nn - 1; ++mm) {
        for (int ii = 1; ii <= nn - mm; ++ii) {
            double rTmp = X[static_cast<std::size_t>(ii)]
                          - X[static_cast<std::size_t>(ii + mm)];
            rTmp = (C[static_cast<std::size_t>(ii + 1)]
                    - D[static_cast<std::size_t>(ii)])
                   / rTmp;
            C[static_cast<std::size_t>(ii)] =
                (X[static_cast<std::size_t>(ii)] - xx) * rTmp;
            D[static_cast<std::size_t>(ii)] =
                (X[static_cast<std::size_t>(ii + mm)] - xx) * rTmp;
        }
        double dyy;
        if (2 * iCl < nn - mm) {
            dyy = C[static_cast<std::size_t>(iCl + 1)];
        } else {
            dyy = D[static_cast<std::size_t>(iCl)];
            iCl = iCl - 1;
        }
        yy += dyy;
    }
    return yy;
}

/// One 3ob channel column as double (tables store float).
void sk_col_3ob(const Dftb3obPair& p, int ch, std::vector<double>& col) {
    col.resize(static_cast<std::size_t>(p.n_grid));
    for (int i = 0; i < p.n_grid; ++i) {
        const Dftb3obSkPoint& pt = p.sk[i];
        double v = 0;
        switch (ch) {
        case 0: v = pt.h_ss; break;
        case 1: v = pt.h_sp; break;
        case 2: v = pt.h_pps; break;
        case 3: v = pt.h_ppp; break;
        case 4: v = pt.s_ss; break;
        case 5: v = pt.s_sp; break;
        case 6: v = pt.s_pps; break;
        case 7: v = pt.s_ppp; break;
        default: break;
        }
        col[static_cast<std::size_t>(i)] = v;
    }
}

/// DFTB+ new-style SK grid value for one 3ob channel [Ha or dimensionless].
double grid_value_3ob(const Dftb3obPair& p, int ch, double r_bohr) {
    constexpr int nInterNew = 8;
    constexpr int nRightInterNew = 4;
    constexpr double deltaR = 1.0e-5;
    constexpr double distFudge = 1.0; // Bohr — DFTB+ default
    const int leng = p.n_grid;
    const double incr = p.grid;
    const double rMax = static_cast<double>(leng) * incr + distFudge;
    if (r_bohr >= rMax)
        return 0.0;
    std::vector<double> tab;
    sk_col_3ob(p, ch, tab);
    const int ind = static_cast<int>(std::floor(r_bohr / incr));
    if (ind < leng) {
        int iLast = std::min(leng, ind + nRightInterNew);
        iLast = std::max(iLast, nInterNew);
        double xa[nInterNew];
        double ya[nInterNew];
        for (int ii = 1; ii <= nInterNew; ++ii) {
            xa[ii - 1] = static_cast<double>(iLast - nInterNew + ii) * incr;
            ya[ii - 1] = tab[static_cast<std::size_t>(iLast - nInterNew + ii - 1)];
        }
        return poly_interp_u1(xa, ya, nInterNew, r_bohr);
    }
    // 5th-order taper over distFudge past the last grid point.
    const double dr = r_bohr - rMax;
    const int iLast = leng;
    double xa[nInterNew];
    double ya[nInterNew];
    for (int ii = 1; ii <= nInterNew; ++ii) {
        xa[ii - 1] = static_cast<double>(iLast - nInterNew + ii) * incr;
        ya[ii - 1] = tab[static_cast<std::size_t>(iLast - nInterNew + ii - 1)];
    }
    const double y0 = poly_interp_u1(xa, ya, nInterNew, xa[nInterNew - 1] - deltaR);
    const double y2 = poly_interp_u1(xa, ya, nInterNew, xa[nInterNew - 1] + deltaR);
    const double y1 = ya[nInterNew - 1];
    const double y1p = (y2 - y0) / (2.0 * deltaR);
    const double y1pp = (y2 + y0 - 2.0 * y1) / (deltaR * deltaR);
    const double dx = -distFudge;
    const double invdx = -1.0 / distFudge;
    const double dx1 = y1p * dx;
    const double dx2 = y1pp * dx * dx;
    const double dd = 10.0 * y1 - 4.0 * dx1 + 0.5 * dx2;
    const double ee = -15.0 * y1 + 7.0 * dx1 - 1.0 * dx2;
    const double ff = 6.0 * y1 - 3.0 * dx1 + 0.5 * dx2;
    const double xr = dr * invdx;
    return ((ff * xr + ee) * xr + dd) * xr * xr * xr;
}

/// SK channels at r [Bohr]: Hss,Hsp,Hpps,Hppp,Sss,Ssp,Spps,Sppp.
void interp_3ob_sk(const Dftb3obPair& p, double r_bohr, double out[8]) {
    for (int ch = 0; ch < 8; ++ch)
        out[ch] = grid_value_3ob(p, ch, r_bohr);
}

/// d(channel)/dR [1/Bohr] via central difference of the same interpolant.
void interp_3ob_sk_dr(const Dftb3obPair& p, double r_bohr, double dout[8]) {
    constexpr double h = 1.0e-5; // Bohr
    const double rp = r_bohr + h;
    const double rm = std::max(r_bohr - h, 1.0e-8);
    double op[8], om[8];
    interp_3ob_sk(p, rp, op);
    interp_3ob_sk(p, rm, om);
    for (int i = 0; i < 8; ++i)
        dout[i] = (op[i] - om[i]) / (rp - rm);
}

double rep_3ob(const Dftb3obPair& p, double r_bohr) {
    if (r_bohr >= p.rep_cutoff)
        return 0.0;
    // First interval start
    const double r0 = p.rep_int[0];
    if (r_bohr < r0)
        return std::exp(-p.rep_a1 * r_bohr + p.rep_a2) + p.rep_a3;
    for (int i = 0; i < p.n_rep_int; ++i) {
        const double* iv = p.rep_int + static_cast<std::size_t>(i) * 8;
        const double ra = iv[0];
        const double rb = iv[1];
        if (r_bohr < rb || std::abs(r_bohr - rb) < 1e-14) {
            const double dr = r_bohr - ra;
            double e = 0;
            double pow = 1.0;
            for (int k = 0; k < 6; ++k) {
                e += iv[2 + k] * pow;
                pow *= dr;
            }
            return e;
        }
    }
    return 0.0;
}

/// dE_rep/dR [Ha/Bohr] of the 3ob repulsive (exp + spline).
double rep_3ob_dr(const Dftb3obPair& p, double r_bohr) {
    if (r_bohr >= p.rep_cutoff)
        return 0.0;
    const double r0 = p.rep_int[0];
    if (r_bohr < r0)
        return -p.rep_a1 * std::exp(-p.rep_a1 * r_bohr + p.rep_a2);
    for (int i = 0; i < p.n_rep_int; ++i) {
        const double* iv = p.rep_int + static_cast<std::size_t>(i) * 8;
        const double ra = iv[0];
        const double rb = iv[1];
        if (r_bohr < rb || std::abs(r_bohr - rb) < 1e-14) {
            const double dr = r_bohr - ra;
            // e = c0 + c1 dr + c2 dr^2 + ... + c5 dr^5
            // de/dr = c1 + 2 c2 dr + ... + 5 c5 dr^4
            double de = 0;
            double pow = 1.0;
            for (int k = 1; k < 6; ++k) {
                de += static_cast<double>(k) * iv[2 + k] * pow;
                pow *= dr;
            }
            return de;
        }
    }
    return 0.0;
}

void fill_sk_block_3ob(const Dftb3obElement& ea, const Dftb3obElement& eb,
                       int za, int zb, double dx, double dy, double dz,
                       double r, linalg::RealMatrix& H0, linalg::RealMatrix& S,
                       int ia0, int ib0) {
    if (r < 1e-12)
        return;
    const double inv_r = 1.0 / r;
    const double l = dx * inv_r;
    const double m = dy * inv_r;
    const double n = dz * inv_r;

    const Dftb3obPair& p_ab = pair_3ob(za, zb);
    double ch[8];
    interp_3ob_sk(p_ab, r, ch);
    double Hss = ch[0], Hsp = ch[1], Hpps = ch[2], Hppp = ch[3];
    double Sss = ch[4], Ssp = ch[5], Spps = ch[6], Sppp = ch[7];
    // X–H tables store only ss; s–p lives in H–X.
    if (za != 1 && zb == 1) {
        const Dftb3obPair& p_ba = pair_3ob(zb, za);
        double rev[8];
        interp_3ob_sk(p_ba, r, rev);
        Hsp = rev[1];
        Ssp = rev[5];
    }

    auto set_hs = [&](int i, int j, double hv, double sv) {
        H0(ia0 + i, ib0 + j) = hv;
        H0(ib0 + j, ia0 + i) = hv;
        S(ia0 + i, ib0 + j) = sv;
        S(ib0 + j, ia0 + i) = sv;
    };

    set_hs(0, 0, Hss, Sss);
    if (ea.n_ao >= 1 && eb.n_ao == 4) {
        set_hs(0, 1, l * Hsp, l * Ssp);
        set_hs(0, 2, m * Hsp, m * Ssp);
        set_hs(0, 3, n * Hsp, n * Ssp);
    }
    if (ea.n_ao == 4 && eb.n_ao >= 1) {
        set_hs(1, 0, -l * Hsp, -l * Ssp);
        set_hs(2, 0, -m * Hsp, -m * Ssp);
        set_hs(3, 0, -n * Hsp, -n * Ssp);
    }
    if (ea.n_ao == 4 && eb.n_ao == 4) {
        const double dir[3] = {l, m, n};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                const double d_ab = (a == b) ? 1.0 : 0.0;
                const double hv = dir[a] * dir[b] * (Hpps - Hppp) + d_ab * Hppp;
                const double sv = dir[a] * dir[b] * (Spps - Sppp) + d_ab * Sppp;
                set_hs(1 + a, 1 + b, hv, sv);
            }
        }
    }
}

/// SK block derivatives w.r.t. bond vector R_B − R_A [Bohr] (3ob tables).
void fill_sk_block_deriv_3ob(const Dftb3obElement& ea, const Dftb3obElement& eb,
                             int za, int zb, double dx, double dy, double dz,
                             double r, linalg::RealMatrix dH0[3],
                             linalg::RealMatrix dS[3], int ia0, int ib0) {
    if (r < 1e-12)
        return;
    const double inv_r = 1.0 / r;
    const double dir[3] = {dx * inv_r, dy * inv_r, dz * inv_r};

    const Dftb3obPair& p_ab = pair_3ob(za, zb);
    double ch[8], dch[8];
    interp_3ob_sk(p_ab, r, ch);
    interp_3ob_sk_dr(p_ab, r, dch);
    double Hss = ch[0], Hsp = ch[1], Hpps = ch[2], Hppp = ch[3];
    double Sss = ch[4], Ssp = ch[5], Spps = ch[6], Sppp = ch[7];
    double dHss = dch[0], dHsp = dch[1], dHpps = dch[2], dHppp = dch[3];
    double dSss = dch[4], dSsp = dch[5], dSpps = dch[6], dSppp = dch[7];
    // X–H tables store only ss; s–p lives in H–X.
    if (za != 1 && zb == 1) {
        const Dftb3obPair& p_ba = pair_3ob(zb, za);
        double rev[8], drev[8];
        interp_3ob_sk(p_ba, r, rev);
        interp_3ob_sk_dr(p_ba, r, drev);
        Hsp = rev[1];
        Ssp = rev[5];
        dHsp = drev[1];
        dSsp = drev[5];
    }

    auto ddir = [&](int alpha, int k) {
        const double dab = (alpha == k) ? 1.0 : 0.0;
        return (dab - dir[alpha] * dir[k]) * inv_r;
    };

    for (int k = 0; k < 3; ++k) {
        const double dr_dk = dir[k];
        dH0[k](ia0, ib0) = dHss * dr_dk;
        dS[k](ia0, ib0) = dSss * dr_dk;

        if (ea.n_ao >= 1 && eb.n_ao == 4) {
            for (int a = 0; a < 3; ++a) {
                dH0[k](ia0, ib0 + 1 + a) =
                    ddir(a, k) * Hsp + dir[a] * dHsp * dr_dk;
                dS[k](ia0, ib0 + 1 + a) =
                    ddir(a, k) * Ssp + dir[a] * dSsp * dr_dk;
            }
        }
        if (ea.n_ao == 4 && eb.n_ao >= 1) {
            for (int a = 0; a < 3; ++a) {
                dH0[k](ia0 + 1 + a, ib0) =
                    -(ddir(a, k) * Hsp + dir[a] * dHsp * dr_dk);
                dS[k](ia0 + 1 + a, ib0) =
                    -(ddir(a, k) * Ssp + dir[a] * dSsp * dr_dk);
            }
        }
        if (ea.n_ao == 4 && eb.n_ao == 4) {
            const double dHsig = dHpps - dHppp;
            const double dSsig = dSpps - dSppp;
            const double Hsig = Hpps - Hppp;
            const double Ssig = Spps - Sppp;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    const double dab = (a == b) ? 1.0 : 0.0;
                    dH0[k](ia0 + 1 + a, ib0 + 1 + b) =
                        (ddir(a, k) * dir[b] + dir[a] * ddir(b, k)) * Hsig
                        + dir[a] * dir[b] * dHsig * dr_dk + dab * dHppp * dr_dk;
                    dS[k](ia0 + 1 + a, ib0 + 1 + b) =
                        (ddir(a, k) * dir[b] + dir[a] * ddir(b, k)) * Ssig
                        + dir[a] * dir[b] * dSsig * dr_dk + dab * dSppp * dr_dk;
                }
            }
        }
    }
}

double gamma_ab_ha(double r_bohr, double ua_ha, double ub_ha) {
    return internal::dftb_gamma::gamma_au(r_bohr, ua_ha, ub_ha);
}

/// dγ/dR [Ha/Bohr] for DFTB+ Elstner γ in a.u.
double dgamma_dr_ha(double r_bohr, double ua_ha, double ub_ha) {
    return internal::dftb_gamma::dgamma_dr_au(r_bohr, ua_ha, ub_ha);
}

struct Scc3obResult {
    double energy_ha = 0;
    double residual = 0;
    int iterations = 0;
    std::vector<double> dq;
    linalg::RealMatrix C;
    linalg::RealVector eps;
    int n_occ = 0;
};

Scc3obResult scc_3ob(const std::vector<const Dftb3obElement*>& el,
                     const std::vector<int>& ao0, const std::vector<int>& Z,
                     const std::vector<double>& x_b,
                     const std::vector<double>& y_b,
                     const std::vector<double>& z_b, int n_at, int n_ao,
                     int n_occ, double scc_tol, int max_scc, double mix_beta) {
    auto dist = [&](int i, int j, double& dx, double& dy, double& dz) {
        dx =
            x_b[static_cast<std::size_t>(j)] - x_b[static_cast<std::size_t>(i)];
        dy =
            y_b[static_cast<std::size_t>(j)] - y_b[static_cast<std::size_t>(i)];
        dz =
            z_b[static_cast<std::size_t>(j)] - z_b[static_cast<std::size_t>(i)];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    linalg::RealMatrix H0 = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix S = xt::zeros<double>({n_ao, n_ao});

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        H0(ia, ia) = ea.e_s;
        S(ia, ia) = 1.0;
        if (ea.n_ao == 4) {
            for (int p = 1; p < 4; ++p) {
                H0(ia + p, ia + p) = ea.e_p;
                S(ia + p, ia + p) = 1.0;
            }
        }
        for (int b = a + 1; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            double dx, dy, dz;
            const double r = dist(a, b, dx, dy, dz);
            if (r > std::max(ea.r_cut, eb.r_cut))
                continue;
            fill_sk_block_3ob(ea, eb, Z[static_cast<std::size_t>(a)],
                              Z[static_cast<std::size_t>(b)], dx, dy, dz, r, H0,
                              S, ia, ao0[static_cast<std::size_t>(b)]);
        }
    }

    const std::size_t n = static_cast<std::size_t>(n_at);
    std::vector<double> dq(n, 0.0);
    std::vector<double> dq_out(n, 0.0);
    linalg::RealMatrix P = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix C = xt::zeros<double>({n_ao, n_ao});
    linalg::RealVector eps = xt::zeros<double>({n_ao});
    double e_scc = 0, residual = 0;
    int iterations = 0;
    bool converged = false;
    internal::ChargeMixer mixer(mix_beta, kSccMixDepth);

    for (int it = 0; it < max_scc; ++it) {
        iterations = it + 1;
        std::vector<double> dV(static_cast<std::size_t>(n_at), 0.0);
        for (int a = 0; a < n_at; ++a) {
            double shift = 0;
            for (int b = 0; b < n_at; ++b) {
                double dx, dy, dz;
                const double r = (a == b) ? 0.0 : dist(a, b, dx, dy, dz);
                const double g =
                    (a == b)
                        ? el[static_cast<std::size_t>(a)]->hubbard_u
                        : gamma_ab_ha(
                              r, el[static_cast<std::size_t>(a)]->hubbard_u,
                              el[static_cast<std::size_t>(b)]->hubbard_u);
                shift += g * dq[static_cast<std::size_t>(b)];
            }
            dV[static_cast<std::size_t>(a)] = shift;
        }

        linalg::RealMatrix H = H0;
        for (int a = 0; a < n_at; ++a) {
            const int ia = ao0[static_cast<std::size_t>(a)];
            const int na = el[static_cast<std::size_t>(a)]->n_ao;
            for (int b = 0; b < n_at; ++b) {
                const int ib = ao0[static_cast<std::size_t>(b)];
                const int nb = el[static_cast<std::size_t>(b)]->n_ao;
                const double half = 0.5
                                    * (dV[static_cast<std::size_t>(a)]
                                       + dV[static_cast<std::size_t>(b)]);
                for (int i = 0; i < na; ++i)
                    for (int j = 0; j < nb; ++j)
                        H(ia + i, ib + j) += half * S(ia + i, ib + j);
            }
        }

        auto sol = linalg::diagonalize_fock_generalized(H, S);
        eps = sol.first;
        C = sol.second;

        P = xt::zeros<double>({n_ao, n_ao});
        for (int o = 0; o < n_occ; ++o)
            for (int mu = 0; mu < n_ao; ++mu)
                for (int nu = 0; nu < n_ao; ++nu)
                    P(mu, nu) += 2.0 * C(mu, o) * C(nu, o);

        linalg::RealMatrix PS = xt::linalg::dot(P, S);
        residual = 0;
        for (int a = 0; a < n_at; ++a) {
            double qa = 0;
            const int ia = ao0[static_cast<std::size_t>(a)];
            const int na = el[static_cast<std::size_t>(a)]->n_ao;
            for (int i = 0; i < na; ++i)
                qa += PS(ia + i, ia + i);
            dq_out[static_cast<std::size_t>(a)] =
                qa - el[static_cast<std::size_t>(a)]->n_val;
            residual =
                std::max(residual, std::abs(dq_out[static_cast<std::size_t>(a)]
                                            - dq[static_cast<std::size_t>(a)]));
        }

        if (residual < scc_tol) {
            dq = dq_out;
            converged = true;
            break;
        }
        dq = mixer.next(dq, dq_out);
    }

    require_scc_converged(converged, residual, scc_tol, max_scc);

    e_scc = 0;
    for (int a = 0; a < n_at; ++a)
        for (int b = 0; b < n_at; ++b) {
            double dx, dy, dz;
            const double r = (a == b) ? 0.0 : dist(a, b, dx, dy, dz);
            const double g =
                (a == b)
                    ? el[static_cast<std::size_t>(a)]->hubbard_u
                    : gamma_ab_ha(r, el[static_cast<std::size_t>(a)]->hubbard_u,
                                  el[static_cast<std::size_t>(b)]->hubbard_u);
            e_scc += 0.5 * g * dq[static_cast<std::size_t>(a)]
                     * dq[static_cast<std::size_t>(b)];
        }

    // E_elec = Tr(P H0) + E_scc  (DFTB2; matches DFTB+ double-counting)
    const double e_h0 = trace_product(P, H0);

    double e_repulsive = 0;
    for (int a = 0; a < n_at; ++a)
        for (int b = a + 1; b < n_at; ++b) {
            double dx, dy, dz;
            const double r = dist(a, b, dx, dy, dz);
            const auto& pr = pair_3ob(Z[static_cast<std::size_t>(a)],
                                      Z[static_cast<std::size_t>(b)]);
            e_repulsive += rep_3ob(pr, r);
        }

    Scc3obResult out;
    out.energy_ha = e_h0 + e_scc + e_repulsive;
    out.residual = residual;
    out.iterations = iterations;
    out.dq = std::move(dq);
    out.C = std::move(C);
    out.eps = std::move(eps);
    out.n_occ = n_occ;
    return out;
}

/// Closed-form nuclear gradient (+ optional intensive stress) for **restricted**
/// SCC-DFTB2 on 3ob tables. Internally Ha/Bohr; writes Frame forces in eV/Å and
/// stress in eV/Å³ Voigt (xx,yy,zz,xy,xz,yz), mechanical convention
/// σ = −(1/Ω)∂E/∂ε — same assembly and same sign as the restricted MioSketch
/// path (`analytic_forces_restricted`).
///
/// `scc_3ob` is a raw-distance **cluster** sum (no lattice images, no
/// min-image), so E depends on geometry only through the pair bond vectors and
/// an affine strain of nuclei+cell gives ∂E/∂ε_αβ = Σ_pairs (∂E/∂R_α) R_β
/// exactly — the pair virial below is FD-exact, not an approximation.
///
/// @param vol_a3 Cell volume Ω [Å³]; only read when `stress_voigt` is non-null.
void analytic_forces_3ob(int n_at, int n_ao, int n_occ,
                         const std::vector<const Dftb3obElement*>& el,
                         const std::vector<int>& ao0, const std::vector<int>& Z,
                         const std::vector<double>& x_b,
                         const std::vector<double>& y_b,
                         const std::vector<double>& z_b,
                         const std::vector<double>& dq,
                         const linalg::RealMatrix& C,
                         const linalg::RealVector& eps, AccReal* fx, AccReal* fy,
                         AccReal* fz, bool want_forces,
                         double* stress_voigt /* nullable, length 6 */,
                         double vol_a3) {
    auto dist = [&](int i, int j, double& dx, double& dy, double& dz) {
        dx =
            x_b[static_cast<std::size_t>(j)] - x_b[static_cast<std::size_t>(i)];
        dy =
            y_b[static_cast<std::size_t>(j)] - y_b[static_cast<std::size_t>(i)];
        dz =
            z_b[static_cast<std::size_t>(j)] - z_b[static_cast<std::size_t>(i)];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    };

    std::vector<double> dV(static_cast<std::size_t>(n_at), 0.0);
    for (int a = 0; a < n_at; ++a) {
        double shift = 0;
        for (int b = 0; b < n_at; ++b) {
            double dx, dy, dz;
            const double r = (a == b) ? 0.0 : dist(a, b, dx, dy, dz);
            const double g =
                (a == b)
                    ? el[static_cast<std::size_t>(a)]->hubbard_u
                    : gamma_ab_ha(r, el[static_cast<std::size_t>(a)]->hubbard_u,
                                  el[static_cast<std::size_t>(b)]->hubbard_u);
            shift += g * dq[static_cast<std::size_t>(b)];
        }
        dV[static_cast<std::size_t>(a)] = shift;
    }

    linalg::RealMatrix P = xt::zeros<double>({n_ao, n_ao});
    linalg::RealMatrix W = xt::zeros<double>({n_ao, n_ao});
    for (int o = 0; o < n_occ; ++o)
        for (int mu = 0; mu < n_ao; ++mu)
            for (int nu = 0; nu < n_ao; ++nu) {
                P(mu, nu) += 2.0 * C(mu, o) * C(nu, o);
                W(mu, nu) += 2.0 * eps(o) * C(mu, o) * C(nu, o);
            }

    // Ha/Bohr → eV/Å: F_eV_A = F_Ha_Bohr * kHa2eV / kBohr2A
    const double fscale = kHa2eV / kBohr2A;

    linalg::RealMatrix dH0[3] = {xt::zeros<double>({n_ao, n_ao}),
                                 xt::zeros<double>({n_ao, n_ao}),
                                 xt::zeros<double>({n_ao, n_ao})};
    linalg::RealMatrix dS[3] = {xt::zeros<double>({n_ao, n_ao}),
                                xt::zeros<double>({n_ao, n_ao}),
                                xt::zeros<double>({n_ao, n_ao})};

    // ∂E/∂ε in Voigt order [Ha] (strain is dimensionless), before −1/Ω.
    double dE_de[6] = {0, 0, 0, 0, 0, 0};

    for (int a = 0; a < n_at; ++a) {
        const auto& ea = *el[static_cast<std::size_t>(a)];
        const int ia = ao0[static_cast<std::size_t>(a)];
        for (int b = a + 1; b < n_at; ++b) {
            const auto& eb = *el[static_cast<std::size_t>(b)];
            double dx, dy, dz;
            const double r = dist(a, b, dx, dy, dz);
            if (r < 1e-12)
                continue;
            const int ib = ao0[static_cast<std::size_t>(b)];
            const double dir[3] = {dx / r, dy / r, dz / r};
            double dE_dbond[3] = {0, 0, 0}; // ∂E/∂bond [Ha/Bohr]

            if (r <= std::max(ea.r_cut, eb.r_cut)) {
                for (int k = 0; k < 3; ++k) {
                    dH0[k] = xt::zeros<double>({n_ao, n_ao});
                    dS[k] = xt::zeros<double>({n_ao, n_ao});
                }
                fill_sk_block_deriv_3ob(ea, eb, Z[static_cast<std::size_t>(a)],
                                        Z[static_cast<std::size_t>(b)], dx, dy,
                                        dz, r, dH0, dS, ia, ib);
                const double shift = 0.5
                                     * (dV[static_cast<std::size_t>(a)]
                                        + dV[static_cast<std::size_t>(b)]);
                const int na = ea.n_ao;
                const int nb = eb.n_ao;
                for (int k = 0; k < 3; ++k) {
                    double acc = 0;
                    for (int i = 0; i < na; ++i)
                        for (int j = 0; j < nb; ++j) {
                            const double p = P(ia + i, ib + j);
                            const double w = W(ia + i, ib + j);
                            const double dh = dH0[k](ia + i, ib + j);
                            const double ds = dS[k](ia + i, ib + j);
                            acc += p * (dh + shift * ds) - w * ds;
                        }
                    dE_dbond[k] += 2.0 * acc;
                }
            }

            // E = Tr(PH0) + ½ Σ_AB γ_AB Δq_A Δq_B + E_rep; with
            // bond = R_B − R_A, ∂E²/∂bond = +Δq_A Δq_B dγ/dr.
            const double dgdR = dgamma_dr_ha(r, ea.hubbard_u, eb.hubbard_u);
            const double dqa = dq[static_cast<std::size_t>(a)];
            const double dqb = dq[static_cast<std::size_t>(b)];
            for (int k = 0; k < 3; ++k)
                dE_dbond[k] += dqa * dqb * dgdR * dir[k];

            const auto& pr = pair_3ob(Z[static_cast<std::size_t>(a)],
                                      Z[static_cast<std::size_t>(b)]);
            const double dErep_dr = rep_3ob_dr(pr, r);
            for (int k = 0; k < 3; ++k)
                dE_dbond[k] += dErep_dr * dir[k];

            // F = −∂E/∂R; bond = R_B − R_A ⇒ F_A += ∂E/∂bond, F_B −= ∂E/∂bond
            // after converting Ha/Bohr → eV/Å.
            if (want_forces && fx && fy && fz) {
                const double fax = dE_dbond[0] * fscale;
                const double fay = dE_dbond[1] * fscale;
                const double faz = dE_dbond[2] * fscale;
                fx[a] += static_cast<AccReal>(fax);
                fy[a] += static_cast<AccReal>(fay);
                fz[a] += static_cast<AccReal>(faz);
                fx[b] -= static_cast<AccReal>(fax);
                fy[b] -= static_cast<AccReal>(fay);
                fz[b] -= static_cast<AccReal>(faz);
            }

            if (stress_voigt) {
                // Pair virial, Voigt order and sign taken from the restricted
                // MioSketch path: off-diagonals are doubled because the affine
                // strain is symmetric (ε_αβ = ε_βα = h). [Ha], bond in Bohr.
                const double R[3] = {dx, dy, dz};
                dE_de[0] += dE_dbond[0] * R[0];
                dE_de[1] += dE_dbond[1] * R[1];
                dE_de[2] += dE_dbond[2] * R[2];
                dE_de[3] += dE_dbond[0] * R[1] + dE_dbond[1] * R[0]; // xy
                dE_de[4] += dE_dbond[0] * R[2] + dE_dbond[2] * R[0]; // xz
                dE_de[5] += dE_dbond[1] * R[2] + dE_dbond[2] * R[1]; // yz
            }
        }
    }

    if (stress_voigt) {
        if (vol_a3 <= 0.0)
            throw std::runtime_error(
                "DFTB2(ThreeOb): non-positive box volume for stress");
        // ∂E/∂ε [Ha] → [eV]; Ω [Å³] ⇒ σ [eV/Å³].
        for (int v = 0; v < 6; ++v)
            stress_voigt[v] = -dE_de[v] * kHa2eV / vol_a3;
    }
}

double launch_3ob(cpu::Frame& f, double scc_tol, int max_scc, double mix_beta,
                  bool with_forces, bool with_stress,
                  std::vector<double>& out_dq,
                  std::array<double, 6>& out_stress, double& out_residual,
                  int& out_iterations) {
    const int n_at = f.nlocal();
    const auto type = f.span(kAtom_type);
    StorageReal* x = f.span(kAtom_x);
    StorageReal* y = f.span(kAtom_y);
    StorageReal* z = f.span(kAtom_z);
    AccReal* fx = f.span(kAtom_fx);
    AccReal* fy = f.span(kAtom_fy);
    AccReal* fz = f.span(kAtom_fz);
    double& energy_meta = f.meta.get(kEnergy);
    out_stress.fill(0.0);

    std::vector<const Dftb3obElement*> el(static_cast<std::size_t>(n_at));
    std::vector<int> ao0(static_cast<std::size_t>(n_at) + 1, 0);
    std::vector<int> Z(static_cast<std::size_t>(n_at));
    std::vector<double> xb(static_cast<std::size_t>(n_at));
    std::vector<double> yb(static_cast<std::size_t>(n_at));
    std::vector<double> zb(static_cast<std::size_t>(n_at));
    double n_elec = 0;
    for (int a = 0; a < n_at; ++a) {
        Z[static_cast<std::size_t>(a)] = type[a];
        el[static_cast<std::size_t>(a)] = &element_3ob(type[a]);
        n_elec += el[static_cast<std::size_t>(a)]->n_val;
        ao0[static_cast<std::size_t>(a) + 1] =
            ao0[static_cast<std::size_t>(a)]
            + el[static_cast<std::size_t>(a)]->n_ao;
        // Å → Bohr for SK/rep (molecular; no min-image in 3ob pin path)
        xb[static_cast<std::size_t>(a)] = static_cast<double>(x[a]) * kA2Bohr;
        yb[static_cast<std::size_t>(a)] = static_cast<double>(y[a]) * kA2Bohr;
        zb[static_cast<std::size_t>(a)] = static_cast<double>(z[a]) * kA2Bohr;
    }
    const int n_ao = ao0[static_cast<std::size_t>(n_at)];
    if (n_ao < 1)
        throw std::runtime_error("DFTB2(ThreeOb): no basis functions");
    if (std::fabs(n_elec - std::round(n_elec)) > 1e-9
        || static_cast<int>(std::lround(n_elec)) % 2 != 0)
        throw std::runtime_error(
            "DFTB2(ThreeOb): odd/non-integer valence electron count");
    const int n_occ = static_cast<int>(std::lround(n_elec)) / 2;

    auto res = scc_3ob(el, ao0, Z, xb, yb, zb, n_at, n_ao, n_occ, scc_tol,
                       max_scc, mix_beta);
    const double energy = res.energy_ha * kHa2eV;
    out_residual = res.residual;
    out_iterations = res.iterations;
    energy_meta += energy;
    // The electronic per-term share of the total, same operator, same value
    // (esmm-contract-05): DFTB2 already accumulated into kEnergy.
    f.meta.get(kEnergyElectronic) += energy;
    out_dq = std::move(res.dq);

    // One post-SCC pair contraction serves both F and σ: the virial is the
    // same bond terms as the gradient (no multi-SCC strain-FD product path).
    if (with_forces || with_stress) {
        double stress_buf[6] = {0, 0, 0, 0, 0, 0};
        analytic_forces_3ob(n_at, n_ao, res.n_occ, el, ao0, Z, xb, yb, zb,
                            out_dq, res.C, res.eps, fx, fy, fz, with_forces,
                            with_stress ? stress_buf : nullptr,
                            f.box.volume());
        if (with_stress) {
            // kStress is the extensive Voigt virial [eV]; `stress_buf` and
            // `out_stress` (last_stress()) stay intensive [eV/Å³].
            const double vol = f.box.volume();
            auto& st = f.meta.get(kStress);
            for (int v = 0; v < 6; ++v) {
                out_stress[static_cast<std::size_t>(v)] = stress_buf[v];
                st[static_cast<std::size_t>(v)] += stress_buf[v] * vol;
            }
        }
    }
    return energy;
}

} // namespace

void DFTB2::build(const BuildContext& /*ctx*/) {}

void DFTB2::launch(const LaunchContext& ctx) {
    auto& f = ctx.frame;
    if (f.nlocal() < 1)
        throw std::invalid_argument("DFTB2: empty frame");

    if (!(mix_beta > 0.0) || mix_beta > 1.0)
        throw std::invalid_argument("DFTB2: mix_beta must be in (0, 1]");

    bool any_u = false;
    for (std::size_t z = 1; z < u_eff.size(); ++z) {
        if (u_eff[z] != 0.0) {
            any_u = true;
            break;
        }
    }
    if (any_u) {
        if (sk_set != DftbSkSet::MioSketch)
            throw std::runtime_error(
                "DFTB2+U: only sk_set=MioSketch is supported (v1 foothold)");
        if (with_stress)
            throw std::runtime_error(
                "DFTB2+U: with_stress not supported — cell virial of E_U is "
                "not implemented");
        if (k_mesh[0] > 1 || k_mesh[1] > 1 || k_mesh[2] > 1)
            throw std::runtime_error(
                "DFTB2+U: multi-k not supported with u_eff");
    }

    last_stress_.fill(0.0);
    e_u_ = 0.0;
    for (int d = 0; d < 3; ++d)
        if (k_mesh[static_cast<std::size_t>(d)] < 1)
            throw std::invalid_argument(
                "DFTB2: k_mesh subdivision must be >= 1 (axis "
                + std::to_string(d) + " got "
                + std::to_string(k_mesh[static_cast<std::size_t>(d)]) + ")");
    const bool multi_k = k_mesh[0] > 1 || k_mesh[1] > 1 || k_mesh[2] > 1;

    // When +U is active, nuclear forces come from central FD of the total
    // energy (SCC + E_U), not the analytic SCC gradient alone — ∂E_U/∂R needs
    // the charge response. Energy-only path stays analytic SCC + closed E_U.
    const bool u_force_fd = any_u && with_forces;
    const bool scc_forces = with_forces && !u_force_fd;

    if (sk_set == DftbSkSet::ThreeOb) {
        if (multi_k)
            throw std::runtime_error(
                "DFTB2(ThreeOb): multi-k not supported — the 3ob radial-table "
                "path has no Bloch-phased H0(k)/S(k) builder (and no lattice "
                "sum at all); use sk_set=MioSketch or k_mesh={1,1,1}");
        if (multiplicity != 1 || charge != 0)
            throw std::runtime_error(
                "DFTB2(ThreeOb): open-shell/charge not supported on this path");
        energy_ =
            launch_3ob(f, scc_tol, max_scc, mix_beta, scc_forces, with_stress,
                       last_dq_, last_stress_, scc_residual_, scc_iterations_);
        last_spin_.assign(static_cast<std::size_t>(f.nlocal()), 0.0);
    } else {
        const int mesh[3] = {k_mesh[0], k_mesh[1], k_mesh[2]};
        energy_ = launch_mio(f, scc_tol, max_scc, mix_beta, scc_forces,
                             with_stress, charge, multiplicity, mesh,
                             k_time_reversal, last_dq_, last_spin_,
                             last_stress_, scc_residual_, scc_iterations_);
    }

    // DFTB+U (Dudarev equal-filling on atomic Mulliken n) — post-SCC additive.
    if (any_u) {
        // Resolve Frame columns once at this launch boundary (13.7× rule).
        const int n_at = f.nlocal();
        const int* type = f.span(kAtom_type).data();
        double& energy_meta = f.meta.get(kEnergy);
        double& energy_el_meta = f.meta.get(kEnergyElectronic);

        auto apply_e_u = [&](const std::vector<double>& dq) {
            double eu = 0;
            if (static_cast<int>(dq.size()) != n_at)
                throw std::runtime_error("DFTB2+U: Mulliken Δq size mismatch");
            for (int a = 0; a < n_at; ++a) {
                const int Z = type[a];
                if (Z < 1 || Z >= static_cast<int>(u_eff.size()))
                    throw std::runtime_error(
                        "DFTB2+U: atomic number out of range");
                const double U = u_eff[static_cast<std::size_t>(Z)];
                if (U == 0.0)
                    continue;
                if (!(U > 0.0) || !std::isfinite(U))
                    throw std::invalid_argument(
                        "DFTB2+U: u_eff[Z] must be finite and > 0 when set");
                const auto& el = element_row(Z);
                const double n = static_cast<double>(el.n_val)
                                 - dq[static_cast<std::size_t>(a)];
                const double Norb = 2.0 * static_cast<double>(el.n_ao);
                if (Norb < 1.0)
                    throw std::runtime_error("DFTB2+U: atom has no AOs");
                eu += 0.5 * U * n * (1.0 - n / Norb);
            }
            return eu;
        };

        e_u_ = apply_e_u(last_dq_);
        energy_ += e_u_;
        energy_meta += e_u_;
        energy_el_meta += e_u_;

        if (u_force_fd) {
            // Central FD of E_tot = E_SCC + E_U (full re-SCC at each displaced
            // geometry). h = 1e-4 Å matches other DFTB force gates.
            //
            // Each probe re-enters launch_mio (many span resolutions). Same
            // pattern as PWEvaluator force/stress and MeanFieldMethod
            // accumulator guards: park the host launch window so nested
            // multi-SCC probes are not billed against this component's
            // single-launch budget (13.7× catches *loops that re-resolve*,
            // not intentional multi-geometry energy probes).
            constexpr double h = 1e-4;
            AccReal* fx = f.span(kAtom_fx);
            AccReal* fy = f.span(kAtom_fy);
            AccReal* fz = f.span(kAtom_fz);
            StorageReal* x = f.span(kAtom_x);
            StorageReal* y = f.span(kAtom_y);
            StorageReal* z = f.span(kAtom_z);
            for (int a = 0; a < n_at; ++a) {
                fx[a] = fy[a] = fz[a] = 0;
            }
            const int mesh[3] = {1, 1, 1};
            const auto saved_window = f.resolution_window();
            auto e_total_at = [&]() {
                // Zero frame energy so launch_mio's += is absolute for this
                // probe.
                energy_meta = 0;
                energy_el_meta = 0;
                std::vector<double> dq, spin;
                std::array<double, 6> st{};
                double res = 0;
                int it = 0;
                double e =
                    launch_mio(f, scc_tol, max_scc, mix_beta, false, false,
                               charge, multiplicity, mesh, k_time_reversal, dq,
                               spin, st, res, it);
                e += apply_e_u(dq);
                return e;
            };
            for (int a = 0; a < n_at; ++a) {
                StorageReal* coords[3] = {x, y, z};
                AccReal* forces[3] = {fx, fy, fz};
                for (int k = 0; k < 3; ++k) {
                    const StorageReal x0 = coords[k][a];
                    coords[k][a] = static_cast<StorageReal>(
                        static_cast<double>(x0) + h);
                    const double ep = e_total_at();
                    coords[k][a] = static_cast<StorageReal>(
                        static_cast<double>(x0) - h);
                    const double em = e_total_at();
                    coords[k][a] = x0;
                    forces[k][a] =
                        static_cast<AccReal>(-(ep - em) / (2.0 * h));
                }
            }
            f.restore_resolution_window(saved_window);
            // Restore frame energy to the equilibrium total.
            energy_meta = energy_;
            energy_el_meta = energy_;
        }
    }

    // Optional Grimme D3(BJ): geometry-only additive correction (no SCC re-entry).
    // Analytic forces always available from dftd3_pbe_bj; stress is the pair
    // virial of those forces (molecular) or strain-FD of E_disp (periodic).
    e_disp_ = 0.0;
    if (d3) {
        const int n_at = f.nlocal();
        const int* type = f.span(kAtom_type).data();
        const StorageReal* x = f.span(kAtom_x);
        const StorageReal* y = f.span(kAtom_y);
        const StorageReal* z = f.span(kAtom_z);

        std::vector<int> Z(static_cast<std::size_t>(n_at));
        std::vector<std::array<double, 3>> pos(static_cast<std::size_t>(n_at));
        for (int a = 0; a < n_at; ++a) {
            const int zi = type[a];
            if (!gpw::D3Params::supports_element(zi))
                throw std::runtime_error(
                    "DFTB2 D3: element Z=" + std::to_string(zi)
                    + " is not in the in-tree D3 table (no fallback)");
            Z[static_cast<std::size_t>(a)] = zi;
            pos[static_cast<std::size_t>(a)] = {
                static_cast<double>(x[a]) * units::kAngstromInBohr,
                static_cast<double>(y[a]) * units::kAngstromInBohr,
                static_cast<double>(z[a]) * units::kAngstromInBohr};
        }

        double cell[3] = {0.0, 0.0, 0.0};
        bool periodic = false;
        if (d3_periodic) {
            if (!f.box.is_orthorhombic())
                throw std::runtime_error(
                    "DFTB2 D3: d3_periodic requires an orthorhombic box");
            cell[0] = f.box.length(0) * units::kAngstromInBohr;
            cell[1] = f.box.length(1) * units::kAngstromInBohr;
            cell[2] = f.box.length(2) * units::kAngstromInBohr;
            periodic = true;
        }
        if (!(d3_cutoff > 0.0))
            throw std::runtime_error("DFTB2 D3: d3_cutoff must be positive");
        const gpw::D3Result res =
            gpw::dftd3_pbe_bj(Z, pos, cell, periodic, gpw::D3Params{},
                              d3_cutoff * units::kAngstromInBohr, d3_s6, d3_s8,
                              d3_a1, d3_a2,
                              /*cp2k_cn=*/false, /*calculate_c9=*/false);

        e_disp_ = res.energy * units::kHartreeInEv;
        energy_ += e_disp_;
        // Total only: dispersion is empirical geometry, not kEnergyElectronic.
        f.meta.get(kEnergy) += e_disp_;

        // Ha/Bohr -> eV/Angstrom for forces; also used for molecular virial.
        const double f_conv = units::kHartreeInEv * units::kAngstromInBohr;
        std::vector<std::array<double, 3>> f_d3_ev(
            static_cast<std::size_t>(n_at), {0.0, 0.0, 0.0});
        for (int a = 0; a < n_at; ++a) {
            const auto& fa = res.forces[static_cast<std::size_t>(a)];
            f_d3_ev[static_cast<std::size_t>(a)] = {
                fa[0] * f_conv, fa[1] * f_conv, fa[2] * f_conv};
        }

        if (with_forces) {
            AccReal* fx = f.span(kAtom_fx);
            AccReal* fy = f.span(kAtom_fy);
            AccReal* fz = f.span(kAtom_fz);
            for (int a = 0; a < n_at; ++a) {
                const auto& fa = f_d3_ev[static_cast<std::size_t>(a)];
                fx[a] += static_cast<AccReal>(fa[0]);
                fy[a] += static_cast<AccReal>(fa[1]);
                fz[a] += static_cast<AccReal>(fa[2]);
            }
        }

        if (with_stress) {
            const double vol = f.box.volume();
            if (!(vol > 0.0))
                throw std::runtime_error(
                    "DFTB2 D3: non-positive box volume for stress");
            // Analytic pair virial from dftd3 (image-aware when periodic).
            // W [Ha] from Bohr·(Ha/Bohr); σ [eV/Å³] = W·(Ha→eV) / V_Å³.
            const double w2e = units::kHartreeInEv / vol;
            auto& st = f.meta.get(kStress);
            for (int v = 0; v < 6; ++v) {
                const double sig = res.virial[v] * w2e;
                last_stress_[static_cast<std::size_t>(v)] += sig;
                // kStress is the extensive Voigt virial [eV] — which for this
                // term is the dftd3 pair virial itself, W·(Ha→eV), i.e. sig
                // without the 1/V that made it intensive.
                st[static_cast<std::size_t>(v)] +=
                    res.virial[v] * units::kHartreeInEv;
            }
        }
    }
}

// ── Linear-response TD-DFTB (Casida) ────────────────────────────────────
//
// Niehaus, Suhai, Della Sala, Lugli, Elstner, Seifert, Frauenheim,
// Phys. Rev. B 63, 085108 (2001). Lives in this translation unit so the
// Slater-Koster / gamma / SCC machinery above is REUSED (anonymous-namespace
// names stay visible for the rest of the TU) rather than extracted into
// one-shot helpers or duplicated in a second method.

TdDftbResult DFTB2::td_casida(const Frame& frame, int n_states,
                              TdDftbChannel channel, int n_occ_window,
                              int n_virt_window,
                              double coupling_scale) const {
    // Dense Casida path only; matches the RKS::tddft n_ov cap.
    constexpr int kMaxPairs = 512;

    if (sk_set != DftbSkSet::MioSketch)
        throw std::runtime_error(
            "DFTB2::td_casida: MioSketch SK path only (linear response is not "
            "implemented on the 3ob tables)");
    if (k_mesh[0] != 1 || k_mesh[1] != 1 || k_mesh[2] != 1)
        throw std::runtime_error(
            "DFTB2::td_casida: Gamma point only (k_mesh must be {1,1,1}); "
            "periodic linear response is not implemented");
    if (multiplicity != 1)
        throw std::runtime_error(
            "DFTB2::td_casida: closed-shell reference only (multiplicity must "
            "be 1); spin-polarized TD-DFTB is not implemented");
    if (n_states < 1)
        throw std::invalid_argument("DFTB2::td_casida: n_states must be >= 1");
    if (n_occ_window < 0 || n_virt_window < 0)
        throw std::invalid_argument(
            "DFTB2::td_casida: occupied/virtual window must be >= 0 (0 = all)");
    if (!std::isfinite(coupling_scale) || coupling_scale < 0.0)
        throw std::invalid_argument(
            "DFTB2::td_casida: coupling_scale must be finite and >= 0");
    if (!(mix_beta > 0.0) || mix_beta > 1.0)
        throw std::invalid_argument(
            "DFTB2::td_casida: mix_beta must be in (0, 1]");

    const int n_at = frame.nlocal();
    if (n_at < 1)
        throw std::invalid_argument("DFTB2::td_casida: empty frame");
    const int* type = frame.span(kAtom_type).data();
    const StorageReal* x = frame.span(kAtom_x).data();
    const StorageReal* y = frame.span(kAtom_y).data();
    const StorageReal* z = frame.span(kAtom_z).data();

    // Element rows + atomic-orbital offsets: the same walk launch_mio does.
    std::vector<const DftbElementRow*> el(static_cast<std::size_t>(n_at));
    std::vector<int> ao0(static_cast<std::size_t>(n_at) + 1, 0);
    double n_elec = -static_cast<double>(charge);
    for (int a = 0; a < n_at; ++a) {
        el[static_cast<std::size_t>(a)] = &element_row(type[a]);
        n_elec += el[static_cast<std::size_t>(a)]->n_val;
        ao0[static_cast<std::size_t>(a) + 1] =
            ao0[static_cast<std::size_t>(a)]
            + el[static_cast<std::size_t>(a)]->n_ao;
    }
    const int n_ao = ao0[static_cast<std::size_t>(n_at)];
    if (n_ao < 1)
        throw std::runtime_error("DFTB2::td_casida: no basis functions");

    const Occupancy occ = resolve_occupancy(n_elec, multiplicity);
    if (occ.unrestricted)
        throw std::runtime_error(
            "DFTB2::td_casida: odd electron count — closed-shell reference "
            "required");
    const int n_occ = occ.n_alpha;
    const int n_virt = n_ao - n_occ;
    if (n_occ < 1 || n_virt < 1)
        throw std::runtime_error(
            "DFTB2::td_casida: need at least one occupied and one virtual "
            "orbital (n_occ=" + std::to_string(n_occ)
            + ", n_virt=" + std::to_string(n_virt) + ")");

    // Ground state at THIS geometry — no cached state, no launch-order coupling.
    const SccResult gs = scc_energy_restricted(n_at, n_ao, n_occ, el, ao0, type,
                                               x, y, z, frame.box, scc_tol,
                                               max_scc, mix_beta);

    // Overlap and gamma from the same builders the energy path uses.
    linalg::RealMatrix H0, S;
    build_h0_s(n_at, n_ao, el, ao0, type, x, y, z, frame.box, H0, S);
    std::vector<double> gamma(static_cast<std::size_t>(n_at) * n_at, 0.0);
    for (int a = 0; a < n_at; ++a)
        for (int b = 0; b < n_at; ++b)
            gamma[static_cast<std::size_t>(a) * n_at + b] =
                gamma_pair(a, b, el, x, y, z, frame.box);

    // Active window: the n_o highest occupied and n_v lowest virtual orbitals.
    const int n_o = (n_occ_window == 0) ? n_occ : std::min(n_occ_window, n_occ);
    const int n_v =
        (n_virt_window == 0) ? n_virt : std::min(n_virt_window, n_virt);
    const int n_pair = n_o * n_v;
    if (n_pair > kMaxPairs)
        throw std::runtime_error(
            "DFTB2::td_casida: pair space " + std::to_string(n_pair)
            + " exceeds the dense cap " + std::to_string(kMaxPairs)
            + " — narrow n_occ_window / n_virt_window (no iterative solver)");
    if (n_states > n_pair)
        throw std::invalid_argument(
            "DFTB2::td_casida: n_states " + std::to_string(n_states)
            + " exceeds the " + std::to_string(n_pair)
            + " occupied->virtual pairs in the window");

    TdDftbResult out;
    out.n_pair = n_pair;
    out.n_atoms = n_at;
    out.channel = channel;
    out.coupling_scale = coupling_scale;
    out.pair_occ.resize(static_cast<std::size_t>(n_pair));
    out.pair_virt.resize(static_cast<std::size_t>(n_pair));
    std::vector<double> de(static_cast<std::size_t>(n_pair), 0.0);
    for (int io = 0; io < n_o; ++io) {
        const int i = n_occ - n_o + io;
        for (int iv = 0; iv < n_v; ++iv) {
            const int a = n_occ + iv;
            const std::size_t p = static_cast<std::size_t>(io) * n_v + iv;
            out.pair_occ[p] = i;
            out.pair_virt[p] = a;
            de[p] = gs.eps(a) - gs.eps(i);
            if (!(de[p] > 0.0))
                throw std::runtime_error(
                    "DFTB2::td_casida: non-positive orbital gap eps_"
                    + std::to_string(a) + " - eps_" + std::to_string(i) + " = "
                    + std::to_string(de[p])
                    + " eV — the Casida sqrt(eps_a - eps_i) form requires a "
                      "gapped closed-shell reference");
        }
    }

    // Mulliken transition charges q^{ia}_A = 1/2 sum_{mu in A}
    //   (c_{mu i} (S c_a)_mu + c_{mu a} (S c_i)_mu).
    const linalg::RealMatrix SC = xt::linalg::dot(S, gs.C);
    std::vector<int> ao_atom(static_cast<std::size_t>(n_ao), 0);
    for (int a = 0; a < n_at; ++a)
        for (int m = ao0[static_cast<std::size_t>(a)];
             m < ao0[static_cast<std::size_t>(a) + 1]; ++m)
            ao_atom[static_cast<std::size_t>(m)] = a;
    out.transition_charge.assign(
        static_cast<std::size_t>(n_pair) * n_at, 0.0);
    for (int p = 0; p < n_pair; ++p) {
        const int i = out.pair_occ[static_cast<std::size_t>(p)];
        const int a = out.pair_virt[static_cast<std::size_t>(p)];
        double* qp = out.transition_charge.data()
                     + static_cast<std::size_t>(p) * n_at;
        for (int mu = 0; mu < n_ao; ++mu)
            qp[ao_atom[static_cast<std::size_t>(mu)]] +=
                0.5 * (gs.C(mu, i) * SC(mu, a) + gs.C(mu, a) * SC(mu, i));
    }

    // Coupling kernel K and the Casida matrix Omega [eV^2]. Assembled exactly
    // as written — never re-symmetrized, so its symmetry stays a real check.
    out.casida_ev2.assign(static_cast<std::size_t>(n_pair) * n_pair, 0.0);
    for (int p = 0; p < n_pair; ++p) {
        const double* qp = out.transition_charge.data()
                           + static_cast<std::size_t>(p) * n_at;
        for (int r = 0; r < n_pair; ++r) {
            const double* qr = out.transition_charge.data()
                               + static_cast<std::size_t>(r) * n_at;
            double k = 0.0;
            if (channel == TdDftbChannel::singlet) {
                for (int a = 0; a < n_at; ++a)
                    for (int b = 0; b < n_at; ++b)
                        k += qp[a]
                             * gamma[static_cast<std::size_t>(a) * n_at + b]
                             * qr[b];
            } else {
                // Magnetization kernel is on-site: gamma^m_AB = delta_AB W_A.
                for (int a = 0; a < n_at; ++a)
                    k += qp[a] * el[static_cast<std::size_t>(a)]->spin_w
                         * qr[a];
            }
            k *= coupling_scale;
            double om = 2.0
                        * std::sqrt(de[static_cast<std::size_t>(p)]) * k
                        * std::sqrt(de[static_cast<std::size_t>(r)]);
            if (p == r)
                om += de[static_cast<std::size_t>(p)]
                      * de[static_cast<std::size_t>(p)];
            out.casida_ev2[static_cast<std::size_t>(p) * n_pair + r] = om;
        }
    }

    linalg::RealMatrix omega = xt::zeros<double>({n_pair, n_pair});
    for (int p = 0; p < n_pair; ++p)
        for (int r = 0; r < n_pair; ++r)
            omega(p, r) =
                out.casida_ev2[static_cast<std::size_t>(p) * n_pair + r];
    // Full eigh: eigenvalues (ascending) + orthonormal eigenvectors F.
    auto [lambda, F] = xt::linalg::eigh(omega);

    // Pair transition dipoles μ_ia = Σ_A q^{ia}_A R_A [e·Å].
    std::vector<double> mu_pair(static_cast<std::size_t>(n_pair) * 3, 0.0);
    for (int p = 0; p < n_pair; ++p) {
        const double* qp = out.transition_charge.data()
                           + static_cast<std::size_t>(p) * n_at;
        for (int a = 0; a < n_at; ++a) {
            mu_pair[static_cast<std::size_t>(p) * 3 + 0] +=
                qp[a] * static_cast<double>(x[a]);
            mu_pair[static_cast<std::size_t>(p) * 3 + 1] +=
                qp[a] * static_cast<double>(y[a]);
            mu_pair[static_cast<std::size_t>(p) * 3 + 2] +=
                qp[a] * static_cast<double>(z[a]);
        }
    }

    out.energy_ev.resize(static_cast<std::size_t>(n_states));
    out.oscillator_strength.assign(static_cast<std::size_t>(n_states), 0.0);
    out.transition_dipole.assign(static_cast<std::size_t>(n_states) * 3, 0.0);
    for (int s = 0; s < n_states; ++s) {
        const double lam = lambda(s);
        if (lam < 0.0)
            throw std::runtime_error(
                "DFTB2::td_casida: negative Casida root omega^2 = "
                + std::to_string(lam)
                + " eV^2 (imaginary excitation — response instability); no "
                  "fallback value");
        const double w = std::sqrt(lam);
        out.energy_ev[static_cast<std::size_t>(s)] = w;

        // Triplet is spin-forbidden for electric dipole → f = 0.
        if (channel != TdDftbChannel::singlet || !(w > 0.0))
            continue;
        // t_α = Σ_p √2 F_{p s} √Δε_p μ_{p,α};  f = (2/3) |t|² / ω
        // (Casida ΩF=ω²F normalisation, length form).
        double t[3] = {0, 0, 0};
        for (int p = 0; p < n_pair; ++p) {
            const double amp =
                std::sqrt(2.0) * F(p, s)
                * std::sqrt(de[static_cast<std::size_t>(p)]);
            t[0] += amp * mu_pair[static_cast<std::size_t>(p) * 3 + 0];
            t[1] += amp * mu_pair[static_cast<std::size_t>(p) * 3 + 1];
            t[2] += amp * mu_pair[static_cast<std::size_t>(p) * 3 + 2];
        }
        out.transition_dipole[static_cast<std::size_t>(s) * 3 + 0] = t[0] / w;
        out.transition_dipole[static_cast<std::size_t>(s) * 3 + 1] = t[1] / w;
        out.transition_dipole[static_cast<std::size_t>(s) * 3 + 2] = t[2] / w;
        const double t2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
        out.oscillator_strength[static_cast<std::size_t>(s)] =
            (2.0 / 3.0) * t2 / w;
    }

    std::vector<double> de_sorted = de;
    std::sort(de_sorted.begin(), de_sorted.end());
    out.ks_gap_ev.assign(de_sorted.begin(),
                         de_sorted.begin() + n_states);
    return out;
}

} // namespace semiempirical
} // namespace cpu
} // namespace ATV
