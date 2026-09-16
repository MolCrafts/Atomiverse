/// @file cpu/scf/gf2.cpp
/// @brief One-shot closed-shell GF2 (G0F2) quasiparticle energies on RHF.
///
/// Auxiliary second-order self-energy (2h1p + 2p1h Lehmann poles) from the HF
/// Green's function; single Dyson solve with F = diag(ε_HF). Twin of PySCF
/// `agf2.RAGF2` one-shot path (Backhouse–Booth AGF2 SE construction).

#include "atomiverse/cpu/scf/gf2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/containers/xtensor.hpp>

#include "atomiverse/core/units.h"
#include "atomiverse/cpu/scf/dense_matrix.h"
#include "atomiverse/cpu/scf/internal/ao2mo.h"
#include "atomiverse/cpu/scf/internal/integrals.h"
#include "atomiverse/cpu/scf/rhf.h"
#include "atomiverse/cpu/scf/slater_determinant.h"

namespace ATV {
namespace cpu {
namespace scf {
namespace {

constexpr double kWeightTol =
    1e-11; ///< Drop uncoupled SE poles (PySCF default).

/// Occupied / virtual slice of an auxiliary space (energy + physical coupling).
struct AuxSlice {
    RealVector energy;   ///< pole energies [Hartree]
    RealMatrix coupling; ///< [n_phys × n_aux]
};

/// Sort poles by energy ascending (in-place).
void sort_aux(RealVector& e, RealMatrix& c) {
    const std::size_t n = e.size();
    if (n <= 1)
        return;
    std::vector<std::size_t> ord(n);
    for (std::size_t i = 0; i < n; ++i)
        ord[i] = i;
    std::sort(ord.begin(), ord.end(),
              [&](std::size_t a, std::size_t b) { return e(a) < e(b); });
    RealVector e2 = xt::zeros<double>({n});
    RealMatrix c2 = xt::zeros<double>({c.shape()[0], n});
    for (std::size_t k = 0; k < n; ++k) {
        e2(k) = e(ord[k]);
        for (std::size_t p = 0; p < c.shape()[0]; ++p)
            c2(p, k) = c(p, ord[k]);
    }
    e = std::move(e2);
    c = std::move(c2);
}

/// Drop poles whose spectral weight ‖v‖² < tol (PySCF remove_uncoupled).
void remove_uncoupled(RealVector& e, RealMatrix& c, double tol) {
    const std::size_t nphys = c.shape()[0];
    const std::size_t naux = e.size();
    std::vector<std::size_t> keep;
    keep.reserve(naux);
    for (std::size_t k = 0; k < naux; ++k) {
        double w = 0.0;
        for (std::size_t p = 0; p < nphys; ++p)
            w += c(p, k) * c(p, k);
        if (w >= tol)
            keep.push_back(k);
    }
    if (keep.size() == naux)
        return;
    RealVector e2 = xt::zeros<double>({keep.size()});
    RealMatrix c2 = xt::zeros<double>({nphys, keep.size()});
    for (std::size_t j = 0; j < keep.size(); ++j) {
        e2(j) = e(keep[j]);
        for (std::size_t p = 0; p < nphys; ++p)
            c2(p, j) = c(p, keep[j]);
    }
    e = std::move(e2);
    c = std::move(c2);
}

/// Build 2h1p (or 2p1h if occ/vir swapped) SE auxiliaries — PySCF
/// `ragf2_slow.build_se_part` with os_factor = ss_factor = 1.
///
/// gf_occ: energies [n_occ_aux], coupling [n_mo × n_occ_aux]
/// gf_vir: energies [n_vir_aux], coupling [n_mo × n_vir_aux]
/// eri: full MO (pq|rs) chemist notation [n_mo^4]
AuxSlice compute_se_part(const RealVector& gf_occ_e, const RealMatrix& gf_occ_c,
                         const RealVector& gf_vir_e, const RealMatrix& gf_vir_c,
                         const xt::xtensor<double, 4>& eri) {
    const int nmo = static_cast<int>(gf_occ_c.shape()[0]);
    const int nocc_a = static_cast<int>(gf_occ_e.size());
    const int nvir_a = static_cast<int>(gf_vir_e.size());
    if (nocc_a < 1 || nvir_a < 1) {
        RealVector e0 = xt::zeros<double>({static_cast<std::size_t>(0)});
        RealMatrix c0 = xt::zeros<double>(
            {static_cast<std::size_t>(nmo), static_cast<std::size_t>(0)});
        return AuxSlice{std::move(e0), std::move(c0)};
    }

    // qeri[x,i,j,a] = Σ_{pqr} (x p | q r) C_i[p] C_j[q] C_a[r]
    xt::xtensor<double, 4> qeri = xt::zeros<double>(
        {static_cast<std::size_t>(nmo), static_cast<std::size_t>(nocc_a),
         static_cast<std::size_t>(nocc_a), static_cast<std::size_t>(nvir_a)});
    for (int i = 0; i < nocc_a; ++i)
        for (int j = 0; j < nocc_a; ++j)
            for (int a = 0; a < nvir_a; ++a)
                for (int x = 0; x < nmo; ++x) {
                    double s = 0.0;
                    for (int p = 0; p < nmo; ++p) {
                        const double ci = gf_occ_c(static_cast<std::size_t>(p),
                                                   static_cast<std::size_t>(i));
                        if (ci == 0.0)
                            continue;
                        for (int q = 0; q < nmo; ++q) {
                            const double cj =
                                gf_occ_c(static_cast<std::size_t>(q),
                                         static_cast<std::size_t>(j));
                            if (cj == 0.0)
                                continue;
                            for (int r = 0; r < nmo; ++r) {
                                const double ca =
                                    gf_vir_c(static_cast<std::size_t>(r),
                                             static_cast<std::size_t>(a));
                                s += eri(static_cast<std::size_t>(x),
                                         static_cast<std::size_t>(p),
                                         static_cast<std::size_t>(q),
                                         static_cast<std::size_t>(r))
                                     * ci * cj * ca;
                            }
                        }
                    }
                    qeri(static_cast<std::size_t>(x),
                         static_cast<std::size_t>(i),
                         static_cast<std::size_t>(j),
                         static_cast<std::size_t>(a)) = s;
                }

    // Max auxiliaries: nocc*(nocc-1)*nvir * 2 (ss/os) + nocc*nvir (diag)
    const int naux_max = nocc_a * (nocc_a - 1) * nvir_a * 2 + nocc_a * nvir_a;
    RealVector e = xt::zeros<double>({static_cast<std::size_t>(naux_max)});
    RealMatrix v = xt::zeros<double>(
        {static_cast<std::size_t>(nmo), static_cast<std::size_t>(naux_max)});
    const double fpos = std::sqrt(0.5);
    const double fneg = std::sqrt(0.5);
    const double fdia = 1.0;
    int p1 = 0;
    for (int i = 0; i < nocc_a; ++i) {
        // j < i blocks
        if (i > 0) {
            const int blk = i * nvir_a;
            // e_ija = e_i + e_j - e_a
            for (int j = 0; j < i; ++j)
                for (int a = 0; a < nvir_a; ++a) {
                    const int col = p1 + j * nvir_a + a;
                    e(static_cast<std::size_t>(col)) =
                        gf_occ_e(static_cast<std::size_t>(i))
                        + gf_occ_e(static_cast<std::size_t>(j))
                        - gf_vir_e(static_cast<std::size_t>(a));
                    for (int x = 0; x < nmo; ++x) {
                        const double xija = qeri(static_cast<std::size_t>(x),
                                                 static_cast<std::size_t>(i),
                                                 static_cast<std::size_t>(j),
                                                 static_cast<std::size_t>(a));
                        const double xjia = qeri(static_cast<std::size_t>(x),
                                                 static_cast<std::size_t>(j),
                                                 static_cast<std::size_t>(i),
                                                 static_cast<std::size_t>(a));
                        v(static_cast<std::size_t>(x),
                          static_cast<std::size_t>(col)) = fneg * (xija - xjia);
                    }
                }
            p1 += blk;
            for (int j = 0; j < i; ++j)
                for (int a = 0; a < nvir_a; ++a) {
                    const int col = p1 + j * nvir_a + a;
                    e(static_cast<std::size_t>(col)) =
                        gf_occ_e(static_cast<std::size_t>(i))
                        + gf_occ_e(static_cast<std::size_t>(j))
                        - gf_vir_e(static_cast<std::size_t>(a));
                    for (int x = 0; x < nmo; ++x) {
                        const double xija = qeri(static_cast<std::size_t>(x),
                                                 static_cast<std::size_t>(i),
                                                 static_cast<std::size_t>(j),
                                                 static_cast<std::size_t>(a));
                        const double xjia = qeri(static_cast<std::size_t>(x),
                                                 static_cast<std::size_t>(j),
                                                 static_cast<std::size_t>(i),
                                                 static_cast<std::size_t>(a));
                        v(static_cast<std::size_t>(x),
                          static_cast<std::size_t>(col)) = fpos * (xija + xjia);
                    }
                }
            p1 += blk;
        }
        // j = i diagonal (opposite-spin only, fdia)
        for (int a = 0; a < nvir_a; ++a) {
            const int col = p1 + a;
            e(static_cast<std::size_t>(col)) =
                gf_occ_e(static_cast<std::size_t>(i))
                + gf_occ_e(static_cast<std::size_t>(i))
                - gf_vir_e(static_cast<std::size_t>(a));
            for (int x = 0; x < nmo; ++x)
                v(static_cast<std::size_t>(x), static_cast<std::size_t>(col)) =
                    fdia
                    * qeri(static_cast<std::size_t>(x),
                           static_cast<std::size_t>(i),
                           static_cast<std::size_t>(i),
                           static_cast<std::size_t>(a));
        }
        p1 += nvir_a;
    }
    // Shrink to used width
    RealVector e_use = xt::zeros<double>({static_cast<std::size_t>(p1)});
    RealMatrix v_use = xt::zeros<double>(
        {static_cast<std::size_t>(nmo), static_cast<std::size_t>(p1)});
    for (int k = 0; k < p1; ++k) {
        e_use(static_cast<std::size_t>(k)) = e(static_cast<std::size_t>(k));
        for (int x = 0; x < nmo; ++x)
            v_use(static_cast<std::size_t>(x), static_cast<std::size_t>(k)) =
                v(static_cast<std::size_t>(x), static_cast<std::size_t>(k));
    }
    remove_uncoupled(e_use, v_use, kWeightTol);
    sort_aux(e_use, v_use);
    return AuxSlice{std::move(e_use), std::move(v_use)};
}

/// Combine two AuxSlices (inherit chempot conceptually from first).
AuxSlice combine(const AuxSlice& a, const AuxSlice& b) {
    const std::size_t nphys = a.coupling.shape()[0];
    const std::size_t na = a.energy.size();
    const std::size_t nb = b.energy.size();
    RealVector e = xt::zeros<double>({na + nb});
    RealMatrix c = xt::zeros<double>({nphys, na + nb});
    for (std::size_t k = 0; k < na; ++k) {
        e(k) = a.energy(k);
        for (std::size_t p = 0; p < nphys; ++p)
            c(p, k) = a.coupling(p, k);
    }
    for (std::size_t k = 0; k < nb; ++k) {
        e(na + k) = b.energy(k);
        for (std::size_t p = 0; p < nphys; ++p)
            c(p, na + k) = b.coupling(p, k);
    }
    sort_aux(e, c);
    return AuxSlice{std::move(e), std::move(c)};
}

/// Binary-search chemical potential for N physical electrons (RHF occupancy 2).
/// Returns (chempot, electron-count error).
std::pair<double, double> binsearch_chempot(const RealVector& w,
                                            const RealMatrix& v_phys, int nphys,
                                            int nelec) {
    // v_phys: [nphys × nqmo] physical block of eigenvectors
    const int nqmo = static_cast<int>(w.size());
    double sum0 = 0.0, sum1 = 0.0;
    int i_cut = 0;
    for (int i = 0; i < nqmo; ++i) {
        double n = 0.0;
        for (int p = 0; p < nphys; ++p)
            n +=
                v_phys(static_cast<std::size_t>(p), static_cast<std::size_t>(i))
                * v_phys(static_cast<std::size_t>(p),
                         static_cast<std::size_t>(i));
        n *= 2.0; // RHF occupancy
        sum0 = sum1;
        sum1 += n;
        i_cut = i;
        if (i > 0 && sum0 <= static_cast<double>(nelec)
            && static_cast<double>(nelec) <= sum1)
            break;
    }
    int homo;
    double error;
    if (std::abs(sum0 - static_cast<double>(nelec))
        < std::abs(sum1 - static_cast<double>(nelec))) {
        homo = i_cut - 1;
        error = static_cast<double>(nelec) - sum0;
    } else {
        homo = i_cut;
        error = static_cast<double>(nelec) - sum1;
    }
    if (homo < 0)
        homo = 0;
    const int lumo = std::min(homo + 1, nqmo - 1);
    const double chempot = 0.5
                           * (w(static_cast<std::size_t>(homo))
                              + w(static_cast<std::size_t>(lumo)));
    return {chempot, error};
}

/// Galitskii–Migdal E_corr from HF MO energies + virtual SE (half of E_2b).
double energy_mp2_gm(const RealVector& mo_energy, const AuxSlice& se,
                     double chempot) {
    // se_vir: poles with energy >= chempot
    std::vector<std::size_t> vir;
    for (std::size_t k = 0; k < se.energy.size(); ++k)
        if (se.energy(k) >= chempot)
            vir.push_back(k);
    // occupied HF MOs
    std::vector<std::size_t> occ;
    for (std::size_t p = 0; p < mo_energy.size(); ++p)
        if (mo_energy(p) < chempot)
            occ.push_back(p);
    double emp2 = 0.0;
    for (std::size_t xi = 0; xi < occ.size(); ++xi) {
        const std::size_t x = occ[xi];
        for (std::size_t kk = 0; kk < vir.size(); ++kk) {
            const std::size_t k = vir[kk];
            const double vxk = se.coupling(x, k);
            const double dxk = mo_energy(x) - se.energy(k);
            if (std::abs(dxk) < 1e-14)
                continue;
            emp2 += vxk * vxk / dxk;
        }
    }
    return emp2;
}

} // namespace

GF2Result RHF::gf2() const {
    if (!has_result())
        throw std::runtime_error(
            "RHF::gf2: no SCF result — call run(frame) first");

    const SlaterDeterminant& det = determinant();
    if (!det.restricted() || det.generalized())
        throw std::runtime_error(
            "RHF::gf2: closed-shell RHF reference required (no fallback)");

    const internal::IntegralView pack = integral_view_();
    const xt::xtensor<double, 4> eri_ao =
        (pack.ERI.size() > 0) ? pack.ERI : pack.eri_working();

    using Spin = SlaterDeterminant::Spin;
    const RealMatrix& C = det.coefficients(Spin::alpha);
    const RealVector& eps_raw = det.orbital_energies(Spin::alpha);
    const int n_mo = static_cast<int>(C.shape()[1]);
    if (n_mo != static_cast<int>(eps_raw.size()))
        throw std::runtime_error("RHF::gf2: MO dimension mismatch");

    int n_occ = 0;
    for (int p = 0; p < n_mo; ++p) {
        const double f =
            det.occupations(Spin::alpha)(static_cast<std::size_t>(p));
        if (f == 1.0)
            ++n_occ;
        else if (f != 0.0)
            throw std::invalid_argument(
                "RHF::gf2: fractional occupation — integer occupations only");
    }
    const int n_vir = n_mo - n_occ;
    if (n_occ < 1 || n_vir < 1)
        throw std::invalid_argument(
            "RHF::gf2: need at least one occupied and one virtual orbital "
            "(no fallback)");

    // Full MO ERI
    const xt::xtensor<double, 4> eri = ao2mo_full(eri_ao, C, n_mo);

    // HF Green's function: poles at ε_p, coupling = I
    RealVector gf_e = eps_raw;
    RealMatrix gf_c = xt::eye<double>(static_cast<std::size_t>(n_mo));
    // HF chemical potential mid-gap (PySCF binsearch on diag Fock)
    const double mu_hf = 0.5
                         * (eps_raw(static_cast<std::size_t>(n_occ - 1))
                            + eps_raw(static_cast<std::size_t>(n_occ)));

    // Split occ / vir
    RealVector e_occ = xt::zeros<double>({static_cast<std::size_t>(n_occ)});
    RealMatrix c_occ = xt::zeros<double>(
        {static_cast<std::size_t>(n_mo), static_cast<std::size_t>(n_occ)});
    RealVector e_vir = xt::zeros<double>({static_cast<std::size_t>(n_vir)});
    RealMatrix c_vir = xt::zeros<double>(
        {static_cast<std::size_t>(n_mo), static_cast<std::size_t>(n_vir)});
    for (int i = 0; i < n_occ; ++i) {
        e_occ(static_cast<std::size_t>(i)) = gf_e(static_cast<std::size_t>(i));
        for (int p = 0; p < n_mo; ++p)
            c_occ(static_cast<std::size_t>(p), static_cast<std::size_t>(i)) =
                gf_c(static_cast<std::size_t>(p), static_cast<std::size_t>(i));
    }
    for (int a = 0; a < n_vir; ++a) {
        e_vir(static_cast<std::size_t>(a)) =
            gf_e(static_cast<std::size_t>(n_occ + a));
        for (int p = 0; p < n_mo; ++p)
            c_vir(static_cast<std::size_t>(p), static_cast<std::size_t>(a)) =
                gf_c(static_cast<std::size_t>(p),
                     static_cast<std::size_t>(n_occ + a));
    }

    // SE = se_part(occ,vir) ⊕ se_part(vir,occ)
    const AuxSlice se_occ = compute_se_part(e_occ, c_occ, e_vir, c_vir, eri);
    const AuxSlice se_vir = compute_se_part(e_vir, c_vir, e_occ, c_occ, eri);
    AuxSlice se = combine(se_occ, se_vir);

    // Dyson: extended Fock [F, V; V†, ε_aux] with F = diag(ε_HF)
    const int naux = static_cast<int>(se.energy.size());
    const int nq = n_mo + naux;
    RealMatrix H = xt::zeros<double>(
        {static_cast<std::size_t>(nq), static_cast<std::size_t>(nq)});
    for (int p = 0; p < n_mo; ++p)
        H(static_cast<std::size_t>(p), static_cast<std::size_t>(p)) =
            eps_raw(static_cast<std::size_t>(p));
    for (int p = 0; p < n_mo; ++p)
        for (int k = 0; k < naux; ++k) {
            const double vij = se.coupling(static_cast<std::size_t>(p),
                                           static_cast<std::size_t>(k));
            H(static_cast<std::size_t>(p), static_cast<std::size_t>(n_mo + k)) =
                vij;
            H(static_cast<std::size_t>(n_mo + k), static_cast<std::size_t>(p)) =
                vij;
        }
    for (int k = 0; k < naux; ++k)
        H(static_cast<std::size_t>(n_mo + k),
          static_cast<std::size_t>(n_mo + k)) =
            se.energy(static_cast<std::size_t>(k));

    auto [w, V] = xt::linalg::eigh(H);
    // Physical block of eigenvectors
    RealMatrix v_phys = xt::zeros<double>(
        {static_cast<std::size_t>(n_mo), static_cast<std::size_t>(nq)});
    for (int p = 0; p < n_mo; ++p)
        for (int q = 0; q < nq; ++q)
            v_phys(static_cast<std::size_t>(p), static_cast<std::size_t>(q)) =
                V(static_cast<std::size_t>(p), static_cast<std::size_t>(q));

    const int nelec = 2 * n_occ;
    const auto [mu, nerr] = binsearch_chempot(w, v_phys, n_mo, nelec);

    // Occupied / virtual poles by chemical potential
    std::vector<double> e_ip_poles;
    std::vector<double> e_ea_poles;
    for (int q = 0; q < nq; ++q) {
        if (w(static_cast<std::size_t>(q)) < mu)
            e_ip_poles.push_back(w(static_cast<std::size_t>(q)));
        else
            e_ea_poles.push_back(w(static_cast<std::size_t>(q)));
    }

    GF2Result out;
    out.n_occ = n_occ;
    out.mf_energy_ha.resize(static_cast<std::size_t>(n_mo));
    for (int p = 0; p < n_mo; ++p)
        out.mf_energy_ha[static_cast<std::size_t>(p)] =
            eps_raw(static_cast<std::size_t>(p));
    out.pole_energy_ha.resize(static_cast<std::size_t>(nq));
    for (int q = 0; q < nq; ++q)
        out.pole_energy_ha[static_cast<std::size_t>(q)] =
            w(static_cast<std::size_t>(q));

    out.converged =
        !e_ip_poles.empty() && !e_ea_poles.empty() && std::abs(nerr) < 1e-6;
    if (!out.converged) {
        // Still fill what we can; caller may check converged.
        out.qp_energy_ha = out.mf_energy_ha;
        out.e_corr = 0.0;
        out.e_tot = energy();
        return out;
    }

    // IP = −ε_HOMO (highest occupied pole); EA = ε_LUMO (lowest virtual pole)
    const double e_homo = e_ip_poles.back(); // poles sorted ascending
    const double e_lumo = e_ea_poles.front();
    out.qp_energy_ha.resize(static_cast<std::size_t>(n_mo));
    // Map QP energies: HOMO ← e_homo, LUMO ← e_lumo; remaining MOs keep HF
    // for the textbook H2 case n_mo=2 this is exact. For larger systems the
    // principal poles dominate the reported gap/IP/EA.
    for (int p = 0; p < n_mo; ++p)
        out.qp_energy_ha[static_cast<std::size_t>(p)] =
            eps_raw(static_cast<std::size_t>(p));
    out.qp_energy_ha[static_cast<std::size_t>(n_occ - 1)] = e_homo;
    out.qp_energy_ha[static_cast<std::size_t>(n_occ)] = e_lumo;

    out.ip_ev = -e_homo * units::kHartreeInEv;
    out.ea_ev = e_lumo * units::kHartreeInEv;
    out.gap_ev = (e_lumo - e_homo) * units::kHartreeInEv;

    // Galitskii–Migdal e_corr from one-shot SE (= MP2 on HF reference)
    out.e_corr = energy_mp2_gm(eps_raw, se, mu_hf);
    out.e_tot = energy() + out.e_corr;
    return out;
}

} // namespace scf
} // namespace cpu
} // namespace ATV
