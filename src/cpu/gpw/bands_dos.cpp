/// @file cpu/gpw/bands_dos.cpp
/// @brief Multi-k GPW density-of-states post-processing.

#include "atomiverse/cpu/gpw/bands_dos.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ATV {
namespace cpu {
namespace gpw {

double gap_from_spectrum(const std::vector<std::vector<double>>& evals,
                         int n_occ, double& e_vbm, double& e_cbm, int& vbm_k,
                         int& cbm_k, int& vbm_b, int& cbm_b) {
    if (evals.empty())
        throw std::invalid_argument("gap_from_spectrum: empty spectrum");
    if (n_occ < 1)
        throw std::invalid_argument("gap_from_spectrum: n_occ < 1");
    const int n_bands = static_cast<int>(evals[0].size());
    if (n_bands < n_occ + 1)
        throw std::invalid_argument("gap_from_spectrum: need at least one "
                                    "empty band (n_bands > n_occ)");

    e_vbm = -1e300;
    e_cbm = 1e300;
    vbm_k = cbm_k = 0;
    vbm_b = n_occ - 1;
    cbm_b = n_occ;

    for (std::size_t ik = 0; ik < evals.size(); ++ik) {
        if (static_cast<int>(evals[ik].size()) < n_occ + 1)
            throw std::invalid_argument(
                "gap_from_spectrum: inconsistent band count");
        const double v = evals[ik][static_cast<std::size_t>(n_occ - 1)];
        const double c = evals[ik][static_cast<std::size_t>(n_occ)];
        if (v > e_vbm) {
            e_vbm = v;
            vbm_k = static_cast<int>(ik);
            vbm_b = n_occ - 1;
        }
        if (c < e_cbm) {
            e_cbm = c;
            cbm_k = static_cast<int>(ik);
            cbm_b = n_occ;
        }
    }
    return e_cbm - e_vbm;
}

void gaussian_dos(const std::vector<double>& evals,
                  const std::vector<double>& weights, double e_min,
                  double e_max, int n_bins, double sigma,
                  std::vector<double>& energy, std::vector<double>& dos,
                  double spin_degeneracy) {
    if (evals.size() != weights.size())
        throw std::invalid_argument(
            "gaussian_dos: evals/weights size mismatch");
    if (n_bins < 2)
        throw std::invalid_argument("gaussian_dos: n_bins < 2");
    if (!(sigma > 0.0))
        throw std::invalid_argument("gaussian_dos: sigma must be > 0");
    if (!(e_max > e_min))
        throw std::invalid_argument("gaussian_dos: need e_max > e_min");
    if (!(spin_degeneracy > 0.0))
        throw std::invalid_argument(
            "gaussian_dos: spin_degeneracy must be > 0");

    energy.resize(static_cast<std::size_t>(n_bins));
    dos.assign(static_cast<std::size_t>(n_bins), 0.0);
    const double de = (e_max - e_min) / static_cast<double>(n_bins - 1);
    const double pref = spin_degeneracy
                        / (std::sqrt(2.0 * 3.14159265358979323846) * sigma);
    const double inv_2s2 = 1.0 / (2.0 * sigma * sigma);

    for (int i = 0; i < n_bins; ++i)
        energy[static_cast<std::size_t>(i)] =
            e_min + de * static_cast<double>(i);

    for (std::size_t n = 0; n < evals.size(); ++n) {
        const double e = evals[n];
        const double w = weights[n];
        for (int i = 0; i < n_bins; ++i) {
            const double x = energy[static_cast<std::size_t>(i)] - e;
            dos[static_cast<std::size_t>(i)] +=
                w * pref * std::exp(-x * x * inv_2s2);
        }
    }
}

double integrate_dos_to(const std::vector<double>& energy,
                        const std::vector<double>& dos, double e_cut) {
    if (energy.size() != dos.size())
        throw std::invalid_argument(
            "integrate_dos_to: energy/dos size mismatch");
    if (energy.size() < 2)
        throw std::invalid_argument("integrate_dos_to: need >= 2 bins");

    const double de = (energy.back() - energy.front())
                      / static_cast<double>(static_cast<int>(energy.size()) - 1);
    double integ = 0;
    for (std::size_t i = 0; i + 1 < energy.size(); ++i) {
        if (energy[i] > e_cut)
            break;
        const double e1 = energy[i];
        const double e2 = energy[i + 1];
        const double d1 = dos[i];
        const double d2 = dos[i + 1];
        if (e2 <= e_cut) {
            integ += 0.5 * (d1 + d2) * de;
        } else {
            const double t = (e_cut - e1) / (e2 - e1);
            const double d_c = d1 + t * (d2 - d1);
            integ += 0.5 * (d1 + d_c) * (e_cut - e1);
            break;
        }
    }
    return integ;
}

namespace {

std::vector<std::vector<double>>
cap_bands(const std::vector<std::vector<double>>& evals, int n_bands) {
    if (n_bands <= 0)
        return evals;
    std::vector<std::vector<double>> out(evals.size());
    for (std::size_t ik = 0; ik < evals.size(); ++ik) {
        const int m = static_cast<int>(evals[ik].size());
        const int take = std::min(n_bands, m);
        out[ik].assign(evals[ik].begin(),
                       evals[ik].begin() + static_cast<std::ptrdiff_t>(take));
    }
    return out;
}

void flatten_spectrum(const std::vector<std::vector<double>>& evals,
                      const std::vector<double>& k_weights, int n_bands_use,
                      std::vector<double>& flat_e, std::vector<double>& flat_w) {
    const int nk = static_cast<int>(evals.size());
    flat_e.clear();
    flat_w.clear();
    flat_e.reserve(static_cast<std::size_t>(nk * std::max(n_bands_use, 1)));
    flat_w.reserve(flat_e.capacity());
    for (int ik = 0; ik < nk; ++ik) {
        const double w = k_weights[static_cast<std::size_t>(ik)];
        const int m = static_cast<int>(evals[static_cast<std::size_t>(ik)].size());
        const int take = (n_bands_use > 0) ? std::min(n_bands_use, m) : m;
        for (int b = 0; b < take; ++b) {
            flat_e.push_back(
                evals[static_cast<std::size_t>(ik)][static_cast<std::size_t>(b)]);
            flat_w.push_back(w);
        }
    }
}

} // namespace

DosResult Dos::compute(GpwDft& method,
                       const std::vector<GpwAtom>& atoms) const {
    if (k_mesh < 1 && !use_method_k_list)
        throw std::invalid_argument("Dos::compute: k_mesh must be >= 1");
    if (!(sigma_ha > 0.0))
        throw std::invalid_argument("Dos::compute: sigma_ha must be > 0");
    if (n_bins < 2)
        throw std::invalid_argument("Dos::compute: n_bins must be >= 2");
    if (n_bands < 0)
        throw std::invalid_argument("Dos::compute: n_bands must be >= 0");

    if (!use_method_k_list) {
        method.k_points.clear();
        method.k_point_weights.clear();
        method.k_mesh[0] = k_mesh;
        method.k_mesh[1] = k_mesh;
        method.k_mesh[2] = k_mesh;
    }

    // Multi-k path requires a non-Γ mesh or an explicit list.
    const bool has_explicit = !method.k_points.empty();
    const bool multi =
        has_explicit || method.k_mesh[0] != 1 || method.k_mesh[1] != 1
        || method.k_mesh[2] != 1;
    if (!multi)
        throw std::invalid_argument(
            "Dos::compute: multi-k mesh required (set Dos::k_mesh >= 2, or "
            "pass use_method_k_list with a non-Γ GpwDft::k_mesh / k_points). "
            "Example: dos.k_mesh = 2; dos.compute(method, atoms);");

    GpwResult& scf = method.run(atoms);

    if (scf.eps_k.empty() || scf.k_weights.empty() || scf.n_kpts < 1)
        throw std::runtime_error(
            "Dos::compute: multi-k SCF left empty eps_k / k_weights "
            "(internal multi-k payload missing)");
    if (static_cast<int>(scf.eps_k.size()) != scf.n_kpts
        || static_cast<int>(scf.k_weights.size()) != scf.n_kpts)
        throw std::runtime_error(
            "Dos::compute: eps_k / k_weights length mismatch vs n_kpts");

    DosResult out;
    out.scf_converged = scf.converged;
    out.scf_energy_ha = scf.e_total;
    out.n_kpts = scf.n_kpts;
    out.e_fermi_ha = scf.e_fermi;

    const bool uks = method.is_uks();
    const bool use_smear = (method.smearing != Smearing::none);
    out.nspin = uks ? 2 : 1;

    // Cap band count; GPW diagonalizes the full AO spectrum.
    const int m0 = static_cast<int>(scf.eps_k[0].size());
    const int n_use = (n_bands > 0) ? std::min(n_bands, m0) : m0;
    if (n_use < 1)
        throw std::runtime_error("Dos::compute: no bands in eps_k");
    out.n_bands = n_use;

    auto evals = cap_bands(scf.eps_k, n_use);

    if (!uks) {
        // ── Restricted (RKS) ───────────────────────────────────────────
        if (method.n_occ < 1)
            throw std::invalid_argument(
                "Dos::compute: RKS requires method.n_occ >= 1");
        if (n_use < method.n_occ)
            throw std::invalid_argument(
                "Dos::compute: n_bands (" + std::to_string(n_use)
                + ") < n_occ (" + std::to_string(method.n_occ) + ")");

        if (n_use > method.n_occ) {
            int vk = 0, ck = 0, vb = 0, cb = 0;
            out.gap_ha = gap_from_spectrum(evals, method.n_occ, out.e_vbm_ha,
                                          out.e_cbm_ha, vk, ck, vb, cb);
        } else {
            // Only occupied spectrum: VBM = max of band n_occ-1; no CBM.
            out.e_vbm_ha = -1e300;
            for (const auto& row : evals) {
                const double v =
                    row[static_cast<std::size_t>(method.n_occ - 1)];
                if (v > out.e_vbm_ha)
                    out.e_vbm_ha = v;
            }
            out.e_cbm_ha = out.e_vbm_ha;
            out.gap_ha = 0;
        }

        std::vector<double> flat_e, flat_w;
        flatten_spectrum(evals, scf.k_weights, n_use, flat_e, flat_w);

        double emin = e_min_ha;
        double emax = e_max_ha;
        if (!(emax > emin)) {
            emin = *std::min_element(flat_e.begin(), flat_e.end())
                   - 4.0 * sigma_ha;
            emax = *std::max_element(flat_e.begin(), flat_e.end())
                   + 4.0 * sigma_ha;
        }

        gaussian_dos(flat_e, flat_w, emin, emax, n_bins, sigma_ha,
                     out.energy_ha, out.dos, /*spin_degeneracy=*/2.0);

        const double e_cut =
            use_smear ? out.e_fermi_ha : out.e_vbm_ha;
        out.n_elec_to_vbm = integrate_dos_to(out.energy_ha, out.dos, e_cut);
        return out;
    }

    // ── Collinear UKS multi-k ───────────────────────────────────────────
    if (scf.eps_k_beta.empty()
        || static_cast<int>(scf.eps_k_beta.size()) != scf.n_kpts)
        throw std::runtime_error(
            "Dos::compute: UKS multi-k missing eps_k_beta payload");
    if (method.n_alpha < 1 || method.n_beta < 1)
        throw std::invalid_argument(
            "Dos::compute: UKS requires n_alpha, n_beta >= 1");

    auto evals_b = cap_bands(scf.eps_k_beta, n_use);
    if (n_use < method.n_alpha || n_use < method.n_beta)
        throw std::invalid_argument(
            "Dos::compute: n_bands must cover max(n_alpha, n_beta)");

    // Joint VBM / CBM over α+β (integer-fill seed).
    {
        double e_vbm = -1e300;
        double e_cbm = 1e300;
        bool have_vbm = false;
        bool have_cbm = false;
        const auto scan = [&](const std::vector<std::vector<double>>& ev,
                              int n_occ) {
            for (const auto& row : ev) {
                if (n_occ > 0
                    && static_cast<int>(row.size()) >= n_occ) {
                    const double v =
                        row[static_cast<std::size_t>(n_occ - 1)];
                    if (!have_vbm || v > e_vbm) {
                        e_vbm = v;
                        have_vbm = true;
                    }
                }
                if (static_cast<int>(row.size()) > n_occ) {
                    const double c =
                        row[static_cast<std::size_t>(n_occ)];
                    if (!have_cbm || c < e_cbm) {
                        e_cbm = c;
                        have_cbm = true;
                    }
                }
            }
        };
        scan(evals, method.n_alpha);
        scan(evals_b, method.n_beta);
        if (have_vbm)
            out.e_vbm_ha = e_vbm;
        if (have_cbm && have_vbm) {
            out.e_cbm_ha = e_cbm;
            out.gap_ha = e_cbm - e_vbm;
        }
    }

    std::vector<double> flat_a, flat_b, flat_w, flat_all, flat_w_all;
    flatten_spectrum(evals, scf.k_weights, n_use, flat_a, flat_w);
    flatten_spectrum(evals_b, scf.k_weights, n_use, flat_b, flat_w);
    flat_all.reserve(flat_a.size() + flat_b.size());
    flat_w_all.reserve(flat_a.size() + flat_b.size());
    for (std::size_t i = 0; i < flat_a.size(); ++i) {
        flat_all.push_back(flat_a[i]);
        flat_w_all.push_back(flat_w[i]);
        flat_all.push_back(flat_b[i]);
        flat_w_all.push_back(flat_w[i]);
    }

    double emin = e_min_ha;
    double emax = e_max_ha;
    if (!(emax > emin)) {
        emin = *std::min_element(flat_all.begin(), flat_all.end())
               - 4.0 * sigma_ha;
        emax = *std::max_element(flat_all.begin(), flat_all.end())
               + 4.0 * sigma_ha;
    }

    gaussian_dos(flat_a, flat_w, emin, emax, n_bins, sigma_ha, out.energy_ha,
                 out.dos_alpha, /*spin_degeneracy=*/1.0);
    std::vector<double> e_beta;
    gaussian_dos(flat_b, flat_w, emin, emax, n_bins, sigma_ha, e_beta,
                 out.dos_beta, /*spin_degeneracy=*/1.0);
    gaussian_dos(flat_all, flat_w_all, emin, emax, n_bins, sigma_ha,
                 out.energy_ha, out.dos, /*spin_degeneracy=*/1.0);

    const double e_cut = use_smear ? out.e_fermi_ha : out.e_vbm_ha;
    out.n_elec_to_vbm = integrate_dos_to(out.energy_ha, out.dos, e_cut);
    return out;
}

} // namespace gpw
} // namespace cpu
} // namespace ATV
