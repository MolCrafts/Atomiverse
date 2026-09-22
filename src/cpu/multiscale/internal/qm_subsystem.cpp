/// @file src/cpu/multiscale/internal/qm_subsystem.cpp
#include "atomiverse/cpu/multiscale/internal/qm_subsystem.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "atomiverse/core/frame_schema.h"
#include "atomiverse/cpu/component/owned_range.h"
#include "atomiverse/cpu/multiscale/link_atom.h"
#include "atomiverse/cpu/multiscale/qm_mask.h"

namespace ATV {
namespace cpu {
namespace multiscale {
namespace internal {

namespace {

constexpr const char* kMaskHowto =
    "Insert and fill kAtomQmZ before build():\n"
    "  frame.blocks.at(\"atoms\").insert(\"qm_z\",\n"
    "      ColumnSpec{ColDType::Int, 1, ColGroup::Core, 0, kColNone,\n"
    "                 kDynamicFieldId});\n"
    "  auto qm = frame.span(cpu::kAtomQmZ);\n"
    "  qm[i] = Z;  // >0 = QM with atomic number Z; 0 = MM\n"
    "See atomiverse/cpu/multiscale/qm_mask.h.";

} // namespace

void QmSubsystem::resolve_mask_(const Frame& frame) {
    qm_rows_.clear();
    qm_z_.clear();
    const int n = static_cast<int>(frame.nlocal());
    const Block* atoms = nullptr;
    try {
        atoms = &frame.blocks.at("atoms");
    } catch (...) {
        throw std::runtime_error(
            std::string("QmSubsystem: atoms block missing. ") + kMaskHowto);
    }
    if (!atoms->contains("qm_z")) {
        throw std::runtime_error(
            std::string("QmSubsystem: kAtomQmZ (atoms.qm_z) column missing. ")
            + kMaskHowto);
    }
    const auto qm = frame.span(kAtomQmZ);
    for (int i = 0; i < n; ++i) {
        const int z = qm[i];
        if (z > 0) {
            qm_rows_.push_back(i);
            qm_z_.push_back(z);
        } else if (z < 0) {
            throw std::invalid_argument(
                "QmSubsystem: kAtomQmZ values must be ≥ 0 (0=MM, >0=QM Z)");
        }
    }
    if (qm_rows_.empty()) {
        throw std::runtime_error(
            std::string("QmSubsystem: kAtomQmZ is all-zero (no QM atoms). ")
            + kMaskHowto);
    }
    n_qm_real_ = static_cast<int>(qm_rows_.size());
}

void QmSubsystem::allocate_private_frame_() {
    n_fqm_ = n_qm_real_ + static_cast<int>(resolved_links_.size());
    qm_frame_ = std::make_unique<Frame>(n_fqm_);
}

void QmSubsystem::build(Frame& frame, int mpi_size, bool require_embedding) {
    if (mpi_size > 1)
        throw std::runtime_error(
            "QmSubsystem: MPI (mpi_size > 1) is not supported in v1 "
            "(serial replicated-frame only)");
    if (!engine_)
        throw std::runtime_error(
            "QmSubsystem: no engine — call set_engine(...) before build()");
    embedding_ = engine_->as_embedding();
    if (require_embedding) {
        if (!embedding_) {
            throw std::runtime_error(
                "QmSubsystem: engine '" + engine_->qm_engine_name()
                + "' is not embedding-capable (as_embedding() is null). "
                  "Use a mean-field method (RHF/RKS/…) or another "
                  "EmbeddingEngine.");
        }
        if (!(embed_cutoff_ > 0.0))
            throw std::invalid_argument(
                "QmSubsystem: embed_cutoff must be > 0");
    }
    if (!(link_scale > 0.0) || !(link_scale < 1.0))
        throw std::invalid_argument("QmSubsystem: link_scale must be in (0,1)");

    resolve_mask_(frame);

    const auto charge = frame.span(kAtom_charge);
    const auto tags = frame.span(kAtom_tag);
    if (require_embedding) {
        // Exclusion invariant: QM charges must be 0 for electrostatic paths.
        std::ostringstream bad;
        bool any_bad = false;
        for (std::size_t k = 0; k < qm_rows_.size(); ++k) {
            const int i = qm_rows_[k];
            if (std::abs(static_cast<double>(charge[i])) > 1e-14) {
                if (any_bad)
                    bad << ", ";
                bad << tags[i];
                any_bad = true;
            }
        }
        if (any_bad)
            throw std::runtime_error(
                "QmSubsystem: QM atoms must have kAtom_charge == 0 "
                "(Coulomb styles exclude them by construction). Offending "
                "tags: "
                + bad.str());
    }

    // Derive covalent QM/MM cuts from mask × bond topology (no user links()).
    // No bonds ⇒ zero links (LJ / nonbonded QM/MM is valid).
    std::unordered_map<TagInt, int> tag_to_row;
    const int n = static_cast<int>(frame.nlocal());
    for (int i = 0; i < n; ++i)
        tag_to_row[tags[i]] = i;

    std::unordered_set<int> qm_row_set(qm_rows_.begin(), qm_rows_.end());
    resolved_links_.clear();
    link_mm_tags_.clear();
    // Dedupe unordered tag pairs (bonds may be listed once on min-tag owner).
    std::unordered_set<std::string> seen_pairs;

    const bool has_bonds = frame.blocks.at("atoms").contains("num_bond")
                           && frame.blocks.at("atoms").contains("bond_atom");
    if (has_bonds) {
        const auto num_bond = frame.span(kAtom_num_bond);
        const auto bond_atom = frame.span(kAtom_bond_atom);
        const auto qm_z = frame.span(kAtomQmZ);
        for (int i = 0; i < n; ++i) {
            const int nb = num_bond[static_cast<std::size_t>(i)];
            for (int m = 0; m < nb; ++m) {
                const TagInt partner_tag = bond_atom(i, m);
                auto it = tag_to_row.find(partner_tag);
                if (it == tag_to_row.end())
                    continue;
                const int j = it->second;
                const bool qi = qm_z[i] > 0;
                const bool qj = qm_z[j] > 0;
                if (qi == qj)
                    continue; // both QM or both MM
                const int qm_row = qi ? i : j;
                const int mm_row = qi ? j : i;
                const TagInt tq = tags[qm_row];
                const TagInt tm = tags[mm_row];
                const TagInt lo = std::min(tq, tm);
                const TagInt hi = std::max(tq, tm);
                const std::string key =
                    std::to_string(static_cast<long long>(lo)) + ":"
                    + std::to_string(static_cast<long long>(hi));
                if (!seen_pairs.insert(key).second)
                    continue;
                link_mm_tags_.insert(tm);
                resolved_links_.push_back({qm_row, mm_row, -1});
            }
        }
    }

    allocate_private_frame_();
    for (std::size_t li = 0; li < resolved_links_.size(); ++li)
        resolved_links_[li].fqm_h_row = n_qm_real_ + static_cast<int>(li);

    built_n_qm_ = n_qm_real_;
    built_ = true;

    cpu::Context dummy;
    cpu::BuildContext bctx{dummy, *qm_frame_};
    engine_->prepare(bctx);
}

void QmSubsystem::refresh_(Frame& frame) {
    resolve_mask_(frame);
    if (n_qm_real_ != built_n_qm_)
        throw std::runtime_error("QmSubsystem: QM atom count changed mid-run "
                                 "(kAtomQmZ is per-run static in v1)");

    const StorageReal* x = frame.span(kAtom_x);
    const StorageReal* y = frame.span(kAtom_y);
    const StorageReal* z = frame.span(kAtom_z);

    auto qx = qm_frame_->span(kAtom_x);
    auto qy = qm_frame_->span(kAtom_y);
    auto qz = qm_frame_->span(kAtom_z);
    auto qtype = qm_frame_->span(kAtom_type);
    auto qtag = qm_frame_->span(kAtom_tag);
    auto qmass = qm_frame_->span(kAtom_mass);
    auto qfx = qm_frame_->span(kAtom_fx);
    auto qfy = qm_frame_->span(kAtom_fy);
    auto qfz = qm_frame_->span(kAtom_fz);

    for (std::size_t k = 0; k < qm_rows_.size(); ++k) {
        const int i = qm_rows_[k];
        qx[k] = x[i];
        qy[k] = y[i];
        qz[k] = z[i];
        qtype[k] = qm_z_[k]; // Z from mask, not force-field type
        qtag[k] = static_cast<TagInt>(k + 1);
        qmass[k] = 1.0;
        qfx[k] = qfy[k] = qfz[k] = 0;
    }
    for (const auto& rl : resolved_links_) {
        const int h = rl.fqm_h_row;
        double hx, hy, hz;
        place_link_hydrogen(x[rl.qm_row], y[rl.qm_row], z[rl.qm_row],
                            x[rl.mm_row], y[rl.mm_row], z[rl.mm_row],
                            link_scale, hx, hy, hz);
        qx[h] = hx;
        qy[h] = hy;
        qz[h] = hz;
        qtype[h] = 1; // H
        qtag[h] = static_cast<TagInt>(h + 1);
        qmass[h] = 1.0;
        qfx[h] = qfy[h] = qfz[h] = 0;
    }
    qm_frame_->box = frame.box;
    qm_frame_->meta.get(kEnergy) = 0.0;
    qm_frame_->meta.get(kEnergyElectronic) = 0.0;
}

void QmSubsystem::collect_sites_(const Frame& frame) {
    sites_.clear();
    site_user_rows_.clear();
    if (!embedding_)
        return; // mechanical path — no embedding sites
    const int n = static_cast<int>(frame.nlocal());
    const auto tags = frame.span(kAtom_tag);
    const auto charge = frame.span(kAtom_charge);
    const StorageReal* x = frame.span(kAtom_x);
    const StorageReal* y = frame.span(kAtom_y);
    const StorageReal* z = frame.span(kAtom_z);
    const double rc2 = embed_cutoff_ * embed_cutoff_;

    std::unordered_set<int> qm_set(qm_rows_.begin(), qm_rows_.end());

    for (int i = 0; i < n; ++i) {
        if (qm_set.count(i))
            continue; // QM atoms are not embedding sites
        if (link_mm_tags_.count(tags[i]))
            continue; // link MM partner excluded (Z1 charge deletion)
        // Distance to nearest QM atom (minimum image).
        bool in_range = false;
        for (int qr : qm_rows_) {
            double dx = x[i] - x[qr];
            double dy = y[i] - y[qr];
            double dz = z[i] - z[qr];
            frame.box.min_image(dx, dy, dz);
            if (dx * dx + dy * dy + dz * dz <= rc2) {
                in_range = true;
                break;
            }
        }
        if (!in_range)
            continue;
        EmbeddingSite s;
        s.x = x[i];
        s.y = y[i];
        s.z = z[i];
        s.q = static_cast<double>(charge[i]);
        sites_.push_back(s);
        site_user_rows_.push_back(i);
    }

    embedding_->set_site_self_energy(false); // MM–MM owned by force field
    embedding_->set_embedding_sites(sites_);
}

void QmSubsystem::scatter_(Frame& frame, bool forces_requested, double e_qm) {
    frame.meta.get(kEnergy) += e_qm;
    frame.meta.get(kEnergyElectronic) += e_qm;

    force_residual_ = {{0, 0, 0}};
    if (!forces_requested)
        return;

    AccReal* fx = frame.span(kAtom_fx);
    AccReal* fy = frame.span(kAtom_fy);
    AccReal* fz = frame.span(kAtom_fz);
    const AccReal* qfx = qm_frame_->span(kAtom_fx);
    const AccReal* qfy = qm_frame_->span(kAtom_fy);
    const AccReal* qfz = qm_frame_->span(kAtom_fz);

    for (std::size_t k = 0; k < qm_rows_.size(); ++k) {
        const int i = qm_rows_[k];
        fx[i] += qfx[k];
        fy[i] += qfy[k];
        fz[i] += qfz[k];
    }
    for (const auto& rl : resolved_links_) {
        const int h = rl.fqm_h_row;
        project_link_force(link_scale, qfx[h], qfy[h], qfz[h], fx[rl.qm_row],
                           fy[rl.qm_row], fz[rl.qm_row], fx[rl.mm_row],
                           fy[rl.mm_row], fz[rl.mm_row]);
    }

    // Site forces from embedding engine (electrostatic/polarizable only).
    if (embedding_) {
        const auto& sf = embedding_->embedding_site_forces();
        if (sf.size() != sites_.size())
            throw std::runtime_error(
                "QmSubsystem: embedding_site_forces size mismatch");
        for (std::size_t s = 0; s < sf.size(); ++s) {
            const int i = site_user_rows_[s];
            fx[i] += sf[s].fx;
            fy[i] += sf[s].fy;
            fz[i] += sf[s].fz;
        }
    }

    // Residual over whole frame (third-law witness for self-energy off).
    const int n = static_cast<int>(frame.nlocal());
    for (int i = 0; i < n; ++i) {
        force_residual_[0] += fx[i];
        force_residual_[1] += fy[i];
        force_residual_[2] += fz[i];
    }
}

void QmSubsystem::launch(Frame& frame, bool forces_requested, double& e_qm) {
    if (!built_)
        throw std::runtime_error("QmSubsystem::launch before build()");
    refresh_(frame);
    collect_sites_(frame);
    if (reset_history_each_launch_)
        engine_->reset_history();

    cpu::Context dummy;
    cpu::LaunchContext lctx{dummy, *qm_frame_, 0, 0};
    lctx.forces_requested = forces_requested;
    engine_->evaluate(lctx);
    e_qm = qm_frame_->meta.get(kEnergy);
    scatter_(frame, forces_requested, e_qm);
}

} // namespace internal
} // namespace multiscale
} // namespace cpu
} // namespace ATV
