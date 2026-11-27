/// @file cpu/longrange/grid_remap.cpp
/// @brief GridRemap: LAMMPS `remap_3d` point-to-point plan.
///
/// Ported from LAMMPS `src/KSPACE/remap.cpp` and `src/pack.h`, checkout
/// e8c4c278a8. LAMMPS: Copyright (2003) Sandia Corporation. GPL-2.0-only.

#include "atomiverse/cpu/longrange/grid_remap.h"

#ifdef ATV_HAS_MPI

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ATV::longrange {
namespace {

/// LAMMPS `extent_3d`.
struct Extent {
    int ilo, ihi, isize;
    int jlo, jhi, jsize;
    int klo, khi, ksize;
};

Extent extent_of(const std::array<int, 6>& b) {
    return {b[0], b[1], b[1] - b[0] + 1, b[2], b[3], b[3] - b[2] + 1,
            b[4], b[5], b[5] - b[4] + 1};
}

/// LAMMPS `remap_3d_collide`: the overlap of two boxes, if any.
bool collide(const Extent& a, const Extent& b, Extent& o) {
    o.ilo = std::max(a.ilo, b.ilo);
    o.ihi = std::min(a.ihi, b.ihi);
    o.jlo = std::max(a.jlo, b.jlo);
    o.jhi = std::min(a.jhi, b.jhi);
    o.klo = std::max(a.klo, b.klo);
    o.khi = std::min(a.khi, b.khi);
    if (o.ilo > o.ihi || o.jlo > o.jhi || o.klo > o.khi)
        return false;
    o.isize = o.ihi - o.ilo + 1;
    o.jsize = o.jhi - o.jlo + 1;
    o.ksize = o.khi - o.klo + 1;
    return true;
}

} // namespace

GridRemap::GridRemap(MPI_Comm comm, const std::array<int, 6>& in_box,
                     const std::array<int, 6>& out_box, int nqty, int permute,
                     int tag)
    : comm_(comm), nqty_(nqty), permute_(permute), tag_(tag) {
    if (nqty < 1 || permute < 0 || permute > 2)
        throw std::invalid_argument(
            "GridRemap: nqty must be >= 1 and permute 0, 1 or 2");
    int me = 0, nprocs = 0;
    MPI_Comm_rank(comm, &me);
    MPI_Comm_size(comm, &nprocs);

    const Extent in = extent_of(in_box);
    const Extent out = extent_of(out_box);
    std::vector<Extent> inarray(static_cast<std::size_t>(nprocs));
    std::vector<Extent> outarray(static_cast<std::size_t>(nprocs));
    MPI_Allgather(&in, sizeof(Extent), MPI_BYTE, inarray.data(), sizeof(Extent),
                  MPI_BYTE, comm);
    MPI_Allgather(&out, sizeof(Extent), MPI_BYTE, outarray.data(),
                  sizeof(Extent), MPI_BYTE, comm);

    // Sends, partners in ring order starting after me, so self comes last.
    Extent overlap{};
    int ibuf = 0;
    int iproc = me;
    for (int i = 0; i < nprocs; ++i) {
        if (++iproc == nprocs)
            iproc = 0;
        if (!collide(in, outarray[static_cast<std::size_t>(iproc)], overlap))
            continue;
        Message m;
        m.proc = iproc;
        m.offset =
            nqty
            * ((overlap.klo - in.klo) * in.jsize * in.isize
               + ((overlap.jlo - in.jlo) * in.isize + overlap.ilo - in.ilo));
        m.plan.nfast = nqty * overlap.isize;
        m.plan.nmid = overlap.jsize;
        m.plan.nslow = overlap.ksize;
        m.plan.nstride_line = nqty * in.isize;
        m.plan.nstride_plane = nqty * in.jsize * in.isize;
        m.size = nqty * overlap.isize * overlap.jsize * overlap.ksize;
        if (iproc == me) {
            self_send_ = m;
        } else {
            m.bufloc = ibuf;
            ibuf += m.size;
            sends_.push_back(m);
        }
    }
    sendbuf_.resize(static_cast<std::size_t>(ibuf));

    // Receives, same ring order.
    ibuf = 0;
    iproc = me;
    for (int i = 0; i < nprocs; ++i) {
        if (++iproc == nprocs)
            iproc = 0;
        if (!collide(out, inarray[static_cast<std::size_t>(iproc)], overlap))
            continue;
        Message m;
        m.proc = iproc;
        m.bufloc = ibuf;
        if (permute == 0) {
            m.offset = nqty
                       * ((overlap.klo - out.klo) * out.jsize * out.isize
                          + (overlap.jlo - out.jlo) * out.isize
                          + (overlap.ilo - out.ilo));
            m.plan.nfast = nqty * overlap.isize;
            m.plan.nmid = overlap.jsize;
            m.plan.nslow = overlap.ksize;
            m.plan.nstride_line = nqty * out.isize;
            m.plan.nstride_plane = nqty * out.jsize * out.isize;
        } else if (permute == 1) {
            m.offset = nqty
                       * ((overlap.ilo - out.ilo) * out.ksize * out.jsize
                          + (overlap.klo - out.klo) * out.jsize
                          + (overlap.jlo - out.jlo));
            m.plan.nfast = overlap.isize;
            m.plan.nmid = overlap.jsize;
            m.plan.nslow = overlap.ksize;
            m.plan.nstride_line = nqty * out.jsize;
            m.plan.nstride_plane = nqty * out.ksize * out.jsize;
        } else {
            m.offset = nqty
                       * ((overlap.jlo - out.jlo) * out.isize * out.ksize
                          + (overlap.ilo - out.ilo) * out.ksize
                          + (overlap.klo - out.klo));
            m.plan.nfast = overlap.isize;
            m.plan.nmid = overlap.jsize;
            m.plan.nslow = overlap.ksize;
            m.plan.nstride_line = nqty * out.ksize;
            m.plan.nstride_plane = nqty * out.isize * out.ksize;
        }
        m.size = nqty * overlap.isize * overlap.jsize * overlap.ksize;
        ibuf += m.size;
        if (iproc == me) {
            self_recv_ = m;
            self_ = true;
        } else {
            recvs_.push_back(m);
        }
    }
    scratch_.resize(static_cast<std::size_t>(ibuf));
    requests_.resize(sends_.size() + recvs_.size());
}

void GridRemap::pack_(const double* data, double* buf,
                      const PackPlan& p) const {
    const std::size_t line = static_cast<std::size_t>(p.nfast);
    for (int slow = 0; slow < p.nslow; ++slow) {
        const double* plane = data + slow * p.nstride_plane;
        for (int mid = 0; mid < p.nmid; ++mid) {
            std::memcpy(buf, plane + mid * p.nstride_line,
                        line * sizeof(double));
            buf += line;
        }
    }
}

void GridRemap::unpack_(const double* buf, double* data,
                        const PackPlan& p) const {
    const int nqty = nqty_;
    if (permute_ == 0) {
        const std::size_t line = static_cast<std::size_t>(p.nfast);
        for (int slow = 0; slow < p.nslow; ++slow) {
            double* plane = data + slow * p.nstride_plane;
            for (int mid = 0; mid < p.nmid; ++mid) {
                std::memcpy(plane + mid * p.nstride_line, buf,
                            line * sizeof(double));
                buf += line;
            }
        }
    } else if (permute_ == 1) {
        // LAMMPS unpack_3d_permute1_n.
        for (int slow = 0; slow < p.nslow; ++slow) {
            const int plane = slow * p.nstride_line;
            for (int mid = 0; mid < p.nmid; ++mid) {
                double* dst = data + plane + nqty * mid;
                for (int fast = 0; fast < p.nfast;
                     ++fast, dst += p.nstride_plane)
                    for (int q = 0; q < nqty; ++q)
                        dst[q] = *buf++;
            }
        }
    } else {
        // LAMMPS unpack_3d_permute2_n.
        for (int slow = 0; slow < p.nslow; ++slow) {
            for (int mid = 0; mid < p.nmid; ++mid) {
                double* dst = data + nqty * slow + mid * p.nstride_plane;
                for (int fast = 0; fast < p.nfast;
                     ++fast, dst += p.nstride_line)
                    for (int q = 0; q < nqty; ++q)
                        dst[q] = *buf++;
            }
        }
    }
}

void GridRemap::perform(const double* in, double* out) {
    const int nrecv = static_cast<int>(recvs_.size());
    const int nsend = static_cast<int>(sends_.size());
    for (int r = 0; r < nrecv; ++r) {
        const Message& m = recvs_[static_cast<std::size_t>(r)];
        MPI_Irecv(scratch_.data() + m.bufloc, m.size, MPI_DOUBLE, m.proc, tag_,
                  comm_, &requests_[static_cast<std::size_t>(r)]);
    }
    for (int s = 0; s < nsend; ++s) {
        const Message& m = sends_[static_cast<std::size_t>(s)];
        pack_(in + m.offset, sendbuf_.data() + m.bufloc, m.plan);
        MPI_Isend(sendbuf_.data() + m.bufloc, m.size, MPI_DOUBLE, m.proc, tag_,
                  comm_, &requests_[static_cast<std::size_t>(nrecv + s)]);
    }
    // in -> scratch -> out for the part that stays here.
    if (self_) {
        pack_(in + self_send_.offset, scratch_.data() + self_recv_.bufloc,
              self_send_.plan);
        unpack_(scratch_.data() + self_recv_.bufloc, out + self_recv_.offset,
                self_recv_.plan);
    }
    for (int i = 0; i < nrecv; ++i) {
        int r = 0;
        MPI_Waitany(nrecv, requests_.data(), &r, MPI_STATUS_IGNORE);
        const Message& m = recvs_[static_cast<std::size_t>(r)];
        unpack_(scratch_.data() + m.bufloc, out + m.offset, m.plan);
    }
    if (nsend > 0)
        MPI_Waitall(nsend, requests_.data() + nrecv, MPI_STATUSES_IGNORE);
}

std::size_t GridRemap::memory_bytes() const {
    return (sendbuf_.capacity() + scratch_.capacity()) * sizeof(double)
           + (sends_.capacity() + recvs_.capacity()) * sizeof(Message)
           + requests_.capacity() * sizeof(MPI_Request);
}

} // namespace ATV::longrange

#endif // ATV_HAS_MPI
