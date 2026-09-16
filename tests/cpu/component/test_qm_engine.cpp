/// @file tests/cpu/component/test_qm_engine.cpp
/// @brief Unit: QmEngine / EmbeddingEngine protocol (esmm-qmmm-02).
///
/// Multi-iter RHF.evaluate / warm-start campaigns: regressions/multiscale/*
/// and architecture/lj_rhf_coupled (e2e-in-tests wave-2).

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/scf/methods.h"
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#include "atomiverse/cpu/semiempirical/dftb2.h"

using namespace ATV;

TEST(QmEngine, GaussianMethodsExposeEmbedding) {
    cpu::scf::RHF rhf;
    cpu::scf::UHF uhf;
    cpu::scf::RKS rks;
    cpu::QmEngine* engines[] = {&rhf, &uhf, &rks};
    for (cpu::QmEngine* e : engines) {
        ASSERT_NE(e->as_embedding(), nullptr);
        EXPECT_FALSE(e->qm_engine_name().empty());
    }
}

TEST(QmEngine, Am1AndDftb2NoEmbedding) {
    cpu::semiempirical::AM1 am1;
    cpu::semiempirical::DFTB2 dftb;
    cpu::QmEngine* engines[] = {&am1, &dftb};
    for (cpu::QmEngine* e : engines) {
        EXPECT_EQ(e->as_embedding(), nullptr);
        EXPECT_FALSE(e->qm_engine_name().empty());
    }
}

TEST(QmEngine, EmbeddingSitesRoundTrip) {
    cpu::scf::RHF rhf;
    cpu::EmbeddingEngine* emb = rhf.as_embedding();
    ASSERT_NE(emb, nullptr);
    emb->set_site_self_energy(false);
    emb->set_embedding_sites({{0.0, 0.0, 3.0, 0.5}});
    EXPECT_EQ(rhf.point_charges.size(), 1u);
    EXPECT_FALSE(rhf.embed_self_energy);
}

#else
TEST(QmEngine, SkippedWithoutHF) { GTEST_SKIP() << "ATV_HAS_HF off"; }
#endif
