/// @file tests/cpu/component/test_qm_engine.cpp
/// @brief Unit: QmEngine / EmbeddingEngine protocol (esmm-qmmm-02).
///
/// Multi-iter RHF.evaluate / warm-start campaigns: regressions/multiscale/*
/// and architecture/lj_rhf_coupled (e2e-in-tests wave-2).

#ifdef ATV_HAS_HF

#include <gtest/gtest.h>

#include "atomiverse/cpu/component/qm_engine.h"
#include "atomiverse/cpu/electronic/aimd_method.h"
#include "atomiverse/cpu/scf/methods.h"
#ifdef ATV_HAS_SEMIEMPIRICAL
#include "atomiverse/cpu/semiempirical/am1_solver.h"
#endif
#ifdef ATV_HAS_SEMIEMPIRICAL
#include "atomiverse/cpu/semiempirical/dftb2.h"
#endif

using namespace ATV;

TEST(QmEngine, GaussianMethodsExposeEmbedding) {
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    cpu::scf::UHF uhf{cpu::scf::BasisSet::STO3G, 0, 0};
    cpu::scf::RKS rks{cpu::scf::BasisSet::STO3G, cpu::scf::XCType::PBE};
    cpu::QmEngine* engines[] = {&rhf, &uhf, &rks};
    for (cpu::QmEngine* e : engines) {
        ASSERT_NE(e->as_embedding(), nullptr);
        EXPECT_FALSE(e->qm_engine_name().empty());
    }
}

#ifdef ATV_HAS_SEMIEMPIRICAL
TEST(QmEngine, Am1AndDftb2NoEmbedding) {
    cpu::semiempirical::AM1 am1;
    cpu::semiempirical::DFTB2 dftb{cpu::semiempirical::DftbSkSet::MioSketch};
    cpu::QmEngine* engines[] = {&am1, &dftb};
    for (cpu::QmEngine* e : engines) {
        EXPECT_EQ(e->as_embedding(), nullptr);
        EXPECT_FALSE(e->qm_engine_name().empty());
    }
}

#endif

TEST(QmEngine, EmbeddingSitesRoundTrip) {
    cpu::scf::RHF rhf{cpu::scf::BasisSet::STO3G};
    cpu::EmbeddingEngine* emb = rhf.as_embedding();
    ASSERT_NE(emb, nullptr);
    emb->set_site_self_energy(false);
    emb->set_embedding_sites({{0.0, 0.0, 3.0, 0.5}});
    EXPECT_EQ(rhf.point_charges.size(), 1u);
    EXPECT_FALSE(rhf.embed_self_energy);
}

TEST(QmEngine, RejectsUnwiredElectronicPropagationBeforeSolving) {
    cpu::Frame frame(0);
    cpu::Context context;
    cpu::LaunchContext launch{context, frame, 0, 0};
    cpu::electronic::CarParrinelloMethod<cpu::scf::RHF> cp{
        cpu::scf::RHF{cpu::scf::BasisSet::STO3G}, 400.0};
    EXPECT_THROW(
        {
            try {
                cp.method().evaluate(launch);
            } catch (const std::runtime_error& error) {
                EXPECT_NE(std::string(error.what())
                              .find("QM/MM does not support CPMD or XL-BOMD"),
                          std::string::npos);
                throw;
            }
        },
        std::runtime_error);
    cpu::electronic::XlBomdMethod<cpu::scf::RHF> xl{
        cpu::scf::RHF{cpu::scf::BasisSet::STO3G}, 5};
    EXPECT_THROW(
        {
            try {
                xl.method().evaluate(launch);
            } catch (const std::runtime_error& error) {
                EXPECT_NE(std::string(error.what())
                              .find("QM/MM does not support CPMD or XL-BOMD"),
                          std::string::npos);
                throw;
            }
        },
        std::runtime_error);
}

#else
TEST(QmEngine, SkippedWithoutHF) { GTEST_SKIP() << "ATV_HAS_HF off"; }
#endif
