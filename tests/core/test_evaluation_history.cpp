#include <gtest/gtest.h>

#include "atomiverse/core/evaluation_history.h"

namespace {
struct History final : ATV::EvaluationHistory {
    int accepted = 1, candidate = 2, commits = 0, discards = 0;
    void commit_evaluation() noexcept override {
        accepted = candidate;
        ++commits;
    }
    void discard_evaluation() noexcept override { ++discards; }
};
} // namespace

TEST(EvaluationHistory, NestedSuccessWaitsForParentAndDeduplicates) {
    ATV::EvaluationTransaction transaction;
    History history;
    {
        ATV::EvaluationTransaction::Scope outer(transaction);
        transaction.enlist(history);
        {
            ATV::EvaluationTransaction::Scope inner(transaction);
            transaction.enlist(history);
            inner.accept();
        }
        EXPECT_EQ(history.accepted, 1);
        EXPECT_FALSE(transaction.empty());
        outer.accept();
    }
    EXPECT_EQ(history.accepted, 2);
    EXPECT_EQ(history.commits, 1);
    EXPECT_EQ(history.discards, 0);
    EXPECT_FALSE(transaction.active());
    EXPECT_TRUE(transaction.empty());
}

TEST(EvaluationHistory, UnacceptedParentDiscardsEveryParticipantAndCanRetry) {
    ATV::EvaluationTransaction transaction;
    History first, second;
    {
        ATV::EvaluationTransaction::Scope outer(transaction);
        transaction.enlist(first);
        transaction.enlist(second);
    }
    EXPECT_EQ(first.accepted, 1);
    EXPECT_EQ(second.accepted, 1);
    EXPECT_EQ(first.discards, 1);
    EXPECT_EQ(second.discards, 1);
    {
        ATV::EvaluationTransaction::Scope retry(transaction);
        transaction.enlist(first);
        retry.accept();
    }
    EXPECT_EQ(first.accepted, 2);
    EXPECT_EQ(first.commits, 1);
    EXPECT_TRUE(transaction.empty());
}

TEST(EvaluationHistory, RegistrationRequiresPhysicalScope) {
    ATV::EvaluationTransaction transaction;
    History history;
    EXPECT_THROW(transaction.enlist(history), std::logic_error);
    EXPECT_TRUE(transaction.empty());
}

TEST(EvaluationHistory, ExceptionDiscardsBeforeParticipantLeavesScope) {
    ATV::EvaluationTransaction transaction;
    History history;
    EXPECT_THROW(
        {
            ATV::EvaluationTransaction::Scope outer(transaction);
            transaction.enlist(history);
            throw std::runtime_error("later component failed");
        },
        std::runtime_error);
    EXPECT_EQ(history.accepted, 1);
    EXPECT_EQ(history.discards, 1);
    EXPECT_FALSE(transaction.active());
}
