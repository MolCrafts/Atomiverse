/// @file core/capture_ledger.cpp
/// @brief CaptureLedger: host-side admission (condition-variable
///        backpressure), capture-once fan-out, in-order release and failure
///        propagation shared by both backends' capture buffers.

#include "atomiverse/core/capture_ledger.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

#include "atomiverse/core/log.h"

namespace ATV {

namespace {

using Clock = std::chrono::steady_clock;

/// Re-check interval of a consumer whose next capture is admitted but whose
/// data is not complete yet [µs]: kPollMin after a capture, doubling to
/// kPollMax. A CUDA capture replayed from a step graph issues no host call
/// that could wake the consumer, so its completion word is polled; a CPU
/// copy wakes it at once (`notify_published`). With nothing admitted the
/// consumer sleeps on the condition variable instead.
constexpr std::chrono::microseconds kPollMin{20};
constexpr std::chrono::microseconds kPollMax{500};

double seconds(std::chrono::nanoseconds d) {
    return std::chrono::duration<double>(d).count();
}

std::uint64_t bit(int id) { return std::uint64_t{1} << id; }

} // namespace

CaptureLedger::CaptureLedger(std::string owner) : owner_(std::move(owner)) {}

CaptureLedger::~CaptureLedger() { stop(); }

int CaptureLedger::subscribe(CaptureSink& sink) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (running_)
        throw std::logic_error(owner_
                               + ": a consumer subscribed while the buffer "
                                 "is running");
    for (std::size_t i = 0; i < sinks_.size(); ++i)
        if (sinks_[i] == &sink)
            return static_cast<int>(i);
    for (std::size_t i = 0; i < sinks_.size(); ++i)
        if (sinks_[i] == nullptr) {
            sinks_[i] = &sink;
            return static_cast<int>(i);
        }
    if (static_cast<int>(sinks_.size()) >= kMaxConsumers)
        throw std::logic_error(owner_ + ": more than "
                               + std::to_string(kMaxConsumers)
                               + " asynchronous consumers on one driver");
    sinks_.push_back(&sink);
    return static_cast<int>(sinks_.size()) - 1;
}

void CaptureLedger::unsubscribe(CaptureSink& sink) {
    std::lock_guard<std::mutex> lk(mtx_);
    if (running_)
        throw std::logic_error(owner_
                               + ": a consumer unsubscribed while the buffer "
                                 "is running");
    for (auto& s : sinks_)
        if (s == &sink)
            s = nullptr;
    while (!sinks_.empty() && sinks_.back() == nullptr)
        sinks_.pop_back();
}

int CaptureLedger::id_of(const CaptureSink& sink) const {
    for (std::size_t i = 0; i < sinks_.size(); ++i)
        if (sinks_[i] == &sink)
            return static_cast<int>(i);
    return -1;
}

std::uint64_t CaptureLedger::capacity_for(const std::string& owner,
                                          std::uint64_t budget,
                                          std::uint64_t frame_bytes,
                                          const std::string& what_frame) {
    if (frame_bytes == 0)
        throw std::logic_error(owner + ": a captured frame has no bytes");
    const std::uint64_t capacity = budget / frame_bytes;
    if (capacity == 0)
        throw std::invalid_argument(
            owner + ": capture_buffer_bytes = " + std::to_string(budget)
            + " cannot hold one captured frame (" + what_frame
            + "), which needs " + std::to_string(frame_bytes)
            + " bytes; set capture_buffer_bytes >= "
            + std::to_string(frame_bytes));
    return capacity;
}

void CaptureLedger::start(std::uint64_t budget, std::uint64_t frame_bytes,
                          ReadyFn ready, const std::string& what_frame) {
    if (running_)
        throw std::logic_error(owner_ + ": start() while running");
    const std::uint64_t capacity =
        capacity_for(owner_, budget, frame_bytes, what_frame);
    std::lock_guard<std::mutex> lk(mtx_);
    budget_ = budget;
    frame_bytes_ = frame_bytes;
    capacity_ = capacity;
    ready_ = std::move(ready);
    admitted_ = tail_ = 0;
    open_ = false;
    abandoned_ = false;
    failed_.store(false, std::memory_order_release);
    failure_.clear();
    failure_raised_ = false;
    mask_.assign(static_cast<std::size_t>(capacity), 0);
    pending_.assign(static_cast<std::size_t>(capacity), 0);
    ticket_.assign(static_cast<std::size_t>(capacity), CaptureTicket{});
    consumers_.clear();
    consumers_.reserve(sinks_.size());
    for (std::size_t i = 0; i < sinks_.size(); ++i) {
        if (!sinks_[i])
            continue;
        Consumer c;
        c.sink = sinks_[i];
        c.id = static_cast<int>(i);
        consumers_.push_back(std::move(c));
    }
    running_ = true;
    for (std::size_t k = 0; k < consumers_.size(); ++k)
        consumers_[k].thread =
            std::thread([this, k] { worker_(static_cast<int>(k)); });
}

void CaptureLedger::release_locked_(std::uint64_t index) {
    --pending_[static_cast<std::size_t>(index % capacity_)];
    bool freed = false;
    while (tail_ < admitted_
           && pending_[static_cast<std::size_t>(tail_ % capacity_)] == 0) {
        mask_[static_cast<std::size_t>(tail_ % capacity_)] = 0;
        ++tail_;
        freed = true;
    }
    if (freed)
        space_.notify_all();
    done_.notify_all();
}

void CaptureLedger::seal_locked_() {
    if (!open_)
        return;
    open_ = false;
    release_locked_(admitted_ - 1); // the producer's pin
    work_.notify_all();             // a consumer not due may move on
}

void CaptureLedger::fail_locked_(std::string message) {
    if (!failed_.load(std::memory_order_relaxed)) {
        failure_ = std::move(message);
        failed_.store(true, std::memory_order_release);
    }
    space_.notify_all();
    done_.notify_all();
    work_.notify_all();
}

void CaptureLedger::throw_failure_locked_() {
    failure_raised_ = true;
    throw std::runtime_error(failure_);
}

std::string CaptureLedger::laggards_locked_() const {
    std::string names;
    if (tail_ >= admitted_)
        return names;
    const std::uint64_t mask =
        mask_[static_cast<std::size_t>(tail_ % capacity_)];
    for (const auto& c : consumers_) {
        if (!(mask & bit(c.id)) || c.cursor > tail_)
            continue;
        if (!names.empty())
            names += ", ";
        names += c.sink->capture_name();
    }
    return names;
}

CaptureLedger::Join CaptureLedger::join(CaptureSink& sink, long long step,
                                        CaptureKind kind) {
    std::unique_lock<std::mutex> lk(mtx_);
    if (!running_)
        throw std::logic_error(owner_
                               + ": a capture was requested before the "
                                 "buffer was allocated");
    const int id = id_of(sink);
    if (id < 0)
        throw std::logic_error(owner_ + ": " + sink.capture_name()
                               + " is not subscribed to this buffer");
    if (failed_.load(std::memory_order_acquire))
        throw_failure_locked_();
    if (abandoned_)
        throw std::logic_error(owner_ + ": the buffer was stopped");
    if (open_ && open_step_ == step) {
        // Capture once: this step is already captured (or being captured).
        const auto slot = static_cast<std::size_t>((admitted_ - 1) % capacity_);
        if (!(mask_[slot] & bit(id))) {
            mask_[slot] |= bit(id);
            ++pending_[slot];
        }
        const CaptureTicket t = ticket_[slot];
        lk.unlock();
        work_.notify_all();
        return {t, false};
    }
    seal_locked_();
    if (admitted_ - tail_ >= capacity_) {
        ++blocked_episodes_;
        if (blocked_episodes_ == episodes_reported_ + 1) {
            // First wait of this run; the rest are summarised by drain().
            const std::string behind = laggards_locked_();
            ATV_WARN("{}: the capture buffer is full ({} frame(s) of {} "
                     "bytes, budget {} bytes) — the run waits for {}; "
                     "nothing is dropped. Waits are summarised at the end of "
                     "the run",
                     owner_, capacity_, frame_bytes_, budget_,
                     behind.empty() ? std::string("its consumers") : behind);
        }
        const auto t0 = Clock::now();
        space_.wait(lk, [&] {
            return admitted_ - tail_ < capacity_
                   || failed_.load(std::memory_order_relaxed) || abandoned_;
        });
        blocked_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
            Clock::now() - t0);
        if (failed_.load(std::memory_order_relaxed))
            throw_failure_locked_();
        if (abandoned_)
            throw std::logic_error(owner_ + ": the buffer was stopped");
    }
    const std::uint64_t index = admitted_++;
    const auto slot = static_cast<std::size_t>(index % capacity_);
    mask_[slot] = bit(id);
    pending_[slot] = 2; // this consumer + the producer's pin
    ticket_[slot] = CaptureTicket{index, slot, step, kind};
    open_ = true;
    open_step_ = step;
    ++captures_;
    peak_frames_ = std::max(peak_frames_, admitted_ - tail_);
    const CaptureTicket t = ticket_[slot];
    lk.unlock();
    work_.notify_all();
    return {t, true};
}

void CaptureLedger::notify_published() { work_.notify_all(); }

void CaptureLedger::check() {
    if (!failed())
        return;
    std::lock_guard<std::mutex> lk(mtx_);
    throw_failure_locked_();
}

void CaptureLedger::worker_(int k) {
    auto idle = kPollMin;
    std::unique_lock<std::mutex> lk(mtx_);
    for (;;) {
        if (abandoned_)
            return;
        Consumer& c = consumers_[static_cast<std::size_t>(k)];
        if (c.cursor < admitted_) {
            const std::uint64_t index = c.cursor;
            const auto slot = static_cast<std::size_t>(index % capacity_);
            if (!(mask_[slot] & bit(c.id))) {
                // Not due on this capture — unless it is still open and this
                // consumer joins it later in the same step.
                if (open_ && index + 1 == admitted_) {
                    work_.wait(lk);
                    continue;
                }
                ++c.cursor;
                continue;
            }
            if (c.failed) {
                // A failed consumer reads nothing more; its holds are
                // released so the producer never waits on them.
                release_locked_(index);
                ++c.cursor;
                continue;
            }
            if (!ready_(c.id, index)) {
                work_.wait_for(lk, idle);
                idle = std::min(idle * 2, kPollMax);
                continue;
            }
            idle = kPollMin;
            const CaptureTicket ticket = ticket_[slot];
            lk.unlock();
            std::string cause;
            bool ok = true;
            try {
                c.sink->consume_capture(ticket);
            } catch (const std::exception& e) {
                ok = false;
                cause = e.what();
            } catch (...) {
                ok = false;
                cause = "unknown error";
            }
            lk.lock();
            if (!ok) {
                c.failed = true;
                fail_locked_(c.sink->capture_name() + " failed at step "
                             + std::to_string(ticket.step) + ": " + cause
                             + " — the run was stopped");
            }
            release_locked_(index);
            ++c.cursor;
            continue;
        }
        work_.wait(lk);
    }
}

void CaptureLedger::drain(bool raise) {
    std::unique_lock<std::mutex> lk(mtx_);
    if (running_ && !abandoned_) {
        seal_locked_();
        // Every copy has completed (precondition): a due capture whose data
        // is still not ready was reserved but never captured — its launch
        // failed. Waiting for it would hang; fail instead.
        for (std::uint64_t index = tail_; index < admitted_; ++index) {
            const std::uint64_t mask =
                mask_[static_cast<std::size_t>(index % capacity_)];
            for (const auto& c : consumers_) {
                if (!(mask & bit(c.id)) || c.failed || c.cursor > index
                    || ready_(c.id, index))
                    continue;
                fail_locked_(
                    owner_ + ": the capture of step "
                    + std::to_string(
                        ticket_[static_cast<std::size_t>(index % capacity_)]
                            .step)
                    + " was reserved for " + c.sink->capture_name()
                    + " but never taken (its launch failed)");
                abandoned_ = true;
                work_.notify_all();
                break;
            }
            if (abandoned_)
                break;
        }
        done_.wait(lk, [&] { return tail_ == admitted_ || abandoned_; });
    }
    const std::uint64_t episodes = blocked_episodes_ - episodes_reported_;
    const auto time = blocked_ - blocked_reported_;
    episodes_reported_ = blocked_episodes_;
    blocked_reported_ = blocked_;
    if (episodes > 0)
        ATV_WARN("{}: the run waited {} time(s) for capture buffer space, "
                 "{:.3f} s in total (budget {} bytes = {} frame(s) of {} "
                 "bytes, peak {} bytes); nothing was dropped. A larger "
                 "capture_buffer_bytes or faster consumers remove the waits",
                 owner_, episodes, seconds(time), budget_, capacity_,
                 frame_bytes_, peak_frames_ * frame_bytes_);
    if (raise && failed_.load(std::memory_order_acquire) && !failure_raised_)
        throw_failure_locked_();
}

void CaptureLedger::stop() noexcept {
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!running_)
            return;
        abandoned_ = true;
    }
    work_.notify_all();
    space_.notify_all();
    done_.notify_all();
    for (auto& c : consumers_)
        if (c.thread.joinable())
            c.thread.join();
    std::lock_guard<std::mutex> lk(mtx_);
    consumers_.clear();
    running_ = false;
    open_ = false;
}

void CaptureLedger::reset_stats() {
    std::lock_guard<std::mutex> lk(mtx_);
    captures_ = peak_frames_ = blocked_episodes_ = episodes_reported_ = 0;
    blocked_ = blocked_reported_ = std::chrono::nanoseconds{0};
}

CaptureStats CaptureLedger::stats() const {
    std::lock_guard<std::mutex> lk(mtx_);
    CaptureStats s;
    s.budget_bytes = budget_;
    s.frame_bytes = frame_bytes_;
    s.capacity = capacity_;
    s.captures = captures_;
    s.peak_bytes = peak_frames_ * frame_bytes_;
    s.blocked_episodes = blocked_episodes_;
    s.blocked_seconds = seconds(blocked_);
    return s;
}

std::string CaptureLedger::failure() const {
    std::lock_guard<std::mutex> lk(mtx_);
    return failure_;
}

} // namespace ATV
