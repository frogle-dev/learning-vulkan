#pragma once

#include <deque>
#include <functional>

#include "error.hpp"

enum class DeletionQueueErrorKind
{
};

using DeletionQueueError = Error<DeletionQueueErrorKind>;

template <typename T> using DeletionQueueResult = std::expected<T, DeletionQueueError>;

class DeletionQueue
{
  public:
    static DeletionQueue init();
    void deinit();

    void pushBack(std::function<void()> cleanup_func);

  private:
    DeletionQueue() = default;

    std::deque<std::function<void()>> deletion_queue_;
};
