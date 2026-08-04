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
    DeletionQueue() = default;

    void deinit();

    void setDevice(vk::Device logical_device) { device_ = logical_device; }
    void pushBack(std::function<void(vk::Device)> cleanup_func);

  private:
    vk::Device device_ = nullptr;
    std::deque<std::function<void(vk::Device)>> deletion_queue_;
};
