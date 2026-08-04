#include "deletion_queue.hpp"

void DeletionQueue::deinit()
{
    for (auto it = deletion_queue_.rbegin(); it != deletion_queue_.rend(); it++)
    {
        (*it)(device_);
    }

    deletion_queue_.clear();
}

void DeletionQueue::pushBack(std::function<void(vk::Device)> cleanup_function)
{
    deletion_queue_.push_back(cleanup_function);
}
