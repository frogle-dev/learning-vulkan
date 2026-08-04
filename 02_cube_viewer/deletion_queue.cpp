#include "deletion_queue.hpp"

DeletionQueue DeletionQueue::init()
{
    DeletionQueue del_queue;
    return del_queue;
}

void DeletionQueue::deinit()
{
    for (auto it = deletion_queue_.rbegin(); it != deletion_queue_.rend(); it++)
    {
        (*it)();
    }

    deletion_queue_.clear();
}

void DeletionQueue::pushBack(std::function<void()> cleanup_function)
{
    deletion_queue_.push_back(cleanup_function);
}
