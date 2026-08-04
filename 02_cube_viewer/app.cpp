#include "app.hpp"

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

[[nodiscard]]
AppResult<App> App::init(Window &window)
{
    App app;

    app.window_  = &window;
    app.running_ = true;

    auto expected = app.initVulkan();

    if (!expected)
        return AppError::unexpected(expected.error());

    return app;
}

[[nodiscard]]
AppResult<void> App::deinit()
{
    deletion_queue_.deinit();

    cleanupSwapchain();

    vk::Result result = logical_device_.waitIdle();

    if (result != vk::Result::eSuccess)
        return AppError::unexpected({"failed to logical device wait idle", result});

    return {};
}

AppResult<void> App::pollEvents()
{
    while (window_->isEventReady())
    {
        SDL_Event &event = window_->getCurrentEvent();
        if (event.type == SDL_EVENT_QUIT)
            running_ = false;
        if (event.type == SDL_EVENT_WINDOW_RESIZED)
        {
            auto expected = recreateSwapchain();

            if (!expected)
                return AppError::unexpected(expected.error());
        }
    }

    return {};
}

bool App::isRunning() { return running_; }

AppResult<void> App::endFrame() { return drawFrame(); }
