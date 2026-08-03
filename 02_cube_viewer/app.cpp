#include "app.hpp"

[[nodiscard]]
Result<App> App::init(Window &window)
{
    App app;

    app.window_  = &window;
    app.running_ = true;

    auto expected = app.initVulkan();

    if (!expected)
        return std::unexpected(expected.error());

    return app;
}

[[nodiscard]]
Result<void> App::deinit()
{
    cleanupSwapchain();

    vk::Result result = logical_device_.waitIdle();

    if (result != vk::Result::eSuccess)
        return AppError::unexpected({"failed to logical device wait idle", result});

    return {};
}

Result<void> App::pollEvents()
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

std::expected<void, AppError> App::endFrame() { return drawFrame(); }
