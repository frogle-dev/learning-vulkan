#include "app.hpp"

std::expected<App, AppError> App::init(Window &window) {
    App app;

    app.window  = &window;
    app.running = true;

    auto expected = app.initVulkan();

    if (!expected)
        return std::unexpected(expected.error());

    return app;
}

std::expected<void, AppError> App::deinit() {
    cleanupSwapchain();

    VkResult result = vkDeviceWaitIdle(logicalDevice);

    if (result != VK_SUCCESS)
        return std::unexpected(result);

    return {};
}

std::expected<void, AppError> App::pollEvents() {
    while (window->isEventReady()) {
        SDL_Event &event = window->getCurrentEvent();
        if (event.type == SDL_EVENT_QUIT)
            running = false;
        if (event.type == SDL_EVENT_WINDOW_RESIZED) {
            auto expected = recreateSwapchain();

            if (!expected)
                return std::unexpected(expected.error());
        }
    }

    return {};
}

bool App::isRunning() { return running; }

std::expected<void, AppError> App::endFrame() { return drawFrame(); }
