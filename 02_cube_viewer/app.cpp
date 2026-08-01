#include "app.hpp"

std::expected<App, AppError> App::init(Window &window) {
    App app;

    app.window = &window;
    app.running = true;

    app.initVulkan();

    return app;
}

void App::deinit() {
    cleanupSwapchain();

    logicalDevice.waitIdle();
}

void App::pollEvents() {
    while (window->isEventReady()) {
        SDL_Event &event = window->getCurrentEvent();
        if (event.type == SDL_EVENT_QUIT)
            running = false;
        if (event.type == SDL_EVENT_WINDOW_RESIZED) {
            recreateSwapchain();
        }
    }
}

bool App::isRunning() { return running; }

std::expected<void, FrameError> App::endFrame() { return drawFrame(); }
