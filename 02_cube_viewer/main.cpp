#include <print>

#include "engine.hpp"

#include "magic_enum.hpp"

constexpr uint16_t width = 960;
constexpr uint16_t height = 960;

int main() {
    auto window = Window::init(width, height);
    if (!window) {
        std::print(stderr, "Window init failed: {}\n", magic_enum::enum_name(window.error()));
        return -1;
    }

    auto app = App::init(window.value());
    if (!app) {
        std::print(stderr, "App init failed: {}\n", magic_enum::enum_name(app.error()));
        return -1;
    }

    while (app->isRunning()) {
        app->pollEvents();
        app->endFrame();
    }

    app->deinit();
    window->deinit();
}
