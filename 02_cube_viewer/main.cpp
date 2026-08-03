#include "engine.hpp"

constexpr uint16_t width  = 960;
constexpr uint16_t height = 960;

int main()
{
    auto window = Window::init(width, height);
    if (!window)
    {
        spdlog::error("Window init failed: {}\n", magic_enum::enum_name(window.error()));
        return -1;
    }

    auto app = App::init(window.value());
    if (!app)
    {
        app.error().log("App init failed");
        return -1;
    }

    while (app->isRunning())
    {
        auto expected = app.value().pollEvents();
        if (!expected)
        {
            expected.error().log("App poll events failed");
            return -1;
        }

        expected = app.value().endFrame();
        if (!expected)
        {
            expected.error().log("App end frame failed");
            return -1;
        }
    }

    auto expected = app.value().deinit();
    if (!expected)
    {
        expected.error().log("App deinit failed");
        return -1;
    }

    window->deinit();
}
