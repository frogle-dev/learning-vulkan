#include "engine.hpp"

#include "error.hpp"

uint16_t constexpr width  = 960;
uint16_t constexpr height = 960;

std::string constexpr app_name = "HelloVulkan";

int main()
{
    auto window = Window::init(width, height, app_name);
    if (!window)
    {
        window.error().log("Window init failed");
        return -1;
    }

    auto app = App::init(window.value(), app_name);
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

    window.value().deinit();
}
