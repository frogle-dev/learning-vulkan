#include "window.hpp"

std::expected<Window, WindowError> Window::init(uint16_t start_width, uint16_t start_height) {
    Window win;

    win.width = start_width;
    win.height = start_height;

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        return std::unexpected(WindowError::SdlInitFailed);
    }

    if (!SDL_SetHint(SDL_HINT_APP_ID, "HelloVulkan")) {
        return std::unexpected(WindowError::SdlSetHintFailed);
    }

    win.window = SDL_CreateWindow("HelloVulkan", win.width, win.height, SDL_WINDOW_RESIZABLE);

    if (win.window == nullptr) {
        return std::unexpected(WindowError::SdlWindowCreationFailed);
    }

    return win;
}

void Window::deinit() {
    SDL_DestroyWindow(window);

    SDL_Quit();
}

SDL_Window *Window::getSDLWindow() const { return window; }

bool Window::isEventReady() { return SDL_PollEvent(&event); }

SDL_Event &Window::getCurrentEvent() { return event; }
