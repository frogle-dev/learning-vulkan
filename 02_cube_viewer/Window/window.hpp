#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_hints.h>

#include <expected>

enum class WindowError {
    SdlInitFailed,
    SdlSetHintFailed,
    SdlWindowCreationFailed,
};

class Window {
  public:
    static std::expected<Window, WindowError> init(uint16_t start_width,
                                                   uint16_t start_height);
    void deinit();

    SDL_Window *getSDLWindow() const;
    bool isEventReady();
    SDL_Event &getCurrentEvent();

  private:
    Window() = default;

    uint16_t width;
    uint16_t height;

    SDL_Window *window = nullptr;
    SDL_Event event;
};
