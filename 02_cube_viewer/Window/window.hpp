#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_hints.h>

#include "error.hpp"

enum class WindowErrorKind
{
    SdlInitFailed,
    SdlSetHintFailed,
    SdlWindowCreationFailed,
};

using WindowError = Error<WindowErrorKind>;

template <typename T> using WindowResult = std::expected<T, WindowError>;

class Window
{
  public:
    static WindowResult<Window> init(uint16_t start_width, uint16_t start_height,
                                     std::string const &window_name);
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
