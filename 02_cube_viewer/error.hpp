#pragma once

#include <vulkan/vulkan.hpp>

#include <source_location>
#include <type_traits>

#include "magic_enum.hpp"

template <class T>
concept ScopedEnum = std::is_scoped_enum_v<T>;

template <ScopedEnum ErrorKind> struct Error
{
    Error(std::string message, ErrorKind kind,
          std::source_location location = std::source_location::current())
        : kind_(kind), message_(std::move(message)), location_(location)
    {
    }

    Error(std::string message, vk::Result result,
          std::source_location location = std::source_location::current())
        : vk_result_(result), message_(std::move(message)), location_(location)
    {
    }

    static std::unexpected<Error> unexpected(Error error)
    {
        return std::unexpected(std::move(error));
    }

    void log(std::string message)
    {
        spdlog::error(message);
        spdlog::error("File: {} | Line: {} | Func: {}", location_.file_name(),
                      location_.line(), location_.function_name());

        if (vk_result_.has_value())
            spdlog::error("Kind: {} | vk::Result: {}",
                          magic_enum::enum_name(kind_.value()),
                          vk::to_string(vk_result_.value()));
        if (kind_.has_value())
            spdlog::error("Kind: {}", magic_enum::enum_name(kind_.value()));

        spdlog::error("Message: {}\n", message_);
    }

    std::optional<ErrorKind> kind_       = std::nullopt;
    std::optional<vk::Result> vk_result_ = std::nullopt;
    std::string message_;
    std::source_location location_;
};
