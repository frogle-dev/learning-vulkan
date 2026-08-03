cmake_minimum_required(VERSION 3.20)
project(learn_vulkan)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

find_package(Vulkan REQUIRED)
find_package(SDL3 REQUIRED)
find_package(glm REQUIRED)

find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

FetchContent_Declare(
    VMA
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(VMA)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(spdlog)

function(add_slang_shader_target TARGET)
    cmake_parse_arguments("SHADER" "" "" "SOURCES" ${ARGN})
    set(SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
    set(ENTRY_POINTS -entry vertMain -entry fragMain)
    add_custom_command(
                OUTPUT ${SHADERS_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
        )
    add_custom_command(
                OUTPUT ${SHADERS_DIR}/slang.spv
                COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES}
                    -target spirv
                    -profile spirv_1_4
                    -emit-spirv-directly
                    -fvk-use-entrypoint-name
                    ${ENTRY_POINTS} -o slang.spv
                WORKING_DIRECTORY ${SHADERS_DIR}
                DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
                COMMENT "Compiling Slang shader"
                VERBATIM
        )
    add_custom_target(${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()

add_executable(${PROJECT_NAME})

target_sources(${PROJECT_NAME} PRIVATE main.cpp)

target_precompile_headers(${PROJECT_NAME} PRIVATE pch.hpp)

target_link_libraries(${PROJECT_NAME}
        Vulkan::Vulkan
        SDL3::SDL3
        glm::glm
        GPUOpen::VulkanMemoryAllocator
        spdlog::spdlog
)

target_compile_options(${PROJECT_NAME} PRIVATE
                                    -fsanitize=address
                                    -fno-exceptions
                                    -fstack-protector
                                    -fsanitize=undefined)
target_compile_options(${PROJECT_NAME} PRIVATE -Wswitch -Wall -Wunused)

add_slang_shader_target(shaders
    SOURCES ${CMAKE_CURRENT_LIST_DIR}/shaders/shader.slang)
add_dependencies(${PROJECT_NAME} shaders)
