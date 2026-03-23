cmake_minimum_required(VERSION 3.20)
project(learn_vulkan)

set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
        Vulkan::Vulkan
        glfw
        glm::glm
)
