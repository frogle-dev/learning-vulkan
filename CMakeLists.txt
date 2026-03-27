cmake_minimum_required(VERSION 3.20)
project(learn_vulkan)

set(CMAKE_CXX_STANDARD 20)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

find_program(SLANGC_EXECUTABLE slangc HINTS $ENV{VULKAN_SDK}/bin REQUIRED)

function (add_slang_shader_target TARGET)
        cmake_parse_arguments ("SHADER" "" "" "SOURCES" ${ARGN})
        set (SHADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/shaders)
        set (ENTRY_POINTS -entry vertMain -entry fragMain)
        add_custom_command (
                OUTPUT ${SHADERS_DIR}
                COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADERS_DIR}
        )
        add_custom_command (
                OUTPUT ${SHADERS_DIR}/slang.spv
                COMMAND ${SLANGC_EXECUTABLE} ${SHADER_SOURCES} -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name ${ENTRY_POINTS} -o slang.spv
                WORKING_DIRECTORY ${SHADERS_DIR}
                DEPENDS ${SHADERS_DIR} ${SHADER_SOURCES}
                COMMENT "Compiling Slang shader"
                VERBATIM
        )
        add_custom_target (${TARGET} DEPENDS ${SHADERS_DIR}/slang.spv)
endfunction()


add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME}
        Vulkan::Vulkan
        glfw
        glm::glm
)


add_slang_shader_target(shaders SOURCES ${CMAKE_CURRENT_LIST_DIR}/shaders/shader.slang)
add_dependencies(${PROJECT_NAME} shaders)
