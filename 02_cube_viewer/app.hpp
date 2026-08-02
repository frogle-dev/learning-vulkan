#pragma once

#include <SDL3/SDL_vulkan.h>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>

#include <glm/ext/matrix_transform.hpp>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <assert.h>

#include <array>
#include <chrono>
#include <expected>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <print>
#include <source_location>
#include <vector>

#include "stb_image.h"

#include "Window/window.hpp"

inline std::filesystem::path appPath() {
    return std::filesystem::canonical("/proc/self/exe").parent_path().parent_path();
}

constexpr uint8_t maxFramesInFlight = 2;

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

constexpr std::array<char const *, 1> validationLayers = {"VK_LAYER_KHRONOS_validation"};

struct Vertex {
    glm::vec2 pos;
    glm::vec3 col;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription() {
        return {.binding   = 0,
                .stride    = sizeof(Vertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return {{
            {.location = 0,
             .binding  = 0,
             .format   = VK_FORMAT_R32G32_SFLOAT,
             .offset   = offsetof(Vertex, pos)},
            {.location = 1,
             .binding  = 0,
             .format   = VK_FORMAT_R32G32B32_SFLOAT,
             .offset   = offsetof(Vertex, col)},
            {.location = 2,
             .binding  = 0,
             .format   = VK_FORMAT_R32G32_SFLOAT,
             .offset   = offsetof(Vertex, texCoord)},
        }};
    }
};

const std::vector<Vertex> vertices = {
    // pos           // col              // tex coord
    {{-0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.5f, 0.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

struct UniformBufferObject {
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

// enum class InstanceError {
//     ValidationLayerNotSupported,
//     ExtensionNotSupported,
// };
//
// enum class SurfaceError {
//     CreationFailed,
// };
//
// enum class PhysicalDeviceError {
//     FailedToFindGPU,
// };
//
// enum class LogicalDeviceError {
//     NoSuitableQueueFamily,
// };
//
// enum class FileError {
//     FailedToOpen,
// };
//
// enum class TransitionError {
//     TransitionNotSupported,
// };
//
// enum class ImageError {
//     FailedToLoad,
// };
//
// enum class MemoryError {
//     NoSuitableMemoryType,
// };
//
// using TextureError = std::variant<VkResult, ImageError>;
// using AppError = std::variant<VkResult, InstanceError, SurfaceError,
// LogicalDeviceError>; using GraphicsPipelineError = std::variant<VkResult, FileError>;

enum class ErrorKind {
    VulkanFailure,
    ValidationLayerNotSupported,
    ExtensionNotSupported,
    SurfaceCreationFailed,
    FailedToFindGPU,
    NoSuitableQueueFamily,
    FailedToOpenFile,
    TransitionNotSupported,
    FailedToLoadImage,
    NoSuitableMemoryType,
};

struct AppError {
    AppError(ErrorKind kind) : kind(kind) {}

    AppError(VkResult vk_result) : vk_result(vk_result) {
        kind = ErrorKind::VulkanFailure;
    }

    ErrorKind kind;
    std::optional<VkResult> vk_result = std::nullopt;

    std::source_location location = std::source_location::current();
};

class App {
  public:
    [[nodiscard]] static std::expected<App, AppError> init(Window &window);
    void deinit();

    bool isRunning();
    void pollEvents();
    [[nodiscard]] std::expected<void, AppError> endFrame();

  private:
    App() = default;

    Window *window = nullptr;

    bool running = false;

    std::array<const char *, 1> required_device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // vk::raii::Context context;
    VkInstance instance             = nullptr;
    VkPhysicalDevice physicalDevice = nullptr; // Physical device represents the GPU
    VkDevice logicalDevice =
        nullptr; // Logical Device is the interface for the physical device
    VkQueue queue      = nullptr;
    uint32_t queue_idx = UINT32_MAX;

    VkDebugUtilsMessengerEXT debugMessenger = nullptr;

    VkSurfaceKHR windowSurface = nullptr; // Surface to render to window

    VkSwapchainKHR swapchain = nullptr;
    std::vector<VkImage> swapchainImages;
    VkSurfaceFormatKHR swapchainSurfaceFormat;
    VkExtent2D swapchainExtent;
    std::vector<VkImageView> swapchainImageViews;

    VkDescriptorSetLayout descriptorSetLayout = nullptr;
    VkPipelineLayout pipelineLayout           = nullptr;
    VkPipeline graphicsPipeline               = nullptr;

    VkCommandPool commandPool = nullptr;
    std::vector<VkCommandBuffer> commandBuffers;

    uint32_t frame_idx = 0;

    // buffers
    VkBuffer vertexBuffer             = nullptr;
    VkDeviceMemory vertexBufferMemory = nullptr;
    VkBuffer indexBuffer              = nullptr;
    VkDeviceMemory indexBufferMemory  = nullptr;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;

    VkDescriptorPool descriptorPool = nullptr;
    std::vector<VkDescriptorSet> descriptorSets;

    // textures
    VkImage textureImage              = nullptr;
    VkDeviceMemory textureImageMemory = nullptr;
    VkImageView textureImageView      = nullptr;
    VkSampler textureSampler          = nullptr;

    // Sync objects
    std::vector<VkSemaphore> presentCompleteSphrs;
    std::vector<VkSemaphore> renderFinishedSphrs;
    std::vector<VkFence> drawFences;

    /* APPLICATION METHODS */

    void initVulkan() {
        createInstance();
        setupDebugMessenger();
        createWindowSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createCommandPool();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    [[nodiscard]]
    std::expected<void, AppError> drawFrame() {
        VkResult result = vkWaitForFences(logicalDevice, 1, &drawFences[frame_idx],
                                          VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS) {
            return std::unexpected(AppError(result));
        }

        uint32_t image_idx;
        result =
            vkAcquireNextImageKHR(logicalDevice, swapchain, UINT64_MAX,
                                  presentCompleteSphrs[frame_idx], nullptr, &image_idx);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return std::unexpected(AppError(result));
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            assert(result == VK_TIMEOUT || result == VK_NOT_READY);

            std::print(stderr, "Failed to acquire swap chain image");
            return std::unexpected(AppError(result));
        }

        updateUniformBuffer(frame_idx);

        result = vkResetFences(logicalDevice, 1, &drawFences[frame_idx]);
        if (result != VK_SUCCESS)
            return std::unexpected(AppError(result));

        result = vkResetCommandBuffer(commandBuffers[frame_idx], 0);
        if (result != VK_SUCCESS)
            return std::unexpected(AppError(result));

        recordCommandBuffer(image_idx);

        VkPipelineStageFlags waitDestinationStageMask(
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        const VkSubmitInfo submitInfo{
            .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &presentCompleteSphrs[frame_idx], // semaphores to wait for
            .pWaitDstStageMask    = &waitDestinationStageMask,
            .commandBufferCount   = 1,
            .pCommandBuffers      = &commandBuffers[frame_idx],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores =
                &renderFinishedSphrs[image_idx], // semaphores to signal when done
        };

        result = vkQueueSubmit(queue, 1, &submitInfo, drawFences[frame_idx]);
        if (result != VK_SUCCESS)
            return std::unexpected(result));

        const VkPresentInfoKHR presentInfoKHR{
            .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &renderFinishedSphrs[image_idx],
            .swapchainCount     = 1,
            .pSwapchains        = &swapchain,
            .pImageIndices      = &image_idx,
        };

        result = vkQueuePresentKHR(queue, &presentInfoKHR);
        assert(result == VK_SUCCESS);
        if (result == VK_SUBOPTIMAL_KHR || result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
        }

        frame_idx = (frame_idx + 1) % maxFramesInFlight;

        return {};
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static std::chrono::time_point startTime =
            std::chrono::high_resolution_clock::now();
        std::chrono::time_point currentTime = std::chrono::high_resolution_clock::now();

        float deltaTime = std::chrono::duration<float, std::chrono::seconds::period>(
                              currentTime - startTime)
                              .count();

        UniformBufferObject ubo;
        ubo.model = glm::rotate(glm::mat4(1.0f), deltaTime * glm::radians(90.0f),
                                glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view  = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                                glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj  = glm::perspective(glm::radians(45.0f),
                                     static_cast<float>(swapchainExtent.width) /
                                         static_cast<float>(swapchainExtent.height),
                                     0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    /* SETUP METHODS */

    [[nodiscard]]
    static std::expected<std::vector<char>, AppError> readFile(const std::string &path) {
        // std::ios::ate - reading starts at the end of file
        // std::ios::binary - reads file as a binary
        std::ifstream fin(path, std::ios::ate | std::ios::binary);

        if (!fin.is_open()) {
            return std::unexpected(AppError(ErrorKind::FailedToOpenFile));
        }

        // get position at end of file to get file length
        std::vector<char> buffer(fin.tellg());

        // go to beginning of file
        fin.seekg(0, std::ios::beg);
        fin.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));

        fin.close();

        return buffer;
    }

    // VKAPI_ATTR, VKAPI_CALL gives the function a signature that vulkan can
    // call
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type,
        const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData) {
        std::cerr << "[Validation layer]: " << std::to_string(severity) << " , "
                  << "[Type]: " << std::to_string(type) << " , "
                  << "[Message]:" << std::endl
                  << std::endl
                  << pCallbackData->pMessage << std::endl
                  << "----------------" << std::endl;

        return VK_FALSE;
    }

    [[nodiscard]]
    std::expected<void, AppError> setupDebugMessenger() {
        if (!enableValidationLayers)
            return {};

        VkDebugUtilsMessageSeverityFlagsEXT severityFlags(
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
        VkDebugUtilsMessageTypeFlagsEXT messageTypeFlags(
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT);

        VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
            .messageSeverity = severityFlags,
            .messageType     = messageTypeFlags,
            .pfnUserCallback = &debugCallback};

        VkResult result = vkCreateDebugUtilsMessengerEXT(
            instance, &debugUtilsMessengerCreateInfoEXT, nullptr, &debugMessenger);
        if (result != VK_SUCCESS)
            return std::unexpected(AppError(result));

        return {};
    }

    std::vector<const char *> getRequiredInstanceExtensions() {
        uint32_t sdlExtensionCount = 0;
        char const *const *sdlExtensions =
            SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);

        std::vector extensions(sdlExtensions, sdlExtensions + sdlExtensionCount);
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    [[nodiscard]]
    std::expected<void, AppError> createInstance() {
        // VULKAN INSTANCE CREATION
        // instance is used to communicate with vulkan
        VkApplicationInfo constexpr appInfo{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                            .pApplicationName = "Learn Vulkan",
                                            .applicationVersion =
                                                VK_MAKE_VERSION(1, 0, 0),
                                            .pEngineName   = "No Engine",
                                            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                            .apiVersion    = VK_API_VERSION_1_3};

        // VALIDATION LAYERS
        std::vector<char const *> requiredLayers;
        if (enableValidationLayers) {
            requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        // check if validation layers are available
        uint32_t layer_count = 0;
        vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
        std::vector<VkLayerProperties> layerProperties(layer_count);
        vkEnumerateInstanceLayerProperties(&layer_count, layerProperties.data());

        for (char const *required_layer : requiredLayers) {
            bool found = false;
            for (VkLayerProperties const &layer : layerProperties) {
                if (strcmp(layer.layerName, required_layer) == 0) {
                    found = true;
                    break;
                }
            }

            if (!found) {
                return std::unexpected(AppError(ErrorKind::ValidationLayerNotSupported));
            }
        }

        // EXTENSIONS
        std::vector<char const *> requiredExtensions = getRequiredInstanceExtensions();

        uint32_t extension_count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> extensionProperties(extension_count);
        vkEnumerateInstanceExtensionProperties(nullptr, &extension_count,
                                               extensionProperties.data());

        for (const char *required_extension : requiredExtensions) {
            bool found = false;
            for (VkExtensionProperties const &extension : extensionProperties) {
                if (strcmp(extension.extensionName, required_extension)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return std::unexpected(AppError(ErrorKind::ExtensionNotSupported));
            }
        }

        // CREATING THE INSTANCE
        VkInstanceCreateInfo createInfo{
            .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
            .ppEnabledLayerNames     = requiredLayers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data()};

        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS) {
            return std::unexpected(AppError(result));
        }

        return {};
    }

    [[nodiscard]]
    std::expected<void, AppError> createWindowSurface() {
        VkSurfaceKHR surface;

        if (!SDL_Vulkan_CreateSurface(window->getSDLWindow(), instance, nullptr,
                                      &surface)) {
            std::print(stderr, "SDL_Vulkan_CreateSurface failed: {}\n", SDL_GetError());
            return std::unexpected(AppError(ErrorKind::SurfaceCreationFailed));
        }

        windowSurface = surface;
        return {};
    }

    bool isDeviceSuitable(VkPhysicalDevice const &physicalDevice) {
        // if supports vulkan 1.3
        VkPhysicalDeviceProperties physical_device_properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physical_device_properties);
        bool supports_vulkan1_3 =
            physical_device_properties.apiVersion >= VK_API_VERSION_1_3;

        // if supports graphics queue family
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queue_family_count,
                                                 nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queue_family_count,
                                                 queue_families.data());

        bool supports_graphics = false;
        for (auto const &queue_family : queue_families) {
            if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                supports_graphics = true;
                break;
            }
        }

        // if supports specific extensions
        uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extension_count,
                                             nullptr);
        std::vector<VkExtensionProperties> available_device_extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extension_count,
                                             available_device_extensions.data());

        // if any of the required device extensions aren't available -> false
        bool supports_all_required_extensions = true;
        for (char const *required_device_extension : required_device_extensions) {
            bool found = true;
            for (auto const &available_device_extension : available_device_extensions) {
                if (strcmp(available_device_extension.extensionName,
                           required_device_extension) == 0) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                supports_all_required_extensions = false;
                break;
            }
        }

        // if supports specific features
        VkPhysicalDeviceExtendedDynamicStateFeaturesEXT extended_dynamic_state_features{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
        };

        VkPhysicalDeviceVulkan13Features vulkan_13_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = &extended_dynamic_state_features,
        };

        VkPhysicalDeviceVulkan11Features vulkan_11_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .pNext = &vulkan_13_features,
        };

        VkPhysicalDeviceFeatures2 features_2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &vulkan_11_features,
        };

        vkGetPhysicalDeviceFeatures2(physicalDevice, &features_2);

        bool supports_required_features =
            vulkan_11_features.shaderDrawParameters &&
            vulkan_13_features.synchronization2 && vulkan_13_features.dynamicRendering &&
            extended_dynamic_state_features.extendedDynamicState;

        return supports_vulkan1_3 && supports_graphics &&
               supports_all_required_extensions && supports_required_features;
    }

    [[nodiscard]]
    std::expected<void, PhysicalDeviceError> pickPhysicalDevice() {
        // checking if physical devices meet requirements

        uint32_t physical_device_count = 0;
        vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr);
        std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
        vkEnumeratePhysicalDevices(instance, &physical_device_count,
                                   physical_devices.data());

        // find if a GPU meets all the requirements
        bool found = false;
        for (auto const &physical_device : physical_devices) {
            if (isDeviceSuitable(physical_device)) {
                found          = true;
                physicalDevice = physical_device;
                break;
            }
        }
        if (!found) {
            return std::unexpected(PhysicalDeviceError::FailedToFindGPU);
        }

        return {};
    }

    [[nodiscard]]
    std::expected<void, AppError> createLogicalDevice() {
        uint32_t queue_family_properties_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice,
                                                 &queue_family_properties_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_family_properties(
            queue_family_properties_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice,
                                                 &queue_family_properties_count,
                                                 queue_family_properties.data());

        // check for support of both graphics and present queue families
        for (uint32_t queue_family_prop_idx = 0;
             queue_family_prop_idx < queue_family_properties.size();
             queue_family_prop_idx++) {
            VkBool32 present_support = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queue_family_prop_idx,
                                                 windowSurface, &present_support);

            if ((queue_family_properties[queue_family_prop_idx].queueFlags &
                 VK_QUEUE_GRAPHICS_BIT) &&
                present_support) {
                queue_idx = queue_family_prop_idx;
                break;
            }
        }

        if (queue_idx == UINT32_MAX) {
            return std::unexpected(AppError{LogicalDeviceError::NoSuitableQueueFamily});
        }

        // getting features
        VkPhysicalDeviceExtendedDynamicStateFeaturesEXT extended_dynamic_state_features{
            .sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
            .extendedDynamicState = VK_TRUE,
        };

        VkPhysicalDeviceVulkan13Features vulkan_13_features{
            .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext            = &extended_dynamic_state_features,
            .synchronization2 = VK_TRUE,
            .dynamicRendering = VK_TRUE,
        };

        VkPhysicalDeviceVulkan11Features vulkan_11_features{
            .sType                = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .pNext                = &vulkan_13_features,
            .shaderDrawParameters = VK_TRUE,
        };

        VkPhysicalDeviceFeatures2 features2{
            .sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext    = &vulkan_11_features,
            .features = {.samplerAnisotropy = VK_TRUE},
        };

        float queuePriority = 0.5f; // priority for scheduling of command buffer
                                    // execution, needed even if there is one queue
        VkDeviceQueueCreateInfo device_queue_create_info{
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = queue_idx,
            .queueCount       = 1,
            .pQueuePriorities = &queuePriority};

        VkDeviceCreateInfo deviceCreateInfo{
            .sType                = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext                = &features2, // connecting the chain of
                                                // features to vulkan
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos    = &device_queue_create_info,
            .enabledExtensionCount =
                static_cast<uint32_t>(required_device_extensions.size()),
            .ppEnabledExtensionNames = required_device_extensions.data()};

        VkResult result =
            vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &logicalDevice);
        if (result != VK_SUCCESS) {
            return std::unexpected(AppError{result});
        }

        vkGetDeviceQueue(logicalDevice, queue_idx, 0, &queue);

        return {};
    }

    VkSurfaceFormatKHR chooseSwapchainSurfaceFormat(
        const std::vector<VkSurfaceFormatKHR> &availableFormats) {
        assert(!availableFormats.empty());

        VkSurfaceFormatKHR surface_format = availableFormats[0];

        bool found = false;
        for (auto const &format : availableFormats) {
            found = format.format == VK_FORMAT_B8G8R8A8_SRGB &&
                    format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
            if (found) {
                surface_format = format;
                break;
            }
        }

        return surface_format;
    }

    VkPresentModeKHR chooseSwapchainPresentMode(
        const std::vector<VkPresentModeKHR> &availablePresentModes) {
        // fifo present mode - stores rendered images in a queue, takes an image
        // from the front of the queue to display every time the display
        // refreshes mailbox present mode - like fifo, but when the queue is
        // full it replaces old images with new ones to display images as fast as
        // possible

        bool found_fifo    = false;
        bool found_mailbox = false;
        for (VkPresentModeKHR present_mode : availablePresentModes) {
            if (present_mode == VK_PRESENT_MODE_FIFO_KHR) {
                found_fifo = true;
                break;
            }
            if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
                found_mailbox = true;
                break;
            }
        }

        assert(found_fifo || found_mailbox);

        // if mailbox present mode is available, use it, otherwise FIFO present mode
        return found_mailbox ? VK_PRESENT_MODE_MAILBOX_KHR : VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D
    chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR &surfaceCapabilities) {
        // extent is the resolution of the images in the swapchain

        if (surfaceCapabilities.currentExtent.width != UINT32_MAX) {
            return surfaceCapabilities.currentExtent;
        }

        int width, height;
        SDL_GetWindowSizeInPixels(window->getSDLWindow(), &width, &height);

        return VkExtent2D{
            std::clamp<uint32_t>(width, surfaceCapabilities.minImageExtent.width,
                                 surfaceCapabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, surfaceCapabilities.minImageExtent.height,
                                 surfaceCapabilities.maxImageExtent.height)};
    }

    uint32_t
    chooseSwapchainMinImageCount(const VkSurfaceCapabilitiesKHR &surfaceCapabilities) {
        uint32_t minImgCount = std::max(uint32_t(3), surfaceCapabilities.minImageCount);

        if ((0 < surfaceCapabilities.maxImageCount) &&
            (surfaceCapabilities.maxImageCount < minImgCount)) {
            minImgCount = surfaceCapabilities.maxImageCount;
        }

        return minImgCount;
    }

    void cleanupSwapchain() {
        swapchainImageViews.clear();
        swapchain = nullptr;
    }

    void recreateSwapchain() {
        int width  = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window->getSDLWindow(), &width, &height);
        while (width == 0 || height == 0) {
            SDL_GetWindowSizeInPixels(window->getSDLWindow(), &width, &height);
            SDL_WaitEvent(&window->getCurrentEvent());
        }

        vkDeviceWaitIdle(logicalDevice);

        swapchainImageViews.clear();
        // cleanupSwapchain();

        createSwapchain();
        createImageViews();
    }

    [[nodiscard]]
    std::expected<void, VkResult> createSwapchain() {
        VkSurfaceCapabilitiesKHR surface_capabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, windowSurface,
                                                  &surface_capabilities);
        swapchainExtent        = chooseSwapchainExtent(surface_capabilities);
        uint32_t minImageCount = chooseSwapchainMinImageCount(surface_capabilities);

        uint32_t available_formats_count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, windowSurface,
                                             &available_formats_count, nullptr);
        std::vector<VkSurfaceFormatKHR> available_formats(available_formats_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, windowSurface,
                                             &available_formats_count,
                                             available_formats.data());

        swapchainSurfaceFormat = chooseSwapchainSurfaceFormat(available_formats);

        uint32_t available_present_modes_count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, windowSurface,
                                                  &available_formats_count, nullptr);
        std::vector<VkPresentModeKHR> available_present_modes(
            available_present_modes_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, windowSurface,
                                                  &available_formats_count,
                                                  available_present_modes.data());

        VkPresentModeKHR presentMode =
            chooseSwapchainPresentMode(available_present_modes);

        VkSwapchainCreateInfoKHR swapchainCreateInfo{
            .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface          = windowSurface,
            .minImageCount    = minImageCount,
            .imageFormat      = swapchainSurfaceFormat.format,
            .imageColorSpace  = swapchainSurfaceFormat.colorSpace,
            .imageExtent      = swapchainExtent,
            .imageArrayLayers = 1,
            .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .preTransform     = surface_capabilities.currentTransform,
            .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode      = presentMode,
            .clipped          = true,
            .oldSwapchain     = swapchain,
        };

        VkResult result = vkCreateSwapchainKHR(logicalDevice, &swapchainCreateInfo,
                                               nullptr, &swapchain);
        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        uint32_t swapchain_image_count = 0;
        vkGetSwapchainImagesKHR(logicalDevice, swapchain, &swapchain_image_count,
                                nullptr);
        swapchainImages.resize(swapchain_image_count);
        vkGetSwapchainImagesKHR(logicalDevice, swapchain, &swapchain_image_count,
                                swapchainImages.data());

        return {};
    }

    [[nodiscard]]
    std::expected<VkImageView, VkResult> createImageView(VkImage &image,
                                                         VkFormat format) {
        VkImageViewCreateInfo viewInfo{
            .sType            = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image            = image,
            .viewType         = VK_IMAGE_VIEW_TYPE_2D,
            .format           = format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };

        VkImageView image_view;
        VkResult result =
            vkCreateImageView(logicalDevice, &viewInfo, nullptr, &image_view);
        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        return image_view;
    }

    [[nodiscard]]
    std::expected<void, VkResult> createImageViews() {
        assert(swapchainImageViews.empty());

        VkImageViewCreateInfo imageViewCreateInfo{
            .viewType         = VK_IMAGE_VIEW_TYPE_2D,
            .format           = swapchainSurfaceFormat.format,
            .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                 .levelCount = 1,
                                 .layerCount = 1}};

        swapchainImageViews.resize(swapchainImages.size());
        for (uint32_t i = 0; i < swapchainImages.size(); i++) {
            imageViewCreateInfo.image = swapchainImages[i];

            VkResult result = vkCreateImageView(logicalDevice, &imageViewCreateInfo,
                                                nullptr, &swapchainImageViews[i]);

            if (result != VK_SUCCESS) {
                return std::unexpected(result);
            }
        }

        return {};
    }

    [[nodiscard]]
    std::expected<void, VkResult> createDescriptorSetLayout() {
        std::array bindings = {
            VkDescriptorSetLayoutBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                         VK_SHADER_STAGE_VERTEX_BIT, nullptr),
            VkDescriptorSetLayoutBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                         VK_SHADER_STAGE_FRAGMENT_BIT, nullptr),
        };

        VkDescriptorSetLayoutCreateInfo layoutInfo{
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = bindings.size(),
            .pBindings    = bindings.data(),
        };

        VkResult result = vkCreateDescriptorSetLayout(logicalDevice, &layoutInfo, nullptr,
                                                      &descriptorSetLayout);

        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        return {};
    }

    [[nodiscard]]
    std::expected<void, GraphicsPipelineError> createGraphicsPipeline() {
        /* SHADER STAGE SETUP */

        auto shader_code = readFile(appPath() / "shaders/slang.spv");
        if (!shader_code) {
            return std::unexpected(GraphicsPipelineError{shader_code.error()});
        }

        VkShaderModule shader_module = createShaderModule(shader_code.value());

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage               = VK_SHADER_STAGE_VERTEX_BIT,
            .module              = shader_module,
            .pName               = "vertMain", // the entrypoint in the slang code
            .pSpecializationInfo = nullptr // used to set constants in shader per-pipeline
        };

        VkPipelineShaderStageCreateInfo fragShaderStageInfo{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = shader_module,
            .pName  = "fragMain"};

        std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages = {
            vertShaderStageInfo, fragShaderStageInfo};

        /* INPUT STAGE SETUP */

        auto bindingDescription    = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        VkPipelineVertexInputStateCreateInfo vertexInputInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions    = &bindingDescription,
            .vertexAttributeDescriptionCount =
                static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()};

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};

        VkPipelineViewportStateCreateInfo viewportState{
            .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount  = 1};

        std::array dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_VIEWPORT};
        VkPipelineDynamicStateCreateInfo dynamicState{
            .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates    = dynamicStates.data()};

        /* RASTERIZATION STAGE SETUP */

        VkPipelineRasterizationStateCreateInfo rasterizer{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE, // if true, fragments past the near or far plane
                                          // will be clamped rather than discarded
            .rasterizerDiscardEnable = VK_FALSE, // if true, skips rasterizer stage
            .polygonMode             = VK_POLYGON_MODE_FILL,
            .cullMode                = VK_CULL_MODE_BACK_BIT,
            .frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable         = VK_FALSE, // if true, rasterizer can make
                                                 // adjustments to depth values
            .lineWidth               = 1.0f};

        VkPipelineMultisampleStateCreateInfo multisampling{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
            .sampleShadingEnable  = VK_FALSE};

        /* COLOR BLENDING STAGE SETUP */

        // linearly interpolated blending
        VkPipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable         = VK_TRUE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp        = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp        = VK_BLEND_OP_ADD,
            .colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                   VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT};

        VkPipelineColorBlendStateCreateInfo colorBlending{
            .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable   = VK_FALSE,
            .logicOp         = VK_LOGIC_OP_COPY,
            .attachmentCount = 1,
            .pAttachments    = &colorBlendAttachment};

        /* PIPELINE SETUP */

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 1,
            .pSetLayouts            = &descriptorSetLayout,
            .pushConstantRangeCount = 0};

        VkResult result = vkCreatePipelineLayout(logicalDevice, &pipelineLayoutInfo,
                                                 nullptr, &pipelineLayout);

        if (result != VK_SUCCESS) {
            return std::unexpected(GraphicsPipelineError{result});
        }

        VkPipelineRenderingCreateInfo pipeline_rendering_create_info{
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount    = 1,
            .pColorAttachmentFormats = &swapchainSurfaceFormat.format};

        VkGraphicsPipelineCreateInfo graphics_pipeline_create_info{
            .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext               = &pipeline_rendering_create_info,
            .stageCount          = 2,
            .pStages             = shaderStages.data(),
            .pVertexInputState   = &vertexInputInfo,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState      = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState   = &multisampling,
            .pColorBlendState    = &colorBlending,
            .pDynamicState       = &dynamicState,
            .layout              = pipelineLayout,
            .renderPass          = VK_NULL_HANDLE // using dynamic rendering
        };

        result = vkCreateGraphicsPipelines(logicalDevice, VK_NULL_HANDLE, 1,
                                           &graphics_pipeline_create_info, nullptr,
                                           &graphicsPipeline);

        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        return {};
    }

    [[nodiscard]]
    std::expected<VkShaderModule, VkResult>
    createShaderModule(const std::vector<char> &code) const {
        VkShaderModuleCreateInfo createInfo{
            .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size() * sizeof(char),
            .pCode    = reinterpret_cast<const uint32_t *>(code.data())};

        VkShaderModule shaderModule;
        VkResult result =
            vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule);

        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        return shaderModule;
    }

    [[nodiscard]]
    std::expected<void, VkResult> createCommandPool() {
        VkCommandPoolCreateInfo poolInfo{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_idx,
        };

        VkResult result =
            vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool);

        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        return {};
    }

    [[nodiscard]]
    std::expected<void, VkResult>
    createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image,
                VkDeviceMemory &imageMemory) {
        VkImageCreateInfo imageInfo{
            .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType   = VK_IMAGE_TYPE_2D,
            .format      = format,
            .extent      = {width, height, 1},
            .mipLevels   = 1,
            .arrayLayers = 1,
            .samples     = VK_SAMPLE_COUNT_1_BIT,
            .tiling      = tiling,
            .usage       = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        VkResult result = vkCreateImage(logicalDevice, &imageInfo, nullptr, &image);
        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(logicalDevice, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize  = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties),
        };

        result = vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &imageMemory);
        if (result != VK_SUCCESS)
            return std::unexpected(result);

        result = vkBindImageMemory(logicalDevice, image, imageMemory, 0);
        if (result != VK_SUCCESS)
            return std::unexpected(result);

        return {};
    }

    std::expected<void, TransitionError> transitionImageLayout(const VkImage &image,
                                                               VkImageLayout oldLayout,
                                                               VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginOneTimeCommandBuffer();

        VkImageMemoryBarrier barrier{
            .sType            = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout        = oldLayout,
            .newLayout        = newLayout,
            .image            = image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };

        VkPipelineStageFlags srcStage;
        VkPipelineStageFlags dstStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED &&
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                   newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            return std::unexpected(TransitionError::TransitionNotSupported);
        }

        vkCmdPipelineBarrier(commandBuffer, srcStage, dstStage, {}, {}, nullptr, 0,
                             nullptr, 1, &barrier);

        endOneTimeCommandBuffer(commandBuffer);
    }

    [[nodiscard]]
    std::expected<void, TextureError> createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load((appPath() / "textures/dirt.png").c_str(), &texWidth,
                                    &texHeight, &texChannels, STBI_rgb_alpha);

        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            return std::unexpected(TextureError{ImageError::FailedToLoad});
        }

        VkBuffer stagingBuffer             = nullptr;
        VkDeviceMemory stagingBufferMemory = nullptr;

        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     stagingBuffer, stagingBufferMemory);

        void *data;
        VkResult result =
            vkMapMemory(logicalDevice, stagingBufferMemory, 0, imageSize, 0, &data);
        if (result != VK_SUCCESS) {
            return std::unexpected(TextureError{result});
        }

        memcpy(data, pixels, imageSize);

        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        stbi_image_free(pixels);

        auto expected = createImage(
            texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

        if (!expected) {
            return std::unexpected(TextureError{expected.error()});
        }

        transitionImageLayout(textureImage, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer, textureImage, texWidth, texHeight);
        transitionImageLayout(textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        return {};
    }

    [[nodiscard]]
    std::expected<void, VkResult> createTextureImageView() {
        auto expected = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB);
        if (!expected) {
            return std::unexpected(expected.error());
        }

        textureImageView = expected.value();

        return {};
    }

    [[nodiscard]]
    std::expected<void, VkResult> createTextureSampler() {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);

        VkSamplerCreateInfo samplerInfo{
            .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter               = VK_FILTER_NEAREST,
            .minFilter               = VK_FILTER_NEAREST,
            .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .mipLodBias              = 0.0f,
            .anisotropyEnable        = VK_TRUE,
            .maxAnisotropy           = properties.limits.maxSamplerAnisotropy,
            .compareEnable           = VK_FALSE,
            .compareOp               = VK_COMPARE_OP_ALWAYS,
            .minLod                  = 0.0f,
            .maxLod                  = 0.0f,
            .borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };

        VkResult result =
            vkCreateSampler(logicalDevice, &samplerInfo, nullptr, &textureSampler);
        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        return {};
    }

    void copyBufferToImage(const VkBuffer &buffer, VkImage &image, uint32_t width,
                           uint32_t height) {
        VkCommandBuffer commandBuffer = beginOneTimeCommandBuffer();

        VkBufferImageCopy region{
            .bufferOffset      = 0,
            .bufferRowLength   = 0,
            .bufferImageHeight = 0,
            .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset       = {0, 0, 0},
            .imageExtent       = {width, height, 1},
        };

        vkCmdCopyBufferToImage(commandBuffer, buffer, image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endOneTimeCommandBuffer(commandBuffer);
    }

    std::expected<uint32_t, MemoryError>
    findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        return std::unexpected(MemoryError::NoSuitableMemoryType);
    }

    [[nodiscard]]
    std::expected<void, VkResult>
    createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                 VkMemoryPropertyFlags properties, VkBuffer &buffer,
                 VkDeviceMemory &bufferMemory) {
        VkBufferCreateInfo bufferInfo{
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size        = size,
            .usage       = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        VkResult result = vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer);
        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

        VkMemoryAllocateInfo memoryAllocateInfo{
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize  = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties),
        };

        bufferMemory = VkDeviceMemory(logicalDevice, memoryAllocateInfo);

        buffer.bindMemory(*bufferMemory, 0);
    }

    VkCommandBuffer beginOneTimeCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = commandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };

        VkCommandBuffer commandBuffer =
            std::move(logicalDevice.allocateCommandBuffers(allocInfo).front());

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        commandBuffer.begin(beginInfo);

        return commandBuffer;
    }

    void endOneTimeCommandBuffer(VkCommandBuffer &commandBuffer) {
        commandBuffer.end();

        VkSubmitInfo submitInfo{
            .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers    = &*commandBuffer,
        };

        queue.submit(submitInfo, nullptr);
        queue.waitIdle();
    }

    void copyBuffer(VkBuffer &srcBuffer, VkBuffer &dstBuffer, VkDeviceSize size) {
        VkCommandBuffer commandCopyBuffer = beginOneTimeCommandBuffer();
        commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, vkBufferCopy(0, 0, size));
        endOneTimeCommandBuffer(commandCopyBuffer);
    }

    void createVertexBuffer() {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        // staging buffer, CPU vertex data will be put here and then transferred
        // to the GPU local vertex buffer
        vk::raii::Buffer stagingBuffer             = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, vertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        // vertex buffer
        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eVertexBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer,
                     vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        // staging buffer, CPU vertex data will be put here and then transferred
        // to the GPU local vertex buffer
        vk::raii::Buffer stagingBuffer             = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, indices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        // vertex buffer
        createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
    }

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (int i = 0; i < maxFramesInFlight; i++) {
            vk::DeviceSize bufferSize        = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer          = nullptr;
            vk::raii::DeviceMemory bufferMem = nullptr;

            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                         vk::MemoryPropertyFlagBits::eHostVisible |
                             vk::MemoryPropertyFlagBits::eHostCoherent,
                         buffer, bufferMem);

            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back(
                uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createDescriptorPool() {
        std::array poolSize{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, maxFramesInFlight),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler,
                                   maxFramesInFlight),
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = maxFramesInFlight,
            .poolSizeCount = poolSize.size(),
            .pPoolSizes    = poolSize.data(),
        };

        descriptorPool = vk::raii::DescriptorPool(logicalDevice, poolInfo);
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(maxFramesInFlight,
                                                     *descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool     = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts        = layouts.data(),
        };

        descriptorSets.clear();
        descriptorSets = logicalDevice.allocateDescriptorSets(allocInfo);

        for (int i = 0; i < maxFramesInFlight; i++) {
            vk::DescriptorBufferInfo bufferInfo{
                .buffer = uniformBuffers[i],
                .offset = 0,
                .range  = sizeof(UniformBufferObject),
            };

            vk::DescriptorImageInfo imageInfo{
                .sampler     = textureSampler,
                .imageView   = textureImageView,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

            std::array descriptorWrites{
                vk::WriteDescriptorSet{
                    .dstSet          = descriptorSets[i],
                    .dstBinding      = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo     = &bufferInfo,
                },
                vk::WriteDescriptorSet{
                    .dstSet          = descriptorSets[i],
                    .dstBinding      = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo      = &imageInfo,
                }};

            logicalDevice.updateDescriptorSets(descriptorWrites, {});
        }
    }

    void createCommandBuffers() {
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool        = commandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = maxFramesInFlight,
        };

        commandBuffers = vk::raii::CommandBuffers(logicalDevice, allocInfo);
    }

    void recordCommandBuffer(uint32_t image_idx) {
        vk::raii::CommandBuffer &commandBuffer = commandBuffers[frame_idx];

        commandBuffer.begin(vk::CommandBufferBeginInfo{});

        // changing image layout from undefined to color attachment optimal
        transitionImageLayout(image_idx, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eColorAttachmentOptimal, {},
                              vk::AccessFlagBits2::eColorAttachmentWrite,
                              vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                              vk::PipelineStageFlagBits2::eColorAttachmentOutput);

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0);
        vk::RenderingAttachmentInfo attachmentInfo = {
            .imageView   = swapchainImageViews[image_idx], // rendering to this image
                                                           // in the swapchain
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = clearColor};

        vk::RenderingInfo renderingInfo = {
            .renderArea           = {.offset = {0, 0}, .extent = swapchainExtent},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &attachmentInfo,
        };

        commandBuffer.beginRendering(renderingInfo);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

        commandBuffer.setViewport(
            0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width),
                            static_cast<float>(swapchainExtent.height), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));

        commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
        commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint16);

        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
                                         0, *descriptorSets[frame_idx], nullptr);
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        commandBuffer.endRendering();

        transitionImageLayout(image_idx, vk::ImageLayout::eColorAttachmentOptimal,
                              vk::ImageLayout::ePresentSrcKHR,
                              vk::AccessFlagBits2::eColorAttachmentWrite, {},
                              vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                              vk::PipelineStageFlagBits2::eBottomOfPipe);

        commandBuffer.end();
    }

    void transitionImageLayout(uint32_t image_idx, VkImageLayout oldLayout,
                               VkImageLayout newLayout, VkAccessFlags2 oldAccessMask,
                               VkAccessFlags2 newAccessMask,
                               VkPipelineStageFlags2 oldStageMask,
                               VkPipelineStageFlags2 newStageMask) {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask        = oldStageMask,
            .srcAccessMask       = oldAccessMask,
            .dstStageMask        = newStageMask,
            .dstAccessMask       = newAccessMask,
            .oldLayout           = oldLayout,
            .newLayout           = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = swapchainImages[image_idx],
            .subresourceRange    = {.aspectMask     = vk::ImageAspectFlagBits::eColor,
                                    .baseMipLevel   = 0,
                                    .levelCount     = 1,
                                    .baseArrayLayer = 0,
                                    .layerCount     = 1},
        };

        vk::DependencyInfo dependencyInfo = {
            .dependencyFlags         = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrier,
        };

        commandBuffers[frame_idx].pipelineBarrier2(dependencyInfo);
    }

    void createSyncObjects() {
        assert(presentCompleteSphrs.empty() && renderFinishedSphrs.empty() &&
               drawFences.empty());

        for (int i = 0; i < swapchainImages.size(); i++) {
            renderFinishedSphrs.emplace_back(logicalDevice, vk::SemaphoreCreateInfo{});
        }

        for (int i = 0; i < maxFramesInFlight; i++) {
            presentCompleteSphrs.emplace_back(logicalDevice, vk::SemaphoreCreateInfo{});
            drawFences.emplace_back(
                logicalDevice,
                vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        }
    }
};
