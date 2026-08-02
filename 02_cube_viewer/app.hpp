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
#include <variant>
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
        return {.binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX};
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        return {{
            {.location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, pos)},
            {.location = 1,
             .binding = 0,
             .format = VK_FORMAT_R32G32B32_SFLOAT,
             .offset = offsetof(Vertex, col)},
            {.location = 2,
             .binding = 0,
             .format = VK_FORMAT_R32G32_SFLOAT,
             .offset = offsetof(Vertex, texCoord)},
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

enum class InstanceError {
    ValidationLayerNotSupported,
    ExtensionNotSupported,
};

enum class SurfaceError {
    CreationFailed,
};

enum class PhysicalDeviceError {
    FailedToFindGPU,
};

enum class LogicalDeviceError {
    NoSuitableQueueFamily,
};

enum class FileError {
    FailedToOpen,
};

using AppError = std::variant<VkResult, InstanceError, SurfaceError, LogicalDeviceError>;

class App {
  public:
    static std::expected<App, AppError> init(Window &window);
    void deinit();

    bool isRunning();
    void pollEvents();
    std::expected<void, VkResult> endFrame();

  private:
    App() = default;

    Window *window = nullptr;

    bool running = false;

    std::array<const char *, 1> required_device_extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    // vk::raii::Context context;
    VkInstance instance = nullptr;
    VkPhysicalDevice physicalDevice = nullptr; // Physical device represents the GPU
    VkDevice logicalDevice = nullptr;          // Logical Device is the interface for the physical device
    VkQueue queue = nullptr;
    uint32_t queue_idx = UINT32_MAX;

    VkDebugUtilsMessengerEXT debugMessenger = nullptr;

    VkSurfaceKHR windowSurface = nullptr; // Surface to render to window

    VkSwapchainKHR swapchain = nullptr;
    std::vector<VkImage> swapchainImages;
    VkSurfaceFormatKHR swapchainSurfaceFormat;
    VkExtent2D swapchainExtent;
    std::vector<VkImageView> swapchainImageViews;

    VkDescriptorSetLayout descriptorSetLayout = nullptr;
    VkPipelineLayout pipelineLayout = nullptr;
    VkPipeline graphicsPipeline = nullptr;

    VkCommandPool commandPool = nullptr;
    std::vector<VkCommandBuffer> commandBuffers;

    uint32_t frame_idx = 0;

    // buffers
    VkBuffer vertexBuffer = nullptr;
    VkDeviceMemory vertexBufferMemory = nullptr;
    VkBuffer indexBuffer = nullptr;
    VkDeviceMemory indexBufferMemory = nullptr;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;

    VkDescriptorPool descriptorPool = nullptr;
    std::vector<VkDescriptorSet> descriptorSets;

    // textures
    VkImage textureImage = nullptr;
    VkDeviceMemory textureImageMemory = nullptr;
    VkImageView textureImageView = nullptr;
    VkSampler textureSampler = nullptr;

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
    std::expected<void, VkResult> drawFrame() {
        VkResult result = vkWaitForFences(logicalDevice, 1, &drawFences[frame_idx], VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        uint32_t image_idx;
        result = vkAcquireNextImageKHR(logicalDevice, swapchain, UINT64_MAX, presentCompleteSphrs[frame_idx],
                                       nullptr, &image_idx);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapchain();
            return std::unexpected(result);
        }
        if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            assert(result == VK_TIMEOUT || result == VK_NOT_READY);

            std::print(stderr, "Failed to acquire swap chain image");
            return std::unexpected(result);
        }

        updateUniformBuffer(frame_idx);

        result = vkResetFences(logicalDevice, 1, &drawFences[frame_idx]);
        if (result != VK_SUCCESS)
            return std::unexpected(result);

        result = vkResetCommandBuffer(commandBuffers[frame_idx], 0);
        if (result != VK_SUCCESS)
            return std::unexpected(result);

        recordCommandBuffer(image_idx);

        VkPipelineStageFlags waitDestinationStageMask(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
        const VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &presentCompleteSphrs[frame_idx], // semaphores to wait for
            .pWaitDstStageMask = &waitDestinationStageMask,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffers[frame_idx],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &renderFinishedSphrs[image_idx], // semaphores to signal when done
        };

        result = vkQueueSubmit(queue, 1, &submitInfo, drawFences[frame_idx]);
        if (result != VK_SUCCESS)
            return std::unexpected(result);

        const VkPresentInfoKHR presentInfoKHR{
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &renderFinishedSphrs[image_idx],
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &image_idx,
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
        static std::chrono::time_point startTime = std::chrono::high_resolution_clock::now();
        std::chrono::time_point currentTime = std::chrono::high_resolution_clock::now();

        float deltaTime =
            std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo;
        ubo.model =
            glm::rotate(glm::mat4(1.0f), deltaTime * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                               glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f),
                                    static_cast<float>(swapchainExtent.width) /
                                        static_cast<float>(swapchainExtent.height),
                                    0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    /* SETUP METHODS */

    static std::expected<std::vector<char>, FileError> readFile(const std::string &path) {
        // std::ios::ate - reading starts at the end of file
        // std::ios::binary - reads file as a binary
        std::ifstream fin(path, std::ios::ate | std::ios::binary);

        if (!fin.is_open()) {
            return std::unexpected(FileError::FailedToOpen);
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
    static VKAPI_ATTR VkBool32 VKAPI_CALL
    debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT type,
                  const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData) {
        std::cerr << "[Validation layer]: " << std::to_string(severity) << " , "
                  << "[Type]: " << std::to_string(type) << " , "
                  << "[Message]:" << std::endl
                  << std::endl
                  << pCallbackData->pMessage << std::endl
                  << "----------------" << std::endl;

        return VK_FALSE;
    }

    std::expected<void, VkResult> setupDebugMessenger() {
        if (!enableValidationLayers)
            return {};

        VkDebugUtilsMessageSeverityFlagsEXT severityFlags(VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
        VkDebugUtilsMessageTypeFlagsEXT messageTypeFlags(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
                                                         VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT);

        VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{.messageSeverity = severityFlags,
                                                                            .messageType = messageTypeFlags,
                                                                            .pfnUserCallback =
                                                                                &debugCallback};

        VkResult result = vkCreateDebugUtilsMessengerEXT(instance, &debugUtilsMessengerCreateInfoEXT, nullptr,
                                                         &debugMessenger);
        if (result != VK_SUCCESS)
            return std::unexpected(result);

        return {};
    }

    std::vector<const char *> getRequiredInstanceExtensions() {
        uint32_t sdlExtensionCount = 0;
        char const *const *sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);

        std::vector extensions(sdlExtensions, sdlExtensions + sdlExtensionCount);
        if (enableValidationLayers) {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    std::expected<void, AppError> createInstance() {
        // VULKAN INSTANCE CREATION
        // instance is used to communicate with vulkan
        VkApplicationInfo constexpr appInfo{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                            .pApplicationName = "Learn Vulkan",
                                            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                            .pEngineName = "No Engine",
                                            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                            .apiVersion = VK_API_VERSION_1_3};

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
                return std::unexpected(AppError{InstanceError::ValidationLayerNotSupported});
            }
        }

        // EXTENSIONS
        std::vector<char const *> requiredExtensions = getRequiredInstanceExtensions();

        uint32_t extension_count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> extensionProperties(extension_count);
        vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensionProperties.data());

        for (const char *required_extension : requiredExtensions) {
            bool found = false;
            for (VkExtensionProperties const &extension : extensionProperties) {
                if (strcmp(extension.extensionName, required_extension)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                return std::unexpected(AppError{InstanceError::ExtensionNotSupported});
            }
        }

        // CREATING THE INSTANCE
        VkInstanceCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                        .pApplicationInfo = &appInfo,
                                        .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
                                        .ppEnabledLayerNames = requiredLayers.data(),
                                        .enabledExtensionCount =
                                            static_cast<uint32_t>(requiredExtensions.size()),
                                        .ppEnabledExtensionNames = requiredExtensions.data()};

        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);
        if (result != VK_SUCCESS) {
            return std::unexpected(AppError{result});
        }

        return {};
    }

    std::expected<void, AppError> createWindowSurface() {
        VkSurfaceKHR surface;

        if (!SDL_Vulkan_CreateSurface(window->getSDLWindow(), instance, nullptr, &surface)) {
            std::print(stderr, "SDL_Vulkan_CreateSurface failed: {}\n", SDL_GetError());
            return std::unexpected(AppError{SurfaceError::CreationFailed});
        }

        windowSurface = surface;
        return {};
    }

    bool isDeviceSuitable(VkPhysicalDevice const &physicalDevice) {
        // if supports vulkan 1.3
        VkPhysicalDeviceProperties physical_device_properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physical_device_properties);
        bool supports_vulkan1_3 = physical_device_properties.apiVersion >= VK_API_VERSION_1_3;

        // if supports graphics queue family
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queue_family_count, queue_families.data());

        bool supports_graphics = false;
        for (auto const &queue_family : queue_families) {
            if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                supports_graphics = true;
                break;
            }
        }

        // if supports specific extensions
        uint32_t extension_count = 0;
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extension_count, nullptr);
        std::vector<VkExtensionProperties> available_device_extensions(extension_count);
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extension_count,
                                             available_device_extensions.data());

        // if any of the required device extensions aren't available -> false
        bool supports_all_required_extensions = true;
        for (char const *required_device_extension : required_device_extensions) {
            bool found = true;
            for (auto const &available_device_extension : available_device_extensions) {
                if (strcmp(available_device_extension.extensionName, required_device_extension) == 0) {
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
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
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
            vulkan_11_features.shaderDrawParameters && vulkan_13_features.synchronization2 &&
            vulkan_13_features.dynamicRendering && extended_dynamic_state_features.extendedDynamicState;

        return supports_vulkan1_3 && supports_graphics && supports_all_required_extensions &&
               supports_required_features;
    }

    std::expected<void, PhysicalDeviceError> pickPhysicalDevice() {
        // checking if physical devices meet requirements

        uint32_t physical_device_count = 0;
        vkEnumeratePhysicalDevices(instance, &physical_device_count, nullptr);
        std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
        vkEnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data());

        // find if a GPU meets all the requirements
        bool found = false;
        for (auto const &physical_device : physical_devices) {
            if (isDeviceSuitable(physical_device)) {
                found = true;
                physicalDevice = physical_device;
                break;
            }
        }
        if (!found) {
            return std::unexpected(PhysicalDeviceError::FailedToFindGPU);
        }

        return {};
    }

    std::expected<void, AppError> createLogicalDevice() {
        uint32_t queue_family_properties_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queue_family_properties_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_properties_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queue_family_properties_count,
                                                 queue_family_properties.data());

        // check for support of both graphics and present queue families
        for (uint32_t queue_family_prop_idx = 0; queue_family_prop_idx < queue_family_properties.size();
             queue_family_prop_idx++) {
            VkBool32 present_support = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queue_family_prop_idx, windowSurface,
                                                 &present_support);

            if ((queue_family_properties[queue_family_prop_idx].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
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
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT,
            .extendedDynamicState = VK_TRUE,
        };

        VkPhysicalDeviceVulkan13Features vulkan_13_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
            .pNext = &extended_dynamic_state_features,
            .synchronization2 = VK_TRUE,
            .dynamicRendering = VK_TRUE,
        };

        VkPhysicalDeviceVulkan11Features vulkan_11_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
            .pNext = &vulkan_13_features,
            .shaderDrawParameters = VK_TRUE,
        };

        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &vulkan_11_features,
            .features = {.samplerAnisotropy = VK_TRUE},
        };

        float queuePriority = 0.5f; // priority for scheduling of command buffer
                                    // execution, needed even if there is one queue
        VkDeviceQueueCreateInfo device_queue_create_info{.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                                         .queueFamilyIndex = queue_idx,
                                                         .queueCount = 1,
                                                         .pQueuePriorities = &queuePriority};

        VkDeviceCreateInfo deviceCreateInfo{.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                            .pNext = &features2, // connecting the chain of
                                                                 // features to vulkan
                                            .queueCreateInfoCount = 1,
                                            .pQueueCreateInfos = &device_queue_create_info,
                                            .enabledExtensionCount =
                                                static_cast<uint32_t>(required_device_extensions.size()),
                                            .ppEnabledExtensionNames = required_device_extensions.data()};

        VkResult result = vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &logicalDevice);
        if (result != VK_SUCCESS) {
            return std::unexpected(AppError{result});
        }

        vkGetDeviceQueue(logicalDevice, queue_idx, 0, &queue);

        return {};
    }

    VkSurfaceFormatKHR chooseSwapchainSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
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

    VkPresentModeKHR chooseSwapchainPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes) {
        // fifo present mode - stores rendered images in a queue, takes an image
        // from the front of the queue to display every time the display
        // refreshes mailbox present mode - like fifo, but when the queue is
        // full it replaces old images with new ones to display images as fast as possible

        bool found_fifo = false;
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

    VkExtent2D chooseSwapchainExtent(const VkSurfaceCapabilitiesKHR &surfaceCapabilities) {
        // extent is the resolution of the images in the swapchain

        if (surfaceCapabilities.currentExtent.width != UINT32_MAX) {
            return surfaceCapabilities.currentExtent;
        }

        int width, height;
        SDL_GetWindowSizeInPixels(window->getSDLWindow(), &width, &height);

        return VkExtent2D{std::clamp<uint32_t>(width, surfaceCapabilities.minImageExtent.width,
                                               surfaceCapabilities.maxImageExtent.width),
                          std::clamp<uint32_t>(height, surfaceCapabilities.minImageExtent.height,
                                               surfaceCapabilities.maxImageExtent.height)};
    }

    uint32_t chooseSwapchainMinImageCount(const VkSurfaceCapabilitiesKHR &surfaceCapabilities) {
        uint32_t minImgCount = std::max(uint32_t(3), surfaceCapabilities.minImageCount);

        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImgCount)) {
            minImgCount = surfaceCapabilities.maxImageCount;
        }

        return minImgCount;
    }

    void cleanupSwapchain() {
        swapchainImageViews.clear();
        swapchain = nullptr;
    }

    void recreateSwapchain() {
        int width = 0;
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

    std::expected<void, VkResult> createSwapchain() {
        VkSurfaceCapabilitiesKHR surface_capabilities;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, windowSurface, &surface_capabilities);
        swapchainExtent = chooseSwapchainExtent(surface_capabilities);
        uint32_t minImageCount = chooseSwapchainMinImageCount(surface_capabilities);

        uint32_t available_formats_count = 0;
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, windowSurface, &available_formats_count,
                                             nullptr);
        std::vector<VkSurfaceFormatKHR> available_formats(available_formats_count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, windowSurface, &available_formats_count,
                                             available_formats.data());

        swapchainSurfaceFormat = chooseSwapchainSurfaceFormat(available_formats);

        uint32_t available_present_modes_count = 0;
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, windowSurface, &available_formats_count,
                                                  nullptr);
        std::vector<VkPresentModeKHR> available_present_modes(available_present_modes_count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, windowSurface, &available_formats_count,
                                                  available_present_modes.data());

        VkPresentModeKHR presentMode = chooseSwapchainPresentMode(available_present_modes);

        VkSwapchainCreateInfoKHR swapchainCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            .surface = windowSurface,
            .minImageCount = minImageCount,
            .imageFormat = swapchainSurfaceFormat.format,
            .imageColorSpace = swapchainSurfaceFormat.colorSpace,
            .imageExtent = swapchainExtent,
            .imageArrayLayers = 1,
            .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .preTransform = surface_capabilities.currentTransform,
            .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = presentMode,
            .clipped = true,
            .oldSwapchain = swapchain,
        };

        VkResult result = vkCreateSwapchainKHR(logicalDevice, &swapchainCreateInfo, nullptr, &swapchain);
        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        uint32_t swapchain_image_count = 0;
        vkGetSwapchainImagesKHR(logicalDevice, swapchain, &swapchain_image_count, nullptr);
        swapchainImages.resize(swapchain_image_count);
        vkGetSwapchainImagesKHR(logicalDevice, swapchain, &swapchain_image_count, swapchainImages.data());

        return {};
    }

    std::expected<VkImageView, VkResult> createImageView(VkImage &image, VkFormat format) {
        VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = format,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        };

        VkImageView image_view;
        VkResult result = vkCreateImageView(logicalDevice, &viewInfo, nullptr, &image_view);
        if (result != VK_SUCCESS) {
            return std::unexpected(result);
        }

        return image_view;
    }

    void createImageViews() {
        assert(swapchainImageViews.empty());

        VkImageViewCreateInfo imageViewCreateInfo{
            .viewType = vk::ImageViewType::e2D,
            .format = swapchainSurfaceFormat.format,
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}};

        for (vk::Image &image : swapchainImages) {
            imageViewCreateInfo.image = image;
            swapchainImageViews.emplace_back(vk::raii::ImageView(logicalDevice, imageViewCreateInfo));
        }
    }

    void createDescriptorSetLayout() {
        std::array bindings = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                           vk::ShaderStageFlagBits::eVertex, nullptr),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1,
                                           vk::ShaderStageFlagBits::eFragment, nullptr),
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo{
            .bindingCount = bindings.size(),
            .pBindings = bindings.data(),
        };
        descriptorSetLayout = vk::raii::DescriptorSetLayout(logicalDevice, layoutInfo);
    }

    void createGraphicsPipeline() {
        /* SHADER STAGE SETUP */

        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(appPath() / "shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = shaderModule,
            .pName = "vertMain",           // the entrypoint in the slang code
            .pSpecializationInfo = nullptr // used to set constants in shader per-pipeline
        };

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        /* INPUT STAGE SETUP */

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()};

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology =
                                                                   vk::PrimitiveTopology::eTriangleList};

        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1,
        };

        std::array dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynamicState{.dynamicStateCount =
                                                            static_cast<uint32_t>(dynamicStates.size()),
                                                        .pDynamicStates = dynamicStates.data()};

        /* RASTERIZATION STAGE SETUP */

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False,        // if true, fragments past the near or far plane
                                                  // will be clamped rather than discarded
            .rasterizerDiscardEnable = vk::False, // if true, skips rasterizer stage
            .polygonMode = vk::PolygonMode::eFill,
            .cullMode = vk::CullModeFlagBits::eBack,
            .frontFace = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable = vk::False, // if true, rasterizer can make
                                          // adjustments to depth values
            .lineWidth = 1.0f};

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

        /* COLOR BLENDING STAGE SETUP */

        // linearly interpolated blending
        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
            .blendEnable = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp = vk::BlendOp::eAdd,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

        vk::PipelineColorBlendStateCreateInfo colorBlending{.logicOpEnable = vk::False,
                                                            .logicOp = vk::LogicOp::eCopy,
                                                            .attachmentCount = 1,
                                                            .pAttachments = &colorBlendAttachment};

        /* PIPELINE SETUP */

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 1, .pSetLayouts = &*descriptorSetLayout, .pushConstantRangeCount = 0};
        pipelineLayout = vk::raii::PipelineLayout(logicalDevice, pipelineLayoutInfo);

        vk::StructureChain pipelineCreateInfoChain = {
            vk::GraphicsPipelineCreateInfo{
                .stageCount = 2,
                .pStages = shaderStages,
                .pVertexInputState = &vertexInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState = &multisampling,
                .pColorBlendState = &colorBlending,
                .pDynamicState = &dynamicState,
                .layout = pipelineLayout,
                .renderPass = nullptr // using dynamic rendering
            },

            vk::PipelineRenderingCreateInfo{.colorAttachmentCount = 1,
                                            .pColorAttachmentFormats = &swapchainSurfaceFormat.format}};

        graphicsPipeline =
            VkPipeline(logicalDevice, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    }

    [[nodiscard]] VkShaderModule createShaderModule(const std::vector<char> &code) const {
        vk::ShaderModuleCreateInfo createInfo{.codeSize = code.size() * sizeof(char),
                                              .pCode = reinterpret_cast<const uint32_t *>(code.data())};

        vk::raii::ShaderModule shaderModule{logicalDevice, createInfo};
        return shaderModule;
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueIdx,
        };
        commandPool = vk::raii::CommandPool(logicalDevice, poolInfo);
    }

    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                     VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage &image,
                     VkDeviceMemory &imageMemory) {
        vk::ImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = tiling,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };

        image = vk::raii::Image(logicalDevice, imageInfo);

        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties),
        };
        imageMemory = vk::raii::DeviceMemory(logicalDevice, allocInfo);
        image.bindMemory(imageMemory, 0);
    }

    void transitionImageLayout(const VkImage &image, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginOneTimeCommandBuffer();

        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .image = image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };

        VkPipelineStageFlags srcStage;
        VkPipelineStageFlags dstStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            srcStage = vk::PipelineStageFlagBits::eTopOfPipe;
            dstStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
                   newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            srcStage = vk::PipelineStageFlagBits::eTransfer;
            dstStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("Unsupported layout transition");
        }

        commandBuffer.pipelineBarrier(srcStage, dstStage, {}, {}, nullptr, barrier);

        endOneTimeCommandBuffer(commandBuffer);
    }

    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load((appPath() / "textures/dirt.png").c_str(), &texWidth, &texHeight,
                                    &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("Failed to load texture image");
        }

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        void *data = stagingBufferMemory.mapMemory(0, imageSize);
        memcpy(data, pixels, imageSize);
        stagingBufferMemory.unmapMemory();

        stbi_image_free(pixels);

        createImage(texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                    vk::MemoryPropertyFlagBits::eDeviceLocal, textureImage, textureImageMemory);

        transitionImageLayout(textureImage, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eTransferDstOptimal);
        copyBufferToImage(stagingBuffer, textureImage, texWidth, texHeight);
        transitionImageLayout(textureImage, vk::ImageLayout::eTransferDstOptimal,
                              vk::ImageLayout::eShaderReadOnlyOptimal);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);
    }

    void createTextureSampler() {
        vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
        vk::SamplerCreateInfo samplerInfo{
            .magFilter = vk::Filter::eNearest,
            .minFilter = vk::Filter::eNearest,
            .mipmapMode = vk::SamplerMipmapMode::eLinear,
            .addressModeU = vk::SamplerAddressMode::eRepeat,
            .addressModeV = vk::SamplerAddressMode::eRepeat,
            .addressModeW = vk::SamplerAddressMode::eRepeat,
            .mipLodBias = 0.0f,
            .anisotropyEnable = vk::True,
            .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
            .compareEnable = vk::False,
            .compareOp = vk::CompareOp::eAlways,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = vk::False,
        };

        textureSampler = VkSampler(logicalDevice, samplerInfo);
    }

    void copyBufferToImage(const VkBuffer &buffer, VkImage &image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginOneTimeCommandBuffer();

        VkBufferImageCopy region{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, 1},
        };

        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, {region});

        endOneTimeCommandBuffer(commandBuffer);
    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
                      VkBuffer &buffer, VkDeviceMemory &bufferMemory) {
        VkBufferCreateInfo bufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };

        buffer = VkBuffer(logicalDevice, bufferInfo);

        VkMemoryRequirements memRequirements = buffer.getMemoryRequirements();

        VkMemoryAllocateInfo memoryAllocateInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties),
        };

        bufferMemory = VkDeviceMemory(logicalDevice, memoryAllocateInfo);

        buffer.bindMemory(*bufferMemory, 0);
    }

    VkCommandBuffer beginOneTimeCommandBuffer() {
        VkCommandBufferAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };

        VkCommandBuffer commandBuffer = std::move(logicalDevice.allocateCommandBuffers(allocInfo).front());

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
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &*commandBuffer,
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
        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, vertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        // vertex buffer
        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, vertexBuffer, vertexBufferMemory);

        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
    }

    void createIndexBuffer() {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        // staging buffer, CPU vertex data will be put here and then transferred
        // to the GPU local vertex buffer
        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer, stagingBufferMemory);

        void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(dataStaging, indices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        // vertex buffer
        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal, indexBuffer, indexBufferMemory);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);
    }

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (int i = 0; i < maxFramesInFlight; i++) {
            vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
            vk::raii::Buffer buffer = nullptr;
            vk::raii::DeviceMemory bufferMem = nullptr;

            createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                         buffer, bufferMem);

            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferSize));
        }
    }

    void createDescriptorPool() {
        std::array poolSize{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, maxFramesInFlight),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, maxFramesInFlight),
        };

        vk::DescriptorPoolCreateInfo poolInfo{
            .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets = maxFramesInFlight,
            .poolSizeCount = poolSize.size(),
            .pPoolSizes = poolSize.data(),
        };

        descriptorPool = vk::raii::DescriptorPool(logicalDevice, poolInfo);
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(maxFramesInFlight, *descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo{
            .descriptorPool = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts = layouts.data(),
        };

        descriptorSets.clear();
        descriptorSets = logicalDevice.allocateDescriptorSets(allocInfo);

        for (int i = 0; i < maxFramesInFlight; i++) {
            vk::DescriptorBufferInfo bufferInfo{
                .buffer = uniformBuffers[i],
                .offset = 0,
                .range = sizeof(UniformBufferObject),
            };

            vk::DescriptorImageInfo imageInfo{.sampler = textureSampler,
                                              .imageView = textureImageView,
                                              .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

            std::array descriptorWrites{vk::WriteDescriptorSet{
                                            .dstSet = descriptorSets[i],
                                            .dstBinding = 0,
                                            .dstArrayElement = 0,
                                            .descriptorCount = 1,
                                            .descriptorType = vk::DescriptorType::eUniformBuffer,
                                            .pBufferInfo = &bufferInfo,
                                        },
                                        vk::WriteDescriptorSet{
                                            .dstSet = descriptorSets[i],
                                            .dstBinding = 1,
                                            .dstArrayElement = 0,
                                            .descriptorCount = 1,
                                            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
                                            .pImageInfo = &imageInfo,
                                        }};

            logicalDevice.updateDescriptorSets(descriptorWrites, {});
        }
    }

    void createCommandBuffers() {
        vk::CommandBufferAllocateInfo allocInfo{
            .commandPool = commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = maxFramesInFlight,
        };

        commandBuffers = vk::raii::CommandBuffers(logicalDevice, allocInfo);
    }

    void recordCommandBuffer(uint32_t image_idx) {
        vk::raii::CommandBuffer &commandBuffer = commandBuffers[frame_idx];

        commandBuffer.begin(vk::CommandBufferBeginInfo{});

        // changing image layout from undefined to color attachment optimal
        transitionImageLayout(
            image_idx, vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal, {},
            vk::AccessFlagBits2::eColorAttachmentWrite, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput);

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0);
        vk::RenderingAttachmentInfo attachmentInfo = {
            .imageView = swapchainImageViews[image_idx], // rendering to this image
                                                         // in the swapchain
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = clearColor};

        vk::RenderingInfo renderingInfo = {
            .renderArea = {.offset = {0, 0}, .extent = swapchainExtent},
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &attachmentInfo,
        };

        commandBuffer.beginRendering(renderingInfo);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

        commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width),
                                                  static_cast<float>(swapchainExtent.height), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));

        commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
        commandBuffer.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint16);

        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
                                         *descriptorSets[frame_idx], nullptr);
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        commandBuffer.endRendering();

        transitionImageLayout(image_idx, vk::ImageLayout::eColorAttachmentOptimal,
                              vk::ImageLayout::ePresentSrcKHR, vk::AccessFlagBits2::eColorAttachmentWrite, {},
                              vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                              vk::PipelineStageFlagBits2::eBottomOfPipe);

        commandBuffer.end();
    }

    void transitionImageLayout(uint32_t image_idx, VkImageLayout oldLayout, VkImageLayout newLayout,
                               VkAccessFlags2 oldAccessMask, VkAccessFlags2 newAccessMask,
                               VkPipelineStageFlags2 oldStageMask, VkPipelineStageFlags2 newStageMask) {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask = oldStageMask,
            .srcAccessMask = oldAccessMask,
            .dstStageMask = newStageMask,
            .dstAccessMask = newAccessMask,
            .oldLayout = oldLayout,
            .newLayout = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = swapchainImages[image_idx],
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor,
                                 .baseMipLevel = 0,
                                 .levelCount = 1,
                                 .baseArrayLayer = 0,
                                 .layerCount = 1},
        };

        vk::DependencyInfo dependencyInfo = {
            .dependencyFlags = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier,
        };

        commandBuffers[frame_idx].pipelineBarrier2(dependencyInfo);
    }

    void createSyncObjects() {
        assert(presentCompleteSphrs.empty() && renderFinishedSphrs.empty() && drawFences.empty());

        for (int i = 0; i < swapchainImages.size(); i++) {
            renderFinishedSphrs.emplace_back(logicalDevice, vk::SemaphoreCreateInfo{});
        }

        for (int i = 0; i < maxFramesInFlight; i++) {
            presentCompleteSphrs.emplace_back(logicalDevice, vk::SemaphoreCreateInfo{});
            drawFences.emplace_back(logicalDevice,
                                    vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        }
    }
};
