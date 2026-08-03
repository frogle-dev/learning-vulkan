#pragma once

#include <complex>
#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include "vulkan/vulkan.hpp"
#include <vulkan/vulkan.hpp>

#include <vk_mem_alloc.h>

#include <SDL3/SDL_vulkan.h>

#include "Window/window.hpp"
#include "stb_image.h"

#include <chrono>
#include <fstream>
#include <source_location>

inline std::filesystem::path appPath()
{
    return std::filesystem::canonical("/proc/self/exe").parent_path().parent_path();
}

constexpr uint8_t max_frames_in_flight = 2;

#ifdef NDEBUG
constexpr bool enable_validation_layers = false;
#else
constexpr bool enable_validation_layers = true;
#endif

constexpr std::array<char const *, 1> validation_layers = {"VK_LAYER_KHRONOS_validation"};

struct Vertex
{
    glm::vec2 pos_;
    glm::vec3 col_;
    glm::vec2 tex_coord_;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        return {.binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex};
    }

    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions()
    {
        return {{
            {.location = 0,
             .binding  = 0,
             .format   = vk::Format::eR32G32Sfloat,
             .offset   = offsetof(Vertex, pos_)},
            {.location = 1,
             .binding  = 0,
             .format   = vk::Format::eR32G32B32Sfloat,
             .offset   = offsetof(Vertex, col_)},
            {.location = 2,
             .binding  = 0,
             .format   = vk::Format::eR32G32Sfloat,
             .offset   = offsetof(Vertex, tex_coord_)},
        }};
    }
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.5f, 0.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
};

const std::vector<uint16_t> indices = {0, 1, 2, 2, 3, 0};

struct UniformBufferObject
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

enum class ErrorKind
{
    VulkanFailure,
    SDLFailure,
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

struct AppError
{
    AppError(std::string message, ErrorKind kind,
             std::source_location location = std::source_location::current())
        : kind_(kind), message_(std::move(message)), location_(location)
    {
    }

    AppError(std::string message, vk::Result result,
             std::source_location location = std::source_location::current())
        : kind_(ErrorKind::VulkanFailure), message_(std::move(message)), vk_result_(result),
          location_(location)
    {
    }

    static std::unexpected<AppError> unexpected(AppError error)
    {
        return std::unexpected(std::move(error));
    }

    ErrorKind kind_;
    std::string message_;
    std::optional<vk::Result> vk_result_ = std::nullopt;
    std::source_location location_;
};

template <typename T> using Result = std::expected<T, AppError>;

class App
{
  public:
    [[nodiscard]] static Result<App> init(Window &window);
    [[nodiscard]] Result<void> deinit();

    bool isRunning();
    [[nodiscard]] Result<void> pollEvents();
    [[nodiscard]] Result<void> endFrame();

  private:
    App() = default;

    Window *window_ = nullptr;

    bool running_ = false;

    std::array<const char *, 1> required_device_extensions_ = {vk::KHRSwapchainExtensionName};

    vk::Instance instance_              = nullptr;
    vk::PhysicalDevice physical_device_ = nullptr; // Physical device represents the GPU
    vk::Device logical_device_ = nullptr; // Logical Device is the interface for the physical device
    vk::Queue queue_           = nullptr;
    uint32_t queue_family_idx_ = UINT32_MAX;

    vk::DebugUtilsMessengerEXT debug_messenger_ = nullptr;

    vk::SurfaceKHR window_surface_ = nullptr; // Surface to render to window

    vk::SwapchainKHR swapchain_ = nullptr;
    std::vector<vk::Image> swapchain_images_;
    vk::SurfaceFormatKHR swapchain_surface_format_;
    vk::Extent2D swapchain_extent_;
    std::vector<vk::ImageView> swapchain_image_views_;

    vk::DescriptorSetLayout descriptor_set_layout_ = nullptr;
    vk::PipelineLayout pipeline_layout_            = nullptr;
    vk::Pipeline graphics_pipeline_                = nullptr;

    vk::CommandPool command_pool_ = nullptr;
    std::vector<vk::CommandBuffer> command_buffers_;

    uint32_t frame_idx_ = 0;

    // buffers
    vk::Buffer vertex_buffer_              = nullptr;
    vk::DeviceMemory vertex_buffer_memory_ = nullptr;
    vk::Buffer index_buffer_               = nullptr;
    vk::DeviceMemory index_buffer_memory_  = nullptr;

    std::vector<vk::Buffer> uniform_buffers_;
    std::vector<vk::DeviceMemory> uniform_buffers_memory_;
    std::vector<void *> uniform_buffers_mapped_;

    vk::DescriptorPool descriptor_pool_ = nullptr;
    std::vector<VkDescriptorSet> descriptor_sets_;

    // textures
    vk::Image texture_image_               = nullptr;
    vk::DeviceMemory texture_image_memory_ = nullptr;
    vk::ImageView texture_image_view_      = nullptr;
    vk::Sampler texture_sampler_           = nullptr;

    // Sync objects
    std::vector<vk::Semaphore> present_complete_sphrs_;
    std::vector<vk::Semaphore> render_finished_sphrs_;
    std::vector<vk::Fence> draw_fences_;

    /* APPLICATION METHODS */

    [[nodiscard]]
    Result<void> initVulkan()
    {
        auto expected = createInstance();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = setupDebugMessenger();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createWindowSurface();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = pickPhysicalDevice();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createLogicalDevice();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createSwapchain();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createImageViews();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createDescriptorSetLayout();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createGraphicsPipeline();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createCommandPool();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createTextureImage();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createTextureImageView();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createTextureSampler();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createVertexBuffer();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createIndexBuffer();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createUniformBuffers();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createDescriptorPool();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createDescriptorSets();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createCommandBuffers();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createSyncObjects();
        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> drawFrame()
    {
        vk::Result result =
            logical_device_.waitForFences(1, &draw_fences_[frame_idx_], VK_TRUE, UINT64_MAX);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to wait for fences", result});

        uint32_t image_idx;
        result = logical_device_.acquireNextImageKHR(
            swapchain_, UINT64_MAX, present_complete_sphrs_[frame_idx_], nullptr, &image_idx);

        if (result == vk::Result::eErrorOutOfDateKHR)
        {
            auto expected = recreateSwapchain();
            if (!expected)
                return AppError::unexpected(expected.error());

            return AppError::unexpected({"Swapchain image out of date", result});
        }

        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
        {
            assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);

            return AppError::unexpected({"Failed to acquire swap chain image", result});
        }

        updateUniformBuffer(frame_idx_);

        result = logical_device_.resetFences(1, &draw_fences_[frame_idx_]);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to reset fences", result});

        result = command_buffers_[frame_idx_].reset({});
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to reset command buffer", result});

        auto expected = recordCommandBuffer(image_idx);
        if (!expected)
            return AppError::unexpected(expected.error());

        vk::PipelineStageFlags waitDestinationStageMask(
            vk::PipelineStageFlagBits::eColorAttachmentOutput);

        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount   = 1,
            .pWaitSemaphores      = &present_complete_sphrs_[frame_idx_], // semaphores to wait for
            .pWaitDstStageMask    = &waitDestinationStageMask,
            .commandBufferCount   = 1,
            .pCommandBuffers      = &command_buffers_[frame_idx_],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores =
                &render_finished_sphrs_[image_idx], // semaphores to signal when done
        };

        result = queue_.submit(1, &submitInfo, draw_fences_[frame_idx_]);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to submit queue", result});

        const vk::PresentInfoKHR presentInfoKHR{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &render_finished_sphrs_[image_idx],
            .swapchainCount     = 1,
            .pSwapchains        = &swapchain_,
            .pImageIndices      = &image_idx,
        };

        result = queue_.presentKHR(&presentInfoKHR);
        assert(result == vk::Result::eSuccess);
        if (result == vk::Result::eSuboptimalKHR || result == vk::Result::eErrorOutOfDateKHR)
        {
            auto expected = recreateSwapchain();

            if (!expected)
                return AppError::unexpected(expected.error());
        }

        frame_idx_ = (frame_idx_ + 1) % max_frames_in_flight;

        return {};
    }

    void updateUniformBuffer(uint32_t current_image)
    {
        static std::chrono::time_point start_time = std::chrono::high_resolution_clock::now();
        std::chrono::time_point current_time      = std::chrono::high_resolution_clock::now();

        float delta_time =
            std::chrono::duration<float, std::chrono::seconds::period>(current_time - start_time)
                .count();

        UniformBufferObject ubo;
        ubo.model = glm::rotate(glm::mat4(1.0f), delta_time * glm::radians(90.0f),
                                glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view  = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                                glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj  = glm::perspective(glm::radians(45.0f),
                                     static_cast<float>(swapchain_extent_.width) /
                                         static_cast<float>(swapchain_extent_.height),
                                     0.1f, 10.0f);

        ubo.proj[1][1] *= -1;

        memcpy(uniform_buffers_mapped_[current_image], &ubo, sizeof(ubo));
    }

    /* SETUP METHODS */

    [[nodiscard]]
    static Result<std::vector<char>> readFile(const std::string &path)
    {
        // std::ios::ate - reading starts at the end of file
        // std::ios::binary - reads file as a binary
        std::ifstream fin(path, std::ios::ate | std::ios::binary);

        if (!fin.is_open())
        {
            return AppError::unexpected({"Failed to open file", ErrorKind::FailedToOpenFile});
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
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT severity, vk::DebugUtilsMessageTypeFlagsEXT type,
        const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
    {
        switch (severity)
        {
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
            spdlog::error("[Validation Layer]: \n \
                                [Type]: {} \n \n \
                                [Message]: {} \n \
                                --------------",
                          vk::to_string(type), pCallbackData->pMessage);

            break;
        case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
            spdlog::warn("[Validation Layer]: \n \
                                [Type]: {} \n \n \
                                [Message]: {} \n \
                                --------------",
                         vk::to_string(type), pCallbackData->pMessage);

            break;
        default:
            break;
        }

        return vk::False;
    }

    [[nodiscard]]
    Result<void> setupDebugMessenger()
    {
        if (!enable_validation_layers)
            return {};

        vk::DebugUtilsMessageSeverityFlagsEXT severity_flags =
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
        vk::DebugUtilsMessageTypeFlagsEXT message_type_flags =
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;

        vk::DebugUtilsMessengerCreateInfoEXT create_info{.messageSeverity = severity_flags,
                                                         .messageType     = message_type_flags,
                                                         .pfnUserCallback = debugCallback};

        auto result = instance_.createDebugUtilsMessengerEXT(create_info);

        if (result.result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to create debug utils messenger", result.result});

        debug_messenger_ = result.value;

        return {};
    }

    std::vector<const char *> getRequiredInstanceExtensions()
    {
        uint32_t sdlExtensionCount       = 0;
        char const *const *sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);

        std::vector extensions(sdlExtensions, sdlExtensions + sdlExtensionCount);
        if (enable_validation_layers)
        {
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }

        return extensions;
    }

    [[nodiscard]]
    Result<void> createInstance()
    {
        // VULKAN INSTANCE CREATION
        // instance is used to communicate with vulkan
        vk::ApplicationInfo constexpr app_info{.pApplicationName   = "Learn Vulkan",
                                               .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
                                               .pEngineName        = "No Engine",
                                               .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
                                               .apiVersion         = VK_API_VERSION_1_3};

        // VALIDATION LAYERS
        std::vector<char const *> required_layers;
        if (enable_validation_layers)
        {
            required_layers.assign(validation_layers.begin(), validation_layers.end());
        }

        // check if validation layers are available
        auto layer_properties = vk::enumerateInstanceLayerProperties();
        if (layer_properties.result != vk::Result::eSuccess)
            return AppError::unexpected(
                {"Failed to enumerate instance layer properties", layer_properties.result});

        for (char const *required_layer : required_layers)
        {
            bool found = false;
            for (auto const &layer : layer_properties.value)
            {
                if (strcmp(layer.layerName, required_layer) == 0)
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                return AppError::unexpected(
                    {"Validation layer not supported", ErrorKind::ValidationLayerNotSupported});
            }
        }

        // EXTENSIONS
        std::vector<char const *> required_extensions = getRequiredInstanceExtensions();

        auto extension_properties = vk::enumerateInstanceExtensionProperties();
        if (extension_properties.result != vk::Result::eSuccess)
            return AppError::unexpected(
                {"Failed to enumerate extension properties", extension_properties.result});

        for (char const *required_extension : required_extensions)
        {
            bool found = false;
            for (auto const &extension : extension_properties.value)
            {
                if (strcmp(extension.extensionName, required_extension))
                {
                    found = true;
                    break;
                }
            }

            if (!found)
            {
                return AppError::unexpected(
                    {"Extension not supported", ErrorKind::ExtensionNotSupported});
            }
        }

        // CREATING THE INSTANCE
        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo        = &app_info,
            .enabledLayerCount       = static_cast<uint32_t>(required_layers.size()),
            .ppEnabledLayerNames     = required_layers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(required_extensions.size()),
            .ppEnabledExtensionNames = required_extensions.data()};

        vk::Result result = vk::createInstance(&createInfo, nullptr, &instance_);
        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected({"Failed to create instance", result});
        }

        return {};
    }

    [[nodiscard]]
    Result<void> createWindowSurface()
    {
        VkSurfaceKHR c_surface;

        if (!SDL_Vulkan_CreateSurface(window_->getSDLWindow(), instance_, nullptr, &c_surface))
        {
            return AppError::unexpected(
                {"SDL_Vulkan_CreateSurface failed: " + std::string(SDL_GetError()),
                 ErrorKind::SurfaceCreationFailed});
        }

        window_surface_ = vk::SurfaceKHR(c_surface);
        return {};
    }

    bool isDeviceSuitable(VkPhysicalDevice const &physicalDevice)
    {
        // if supports vulkan 1.3
        VkPhysicalDeviceProperties physical_device_properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &physical_device_properties);
        bool supports_vulkan1_3 = physical_device_properties.apiVersion >= VK_API_VERSION_1_3;

        // if supports graphics queue family
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queue_family_count,
                                                 queue_families.data());

        bool supports_graphics = false;
        for (auto const &queue_family : queue_families)
        {
            if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
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
        for (char const *required_device_extension : required_device_extensions_)
        {
            bool found = true;
            for (auto const &available_device_extension : available_device_extensions)
            {
                if (strcmp(available_device_extension.extensionName, required_device_extension) ==
                    0)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
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

        bool supports_required_features = vulkan_11_features.shaderDrawParameters &&
                                          vulkan_13_features.synchronization2 &&
                                          vulkan_13_features.dynamicRendering &&
                                          extended_dynamic_state_features.extendedDynamicState;

        return supports_vulkan1_3 && supports_graphics && supports_all_required_extensions &&
               supports_required_features;
    }

    [[nodiscard]]
    Result<void> pickPhysicalDevice()
    {
        // checking if physical devices meet requirements

        auto physical_devices = instance_.enumeratePhysicalDevices();
        if (physical_devices.result != vk::Result::eSuccess)
            return AppError::unexpected(
                {"Failed to enumerate physical device", physical_devices.result});

        // find if a GPU meets all the requirements
        bool found = false;
        for (auto const &physical_device : physical_devices.value)
        {
            if (isDeviceSuitable(physical_device))
            {
                found            = true;
                physical_device_ = physical_device;
                break;
            }
        }
        if (!found)
        {
            return AppError::unexpected(
                {"Failed to find physical device", ErrorKind::FailedToFindGPU});
        }

        return {};
    }

    [[nodiscard]]
    Result<void> createLogicalDevice()
    {
        auto queue_family_properties = physical_device_.getQueueFamilyProperties();

        // check for support of both graphics and present queue families
        for (uint32_t queue_family_prop_idx = 0;
             queue_family_prop_idx < queue_family_properties.size(); queue_family_prop_idx++)
        {
            vk::Bool32 present_support = VK_FALSE;

            vk::Result result = physical_device_.getSurfaceSupportKHR(
                queue_family_prop_idx, window_surface_, &present_support);
            if (result != vk::Result::eSuccess)
                return AppError::unexpected(
                    {"Failed to get physical device surface support", result});

            if ((queue_family_properties[queue_family_prop_idx].queueFlags &
                 vk::QueueFlagBits::eGraphics) &&
                present_support)
            {
                queue_family_idx_ = queue_family_prop_idx;
                break;
            }
        }

        if (queue_family_idx_ == UINT32_MAX)
        {
            return AppError::unexpected(
                {"Couldnt find suitable queue family", ErrorKind::NoSuitableQueueFamily});
        }

        // getting features
        vk::StructureChain feature_chain = {
            vk::PhysicalDeviceFeatures2{
                .features = {.samplerAnisotropy = vk::True},
            },
            vk::PhysicalDeviceVulkan11Features{
                .shaderDrawParameters = vk::True,
            },
            vk::PhysicalDeviceVulkan13Features{
                .synchronization2 = vk::True,
                .dynamicRendering = vk::True,
            },
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT{
                .extendedDynamicState = vk::True,
            },
        };

        float queue_priority = 0.5f; // priority for scheduling of command buffer
                                     // execution, needed even if there is one queue
        vk::DeviceQueueCreateInfo device_queue_create_info{.queueFamilyIndex = queue_family_idx_,
                                                           .queueCount       = 1,
                                                           .pQueuePriorities = &queue_priority};

        vk::DeviceCreateInfo device_create_info{
            .pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(), // connecting the chain of
                                                                        // features to vulkan
            .queueCreateInfoCount    = 1,
            .pQueueCreateInfos       = &device_queue_create_info,
            .enabledExtensionCount   = static_cast<uint32_t>(required_device_extensions_.size()),
            .ppEnabledExtensionNames = required_device_extensions_.data()};

        vk::Result result =
            physical_device_.createDevice(&device_create_info, nullptr, &logical_device_);
        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected({"Failed to create logical device", result});
        }

        queue_ = logical_device_.getQueue(queue_family_idx_, 0);

        return {};
    }

    vk::SurfaceFormatKHR
    chooseSwapchainSurfaceFormat(std::vector<vk::SurfaceFormatKHR> const &available_formats)
    {
        assert(!available_formats.empty());

        vk::SurfaceFormatKHR surface_format = available_formats[0];

        bool found = false;
        for (auto const &format : available_formats)
        {
            found = format.format == vk::Format::eB8G8R8A8Srgb &&
                    format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
            if (found)
            {
                surface_format = format;
                break;
            }
        }

        return surface_format;
    }

    vk::PresentModeKHR
    chooseSwapchainPresentMode(std::vector<vk::PresentModeKHR> const &available_present_modes)
    {
        // fifo present mode - stores rendered images in a queue, takes an image
        // from the front of the queue to display every time the display
        // refreshes mailbox present mode - like fifo, but when the queue is
        // full it replaces old images with new ones to display images as fast as
        // possible

        bool found_fifo    = false;
        bool found_mailbox = false;
        for (vk::PresentModeKHR present_mode : available_present_modes)
        {
            if (present_mode == vk::PresentModeKHR::eFifo)
            {
                found_fifo = true;
                break;
            }
            if (present_mode == vk::PresentModeKHR::eMailbox)
            {
                found_mailbox = true;
                break;
            }
        }

        assert(found_fifo || found_mailbox);

        // if mailbox present mode is available, use it, otherwise FIFO present
        // mode
        return found_mailbox ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }

    [[nodiscard]]
    Result<vk::Extent2D>
    chooseSwapchainExtent(vk::SurfaceCapabilitiesKHR const &surface_capabilities)
    {
        // extent is the resolution of the images in the swapchain

        if (surface_capabilities.currentExtent.width != UINT32_MAX)
        {
            return surface_capabilities.currentExtent;
        }

        int width, height;
        bool success = SDL_GetWindowSizeInPixels(window_->getSDLWindow(), &width, &height);
        if (!success)
            return AppError::unexpected(
                {"SDL_GetWindowSizeInPixels failed" + std::string(SDL_GetError()),
                 ErrorKind::SDLFailure});

        return vk::Extent2D{std::clamp<uint32_t>(width, surface_capabilities.minImageExtent.width,
                                                 surface_capabilities.maxImageExtent.width),
                            std::clamp<uint32_t>(height, surface_capabilities.minImageExtent.height,
                                                 surface_capabilities.maxImageExtent.height)};
    }

    uint32_t chooseSwapchainMinImageCount(vk::SurfaceCapabilitiesKHR const &surface_capabilities)
    {
        uint32_t min_img_count = std::max(uint32_t(3), surface_capabilities.minImageCount);

        if ((0 < surface_capabilities.maxImageCount) &&
            (surface_capabilities.maxImageCount < min_img_count))
        {
            min_img_count = surface_capabilities.maxImageCount;
        }

        return min_img_count;
    }

    void cleanupSwapchain()
    {
        swapchain_image_views_.clear();
        swapchain_ = nullptr;
    }

    [[nodiscard]]
    Result<void> recreateSwapchain()
    {
        int width    = 0;
        int height   = 0;
        bool success = SDL_GetWindowSizeInPixels(window_->getSDLWindow(), &width, &height);
        if (!success)
            return AppError::unexpected(
                {"SDL_GetWindowSizeInPixels failed" + std::string(SDL_GetError()),
                 ErrorKind::SDLFailure});

        while (width == 0 || height == 0)
        {
            success = SDL_GetWindowSizeInPixels(window_->getSDLWindow(), &width, &height);
            if (!success)
                return AppError::unexpected(
                    {"SDL_GetWindowSizeInPixels failed" + std::string(SDL_GetError()),
                     ErrorKind::SDLFailure});

            success = SDL_WaitEvent(&window_->getCurrentEvent());
            if (!success)
                return AppError::unexpected(
                    {"SDL_WaitEvent failed" + std::string(SDL_GetError()), ErrorKind::SDLFailure});
        }

        vk::Result result = logical_device_.waitIdle();
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Logical device wait idle failed", result});

        swapchain_image_views_.clear();
        cleanupSwapchain();

        auto expected = createSwapchain();
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = createImageViews();
        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> createSwapchain()
    {
        vk::SurfaceCapabilitiesKHR surface_capabilities;
        vk::Result result =
            physical_device_.getSurfaceCapabilitiesKHR(window_surface_, &surface_capabilities);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected(
                {"Failed to get physical device surface capabilites", result});

        auto extent = chooseSwapchainExtent(surface_capabilities);
        if (!extent)
            return AppError::unexpected(extent.error());
        swapchain_extent_ = extent.value();

        uint32_t min_img_count = chooseSwapchainMinImageCount(surface_capabilities);

        auto available_formats = physical_device_.getSurfaceFormatsKHR();
        if (available_formats.result != vk::Result::eSuccess)
            return AppError::unexpected(
                {"Failed to get physical device surface formats", available_formats.result});

        swapchain_surface_format_ = chooseSwapchainSurfaceFormat(available_formats.value);

        auto available_present_modes = physical_device_.getSurfacePresentModesKHR();
        if (available_present_modes.result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to get physical device surface present modes",
                                         available_present_modes.result});

        vk::PresentModeKHR present_mode = chooseSwapchainPresentMode(available_present_modes.value);

        vk::SwapchainCreateInfoKHR swapchain_create_info{
            .surface          = window_surface_,
            .minImageCount    = min_img_count,
            .imageFormat      = swapchain_surface_format_.format,
            .imageColorSpace  = swapchain_surface_format_.colorSpace,
            .imageExtent      = swapchain_extent_,
            .imageArrayLayers = 1,
            .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform     = surface_capabilities.currentTransform,
            .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode      = present_mode,
            .clipped          = true,
            .oldSwapchain     = swapchain_,
        };

        result = logical_device_.createSwapchainKHR(&swapchain_create_info, nullptr, &swapchain_);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to create swapchain", result});

        auto swapchain_imgs = logical_device_.getSwapchainImagesKHR(swapchain_);
        if (swapchain_imgs.result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to get swapchain images", swapchain_imgs.result});

        swapchain_images_.assign(swapchain_imgs.value.begin(), swapchain_imgs.value.end());

        return {};
    }

    [[nodiscard]]
    Result<vk::ImageView> createImageView(vk::Image &image, vk::Format format)
    {
        vk::ImageViewCreateInfo view_info{
            .image            = image,
            .viewType         = vk::ImageViewType::e2D,
            .format           = format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };

        vk::ImageView image_view;
        vk::Result result = logical_device_.createImageView(&view_info, nullptr, &image_view);
        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected({"Failed to create image view", result});
        }

        return image_view;
    }

    [[nodiscard]]
    Result<void> createImageViews()
    {
        assert(swapchain_image_views_.empty());

        swapchain_image_views_.resize(swapchain_images_.size());
        for (uint32_t i = 0; i < swapchain_images_.size(); i++)
        {
            auto image_view =
                createImageView(swapchain_images_[i], swapchain_surface_format_.format);

            if (!image_view)
            {
                return AppError::unexpected(image_view.error());
            }

            swapchain_image_views_[i] = image_view.value();
        }

        return {};
    }

    [[nodiscard]]
    Result<void> createDescriptorSetLayout()
    {
        std::array bindings = {
            vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                           vk::ShaderStageFlagBits::eVertex, nullptr),
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1,
                                           vk::ShaderStageFlagBits::eFragment, nullptr),
        };

        vk::DescriptorSetLayoutCreateInfo layout_info{
            .bindingCount = bindings.size(),
            .pBindings    = bindings.data(),
        };

        vk::Result result = logical_device_.createDescriptorSetLayout(&layout_info, nullptr,
                                                                      &descriptor_set_layout_);

        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected({"Failed to create descriptor set layout", result});
        }

        return {};
    }

    [[nodiscard]]
    Result<void> createGraphicsPipeline()
    {
        /* SHADER STAGE SETUP */

        auto shader_code = readFile(appPath() / "shaders/slang.spv");
        if (!shader_code)
        {
            return AppError::unexpected(shader_code.error());
        }

        auto shader_module = createShaderModule(shader_code.value());
        if (!shader_module)
            return AppError::unexpected(shader_module.error());

        vk::PipelineShaderStageCreateInfo vert_shader_stage_info{
            .stage               = vk::ShaderStageFlagBits::eVertex,
            .module              = shader_module.value(),
            .pName               = "vertMain", // the entrypoint in the slang code
            .pSpecializationInfo = nullptr     // used to set constants in shader per-pipeline
        };

        vk::PipelineShaderStageCreateInfo frag_shader_stage_info{
            .stage  = vk::ShaderStageFlagBits::eFragment,
            .module = shader_module.value(),
            .pName  = "fragMain"};

        std::array<vk::PipelineShaderStageCreateInfo, 2> shader_stages = {vert_shader_stage_info,
                                                                          frag_shader_stage_info};

        /* INPUT STAGE SETUP */

        auto bindingDescription    = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertex_input_info{
            .vertexBindingDescriptionCount   = 1,
            .pVertexBindingDescriptions      = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions    = attributeDescriptions.data()};

        vk::PipelineInputAssemblyStateCreateInfo input_assembly{
            .topology = vk::PrimitiveTopology::eTriangleList};

        vk::PipelineViewportStateCreateInfo viewport_state{.viewportCount = 1, .scissorCount = 1};

        std::array dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

        vk::PipelineDynamicStateCreateInfo dynamic_state{
            .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
            .pDynamicStates    = dynamic_states.data()};

        /* RASTERIZATION STAGE SETUP */

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable        = vk::False, // if true, fragments past the near or far plane
                                                  // will be clamped rather than discarded
            .rasterizerDiscardEnable = vk::False, // if true, skips rasterizer stage
            .polygonMode             = vk::PolygonMode::eFill,
            .cullMode                = vk::CullModeFlagBits::eBack,
            .frontFace               = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable         = vk::False, // if true, rasterizer can make
                                                  // adjustments to depth values
            .lineWidth               = 1.0f};

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

        /* COLOR BLENDING STAGE SETUP */

        // linearly interpolated blending
        vk::PipelineColorBlendAttachmentState color_blend_attachment{
            .blendEnable         = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp        = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp        = vk::BlendOp::eAdd,
            .colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                   vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

        vk::PipelineColorBlendStateCreateInfo color_blending{.logicOpEnable   = vk::False,
                                                             .logicOp         = vk::LogicOp::eCopy,
                                                             .attachmentCount = 1,
                                                             .pAttachments =
                                                                 &color_blend_attachment};

        /* PIPELINE SETUP */

        vk::PipelineLayoutCreateInfo pipelione_layout_info{.setLayoutCount = 1,
                                                           .pSetLayouts = &descriptor_set_layout_,
                                                           .pushConstantRangeCount = 0};

        vk::Result result = logical_device_.createPipelineLayout(&pipelione_layout_info, nullptr,
                                                                 &pipeline_layout_);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to create pipeline layout", result});

        vk::PipelineRenderingCreateInfo pipeline_rendering_create_info{
            .colorAttachmentCount    = 1,
            .pColorAttachmentFormats = &swapchain_surface_format_.format};

        vk::GraphicsPipelineCreateInfo graphics_pipeline_create_info{
            .pNext               = &pipeline_rendering_create_info,
            .stageCount          = 2,
            .pStages             = shader_stages.data(),
            .pVertexInputState   = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState      = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState   = &multisampling,
            .pColorBlendState    = &color_blending,
            .pDynamicState       = &dynamic_state,
            .layout              = pipeline_layout_,
            .renderPass          = VK_NULL_HANDLE // using dynamic rendering
        };

        result = logical_device_.createGraphicsPipelines(
            VK_NULL_HANDLE, 1, &graphics_pipeline_create_info, nullptr, &graphics_pipeline_);

        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected({"Failed to create graphics pipeline", result});
        }

        return {};
    }

    [[nodiscard]]
    Result<vk::ShaderModule> createShaderModule(const std::vector<char> &code) const
    {
        VkShaderModuleCreateInfo createInfo{.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                            .codeSize = code.size() * sizeof(char),
                                            .pCode =
                                                reinterpret_cast<const uint32_t *>(code.data())};

        VkShaderModule shaderModule;
        VkResult result = vkCreateShaderModule(logicalDevice, &createInfo, nullptr, &shaderModule);

        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected(result);
        }

        return shaderModule;
    }

    [[nodiscard]]
    Result<void> createCommandPool()
    {
        VkCommandPoolCreateInfo poolInfo{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_idx,
        };

        VkResult result = vkCreateCommandPool(logicalDevice, &poolInfo, nullptr, &commandPool);

        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected(result);
        }

        return {};
    }

    [[nodiscard]]
    Result<void> createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
                             VkImageUsageFlags usage, VkMemoryPropertyFlags properties,
                             VkImage &image, VkDeviceMemory &imageMemory)
    {
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
        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected(result);
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(logicalDevice, image, &memRequirements);

        auto memory_type_idx = findMemoryType(memRequirements.memoryTypeBits, properties);
        if (!memory_type_idx)
            return AppError::unexpected(memory_type_idx.error());

        VkMemoryAllocateInfo allocInfo{
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize  = memRequirements.size,
            .memoryTypeIndex = memory_type_idx.value(),
        };

        result = vkAllocateMemory(logicalDevice, &allocInfo, nullptr, &imageMemory);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        result = vkBindImageMemory(logicalDevice, image, imageMemory, 0);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        return {};
    }

    [[nodiscard]]
    Result<void> transitionImageLayout(const VkImage &image, VkImageLayout oldLayout,
                                       VkImageLayout newLayout)
    {
        auto commandBuffer = beginOneTimeCommandBuffer();
        if (!commandBuffer)
            return AppError::unexpected(commandBuffer.error());

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
            newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                 newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else
        {
            return AppError::unexpected(ErrorKind::TransitionNotSupported);
        }

        vkCmdPipelineBarrier(commandBuffer.value(), srcStage, dstStage, {}, {}, nullptr, 0, nullptr,
                             1, &barrier);

        auto expected = endOneTimeCommandBuffer(commandBuffer.value());

        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> createTextureImage()
    {
        int texWidth, texHeight, texChannels;
        stbi_uc *pixels = stbi_load((appPath() / "textures/dirt.png").c_str(), &texWidth,
                                    &texHeight, &texChannels, STBI_rgb_alpha);

        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels)
        {
            return AppError::unexpected(ErrorKind::FailedToLoadImage);
        }

        VkBuffer stagingBuffer             = nullptr;
        VkDeviceMemory stagingBufferMemory = nullptr;

        auto expected =
            createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stagingBuffer, stagingBufferMemory);

        if (!expected)
            return AppError::unexpected(expected.error());

        void *data;
        VkResult result = vkMapMemory(logicalDevice, stagingBufferMemory, 0, imageSize, 0, &data);
        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected(result);
        }

        memcpy(data, pixels, imageSize);

        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        stbi_image_free(pixels);

        expected =
            createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
                        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, textureImage, textureImageMemory);

        if (!expected)
        {
            return AppError::unexpected(expected.error());
        }

        expected = transitionImageLayout(textureImage, VK_IMAGE_LAYOUT_UNDEFINED,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = copyBufferToImage(stagingBuffer, textureImage, texWidth, texHeight);
        if (!expected)
            return AppError::unexpected(expected.error());

        expected = transitionImageLayout(textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> createTextureImageView()
    {
        auto expected = createImageView(textureImage, VK_FORMAT_R8G8B8A8_SRGB);
        if (!expected)
        {
            return AppError::unexpected(expected.error());
        }

        textureImageView = expected.value();

        return {};
    }

    [[nodiscard]]
    Result<void> createTextureSampler()
    {
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

        VkResult result = vkCreateSampler(logicalDevice, &samplerInfo, nullptr, &textureSampler);
        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected(result);
        }

        return {};
    }

    [[nodiscard]]
    Result<void> copyBufferToImage(const VkBuffer &buffer, VkImage &image, uint32_t width,
                                   uint32_t height)
    {
        auto commandBuffer = beginOneTimeCommandBuffer();
        if (!commandBuffer)
            return AppError::unexpected(commandBuffer.error());

        VkBufferImageCopy region{
            .bufferOffset      = 0,
            .bufferRowLength   = 0,
            .bufferImageHeight = 0,
            .imageSubresource  = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageOffset       = {0, 0, 0},
            .imageExtent       = {width, height, 1},
        };

        vkCmdCopyBufferToImage(commandBuffer.value(), buffer, image,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        auto expected = endOneTimeCommandBuffer(commandBuffer.value());
        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<uint32_t> findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
    {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        return AppError::unexpected(ErrorKind::NoSuitableMemoryType);
    }

    [[nodiscard]]
    Result<void> createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags properties, VkBuffer &buffer,
                              VkDeviceMemory &bufferMemory)
    {
        VkBufferCreateInfo bufferInfo{
            .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size        = size,
            .usage       = usage,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };

        VkResult result = vkCreateBuffer(logicalDevice, &bufferInfo, nullptr, &buffer);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(logicalDevice, buffer, &memRequirements);

        auto memory_type_idx = findMemoryType(memRequirements.memoryTypeBits, properties);
        if (!memory_type_idx)
            return AppError::unexpected(memory_type_idx.error());

        VkMemoryAllocateInfo memoryAllocateInfo{
            .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize  = memRequirements.size,
            .memoryTypeIndex = memory_type_idx.value(),
        };

        result = vkAllocateMemory(logicalDevice, &memoryAllocateInfo, nullptr, &bufferMemory);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        result = vkBindBufferMemory(logicalDevice, buffer, bufferMemory, 0);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        return {};
    }

    [[nodiscard]]
    Result<vk::CommandBuffer> beginOneTimeCommandBuffer()
    {
        VkCommandBufferAllocateInfo allocInfo{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = commandPool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        VkCommandBuffer commandBuffer;
        VkResult result = vkAllocateCommandBuffers(logicalDevice, &allocInfo, &commandBuffer);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };

        result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        return commandBuffer;
    }

    [[nodiscard]]
    Result<void> endOneTimeCommandBuffer(VkCommandBuffer &commandBuffer)
    {
        VkResult result = vkEndCommandBuffer(commandBuffer);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        VkSubmitInfo submitInfo{
            .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers    = &commandBuffer,
        };

        result = vkQueueSubmit(queue, 1, &submitInfo, nullptr);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        result = vkQueueWaitIdle(queue);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        return {};
    }

    [[nodiscard]]
    Result<void> copyBuffer(VkBuffer &srcBuffer, VkBuffer &dstBuffer, VkDeviceSize size)
    {
        auto commandCopyBuffer = beginOneTimeCommandBuffer();
        if (!commandCopyBuffer)
            return AppError::unexpected(commandCopyBuffer.error());

        VkBufferCopy buffer_copy_region = {0, 0, size};
        vkCmdCopyBuffer(commandCopyBuffer.value(), srcBuffer, dstBuffer, 1, &buffer_copy_region);

        auto expected = endOneTimeCommandBuffer(commandCopyBuffer.value());
        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> createVertexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        // staging buffer, CPU vertex data will be put here and then transferred
        // to the GPU local vertex buffer
        VkBuffer stagingBuffer             = nullptr;
        VkDeviceMemory stagingBufferMemory = nullptr;

        auto expected =
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stagingBuffer, stagingBufferMemory);

        if (!expected)
            return AppError::unexpected(expected.error());

        void *dataStaging;
        VkResult result =
            vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &dataStaging);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        memcpy(dataStaging, vertices.data(), bufferSize);

        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // vertex buffer
        expected = createBuffer(
            bufferSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer, vertexBufferMemory);

        if (!expected)
            return AppError::unexpected(expected.error());

        expected = copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> createIndexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        // staging buffer, CPU vertex data will be put here and then transferred
        // to the GPU local vertex buffer
        VkBuffer stagingBuffer             = nullptr;
        VkDeviceMemory stagingBufferMemory = nullptr;

        auto expected =
            createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         stagingBuffer, stagingBufferMemory);

        if (!expected)
            return AppError::unexpected(expected.error());

        void *dataStaging;
        VkResult result =
            vkMapMemory(logicalDevice, stagingBufferMemory, 0, bufferSize, 0, &dataStaging);

        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected(result);
        }

        memcpy(dataStaging, indices.data(), bufferSize);

        vkUnmapMemory(logicalDevice, stagingBufferMemory);

        // vertex buffer
        expected = createBuffer(
            bufferSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);

        if (!expected)
            return AppError::unexpected(expected.error());

        expected = copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> createUniformBuffers()
    {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        for (int i = 0; i < maxFramesInFlight; i++)
        {
            VkDeviceSize bufferSize  = sizeof(UniformBufferObject);
            VkBuffer buffer          = nullptr;
            VkDeviceMemory bufferMem = nullptr;

            auto expected = createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                         buffer, bufferMem);

            if (!expected)
                return AppError::unexpected(expected.error());

            vkMapMemory(logicalDevice, uniformBuffersMemory[i], 0, bufferSize, 0, nullptr);

            // why are there std::move here? Can it just be removed?
            uniformBuffers.emplace_back(std::move(buffer));
            uniformBuffersMemory.emplace_back(std::move(bufferMem));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory[i]);
        }

        return {};
    }

    [[nodiscard]]
    Result<void> createDescriptorPool()
    {
        std::array poolSize{
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, maxFramesInFlight),
            VkDescriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maxFramesInFlight),
        };

        VkDescriptorPoolCreateInfo poolInfo{
            .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets       = maxFramesInFlight,
            .poolSizeCount = poolSize.size(),
            .pPoolSizes    = poolSize.data(),
        };

        VkResult result =
            vkCreateDescriptorPool(logicalDevice, &poolInfo, nullptr, &descriptorPool);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        return {};
    }

    [[nodiscard]]
    Result<void> createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(maxFramesInFlight, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{
            .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool     = descriptorPool,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts        = layouts.data(),
        };

        descriptorSets.clear();

        VkResult result =
            vkAllocateDescriptorSets(logicalDevice, &allocInfo, descriptorSets.data());

        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        for (int i = 0; i < maxFramesInFlight; i++)
        {
            VkDescriptorBufferInfo bufferInfo{
                .buffer = uniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject)};

            VkDescriptorImageInfo imageInfo{.sampler   = textureSampler,
                                            .imageView = textureImageView,
                                            .imageLayout =
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};

            std::array descriptorWrites{
                VkWriteDescriptorSet{
                    .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet          = descriptorSets[i],
                    .dstBinding      = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .pBufferInfo     = &bufferInfo,
                },
                VkWriteDescriptorSet{
                    .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet          = descriptorSets[i],
                    .dstBinding      = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo      = &imageInfo,
                }};

            vkUpdateDescriptorSets(logicalDevice, descriptorWrites.size(), descriptorWrites.data(),
                                   0, nullptr);
        }

        return {};
    }

    [[nodiscard]]
    Result<void> createCommandBuffers()
    {
        VkCommandBufferAllocateInfo allocInfo{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = commandPool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = maxFramesInFlight,
        };

        VkResult result =
            vkAllocateCommandBuffers(logicalDevice, &allocInfo, commandBuffers.data());

        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        return {};
    }

    [[nodiscard]]
    Result<void> recordCommandBuffer(uint32_t image_idx)
    {
        VkCommandBuffer &commandBuffer = commandBuffers[frame_idx];

        VkResult result = vkBeginCommandBuffer(commandBuffer, {});
        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        // changing image layout from undefined to color attachment optimal
        transitionImageLayout(
            image_idx, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, {},
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);

        VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0}}};

        VkRenderingAttachmentInfo attachmentInfo = {
            .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView   = swapchainImageViews[image_idx], // rendering to this
                                                           // image in the swapchain
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue  = clearColor};

        VkRenderingInfo renderingInfo = {
            .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea           = {.offset = {0, 0}, .extent = swapchainExtent},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &attachmentInfo,
        };

        vkCmdBeginRendering(commandBuffer, &renderingInfo);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport = {0.0f,
                               0.0f,
                               static_cast<float>(swapchainExtent.width),
                               static_cast<float>(swapchainExtent.height),
                               0.0f,
                               1.0f};

        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor = {VkOffset2D(0, 0), swapchainExtent};

        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, nullptr);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT16);

        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
                                1, &descriptorSets[frame_idx], 0, nullptr);

        vkCmdDrawIndexed(commandBuffer, static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        vkCmdEndRendering(commandBuffer);

        transitionImageLayout(image_idx, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                              VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                              VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, {},
                              VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                              VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

        result = vkEndCommandBuffer(commandBuffer);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected(result);

        return {};
    }

    void transitionImageLayout(uint32_t image_idx, VkImageLayout oldLayout, VkImageLayout newLayout,
                               VkAccessFlags2 oldAccessMask, VkAccessFlags2 newAccessMask,
                               VkPipelineStageFlags2 oldStageMask,
                               VkPipelineStageFlags2 newStageMask)
    {
        VkImageMemoryBarrier2 barrier = {
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = oldStageMask,
            .srcAccessMask       = oldAccessMask,
            .dstStageMask        = newStageMask,
            .dstAccessMask       = newAccessMask,
            .oldLayout           = oldLayout,
            .newLayout           = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = swapchainImages[image_idx],
            .subresourceRange    = {.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                                    .baseMipLevel   = 0,
                                    .levelCount     = 1,
                                    .baseArrayLayer = 0,
                                    .layerCount     = 1},
        };

        VkDependencyInfo dependencyInfo = {
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .dependencyFlags         = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrier,
        };

        vkCmdPipelineBarrier2(commandBuffers[frame_idx], &dependencyInfo);
    }

    [[nodiscard]]
    Result<void> createSyncObjects()
    {
        assert(presentCompleteSphrs.empty() && renderFinishedSphrs.empty() && drawFences.empty());

        for (int i = 0; i < swapchainImages.size(); i++)
        {
            VkResult result =
                vkCreateSemaphore(logicalDevice, nullptr, nullptr, &renderFinishedSphrs[i]);

            if (result != vk::Result::eSuccess)
                return AppError::unexpected(result);
        }

        for (int i = 0; i < maxFramesInFlight; i++)
        {
            VkResult result =
                vkCreateSemaphore(logicalDevice, nullptr, nullptr, &presentCompleteSphrs[i]);

            if (result != vk::Result::eSuccess)
                return AppError::unexpected(result);

            VkFenceCreateInfo fence_create_info = {.flags = VK_FENCE_CREATE_SIGNALED_BIT};

            result = vkCreateFence(logicalDevice, &fence_create_info, nullptr, &drawFences[i]);

            if (result != vk::Result::eSuccess)
                return AppError::unexpected(result);
        }

        return {};
    }
};
