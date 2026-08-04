#pragma once

#define VK_NO_PROTOTYPES
#include <volk.h>

#define VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#include <vk_mem_alloc.h>

#include <SDL3/SDL_vulkan.h>

#include "Window/window.hpp"
#include "magic_enum.hpp"
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
        return {.binding   = 0,
                .stride    = sizeof(Vertex),
                .inputRate = vk::VertexInputRate::eVertex};
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
        : kind_(ErrorKind::VulkanFailure), message_(std::move(message)),
          vk_result_(result), location_(location)
    {
    }

    static std::unexpected<AppError> unexpected(AppError error)
    {
        return std::unexpected(std::move(error));
    }

    void log(std::string message)
    {
        spdlog::error(message);
        spdlog::error("File: {} | Line: {} | Func: {}", location_.file_name(),
                      location_.line(), location_.function_name());

        if (vk_result_.has_value())
            spdlog::error("Kind: {} | vk::Result: {}", magic_enum::enum_name(kind_),
                          vk::to_string(vk_result_.value()));
        else
            spdlog::error("Kind: {}", magic_enum::enum_name(kind_));

        spdlog::error("Message: {}\n", message_);
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

    std::array<const char *, 1> required_device_extensions_ = {
        vk::KHRSwapchainExtensionName};

    vk::Instance instance_              = nullptr;
    vk::PhysicalDevice physical_device_ = nullptr; // Physical device represents the GPU
    vk::Device logical_device_ =
        nullptr; // Logical Device is the interface for the physical device
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
    std::vector<vk::DescriptorSet> descriptor_sets_;

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
        VkResult volk_result = volkInitialize();
        if (volk_result != VK_SUCCESS)
            return AppError::unexpected(
                {"Failed to initialize volk", vk::Result(volk_result)});

        VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

        auto expected = createInstance();
        if (!expected)
            return AppError::unexpected(expected.error());

        volkLoadInstance(static_cast<VkInstance>(instance_));
        VULKAN_HPP_DEFAULT_DISPATCHER.init(instance_);

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

        volkLoadDevice(static_cast<VkDevice>(logical_device_));
        VULKAN_HPP_DEFAULT_DISPATCHER.init(logical_device_);

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
        vk::Result result = logical_device_.waitForFences(1, &draw_fences_[frame_idx_],
                                                          VK_TRUE, UINT64_MAX);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to wait for fences", result});

        uint32_t image_idx;
        result = logical_device_.acquireNextImageKHR(swapchain_, UINT64_MAX,
                                                     present_complete_sphrs_[frame_idx_],
                                                     nullptr, &image_idx);

        if (result == vk::Result::eErrorOutOfDateKHR)
        {
            auto expected = recreateSwapchain();
            if (!expected)
                return AppError::unexpected(expected.error());

            return {}; // skip to next frame
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
            .waitSemaphoreCount = 1,
            .pWaitSemaphores =
                &present_complete_sphrs_[frame_idx_], // semaphores to wait for
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
        if (result == vk::Result::eSuboptimalKHR ||
            result == vk::Result::eErrorOutOfDateKHR)
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
        static std::chrono::time_point start_time =
            std::chrono::high_resolution_clock::now();
        std::chrono::time_point current_time = std::chrono::high_resolution_clock::now();

        float delta_time = std::chrono::duration<float, std::chrono::seconds::period>(
                               current_time - start_time)
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
            return AppError::unexpected(
                {"Failed to open file", ErrorKind::FailedToOpenFile});
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
        vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
        vk::DebugUtilsMessageTypeFlagsEXT type,
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

        vk::DebugUtilsMessengerCreateInfoEXT create_info{
            .messageSeverity = severity_flags,
            .messageType     = message_type_flags,
            .pfnUserCallback = debugCallback};

        auto result = instance_.createDebugUtilsMessengerEXT(create_info);

        if (result.result != vk::Result::eSuccess)
            return AppError::unexpected(
                {"Failed to create debug utils messenger", result.result});

        debug_messenger_ = result.value;

        return {};
    }

    std::vector<const char *> getRequiredInstanceExtensions()
    {
        uint32_t sdlExtensionCount = 0;
        char const *const *sdlExtensions =
            SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);

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
        vk::ApplicationInfo constexpr app_info{.pApplicationName = "Learn Vulkan",
                                               .applicationVersion =
                                                   VK_MAKE_VERSION(1, 0, 0),
                                               .pEngineName   = "No Engine",
                                               .engineVersion = VK_MAKE_VERSION(1, 0, 0),
                                               .apiVersion    = VK_API_VERSION_1_3};

        // VALIDATION LAYERS
        std::vector<char const *> required_layers;
        if (enable_validation_layers)
        {
            required_layers.assign(validation_layers.begin(), validation_layers.end());
        }

        // check if validation layers are available
        auto layer_properties = vk::enumerateInstanceLayerProperties();
        if (layer_properties.result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to enumerate instance layer properties",
                                         layer_properties.result});

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
                return AppError::unexpected({"Validation layer not supported",
                                             ErrorKind::ValidationLayerNotSupported});
            }
        }

        // EXTENSIONS
        std::vector<char const *> required_extensions = getRequiredInstanceExtensions();

        auto extension_properties = vk::enumerateInstanceExtensionProperties();
        if (extension_properties.result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to enumerate extension properties",
                                         extension_properties.result});

        for (char const *required_extension : required_extensions)
        {
            bool found = false;
            for (auto const &extension : extension_properties.value)
            {
                if (strcmp(extension.extensionName, required_extension) == 0)
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

        if (!SDL_Vulkan_CreateSurface(window_->getSDLWindow(), instance_, nullptr,
                                      &c_surface))
        {
            return AppError::unexpected(
                {"SDL_Vulkan_CreateSurface failed: " + std::string(SDL_GetError()),
                 ErrorKind::SurfaceCreationFailed});
        }

        window_surface_ = vk::SurfaceKHR(c_surface);
        return {};
    }

    [[nodiscard]]
    Result<bool> isDeviceSuitable(vk::PhysicalDevice const &physical_device)
    {
        // if supports vulkan 1.3
        vk::PhysicalDeviceProperties physical_device_properties;
        physical_device.getProperties(&physical_device_properties);
        bool supports_vulkan1_3 =
            physical_device_properties.apiVersion >= VK_API_VERSION_1_3;

        // if supports graphics queue family
        auto queue_families = physical_device.getQueueFamilyProperties();

        bool supports_graphics = false;
        for (auto const &queue_family : queue_families)
        {
            if (queue_family.queueFlags & vk::QueueFlagBits::eGraphics)
            {
                supports_graphics = true;
                break;
            }
        }

        // if supports specific extensions
        auto available_device_extensions =
            physical_device.enumerateDeviceExtensionProperties();
        if (available_device_extensions.result != vk::Result::eSuccess)
            return AppError::unexpected(
                {"Failed to enumerate device extension properties",
                 available_device_extensions.result});

        // if any of the required device extensions aren't available -> false
        bool supports_all_required_extensions = true;
        for (char const *required_device_extension : required_device_extensions_)
        {
            bool found = false;
            for (auto const &available_device_extension :
                 available_device_extensions.value)
            {
                if (strcmp(available_device_extension.extensionName,
                           required_device_extension) == 0)
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
        auto features = physical_device.template getFeatures2<
            vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

        bool supports_required_features =
            features.template get<vk::PhysicalDeviceVulkan11Features>()
                .shaderDrawParameters &&
            features.template get<vk::PhysicalDeviceVulkan13Features>()
                .synchronization2 &&
            features.template get<vk::PhysicalDeviceVulkan13Features>()
                .dynamicRendering &&
            features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
                .extendedDynamicState;

        return supports_vulkan1_3 && supports_graphics &&
               supports_all_required_extensions && supports_required_features;
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
            auto suitable = isDeviceSuitable(physical_device);
            if (!suitable)
                return AppError::unexpected(suitable.error());

            if (suitable.value())
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
             queue_family_prop_idx < queue_family_properties.size();
             queue_family_prop_idx++)
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
        vk::DeviceQueueCreateInfo device_queue_create_info{
            .queueFamilyIndex = queue_family_idx_,
            .queueCount       = 1,
            .pQueuePriorities = &queue_priority};

        vk::DeviceCreateInfo device_create_info{
            .pNext = &feature_chain
                          .get<vk::PhysicalDeviceFeatures2>(), // connecting the chain of
                                                               // features to vulkan
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos    = &device_queue_create_info,
            .enabledExtensionCount =
                static_cast<uint32_t>(required_device_extensions_.size()),
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

    vk::SurfaceFormatKHR chooseSwapchainSurfaceFormat(
        std::vector<vk::SurfaceFormatKHR> const &available_formats)
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

    vk::PresentModeKHR chooseSwapchainPresentMode(
        std::vector<vk::PresentModeKHR> const &available_present_modes)
    {
        // fifo present mode - stores rendered images in a queue, takes an image
        // from the front of the queue to display every time the display
        // refreshes mailbox present mode - like fifo, but when the queue is
        // full it replaces old images with new ones to display images as fast as
        // possible

        bool found_mailbox = false;
        for (vk::PresentModeKHR present_mode : available_present_modes)
        {
            if (present_mode == vk::PresentModeKHR::eMailbox)
            {
                found_mailbox = true;
                break;
            }
        }

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
        bool success =
            SDL_GetWindowSizeInPixels(window_->getSDLWindow(), &width, &height);
        if (!success)
            return AppError::unexpected(
                {"SDL_GetWindowSizeInPixels failed" + std::string(SDL_GetError()),
                 ErrorKind::SDLFailure});

        return vk::Extent2D{
            std::clamp<uint32_t>(width, surface_capabilities.minImageExtent.width,
                                 surface_capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, surface_capabilities.minImageExtent.height,
                                 surface_capabilities.maxImageExtent.height)};
    }

    uint32_t
    chooseSwapchainMinImageCount(vk::SurfaceCapabilitiesKHR const &surface_capabilities)
    {
        uint32_t min_img_count =
            std::max(uint32_t(3), surface_capabilities.minImageCount);

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
        int width  = 0;
        int height = 0;
        bool success =
            SDL_GetWindowSizeInPixels(window_->getSDLWindow(), &width, &height);
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
                    {"SDL_WaitEvent failed" + std::string(SDL_GetError()),
                     ErrorKind::SDLFailure});
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
        vk::Result result = physical_device_.getSurfaceCapabilitiesKHR(
            window_surface_, &surface_capabilities);
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
            return AppError::unexpected({"Failed to get physical device surface formats",
                                         available_formats.result});

        swapchain_surface_format_ = chooseSwapchainSurfaceFormat(available_formats.value);

        auto available_present_modes = physical_device_.getSurfacePresentModesKHR();
        if (available_present_modes.result != vk::Result::eSuccess)
            return AppError::unexpected(
                {"Failed to get physical device surface present modes",
                 available_present_modes.result});

        vk::PresentModeKHR present_mode =
            chooseSwapchainPresentMode(available_present_modes.value);

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

        result = logical_device_.createSwapchainKHR(&swapchain_create_info, nullptr,
                                                    &swapchain_);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to create swapchain", result});

        auto swapchain_imgs = logical_device_.getSwapchainImagesKHR(swapchain_);
        if (swapchain_imgs.result != vk::Result::eSuccess)
            return AppError::unexpected(
                {"Failed to get swapchain images", swapchain_imgs.result});

        swapchain_images_.assign(swapchain_imgs.value.begin(),
                                 swapchain_imgs.value.end());

        return {};
    }

    [[nodiscard]]
    Result<vk::ImageView> createImageView(vk::Image const &image, vk::Format format)
    {
        vk::ImageViewCreateInfo view_info{
            .image            = image,
            .viewType         = vk::ImageViewType::e2D,
            .format           = format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };

        vk::ImageView image_view;
        vk::Result result =
            logical_device_.createImageView(&view_info, nullptr, &image_view);
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
        for (size_t i = 0; i < swapchain_images_.size(); i++)
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
            vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler,
                                           1, vk::ShaderStageFlagBits::eFragment,
                                           nullptr),
        };

        vk::DescriptorSetLayoutCreateInfo layout_info{
            .bindingCount = static_cast<uint32_t>(bindings.size()),
            .pBindings    = bindings.data(),
        };

        vk::Result result = logical_device_.createDescriptorSetLayout(
            &layout_info, nullptr, &descriptor_set_layout_);

        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected(
                {"Failed to create descriptor set layout", result});
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
            .pSpecializationInfo = nullptr // used to set constants in shader per-pipeline
        };

        vk::PipelineShaderStageCreateInfo frag_shader_stage_info{
            .stage  = vk::ShaderStageFlagBits::eFragment,
            .module = shader_module.value(),
            .pName  = "fragMain"};

        std::array<vk::PipelineShaderStageCreateInfo, 2> shader_stages = {
            vert_shader_stage_info, frag_shader_stage_info};

        /* INPUT STAGE SETUP */

        auto bindingDescription    = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertex_input_info{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions    = &bindingDescription,
            .vertexAttributeDescriptionCount =
                static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions = attributeDescriptions.data()};

        vk::PipelineInputAssemblyStateCreateInfo input_assembly{
            .topology = vk::PrimitiveTopology::eTriangleList};

        vk::PipelineViewportStateCreateInfo viewport_state{.viewportCount = 1,
                                                           .scissorCount  = 1};

        std::array dynamic_states = {vk::DynamicState::eViewport,
                                     vk::DynamicState::eScissor};

        vk::PipelineDynamicStateCreateInfo dynamic_state{
            .dynamicStateCount = static_cast<uint32_t>(dynamic_states.size()),
            .pDynamicStates    = dynamic_states.data()};

        /* RASTERIZATION STAGE SETUP */

        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable = vk::False, // if true, fragments past the near or far
                                           // plane will be clamped rather than discarded
            .rasterizerDiscardEnable = vk::False, // if true, skips rasterizer stage
            .polygonMode             = vk::PolygonMode::eFill,
            .cullMode                = vk::CullModeFlagBits::eBack,
            .frontFace               = vk::FrontFace::eCounterClockwise,
            .depthBiasEnable         = vk::False, // if true, rasterizer can make
                                                  // adjustments to depth values
            .lineWidth               = 1.0f};

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable  = vk::False};

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
            .colorWriteMask =
                vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

        vk::PipelineColorBlendStateCreateInfo color_blending{
            .logicOpEnable   = vk::False,
            .logicOp         = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments    = &color_blend_attachment};

        /* PIPELINE SETUP */

        vk::PipelineLayoutCreateInfo pipelione_layout_info{.setLayoutCount = 1,
                                                           .pSetLayouts =
                                                               &descriptor_set_layout_,
                                                           .pushConstantRangeCount = 0};

        vk::Result result = logical_device_.createPipelineLayout(
            &pipelione_layout_info, nullptr, &pipeline_layout_);

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

        result = logical_device_.createGraphicsPipelines(VK_NULL_HANDLE, 1,
                                                         &graphics_pipeline_create_info,
                                                         nullptr, &graphics_pipeline_);

        if (result != vk::Result::eSuccess)
        {
            return AppError::unexpected({"Failed to create graphics pipeline", result});
        }

        return {};
    }

    [[nodiscard]]
    Result<vk::ShaderModule> createShaderModule(const std::vector<char> &code) const
    {
        vk::ShaderModuleCreateInfo createInfo{
            .codeSize = code.size() * sizeof(char),
            .pCode    = reinterpret_cast<const uint32_t *>(code.data())};

        vk::ShaderModule shaderModule;
        vk::Result result =
            logical_device_.createShaderModule(&createInfo, nullptr, &shaderModule);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to create shader module", result});

        return shaderModule;
    }

    [[nodiscard]]
    Result<void> createCommandPool()
    {
        vk::CommandPoolCreateInfo pool_info{
            .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queue_family_idx_,
        };

        vk::Result result =
            logical_device_.createCommandPool(&pool_info, nullptr, &command_pool_);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to create command pool", result});

        return {};
    }

    [[nodiscard]]
    Result<void> createImage(uint32_t width, uint32_t height, vk::Format format,
                             vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                             vk::MemoryPropertyFlags properties, vk::Image &image,
                             vk::DeviceMemory &image_memory)
    {
        vk::ImageCreateInfo image_info{
            .imageType   = vk::ImageType::e2D,
            .format      = format,
            .extent      = {width, height, 1},
            .mipLevels   = 1,
            .arrayLayers = 1,
            .samples     = vk::SampleCountFlagBits::e1,
            .tiling      = tiling,
            .usage       = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };

        vk::Result result = logical_device_.createImage(&image_info, nullptr, &image);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to create image", result});

        vk::MemoryRequirements memRequirements;
        logical_device_.getImageMemoryRequirements(image, &memRequirements);

        auto memory_type_idx = findMemoryType(memRequirements.memoryTypeBits, properties);
        if (!memory_type_idx)
            return AppError::unexpected(memory_type_idx.error());

        vk::MemoryAllocateInfo allocInfo{
            .allocationSize  = memRequirements.size,
            .memoryTypeIndex = memory_type_idx.value(),
        };

        result = logical_device_.allocateMemory(&allocInfo, nullptr, &image_memory);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to allocated memory", result});

        result = logical_device_.bindImageMemory(image, image_memory, 0);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to bind image memory", result});

        return {};
    }

    [[nodiscard]]
    Result<void> transitionImageLayout(vk::Image const &image, vk::ImageLayout old_layout,
                                       vk::ImageLayout new_layout)
    {
        auto command_buffer = beginOneTimeCommandBuffer();
        if (!command_buffer)
            return AppError::unexpected(command_buffer.error());

        vk::ImageMemoryBarrier barrier{
            .oldLayout        = old_layout,
            .newLayout        = new_layout,
            .image            = image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };

        vk::PipelineStageFlags src_stage;
        vk::PipelineStageFlags dst_stage;

        if (old_layout == vk::ImageLayout::eUndefined &&
            new_layout == vk::ImageLayout::eTransferDstOptimal)
        {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            src_stage = vk::PipelineStageFlagBits::eTopOfPipe;
            dst_stage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (old_layout == vk::ImageLayout::eTransferDstOptimal &&
                 new_layout == vk::ImageLayout::eShaderReadOnlyOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            src_stage = vk::PipelineStageFlagBits::eTransfer;
            dst_stage = vk::PipelineStageFlagBits::eFragmentShader;
        }
        else
        {
            return AppError::unexpected({"Image layout transition not supported",
                                         ErrorKind::TransitionNotSupported});
        }

        command_buffer.value().pipelineBarrier(src_stage, dst_stage, {}, {}, nullptr, 0,
                                               nullptr, 1, &barrier);

        auto expected = endOneTimeCommandBuffer(command_buffer.value());

        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> createTextureImage()
    {
        int tex_width, tex_height, tex_channels;
        stbi_uc *pixels = stbi_load((appPath() / "textures/dirt.png").c_str(), &tex_width,
                                    &tex_height, &tex_channels, STBI_rgb_alpha);

        vk::DeviceSize image_size = tex_width * tex_height * STBI_rgb_alpha;

        if (!pixels)
            return AppError::unexpected(
                {"stbi_load failed", ErrorKind::FailedToLoadImage});

        vk::Buffer staging_buffer              = nullptr;
        vk::DeviceMemory staging_buffer_memory = nullptr;

        auto expected = createBuffer(image_size, vk::BufferUsageFlagBits::eTransferSrc,
                                     vk::MemoryPropertyFlagBits::eHostVisible |
                                         vk::MemoryPropertyFlagBits::eHostCoherent,
                                     staging_buffer, staging_buffer_memory);

        if (!expected)
            return AppError::unexpected(expected.error());

        void *data;
        vk::Result result =
            logical_device_.mapMemory(staging_buffer_memory, 0, image_size, {}, &data);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to map memory", result});

        memcpy(data, pixels, image_size);

        logical_device_.unmapMemory(staging_buffer_memory);

        stbi_image_free(pixels);

        expected = createImage(
            tex_width, tex_height, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal,
            vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
            vk::MemoryPropertyFlagBits::eDeviceLocal, texture_image_,
            texture_image_memory_);

        if (!expected)
            return AppError::unexpected(expected.error());

        expected = transitionImageLayout(texture_image_, vk::ImageLayout::eUndefined,
                                         vk::ImageLayout::eTransferDstOptimal);
        if (!expected)
            return AppError::unexpected(expected.error());

        expected =
            copyBufferToImage(staging_buffer, texture_image_, tex_width, tex_height);

        if (!expected)
            return AppError::unexpected(expected.error());

        expected =
            transitionImageLayout(texture_image_, vk::ImageLayout::eTransferDstOptimal,
                                  vk::ImageLayout::eShaderReadOnlyOptimal);
        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> createTextureImageView()
    {
        auto expected = createImageView(texture_image_, vk::Format::eR8G8B8A8Srgb);
        if (!expected)
            return AppError::unexpected(expected.error());

        texture_image_view_ = expected.value();

        return {};
    }

    [[nodiscard]]
    Result<void> createTextureSampler()
    {
        vk::PhysicalDeviceProperties properties;
        physical_device_.getProperties(&properties);

        vk::SamplerCreateInfo sampler_info{
            .magFilter               = vk::Filter::eNearest,
            .minFilter               = vk::Filter::eNearest,
            .mipmapMode              = vk::SamplerMipmapMode::eLinear,
            .addressModeU            = vk::SamplerAddressMode::eRepeat,
            .addressModeV            = vk::SamplerAddressMode::eRepeat,
            .addressModeW            = vk::SamplerAddressMode::eRepeat,
            .mipLodBias              = 0.0f,
            .anisotropyEnable        = vk::True,
            .maxAnisotropy           = properties.limits.maxSamplerAnisotropy,
            .compareEnable           = vk::False,
            .compareOp               = vk::CompareOp::eAlways,
            .minLod                  = 0.0f,
            .maxLod                  = 0.0f,
            .borderColor             = vk::BorderColor::eIntOpaqueBlack,
            .unnormalizedCoordinates = vk::False,
        };

        vk::Result result =
            logical_device_.createSampler(&sampler_info, nullptr, &texture_sampler_);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to create sampler", result});

        return {};
    }

    [[nodiscard]]
    Result<void> copyBufferToImage(vk::Buffer const &buffer, vk::Image &image,
                                   uint32_t width, uint32_t height)
    {
        auto command_buffer = beginOneTimeCommandBuffer();
        if (!command_buffer)
            return AppError::unexpected(command_buffer.error());

        vk::BufferImageCopy region{
            .bufferOffset      = 0,
            .bufferRowLength   = 0,
            .bufferImageHeight = 0,
            .imageSubresource  = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .imageOffset       = {0, 0, 0},
            .imageExtent       = {width, height, 1},
        };

        command_buffer.value().copyBufferToImage(
            buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

        auto expected = endOneTimeCommandBuffer(command_buffer.value());
        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<uint32_t> findMemoryType(uint32_t typeFilter,
                                    vk::MemoryPropertyFlags properties)
    {
        vk::PhysicalDeviceMemoryProperties mem_properties;
        physical_device_.getMemoryProperties(&mem_properties);

        for (size_t i = 0; i < mem_properties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) &&
                (mem_properties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        return AppError::unexpected(
            {"Failed to find memory type", ErrorKind::NoSuitableMemoryType});
    }

    [[nodiscard]]
    Result<void> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
                              vk::MemoryPropertyFlags properties, vk::Buffer &buffer,
                              vk::DeviceMemory &buffer_memory)
    {
        vk::BufferCreateInfo buffer_info{
            .size        = size,
            .usage       = usage,
            .sharingMode = vk::SharingMode::eExclusive,
        };

        vk::Result result = logical_device_.createBuffer(&buffer_info, nullptr, &buffer);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to create buffer", result});

        vk::MemoryRequirements mem_requirements;
        logical_device_.getBufferMemoryRequirements(buffer, &mem_requirements);

        auto memory_type_idx =
            findMemoryType(mem_requirements.memoryTypeBits, properties);
        if (!memory_type_idx)
            return AppError::unexpected(memory_type_idx.error());

        vk::MemoryAllocateInfo mem_alloc_info{
            .allocationSize  = mem_requirements.size,
            .memoryTypeIndex = memory_type_idx.value(),
        };

        result = logical_device_.allocateMemory(&mem_alloc_info, nullptr, &buffer_memory);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to allocate memory", result});

        result = logical_device_.bindBufferMemory(buffer, buffer_memory, 0);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to bind buffer memory", result});

        return {};
    }

    [[nodiscard]]
    Result<vk::CommandBuffer> beginOneTimeCommandBuffer()
    {
        vk::CommandBufferAllocateInfo alloc_info{
            .commandPool        = command_pool_,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };

        vk::CommandBuffer command_buffer;
        vk::Result result =
            logical_device_.allocateCommandBuffers(&alloc_info, &command_buffer);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to allocate command buffer", result});

        vk::CommandBufferBeginInfo begin_info{
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };

        result = command_buffer.begin(&begin_info);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to begin command buffer", result});

        return command_buffer;
    }

    [[nodiscard]]
    Result<void> endOneTimeCommandBuffer(vk::CommandBuffer &command_buffer)
    {
        vk::Result result = command_buffer.end();
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to end command buffer", result});

        vk::SubmitInfo submit_info{
            .commandBufferCount = 1,
            .pCommandBuffers    = &command_buffer,
        };

        result = queue_.submit(1, &submit_info, nullptr);
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to submit queue", result});

        result = queue_.waitIdle();
        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to queue wait idle", result});

        return {};
    }

    [[nodiscard]]
    Result<void> copyBuffer(vk::Buffer &src_buffer, vk::Buffer &dst_buffer,
                            vk::DeviceSize size)
    {
        auto copy_command_buf = beginOneTimeCommandBuffer();
        if (!copy_command_buf)
            return AppError::unexpected(copy_command_buf.error());

        vk::BufferCopy buffer_copy_region = {0, 0, size};
        copy_command_buf.value().copyBuffer(src_buffer, dst_buffer, 1,
                                            &buffer_copy_region);

        auto expected = endOneTimeCommandBuffer(copy_command_buf.value());
        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> createVertexBuffer()
    {
        vk::DeviceSize buffer_size = sizeof(vertices[0]) * vertices.size();

        // staging buffer, CPU vertex data will be put here and then transferred
        // to the GPU local vertex buffer
        vk::Buffer staging_buffer              = nullptr;
        vk::DeviceMemory staging_buffer_memory = nullptr;

        auto expected = createBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc,
                                     vk::MemoryPropertyFlagBits::eHostVisible |
                                         vk::MemoryPropertyFlagBits::eHostCoherent,
                                     staging_buffer, staging_buffer_memory);

        if (!expected)
            return AppError::unexpected(expected.error());

        void *data_staging;
        vk::Result result = logical_device_.mapMemory(staging_buffer_memory, 0,
                                                      buffer_size, {}, &data_staging);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to map memory", result});

        memcpy(data_staging, vertices.data(), buffer_size);

        logical_device_.unmapMemory(staging_buffer_memory);

        // vertex buffer
        expected = createBuffer(buffer_size,
                                vk::BufferUsageFlagBits::eVertexBuffer |
                                    vk::BufferUsageFlagBits::eTransferDst,
                                vk::MemoryPropertyFlagBits::eDeviceLocal, vertex_buffer_,
                                vertex_buffer_memory_);

        if (!expected)
            return AppError::unexpected(expected.error());

        expected = copyBuffer(staging_buffer, vertex_buffer_, buffer_size);

        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> createIndexBuffer()
    {
        vk::DeviceSize buffer_size = sizeof(indices[0]) * indices.size();

        // staging buffer, CPU vertex data will be put here and then transferred
        // to the GPU local vertex buffer
        vk::Buffer staging_buffer              = nullptr;
        vk::DeviceMemory staging_buffer_memory = nullptr;

        auto expected = createBuffer(buffer_size, vk::BufferUsageFlagBits::eTransferSrc,
                                     vk::MemoryPropertyFlagBits::eHostVisible |
                                         vk::MemoryPropertyFlagBits::eHostCoherent,
                                     staging_buffer, staging_buffer_memory);

        if (!expected)
            return AppError::unexpected(expected.error());

        void *data_staging;
        vk::Result result = logical_device_.mapMemory(staging_buffer_memory, 0,
                                                      buffer_size, {}, &data_staging);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to map memory", result});

        memcpy(data_staging, indices.data(), buffer_size);

        logical_device_.unmapMemory(staging_buffer_memory);

        // vertex buffer
        expected = createBuffer(buffer_size,
                                vk::BufferUsageFlagBits::eIndexBuffer |
                                    vk::BufferUsageFlagBits::eTransferDst,
                                vk::MemoryPropertyFlagBits::eDeviceLocal, index_buffer_,
                                index_buffer_memory_);

        if (!expected)
            return AppError::unexpected(expected.error());

        expected = copyBuffer(staging_buffer, index_buffer_, buffer_size);

        if (!expected)
            return AppError::unexpected(expected.error());

        return {};
    }

    [[nodiscard]]
    Result<void> createUniformBuffers()
    {
        uniform_buffers_.clear();
        uniform_buffers_memory_.clear();
        uniform_buffers_mapped_.clear();

        for (size_t i = 0; i < max_frames_in_flight; i++)
        {
            vk::DeviceSize buffer_size  = sizeof(UniformBufferObject);
            vk::Buffer buffer           = nullptr;
            vk::DeviceMemory buffer_mem = nullptr;

            auto expected =
                createBuffer(buffer_size, vk::BufferUsageFlagBits::eUniformBuffer,
                             vk::MemoryPropertyFlagBits::eHostVisible |
                                 vk::MemoryPropertyFlagBits::eHostCoherent,
                             buffer, buffer_mem);

            if (!expected)
                return AppError::unexpected(expected.error());

            void *data = nullptr;
            vk::Result result =
                logical_device_.mapMemory(buffer_mem, 0, buffer_size, {}, &data);

            if (result != vk::Result::eSuccess)
                return AppError::unexpected({"Failed to map memory", result});

            uniform_buffers_.emplace_back(std::move(buffer));
            uniform_buffers_memory_.emplace_back(std::move(buffer_mem));
            uniform_buffers_mapped_.emplace_back(std::move(data));
        }

        return {};
    }

    [[nodiscard]]
    Result<void> createDescriptorPool()
    {
        std::array pool_size{
            vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
                                   max_frames_in_flight),
            vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler,
                                   max_frames_in_flight),
        };

        vk::DescriptorPoolCreateInfo pool_info{
            .flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
            .maxSets       = max_frames_in_flight,
            .poolSizeCount = static_cast<uint32_t>(pool_size.size()),
            .pPoolSizes    = pool_size.data(),
        };

        vk::Result result =
            logical_device_.createDescriptorPool(&pool_info, nullptr, &descriptor_pool_);

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to create descriptor pool", result});

        return {};
    }

    [[nodiscard]]
    Result<void> createDescriptorSets()
    {
        std::vector<vk::DescriptorSetLayout> layouts(max_frames_in_flight,
                                                     descriptor_set_layout_);

        vk::DescriptorSetAllocateInfo alloc_info{
            .descriptorPool     = descriptor_pool_,
            .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
            .pSetLayouts        = layouts.data(),
        };

        descriptor_sets_.clear();

        descriptor_sets_.reserve(layouts.size());
        vk::Result result =
            logical_device_.allocateDescriptorSets(&alloc_info, descriptor_sets_.data());

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to allocated descriptor sets", result});

        for (size_t i = 0; i < max_frames_in_flight; i++)
        {
            vk::DescriptorBufferInfo buffer_info{.buffer = uniform_buffers_[i],
                                                 .offset = 0,
                                                 .range  = sizeof(UniformBufferObject)};

            vk::DescriptorImageInfo image_info{
                .sampler     = texture_sampler_,
                .imageView   = texture_image_view_,
                .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};

            std::array descriptor_writes{
                vk::WriteDescriptorSet{
                    .dstSet          = descriptor_sets_[i],
                    .dstBinding      = 0,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eUniformBuffer,
                    .pBufferInfo     = &buffer_info,
                },
                vk::WriteDescriptorSet{
                    .dstSet          = descriptor_sets_[i],
                    .dstBinding      = 1,
                    .dstArrayElement = 0,
                    .descriptorCount = 1,
                    .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                    .pImageInfo      = &image_info,
                }};

            logical_device_.updateDescriptorSets(descriptor_writes.size(),
                                                 descriptor_writes.data(), 0, nullptr);
        }

        return {};
    }

    [[nodiscard]]
    Result<void> createCommandBuffers()
    {
        vk::CommandBufferAllocateInfo alloc_info{
            .commandPool        = command_pool_,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = max_frames_in_flight,
        };

        command_buffers_.resize(max_frames_in_flight);
        vk::Result result =
            logical_device_.allocateCommandBuffers(&alloc_info, command_buffers_.data());

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to allocate command buffers", result});

        return {};
    }

    [[nodiscard]]
    Result<void> recordCommandBuffer(uint32_t image_idx)
    {
        vk::CommandBuffer &command_buffer = command_buffers_[frame_idx_];

        vk::Result result = command_buffer.begin({});

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to begin command buffer", result});

        // changing image layout from undefined to color attachment optimal
        transitionImageLayout(image_idx, vk::ImageLayout::eUndefined,
                              vk::ImageLayout::eColorAttachmentOptimal, {},
                              vk::AccessFlagBits2::eColorAttachmentWrite,
                              vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                              vk::PipelineStageFlagBits2::eColorAttachmentOutput);

        vk::ClearValue clear_color = {{0.0f, 0.0f, 0.0f, 1.0}};

        vk::RenderingAttachmentInfo attachment_info = {
            .imageView   = swapchain_image_views_[image_idx], // rendering to this
                                                              // image in the swapchain
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = clear_color};

        vk::RenderingInfo renderingInfo = {
            .renderArea           = {.offset = {0, 0}, .extent = swapchain_extent_},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &attachment_info,
        };

        command_buffer.beginRendering(&renderingInfo);

        command_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphics_pipeline_);

        vk::Viewport viewport = {0.0f,
                                 0.0f,
                                 static_cast<float>(swapchain_extent_.width),
                                 static_cast<float>(swapchain_extent_.height),
                                 0.0f,
                                 1.0f};

        command_buffer.setViewport(0, 1, &viewport);

        vk::Rect2D scissor = {vk::Offset2D(0, 0), swapchain_extent_};

        command_buffer.setScissor(0, 1, &scissor);

        command_buffer.bindVertexBuffers(0, 1, &vertex_buffer_, nullptr);
        command_buffer.bindIndexBuffer(index_buffer_, 0, vk::IndexType::eUint16);

        command_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                          pipeline_layout_, 0, 1,
                                          &descriptor_sets_[frame_idx_], 0, nullptr);

        command_buffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);

        command_buffer.endRendering();

        transitionImageLayout(image_idx, vk::ImageLayout::eColorAttachmentOptimal,
                              vk::ImageLayout::ePresentSrcKHR,
                              vk::AccessFlagBits2::eColorAttachmentWrite, {},
                              vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                              vk::PipelineStageFlagBits2::eBottomOfPipe);

        result = command_buffer.end();

        if (result != vk::Result::eSuccess)
            return AppError::unexpected({"Failed to end command buffer", result});

        return {};
    }

    void transitionImageLayout(uint32_t image_idx, vk::ImageLayout old_layout,
                               vk::ImageLayout new_layout,
                               vk::AccessFlags2 old_access_mask,
                               vk::AccessFlags2 new_access_mask,
                               vk::PipelineStageFlags2 old_stage_mask,
                               vk::PipelineStageFlags2 new_stage_mask)
    {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask        = old_stage_mask,
            .srcAccessMask       = old_access_mask,
            .dstStageMask        = new_stage_mask,
            .dstAccessMask       = new_access_mask,
            .oldLayout           = old_layout,
            .newLayout           = new_layout,
            .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
            .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
            .image               = swapchain_images_[image_idx],
            .subresourceRange    = {.aspectMask     = vk::ImageAspectFlagBits::eColor,
                                    .baseMipLevel   = 0,
                                    .levelCount     = 1,
                                    .baseArrayLayer = 0,
                                    .layerCount     = 1},
        };

        vk::DependencyInfo dependency_info = {
            .dependencyFlags         = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrier,
        };

        command_buffers_[frame_idx_].pipelineBarrier2(&dependency_info);
    }

    [[nodiscard]]
    Result<void> createSyncObjects()
    {
        assert(present_complete_sphrs_.empty() && render_finished_sphrs_.empty() &&
               draw_fences_.empty());

        render_finished_sphrs_.reserve(swapchain_images_.size());
        for (size_t i = 0; i < swapchain_images_.size(); i++)
        {
            vk::Result result = logical_device_.createSemaphore(
                nullptr, nullptr, &render_finished_sphrs_[i]);

            if (result != vk::Result::eSuccess)
                return AppError::unexpected({"Failed to create semaphore", result});
        }

        present_complete_sphrs_.reserve(max_frames_in_flight);
        draw_fences_.reserve(max_frames_in_flight);
        for (size_t i = 0; i < max_frames_in_flight; i++)
        {
            vk::Result result = logical_device_.createSemaphore(
                nullptr, nullptr, &present_complete_sphrs_[i]);

            if (result != vk::Result::eSuccess)
                return AppError::unexpected({"Failed to create semaphore", result});

            vk::FenceCreateInfo fence_create_info = {
                .flags = vk::FenceCreateFlagBits::eSignaled};

            result = logical_device_.createFence(&fence_create_info, nullptr,
                                                 &draw_fences_[i]);

            if (result != vk::Result::eSuccess)
                return AppError::unexpected({"Failed to create fence", result});
        }

        return {};
    }
};
