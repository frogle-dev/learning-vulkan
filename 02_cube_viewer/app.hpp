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
#include "deletion_queue.hpp"
#include "error.hpp"
#include "stb_image.h"

inline std::filesystem::path appPath()
{
    return std::filesystem::canonical("/proc/self/exe").parent_path().parent_path();
}

uint8_t constexpr max_frames_in_flight = 2;
uint32_t constexpr api_version         = vk::ApiVersion14;

#ifdef NDEBUG
bool constexpr enable_validation_layers = false;
#else
bool constexpr enable_validation_layers = true;
#endif

std::array<char const *, 1> constexpr validation_layers = {"VK_LAYER_KHRONOS_validation"};

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

auto constexpr vertices = std::to_array<Vertex>({
    {{-0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.5f, 0.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
});

auto constexpr indices = std::to_array<uint16_t>({0, 1, 2, 2, 3, 0});

struct UniformBufferObject
{
    alignas(16) glm::mat4 model;
    alignas(16) glm::mat4 view;
    alignas(16) glm::mat4 proj;
};

struct Texture
{
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImage image            = VK_NULL_HANDLE;
    VkImageView view         = VK_NULL_HANDLE;
    VkSampler sampler        = VK_NULL_HANDLE;
};

enum class AppErrorKind
{
    VulkanFailure,
    SDLFailure,
    ValidationLayerNotSupported,
    ExtensionNotSupported,
    FailedToFindGPU,
    NoSuitableQueueFamily,
    FailedToOpenFile,
    TransitionNotSupported,
    FailedToLoadImage,
    NoSuitableMemoryType,
};

using AppError = Error<AppErrorKind>;

template <typename T> using AppResult = std::expected<T, AppError>;

class App
{
  public:
    [[nodiscard]] static AppResult<App> init(Window &window, std::string const &app_name);
    [[nodiscard]] AppResult<void> deinit();

    bool isRunning();
    [[nodiscard]] AppResult<void> pollEvents();
    [[nodiscard]] AppResult<void> endFrame();

  private:
    App() = default;

    std::string app_name_;

    DeletionQueue deletion_queue_;

    Window *window_ = nullptr;

    bool running_ = false;

    std::array<char const *, 1> required_device_extensions_ = {
        vk::KHRSwapchainExtensionName};

    vk::Instance instance_              = VK_NULL_HANDLE;
    vk::PhysicalDevice physical_device_ = VK_NULL_HANDLE; // Physical device represents
                                                          // the GPU
    vk::Device logical_device_ = nullptr; // Logical Device is the interface for the
                                          // physical device
    vk::Queue queue_           = VK_NULL_HANDLE;
    uint32_t queue_family_idx_ = UINT32_MAX;

    vk::DebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;

    vk::SurfaceKHR window_surface_ = VK_NULL_HANDLE; // Surface to render to window

    /* SWAPCHAIN */

    vk::SwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<vk::Image> swapchain_images_;
    vk::SurfaceFormatKHR swapchain_surface_format_;
    vk::Extent2D swapchain_extent_;
    std::vector<vk::ImageView> swapchain_image_views_;
    DeletionQueue swapchain_deletion_queue_;

    vk::DescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    vk::PipelineLayout pipeline_layout_            = VK_NULL_HANDLE;
    vk::Pipeline graphics_pipeline_                = VK_NULL_HANDLE;

    vk::CommandPool command_pool_ = VK_NULL_HANDLE;
    std::array<vk::CommandBuffer, max_frames_in_flight> command_buffers_;

    uint32_t frame_idx_ = 0;

    /* BUFFERS */

    // vk::Buffer vertex_buffer_              = VK_NULL_HANDLE;
    // vk::DeviceMemory vertex_buffer_memory_ = VK_NULL_HANDLE;
    // vk::Buffer index_buffer_               = VK_NULL_HANDLE;
    // vk::DeviceMemory index_buffer_memory_  = VK_NULL_HANDLE;
    //
    // std::vector<vk::Buffer> uniform_buffers_;
    // std::vector<vk::DeviceMemory> uniform_buffers_memory_;
    // std::vector<void *> uniform_buffers_mapped_;

    VmaAllocation vertex_buffer_allocation_ = VK_NULL_HANDLE;
    VkBuffer vertex_buffer_                 = VK_NULL_HANDLE;
    VmaAllocation index_buffer_allocation_  = VK_NULL_HANDLE;
    VkBuffer index_buffer_                  = VK_NULL_HANDLE;

    vk::DescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    std::vector<vk::DescriptorSet> descriptor_sets_;

    /* TEXTURES */

    vk::Image texture_image_               = VK_NULL_HANDLE;
    vk::DeviceMemory texture_image_memory_ = VK_NULL_HANDLE;
    vk::ImageView texture_image_view_      = VK_NULL_HANDLE;
    vk::Sampler texture_sampler_           = VK_NULL_HANDLE;

    /* SYNC OBJECTS */

    std::vector<vk::Semaphore> image_acquire_sphrs_;
    std::vector<vk::Semaphore> render_complete_sphrs_;
    std::vector<vk::Fence> draw_fences_;

    /* APPLICATION METHODS */

    [[nodiscard]] AppResult<void> initVulkan();
    [[nodiscard]] AppResult<void> drawFrame();
    void updateUniformBuffer(uint32_t current_image);

    /* SETUP METHODS */

    [[nodiscard]] static AppResult<std::vector<char>> readFile(std::string const &path);

    /// VKAPI_ATTR, VKAPI_CALL gives the function a signature that vulkan can
    /// call
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
        vk::DebugUtilsMessageTypeFlagsEXT type,
        vk::DebugUtilsMessengerCallbackDataEXT const *pCallbackData, void *pUserData);

    [[nodiscard]] AppResult<void> setupDebugMessenger();

    std::vector<char const *> getRequiredInstanceExtensions();
    [[nodiscard]] AppResult<void> createInstance();
    [[nodiscard]] AppResult<void> createWindowSurface();

    [[nodiscard]] AppResult<bool> isDeviceSuitable(
        vk::PhysicalDevice const &physical_device);

    [[nodiscard]] AppResult<void> pickPhysicalDevice();
    [[nodiscard]] AppResult<void> createLogicalDevice();

    vk::SurfaceFormatKHR chooseSwapchainSurfaceFormat(
        std::vector<vk::SurfaceFormatKHR> const &available_formats);

    vk::PresentModeKHR chooseSwapchainPresentMode(
        std::vector<vk::PresentModeKHR> const &available_present_modes);

    [[nodiscard]] AppResult<vk::Extent2D> chooseSwapchainExtent(
        vk::SurfaceCapabilitiesKHR const &surface_capabilities);

    [[nodiscard]] AppResult<void> recreateSwapchain();

    /// if recreate is on, the swapchain cleanup will have to be handled manually
    /// only turn on recreate when the swapchain must be recreated and the old swapchain
    /// deletion queue has to be flushed before a new swapchain is appended
    [[nodiscard]] AppResult<void> createSwapchain(bool recreate);

    [[nodiscard]] AppResult<vk::ImageView> createImageView(
        vk::Image const &image, vk::Format format, bool for_swapchain);

    [[nodiscard]] AppResult<void> createImageViews();
    [[nodiscard]] AppResult<void> createDescriptorSetLayout();
    [[nodiscard]] AppResult<void> createGraphicsPipeline();

    [[nodiscard]] AppResult<vk::ShaderModule> createShaderModule(
        std::vector<char> const &code);

    [[nodiscard]] AppResult<void> createCommandPool();

    [[nodiscard]] AppResult<void> createImage(
        uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling,
        vk::ImageUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Image &image,
        vk::DeviceMemory &image_memory);

    [[nodiscard]] AppResult<void> transitionImageLayout(
        vk::Image const &image, vk::ImageLayout old_layout, vk::ImageLayout new_layout);

    void transitionImageLayout(
        uint32_t image_idx, vk::ImageLayout old_layout, vk::ImageLayout new_layout,
        vk::AccessFlags2 old_access_mask, vk::AccessFlags2 new_access_mask,
        vk::PipelineStageFlags2 old_stage_mask, vk::PipelineStageFlags2 new_stage_mask);

    [[nodiscard]] AppResult<void> createTextureImage();
    [[nodiscard]] AppResult<void> createTextureImageView();
    [[nodiscard]] AppResult<void> createTextureSampler();

    [[nodiscard]] AppResult<void> copyBufferToImage(
        vk::Buffer const &buffer, vk::Image &image, uint32_t width, uint32_t height);

    [[nodiscard]] AppResult<uint32_t> findMemoryType(uint32_t typeFilter,
                                                     vk::MemoryPropertyFlags properties);

    [[nodiscard]] AppResult<void> createBuffer(
        vk::DeviceSize size, vk::BufferUsageFlags usage,
        vk::MemoryPropertyFlags properties, vk::Buffer &buffer,
        vk::DeviceMemory &buffer_memory);

    [[nodiscard]] AppResult<vk::CommandBuffer> beginOneTimeCommandBuffer();

    [[nodiscard]] AppResult<void> endOneTimeCommandBuffer(
        vk::CommandBuffer &command_buffer);

    [[nodiscard]] AppResult<void> copyBuffer(vk::Buffer &src_buffer,
                                             vk::Buffer &dst_buffer, vk::DeviceSize size);

    [[nodiscard]] AppResult<void> createVertexBuffer();
    [[nodiscard]] AppResult<void> createIndexBuffer();
    [[nodiscard]] AppResult<void> createUniformBuffers();

    [[nodiscard]] AppResult<void> createDescriptorPool();
    [[nodiscard]] AppResult<void> createDescriptorSets();

    [[nodiscard]] AppResult<void> createCommandBuffers();
    [[nodiscard]] AppResult<void> recordCommandBuffer(uint32_t image_idx);

    [[nodiscard]] AppResult<void> createSyncObjects();
};
