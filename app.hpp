#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <SDL3/SDL_hints.h>

#include <vulkan/vulkan_core.h>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include "vulkan/vulkan.hpp"
#include <vulkan/vulkan_raii.hpp>

#include <glm/glm.hpp>

#include <assert.h>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <array>
#include <iostream>
#include <stdexcept>
#include <fstream>

inline std::filesystem::path appPath()
{
    return std::filesystem::canonical("/proc/self/exe").parent_path().parent_path();
}

constexpr uint16_t width = 960;
constexpr uint16_t height = 960;

constexpr uint8_t maxFramesInFlight = 2;

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

constexpr std::array<char const*, 1> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 col;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        return {.binding = 0, .stride = sizeof(Vertex), .inputRate = vk::VertexInputRate::eVertex};
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        return {{{.location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat, .offset = offsetof(Vertex, pos)},
                 {.location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, col)}}};
    }
};

const std::vector<Vertex> vertices = {
    // pos           // col
    {{ 0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{ 0.5f,  0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}}
};

class Application
{
public:
    void run() 
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    SDL_Window* window = nullptr;

    SDL_Event event;
    bool quit = false;

    std::array<const char*, 1> requiredDeviceExtensions = {
        vk::KHRSwapchainExtensionName
    };

    vk::raii::Context                context;
    vk::raii::Instance               instance       = nullptr;
    vk::raii::PhysicalDevice         physicalDevice = nullptr; // Physical device represents the GPU
    vk::raii::Device                 logicalDevice  = nullptr; // Logical Device is the interface for the physical device
    vk::raii::Queue                  queue          = nullptr;
    uint32_t                         queueIdx       = UINT32_MAX;

    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

    vk::raii::SurfaceKHR             windowSurface  = nullptr;  // Surface to render to window

    vk::raii::SwapchainKHR           swapchain      = nullptr;
    std::vector<vk::Image>           swapchainImages;
    vk::SurfaceFormatKHR             swapchainSurfaceFormat;
    vk::Extent2D                     swapchainExtent;
    std::vector<vk::raii::ImageView> swapchainImageViews;

    vk::raii::PipelineLayout         pipelineLayout   = nullptr;
    vk::raii::Pipeline               graphicsPipeline = nullptr;

    vk::raii::CommandPool            commandPool      = nullptr;
    std::vector
        <vk::raii::CommandBuffer>    commandBuffers;

    uint32_t                         frameIdx = 0;
    bool                             framebufferResized = false;

    vk::raii::Buffer       vertexBuffer       = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;

    // Sync objects
    std::vector<vk::raii::Semaphore> presentCompleteSphrs;
    std::vector<vk::raii::Semaphore> renderFinishedSphrs;
    std::vector<vk::raii::Fence>     drawFences;

    /* APPLICATION METHODS */

    void initWindow()
    {
        SDL_Init(SDL_INIT_VIDEO);

        SDL_SetHint(SDL_HINT_APP_ID, "HelloVulkan");
        window = SDL_CreateWindow("HelloVulkan", width, height, SDL_WINDOW_RESIZABLE);
    }

    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createWindowSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createImageViews();
        createGraphicsPipeline();
        createCommandPool();
        createVertexBuffer();
        createCommandBuffers();
        createSyncObjects();
    }

    void mainLoop()
    {
        while (!quit) {
            while (SDL_PollEvent(&event) == true)
            {
                if (event.type == SDL_EVENT_QUIT) quit = true;
                if (event.type == SDL_EVENT_WINDOW_RESIZED)
                {
                    framebufferResized = true;
                }
            }

            // std::cout << "hello, vulkan! " << frameIdx << std::endl;
            drawFrame();
        }

        logicalDevice.waitIdle();
    }

    void drawFrame()
    {
        vk::Result fenceResult = logicalDevice.waitForFences(*drawFences[frameIdx], vk::True, UINT64_MAX);
        if (fenceResult != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to wait for fence");
        }

        uint32_t imageIdx;
        try
        {
            auto [result, idx] = swapchain.acquireNextImage(UINT64_MAX, *presentCompleteSphrs[frameIdx], nullptr);

            if (result == vk::Result::eErrorOutOfDateKHR)
            {
                recreateSwapchain();
                return;
            }
            if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
            {
                assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
                throw std::runtime_error("Failed to acquire swap chain image");
            }

            imageIdx = idx;
        }
        catch (vk::OutOfDateKHRError)
        {
            recreateSwapchain();
            return;
        }

        logicalDevice.resetFences(*drawFences[frameIdx]);

        commandBuffers[frameIdx].reset();
        recordCommandBuffer(imageIdx);

        vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        const vk::SubmitInfo submitInfo
        {
            .waitSemaphoreCount   = 1,
            .pWaitSemaphores      = &*presentCompleteSphrs[frameIdx], // semaphores to wait for
            .pWaitDstStageMask    = &waitDestinationStageMask,
            .commandBufferCount   = 1,
            .pCommandBuffers      = &*commandBuffers[frameIdx],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores    = &*renderFinishedSphrs[imageIdx], // semaphores to signal when done
        };

        queue.submit(submitInfo, *drawFences[frameIdx]);

        const vk::PresentInfoKHR presentInfoKHR
        {
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &*renderFinishedSphrs[imageIdx],
            .swapchainCount     = 1,
            .pSwapchains        = &*swapchain,
            .pImageIndices      = &imageIdx,
        };

        try
        {
            vk::Result result = queue.presentKHR(presentInfoKHR);
            if (result == vk::Result::eSuboptimalKHR || result == vk::Result::eErrorOutOfDateKHR || framebufferResized)
            {
                framebufferResized = false;
                recreateSwapchain();
            }
            else
            {
                assert(result == vk::Result::eSuccess);
            }
        }
        catch (vk::OutOfDateKHRError&)
        {
            framebufferResized = false;
            recreateSwapchain();
        }

        frameIdx = (frameIdx + 1) % maxFramesInFlight;
    }

    void cleanup()
    {
        cleanupSwapchain();

        SDL_DestroyWindow(window);

        SDL_Quit();
    }

    /* SETUP METHODS */

    static std::vector<char> readFile(const std::string &path)
    {
        // std::ios::ate - reading starts at the end of file
        // std::ios::binary - reads file as a binary
        std::ifstream fin(path, std::ios::ate | std::ios::binary);

        if (!fin.is_open())
        {
            throw std::runtime_error("Failed to open file");
        }

        // get position at end of file to get file length
        std::vector<char> buffer(fin.tellg());

        // go to beginning of file
        fin.seekg(0, std::ios::beg);
        fin.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));

        fin.close();

        return buffer;
    }

    // VKAPI_ATTR, VKAPI_CALL gives the function a signature that vulkan can call
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
            vk::DebugUtilsMessageSeverityFlagBitsEXT severity, 
            vk::DebugUtilsMessageTypeFlagsEXT type, 
            const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, 
            void *pUserData)
    {
        std::cerr << "[Validation layer]: " << to_string(severity) << " , " << "[Type]: " << to_string(type) << " , " << "[Message]:" << std::endl << std::endl << pCallbackData->pMessage << std::endl << "----------------" << std::endl;

        return vk::False;
    }

    void setupDebugMessenger()
    {
        if (!enableValidationLayers) return;

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | 
                                                            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
        vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                                           vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                                                           vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT
        {
            .messageSeverity = severityFlags,
            .messageType     = messageTypeFlags,
            .pfnUserCallback = &debugCallback
        };

        debugMessenger = instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }

    std::vector<const char*> getRequiredInstanceExtensions() 
    {
        uint32_t sdlExtensionCount = 0;
        const char *const * sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);

        std::vector extensions(sdlExtensions, sdlExtensions + sdlExtensionCount);
        if (enableValidationLayers)
        {
            extensions.push_back(vk::EXTDebugUtilsExtensionName);
        }

        return extensions;
    }

    void createInstance()
    {
        // VULKAN INSTANCE CREATION
        // instance is used to communicate with vulkan
        constexpr vk::ApplicationInfo appInfo
        {
            .pApplicationName   = "Learn Vulkan",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "No Engine",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = vk::ApiVersion14
        };


        // VALIDATION LAYERS
        std::vector<const char*> requiredLayers;
        if (enableValidationLayers) 
        {
            requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        // check if validation layers are available
        std::vector<vk::LayerProperties> layerProperties = context.enumerateInstanceLayerProperties();
        auto unsupportedLayerIterator = std::ranges::find_if(requiredLayers, // returns iterator to first layer that is not supported
            [&layerProperties](const char*& requiredLayer) 
            {
                return std::ranges::none_of(layerProperties, // check if any required layers are missing from the instance's supported layers
                    [requiredLayer](const vk::LayerProperties& layerProperty) 
                    { 
                        return strcmp(layerProperty.layerName, requiredLayer) == 0; 
                    });
            });

        // find_if returns end() iterator if all the layers are supported
        // this will run only if there is a required layer that isnt supported
        if (unsupportedLayerIterator != requiredLayers.end()) 
        {
            throw std::runtime_error("Required layer not supported: " + std::string(*unsupportedLayerIterator));
        }

        // EXTENSIONS
        std::vector<const char*> requiredExtensions = getRequiredInstanceExtensions();

        std::vector<vk::ExtensionProperties> extensionProperties = context.enumerateInstanceExtensionProperties();
        auto unsupportedPropertyIterator = std::ranges::find_if(requiredExtensions,
            [&extensionProperties](const char*& requiredExtension) 
            {
                return std::ranges::none_of(extensionProperties,
                    [requiredExtension](const vk::ExtensionProperties& extensionProperty) 
                    {
                        return strcmp(extensionProperty.extensionName, requiredExtension) == 0;
                    });
            });

        if (unsupportedPropertyIterator != requiredExtensions.end())
        {
            throw std::runtime_error("Required extension not supported: " + std::string(*unsupportedPropertyIterator));
        }

        // CREATING THE INSTANCE
        vk::InstanceCreateInfo createInfo
        {
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
            .ppEnabledLayerNames     = requiredLayers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data()
        };

        instance = vk::raii::Instance(context, createInfo);
    }

    void createWindowSurface()
    {
        VkSurfaceKHR c_api_Surface; // sdl only handles vulkan's C api, so a vulkan C surface is needed

        if (!SDL_Vulkan_CreateSurface(window, *instance, nullptr, &c_api_Surface))
        {
            throw std::runtime_error("Failed to create window surface");
        }

        windowSurface = vk::raii::SurfaceKHR(instance, c_api_Surface);
    }

    bool isDeviceSuitable(const vk::raii::PhysicalDevice &physicalDevice)
    {
        // if supports vulkan 1.3
        bool supportsVulkan1_3 = physicalDevice.getProperties().apiVersion >= vk::ApiVersion13;

        // if supports graphics queue family
        std::vector<vk::QueueFamilyProperties> queueFamilies = physicalDevice.getQueueFamilyProperties();

        bool supportsGraphics = std::ranges::any_of(queueFamilies, 
            [](const vk::QueueFamilyProperties& queueFamilyProp)
            {
                return static_cast<bool>(queueFamilyProp.queueFlags & vk::QueueFlagBits::eGraphics);
            });

        // if supports specific extensions
        std::vector<vk::ExtensionProperties> availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();

        // if any of the required device extensions aren't available -> false
        bool supportsAllRequiredExtensions = std::ranges::all_of(requiredDeviceExtensions,
            [&availableDeviceExtensions](const char*& requiredDeviceExtension)
            {
                return std::ranges::any_of(availableDeviceExtensions,
                    [requiredDeviceExtension](const vk::ExtensionProperties& availableDeviceExtension)
                    {
                        return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
                    });
            });

        // if supports specific features
        auto features = physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
        bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters &&
                                        features.template get<vk::PhysicalDeviceVulkan13Features>().synchronization2 &&
                                        features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                        features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

        return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
    }

    void pickPhysicalDevice()
    {
        // checking if physical devices meet requirements

        std::vector<vk::raii::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();

        // find if a GPU meets all the requirements
        const auto deviceIterator = std::ranges::find_if(physicalDevices,
            [&](const vk::raii::PhysicalDevice &physDevice)
            {
                return isDeviceSuitable(physDevice);
            });

        if(deviceIterator == physicalDevices.end())
        {
            throw std::runtime_error("Failed to find a GPU with support for all requirements");
        }

        physicalDevice = *deviceIterator;
    }

    void createLogicalDevice()
    {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // check for support of both 'graphics' and 'present' queue families
        for (uint32_t queueFamilyPropIdx = 0; queueFamilyPropIdx < queueFamilyProperties.size(); queueFamilyPropIdx++)
        {
            if ((queueFamilyProperties[queueFamilyPropIdx].queueFlags & vk::QueueFlagBits::eGraphics) && physicalDevice.getSurfaceSupportKHR(queueFamilyPropIdx, *windowSurface))
            {
                queueIdx = queueFamilyPropIdx;
                break;
            }
        }

        if (queueIdx == UINT32_MAX)
        {
            throw std::runtime_error("Graphics and Present queue families not found");
        }

        // getting features
        // structure chain connects each feature struct with pointers, making moving through them easy
        vk::StructureChain featureChain = {
            vk::PhysicalDeviceFeatures2 {},
            vk::PhysicalDeviceVulkan11Features {.shaderDrawParameters = true},
            vk::PhysicalDeviceVulkan13Features {.synchronization2 = true, .dynamicRendering = true},
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT {.extendedDynamicState = true}
        };

        float queuePriority = 0.5f; // priority for scheduling of command buffer execution, needed even if there is one queue
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo
        {
            .queueFamilyIndex = queueIdx,
            .queueCount       = 1,
            .pQueuePriorities = &queuePriority
        };

        vk::DeviceCreateInfo deviceCreateInfo
        {
            .pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(), // connecting the chain of features to vulkan
            .queueCreateInfoCount    = 1,
            .pQueueCreateInfos       = &deviceQueueCreateInfo,
            .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtensions.size()),
            .ppEnabledExtensionNames = requiredDeviceExtensions.data()
        };

        logicalDevice = vk::raii::Device(physicalDevice, deviceCreateInfo);
        queue = vk::raii::Queue(logicalDevice, queueIdx, 0);
    }

    vk::SurfaceFormatKHR chooseSwapchainSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
    {
        assert(!availableFormats.empty());

        const auto formatIterator = std::ranges::find_if(availableFormats,
            [](const vk::SurfaceFormatKHR &format)
            {
                return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
            });

        return formatIterator != availableFormats.end() ? *formatIterator : availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapchainPresentMode(const std::vector<vk::PresentModeKHR> &availablePresentModes)
    {
        // fifo present mode - stores rendered images in a queue, takes an image from the front of the queue to display every time the display refreshes
        // mailbox present mode - like fifo, but when the queue is full it replaces old images with new ones to display images as fast as possible

        assert(std::ranges::any_of(availablePresentModes,
            [](vk::PresentModeKHR presentMode)
            {
                return presentMode == vk::PresentModeKHR::eFifo;
            }));

        // if mailbox present mode is available, use it, otherwise FIFO present mode
        return std::ranges::any_of(availablePresentModes,
            [](const vk::PresentModeKHR presentMode)
            {
                return vk::PresentModeKHR::eMailbox == presentMode;
            }) 
            ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapchainExtent(const vk::SurfaceCapabilitiesKHR &surfaceCapabilities)
    {
        // extent is the resolution of the images in the swapchain

        if (surfaceCapabilities.currentExtent.width != UINT32_MAX)
        {
            return surfaceCapabilities.currentExtent;
        }

        int width, height;
        SDL_GetWindowSizeInPixels(window, &width, &height);

        return vk::Extent2D{
            std::clamp<uint32_t>(width, surfaceCapabilities.minImageExtent.width, surfaceCapabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, surfaceCapabilities.minImageExtent.height, surfaceCapabilities.maxImageExtent.height)
        };
    }

    uint32_t chooseSwapchainMinImageCount(const vk::SurfaceCapabilitiesKHR &surfaceCapabilities)
    {
        uint32_t minImgCount = std::max(uint32_t(3), surfaceCapabilities.minImageCount);

        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImgCount))
        {
            minImgCount = surfaceCapabilities.maxImageCount;
        }

        return minImgCount;
    }

    void cleanupSwapchain()
    {
        swapchainImageViews.clear();
        swapchain = nullptr;
    }

    void recreateSwapchain()
    {
        int width = 0;
        int height = 0;
        SDL_GetWindowSizeInPixels(window, &width, &height);
        while (width == 0 || height == 0)
        {
            SDL_GetWindowSizeInPixels(window, &width, &height);
            SDL_WaitEvent(&event);
        }

        logicalDevice.waitIdle();

        swapchainImageViews.clear();
        // cleanupSwapchain();

        createSwapchain();
        createImageViews();
    }

    void createSwapchain() 
    {
        vk::SurfaceCapabilitiesKHR surfaceCapabilities = physicalDevice.getSurfaceCapabilitiesKHR(*windowSurface);
        swapchainExtent = chooseSwapchainExtent(surfaceCapabilities);
        uint32_t minImageCount = chooseSwapchainMinImageCount(surfaceCapabilities);

        std::vector<vk::SurfaceFormatKHR> availableFormats = physicalDevice.getSurfaceFormatsKHR(windowSurface);
        swapchainSurfaceFormat = chooseSwapchainSurfaceFormat(availableFormats);

        std::vector<vk::PresentModeKHR> availablePresentModes = physicalDevice.getSurfacePresentModesKHR(windowSurface);
        vk::PresentModeKHR presentMode = chooseSwapchainPresentMode(availablePresentModes);

        vk::SwapchainCreateInfoKHR swapchainCreateInfo
        {
            .surface          = *windowSurface,
            .minImageCount    = minImageCount,
            .imageFormat      = swapchainSurfaceFormat.format,
            .imageColorSpace  = swapchainSurfaceFormat.colorSpace,
            .imageExtent      = swapchainExtent,
            .imageArrayLayers = 1,
            .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform     = surfaceCapabilities.currentTransform,
            .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode      = presentMode,
            .clipped          = true,
            .oldSwapchain     = *swapchain,
        };

        swapchain = vk::raii::SwapchainKHR(logicalDevice, swapchainCreateInfo);

        swapchainImages = swapchain.getImages();
    }

    void createImageViews()
    {
        assert(swapchainImageViews.empty());

        vk::ImageViewCreateInfo imageViewCreateInfo
        {
            .viewType         = vk::ImageViewType::e2D,
            .format           = swapchainSurfaceFormat.format,
            .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1 }
        };

        for (vk::Image &image : swapchainImages)
        {
            imageViewCreateInfo.image = image;
            swapchainImageViews.emplace_back(vk::raii::ImageView(logicalDevice, imageViewCreateInfo));
        }
    }

    void createGraphicsPipeline()
    {
        /* SHADER STAGE SETUP */

        vk::raii::ShaderModule shaderModule = createShaderModule(readFile(appPath() / "shaders/slang.spv"));

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo
        {
            .stage               = vk::ShaderStageFlagBits::eVertex,
            .module              = shaderModule,
            .pName               = "vertMain", // the entrypoint in the slang code
            .pSpecializationInfo = nullptr     // used to set constants in shader per-pipeline
        };

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo
        {
            .stage  = vk::ShaderStageFlagBits::eFragment,
            .module = shaderModule,
            .pName  = "fragMain"
        };

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        /* INPUT STAGE SETUP */

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo
        {
            .vertexBindingDescriptionCount   = 1,
            .pVertexBindingDescriptions      = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions    = attributeDescriptions.data()
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly
        {
            .topology = vk::PrimitiveTopology::eTriangleList
        };

        vk::PipelineViewportStateCreateInfo viewportState
        {
            .viewportCount = 1,
            .scissorCount  = 1,
        };

        std::array dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynamicState
        {
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates    = dynamicStates.data()
        };

        /* RASTERIZATION STAGE SETUP */

        vk::PipelineRasterizationStateCreateInfo rasterizer
        {
            .depthClampEnable        = vk::False, // if true, fragments past the near or far plane will be clamped rather than discarded
            .rasterizerDiscardEnable = vk::False, // if true, skips rasterizer stage
            .polygonMode             = vk::PolygonMode::eFill,
            .cullMode                = vk::CullModeFlagBits::eBack,
            .frontFace               = vk::FrontFace::eClockwise,
            .depthBiasEnable         = vk::False, // if true, rasterizer can make adjustments to depth values
            .lineWidth               = 1.0f
        };

        vk::PipelineMultisampleStateCreateInfo multisampling
        {
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable  = vk::False
        };

        /* COLOR BLENDING STAGE SETUP */

        // linearly interpolated blending
        vk::PipelineColorBlendAttachmentState colorBlendAttachment
        {
            .blendEnable         = vk::True,
            .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
            .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
            .colorBlendOp        = vk::BlendOp::eAdd,
            .srcAlphaBlendFactor = vk::BlendFactor::eOne,
            .dstAlphaBlendFactor = vk::BlendFactor::eZero,
            .alphaBlendOp        = vk::BlendOp::eAdd,
            .colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending
        {
            .logicOpEnable   = vk::False,
            .logicOp         = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments    = &colorBlendAttachment
        };

        /* PIPELINE SETUP */

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo
        {
            .setLayoutCount         = 0,
            .pushConstantRangeCount = 0
        };
        pipelineLayout = vk::raii::PipelineLayout(logicalDevice, pipelineLayoutInfo);

        vk::StructureChain pipelineCreateInfoChain = {
            vk::GraphicsPipelineCreateInfo
            {
                .stageCount          = 2,
                .pStages             = shaderStages,
                .pVertexInputState   = &vertexInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState      = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState   = &multisampling,
                .pColorBlendState    = &colorBlending,
                .pDynamicState       = &dynamicState,
                .layout              = pipelineLayout,
                .renderPass          = nullptr // using dynamic rendering
            },

            vk::PipelineRenderingCreateInfo
            {
                .colorAttachmentCount    = 1,
                .pColorAttachmentFormats = &swapchainSurfaceFormat.format
            }
        };

        graphicsPipeline = vk::raii::Pipeline(logicalDevice, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char> &code) const
    {
        vk::ShaderModuleCreateInfo createInfo
        {
            .codeSize = code.size() * sizeof(char),
            .pCode    = reinterpret_cast<const uint32_t*>(code.data())
        };

        vk::raii::ShaderModule shaderModule {logicalDevice, createInfo};
        return shaderModule;
    }

    void createCommandPool()
    {
        vk::CommandPoolCreateInfo poolInfo
        {
            .flags            = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = queueIdx,
        };
        commandPool = vk::raii::CommandPool(logicalDevice, poolInfo);
    }

    void createVertexBuffer()
    {
        vk::BufferCreateInfo bufferInfo
        {
            .size        = sizeof(vertices[0]) * vertices.size(),
            .usage       = vk::BufferUsageFlagBits::eVertexBuffer,
            .sharingMode = vk::SharingMode::eExclusive,
        };

        vertexBuffer = vk::raii::Buffer(logicalDevice, bufferInfo);

        vk::MemoryRequirements memRequirements = vertexBuffer.getMemoryRequirements();

        vk::MemoryAllocateInfo memoryAllocateInfo
        {
            .allocationSize  = memRequirements.size,
            .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
        };

        vertexBufferMemory = vk::raii::DeviceMemory(logicalDevice, memoryAllocateInfo);

        vertexBuffer.bindMemory(*vertexBufferMemory, 0);

        void* data = vertexBufferMemory.mapMemory(0, bufferInfo.size);
        memcpy(data, vertices.data(), bufferInfo.size);
        vertexBufferMemory.unmapMemory();
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        vk::PhysicalDeviceMemoryProperties memProperties = physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
        {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }

    void createCommandBuffers()
    {
        vk::CommandBufferAllocateInfo allocInfo
        {
            .commandPool        = commandPool,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = maxFramesInFlight,
        };

        commandBuffers = vk::raii::CommandBuffers(logicalDevice, allocInfo);
    }

    void recordCommandBuffer(uint32_t imageIdx)
    {
        vk::raii::CommandBuffer& commandBuffer = commandBuffers[frameIdx];

        commandBuffer.begin(vk::CommandBufferBeginInfo{});

        // changing image layout from undefined to color attachment optimal
        transitionImageLayout(
            imageIdx,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput
        );

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0);
        vk::RenderingAttachmentInfo attachmentInfo = 
        {
            .imageView   = swapchainImageViews[imageIdx], // rendering to this image in the swapchain
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = clearColor
        };

        vk::RenderingInfo renderingInfo = 
        {
            .renderArea = {
                .offset = {0, 0},
                .extent = swapchainExtent
            },
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &attachmentInfo,
        };

        commandBuffer.beginRendering(renderingInfo);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);

        commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapchainExtent.width), static_cast<float>(swapchainExtent.height), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));

        commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});

        commandBuffer.draw(static_cast<uint32_t>(vertices.size()), 1, 0, 0);

        commandBuffer.endRendering();

        transitionImageLayout(
            imageIdx,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe
        );

        commandBuffer.end();
    }

    void transitionImageLayout (
            uint32_t imageIdx,
            vk::ImageLayout oldLayout,
            vk::ImageLayout newLayout,
            vk::AccessFlags2 oldAccessMask,
            vk::AccessFlags2 newAccessMask,
            vk::PipelineStageFlags2 oldStageMask,
            vk::PipelineStageFlags2 newStageMask)
    {
        vk::ImageMemoryBarrier2 barrier = {
            .srcStageMask        = oldStageMask,
            .srcAccessMask       = oldAccessMask,
            .dstStageMask        = newStageMask,
            .dstAccessMask       = newAccessMask,
            .oldLayout           = oldLayout,
            .newLayout           = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = swapchainImages[imageIdx],
            .subresourceRange    = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
        };

        vk::DependencyInfo dependencyInfo = {
            .dependencyFlags         = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrier,
        };

        commandBuffers[frameIdx].pipelineBarrier2(dependencyInfo);
    }

    void createSyncObjects()
    {
        assert(presentCompleteSphrs.empty() && renderFinishedSphrs.empty() && drawFences.empty());

        for (int i = 0; i < swapchainImages.size(); i++)
        {
            renderFinishedSphrs.emplace_back(logicalDevice, vk::SemaphoreCreateInfo{});
        }

        for (int i = 0; i < maxFramesInFlight; i++)
        {
            presentCompleteSphrs.emplace_back(logicalDevice, vk::SemaphoreCreateInfo{});
            drawFences.emplace_back(logicalDevice, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        }
    }
};
