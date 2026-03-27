#pragma once

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include "vulkan/vulkan.hpp"
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <assert.h>
#include <algorithm>
#include <filesystem>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <fstream>

constexpr uint32_t width = 960;
constexpr uint32_t height = 540;

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

inline std::string path(const char* path) 
{
    return (std::filesystem::canonical("/proc/self/exe").parent_path() / path).string();
}

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
    GLFWwindow* window = nullptr;

    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

    std::vector<const char*> requiredDeviceExtensions = {
        vk::KHRSwapchainExtensionName
    };

    vk::raii::Context        context;
    vk::raii::Instance       instance        = nullptr;
    vk::raii::PhysicalDevice physicalDevice  = nullptr;
    vk::raii::Device         logicalDevice   = nullptr;
    vk::raii::Queue          queue           = nullptr;

    vk::raii::SurfaceKHR     windowSurface   = nullptr;  // surface to render to window

    vk::raii::SwapchainKHR           swapchain       = nullptr;
    std::vector<vk::Image>           swapchainImages;
    vk::SurfaceFormatKHR             swapchainSurfaceFormat;
    vk::Extent2D                     swapchainExtent;
    std::vector<vk::raii::ImageView> swapchainImageViews;

    /* APPLICATION LIFETIME METHODS */

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(width, height, "Learn Vulkan", nullptr, nullptr);
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
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window)) {
            // std::cout << "hello, vulkan!" << std::endl;
            glfwPollEvents();
        }
    }

    void cleanup()
    {
        glfwDestroyWindow(window);

        glfwTerminate();
    }

    /* SETUP METHODS */

    static std::vector<char> readFile(const std::string &fileName)
    {
        // std::ios::ate - reading starts at the end of file
        // std::ios::binary - reads file as a binary
        std::ifstream fin(fileName, std::ios::ate | std::ios::binary);

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
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, 
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
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
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
        std::vector<char const*> requiredLayers;
        if (enableValidationLayers) 
        {
            requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        // check if validation layers are available
        auto layerProperties = context.enumerateInstanceLayerProperties();
        auto unsupportedLayerIterator = std::ranges::find_if(requiredLayers, // returns iterator to first layer that is not supported
            [&layerProperties](const auto& requiredLayer) 
            {
                return std::ranges::none_of(layerProperties, // check if any required layers are missing from the instance's supported layers
                    [requiredLayer](const auto& layerProperty) 
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
        auto requiredExtensions = getRequiredInstanceExtensions();

        auto extensionProperties = context.enumerateInstanceExtensionProperties();
        auto unsupportedPropertyIterator = std::ranges::find_if(requiredExtensions,
            [&extensionProperties](const auto& requiredExtension) 
            {
                return std::ranges::none_of(extensionProperties,
                    [requiredExtension](const auto& extensionProperty) 
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
        VkSurfaceKHR c_api_Surface; // glfw only handles vulkan's C api, so a vulkan C surface is needed

        if (glfwCreateWindowSurface(*instance, window, nullptr, &c_api_Surface) != 0)
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
        auto queueFamilies = physicalDevice.getQueueFamilyProperties();
        bool supportsGraphics = std::ranges::any_of(queueFamilies, 
            [](const auto& queueFamilyProp)
            {
                return static_cast<bool>(queueFamilyProp.queueFlags & vk::QueueFlagBits::eGraphics);
            });

        // if supports specific extensions
        auto availableDeviceExtensions = physicalDevice.enumerateDeviceExtensionProperties();

        // if any of the required device extensions aren't available -> false
        bool supportsAllRequiredExtensions = std::ranges::all_of(requiredDeviceExtensions,
            [&availableDeviceExtensions](const auto& requiredDeviceExtension)
            {
                return std::ranges::any_of(availableDeviceExtensions,
                    [requiredDeviceExtension](const auto& availableDeviceExtension)
                    {
                        return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
                    });
            });

        // if supports specific features
        auto features = physicalDevice.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
        bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                        features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

        return supportsVulkan1_3 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
    }

    void pickPhysicalDevice()
    {
        // checking if physical devices meet requirements

        auto physicalDevices = instance.enumeratePhysicalDevices();

        // find if a GPU meets all the requirements
        const auto deviceIterator = std::ranges::find_if(physicalDevices,
            [&](const auto &physDevice)
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
        std::optional<uint32_t> queueIndex;
        for (uint32_t queueFamilyPropIdx = 0; queueFamilyPropIdx < queueFamilyProperties.size(); queueFamilyPropIdx++)
        {
            if ((queueFamilyProperties[queueFamilyPropIdx].queueFlags & vk::QueueFlagBits::eGraphics) && physicalDevice.getSurfaceSupportKHR(queueFamilyPropIdx, *windowSurface))
            {
                queueIndex = queueFamilyPropIdx;
                break;
            }
        }

        if (!queueIndex.has_value())
        {
            throw std::runtime_error("Graphics and Present queue families not found");
        }

        // getting features
        // structure chain connects each feature struct with pointers, making moving through them easyty
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain = {
            {},                               // physical device features 2 - empty
            {.dynamicRendering = true},       // vulkan 1.3 features - enable dynamic rendering
            {.extendedDynamicState = true}    // extended dynamic state - enable extension
        };

        float queuePriority = 0.5f; // priority for scheduling of command buffer execution, needed even if there is one queue
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo
        {
            .queueFamilyIndex = queueIndex.value(),
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
        queue = vk::raii::Queue(logicalDevice, queueIndex.value(), 0);
    }

    vk::SurfaceFormatKHR chooseSwapchainSurfaceFormat(const std::vector<vk::SurfaceFormatKHR> &availableFormats)
    {
        assert(!availableFormats.empty());

        const auto formatIterator = std::ranges::find_if(availableFormats,
            [](const auto &format)
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
            [](auto presentMode)
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
        glfwGetFramebufferSize(window, &width, &height);

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

            .oldSwapchain     = nullptr
        };

        swapchain = vk::raii::SwapchainKHR(logicalDevice, swapchainCreateInfo);
        swapchainImages = swapchain.getImages();
    }

    void createImageViews()
    {
        assert(swapchainImageViews.empty());

        vk::ImageViewCreateInfo imageViewCreateInfo
        {
            .viewType = vk::ImageViewType::e2D,
            .format = swapchainSurfaceFormat.format,
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
        std::vector<char> shaderCode = readFile(path("shaders/slang.spv"));
    }
};
