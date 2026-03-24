#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include "vulkan/vulkan.hpp"
#include <vulkan/vulkan_raii.hpp>
#include <SDL3/SDL.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

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

class HelloTriangleApplication
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

    std::vector<const char*> requiredDeviceExtension = {
        vk::KHRSwapchainExtensionName
    };

    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

    vk::raii::Context           context;
    vk::raii::Instance          instance        = nullptr;
    vk::raii::PhysicalDevice    physicalDevice  = nullptr;
    vk::raii::Device            device          = nullptr;
    vk::raii::Queue             graphicsQueue   = nullptr;

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
        pickPhysicalDevice();
        createLogicalDevice();
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

    // VKAPI_ATTR, VKAPI_CALL gives the function a signature that vulkan can call
    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity, 
                                                          vk::DebugUtilsMessageTypeFlagsEXT type, 
                                                          const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, 
                                                          void *pUserData)
    {
        std::cerr << "(Validation layer): severity - " << to_string(severity) << "type - " << to_string(type) << " | msg: " << pCallbackData->pMessage << std::endl;

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
            .messageType = messageTypeFlags,
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
            .pApplicationName = "Learn Vulkan",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion = vk::ApiVersion14
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
            .pApplicationInfo = &appInfo,
            .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
            .ppEnabledLayerNames = requiredLayers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data()
        };

        instance = vk::raii::Instance(context, createInfo);
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
        bool supportsAllRequiredExtensions = std::ranges::all_of(requiredDeviceExtension,
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
        // CHECKING IF DEVICES MEET REQUIREMENTS
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

        auto graphicsQueueFamilyProperty = std::ranges::find_if(queueFamilyProperties,
            [](const auto& queueFamilyProp)
            {
                return (queueFamilyProp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
            });

        auto graphicsIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));

        float queuePriority = 0.5f; // priority for scheduling of command buffer execution, needed even if there is one queue
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo
        {
            .queueFamilyIndex = graphicsIndex,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority
        };

        
        // enabling features
        // structure chain connects each feature struct with pointers, making moving through them easyty
        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain = {
            {},                               // physical device features 2 - empty
            {.dynamicRendering = true},       // vulkan 1.3 features - enable dynamic rendering
            {.extendedDynamicState = true}    // extended dynamic state - enable extension
        };

        vk::DeviceCreateInfo deviceCreateInfo
        {
            .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(), // connecting the chain of features to vulkan
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &deviceQueueCreateInfo,
            .enabledExtensionCount = static_cast<uint32_t>(requiredDeviceExtension.size()),
            .ppEnabledExtensionNames = requiredDeviceExtension.data()
        };

        device = vk::raii::Device(physicalDevice, deviceCreateInfo);
        graphicsQueue = vk::raii::Queue(device, graphicsIndex, 0);
    }
};

int main()
{
    try
    {
        HelloTriangleApplication app;
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

