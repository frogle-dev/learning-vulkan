#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include "vulkan/vulkan.hpp"
#include <vulkan/vulkan_raii.hpp>
#include <SDL3/SDL.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

constexpr uint32_t width = 960;
constexpr uint32_t height = 540;

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

class HelloTriangleApplication
{
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    vk::raii::Context context;
    vk::raii::Instance instance = nullptr;

    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(width, height, "Learn Vulkan", nullptr, nullptr);
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

    void initVulkan()
    {
        createInstance();
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

