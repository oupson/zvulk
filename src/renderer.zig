const std = @import("std");
const builtin = @import("builtin");

const log = std.log.scoped(.renderer);

const vulkan = @cImport({
    @cInclude("vulkan/vulkan.h");
});

const validationLayerName: [1][:0]const u8 = .{
    "VK_LAYER_KHRONOS_validation",
};

const Chunk = @import("chunk.zig");
const TextureManager = @import("texture_manager.zig");
const BrickMap = @import("brickmap.zig");

const deviceExtensions: [3]*align(1) const [:0]u8 = .{
    @ptrCast(vulkan.VK_KHR_SWAPCHAIN_EXTENSION_NAME),
    @ptrCast(vulkan.VK_KHR_MAINTENANCE_3_EXTENSION_NAME),
    @ptrCast(vulkan.VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME),
};

const requiredExtensions: [3]*align(1) const [:0]u8 = .{
    @ptrCast(vulkan.VK_KHR_SURFACE_EXTENSION_NAME),
    @ptrCast(vulkan.VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME),
    @ptrCast(vulkan.VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME),
};

const MAILBOX: bool = false;
const MAX_FRAMES_IN_FLIGHT: u32 = 2;
const ZFAR = 1000.0;

const Allocator = std.mem.Allocator;
const Self = @This();

pub const Camera = struct {
    yaw: f32 = 0,
    pitch: f32 = 0,
    x: f32 = 0.0,
    y: f32 = 1.0,
    z: f32 = 0.0,
};

pub const Instance = struct {
    instance: vulkan.VkInstance,

    pub fn init(allocator: Allocator, platformRequiredExtensions: []const *align(1) const [:0]u8) !@This() {
        const instance = try createInstance(allocator, platformRequiredExtensions);
        return @This(){
            .instance = instance,
        };
    }

    pub fn deinit(self: *@This()) void {
        std.log.debug("deinit instance", .{});
        vulkan.vkDestroyInstance(self.instance, null);
    }
};

const Vertex = struct {
    // TODO: 2D
    pos: @Vector(3, f32),

    fn getBindingDescription() vulkan.VkVertexInputBindingDescription {
        var bindingDescription = vulkan.VkVertexInputBindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = @sizeOf(@This());
        bindingDescription.inputRate = vulkan.VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    fn getAttributeDescriptions() [1]vulkan.VkVertexInputAttributeDescription {
        var attributeDescriptions: [1]vulkan.VkVertexInputAttributeDescription = .{.{}};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vulkan.VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = @offsetOf(Vertex, "pos");
        return attributeDescriptions;
    }
};

const ROW_LENGTH = 64;
const CHUNK_SIZE = ROW_LENGTH * ROW_LENGTH * ROW_LENGTH;

const UniformBufferObject = extern struct {
    camera_pos: @Vector(4, f32) align(4),
    camera_rot: @Vector(2, f32) align(2),
    resolution: @Vector(2, f32) align(2),
};

const VoxelsBuffer = extern struct {
    voxels: [CHUNK_SIZE * 10 * 10 * 10]u32 = undefined,
};

const Texture = struct {
    image: vulkan.VkImage,
    imageView: vulkan.VkImageView,
    memory: vulkan.VkDeviceMemory,

    fn deinit(self: *@This(), device: vulkan.VkDevice) void {
        vulkan.vkDestroyImageView(device, self.imageView, null);
        vulkan.vkDestroyImage(device, self.image, null);
        vulkan.vkFreeMemory(device, self.memory, null);
    }
};

const Swapchain = struct {
    device: vulkan.VkDevice,
    allocator: Allocator,
    swapChain: vulkan.VkSwapchainKHR,
    extent: vulkan.VkExtent2D,
    imageViewList: []vulkan.VkImageView,
    renderPass: vulkan.VkRenderPass,
    pipelineLayout: vulkan.VkPipelineLayout,
    pipeline: vulkan.VkPipeline,
    swapChainFramebuffers: []vulkan.VkFramebuffer,
    submitSemaphores: []vulkan.VkSemaphore,
    depthImage: vulkan.VkImage, // TODO: remove ?
    depthImageMemory: vulkan.VkDeviceMemory,
    depthImageView: vulkan.VkImageView,
    colorImage: vulkan.VkImage,
    colorImageMemory: vulkan.VkDeviceMemory,
    colorImageView: vulkan.VkImageView,

    pub fn create(
        allocator: Allocator,
        device: vulkan.VkDevice,
        physicalDevice: vulkan.VkPhysicalDevice,
        msaaSamples: vulkan.VkSampleCountFlagBits,
        surface: vulkan.VkSurfaceKHR,
        descriptorSetLayout: vulkan.VkDescriptorSetLayout,
        commandPool: vulkan.VkCommandPool,
        graphicQueue: vulkan.VkQueue,
        width: i32,
        height: i32,
    ) !@This() {
        log.debug("creating", .{});
        const swapChain, const format, const extent = try createSwapChain(
            allocator,
            device,
            physicalDevice,
            surface,
            width,
            height,
        );

        const imageList = try getSwapchainImages(allocator, device, swapChain);
        defer allocator.free(imageList);

        const imageViewList = try getImageViewList(allocator, device, imageList, format);

        const renderPass = try createRenderPass(device, physicalDevice, format.format, msaaSamples);
        const pipelineLayout, const pipeline = try createGraphicPipeline(
            allocator,
            device,
            extent,
            renderPass,
            descriptorSetLayout,
            msaaSamples,
        );

        const colorImage, const colorImageMemory, const colorImageView = try createColorResources(
            device,
            physicalDevice,
            extent,
            format.format,
            msaaSamples,
        );

        const depthImage, const depthImageMemory, const depthImageView = try createDepthResources(
            device,
            physicalDevice,
            commandPool,
            graphicQueue,
            extent,
            msaaSamples,
        );

        const swapChainFramebuffers = try createFramebuffers(
            allocator,
            device,
            imageViewList,
            renderPass,
            extent,
            depthImageView,
            colorImageView,
        );

        const submitSemaphores = try createSubmitSemaphores(allocator, device, imageViewList.len);

        return @This(){
            .device = device,
            .allocator = allocator,
            .swapChain = swapChain,
            .extent = extent,
            .imageViewList = imageViewList,
            .renderPass = renderPass,
            .pipelineLayout = pipelineLayout,
            .pipeline = pipeline,
            .swapChainFramebuffers = swapChainFramebuffers,
            .submitSemaphores = submitSemaphores,
            .depthImage = depthImage,
            .depthImageMemory = depthImageMemory,
            .depthImageView = depthImageView,
            .colorImage = colorImage,
            .colorImageMemory = colorImageMemory,
            .colorImageView = colorImageView,
        };
    }

    pub fn deinit(self: *@This()) void {
        vulkan.vkDestroyImageView(self.device, self.colorImageView, null);
        vulkan.vkDestroyImage(self.device, self.colorImage, null);
        vulkan.vkFreeMemory(self.device, self.colorImageMemory, null);

        vulkan.vkDestroyImageView(self.device, self.depthImageView, null);
        vulkan.vkDestroyImage(self.device, self.depthImage, null);
        vulkan.vkFreeMemory(self.device, self.depthImageMemory, null);

        for (self.submitSemaphores) |semaphore| {
            vulkan.vkDestroySemaphore(self.device, semaphore, null);
        }
        self.allocator.free(self.submitSemaphores);

        for (self.swapChainFramebuffers) |framebuffer| {
            vulkan.vkDestroyFramebuffer(self.device, framebuffer, null);
        }
        self.allocator.free(self.swapChainFramebuffers);

        vulkan.vkDestroyPipeline(self.device, self.pipeline, null);
        vulkan.vkDestroyPipelineLayout(self.device, self.pipelineLayout, null);
        vulkan.vkDestroyRenderPass(self.device, self.renderPass, null);

        for (self.imageViewList) |imageView| {
            vulkan.vkDestroyImageView(self.device, imageView, null);
        }
        self.allocator.free(self.imageViewList);

        vulkan.vkDestroySwapchainKHR(self.device, self.swapChain, null);
    }
};

allocator: Allocator,
instance: vulkan.VkInstance,
vulkanSurface: vulkan.VkSurfaceKHR,
device: vulkan.VkDevice,
physicalDevice: vulkan.VkPhysicalDevice,
msaaSamples: vulkan.VkSampleCountFlagBits,
graphicQueue: vulkan.VkQueue,
presentQueue: vulkan.VkQueue,
swapChain: ?Swapchain,
commandPool: vulkan.VkCommandPool,
vertexBuffer: vulkan.VkBuffer,
vertexBufferMemory: vulkan.VkDeviceMemory,
indexBuffer: vulkan.VkBuffer,
indexBufferMemory: vulkan.VkDeviceMemory,
commandBuffers: []vulkan.VkCommandBuffer,
imageAvailableSemaphores: []vulkan.VkSemaphore,
renderFinishedSemaphores: []vulkan.VkSemaphore,
inFlightFences: []vulkan.VkFence,
currentFrame: usize = 0,
descriptorSetLayout: vulkan.VkDescriptorSetLayout,
uniformBuffers: []vulkan.VkBuffer,
uniformBuffersMemory: []vulkan.VkDeviceMemory,
uniformBuffersMapped: [][]u8,
descriptorPool: vulkan.VkDescriptorPool,
descriptorSets: []vulkan.VkDescriptorSet,
textures: std.MultiArrayList(Texture),
textureSamplers: std.ArrayListUnmanaged(vulkan.VkSampler), // TODO: avoid create one sampler per texture
indexCount: u32,
voxelsBuffers: []vulkan.VkBuffer,
voxelsBuffersMemory: []vulkan.VkDeviceMemory,
voxelsBuffersMapped: [][]u8,
textureInfoBuffers: []vulkan.VkBuffer,
textureInfoBuffersMemory: []vulkan.VkDeviceMemory,
textureInfoBuffersMapped: [][]u8,

pub fn new(
    W: type,
    vulkanInstance: Instance,
    allocator: Allocator,
    textureManager: TextureManager,
    wsi: W,
) !Self {
    log.debug("new", .{});
    const instance = vulkanInstance.instance;
    const vulkanSurface: vulkan.VkSurfaceKHR = @ptrCast(try wsi.createVulkanSurface(@ptrCast(instance)));

    const physicalDevice, const msaaSamples = try getPhysicalDevice(instance, allocator, vulkanSurface);

    const familyIndice = try findQueueFamilyIndice(physicalDevice, allocator, vulkanSurface);

    const device = try createDevice(physicalDevice, familyIndice);

    var graphicQueue: vulkan.VkQueue = null;
    vulkan.vkGetDeviceQueue(device, familyIndice.graphics.?, 0, &graphicQueue);
    var presentQueue: vulkan.VkQueue = null;
    vulkan.vkGetDeviceQueue(device, familyIndice.present.?, 0, &presentQueue);

    const textureCount = @as(u32, @intCast(textureManager.getTextureCount()));

    const commandPool = try createCommandPool(allocator, physicalDevice, device, vulkanSurface);

    var textures = std.MultiArrayList(Texture).empty;

    var textureIndexList = try allocator.alloc(
        u32,
        textureManager.getVoxelCount() * 6,
    );
    defer allocator.free(textureIndexList);

    // TODO: Check if multiple sampler is a good idea
    var textureSamplers = std.ArrayListUnmanaged(vulkan.VkSampler).empty;

    {
        var index = @as(usize, 0);
        var textureIndex = @as(u32, 0);
        while (textureManager.getVoxelInfo(index)) |info| : (index += 1) {
            for (info.pixels) |pixels| {
                const textureImage, const textureImageMemory, const mipLevels = try createTextureImage(
                    device,
                    physicalDevice,
                    commandPool,
                    graphicQueue,
                    pixels,
                    info.width,
                    info.height,
                    info.channels,
                );
                const textureImageView = try createTextureImageView(device, textureImage, mipLevels);
                const textureSampler = try createTextureSampler(device, physicalDevice, mipLevels);

                try textures.append(allocator, .{
                    .image = textureImage,
                    .imageView = textureImageView,
                    .memory = textureImageMemory,
                });
                try textureSamplers.append(allocator, textureSampler);
            }
            textureIndex = info.fillIndex(textureIndexList[index * 6 ..][0..6], textureIndex);
        }
    }

    const vertex = [_]Vertex{
        .{ .pos = .{ -1.0, -1.0, 0.0 } },
        .{ .pos = .{ 1.0, -1.0, 0.0 } },
        .{ .pos = .{ 1.0, 1.0, 0.0 } },
        .{ .pos = .{ -1.0, 1.0, 0.0 } },
    };

    const index = [_]u32{
        3, 1, 0, 3, 2, 1,
    };

    const vertexBuffer, const vertexBufferMemory = try createVertexBuffer(
        device,
        physicalDevice,
        commandPool,
        graphicQueue,
        @ptrCast(@constCast(&vertex)),
    );
    const indexBuffer, const indexBufferMemory = try createIndexBuffer(
        device,
        physicalDevice,
        commandPool,
        graphicQueue,
        @ptrCast(@constCast(&index)),
    );
    const uniformBuffers, const uniformBuffersMemory, const uniformBuffersMapped = try createUniformBuffers(
        UniformBufferObject,
        allocator,
        device,
        physicalDevice,
        vulkan.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    );

    // TODO: one time transfer so don't keep mapping / change location
    const voxelsBuffers, const voxelsBuffersMemory, const voxelsBuffersMapped = try createUniformBuffers(
        VoxelsBuffer,
        allocator,
        device,
        physicalDevice,
        vulkan.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    );

    const textureInfoBuffers, const textureInfoBuffersMemory, const textureInfoBuffersMapped = try createUniformBuffersWithSize(
        @sizeOf(u32) * textureIndexList.len,
        allocator,
        device,
        physicalDevice,
        vulkan.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    );

    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        @memcpy(@as([]u32, @ptrCast(@alignCast(textureInfoBuffersMapped[i]))), textureIndexList);
        vulkan.vkUnmapMemory(device, textureInfoBuffersMemory[i]);
    }

    const descriptorSetLayout = try createDescriptorSetLayout(device, textureCount);

    const descriptorPool = try createDescriptorPool(device, textureCount);
    const descriptorSets = try createDescriptorSet(
        allocator,
        device,
        descriptorPool,
        uniformBuffers,
        descriptorSetLayout,
        textures.items(.imageView),
        textureSamplers.items,
        voxelsBuffers,
        textureInfoBuffers,
        @intCast(textureIndexList.len),
    );

    const commandBuffers = try createCommandBuffers(allocator, device, commandPool);
    const imageAvailableSemaphores, const renderFinishedSemaphores, const inFlightFences = try createSyncObjects(allocator, device);
    return Self{
        .allocator = allocator,
        .instance = instance,
        .vulkanSurface = vulkanSurface,
        .device = device,
        .physicalDevice = physicalDevice,
        .msaaSamples = msaaSamples,
        .graphicQueue = graphicQueue,
        .presentQueue = presentQueue,
        .swapChain = null,
        .commandPool = commandPool,
        .vertexBuffer = vertexBuffer,
        .vertexBufferMemory = vertexBufferMemory,
        .indexBuffer = indexBuffer,
        .indexBufferMemory = indexBufferMemory,
        .uniformBuffers = uniformBuffers,
        .uniformBuffersMemory = uniformBuffersMemory,
        .uniformBuffersMapped = uniformBuffersMapped,
        .commandBuffers = commandBuffers,
        .imageAvailableSemaphores = imageAvailableSemaphores,
        .renderFinishedSemaphores = renderFinishedSemaphores,
        .inFlightFences = inFlightFences,
        .descriptorSetLayout = descriptorSetLayout,
        .descriptorPool = descriptorPool,
        .descriptorSets = descriptorSets,
        .textures = textures,
        .textureSamplers = textureSamplers,
        .indexCount = @intCast(index.len), // TODO : remove
        .voxelsBuffers = voxelsBuffers,
        .voxelsBuffersMapped = voxelsBuffersMapped,
        .voxelsBuffersMemory = voxelsBuffersMemory,
        .textureInfoBuffers = textureInfoBuffers,
        .textureInfoBuffersMapped = textureInfoBuffersMapped,
        .textureInfoBuffersMemory = textureInfoBuffersMemory,
    };
}

pub fn deinit(self: *Self) !void {
    if (vulkan.VK_SUCCESS != vulkan.vkDeviceWaitIdle(self.device)) return error.VulkanError;

    if (self.swapChain) |*swapchain| {
        swapchain.deinit();
    }

    {
        var s = self.textures.slice();
        for (0..s.len) |i| {
            var item = s.get(i);
            item.deinit(self.device);
        }
        self.textures.deinit(self.allocator);
    }

    for (self.textureSamplers.items) |sampler| {
        vulkan.vkDestroySampler(self.device, sampler, null);
    }
    self.textureSamplers.deinit(self.allocator);

    vulkan.vkDestroyBuffer(self.device, self.vertexBuffer, null);
    vulkan.vkFreeMemory(self.device, self.vertexBufferMemory, null);

    vulkan.vkDestroyBuffer(self.device, self.indexBuffer, null);
    vulkan.vkFreeMemory(self.device, self.indexBufferMemory, null);

    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        vulkan.vkDestroyBuffer(self.device, self.uniformBuffers[i], null);
        vulkan.vkFreeMemory(self.device, self.uniformBuffersMemory[i], null);
        vulkan.vkDestroyBuffer(self.device, self.voxelsBuffers[i], null);
        vulkan.vkFreeMemory(self.device, self.voxelsBuffersMemory[i], null);
        vulkan.vkDestroyBuffer(self.device, self.textureInfoBuffers[i], null);
        vulkan.vkFreeMemory(self.device, self.textureInfoBuffersMemory[i], null);
    }
    self.allocator.free(self.uniformBuffers);
    self.allocator.free(self.uniformBuffersMemory);
    self.allocator.free(self.uniformBuffersMapped);
    self.allocator.free(self.voxelsBuffers);
    self.allocator.free(self.voxelsBuffersMemory);
    self.allocator.free(self.voxelsBuffersMapped);
    self.allocator.free(self.textureInfoBuffers);
    self.allocator.free(self.textureInfoBuffersMemory);
    self.allocator.free(self.textureInfoBuffersMapped);

    vulkan.vkDestroyDescriptorSetLayout(self.device, self.descriptorSetLayout, null);
    vulkan.vkDestroyDescriptorPool(self.device, self.descriptorPool, null);

    self.allocator.free(self.descriptorSets);

    for (self.inFlightFences) |fence| {
        vulkan.vkDestroyFence(self.device, fence, null);
    }
    self.allocator.free(self.inFlightFences);

    for (self.renderFinishedSemaphores) |semaphore| {
        vulkan.vkDestroySemaphore(self.device, semaphore, null);
    }
    self.allocator.free(self.renderFinishedSemaphores);

    for (self.imageAvailableSemaphores) |semaphore| {
        vulkan.vkDestroySemaphore(self.device, semaphore, null);
    }
    self.allocator.free(self.imageAvailableSemaphores);

    self.allocator.free(self.commandBuffers);

    vulkan.vkDestroyCommandPool(self.device, self.commandPool, null);

    vulkan.vkDestroyDevice(self.device, null);

    vulkan.vkDestroySurfaceKHR(self.instance, self.vulkanSurface, null);
}

pub fn recreate(self: *Self, width: i32, height: i32) !void {
    if (vulkan.VK_SUCCESS != vulkan.vkDeviceWaitIdle(self.device)) {
        return error.WaitDeviceIdleFailed;
    }
    log.debug("recreate : {}x{}", .{ width, height });

    if (self.swapChain) |*swapchain| {
        swapchain.deinit();
    }

    self.swapChain = try Swapchain.create(
        self.allocator,
        self.device,
        self.physicalDevice,
        self.msaaSamples,
        self.vulkanSurface,
        self.descriptorSetLayout,
        self.commandPool,
        self.graphicQueue,
        width,
        height,
    );
}

pub fn draw(self: *Self, camera: *Camera) !void {
    const swapchain = self.swapChain orelse return error.MissingSwapchain;

    if (vulkan.VK_SUCCESS != vulkan.vkWaitForFences(self.device, 1, &self.inFlightFences[self.currentFrame], vulkan.VK_TRUE, std.math.maxInt(u64))) return error.VulkanError;
    var imageIndex: u32 = 0;

    const resCode = vulkan.vkAcquireNextImageKHR(
        self.device,
        swapchain.swapChain,
        std.math.maxInt(u64),
        self.imageAvailableSemaphores[self.currentFrame],
        null,
        &imageIndex,
    );

    // TODO: proper error handling
    if ((resCode == vulkan.VK_SUBOPTIMAL_KHR and false) or resCode == vulkan.VK_ERROR_OUT_OF_DATE_KHR) {
        return error.RecreateSwapchain;
    } else if (resCode < 0) {
        return error.VulkanError;
    }

    try self.updateUniformBuffer(camera, &swapchain);

    if (vulkan.VK_SUCCESS != vulkan.vkResetFences(self.device, 1, &self.inFlightFences[self.currentFrame])) return error.VulkanError;

    if (vulkan.VK_SUCCESS != vulkan.vkResetCommandBuffer(self.commandBuffers[self.currentFrame], 0)) return error.VulkanError;
    try recordCommandBuffer(
        self.commandBuffers[self.currentFrame],
        swapchain.renderPass,
        swapchain.extent,
        swapchain.swapChainFramebuffers[imageIndex],
        swapchain.pipeline,
        self.vertexBuffer,
        self.indexBuffer,
        swapchain.pipelineLayout,
        &self.descriptorSets[self.currentFrame],
        self.indexCount,
    );

    const waitSemaphores = [_]vulkan.VkSemaphore{self.imageAvailableSemaphores[self.currentFrame]};
    const waitStages = [_]vulkan.VkPipelineStageFlags{vulkan.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    const submitSemaphores = [_]vulkan.VkSemaphore{swapchain.submitSemaphores[imageIndex]};

    var submitInfo = vulkan.VkSubmitInfo{};
    submitInfo.sType = vulkan.VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = waitSemaphores.len;
    submitInfo.pWaitSemaphores = &waitSemaphores;
    submitInfo.pWaitDstStageMask = &waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &self.commandBuffers[
        self.currentFrame
    ];
    submitInfo.signalSemaphoreCount = submitSemaphores.len;
    submitInfo.pSignalSemaphores = &submitSemaphores;

    if (vulkan.VK_SUCCESS != vulkan.vkQueueSubmit(self.graphicQueue, 1, &submitInfo, self.inFlightFences[self.currentFrame])) {
        return error.VulkanError;
    }

    const swapChains = [_]vulkan.VkSwapchainKHR{swapchain.swapChain};

    var presentInfo = vulkan.VkPresentInfoKHR{};
    presentInfo.sType = vulkan.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = submitSemaphores.len;
    presentInfo.pWaitSemaphores = &submitSemaphores;
    presentInfo.swapchainCount = swapChains.len;
    presentInfo.pSwapchains = &swapChains;
    presentInfo.pImageIndices = &imageIndex;
    presentInfo.pResults = null; // Optional

    if (vulkan.VK_SUCCESS != vulkan.vkQueuePresentKHR(self.presentQueue, &presentInfo)) {
        return error.VulkanError;
    }

    self.currentFrame = (self.currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

fn createInstance(allocator: Allocator, platformRequiredExtensions: []const *align(1) const [:0]u8) !vulkan.VkInstance {
    var appInfo = vulkan.VkApplicationInfo{};
    appInfo.sType = vulkan.VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Test Zig";
    appInfo.applicationVersion = vulkan.VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = vulkan.VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = vulkan.VK_API_VERSION_1_0;

    const allRequiredExtensions = try allocator.alloc(*align(1) const [:0]u8, requiredExtensions.len + platformRequiredExtensions.len);
    for (requiredExtensions, 0..) |ext, i| {
        allRequiredExtensions[i] = ext;
    }

    for (platformRequiredExtensions, requiredExtensions.len..) |ext, i| {
        allRequiredExtensions[i] = ext;
    }

    defer allocator.free(allRequiredExtensions);

    var createInfo = vulkan.VkInstanceCreateInfo{};
    createInfo.sType = vulkan.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = 0;
    createInfo.flags = createInfo.flags | vulkan.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    createInfo.enabledExtensionCount = @intCast(allRequiredExtensions.len);
    createInfo.ppEnabledExtensionNames = @ptrCast(allRequiredExtensions.ptr);

    if (builtin.mode == .Debug) {
        const checkValidationLayerSupport = layerSupported: {
            var layerCount: u32 = 0;
            if (vulkan.VK_SUCCESS != vulkan.vkEnumerateInstanceLayerProperties(&layerCount, null)) {
                return error.VulkanError; // todo
            }

            const layerList = try allocator.alloc(vulkan.VkLayerProperties, layerCount);
            defer allocator.free(layerList);

            if (vulkan.VK_SUCCESS != vulkan.vkEnumerateInstanceLayerProperties(&layerCount, layerList.ptr)) {
                return error.VulkanError; // todo
            }

            for (layerList) |layer| {
                if (std.mem.orderZ(u8, @ptrCast(&layer.layerName), validationLayerName[0]) == .eq) {
                    break :layerSupported true;
                }
            }

            break :layerSupported false;
        };

        if (!checkValidationLayerSupport) {
            std.log.warn("missing validation layer", .{});
        } else {
            createInfo.enabledLayerCount = validationLayerName.len;
            createInfo.ppEnabledLayerNames = @ptrCast(&validationLayerName);
        }
    }

    var instance: vulkan.VkInstance = null;
    const result = vulkan.vkCreateInstance(&createInfo, null, &instance);
    if (result != vulkan.VK_SUCCESS) {
        return error.VulkanInitFailed;
    }
    return instance;
}

fn getPhysicalDevice(instance: vulkan.VkInstance, allocator: Allocator, surface: vulkan.VkSurfaceKHR) !struct { vulkan.VkPhysicalDevice, vulkan.VkSampleCountFlagBits } {
    var deviceCount: u32 = 0;
    if (vulkan.VK_SUCCESS != vulkan.vkEnumeratePhysicalDevices(instance, &deviceCount, null)) return error.VulkanError; // todo

    if (deviceCount == 0) {
        return error.NoVulkanDevice;
    }

    const deviceList = try allocator.alloc(vulkan.VkPhysicalDevice, deviceCount);
    defer allocator.free(deviceList);
    if (vulkan.VK_SUCCESS != vulkan.vkEnumeratePhysicalDevices(instance, &deviceCount, @ptrCast(deviceList.ptr))) return error.VulkanError; // todo

    for (deviceList) |device| {
        if (try isDeviceSuitable(device, allocator, surface)) {
            const msaaSamples = getMaxUsableSampleCount(device);
            return .{ device, msaaSamples };
        }
    }

    return error.NoVulkanDevice;
}

// todo: improve on multi gpu setup
fn isDeviceSuitable(device: vulkan.VkPhysicalDevice, allocator: Allocator, surface: vulkan.VkSurfaceKHR) !bool {
    var deviceProperties: vulkan.VkPhysicalDeviceProperties = undefined;
    var deviceFeatures: vulkan.VkPhysicalDeviceFeatures = undefined;
    vulkan.vkGetPhysicalDeviceProperties(device, &deviceProperties);
    vulkan.vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

    if (!(deviceProperties.deviceType == vulkan.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU or deviceProperties.deviceType == vulkan.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)) {
        return false;
    }

    if (!try deviceExtensionSupported(device, allocator)) {
        return false;
    }

    var swapChainSupport = try SwapChainSupportDetails.tryInit(allocator, device, surface);
    defer swapChainSupport.deinit();

    if (swapChainSupport.formats.len == 0 or swapChainSupport.presentModes.len == 0) {
        return false;
    }

    var supportedFeatures = vulkan.VkPhysicalDeviceFeatures{};
    vulkan.vkGetPhysicalDeviceFeatures(device, &supportedFeatures);
    if (supportedFeatures.samplerAnisotropy != vulkan.VK_TRUE) {
        return false;
    }

    return true;
}

fn deviceExtensionSupported(device: vulkan.VkPhysicalDevice, allocator: Allocator) !bool {
    var extensionCount: u32 = 0;
    if (vulkan.VK_SUCCESS != vulkan.vkEnumerateDeviceExtensionProperties(device, null, &extensionCount, null)) return error.VulkanError;

    const availableExtensions = try allocator.alloc(vulkan.VkExtensionProperties, @intCast(extensionCount));
    defer allocator.free(availableExtensions);
    if (vulkan.VK_SUCCESS != vulkan.vkEnumerateDeviceExtensionProperties(
        device,
        null,
        &extensionCount,
        availableExtensions.ptr,
    )) return error.VulkanError;

    for (availableExtensions) |ext| {
        // todo
        if (std.mem.orderZ(u8, @ptrCast(&ext.extensionName), @ptrCast(deviceExtensions[0])) == .eq) {
            return true;
        }
    }

    return false;
}

const QueueFamily = struct {
    graphics: ?u32 = null,
    present: ?u32 = null,
};

fn findQueueFamilyIndice(device: vulkan.VkPhysicalDevice, allocator: Allocator, surface: vulkan.VkSurfaceKHR) !QueueFamily {
    var queueFamilyCount: u32 = 0;
    vulkan.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, null);

    const queueFamilies = try allocator.alloc(vulkan.VkQueueFamilyProperties, queueFamilyCount);
    defer allocator.free(queueFamilies);

    vulkan.vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.ptr);

    var res = QueueFamily{};

    for (queueFamilies, 0..) |family, i| {
        if ((family.queueFlags & vulkan.VK_QUEUE_GRAPHICS_BIT) != 0) {
            res.graphics = @intCast(i);
        }
        var presentSupport: vulkan.VkBool32 = vulkan.VK_FALSE;
        if (vulkan.VK_SUCCESS != vulkan.vkGetPhysicalDeviceSurfaceSupportKHR(device, @intCast(i), surface, &presentSupport)) return error.VulkanFailed;
        if (presentSupport == vulkan.VK_TRUE) {
            res.present = @intCast(i);
        }
    }
    return res;
}

fn createDevice(physicalDevice: vulkan.VkPhysicalDevice, familyIndice: QueueFamily) !vulkan.VkDevice {
    var queuePriority: f32 = 1.0;

    var infos = [2]vulkan.VkDeviceQueueCreateInfo{
        .{
            .sType = vulkan.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
            .queueFamilyIndex = familyIndice.graphics.?, // todo
        },
        .{
            .sType = vulkan.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueCount = 1,
            .pQueuePriorities = &queuePriority,
            .queueFamilyIndex = familyIndice.present.?, // todo
        },
    };

    const count: u32 = if (familyIndice.present.? == familyIndice.graphics.?) 1 else 2;

    var deviceFeatures = vulkan.VkPhysicalDeviceFeatures{};
    deviceFeatures.samplerAnisotropy = vulkan.VK_TRUE;
    deviceFeatures.shaderSampledImageArrayDynamicIndexing = vulkan.VK_TRUE;

    const deviceFeatures12 = vulkan.VkPhysicalDeviceVulkan12Features{
        .sType = vulkan.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .descriptorIndexing = vulkan.VK_TRUE,
        .runtimeDescriptorArray = vulkan.VK_TRUE,
        .shaderSampledImageArrayNonUniformIndexing = vulkan.VK_TRUE,
    };

    var createInfo = vulkan.VkDeviceCreateInfo{};
    createInfo.sType = vulkan.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &deviceFeatures12;

    createInfo.pQueueCreateInfos = &infos;
    createInfo.queueCreateInfoCount = count;

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = deviceExtensions.len;
    createInfo.ppEnabledExtensionNames = @ptrCast(&deviceExtensions);

    // if (enableValidationLayers) {
    //    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    //    createInfo.ppEnabledLayerNames = validationLayers.data();
    // } else {
    //        createInfo.enabledLayerCount = 0;
    //  }
    //
    var device: vulkan.VkDevice = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateDevice(physicalDevice, &createInfo, null, &device)) {
        return error.FailedToCreateDevice;
    }
    return device;
}

const SwapChainSupportDetails = struct {
    allocator: Allocator,
    capabilities: vulkan.VkSurfaceCapabilitiesKHR,
    formats: []vulkan.VkSurfaceFormatKHR,
    presentModes: []vulkan.VkPresentModeKHR,

    fn tryInit(allocator: Allocator, device: vulkan.VkPhysicalDevice, surface: vulkan.VkSurfaceKHR) !@This() {
        var capabilities = vulkan.VkSurfaceCapabilitiesKHR{};
        if (vulkan.VK_SUCCESS != vulkan.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities)) return error.VulkanError;

        var formatCount: u32 = 0;
        if (vulkan.VK_SUCCESS != vulkan.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, null)) return error.VulkanError;
        const formats = try allocator.alloc(vulkan.VkSurfaceFormatKHR, formatCount);
        if (vulkan.VK_SUCCESS != vulkan.vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, formats.ptr)) return error.VulkanError;

        var presentModeCount: u32 = 0;
        if (vulkan.VK_SUCCESS != vulkan.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, null)) return error.VulkanError;
        const presentModes = try allocator.alloc(vulkan.VkPresentModeKHR, presentModeCount);
        if (vulkan.VK_SUCCESS != vulkan.vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, presentModes.ptr)) return error.VulkanError;

        return @This(){
            .allocator = allocator,
            .capabilities = capabilities,
            .formats = formats,
            .presentModes = presentModes,
        };
    }

    fn deinit(self: @This()) void {
        self.allocator.free(self.formats);
        self.allocator.free(self.presentModes);
    }
};

fn createSwapChain(
    allocator: Allocator,
    device: vulkan.VkDevice,
    physicalDevice: vulkan.VkPhysicalDevice,
    surface: vulkan.VkSurfaceKHR,
    width: i32,
    height: i32,
) !struct { vulkan.VkSwapchainKHR, vulkan.VkSurfaceFormatKHR, vulkan.VkExtent2D } {
    const swapChainSupport = try SwapChainSupportDetails.tryInit(
        allocator,
        physicalDevice,
        surface,
    ); // todo
    defer swapChainSupport.deinit();
    const surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    const presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    const extent = chooseSwapExtent(&swapChainSupport.capabilities, width, height);

    var imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 and imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    var createInfo = vulkan.VkSwapchainCreateInfoKHR{};
    createInfo.sType = vulkan.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vulkan.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    const queueIndices = try findQueueFamilyIndice(physicalDevice, allocator, surface);
    const queueFamilyIndices = [_]u32{ queueIndices.graphics.?, queueIndices.present.? };

    if (queueIndices.graphics != queueIndices.present) {
        createInfo.imageSharingMode = vulkan.VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = &queueFamilyIndices;
    } else {
        createInfo.imageSharingMode = vulkan.VK_SHARING_MODE_EXCLUSIVE;
        createInfo.queueFamilyIndexCount = 0; // Optional
        createInfo.pQueueFamilyIndices = null; // Optional
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = vulkan.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    createInfo.presentMode = presentMode;
    createInfo.clipped = vulkan.VK_TRUE;

    createInfo.oldSwapchain = null;

    var swapChain: vulkan.VkSwapchainKHR = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateSwapchainKHR(device, &createInfo, null, &swapChain)) {
        return error.FailedToCreateSwapChain;
    }

    return .{ swapChain, surfaceFormat, extent };
}

fn chooseSwapSurfaceFormat(availableFormats: []vulkan.VkSurfaceFormatKHR) vulkan.VkSurfaceFormatKHR {
    for (availableFormats) |format| {
        if (format.format == vulkan.VK_FORMAT_B8G8R8A8_SRGB and format.colorSpace == vulkan.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    //todo
    return availableFormats[0];
}

fn chooseSwapPresentMode(availablePresentModes: []vulkan.VkPresentModeKHR) vulkan.VkPresentModeKHR {
    for (availablePresentModes) |m| {
        if (m == vulkan.VK_PRESENT_MODE_MAILBOX_KHR and MAILBOX) {
            return m;
        }
    }
    return vulkan.VK_PRESENT_MODE_FIFO_KHR;
}

fn chooseSwapExtent(capabilities: *const vulkan.VkSurfaceCapabilitiesKHR, width: i32, height: i32) vulkan.VkExtent2D {
    // todo
    if (capabilities.currentExtent.width != std.math.maxInt(u32)) {
        return capabilities.currentExtent;
    } else {
        return vulkan.VkExtent2D{
            .width = @intCast(width),
            .height = @intCast(height),
        };
    }
}

fn getSwapchainImages(allocator: Allocator, device: vulkan.VkDevice, swapChain: vulkan.VkSwapchainKHR) ![]vulkan.VkImage {
    var imageCount: u32 = 0;
    if (vulkan.VK_SUCCESS != vulkan.vkGetSwapchainImagesKHR(device, swapChain, &imageCount, null)) return error.VulkanFailed;
    const swapChainImages = try allocator.alloc(vulkan.VkImage, imageCount);
    if (vulkan.VK_SUCCESS != vulkan.vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.ptr)) return error.VulkanFailed;
    return swapChainImages;
}

fn getImageViewList(allocator: Allocator, device: vulkan.VkDevice, images: []vulkan.VkImage, imageFormat: vulkan.VkSurfaceFormatKHR) ![]vulkan.VkImageView {
    var swapChainImageViews = try allocator.alloc(vulkan.VkImageView, images.len);
    for (0..images.len) |i| {
        swapChainImageViews[i] = try createImageView(device, images[i], imageFormat.format, vulkan.VK_IMAGE_ASPECT_COLOR_BIT, 1);
    }

    return swapChainImageViews;
}

fn createGraphicPipeline(
    allocator: Allocator,
    device: vulkan.VkDevice,
    swapChainExtent: vulkan.VkExtent2D,
    renderPass: vulkan.VkRenderPass,
    descriptorSetLayout: vulkan.VkDescriptorSetLayout,
    msaaSamples: vulkan.VkSampleCountFlagBits,
) !struct { vulkan.VkPipelineLayout, vulkan.VkPipeline } {
    const vertShaderCode = @embedFile("shaders/vertex.spv");
    const fragShaderCode = @embedFile("shaders/fragment.spv");

    const vertexModule = try createShaderModule(allocator, device, vertShaderCode);
    defer vulkan.vkDestroyShaderModule(device, vertexModule, null);
    const fragModule = try createShaderModule(allocator, device, fragShaderCode);
    defer vulkan.vkDestroyShaderModule(device, fragModule, null);

    var vertShaderStageInfo = vulkan.VkPipelineShaderStageCreateInfo{};
    vertShaderStageInfo.sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = vulkan.VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertexModule;
    vertShaderStageInfo.pName = "main";

    var fragShaderStageInfo = vulkan.VkPipelineShaderStageCreateInfo{};
    fragShaderStageInfo.sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = vulkan.VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragModule;
    fragShaderStageInfo.pName = "main";

    const shaderStages = [_]vulkan.VkPipelineShaderStageCreateInfo{ vertShaderStageInfo, fragShaderStageInfo };

    const dynamicStates = [_]vulkan.VkDynamicState{
        vulkan.VK_DYNAMIC_STATE_VIEWPORT,
        vulkan.VK_DYNAMIC_STATE_SCISSOR,
    };

    var dynamicState = vulkan.VkPipelineDynamicStateCreateInfo{};
    dynamicState.sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = dynamicStates.len;
    dynamicState.pDynamicStates = &dynamicStates;

    const bindingDescription = Vertex.getBindingDescription();
    const attributeDescriptions = Vertex.getAttributeDescriptions();

    var vertexInputInfo = vulkan.VkPipelineVertexInputStateCreateInfo{};
    vertexInputInfo.sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = attributeDescriptions.len;
    vertexInputInfo.pVertexAttributeDescriptions = &attributeDescriptions;

    var inputAssembly = vulkan.VkPipelineInputAssemblyStateCreateInfo{};
    inputAssembly.sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = vulkan.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = vulkan.VK_FALSE;

    var viewport = vulkan.VkViewport{};
    viewport.x = 0.0;
    viewport.y = 0.0;
    viewport.width = @floatFromInt(swapChainExtent.width);
    viewport.height = @floatFromInt(swapChainExtent.height);
    viewport.minDepth = 0.0;
    viewport.maxDepth = 1.0;

    var scissor = vulkan.VkRect2D{};
    scissor.offset = vulkan.VkOffset2D{ .x = 0, .y = 0 };
    scissor.extent = swapChainExtent;

    var viewportState = vulkan.VkPipelineViewportStateCreateInfo{};
    viewportState.sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    var rasterizer = vulkan.VkPipelineRasterizationStateCreateInfo{};
    rasterizer.sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = vulkan.VK_FALSE;
    rasterizer.polygonMode = vulkan.VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0;
    rasterizer.cullMode = vulkan.VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = vulkan.VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = vulkan.VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0; // Optional
    rasterizer.depthBiasClamp = 0.0; // Optional
    rasterizer.depthBiasSlopeFactor = 0.0; // Optional

    var multisampling = vulkan.VkPipelineMultisampleStateCreateInfo{};
    multisampling.sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = vulkan.VK_FALSE;
    multisampling.rasterizationSamples = msaaSamples;
    multisampling.minSampleShading = 1.0; // Optional
    multisampling.pSampleMask = null; // Optional
    multisampling.alphaToCoverageEnable = vulkan.VK_FALSE; // Optional
    multisampling.alphaToOneEnable = vulkan.VK_FALSE; // Optional

    var colorBlendAttachment = vulkan.VkPipelineColorBlendAttachmentState{};
    colorBlendAttachment.colorWriteMask = vulkan.VK_COLOR_COMPONENT_R_BIT | vulkan.VK_COLOR_COMPONENT_G_BIT | vulkan.VK_COLOR_COMPONENT_B_BIT | vulkan.VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = vulkan.VK_FALSE;
    colorBlendAttachment.srcColorBlendFactor = vulkan.VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachment.dstColorBlendFactor = vulkan.VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.colorBlendOp = vulkan.VK_BLEND_OP_ADD; // Optional
    colorBlendAttachment.srcAlphaBlendFactor = vulkan.VK_BLEND_FACTOR_ONE; // Optional
    colorBlendAttachment.dstAlphaBlendFactor = vulkan.VK_BLEND_FACTOR_ZERO; // Optional
    colorBlendAttachment.alphaBlendOp = vulkan.VK_BLEND_OP_ADD; // Optional

    var colorBlending = vulkan.VkPipelineColorBlendStateCreateInfo{};
    colorBlending.sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = vulkan.VK_FALSE;
    colorBlending.logicOp = vulkan.VK_LOGIC_OP_COPY; // Optional
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0; // Optional
    colorBlending.blendConstants[1] = 0.0; // Optional
    colorBlending.blendConstants[2] = 0.0; // Optional
    colorBlending.blendConstants[3] = 0.0; // Optional

    var pipelineLayoutInfo = vulkan.VkPipelineLayoutCreateInfo{};
    pipelineLayoutInfo.sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
    pipelineLayoutInfo.pPushConstantRanges = null; // Optional

    var pipelineLayout: vulkan.VkPipelineLayout = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreatePipelineLayout(device, &pipelineLayoutInfo, null, &pipelineLayout)) {
        return error.VulkanError;
    }

    var depthStencil = vulkan.VkPipelineDepthStencilStateCreateInfo{};
    depthStencil.sType = vulkan.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = vulkan.VK_TRUE;
    depthStencil.depthWriteEnable = vulkan.VK_TRUE;
    depthStencil.depthCompareOp = vulkan.VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = vulkan.VK_FALSE;
    depthStencil.minDepthBounds = 0.0; // Optional
    depthStencil.maxDepthBounds = 1.0; // Optional
    depthStencil.stencilTestEnable = vulkan.VK_FALSE;
    depthStencil.front = .{}; // Optional
    depthStencil.back = .{}; // Optional

    var pipelineInfo = vulkan.VkGraphicsPipelineCreateInfo{};
    pipelineInfo.sType = vulkan.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = shaderStages.len;
    pipelineInfo.pStages = &shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = null; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional

    var graphicsPipeline: vulkan.VkPipeline = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateGraphicsPipelines(device, null, 1, &pipelineInfo, null, &graphicsPipeline)) {
        return error.VulkanError;
    }

    return .{ pipelineLayout, graphicsPipeline };
}

fn createShaderModule(allocator: Allocator, device: vulkan.VkDevice, code: [:0]const u8) !vulkan.VkShaderModule {
    // todo better way
    const alignedCode = try allocator.alignedAlloc(u8, 32, code.len);
    defer allocator.free(alignedCode);
    @memcpy(alignedCode, code);

    var createInfo = vulkan.VkShaderModuleCreateInfo{};
    createInfo.sType = vulkan.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.len;
    createInfo.pCode = @ptrCast(alignedCode.ptr);

    var module: vulkan.VkShaderModule = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateShaderModule(device, &createInfo, null, &module)) return error.VulkanError;

    return module;
}

fn createRenderPass(device: vulkan.VkDevice, physicalDevice: vulkan.VkPhysicalDevice, swapChainImageFormat: vulkan.VkFormat, msaaSamples: vulkan.VkSampleCountFlagBits) !vulkan.VkRenderPass {
    var colorAttachment = vulkan.VkAttachmentDescription{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = msaaSamples;
    colorAttachment.loadOp = vulkan.VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = vulkan.VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = vulkan.VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = vulkan.VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = vulkan.VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = vulkan.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    var colorAttachmentRef = vulkan.VkAttachmentReference{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vulkan.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // TODO: Remove ?
    var depthAttachment = vulkan.VkAttachmentDescription{};
    depthAttachment.format = try findDepthFormat(physicalDevice);
    depthAttachment.samples = msaaSamples;
    depthAttachment.loadOp = vulkan.VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = vulkan.VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = vulkan.VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = vulkan.VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = vulkan.VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = vulkan.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    var depthAttachmentRef = vulkan.VkAttachmentReference{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vulkan.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    var colorAttachmentResolve = vulkan.VkAttachmentDescription{};
    colorAttachmentResolve.format = swapChainImageFormat;
    colorAttachmentResolve.samples = vulkan.VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = vulkan.VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = vulkan.VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = vulkan.VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = vulkan.VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = vulkan.VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = vulkan.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    var colorAttachmentResolveRef = vulkan.VkAttachmentReference{};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout = vulkan.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    var subpass = vulkan.VkSubpassDescription{};
    subpass.pipelineBindPoint = vulkan.VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;

    var dependency = vulkan.VkSubpassDependency{};
    dependency.srcSubpass = vulkan.VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vulkan.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | vulkan.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = vulkan.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | vulkan.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = vulkan.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | vulkan.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    var attachments = [_]vulkan.VkAttachmentDescription{ colorAttachment, depthAttachment, colorAttachmentResolve };

    var renderPassInfo = vulkan.VkRenderPassCreateInfo{};
    renderPassInfo.sType = vulkan.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = attachments.len;
    renderPassInfo.pAttachments = &attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    var renderPass: vulkan.VkRenderPass = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateRenderPass(device, &renderPassInfo, null, &renderPass)) return error.VulkanError;

    return renderPass;
}

fn createFramebuffers(
    allocator: Allocator,
    device: vulkan.VkDevice,
    swapChainImageViews: []vulkan.VkImageView,
    renderPass: vulkan.VkRenderPass,
    swapChainExtent: vulkan.VkExtent2D,
    depthImageView: vulkan.VkImageView,
    colorImageView: vulkan.VkImageView,
) ![]vulkan.VkFramebuffer {
    const swapChainFramebuffers = try allocator.alloc(vulkan.VkFramebuffer, swapChainImageViews.len);

    for (swapChainImageViews, 0..) |imageView, i| {
        var attachments = [_]vulkan.VkImageView{
            colorImageView,
            depthImageView,
            imageView,
        };

        var framebufferInfo = vulkan.VkFramebufferCreateInfo{};
        framebufferInfo.sType = vulkan.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = attachments.len;
        framebufferInfo.pAttachments = &attachments;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vulkan.VK_SUCCESS != vulkan.vkCreateFramebuffer(device, &framebufferInfo, null, &swapChainFramebuffers[i])) {
            return error.VulkanError;
        }
    }
    return swapChainFramebuffers;
}

fn createCommandPool(allocator: Allocator, physicalDevice: vulkan.VkPhysicalDevice, device: vulkan.VkDevice, surface: vulkan.VkSurfaceKHR) !vulkan.VkCommandPool {
    const queueFamilyIndices = try findQueueFamilyIndice(physicalDevice, allocator, surface);

    var poolInfo = vulkan.VkCommandPoolCreateInfo{};
    poolInfo.sType = vulkan.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = vulkan.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphics.?;

    var commandPool: vulkan.VkCommandPool = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateCommandPool(device, &poolInfo, null, &commandPool)) {
        return error.VulkanError;
    }
    return commandPool;
}

fn createCommandBuffers(allocator: Allocator, device: vulkan.VkDevice, commandPool: vulkan.VkCommandPool) ![]vulkan.VkCommandBuffer {
    const commandBuffers: []vulkan.VkCommandBuffer = try allocator.alloc(vulkan.VkCommandBuffer, MAX_FRAMES_IN_FLIGHT);

    var allocInfo = vulkan.VkCommandBufferAllocateInfo{};
    allocInfo.sType = vulkan.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = vulkan.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = @intCast(commandBuffers.len);

    if (vulkan.VK_SUCCESS != vulkan.vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.ptr)) {
        return error.VulkanError;
    }

    return commandBuffers;
}

fn recordCommandBuffer(
    commandBuffer: vulkan.VkCommandBuffer,
    renderPass: vulkan.VkRenderPass,
    swapChainExtent: vulkan.VkExtent2D,
    swapChainFramebuffer: vulkan.VkFramebuffer,
    graphicsPipeline: vulkan.VkPipeline,
    vertexBuffer: vulkan.VkBuffer,
    indexBuffer: vulkan.VkBuffer,
    pipelineLayout: vulkan.VkPipelineLayout,
    descriptorSet: *vulkan.VkDescriptorSet,
    indexCount: u32,
) !void {
    var beginInfo = vulkan.VkCommandBufferBeginInfo{};
    beginInfo.sType = vulkan.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // Optional
    beginInfo.pInheritanceInfo = null; // Optional

    if (vulkan.VK_SUCCESS != vulkan.vkBeginCommandBuffer(commandBuffer, &beginInfo)) {
        return error.VulkanError;
    }

    var renderPassInfo = vulkan.VkRenderPassBeginInfo{};
    renderPassInfo.sType = vulkan.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffer;
    renderPassInfo.renderArea.offset = .{ .x = 0, .y = 0 };
    renderPassInfo.renderArea.extent = swapChainExtent;

    const clearValues = [_]vulkan.VkClearValue{
        .{ .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } } },
        .{ .depthStencil = .{ .depth = 1.0, .stencil = 0.0 } },
    };
    renderPassInfo.clearValueCount = clearValues.len;
    renderPassInfo.pClearValues = &clearValues;

    vulkan.vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, vulkan.VK_SUBPASS_CONTENTS_INLINE);

    {
        vulkan.vkCmdBindPipeline(commandBuffer, vulkan.VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        var viewport = vulkan.VkViewport{};
        viewport.x = 0.0;
        viewport.y = 0.0;
        viewport.width = @floatFromInt(swapChainExtent.width);
        viewport.height = @floatFromInt(swapChainExtent.height);
        viewport.minDepth = 0.0;
        viewport.maxDepth = 1.0;
        vulkan.vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        var scissor = vulkan.VkRect2D{};
        scissor.offset = .{ .x = 0, .y = 0 };
        scissor.extent = swapChainExtent;
        vulkan.vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        const vertexBuffers = [_]vulkan.VkBuffer{vertexBuffer};
        const offsets = [_]vulkan.VkDeviceSize{0};
        vulkan.vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffers, &offsets);
        vulkan.vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, vulkan.VK_INDEX_TYPE_UINT32);
        vulkan.vkCmdBindDescriptorSets(
            commandBuffer,
            vulkan.VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout,
            0,
            1,
            descriptorSet,
            0,
            null,
        );

        vulkan.vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);
    }
    vulkan.vkCmdEndRenderPass(commandBuffer);

    if (vulkan.VK_SUCCESS != vulkan.vkEndCommandBuffer(commandBuffer)) {
        return error.VulkanError;
    }
}

fn createSyncObjects(allocator: Allocator, device: vulkan.VkDevice) !struct { []vulkan.VkSemaphore, []vulkan.VkSemaphore, []vulkan.VkFence } {
    var semaphoreInfo = vulkan.VkSemaphoreCreateInfo{};
    semaphoreInfo.sType = vulkan.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    var fenceInfo = vulkan.VkFenceCreateInfo{};
    fenceInfo.sType = vulkan.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = vulkan.VK_FENCE_CREATE_SIGNALED_BIT;

    var imageAvailableSemaphores: []vulkan.VkSemaphore = try allocator.alloc(vulkan.VkSemaphore, MAX_FRAMES_IN_FLIGHT);
    var renderFinishedSemaphores: []vulkan.VkSemaphore = try allocator.alloc(vulkan.VkSemaphore, MAX_FRAMES_IN_FLIGHT);
    var inFlightFences: []vulkan.VkFence = try allocator.alloc(vulkan.VkFence, MAX_FRAMES_IN_FLIGHT);

    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        if (vulkan.vkCreateSemaphore(device, &semaphoreInfo, null, &imageAvailableSemaphores[i]) != vulkan.VK_SUCCESS or
            vulkan.vkCreateSemaphore(device, &semaphoreInfo, null, &renderFinishedSemaphores[i]) != vulkan.VK_SUCCESS or
            vulkan.vkCreateFence(device, &fenceInfo, null, &inFlightFences[i]) != vulkan.VK_SUCCESS)
        {
            return error.VulkanError;
        }
    }

    return .{ imageAvailableSemaphores, renderFinishedSemaphores, inFlightFences };
}

fn createSubmitSemaphores(allocator: Allocator, device: vulkan.VkDevice, swapChainCount: usize) ![]vulkan.VkSemaphore {
    var semaphoreInfo = vulkan.VkSemaphoreCreateInfo{
        .sType = vulkan.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    var semaphores = try allocator.alloc(vulkan.VkSemaphore, swapChainCount);
    for (0..swapChainCount) |i| {
        if (vulkan.vkCreateSemaphore(device, &semaphoreInfo, null, &semaphores[i]) != vulkan.VK_SUCCESS) {
            return error.VulkanError;
        }
    }
    return semaphores;
}

pub fn createVertexBuffer(
    device: vulkan.VkDevice,
    physicalDevice: vulkan.VkPhysicalDevice,
    commandPool: vulkan.VkCommandPool,
    graphicQueue: vulkan.VkQueue,
    vertex: []u8,
) !struct { vulkan.VkBuffer, vulkan.VkDeviceMemory } {
    const bufferSize = vertex.len;
    const stagingBuffer, const stagingBufferMemory = try createBuffer(
        device,
        physicalDevice,
        bufferSize,
        vulkan.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        vulkan.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vulkan.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    );
    defer {
        vulkan.vkDestroyBuffer(device, stagingBuffer, null);
        vulkan.vkFreeMemory(device, stagingBufferMemory, null);
    }

    var data: [*c]u8 = undefined;
    if (vulkan.VK_SUCCESS != vulkan.vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, @ptrCast(&data))) {
        return error.MapMemoryFailed;
    }
    @memcpy(data[0..bufferSize], vertex);
    vulkan.vkUnmapMemory(device, stagingBufferMemory);

    const vertexBuffer, const vertexBufferMemory = try createBuffer(
        device,
        physicalDevice,
        bufferSize,
        vulkan.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vulkan.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        vulkan.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    );

    try copyBuffer(
        device,
        graphicQueue,
        commandPool,
        stagingBuffer,
        vertexBuffer,
        @intCast(bufferSize),
    );

    return .{ vertexBuffer, vertexBufferMemory };
}

fn findMemoryType(physicalDevice: vulkan.VkPhysicalDevice, typeFilter: u32, properties: vulkan.VkMemoryPropertyFlags) !u32 {
    var memProperties = vulkan.VkPhysicalDeviceMemoryProperties{};
    vulkan.vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (0..memProperties.memoryTypeCount) |i| {
        if ((typeFilter & @shlExact(i, 1) != 0) and ((memProperties.memoryTypes[i].propertyFlags & properties) == properties)) {
            return @intCast(i);
        }
    }

    return error.FailedToFindMemoryType;
}

fn createBuffer(
    device: vulkan.VkDevice,
    physicalDevice: vulkan.VkPhysicalDevice,
    size: usize,
    usage: vulkan.VkBufferUsageFlags,
    properties: vulkan.VkMemoryPropertyFlags,
) !struct { vulkan.VkBuffer, vulkan.VkDeviceMemory } {
    var bufferInfo = vulkan.VkBufferCreateInfo{};
    bufferInfo.sType = vulkan.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vulkan.VK_SHARING_MODE_EXCLUSIVE;

    var vertexBuffer: vulkan.VkBuffer = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateBuffer(device, &bufferInfo, null, &vertexBuffer)) {
        return error.FailedToCreateBuffer;
    }

    var memRequirements = vulkan.VkMemoryRequirements{};
    vulkan.vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

    var allocInfo = vulkan.VkMemoryAllocateInfo{};
    allocInfo.sType = vulkan.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = try findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    var vertexBufferMemory: vulkan.VkDeviceMemory = null;
    if (vulkan.VK_SUCCESS != vulkan.vkAllocateMemory(device, &allocInfo, null, &vertexBufferMemory)) {
        return error.VulkanAllocMemoryFailed;
    }

    if (vulkan.VK_SUCCESS != vulkan.vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0)) {
        return error.BindBufferFailed;
    }

    return .{ vertexBuffer, vertexBufferMemory };
}

fn copyBuffer(
    device: vulkan.VkDevice,
    graphicQueue: vulkan.VkQueue,
    commandPool: vulkan.VkCommandPool,
    srcBuffer: vulkan.VkBuffer,
    dstBuffer: vulkan.VkBuffer,
    size: u32,
) !void {
    const commandBuffer = try beginSingleTimeCommands(device, commandPool);

    var copyRegion = vulkan.VkBufferCopy{};
    copyRegion.srcOffset = 0; // Optional
    copyRegion.dstOffset = 0; // Optional
    copyRegion.size = size;
    vulkan.vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    try endSingleTimeCommands(device, commandPool, commandBuffer, graphicQueue);
}

fn createIndexBuffer(
    device: vulkan.VkDevice,
    physicalDevice: vulkan.VkPhysicalDevice,
    commandPool: vulkan.VkCommandPool,
    graphicQueue: vulkan.VkQueue,
    index: []u8,
) !struct { vulkan.VkBuffer, vulkan.VkDeviceMemory } {
    const bufferSize = index.len;
    const stagingBuffer, const stagingBufferMemory = try createBuffer(
        device,
        physicalDevice,
        bufferSize,
        vulkan.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        vulkan.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vulkan.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    );
    defer {
        vulkan.vkDestroyBuffer(device, stagingBuffer, null);
        vulkan.vkFreeMemory(device, stagingBufferMemory, null);
    }

    var data: [*c]u8 = undefined;
    if (vulkan.VK_SUCCESS != vulkan.vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, @ptrCast(&data))) {
        return error.MapMemoryFailed;
    }
    @memcpy(data[0..bufferSize], index);
    vulkan.vkUnmapMemory(device, stagingBufferMemory);

    const indexBuffer, const indexBufferMemory = try createBuffer(
        device,
        physicalDevice,
        bufferSize,
        vulkan.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vulkan.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
        vulkan.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    );

    try copyBuffer(
        device,
        graphicQueue,
        commandPool,
        stagingBuffer,
        indexBuffer,
        @intCast(bufferSize),
    );

    return .{ indexBuffer, indexBufferMemory };
}

fn createDescriptorSetLayout(device: vulkan.VkDevice, textureCount: u32) !vulkan.VkDescriptorSetLayout {
    var uboLayoutBinding = vulkan.VkDescriptorSetLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = vulkan.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = vulkan.VK_SHADER_STAGE_FRAGMENT_BIT;
    uboLayoutBinding.pImmutableSamplers = null; // Optional

    var samplerLayoutBinding = vulkan.VkDescriptorSetLayoutBinding{};
    samplerLayoutBinding.binding = 1;
    samplerLayoutBinding.descriptorCount = textureCount;
    samplerLayoutBinding.descriptorType = vulkan.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerLayoutBinding.pImmutableSamplers = null;
    samplerLayoutBinding.stageFlags = vulkan.VK_SHADER_STAGE_FRAGMENT_BIT;

    var voxelBindings = vulkan.VkDescriptorSetLayoutBinding{};
    voxelBindings.binding = 2;
    voxelBindings.descriptorCount = 1;
    voxelBindings.descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    voxelBindings.pImmutableSamplers = null;
    voxelBindings.stageFlags = vulkan.VK_SHADER_STAGE_FRAGMENT_BIT;

    var textureInfosBinding = vulkan.VkDescriptorSetLayoutBinding{};
    textureInfosBinding.binding = 3;
    textureInfosBinding.descriptorCount = 1;
    textureInfosBinding.descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    textureInfosBinding.pImmutableSamplers = null;
    textureInfosBinding.stageFlags = vulkan.VK_SHADER_STAGE_FRAGMENT_BIT;

    const bindings = [_]vulkan.VkDescriptorSetLayoutBinding{
        uboLayoutBinding,
        samplerLayoutBinding,
        voxelBindings,
        textureInfosBinding,
    };

    var layoutInfo = vulkan.VkDescriptorSetLayoutCreateInfo{};
    layoutInfo.sType = vulkan.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = bindings.len;
    layoutInfo.pBindings = &bindings;

    var descriptorSetLayout: vulkan.VkDescriptorSetLayout = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateDescriptorSetLayout(device, &layoutInfo, null, &descriptorSetLayout)) {
        return error.FailedToCreateDescriptorSetLayout;
    }

    return descriptorSetLayout;
}

fn createUniformBuffers(varType: anytype, allocator: Allocator, device: vulkan.VkDevice, physicalDevice: vulkan.VkPhysicalDevice, usage: vulkan.VkBufferUsageFlags) !struct {
    []vulkan.VkBuffer,
    []vulkan.VkDeviceMemory,
    [][]u8,
} {
    return createUniformBuffersWithSize(@sizeOf(varType), allocator, device, physicalDevice, usage);
}

fn createUniformBuffersWithSize(bufferSize: usize, allocator: Allocator, device: vulkan.VkDevice, physicalDevice: vulkan.VkPhysicalDevice, usage: vulkan.VkBufferUsageFlags) !struct {
    []vulkan.VkBuffer,
    []vulkan.VkDeviceMemory,
    [][]u8,
} {
    const uniformBuffers = try allocator.alloc(vulkan.VkBuffer, MAX_FRAMES_IN_FLIGHT);
    const uniformBuffersMemory = try allocator.alloc(vulkan.VkDeviceMemory, MAX_FRAMES_IN_FLIGHT);
    const uniformBuffersMapped = try allocator.alloc([]u8, MAX_FRAMES_IN_FLIGHT);

    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        const b, const m = try createBuffer(
            device,
            physicalDevice,
            bufferSize,
            usage,
            vulkan.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vulkan.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );

        uniformBuffers[i] = b;
        uniformBuffersMemory[i] = m;

        var memory: [*c]u8 = null;
        if (vulkan.VK_SUCCESS != vulkan.vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, @ptrCast(&memory))) {
            return error.FailedToMapMemory;
        }
        uniformBuffersMapped[i] = memory[0..bufferSize];
    }

    return .{ uniformBuffers, uniformBuffersMemory, uniformBuffersMapped };
}

fn updateUniformBuffer(self: *Self, camera: *Camera, swapchain: *const Swapchain) !void {
    const ubo: *UniformBufferObject = @alignCast(@ptrCast(self.uniformBuffersMapped[self.currentFrame]));

    // TODO: init resolution at creation time
    ubo.camera_pos = .{ camera.x, -camera.y, camera.z, std.math.degreesToRadians(45) };
    ubo.camera_rot = .{ std.math.degreesToRadians(camera.yaw), std.math.degreesToRadians(camera.pitch) };
    ubo.resolution = .{ @floatFromInt(swapchain.extent.width), @floatFromInt(swapchain.extent.height) };
}

fn createDescriptorPool(device: vulkan.VkDevice, textureCount: u32) !vulkan.VkDescriptorPool {
    var poolSizes = [_]vulkan.VkDescriptorPoolSize{
        .{
            .type = vulkan.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT,
        },
        .{
            .type = vulkan.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT * textureCount,
        },
        .{
            .type = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT,
        },
        .{
            .type = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = MAX_FRAMES_IN_FLIGHT,
        },
    };

    var poolInfo = vulkan.VkDescriptorPoolCreateInfo{};
    poolInfo.sType = vulkan.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = poolSizes.len;
    poolInfo.pPoolSizes = &poolSizes;
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;

    var descriptorPool: vulkan.VkDescriptorPool = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateDescriptorPool(device, &poolInfo, null, &descriptorPool)) {
        return error.FailedToAllocateDescriptorPool;
    }

    return descriptorPool;
}

fn createDescriptorSet(
    allocator: Allocator,
    device: vulkan.VkDevice,
    descriptorPool: vulkan.VkDescriptorPool,
    uniformBuffers: []vulkan.VkBuffer,
    descriptorSetLayout: vulkan.VkDescriptorSetLayout,
    textureImageViews: []vulkan.VkImageView,
    textureSamplers: []vulkan.VkSampler,
    voxelsBuffers: []vulkan.VkBuffer,
    textureBuffers: []vulkan.VkBuffer,
    textureBufferLength: u32,
) ![]vulkan.VkDescriptorSet {
    var layouts: [MAX_FRAMES_IN_FLIGHT]vulkan.VkDescriptorSetLayout = .{ descriptorSetLayout, descriptorSetLayout };

    var allocInfo = vulkan.VkDescriptorSetAllocateInfo{};
    allocInfo.sType = vulkan.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = layouts.len;
    allocInfo.pSetLayouts = &layouts;

    const descriptorSets = try allocator.alloc(vulkan.VkDescriptorSet, MAX_FRAMES_IN_FLIGHT);
    if (vulkan.VK_SUCCESS != vulkan.vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.ptr)) {
        return error.FailedToAllocateDescriptorSets;
    }

    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        var bufferInfo = vulkan.VkDescriptorBufferInfo{};
        bufferInfo.buffer = uniformBuffers[i];
        bufferInfo.offset = 0;
        bufferInfo.range = @sizeOf(UniformBufferObject);

        var imagesInfos = try allocator.alloc(vulkan.VkDescriptorImageInfo, textureImageViews.len);
        defer allocator.free(imagesInfos);
        for (0..textureImageViews.len) |ti| {
            imagesInfos[ti] = .{
                .imageLayout = vulkan.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                .imageView = textureImageViews[ti],
                .sampler = textureSamplers[ti],
            };
        }

        var voxelsInfo = vulkan.VkDescriptorBufferInfo{};
        voxelsInfo.buffer = voxelsBuffers[i];
        voxelsInfo.offset = 0;
        voxelsInfo.range = @sizeOf(VoxelsBuffer);

        var textureInfo = vulkan.VkDescriptorBufferInfo{};
        textureInfo.buffer = textureBuffers[i];
        textureInfo.offset = 0;
        textureInfo.range = @sizeOf(u32) * textureBufferLength;

        var descriptorWrites = [_]vulkan.VkWriteDescriptorSet{
            .{
                .sType = vulkan.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSets[i],
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .descriptorCount = 1,
                .pBufferInfo = &bufferInfo,
                .pImageInfo = null, // Optional
                .pTexelBufferView = null, // Optional
            },
            .{
                .sType = vulkan.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSets[i],
                .dstBinding = 1,
                .dstArrayElement = 0,
                .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = @intCast(imagesInfos.len),
                .pBufferInfo = null,
                .pImageInfo = imagesInfos.ptr, // Optional
                .pTexelBufferView = null, // Optional
            },
            .{
                .sType = vulkan.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSets[i],
                .dstBinding = 2,
                .dstArrayElement = 0,
                .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .pBufferInfo = &voxelsInfo,
                .pImageInfo = null, // Optional
                .pTexelBufferView = null, // Optional
            },
            .{
                .sType = vulkan.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = descriptorSets[i],
                .dstBinding = 3,
                .dstArrayElement = 0,
                .descriptorType = vulkan.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .pBufferInfo = &textureInfo,
                .pImageInfo = null, // Optional
                .pTexelBufferView = null, // Optional
            },
        };

        vulkan.vkUpdateDescriptorSets(device, descriptorWrites.len, &descriptorWrites, 0, null);
    }
    return descriptorSets;
}

fn createTextureImage(
    device: vulkan.VkDevice,
    physicalDevice: vulkan.VkPhysicalDevice,
    commandPool: vulkan.VkCommandPool,
    graphicsQueue: vulkan.VkQueue,
    pixels: []const u8,
    width: u32,
    height: u32,
    channels: u32,
) !struct { vulkan.VkImage, vulkan.VkDeviceMemory, u32 } {
    const mipLevels = std.math.log2(@max(width, height)) + 1;

    const bufferSize = width * height * channels;
    const stagingBuffer, const stagingBufferMemory = try createBuffer(
        device,
        physicalDevice,
        bufferSize,
        vulkan.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        vulkan.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vulkan.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    );

    defer {
        vulkan.vkDestroyBuffer(device, stagingBuffer, null);
        vulkan.vkFreeMemory(device, stagingBufferMemory, null);
    }

    var data: [*c]u8 = undefined;
    if (vulkan.VK_SUCCESS != vulkan.vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, @ptrCast(&data))) {
        return error.MapMemoryFailed;
    }
    @memcpy(data[0..bufferSize], pixels);
    vulkan.vkUnmapMemory(device, stagingBufferMemory);

    const textureImage, const textureImageMemory = try createImage(
        device,
        physicalDevice,
        width,
        height,
        vulkan.VK_FORMAT_R8G8B8A8_SRGB,
        vulkan.VK_IMAGE_TILING_OPTIMAL,
        vulkan.VK_IMAGE_USAGE_TRANSFER_SRC_BIT | vulkan.VK_IMAGE_USAGE_TRANSFER_DST_BIT | vulkan.VK_IMAGE_USAGE_SAMPLED_BIT,
        vulkan.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        mipLevels,
        vulkan.VK_SAMPLE_COUNT_1_BIT,
    );

    try transitionImageLayout(
        device,
        commandPool,
        graphicsQueue,
        textureImage,
        vulkan.VK_FORMAT_R8G8B8A8_SRGB,
        vulkan.VK_IMAGE_LAYOUT_UNDEFINED,
        vulkan.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        mipLevels,
    );
    try copyBufferToImage(
        device,
        commandPool,
        graphicsQueue,
        stagingBuffer,
        textureImage,
        width,
        height,
    );

    try generateMipmaps(
        device,
        physicalDevice,
        commandPool,
        graphicsQueue,
        textureImage,
        vulkan.VK_FORMAT_R8G8B8A8_SRGB,
        @intCast(width),
        @intCast(height),
        mipLevels,
    );

    return .{ textureImage, textureImageMemory, mipLevels };
}

fn createImage(
    device: vulkan.VkDevice,
    physicalDevice: vulkan.VkPhysicalDevice,
    width: u32,
    height: u32,
    format: vulkan.VkFormat,
    tiling: vulkan.VkImageTiling,
    usage: vulkan.VkImageUsageFlags,
    properties: vulkan.VkMemoryPropertyFlags,
    mipLevels: u32,
    numSamples: vulkan.VkSampleCountFlagBits,
) !struct { vulkan.VkImage, vulkan.VkDeviceMemory } {
    var imageInfo = vulkan.VkImageCreateInfo{};
    imageInfo.sType = vulkan.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = vulkan.VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = vulkan.VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vulkan.VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = numSamples;
    imageInfo.flags = 0; // Optional

    var textureImage: vulkan.VkImage = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateImage(device, &imageInfo, null, &textureImage)) {
        return error.FailedToCreateImage;
    }

    var memRequirements = vulkan.VkMemoryRequirements{};
    vulkan.vkGetImageMemoryRequirements(device, textureImage, &memRequirements);

    var allocInfo = vulkan.VkMemoryAllocateInfo{};
    allocInfo.sType = vulkan.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = try findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    var textureImageMemory: vulkan.VkDeviceMemory = null;
    if (vulkan.VK_SUCCESS != vulkan.vkAllocateMemory(device, &allocInfo, null, &textureImageMemory)) {
        return error.FailedToAllocateMemory;
    }

    if (vulkan.VK_SUCCESS != vulkan.vkBindImageMemory(device, textureImage, textureImageMemory, 0)) {
        return error.FailedToBindMemory;
    }
    return .{ textureImage, textureImageMemory };
}

fn beginSingleTimeCommands(device: vulkan.VkDevice, commandPool: vulkan.VkCommandPool) !vulkan.VkCommandBuffer {
    var allocInfo = vulkan.VkCommandBufferAllocateInfo{};
    allocInfo.sType = vulkan.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = vulkan.VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    var commandBuffer: vulkan.VkCommandBuffer = null;
    if (vulkan.VK_SUCCESS != vulkan.vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer)) {
        return error.AllocateCommandBufferFailed;
    }

    var beginInfo = vulkan.VkCommandBufferBeginInfo{};
    beginInfo.sType = vulkan.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = vulkan.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vulkan.VK_SUCCESS != vulkan.vkBeginCommandBuffer(commandBuffer, &beginInfo)) {
        return error.BeginCommandBufferFailed;
    }

    return commandBuffer;
}

fn endSingleTimeCommands(
    device: vulkan.VkDevice,
    commandPool: vulkan.VkCommandPool,
    commandBuffer: vulkan.VkCommandBuffer,
    graphicsQueue: vulkan.VkQueue,
) !void {
    if (vulkan.VK_SUCCESS != vulkan.vkEndCommandBuffer(commandBuffer)) {
        return error.EndCommandBufferFailed;
    }

    var submitInfo = vulkan.VkSubmitInfo{};
    submitInfo.sType = vulkan.VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    const e = vulkan.vkQueueSubmit(graphicsQueue, 1, &submitInfo, null);
    if (vulkan.VK_SUCCESS != e) {
        log.err("failed {}", .{e});
        return error.QueueSubmitFailed;
    }
    if (vulkan.VK_SUCCESS != vulkan.vkQueueWaitIdle(graphicsQueue)) {
        return error.QueueWaitFailed;
    }

    vulkan.vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

fn transitionImageLayout(
    device: vulkan.VkDevice,
    commandPool: vulkan.VkCommandPool,
    graphicsQueue: vulkan.VkQueue,
    image: vulkan.VkImage,
    format: vulkan.VkFormat,
    oldLayout: vulkan.VkImageLayout,
    newLayout: vulkan.VkImageLayout,
    mipLevels: u32,
) !void {
    const commandBuffer = try beginSingleTimeCommands(device, commandPool);

    var barrier = vulkan.VkImageMemoryBarrier{};
    barrier.sType = vulkan.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vulkan.VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    var sourceStage: vulkan.VkPipelineStageFlags = undefined;
    var destinationStage: vulkan.VkPipelineStageFlags = undefined;

    if (oldLayout == vulkan.VK_IMAGE_LAYOUT_UNDEFINED and newLayout == vulkan.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = vulkan.VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = vulkan.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = vulkan.VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == vulkan.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and newLayout == vulkan.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = vulkan.VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = vulkan.VK_ACCESS_SHADER_READ_BIT;

        sourceStage = vulkan.VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = vulkan.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else if (oldLayout == vulkan.VK_IMAGE_LAYOUT_UNDEFINED and newLayout == vulkan.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = vulkan.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | vulkan.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        sourceStage = vulkan.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = vulkan.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    } else {
        return error.UnsupportedLayoutTransition;
    }

    if (newLayout == vulkan.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = vulkan.VK_IMAGE_ASPECT_DEPTH_BIT;

        if (hasStencilComponent(format)) {
            barrier.subresourceRange.aspectMask |= vulkan.VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    } else {
        barrier.subresourceRange.aspectMask = vulkan.VK_IMAGE_ASPECT_COLOR_BIT;
    }

    vulkan.vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage,
        destinationStage,
        0,
        0,
        null,
        0,
        null,
        1,
        &barrier,
    );

    try endSingleTimeCommands(device, commandPool, commandBuffer, graphicsQueue);
}

fn copyBufferToImage(
    device: vulkan.VkDevice,
    commandPool: vulkan.VkCommandPool,
    graphicsQueue: vulkan.VkQueue,
    buffer: vulkan.VkBuffer,
    image: vulkan.VkImage,
    width: u32,
    height: u32,
) !void {
    const commandBuffer = try beginSingleTimeCommands(device, commandPool);

    var region = vulkan.VkBufferImageCopy{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;

    region.imageSubresource.aspectMask = vulkan.VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;

    region.imageOffset = .{ .x = 0, .y = 0, .z = 0 };
    region.imageExtent = .{ .width = width, .height = height, .depth = 1 };

    vulkan.vkCmdCopyBufferToImage(
        commandBuffer,
        buffer,
        image,
        vulkan.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &region,
    );

    try endSingleTimeCommands(device, commandPool, commandBuffer, graphicsQueue);
}

fn createTextureImageView(device: vulkan.VkDevice, textureImage: vulkan.VkImage, mipLevels: u32) !vulkan.VkImageView {
    return createImageView(
        device,
        textureImage,
        vulkan.VK_FORMAT_R8G8B8A8_SRGB,
        vulkan.VK_IMAGE_ASPECT_COLOR_BIT,
        mipLevels,
    );
}

fn createImageView(
    device: vulkan.VkDevice,
    image: vulkan.VkImage,
    format: vulkan.VkFormat,
    aspectFlags: vulkan.VkImageAspectFlags,
    mipLevels: u32,
) !vulkan.VkImageView {
    var viewInfo = vulkan.VkImageViewCreateInfo{};
    viewInfo.sType = vulkan.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = image;
    viewInfo.viewType = vulkan.VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = mipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    var imageView: vulkan.VkImageView = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateImageView(device, &viewInfo, null, &imageView)) {
        return error.CreateImageViewFailed;
    }

    return imageView;
}

fn createTextureSampler(device: vulkan.VkDevice, physicalDevice: vulkan.VkPhysicalDevice, mipLevels: u32) !vulkan.VkSampler {
    var properties = vulkan.VkPhysicalDeviceProperties{};
    vulkan.vkGetPhysicalDeviceProperties(physicalDevice, &properties);

    var samplerInfo = vulkan.VkSamplerCreateInfo{};
    samplerInfo.sType = vulkan.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = vulkan.VK_FILTER_NEAREST;
    samplerInfo.minFilter = vulkan.VK_FILTER_NEAREST;
    samplerInfo.addressModeU = vulkan.VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = vulkan.VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = vulkan.VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = vulkan.VK_TRUE;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.borderColor = vulkan.VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = vulkan.VK_FALSE;
    samplerInfo.compareEnable = vulkan.VK_FALSE;
    samplerInfo.compareOp = vulkan.VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = vulkan.VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.mipLodBias = 0.0;
    samplerInfo.minLod = 0.0;
    samplerInfo.maxLod = @floatFromInt(mipLevels);

    var textureSampler: vulkan.VkSampler = null;
    if (vulkan.VK_SUCCESS != vulkan.vkCreateSampler(device, &samplerInfo, null, &textureSampler)) {
        return error.FailedToCreateSampler;
    }
    return textureSampler;
}

fn createDepthResources(
    device: vulkan.VkDevice,
    physicalDevice: vulkan.VkPhysicalDevice,
    commandPool: vulkan.VkCommandPool,
    graphicsQueue: vulkan.VkQueue,
    swapChainExtent: vulkan.VkExtent2D,
    msaaSamples: vulkan.VkSampleCountFlagBits,
) !struct { vulkan.VkImage, vulkan.VkDeviceMemory, vulkan.VkImageView } {
    const depthFormat = try findDepthFormat(physicalDevice);
    const depthImage, const depthImageMemory = try createImage(
        device,
        physicalDevice,
        swapChainExtent.width,
        swapChainExtent.height,
        depthFormat,
        vulkan.VK_IMAGE_TILING_OPTIMAL,
        vulkan.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
        vulkan.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        1,
        msaaSamples,
    );

    const depthImageView = try createImageView(device, depthImage, depthFormat, vulkan.VK_IMAGE_ASPECT_DEPTH_BIT, 1);

    try transitionImageLayout(
        device,
        commandPool,
        graphicsQueue,
        depthImage,
        depthFormat,
        vulkan.VK_IMAGE_LAYOUT_UNDEFINED,
        vulkan.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        1,
    );

    return .{ depthImage, depthImageMemory, depthImageView };
}

fn findSupportedFormat(
    physicalDevice: vulkan.VkPhysicalDevice,
    candidates: []vulkan.VkFormat,
    tiling: vulkan.VkImageTiling,
    features: vulkan.VkFormatFeatureFlags,
) !vulkan.VkFormat {
    for (candidates) |format| {
        var props = vulkan.VkFormatProperties{};
        vulkan.vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        if (tiling == vulkan.VK_IMAGE_TILING_LINEAR and (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == vulkan.VK_IMAGE_TILING_OPTIMAL and (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    return error.FailedToFindFormat;
}

fn findDepthFormat(physicalDevice: vulkan.VkPhysicalDevice) !vulkan.VkFormat {
    var formats = [_]vulkan.VkFormat{ vulkan.VK_FORMAT_D32_SFLOAT, vulkan.VK_FORMAT_D32_SFLOAT_S8_UINT, vulkan.VK_FORMAT_D24_UNORM_S8_UINT };
    return findSupportedFormat(
        physicalDevice,
        &formats,
        vulkan.VK_IMAGE_TILING_OPTIMAL,
        vulkan.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
    );
}

fn hasStencilComponent(format: vulkan.VkFormat) bool {
    return format == vulkan.VK_FORMAT_D32_SFLOAT_S8_UINT or format == vulkan.VK_FORMAT_D24_UNORM_S8_UINT;
}

fn generateMipmaps(
    device: vulkan.VkDevice,
    physicalDevice: vulkan.VkPhysicalDevice,
    commandPool: vulkan.VkCommandPool,
    graphicsQueue: vulkan.VkQueue,
    image: vulkan.VkImage,
    imageFormat: vulkan.VkFormat,
    texWidth: i32,
    texHeight: i32,
    mipLevels: u32,
) !void {
    var formatProperties: vulkan.VkFormatProperties = undefined;
    vulkan.vkGetPhysicalDeviceFormatProperties(physicalDevice, imageFormat, &formatProperties);

    if ((formatProperties.optimalTilingFeatures & vulkan.VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) == 0) {
        return error.TextureFormatDoesNotSupportBlittering;
    }

    const commandBuffer = try beginSingleTimeCommands(device, commandPool);

    var barrier = vulkan.VkImageMemoryBarrier{};
    barrier.sType = vulkan.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = vulkan.VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = vulkan.VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    var mipWidth = texWidth;
    var mipHeight = texHeight;
    for (1..mipLevels) |is| {
        const i = @as(u32, @intCast(is));
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = vulkan.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = vulkan.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = vulkan.VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = vulkan.VK_ACCESS_TRANSFER_READ_BIT;

        vulkan.vkCmdPipelineBarrier(
            commandBuffer,
            vulkan.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vulkan.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            null,
            0,
            null,
            1,
            &barrier,
        );

        var blit = vulkan.VkImageBlit{};
        blit.srcOffsets[0] = .{ .x = 0, .y = 0, .z = 0 };
        blit.srcOffsets[1] = .{ .x = mipWidth, .y = mipHeight, .z = 1 };
        blit.srcSubresource.aspectMask = vulkan.VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = .{ .x = 0, .y = 0, .z = 0 };
        blit.dstOffsets[1] = .{
            .x = if (mipWidth > 1) @divTrunc(mipWidth, 2) else 1,
            .y = if (mipHeight > 1) @divTrunc(mipHeight, 2) else 1,
            .z = 1,
        };
        blit.dstSubresource.aspectMask = vulkan.VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vulkan.vkCmdBlitImage(
            commandBuffer,
            image,
            vulkan.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            image,
            vulkan.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &blit,
            vulkan.VK_FILTER_LINEAR,
        );

        barrier.oldLayout = vulkan.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = vulkan.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = vulkan.VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = vulkan.VK_ACCESS_SHADER_READ_BIT;

        vulkan.vkCmdPipelineBarrier(
            commandBuffer,
            vulkan.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vulkan.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0,
            null,
            0,
            null,
            1,
            &barrier,
        );

        if (mipWidth > 1) {
            mipWidth = @divTrunc(mipWidth, 2);
        }
        if (mipHeight > 1) {
            mipHeight = @divTrunc(mipHeight, 2);
        }
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vulkan.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = vulkan.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = vulkan.VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = vulkan.VK_ACCESS_SHADER_READ_BIT;

    vulkan.vkCmdPipelineBarrier(
        commandBuffer,
        vulkan.VK_PIPELINE_STAGE_TRANSFER_BIT,
        vulkan.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0,
        null,
        0,
        null,
        1,
        &barrier,
    );

    try endSingleTimeCommands(device, commandPool, commandBuffer, graphicsQueue);
}

fn getMaxUsableSampleCount(physicalDevice: vulkan.VkPhysicalDevice) vulkan.VkSampleCountFlagBits {
    var physicalDeviceProperties: vulkan.VkPhysicalDeviceProperties = undefined;
    vulkan.vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

    const counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & vulkan.VK_SAMPLE_COUNT_64_BIT != 0) {
        return vulkan.VK_SAMPLE_COUNT_64_BIT;
    }
    if (counts & vulkan.VK_SAMPLE_COUNT_32_BIT != 0) {
        return vulkan.VK_SAMPLE_COUNT_32_BIT;
    }
    if (counts & vulkan.VK_SAMPLE_COUNT_16_BIT != 0) {
        return vulkan.VK_SAMPLE_COUNT_16_BIT;
    }
    if (counts & vulkan.VK_SAMPLE_COUNT_8_BIT != 0) {
        return vulkan.VK_SAMPLE_COUNT_8_BIT;
    }
    if (counts & vulkan.VK_SAMPLE_COUNT_4_BIT != 0) {
        return vulkan.VK_SAMPLE_COUNT_4_BIT;
    }
    if (counts & vulkan.VK_SAMPLE_COUNT_2_BIT != 0) {
        return vulkan.VK_SAMPLE_COUNT_2_BIT;
    }

    return vulkan.VK_SAMPLE_COUNT_1_BIT;
}

fn createColorResources(
    device: vulkan.VkDevice,
    physicalDevice: vulkan.VkPhysicalDevice,
    swapChainExtent: vulkan.VkExtent2D,
    colorFormat: vulkan.VkFormat,
    msaaSamples: vulkan.VkSampleCountFlagBits,
) !struct { vulkan.VkImage, vulkan.VkDeviceMemory, vulkan.VkImageView } {
    const colorImage, const colorImageMemory = try createImage(
        device,
        physicalDevice,
        swapChainExtent.width,
        swapChainExtent.height,
        colorFormat,
        vulkan.VK_IMAGE_TILING_OPTIMAL,
        vulkan.VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | vulkan.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        vulkan.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        1,
        msaaSamples,
    );
    const colorImageView = try createImageView(device, colorImage, colorFormat, vulkan.VK_IMAGE_ASPECT_COLOR_BIT, 1);
    return .{ colorImage, colorImageMemory, colorImageView };
}

pub fn updateWorld(self: *Self, world: [][10][10]Chunk) void {
    for (0..MAX_FRAMES_IN_FLIGHT) |f| {
        const vox: *VoxelsBuffer = @alignCast(@ptrCast(self.voxelsBuffersMapped[f]));

        for (0..10) |zchunk| {
            for (0..10) |ychunk| {
                for (0..10) |xchunk| {
                    const chunk = &world[zchunk][ychunk][xchunk];
                    const chunk_slice = vox.voxels[(zchunk * 10 * 10 + ychunk * 10 + xchunk) * CHUNK_SIZE ..][0..CHUNK_SIZE];

                    for (0..64) |z| {
                        for (0..64) |y| {
                            for (0..64) |x| {
                                const block = chunk.getBlock(x, y, z);
                                chunk_slice[z * ROW_LENGTH * ROW_LENGTH + y * ROW_LENGTH + x] = @as(u32, block);
                            }
                        }
                    }
                }
            }
        }
    }
}
