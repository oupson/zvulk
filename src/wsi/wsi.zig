const vulkan = @cImport({
    @cInclude("vulkan/vulkan.h");
});

ptr: *anyopaque,
vtable: struct {
    connect: *const fn (*anyopaque) anyerror!void,
    createVulkanSurface: *const fn (*anyopaque, vulkan.VkInstance) anyerror!vulkan.VkSurfaceKHR,
    dispatch: *const fn (*anyopaque) anyerror!bool,
    deinit: *const fn (*anyopaque) void,
},

pub fn connect(self: @This()) anyerror!void {
    return try self.vtable.connect(self.ptr);
}

pub fn dispatch(self: @This()) anyerror!bool {
    return try self.vtable.dispatch(self.ptr);
}

pub fn createVulkanSurface(self: @This(), instance: vulkan.VkInstance) !vulkan.VkSurfaceKHR {
    return try self.vtable.createVulkanSurface(self.ptr, instance);
}

pub fn deinit(self: @This()) void {
    self.vtable.deinit(self.ptr);
}

pub const EventType = enum {
    Close,
    Draw,
    RecreateSurface,
    KeyboardEvent,
};

pub const KeyboardEventType = enum { KeyUp, KeyDown };

pub const KeyboardEvent = union(KeyboardEventType) {
    KeyUp: Key,
    KeyDown: Key,
};

pub const KeyType = enum(u8) {
    Space,
    Shift,
    Escape,
    Char,
};

pub const Key = union(KeyType) {
    Space,
    Shift,
    Escape,
    Char: u8,
};

pub const Event = union(EventType) {
    Close,
    Draw,
    RecreateSurface,
    KeyboardEvent: KeyboardEvent,
};

pub const Listener = struct {
    ptr: *anyopaque,
    vtable: ListenerVtable,
};

pub const ListenerVtable = struct {
    keyboardEvent: *const fn (*anyopaque, KeyboardEvent) void,
    mouseMove: *const fn (*anyopaque, dx: f64, dy: f64) void,
    close: *const fn (*anyopaque) void,
    draw: *const fn (*anyopaque) void,
    invalidateSurface: *const fn (*anyopaque, i32, i32) void,
};
